# 360 Image Auto-Enhance — Research Benchmark Plan

## 0. Context (read first)

We want an "AI Enhance" button for user-uploaded 360° (equirectangular) images. Before any product or
architecture work, we run a **research benchmark**: implement 5 classical, deterministic image-enhancement
methods in Python, run them over every frame of 5 real 360° datasets, measure them, and pick a winner.

Hard constraints:

- **No machine learning in the enhancement pipeline.** Every method is a deterministic algorithm: the same
  input and config must give a bit-identical output. Parameters are either fixed or computed from image
  statistics with closed-form formulas.
- Learned image-quality metrics (NIQE, BRISQUE via `pyiqa`) are allowed **for evaluation only**.
- This is a standalone research project. It is not part of any company repo. Do not commit or push anywhere.
- Never modify the input zip files or the extracted originals.

Deliverables:

1. A Python package that runs all 5 methods (plus 2 baselines) over all frames.
2. A metrics table (parquet + CSV) covering every frame × method.
3. A Streamlit UI for visual comparison, 360 viewing, and blind A/B voting.
4. An auto-generated `report.md` with aggregate results and a recommendation.

---

## 1. Environment

- Python 3.11, in a venv inside the project folder (`python -m venv .venv`), or `uv` if available.
- Dependencies (pin them in `pyproject.toml`):
  - `numpy`, `scipy`, `opencv-contrib-python` (needed for `cv2.ximgproc.guidedFilter` and `cv2.createMergeMertens`)
  - `scikit-image`, `pillow`, `piexif`
  - `pandas`, `pyarrow`, `pyyaml`, `tqdm`
  - `pyiqa` + `torch` (CPU build is fine; first use downloads metric weights, so it needs internet)
  - `streamlit`, `streamlit-image-comparison`, `plotly`
  - `pytest`
- No GPU is required.
- Record the CPU model, core count, and RAM in the report, because the runtime numbers depend on them.

---

## 2. Project layout

```
360-enhance-research/
  plan.md
  pyproject.toml
  config/default.yaml            # all method params + shared settings
  data/zips/                     # INPUT: the 5 zip files go here (read-only)
  data/frames/<dataset>/...      # extracted originals (read-only after ingest)
  outputs/<run_id>/
    images/<method>/<dataset>/<frame>.jpg      # full-res results
    previews/<method>/<dataset>/<frame>.jpg    # 2048x1024 for UI + metrics
    metrics.parquet / metrics.csv
    runtime.csv
    votes.sqlite
    report.md
    config.yaml                  # frozen copy of the config used
  src/enhance/
    io.py          # zip ingest, decode, sRGB<->linear, save JPEG keeping metadata
    equirect.py    # wrap padding, latitude weights, seam crop
    common.py      # white balance, levels/gamma, colour recombine, vibrance, blend
    methods/
      base.py      # registry + interface
      clahe.py
      msrcr.py
      lime.py
      fusion.py
      llf.py
    run.py         # CLI: batch processing
    metrics.py
    benchmark.py   # CLI: compute metrics over a run
    report.py      # CLI: build report.md
  app/app.py       # Streamlit UI (multipage)
  tests/
```

`run_id` = short hash of the frozen config plus a timestamp. Runs are idempotent: an output that already
exists for the same run_id is skipped.

---

## 3. Data ingest (`io.py`)

1. Extract each zip in `data/zips/` to `data/frames/<zip_stem>/`. Do not trust paths inside the zip: reject
   absolute paths and `..` entries (zip-slip).
2. Walk the extracted folders recursively and collect `.jpg/.jpeg/.png/.tif/.tiff` files. Ignore everything else
   (log the skipped files).
3. **Validate** each image:
   - Aspect ratio must be 2:1 (±1%). Anything else is probably an unstitched dual-fisheye or a non-360 image:
     **list those files and stop, so the human can decide.** Do not silently process them.
   - Record the resolution, bit depth, and whether EXIF/XMP is present (360 viewers rely on the GPano XMP tags).
4. Write `data/manifest.csv`: dataset, frame_id, path, width, height, has_xmp.

Decoding: load as float32 RGB in [0,1] (the sRGB-encoded values). Apply EXIF orientation.

Saving: JPEG quality 95, 4:4:4 chroma, **copy EXIF and XMP from the source** (piexif for EXIF; copy the XMP
APP1 segment as raw bytes, or use `exiftool -tagsFromFile` if it is installed). Check afterwards that GPano tags
are still present.

---

## 4. Shared building blocks

### 4.1 Colour spaces

- sRGB ↔ linear: use the standard piecewise sRGB transfer function.
- Luminance (on linear RGB): `Y = 0.2126 R + 0.7152 G + 0.0722 B`.
- **Methods operate on sRGB-encoded float32 [0,1]** unless stated otherwise. White balance runs in linear space and
  the result is re-encoded.

### 4.2 Equirectangular handling (`equirect.py`) — important

An equirect image wraps horizontally: column 0 is next to column W−1. Any neighbourhood operation (blur, tiles,
pyramids, gradients) will produce a visible **seam** unless this is handled.

- `pad_equirect(img, p)`:
  - Horizontal: `np.pad(..., mode="wrap")` by `p` columns on each side.
  - Vertical (geometrically correct pole continuation): the `p` rows above the top are the top `p` rows
    flipped vertically **and rolled by W/2**. Do the same at the bottom. (Crossing a pole lands you on the
    opposite meridian.)
- `crop_equirect(img, p)`: inverse of the padding.
- Rule: **every method pads before processing and crops after**. `p` must be at least the method's spatial
  support (≈3σ of the largest blur, one CLAHE tile, or the top pyramid level's footprint).
- `lat_weights(H)`: the row weight is `w_i = cos(φ_i)`, where `φ_i = π·(0.5 − (i+0.5)/H)`. Use it for **all
  global statistics** (percentiles, means, medians), so the oversampled pole rows don't dominate.
- `seam_crop(img, k=256)`: concatenate the last `k` columns with the first `k` columns, for visual seam checks in the UI.

### 4.3 Shared pipeline (`common.py`)

Every method runs inside the same wrapper, so the comparison is fair:

```
I (sRGB float) ─► [pre] white balance ─► METHOD CORE ─► [post] auto-levels on luminance ─► (optional) vibrance ─► clip ─► save
```

All steps are toggleable in `config/default.yaml`. Defaults: WB on, post-levels on, vibrance off.

**White balance — Shades-of-Gray (Finlayson & Trezzi 2004), partial strength:**

```
work in linear RGB, lat-weighted
e_c = ( Σ w·I_c^p / Σ w )^(1/p),  p = 6        # p=1 is Gray-World, p=∞ is White-Patch
ē   = (e_R + e_G + e_B) / 3
g_c = ē / e_c
g'_c = 1 + α·(g_c − 1),  α = 0.3               # partial: real warm scenes stay warm
I_c ← clip(g'_c · I_c)
```

**Auto-levels + gamma on luminance (also the first stage of the CLAHE method):**

```
p_lo, p_hi = weighted percentiles 0.5% / 99.5% of Y
Y1 = clip((Y − p_lo)/(p_hi − p_lo), 0, 1)
γ  = ln(0.5) / ln(weighted_median(Y1)),  clamp γ to [0.5, 2.0]
Y2 = Y1^γ
```

(The `post` step uses the same stretch but **without** gamma, so it doesn't override each method's tone decisions.)

**Colour recombine (hue-preserving), used by every method that works on luminance only:**

```
r = Y_out / max(Y_in, 1e-4)
C_out = clip(C_in · r)     for C in R,G,B
```

Clipping can desaturate highlights. That is acceptable, but log the % of pixels clipped.

**Vibrance (optional):** in Lab, chroma `c = sqrt(a²+b²)`, `s = 1 + k·(1 − c/c_max)` with `k = 0.2` and
`c_max = 128`; then `a' = s·a`, `b' = s·b`.

**Strength blend (needed for the product later; implement it, but benchmark at t = 1):**
`Out = (1 − t)·I + t·E`.

### 4.4 Method interface (`methods/base.py`)

```python
class Method(Protocol):
    name: str
    def __call__(self, img: np.ndarray, params: dict) -> np.ndarray: ...
    # img: HxWx3 float32 sRGB-encoded in [0,1] (already white-balanced); returns same shape/dtype
```

Register: `original` (identity), `wb_only` (identity core, so the output is just shared pre/post), `clahe`,
`msrcr`, `lime`, `fusion`, `llf`.

---

## 5. The five methods

### Theory shared by all of them: Retinex

`I(x) = R(x) · L(x)`, where R is reflectance (scene detail and colour) and L is illumination (smooth, the cause
of dark corners and blown-out windows). Each method compresses the dynamic range of L while preserving R, using a
different estimator or strategy.

### 5.1 CLAHE (Zuiderveld 1994)

1. Convert to Lab and take `L*`, scaled to [0,1].
2. Apply auto-levels + gamma (§4.3) to `L*`.
3. CLAHE on `L*`: `clipLimit = 2.0`, a tile grid of **8 rows × 16 cols** on the original image.
   - Equirect wrap: tile width `tw = W/16`. Pad horizontally by `tw` on each side (wrap) and vertically by
     one tile height (pole rule). Run `cv2.createCLAHE` with a grid that matches the padded size
     (rows+2, cols+2), then crop. Convert to **uint16** before CLAHE to keep precision.
4. Put `L*` back and convert Lab → sRGB.

Math, for the report: plain histogram equalisation maps `T(k) = CDF(k)`, so the slope `T'(k) ∝ h(k)`. CLAHE clips
each tile's histogram at `clipLimit × mean_bin_count` and redistributes the excess, which caps the slope (the
contrast gain). Each pixel bilinearly interpolates the mappings of its 4 nearest tiles.

Params to expose: `clip_limit` {1.5, 2.0, 3.0}, `grid` {(4,8), (8,16)}.

### 5.2 MSRCR — Multi-Scale Retinex with Colour Restoration (Jobson, Rahman, Woodell 1997)

Reference implementation to follow: Petro, Sbert, Morel, "Multiscale Retinex", IPOL 2014.

```
per channel c, on I scaled to [1, 256] (avoid log 0):
MSR_c   = Σ_i w_i [ log I_c − log(G_σi * I_c) ],   w_i = 1/3
CR_c    = β [ log(α I_c) − log(Σ_c I_c) ],          α = 125, β = 46
MSRCR_c = G · (CR_c · MSR_c + b),                   G = 192, b = −30
then "simplest colour balance": clip 1% low / 1% high per channel, stretch to [0,1]
```

- Scales σ = {15, 80, 250} **at a reference width of 2048 px**. Scale proportionally to the actual width.
- Speed: compute the Gaussian blurs on a downsampled copy (to width 2048), then upsample the blurred
  illumination bilinearly. This is valid because the blurs are very smooth.
- Equirect: pad by `3·σ_max` (wrap + pole rule) before blurring. Do not rely on OpenCV border modes (they
  don't support wrap for blurs).

Known failure modes to watch for: halos at strong edges, greyish/washed-out colour.

### 5.3 LIME — Low-light Image Enhancement via Illumination Map Estimation (Guo, Li, Ling, IEEE TIP 2017)

```
1. Initial illumination:  L̂(x) = max_c I_c(x)
2. Refine (the paper's "sped-up" solver): solve the linear system
       (Id + α Σ_{d∈{h,v}} D_dᵀ diag(W_d) D_d) t = vec(L̂)
   W_d(x) = 1 / (|∇_d L̂(x)| + ε)            # strategy II weights; ε = 1e-3
   α = 0.15
3. Output:  R_c(x) = I_c(x) / max(T(x)^γ, ε),  γ = 0.8
```

- Solve at a **reduced resolution** (width 1024; try 2048 if fast enough) with `scipy.sparse` +
  `scipy.sparse.linalg.cg` (preconditioned) or `spsolve`. Then upsample T to full resolution with a
  **guided filter** (`cv2.ximgproc.guidedFilter`, guide = full-res max-channel, radius ≈ 8 px at full res,
  eps = 1e-3). Pad before guided filtering.
- Equirect: make the horizontal difference operator `D_h` **circular** (the last column differences against the
  first). That handles the wrap exactly inside the solver, with no padding needed for step 2.
- The paper optionally denoises with BM3D. **Skip it**, but measure noise amplification (§6), because LIME is known
  to amplify noise in dark areas.

Params: `alpha` {0.08, 0.15, 0.3}, `gamma` {0.6, 0.8}.

### 5.4 Exposure fusion (Mertens, Kautz, Van Reeth 2007), on synthetic exposures

1. Convert to linear RGB.
2. Synthesise exposures: `E_k = clip(I_lin · 2^{ev_k})` with `ev = {−1, 0, +1, +2}`, then re-encode each one to sRGB.
3. Fuse with `cv2.createMergeMertens(contrast_weight=1, saturation_weight=1, exposure_weight=1)`.
   The weights per pixel are contrast (|Laplacian|), saturation (std across RGB), and well-exposedness
   `exp(−(v−0.5)²/(2·0.2²))`. Blending uses Laplacian/Gaussian pyramids.
4. Equirect: pad (wrap + pole rule) by `2^levels` px (at least 256) before merging, then crop.

Memory: 4 exposures at 8K in float32 is about 1.5 GB plus pyramids. Keep worker count in check (§7).

Optional stretch goal: the BIMEF variant (Ying et al. 2017), which uses a camera response model instead of a plain gain.

Params: `ev` sets {(-1,0,1,2), (0,1,2)}.

### 5.5 Local Laplacian filter (Paris, Hasinoff, Kautz 2011; fast version Aubry et al. 2014)

Edge-aware detail and tone manipulation: the method behind "Clarity"-style edits.

1. Work on `l = log(Y + 1e-4)` (linear luminance).
2. Remapping function, for a reference value `g` (σ_r = 0.4 in the log domain):
   ```
   detail   (|i−g| ≤ σ_r):  r(i) = g + sign(i−g)·σ_r·(|i−g|/σ_r)^α_d       α_d = 0.6  (<1 boosts detail)
   edges    (|i−g| >  σ_r):  r(i) = g + sign(i−g)·(β·(|i−g|−σ_r) + σ_r)     β   = 0.7  (<1 compresses range)
   ```
3. Fast algorithm (Aubry 2014): sample K = 10 reference values `g_j` evenly across the range of `l`. For each j,
   remap the whole image with `r(·; g_j)` and build its Laplacian pyramid. Build the Gaussian pyramid of `l`. At
   each level and pixel, linearly interpolate the output Laplacian coefficient between the two `g_j`
   bracketing the Gaussian-pyramid value. Collapse.
   - Stream over j and accumulate, so you don't hold K full pyramids at once.
4. `Y_out = exp(l_out)`, colour recombine (§4.3), then let the shared post-levels normalise.
5. Equirect: pad by `2^n_levels` px (wrap + pole rule) before building pyramids.

Params: `alpha_d` {0.5, 0.7}, `beta` {0.6, 0.8}.

### 5.6 Parameters

- `config/default.yaml` holds the default values above. The **main benchmark uses the defaults only**.
- An optional sweep (CLI flag `--sweep`) runs the small grids listed per method on **one** dataset, so we can
  tell whether a method lost because of a bad default.

---

## 6. Metrics (`metrics.py`, `benchmark.py`)

There is no ground truth, so PSNR/SSIM against a reference are not meaningful. Use no-reference and
artifact metrics. Unless stated otherwise, compute them on the **2048×1024 preview** so they are consistent and fast.

| Metric | What it tells us | Better |
|---|---|---|
| NIQE (`pyiqa`) | Naturalness | lower |
| BRISQUE (`pyiqa`) | Perceptual quality | lower |
| LOE, Lightness Order Error | Is the relative brightness order kept (naturalness)? Compute on 200×100 downsample, lightness = max(R,G,B), mean count of order flips per pixel vs the original | lower |
| Entropy of Y | Use of tonal range | higher (diminishing) |
| RMS contrast (lat-weighted std of Y) | Global contrast | context |
| Mean brightness (lat-weighted) | Exposure | close to ~0.45–0.5 |
| Clipping % (Y ≤ 2/255 or ≥ 253/255) | Lost shadows/highlights | lower |
| Colourfulness (Hasler & Süsstrunk 2003) | Colour vividness | context (over-boost is bad) |
| Noise amplification = σ̂_out / σ̂_in (`skimage.restoration.estimate_sigma` on Y) | Noise boosted in flat/dark areas | lower, ≈1 ideal |
| SSIM vs original (on Y) | Structure preservation sanity check (should be high, not 1) | > ~0.6 |
| **Seam delta** (full res): mean abs Y diff between col 0 and col W−1, output minus input, in 8-bit units | Equirect seam artifact | ≈0, **gate: < 1.0** |
| Runtime (s/frame, full res) and peak RAM | Feasibility for production | lower |

Outputs: one row per (dataset, frame, method) in `metrics.parquet`/`.csv`. Aggregates are the median and IQR per
method, overall and per dataset.

---

## 7. Batch runner (`run.py`)

```
python -m enhance.run --zips data/zips --config config/default.yaml [--methods all] [--datasets all] [--sweep]
python -m enhance.benchmark --run outputs/<run_id>
python -m enhance.report   --run outputs/<run_id>
```

- Parallelise across frames with `ProcessPoolExecutor`. Workers = `min(cpu_count, floor(RAM_GB / 4))`.
  An 8K frame needs roughly 2–4 GB at peak in the heavier methods.
- For each output, write the full-res JPEG and the 2048×1024 preview, and append to `runtime.csv` (method, frame,
  seconds, peak RSS via `resource`/`psutil`).
- Catch exceptions per frame/method. Log them and continue; failures appear as a "failed" count in the report.
- Determinism check: for 3 random frames, run each method twice and assert the float arrays are identical
  (compare hashes before JPEG encoding).

---

## 8. UI (`app/app.py`, Streamlit, multipage)

`streamlit run app/app.py -- --run outputs/<run_id>`

1. **Browse / Compare**
   - Choose a dataset, then a frame. Show a grid: original, wb_only, and the 5 methods (previews), each labelled
     with its key metrics.
   - A before/after slider (`streamlit-image-comparison`) for original vs one selected method.
   - A seam strip view (`seam_crop`) for every method, stacked.
   - A full-res crop viewer: choose an (x, y, size) region and show the same crop from all methods side by side
     (from full-res outputs), for checking noise and halos.
2. **360 Viewer**
   - Embed Pannellum through `streamlit.components.v1.html`, with the preview served as base64 (4096×2048 max).
     Use a method dropdown and keep the camera yaw/pitch fixed when switching methods if possible.
3. **Blind Vote** (the most important part)
   - Show the original on top. Below it, two outputs from randomly chosen different methods (not `original`),
     labelled only "A" and "B", in random order. Use the 360 viewer or the flat preview (toggle).
   - Buttons: A better / B better / Tie / Both worse than original.
   - Store in `votes.sqlite`: voter name (text box), timestamp, frame, method_a, method_b, choice.
   - Sample pairs so every method pair and dataset gets roughly equal coverage.
4. **Results**
   - Metric tables (median/IQR per method; toggle overall vs per dataset).
   - Box plots per metric (plotly).
   - Vote results: win rate per method (wins / comparisons, ties = 0.5), a pairwise win matrix, and the
     "both worse than original" rate per method.
   - Runtime and RAM per method.

---

## 9. Decision framework (`report.py`)

**Hard gates.** A method that fails any of these is disqualified:

- Seam delta < 1.0 on every frame.
- No crashes on any frame.
- Deterministic (identical output on repeated runs).
- "Both worse than original" rate < 30%.

**Score for the methods that pass the gates** (rank-based, rank per criterion across methods, weighted average):

| Criterion | Weight |
|---|---|
| Human blind-vote win rate | 50% |
| Naturalness: NIQE + LOE (mean of ranks) | 20% |
| Artifacts: noise amplification + clipping % (mean of ranks) | 20% |
| Runtime | 10% |

`report.md` contains: hardware, dataset summary, per-method aggregates, vote results, gate results, the final
ranking, 3 example frames per method (best/median/worst by vote), and **notes on per-dataset differences**. If one
method wins indoors and another outdoors, call that out, because it may mean we need a hybrid.

---

## 10. Tests (`tests/`)

Keep these small and targeted:

- `test_equirect.py`: pad→crop round-trip is the identity; vertical pole padding matches the flip+roll rule;
  latitude weights are symmetric and sum correctly.
- `test_seam.py`: for each method, a synthetic equirect image that is continuous across the wrap
  (for example a horizontal sinusoid with a period that divides W) must produce output with seam delta < 1.0.
- `test_determinism.py`: each method on a small random image, run twice, gives identical arrays.
- `test_io.py`: EXIF/XMP survive save; a non-2:1 image is rejected.
- `test_wb.py`: a grey image with a colour cast moves toward neutral by ≈α.

Run them with `pytest tests/ -q`.

---

## 11. Milestones and estimates

| # | Milestone | Done when | Est. |
|---|---|---|---|
| M0 | Env + ingest + manifest | Manifest lists all frames; non-2:1 files flagged | 1 h |
| M1 | Shared utils + `original`/`wb_only` + CLAHE | Tests pass; CLAHE outputs for 1 dataset look right | 3 h |
| M2 | MSRCR + exposure fusion | Outputs for 1 dataset; seam test passes | 3 h |
| M3 | LIME | Same; runtime per 8K frame recorded | 4 h |
| M4 | Local Laplacian | Same | 4 h |
| M5 | Full batch run (all datasets × all methods) | No failures; determinism check passes | 2 h + compute |
| M6 | Metrics + benchmark | `metrics.parquet` complete | 3 h |
| M7 | Streamlit UI (4 pages) | All pages load against a real run | 4 h |
| M8 | Human vote session + report | ≥ 30 votes per method pair; `report.md` generated | 2 h + human time |

After each milestone, stop and show the human a short summary plus 2–3 example images before continuing.
If a method looks clearly broken (for example extreme halos or colour casts), check the implementation against the
cited paper or reference before concluding that the method is bad.

---

## 12. Out of scope (do not build)

- Production architecture, API, storage, or frontend integration.
- Any ML/learned enhancement model.
- Denoising (BM3D etc.), sharpening, and HDR merging of real brackets.
- Performance optimisation beyond what makes the benchmark finish in reasonable time.

## 13. References

- K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV, 1994.
- D. Jobson, Z. Rahman, G. Woodell, "A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes", IEEE TIP 1997.
- A. B. Petro, C. Sbert, J.-M. Morel, "Multiscale Retinex", IPOL 2014.
- X. Guo, Y. Li, H. Ling, "LIME: Low-Light Image Enhancement via Illumination Map Estimation", IEEE TIP 2017.
- T. Mertens, J. Kautz, F. Van Reeth, "Exposure Fusion", Pacific Graphics 2007.
- Z. Ying et al., "A Bio-Inspired Multi-Exposure Fusion Framework for Low-light Image Enhancement" (BIMEF), 2017.
- S. Paris, S. Hasinoff, J. Kautz, "Local Laplacian Filters: Edge-aware Image Processing with a Laplacian Pyramid", SIGGRAPH 2011.
- M. Aubry et al., "Fast Local Laplacian Filters: Theory and Applications", ACM TOG 2014.
- G. Finlayson, E. Trezzi, "Shades of Gray and Colour Constancy", CIC 2004.
- D. Hasler, S. Süsstrunk, "Measuring Colourfulness in Natural Images", SPIE 2003.
- K. He, J. Sun, X. Tang, "Guided Image Filtering", ECCV 2010 / TPAMI 2013.
