"""Zip ingest, decode/encode, EXIF/XMP preservation."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import piexif
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class AspectRatioError(ValueError):
    """Raised when one or more frames are not 2:1 equirectangular."""


def _is_safe_zip_member(name: str) -> bool:
    p = Path(name)
    if p.is_absolute() or name.startswith("/") or name.startswith("\\"):
        return False
    parts = p.parts
    if any(part == ".." for part in parts):
        return False
    return True


def extract_zips(zips_dir: Path, frames_dir: Path, overwrite: bool = False) -> list[Path]:
    """Extract each zip in ``zips_dir`` to ``frames_dir/<stem>/`` with zip-slip checks."""
    zips_dir = Path(zips_dir)
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for zpath in sorted(zips_dir.glob("*.zip")):
        dest = frames_dir / zpath.stem
        if dest.exists() and any(dest.rglob("*")) and not overwrite:
            log.info("skip extract (exists): %s", dest)
            extracted.append(dest)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if not _is_safe_zip_member(info.filename):
                    raise ValueError(f"unsafe zip path rejected: {info.filename!r} in {zpath}")
                target = dest / info.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        log.info("extracted %s -> %s", zpath.name, dest)
        extracted.append(dest)
    return extracted


def collect_images(root: Path) -> tuple[list[Path], list[Path]]:
    """Walk recursively; return (image_paths, skipped_paths)."""
    images: list[Path] = []
    skipped: list[Path] = []
    for p in sorted(Path(root).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        else:
            skipped.append(p)
            log.info("skipped non-image: %s", p)
    return images, skipped


def _has_xmp(raw: bytes) -> bool:
    # APP1 XMP marker or plain xpacket
    return b"http://ns.adobe.com/xap/1.0/" in raw or b"<x:xmpmeta" in raw


def _extract_xmp_app1(raw: bytes) -> bytes | None:
    """Return the full APP1 segment bytes that contain XMP (marker + length + payload)."""
    if raw[:2] != b"\xff\xd8":
        return None
    i = 2
    n = len(raw)
    while i + 4 <= n:
        if raw[i] != 0xFF:
            break
        marker = raw[i : i + 2]
        if marker in (b"\xff\xd8", b"\xff\xd9"):
            i += 2
            continue
        if marker == b"\xff\xda":  # SOS — image data follows
            break
        length = int.from_bytes(raw[i + 2 : i + 4], "big")
        seg_end = i + 2 + length
        if seg_end > n:
            break
        payload = raw[i + 4 : seg_end]
        if marker == b"\xff\xe1" and (
            payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00") or b"<x:xmpmeta" in payload
        ):
            return raw[i:seg_end]
        i = seg_end
    return None


def _inject_xmp_app1(jpeg_bytes: bytes, xmp_seg: bytes) -> bytes:
    """Insert/replace XMP APP1 right after SOI."""
    if jpeg_bytes[:2] != b"\xff\xd8":
        return jpeg_bytes
    # Strip existing XMP APP1s then insert ours after SOI
    out = bytearray(b"\xff\xd8")
    i = 2
    n = len(jpeg_bytes)
    while i + 4 <= n:
        if jpeg_bytes[i] != 0xFF:
            out.extend(jpeg_bytes[i:])
            break
        marker = jpeg_bytes[i : i + 2]
        if marker == b"\xff\xda":
            out.extend(xmp_seg)
            out.extend(jpeg_bytes[i:])
            return bytes(out)
        if marker in (b"\xff\xd8",):
            i += 2
            continue
        if marker == b"\xff\xd9":
            out.extend(xmp_seg)
            out.extend(jpeg_bytes[i:])
            return bytes(out)
        length = int.from_bytes(jpeg_bytes[i + 2 : i + 4], "big")
        seg_end = i + 2 + length
        payload = jpeg_bytes[i + 4 : seg_end]
        is_xmp = marker == b"\xff\xe1" and (
            payload.startswith(b"http://ns.adobe.com/xap/1.0/\x00") or b"<x:xmpmeta" in payload
        )
        if not is_xmp:
            out.extend(jpeg_bytes[i:seg_end])
        i = seg_end
    else:
        out.extend(xmp_seg)
    return bytes(out)


def load_image(path: Path) -> np.ndarray:
    """Load as float32 RGB in [0,1] (sRGB-encoded), applying EXIF orientation."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def save_jpeg(
    path: Path,
    img: np.ndarray,
    source_path: Path | None = None,
    quality: int = 95,
) -> None:
    """Save JPEG quality 95, 4:4:4; copy EXIF + XMP from source when available."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    pil = Image.fromarray(u8, mode="RGB")

    exif_bytes = b""
    xmp_seg = None
    if source_path is not None and Path(source_path).exists():
        raw = Path(source_path).read_bytes()
        try:
            exif_dict = piexif.load(raw)
            # Drop thumbnail to avoid size issues; keep 0th/Exif/GPS
            exif_dict.pop("thumbnail", None)
            if "1st" in exif_dict:
                exif_dict["1st"] = {}
            exif_bytes = piexif.dump(exif_dict)
        except Exception as e:  # noqa: BLE001
            log.debug("no/invalid EXIF on %s: %s", source_path, e)
        xmp_seg = _extract_xmp_app1(raw)

    buf = io.BytesIO()
    save_kwargs: dict = {
        "format": "JPEG",
        "quality": quality,
        "subsampling": 0,  # 4:4:4
        "optimize": False,
    }
    if exif_bytes:
        save_kwargs["exif"] = exif_bytes
    pil.save(buf, **save_kwargs)
    data = buf.getvalue()
    if xmp_seg is not None:
        data = _inject_xmp_app1(data, xmp_seg)
    path.write_bytes(data)

    # Verify GPano / XMP still present when source had it
    if xmp_seg is not None and not _has_xmp(path.read_bytes()):
        log.warning("XMP missing after save: %s", path)


def validate_equirect(path: Path, tol: float = 0.01) -> dict:
    """Return metadata dict; raises AspectRatioError info via caller aggregation."""
    with Image.open(path) as im:
        w, h = im.size
        mode = im.mode
        # bit depth heuristic
        bit_depth = 16 if im.mode in ("I;16", "I;16B", "I") else 8
    ratio = w / max(h, 1)
    ok = abs(ratio - 2.0) <= 2.0 * tol
    raw = path.read_bytes() if path.suffix.lower() in {".jpg", ".jpeg"} else b""
    has_xmp = _has_xmp(raw) if raw else False
    has_exif = False
    try:
        piexif.load(str(path))
        has_exif = True
    except Exception:  # noqa: BLE001
        has_exif = False
    return {
        "path": path,
        "width": w,
        "height": h,
        "ratio": ratio,
        "ok_ratio": ok,
        "bit_depth": bit_depth,
        "mode": mode,
        "has_xmp": has_xmp,
        "has_exif": has_exif,
    }


def build_manifest(
    frames_dir: Path,
    out_csv: Path,
    stop_on_bad_ratio: bool = True,
) -> pd.DataFrame:
    """Walk frames, validate 2:1, write data/manifest.csv."""
    frames_dir = Path(frames_dir)
    rows = []
    bad: list[str] = []
    for dataset_dir in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
        images, _ = collect_images(dataset_dir)
        for img_path in images:
            meta = validate_equirect(img_path)
            rel = img_path.relative_to(dataset_dir)
            frame_id = rel.as_posix().rsplit(".", 1)[0].replace("/", "__")
            if not meta["ok_ratio"]:
                bad.append(f"{dataset_dir.name}/{rel} ({meta['width']}x{meta['height']})")
            rows.append(
                {
                    "dataset": dataset_dir.name,
                    "frame_id": frame_id,
                    "path": str(img_path.resolve()),
                    "width": meta["width"],
                    "height": meta["height"],
                    "has_xmp": meta["has_xmp"],
                    "has_exif": meta["has_exif"],
                    "ok_ratio": meta["ok_ratio"],
                }
            )
    if bad and stop_on_bad_ratio:
        msg = "Non-2:1 images found (tol ±1%). Decide how to handle:\n  - " + "\n  - ".join(bad)
        raise AspectRatioError(msg)
    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("wrote manifest %s (%d frames)", out_csv, len(df))
    return df


def ingest(zips_dir: Path, frames_dir: Path, manifest_csv: Path) -> pd.DataFrame:
    extract_zips(zips_dir, frames_dir)
    return build_manifest(frames_dir, manifest_csv)


def resize_preview(img: np.ndarray, size: tuple[int, int] = (2048, 1024)) -> np.ndarray:
    """Resize to preview WxH with area interpolation."""
    import cv2

    w, h = size
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def iter_manifest(manifest: Path | pd.DataFrame, datasets: Iterable[str] | None = None):
    df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest
    if datasets is not None:
        ds = set(datasets)
        df = df[df["dataset"].isin(ds)]
    for row in df.itertuples(index=False):
        yield row
