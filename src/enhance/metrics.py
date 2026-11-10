"""No-reference and artifact metrics for equirect enhancement outputs."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage.restoration import estimate_sigma

from enhance.common import luminance_srgb
from enhance.equirect import lat_weights, seam_delta_y

log = logging.getLogger(__name__)

_IQA = None


def _get_iqa():
    global _IQA
    if _IQA is None:
        import pyiqa
        import torch

        device = torch.device("cpu")
        _IQA = {
            "niqe": pyiqa.create_metric("niqe", device=device),
            "brisque": pyiqa.create_metric("brisque", device=device),
        }
    return _IQA


def _to_torch_img(img: np.ndarray):
    import torch

    # pyiqa expects NCHW float [0,1]
    t = torch.from_numpy(np.transpose(img.astype(np.float32), (2, 0, 1))).unsqueeze(0)
    return t.clamp(0, 1)


def niqe_brisque(img: np.ndarray) -> tuple[float, float]:
    iqa = _get_iqa()
    t = _to_torch_img(img)
    with __import__("torch").no_grad():
        n = float(iqa["niqe"](t).item())
        b = float(iqa["brisque"](t).item())
    return n, b


def lightness_order_error(enhanced: np.ndarray, original: np.ndarray, size=(200, 100)) -> float:
    """LOE: mean count of lightness-order flips vs original (lower better)."""
    def light(x):
        small = cv2.resize(x, size, interpolation=cv2.INTER_AREA)
        return np.max(small, axis=2)

    le = light(enhanced).reshape(-1)
    lo = light(original).reshape(-1)
    n = le.size
    # Pairwise order: sample for speed if large
    # Full O(n^2) is too heavy even at 20k; use random pairs
    rng = np.random.default_rng(0)
    m = min(5000, n * 10)
    i = rng.integers(0, n, size=m)
    j = rng.integers(0, n, size=m)
    mask = i != j
    i, j = i[mask], j[mask]
    order_o = lo[i] >= lo[j]
    order_e = le[i] >= le[j]
    return float(np.mean(order_o != order_e))


def entropy_y(img: np.ndarray) -> float:
    y = (np.clip(luminance_srgb(img), 0, 1) * 255).astype(np.uint8)
    hist = np.bincount(y.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def rms_contrast(img: np.ndarray) -> float:
    y = luminance_srgb(img)
    w = lat_weights(y.shape[0])
    mu = float(np.average(y, weights=np.broadcast_to(w[:, None], y.shape)))
    var = float(np.average((y - mu) ** 2, weights=np.broadcast_to(w[:, None], y.shape)))
    return float(np.sqrt(var))


def mean_brightness(img: np.ndarray) -> float:
    y = luminance_srgb(img)
    w = lat_weights(y.shape[0])
    return float(np.average(y, weights=np.broadcast_to(w[:, None], y.shape)))


def clipping_pct(img: np.ndarray) -> float:
    y = luminance_srgb(img)
    lo = y <= (2 / 255.0)
    hi = y >= (253 / 255.0)
    return float(100.0 * np.mean(lo | hi))


def colourfulness(img: np.ndarray) -> float:
    """Hasler & Süsstrunk 2003."""
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2))


def noise_amplification(enhanced: np.ndarray, original: np.ndarray) -> float:
    ye = luminance_srgb(enhanced)
    yo = luminance_srgb(original)
    sig_e = float(estimate_sigma(ye, channel_axis=None, average_sigmas=True))
    sig_o = float(estimate_sigma(yo, channel_axis=None, average_sigmas=True))
    if sig_o < 1e-8:
        return float("nan")
    return sig_e / sig_o


def ssim_vs_original(enhanced: np.ndarray, original: np.ndarray) -> float:
    ye = luminance_srgb(enhanced)
    yo = luminance_srgb(original)
    return float(structural_similarity(yo, ye, data_range=1.0))


def compute_metrics(
    enhanced: np.ndarray,
    original: np.ndarray,
    full_enhanced: np.ndarray | None = None,
    full_original: np.ndarray | None = None,
    skip_iqa: bool = False,
) -> dict[str, Any]:
    """Compute metrics on preview-sized arrays; seam delta on full-res if provided."""
    row: dict[str, Any] = {}
    if not skip_iqa:
        try:
            n, b = niqe_brisque(enhanced)
            row["niqe"] = n
            row["brisque"] = b
        except Exception as e:  # noqa: BLE001
            log.warning("IQA metrics failed: %s", e)
            row["niqe"] = float("nan")
            row["brisque"] = float("nan")
    else:
        row["niqe"] = float("nan")
        row["brisque"] = float("nan")

    row["loe"] = lightness_order_error(enhanced, original)
    row["entropy"] = entropy_y(enhanced)
    row["rms_contrast"] = rms_contrast(enhanced)
    row["mean_brightness"] = mean_brightness(enhanced)
    row["clipping_pct"] = clipping_pct(enhanced)
    row["colourfulness"] = colourfulness(enhanced)
    row["noise_amp"] = noise_amplification(enhanced, original)
    row["ssim"] = ssim_vs_original(enhanced, original)

    fe = full_enhanced if full_enhanced is not None else enhanced
    fo = full_original if full_original is not None else original
    row["seam_delta"] = seam_delta_y(fe, fo)
    return row
