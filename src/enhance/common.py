"""Shared colour / tone pipeline used by every method."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from enhance.equirect import lat_weights

log = logging.getLogger(__name__)


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    a = 0.055
    return np.where(
        x <= 0.0031308, 12.92 * x, (1 + a) * np.power(np.clip(x, 0, None), 1 / 2.4) - a
    ).astype(np.float32)


def luminance_linear(rgb_lin: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    ).astype(np.float32)


def luminance_srgb(rgb: np.ndarray) -> np.ndarray:
    """Approximate luminance from sRGB-encoded values (for metrics/preview)."""
    return (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = values.reshape(-1).astype(np.float64)
    w = np.broadcast_to(weights[:, None] if values.ndim == 2 else weights, values.shape)
    w = w.reshape(-1).astype(np.float64)
    s = w.sum()
    if s <= 0:
        return float(np.mean(v))
    return float(np.sum(w * v) / s)


def weighted_percentile(values: np.ndarray, weights_1d: np.ndarray, q: float) -> float:
    """Lat-weighted percentile of a 2D array; ``weights_1d`` has length H."""
    v = values.astype(np.float64)
    h, w = v.shape
    ww = np.broadcast_to(weights_1d[:, None], (h, w)).reshape(-1)
    flat = v.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    flat_s = flat[order]
    ww_s = ww[order]
    cdf = np.cumsum(ww_s)
    total = cdf[-1]
    if total <= 0:
        return float(np.percentile(flat, q))
    target = (q / 100.0) * total
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = min(max(idx, 0), len(flat_s) - 1)
    return float(flat_s[idx])


def weighted_median(values: np.ndarray, weights_1d: np.ndarray) -> float:
    return weighted_percentile(values, weights_1d, 50.0)


def white_balance_shades_of_gray(
    img: np.ndarray, p: float = 6.0, alpha: float = 0.3
) -> np.ndarray:
    """Shades-of-Gray WB in linear RGB with partial strength α."""
    lin = srgb_to_linear(img)
    h = lin.shape[0]
    wts = lat_weights(h)
    gains = []
    for c in range(3):
        ch = np.clip(lin[..., c], 0, None)
        # e_c = (Σ w·I^p / Σ w)^(1/p)
        powered = np.power(ch, p)
        e_c = weighted_mean(powered, wts) ** (1.0 / p)
        gains.append(max(e_c, 1e-8))
    gains = np.array(gains, dtype=np.float64)
    e_bar = gains.mean()
    g = e_bar / gains
    g_partial = 1.0 + alpha * (g - 1.0)
    out = lin * g_partial.astype(np.float32)
    return np.clip(linear_to_srgb(out), 0.0, 1.0).astype(np.float32)


def auto_levels_gamma(
    y: np.ndarray,
    p_lo: float = 0.5,
    p_hi: float = 99.5,
    apply_gamma: bool = True,
) -> np.ndarray:
    """Percentile stretch (+ optional midtone gamma) on a luminance map."""
    wts = lat_weights(y.shape[0])
    lo = weighted_percentile(y, wts, p_lo)
    hi = weighted_percentile(y, wts, p_hi)
    if hi <= lo + 1e-8:
        y1 = np.clip(y, 0, 1).astype(np.float32)
    else:
        y1 = np.clip((y - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if not apply_gamma:
        return y1
    med = weighted_median(y1, wts)
    med = float(np.clip(med, 1e-4, 1 - 1e-4))
    gamma = np.log(0.5) / np.log(med)
    gamma = float(np.clip(gamma, 0.5, 2.0))
    return np.power(y1, gamma).astype(np.float32)


def recombine_colour(rgb: np.ndarray, y_in: np.ndarray, y_out: np.ndarray) -> np.ndarray:
    """Hue-preserving luminance recombine; logs clipped pixel fraction."""
    ratio = y_out / np.maximum(y_in, 1e-4)
    out = rgb * ratio[..., None]
    clipped = float(np.mean((out < 0) | (out > 1)))
    if clipped > 0.01:
        log.debug("colour recombine clipped %.2f%% of pixels", 100 * clipped)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def vibrance(img: np.ndarray, k: float = 0.2, c_max: float = 128.0) -> np.ndarray:
    """Mild vibrance in Lab: boost low-chroma more than high-chroma."""
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    c = np.sqrt(a * a + b * b)
    s = 1.0 + k * (1.0 - np.clip(c / c_max, 0, 1))
    lab[..., 1] = np.clip(s * a + 128.0, 0, 255)
    lab[..., 2] = np.clip(s * b + 128.0, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return (out.astype(np.float32) / 255.0).astype(np.float32)


def blend(original: np.ndarray, enhanced: np.ndarray, t: float) -> np.ndarray:
    t = float(np.clip(t, 0.0, 1.0))
    if t >= 1.0:
        return enhanced
    if t <= 0.0:
        return original
    return ((1.0 - t) * original + t * enhanced).astype(np.float32)


def apply_shared_pipeline(
    img: np.ndarray,
    core_fn,
    params: dict[str, Any],
    shared: dict[str, Any],
) -> np.ndarray:
    """
    I → [pre] WB → METHOD CORE → [post] auto-levels on Y (no gamma)
      → optional vibrance → strength blend → clip
    """
    work = img
    if shared.get("white_balance", True):
        wb = shared.get("wb", {})
        work = white_balance_shades_of_gray(
            work, p=float(wb.get("p", 6)), alpha=float(wb.get("alpha", 0.3))
        )
    enhanced = core_fn(work, params)
    if shared.get("post_levels", True):
        lv = shared.get("levels", {})
        y = luminance_srgb(enhanced)
        y2 = auto_levels_gamma(
            y,
            p_lo=float(lv.get("p_lo", 0.5)),
            p_hi=float(lv.get("p_hi", 99.5)),
            apply_gamma=False,
        )
        enhanced = recombine_colour(enhanced, y, y2)
    if shared.get("vibrance", False):
        enhanced = vibrance(enhanced, k=float(shared.get("vibrance_k", 0.2)))
    t = float(shared.get("strength", 1.0))
    out = blend(img, enhanced, t)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
