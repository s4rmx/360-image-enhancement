"""Fast Local Laplacian Filter (Paris / Aubry et al.)."""

from __future__ import annotations

import cv2
import numpy as np

from enhance.common import luminance_linear, recombine_colour, srgb_to_linear, linear_to_srgb
from enhance.equirect import clamp_pad, crop_equirect, pad_equirect
from enhance.methods.base import register


def _gaussian_pyramid(img: np.ndarray, n_levels: int) -> list[np.ndarray]:
    pyr = [img]
    cur = img
    for _ in range(n_levels - 1):
        if min(cur.shape[:2]) < 2:
            break
        cur = cv2.pyrDown(cur)
        pyr.append(cur)
    return pyr


def _laplacian_pyramid(img: np.ndarray, n_levels: int) -> list[np.ndarray]:
    g = _gaussian_pyramid(img, n_levels)
    lap: list[np.ndarray] = []
    for i in range(len(g) - 1):
        size = (g[i].shape[1], g[i].shape[0])
        up = cv2.pyrUp(g[i + 1], dstsize=size)
        lap.append(g[i] - up)
    lap.append(g[-1])
    return lap


def _collapse_laplacian(lap: list[np.ndarray]) -> np.ndarray:
    img = lap[-1]
    for i in range(len(lap) - 2, -1, -1):
        size = (lap[i].shape[1], lap[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + lap[i]
    return img


def _remap(i: np.ndarray, g: float, sigma_r: float, alpha_d: float, beta: float) -> np.ndarray:
    """Edge-aware remapping function for reference value g."""
    d = i - g
    ad = np.abs(d)
    sign = np.sign(d)
    sign = np.where(sign == 0, 1.0, sign)
    detail = g + sign * sigma_r * np.power(ad / sigma_r, alpha_d)
    edges = g + sign * (beta * (ad - sigma_r) + sigma_r)
    return np.where(ad <= sigma_r, detail, edges).astype(np.float32)


@register("llf")
def llf(img: np.ndarray, params: dict) -> np.ndarray:
    sigma_r = float(params.get("sigma_r", 0.4))
    alpha_d = float(params.get("alpha_d", 0.6))
    beta = float(params.get("beta", 0.7))
    n_samples = int(params.get("n_samples", 10))
    n_levels = int(params.get("n_levels", 8))

    lin = srgb_to_linear(img)
    y = luminance_linear(lin)
    l = np.log(y + 1e-4).astype(np.float32)

    pad = clamp_pad(2 ** max(n_levels - 1, 1), l.shape[0], l.shape[1])
    lp = pad_equirect(l, pad)

    # Effective levels limited by size
    max_levels = 1
    s = min(lp.shape[:2])
    while s >= 2 and max_levels < n_levels:
        s //= 2
        max_levels += 1
    n_levels = max_levels

    g_pyr = _gaussian_pyramid(lp, n_levels)
    # Sample reference values across range of l
    lo, hi = float(np.min(lp)), float(np.max(lp))
    if hi <= lo + 1e-6:
        out_l = crop_equirect(lp, pad)
        y_out = np.exp(out_l).astype(np.float32)
        rgb_s = linear_to_srgb(lin)
        return recombine_colour(rgb_s, luminance_linear(lin), y_out)

    gs = np.linspace(lo, hi, n_samples, dtype=np.float64)

    # Accumulate output Laplacian by streaming over g_j
    out_lap = [np.zeros_like(level) for level in g_pyr]
    # Also need to track which samples contribute — we interpolate per pixel
    # Algorithm: for each g_j, build remapped Laplacian; then for each level,
    # linearly interpolate between bracketing g_j based on Gaussian pyramid value.

    remapped_laps: list[list[np.ndarray]] = []
    for g in gs:
        rem = _remap(lp, float(g), sigma_r, alpha_d, beta)
        remapped_laps.append(_laplacian_pyramid(rem, n_levels))

    for lev in range(n_levels):
        gp = g_pyr[lev]
        t = (gp - lo) / (hi - lo) * (n_samples - 1)
        t = np.clip(t, 0, n_samples - 1)
        i0 = np.floor(t).astype(np.int32)
        i1 = np.minimum(i0 + 1, n_samples - 1)
        a = (t - i0).astype(np.float32)
        lap0 = np.zeros_like(gp)
        lap1 = np.zeros_like(gp)
        for j in range(n_samples):
            lap0 = np.where(i0 == j, remapped_laps[j][lev], lap0)
            lap1 = np.where(i1 == j, remapped_laps[j][lev], lap1)
        out_lap[lev] = ((1.0 - a) * lap0 + a * lap1).astype(np.float32)

    out_padded = _collapse_laplacian(out_lap)
    out_l = crop_equirect(out_padded, pad)
    if out_l.shape[:2] != l.shape[:2]:
        out_l = cv2.resize(out_l, (l.shape[1], l.shape[0]), interpolation=cv2.INTER_LINEAR)

    y_out = np.exp(out_l).astype(np.float32)
    # Recombine in linear then encode
    ratio = y_out / np.maximum(y, 1e-4)
    lin_out = np.clip(lin * ratio[..., None], 0.0, 1.0)
    return linear_to_srgb(lin_out).astype(np.float32)
