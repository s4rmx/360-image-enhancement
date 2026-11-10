"""Multi-Scale Retinex with Colour Restoration (Jobson et al. / IPOL 2014)."""

from __future__ import annotations

import cv2
import numpy as np

from enhance.equirect import crop_equirect, lat_weights, pad_equirect
from enhance.methods.base import register
from enhance.common import weighted_percentile


def _gaussian_blur_equirect(img: np.ndarray, sigma: float) -> np.ndarray:
    """Blur with equirect padding; do not rely on OpenCV border modes."""
    from enhance.equirect import clamp_pad

    h, w = img.shape[:2]
    # kernel support ≈ 3σ; clamp so tiny images (tests) still run
    p = clamp_pad(max(int(np.ceil(3 * sigma)), 1), h, w)
    # Also shrink sigma if pad can't cover 3σ
    if p > 0:
        sigma = min(sigma, p / 3.0)
    padded = pad_equirect(img, p)
    k = max(int(2 * np.ceil(3 * max(sigma, 0.5)) + 1), 3)
    if k % 2 == 0:
        k += 1
    blurred = cv2.GaussianBlur(
        padded, (k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE
    )
    return crop_equirect(blurred, p) if p > 0 else blurred


def _simplest_colour_balance(img: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Clip pct% low/high per channel (lat-weighted) and stretch to [0,1]."""
    out = np.empty_like(img)
    wts = lat_weights(img.shape[0])
    for c in range(img.shape[2]):
        ch = img[..., c]
        lo = weighted_percentile(ch, wts, pct)
        hi = weighted_percentile(ch, wts, 100.0 - pct)
        if hi <= lo + 1e-8:
            out[..., c] = np.clip(ch, 0, 1)
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
    return out.astype(np.float32)


@register("msrcr")
def msrcr(img: np.ndarray, params: dict) -> np.ndarray:
    sigmas_ref = list(params.get("sigmas", [15, 80, 250]))
    ref_width = int(params.get("ref_width", 2048))
    work_width = int(params.get("work_width", 2048))
    alpha = float(params.get("alpha", 125.0))
    beta = float(params.get("beta", 46.0))
    gain = float(params.get("gain", 192.0))
    offset = float(params.get("offset", -30.0))
    balance_pct = float(params.get("balance_pct", 1.0))

    h, w = img.shape[:2]
    scale = w / float(ref_width)
    sigmas = [s * scale for s in sigmas_ref]

    # Work on downsampled copy for speed; upsample illumination
    if w > work_width:
        wh = int(round(h * work_width / w))
        small = cv2.resize(img, (work_width, wh), interpolation=cv2.INTER_AREA)
        sigma_scale = work_width / float(w)
        sigmas_work = [s * sigma_scale for s in sigmas]
    else:
        small = img
        sigmas_work = sigmas

    # Scale to [1, 256] to avoid log(0)
    i = np.clip(small, 0, 1).astype(np.float32) * 255.0 + 1.0
    n_scales = len(sigmas_work)
    w_i = 1.0 / n_scales
    msr = np.zeros_like(i)
    for sigma in sigmas_work:
        blur = _gaussian_blur_equirect(i, sigma)
        blur = np.maximum(blur, 1.0)
        msr += w_i * (np.log(i) - np.log(blur))

    # Colour restoration
    sum_c = np.sum(i, axis=2, keepdims=True)
    cr = beta * (np.log(alpha * i) - np.log(np.maximum(sum_c, 1.0)))
    msrcr = gain * (cr * msr + offset)

    # Upsample MSRCR result to full res if needed
    if msrcr.shape[:2] != (h, w):
        msrcr = cv2.resize(msrcr, (w, h), interpolation=cv2.INTER_LINEAR)

    out = _simplest_colour_balance(msrcr, pct=balance_pct)
    return out.astype(np.float32)
