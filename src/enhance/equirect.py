"""Equirectangular geometry helpers: wrap padding, latitude weights, seam crop."""

from __future__ import annotations

import numpy as np


def lat_weights(h: int) -> np.ndarray:
    """Row weights w_i = cos(φ_i) with φ_i = π·(0.5 − (i+0.5)/H)."""
    i = np.arange(h, dtype=np.float64)
    phi = np.pi * (0.5 - (i + 0.5) / h)
    return np.cos(phi).astype(np.float64)


def max_pad(h: int, w: int) -> int:
    """Largest pad that still fits both axes (pole flip needs p ≤ H)."""
    return max(0, min(h, w))


def clamp_pad(p: int, h: int, w: int) -> int:
    """Clamp requested pad so pad_equirect never rejects small images."""
    return int(max(0, min(int(p), max_pad(h, w))))


def pad_equirect(img: np.ndarray, p: int) -> np.ndarray:
    """Pad equirect image with horizontal wrap and geometrically correct poles.

    Horizontal: wrap by p columns on each side.
    Vertical: the p rows above the top are the top p rows flipped vertically
    and rolled by W/2 (crossing a pole lands on the opposite meridian). Same
    at the bottom.

    ``p`` is clamped to fit the image so callers can request a theoretical
    support and still run on small test images.
    """
    if p <= 0:
        return img
    h, w = img.shape[:2]
    p = clamp_pad(p, h, w)
    if p <= 0:
        return img

    roll = w // 2
    top = np.flip(img[:p], axis=0)
    top = np.roll(top, roll, axis=1)
    bottom = np.flip(img[-p:], axis=0)
    bottom = np.roll(bottom, roll, axis=1)
    vert = np.concatenate([top, img, bottom], axis=0)
    return np.pad(vert, ((0, 0), (p, p)) + ((0, 0),) * (img.ndim - 2), mode="wrap")


def pad_amount(img: np.ndarray, p: int) -> int:
    """Actual pad that ``pad_equirect`` will apply for this image."""
    if p <= 0:
        return 0
    h, w = img.shape[:2]
    return clamp_pad(p, h, w)


def crop_equirect(img: np.ndarray, p: int) -> np.ndarray:
    """Inverse of pad_equirect."""
    if p <= 0:
        return img
    return img[p:-p, p:-p]


def seam_crop(img: np.ndarray, k: int = 256) -> np.ndarray:
    """Concatenate last k columns with first k columns for seam visualisation."""
    k = min(k, img.shape[1] // 2)
    return np.concatenate([img[:, -k:], img[:, :k]], axis=1)


def seam_delta_y(img: np.ndarray, ref: np.ndarray | None = None) -> float:
    """Mean abs Y diff between col 0 and col W-1, in 8-bit units.

    If ``ref`` is given, returns (delta_out - delta_in) as specified for the
    seam artifact metric (output minus input).
    """
    from enhance.common import luminance_srgb

    def _delta(x: np.ndarray) -> float:
        y = luminance_srgb(x)
        return float(np.mean(np.abs(y[:, 0] - y[:, -1])) * 255.0)

    d_out = _delta(img)
    if ref is None:
        return d_out
    return d_out - _delta(ref)
