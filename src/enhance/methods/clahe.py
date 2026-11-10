"""CLAHE on L* with equirect-aware tile padding (Zuiderveld 1994)."""

from __future__ import annotations

import cv2
import numpy as np

from enhance.common import auto_levels_gamma
from enhance.equirect import clamp_pad, crop_equirect, pad_equirect
from enhance.methods.base import register


@register("clahe")
def clahe(img: np.ndarray, params: dict) -> np.ndarray:
    clip_limit = float(params.get("clip_limit", 2.0))
    grid = params.get("grid", [8, 16])
    rows, cols = int(grid[0]), int(grid[1])

    h, w = img.shape[:2]
    tw = max(w // cols, 1)
    th = max(h // rows, 1)
    p = clamp_pad(max(tw, th), h, w)

    # Lab on float via uint8 path for OpenCV; keep precision with uint16 CLAHE
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_star = lab[..., 0] / 255.0  # [0,1]

    l_star = auto_levels_gamma(l_star, apply_gamma=True)

    padded = pad_equirect(l_star, p)
    ph, pw = padded.shape[:2]
    # grid that matches padded size: original + 2 border tiles
    grid_h = max(rows + 2, 2)
    grid_w = max(cols + 2, 2)
    # ensure tile sizes work; OpenCV CLAHE uses floor division
    clahe_op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_w, grid_h))
    # uint16 for precision
    u16 = np.clip(padded * 65535.0, 0, 65535).astype(np.uint16)
    eq = clahe_op.apply(u16).astype(np.float32) / 65535.0
    eq = crop_equirect(eq, p)
    # safety resize if pad/crop drifted
    if eq.shape[:2] != (h, w):
        eq = cv2.resize(eq, (w, h), interpolation=cv2.INTER_LINEAR)

    lab[..., 0] = np.clip(eq * 255.0, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return (out.astype(np.float32) / 255.0).astype(np.float32)
