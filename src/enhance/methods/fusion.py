"""Exposure fusion on synthetic exposures (Mertens et al. 2007)."""

from __future__ import annotations

import cv2
import numpy as np

from enhance.common import linear_to_srgb, srgb_to_linear
from enhance.equirect import clamp_pad, crop_equirect, pad_equirect
from enhance.methods.base import register


@register("fusion")
def fusion(img: np.ndarray, params: dict) -> np.ndarray:
    # OpenCV MergeMertens / pyramids are not bit-stable with multi-threading
    cv2.setNumThreads(1)
    evs = list(params.get("ev", [-1, 0, 1, 2]))
    cw = float(params.get("contrast_weight", 1.0))
    sw = float(params.get("saturation_weight", 1.0))
    ew = float(params.get("exposure_weight", 1.0))
    h, w = img.shape[:2]
    pad = clamp_pad(int(params.get("pad", 256)), h, w)

    lin = srgb_to_linear(img)
    exposures_srgb = []
    for ev in evs:
        e = np.clip(lin * (2.0 ** float(ev)), 0.0, 1.0)
        exposures_srgb.append(linear_to_srgb(e))

    padded = [pad_equirect(e, pad) for e in exposures_srgb]
    # OpenCV MergeMertens expects float32 [0,1] or uint8
    merger = cv2.createMergeMertens(
        contrast_weight=cw, saturation_weight=sw, exposure_weight=ew
    )
    fused = merger.process([p.astype(np.float32) for p in padded])
    fused = np.clip(fused, 0.0, 1.0).astype(np.float32)
    out = crop_equirect(fused, pad)
    if out.shape[:2] != img.shape[:2]:
        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)
