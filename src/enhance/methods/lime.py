"""LIME: Low-light Image Enhancement via Illumination Map Estimation (Guo et al. 2017)."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from enhance.equirect import clamp_pad, crop_equirect, pad_equirect
from enhance.methods.base import register


def _build_weight_matrices(l_hat: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Strategy-II weights W_d = 1 / (|∇_d L̂| + ε)."""
    # horizontal gradient with circular wrap
    dx = np.roll(l_hat, -1, axis=1) - l_hat
    # vertical (no wrap at poles for the solver; edges use forward diff)
    dy = np.zeros_like(l_hat)
    dy[:-1] = l_hat[1:] - l_hat[:-1]
    wh = 1.0 / (np.abs(dx) + eps)
    wv = 1.0 / (np.abs(dy) + eps)
    return wh.astype(np.float64), wv.astype(np.float64)


def _solve_illumination(l_hat: np.ndarray, alpha: float, eps: float) -> np.ndarray:
    """Solve (Id + α Σ_d D_dᵀ diag(W_d) D_d) t = vec(L̂) with circular D_h."""
    h, w = l_hat.shape
    n = h * w
    wh, wv = _build_weight_matrices(l_hat, eps)

    # Build sparse Laplacian-like system via finite differences
    # For each pixel, contribution from horizontal circular edges and vertical edges
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    def add(r: int, c: int, v: float) -> None:
        rows.append(r)
        cols.append(c)
        data.append(v)

    # Identity
    for i in range(n):
        add(i, i, 1.0)

    # Horizontal circular differences: edge between (y,x) and (y, x+1 mod w)
    for y in range(h):
        for x in range(w):
            i = y * w + x
            j = y * w + ((x + 1) % w)
            wt = alpha * float(wh[y, x])
            add(i, i, wt)
            add(j, j, wt)
            add(i, j, -wt)
            add(j, i, -wt)

    # Vertical differences: edge between (y,x) and (y+1,x)
    for y in range(h - 1):
        for x in range(w):
            i = y * w + x
            j = (y + 1) * w + x
            wt = alpha * float(wv[y, x])
            add(i, i, wt)
            add(j, j, wt)
            add(i, j, -wt)
            add(j, i, -wt)

    a = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    b = l_hat.reshape(-1).astype(np.float64)
    # Diagonal preconditioner
    diag = a.diagonal()
    m_inv = sparse.diags(1.0 / np.maximum(diag, 1e-8))
    t, info = cg(a, b, M=m_inv, rtol=1e-4, maxiter=500)
    if info != 0:
        # fall back toward initial estimate on non-convergence
        t = 0.7 * t + 0.3 * b
    return np.clip(t.reshape(h, w), 1e-3, 1.0).astype(np.float32)


@register("lime")
def lime(img: np.ndarray, params: dict) -> np.ndarray:
    alpha = float(params.get("alpha", 0.15))
    gamma = float(params.get("gamma", 0.8))
    eps = float(params.get("eps", 1e-3))
    solve_width = int(params.get("solve_width", 1024))
    guided_radius = int(params.get("guided_radius", 8))
    guided_eps = float(params.get("guided_eps", 1e-3))

    h, w = img.shape[:2]
    l_hat_full = np.max(img, axis=2).astype(np.float32)

    if w > solve_width:
        sh = int(round(h * solve_width / w))
        l_small = cv2.resize(l_hat_full, (solve_width, sh), interpolation=cv2.INTER_AREA)
    else:
        l_small = l_hat_full

    t_small = _solve_illumination(l_small, alpha=alpha, eps=eps)
    t_up = cv2.resize(t_small, (w, h), interpolation=cv2.INTER_LINEAR)

    # Guided filter upsample refinement
    p = clamp_pad(max(guided_radius * 2, 8), h, w)
    guide = pad_equirect(l_hat_full, p)
    src = pad_equirect(t_up, p)
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        refined = cv2.ximgproc.guidedFilter(guide, src, guided_radius, guided_eps)
    else:
        # Fallback: bilateral-ish smooth
        refined = cv2.bilateralFilter(src, d=guided_radius * 2 + 1, sigmaColor=0.1, sigmaSpace=guided_radius)
    t = crop_equirect(refined, p)
    t = np.clip(t, eps, 1.0).astype(np.float32)

    denom = np.maximum(np.power(t, gamma), eps)
    out = img / denom[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
