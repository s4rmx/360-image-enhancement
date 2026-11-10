import numpy as np

from enhance.common import apply_shared_pipeline
from enhance.equirect import seam_delta_y
from enhance.methods.base import list_methods, get_method


def _synthetic_equirect(h=64, w=128):
    """Equirect test image with matching left/right edges (seamless wrap).

    Uses endpoint=True so col 0 and col W-1 share the same phase value,
    keeping the plan's seam-delta metric near zero on the input.
    """
    xs = np.linspace(0, 2 * np.pi, w, endpoint=True, dtype=np.float32)
    ys = np.linspace(0, np.pi, h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    base = 0.5 + 0.25 * np.cos(xv) * np.sin(yv)
    # Force exact edge match after float ops
    base[:, -1] = base[:, 0]
    img = np.stack([base, base * 0.95 + 0.02, base * 0.9 + 0.04], axis=-1).astype(np.float32)
    img[:, -1] = img[:, 0]
    return np.clip(img, 0, 1)


def test_methods_seam_delta():
    img = _synthetic_equirect()
    shared = {
        "white_balance": True,
        "post_levels": True,
        "vibrance": False,
        "strength": 1.0,
        "wb": {"p": 6, "alpha": 0.3},
        "levels": {"p_lo": 0.5, "p_hi": 99.5},
    }
    params = {
        "clahe": {"clip_limit": 2.0, "grid": [4, 8]},
        "msrcr": {"sigmas": [5, 15, 40], "ref_width": 128, "work_width": 128},
        "lime": {"alpha": 0.15, "gamma": 0.8, "solve_width": 64, "guided_radius": 4},
        "fusion": {"ev": [-1, 0, 1], "pad": 16},
        "llf": {"n_levels": 4, "n_samples": 5},
        "wb_only": {},
        "original": {},
    }
    for name in list_methods():
        if name == "original":
            out = img.copy()
        else:
            out = apply_shared_pipeline(img, get_method(name), params.get(name, {}), shared)
        delta = abs(seam_delta_y(out, img))
        assert delta < 1.0, f"{name} seam_delta={delta}"
