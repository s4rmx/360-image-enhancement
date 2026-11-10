import numpy as np

from enhance.common import apply_shared_pipeline
from enhance.methods.base import get_method, list_methods


def test_methods_deterministic():
    rng = np.random.default_rng(1)
    img = rng.random((48, 96, 3), dtype=np.float32)
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
        "msrcr": {"sigmas": [4, 10, 20], "ref_width": 96, "work_width": 96},
        "lime": {"alpha": 0.15, "gamma": 0.8, "solve_width": 48, "guided_radius": 3},
        "fusion": {"ev": [0, 1], "pad": 8},
        "llf": {"n_levels": 3, "n_samples": 4},
        "wb_only": {},
    }
    for name in list_methods():
        if name == "original":
            continue
        core = get_method(name)
        p = params.get(name, {})
        a = apply_shared_pipeline(img, core, p, shared)
        b = apply_shared_pipeline(img, core, p, shared)
        assert np.array_equal(a, b), name
