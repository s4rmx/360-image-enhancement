import numpy as np

from enhance.common import white_balance_shades_of_gray


def test_wb_moves_cast_toward_neutral():
    # Grey scene with reddish cast
    base = np.full((64, 128, 3), 0.4, dtype=np.float32)
    cast = base.copy()
    cast[..., 0] *= 1.3  # boost red
    cast[..., 2] *= 0.7  # cut blue
    cast = np.clip(cast, 0, 1)
    alpha = 0.3
    out = white_balance_shades_of_gray(cast, p=6, alpha=alpha)
    # Channel means should move toward each other vs input
    in_means = cast.reshape(-1, 3).mean(axis=0)
    out_means = out.reshape(-1, 3).mean(axis=0)
    in_spread = float(in_means.max() - in_means.min())
    out_spread = float(out_means.max() - out_means.min())
    assert out_spread < in_spread
    # Partial strength: not fully neutralized
    assert out_spread > 0.01
