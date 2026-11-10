import numpy as np

from enhance.equirect import crop_equirect, lat_weights, pad_equirect


def test_pad_crop_roundtrip():
    rng = np.random.default_rng(0)
    img = rng.random((64, 128, 3), dtype=np.float32)
    p = 8
    out = crop_equirect(pad_equirect(img, p), p)
    assert out.shape == img.shape
    np.testing.assert_allclose(out, img, atol=0)


def test_pole_padding_flip_roll():
    h, w = 32, 64
    img = np.zeros((h, w), dtype=np.float32)
    img[0, :] = np.arange(w, dtype=np.float32)
    img[-1, :] = np.arange(w, dtype=np.float32)[::-1]
    p = 4
    padded = pad_equirect(img, p)
    # Top padded rows: flip of top p rows, rolled by W/2
    expected_top = np.flip(img[:p], axis=0)
    expected_top = np.roll(expected_top, w // 2, axis=1)
    np.testing.assert_allclose(padded[:p, p:-p], expected_top)
    expected_bot = np.flip(img[-p:], axis=0)
    expected_bot = np.roll(expected_bot, w // 2, axis=1)
    np.testing.assert_allclose(padded[-p:, p:-p], expected_bot)


def test_lat_weights_symmetric_sum():
    w = lat_weights(100)
    assert w.shape == (100,)
    np.testing.assert_allclose(w, w[::-1], atol=1e-10)
    # Mid-latitude weight near 1, poles near 0
    assert w[0] < 0.1
    assert w[50] > 0.9
    assert w.sum() > 0
