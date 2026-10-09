import numpy as np
from scipy.signal import butter, sosfiltfilt

from fecg.transforms import CWTConfig, cwt, icwt, synthesis_weights


def _band_noise(cfg, seconds, lo, hi, seed=1):
    x = np.random.default_rng(seed).standard_normal(int(cfg.fs * seconds))
    return sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=cfg.fs, output="sos"), x)


def test_scalogram_shape_and_dtype():
    cfg = CWTConfig()
    w = cwt(np.random.default_rng(0).standard_normal(1000), cfg)
    assert w.shape == (cfg.n_scales, 1000) and w.dtype == np.complex64


def test_frequencies_are_ascending_and_in_band():
    f = CWTConfig().freqs
    assert np.all(np.diff(f) > 0) and f[0] == 1.0 and f[-1] == 100.0


def test_energy_concentrates_at_the_tone_frequency():
    cfg = CWTConfig()
    t = np.arange(int(cfg.fs * 8)) / cfg.fs
    mag = np.abs(cwt(np.sin(2 * np.pi * 20 * t), cfg))[:, 500:-500].mean(1)
    assert abs(cfg.freqs[np.argmax(mag)] - 20) < 3


def test_inverse_transform_has_unit_gain_and_high_correlation():
    cfg = CWTConfig()
    x = _band_noise(cfg, 40, 3, 40)
    y = icwt(cwt(x, cfg), cfg)
    e = slice(1000, -1000)
    assert np.corrcoef(x[e], y[e])[0, 1] > 0.99
    assert abs(np.dot(x[e], y[e]) / np.dot(y[e], y[e]) - 1) < 0.05


def test_synthesis_weights_match_icwt():
    cfg = CWTConfig()
    w = cwt(np.random.default_rng(2).standard_normal(500), cfg)
    np.testing.assert_allclose(icwt(w, cfg), synthesis_weights(cfg) @ w.real, rtol=1e-5, atol=1e-6)
