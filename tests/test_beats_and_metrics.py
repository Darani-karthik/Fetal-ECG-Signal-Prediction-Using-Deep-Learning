import numpy as np

from fecg.beats import detect_beats, match_beats, precision_recall_f1, score_beats
from fecg.metrics import pearson, scalogram_errors, snr_db, windowed_correlation
from helpers import pulse_train

FS = 250.0


def _ecg(beat_times_s, seed=0, noise=0.02):
    n = int(60 * FS)
    beats = (np.asarray(beat_times_s) * FS).astype(int)
    rng = np.random.default_rng(seed)
    return pulse_train(n, beats, width=2.0, amp=1.0) + rng.normal(0, noise, n), beats


def test_detector_finds_regular_beats():
    times = np.arange(1.0, 58.0, 0.45)
    x, beats = _ecg(times)
    scores = score_beats(beats, x, FS, tol_ms=50)
    assert scores["f1"] > 0.98


def test_detector_ignores_flat_noise_floor_between_beats_and_negative_r_waves():
    x, beats = _ecg(np.arange(1.0, 58.0, 0.5))
    detected = detect_beats(-x, FS)  # inverted polarity
    assert match_beats(beats, detected, 0.05 * FS)[0] >= len(beats) - 2


def test_match_beats_is_one_to_one_and_respects_tolerance():
    ref = np.array([100, 200, 300])
    assert match_beats(ref, np.array([102, 199, 330]), tol_samples=5) == (2, 1, 1)
    # one detection cannot satisfy two reference beats
    assert match_beats(np.array([100, 104]), np.array([102]), tol_samples=5) == (1, 0, 1)


def test_precision_recall_f1_edge_cases():
    assert precision_recall_f1(0, 0, 0)["f1"] == 0.0
    r = precision_recall_f1(8, 2, 2)
    assert (r["precision"], r["recall"]) == (0.8, 0.8) and abs(r["f1"] - 0.8) < 1e-12


def test_pearson_and_snr():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000)
    assert pearson(x, 3 * x + 1) > 0.9999
    assert snr_db(x, 5 * x) > 80  # pure gain mismatch is not penalised
    assert snr_db(x, x + rng.standard_normal(1000)) < 10
    assert snr_db(x, x + 0.1 * rng.standard_normal(1000)) > snr_db(x, x + rng.standard_normal(1000))


def test_scalogram_errors_are_zero_for_identical_inputs_and_bounded():
    w = (np.random.default_rng(0).normal(size=(8, 8)) + 0j).astype(np.complex64)
    assert scalogram_errors(w, w, mag_norm=2.0)["mse"] == 0.0
    m = scalogram_errors(w, np.zeros_like(w), mag_norm=2.0)
    assert 0 < m["mae"] <= 1 and abs(m["rmse"] ** 2 - m["mse"]) < 1e-9


def test_windowed_correlation_skips_flat_windows():
    x = np.concatenate([np.sin(np.linspace(0, 20, 500)), np.zeros(500)])
    c = windowed_correlation(x, x, window=250, hop=250)
    assert c.size == 2 and np.allclose(c, 1.0)
