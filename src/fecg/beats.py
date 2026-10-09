"""Fetal R-peak detection and beat-level scoring."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt


def detect_beats(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float] = (8.0, 35.0),
    min_rr_s: float = 0.3,
    rel_threshold: float = 0.3,
    refine_ms: float = 40.0,
) -> np.ndarray:
    """Detect QRS complexes in a 1-D ECG with an energy-envelope (Pan-Tompkins style) detector.

    The signal is band-passed to the QRS band, squared and smoothed; peaks of that envelope
    must be at least ``min_rr_s`` apart and exceed ``rel_threshold`` of the local (4 s)
    maximum envelope, so the detector adapts to slow amplitude changes. Each detection is
    then refined to the sample of largest ``|x|`` near the envelope peak.
    """
    sos = butter(2, band, btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, x)
    env = uniform_filter1d(y**2, size=max(1, int(0.06 * fs)))
    local_max = maximum_filter1d(env, size=int(4 * fs))
    peaks, _ = find_peaks(env, distance=int(min_rr_s * fs), height=rel_threshold * local_max)
    if peaks.size == 0:
        return peaks.astype(np.int64)
    r = int(refine_ms / 1000 * fs)
    refined = [max(0, p - r) + int(np.argmax(np.abs(y[max(0, p - r) : p + r + 1]))) for p in peaks]
    return np.unique(np.asarray(refined, dtype=np.int64))


def match_beats(reference: np.ndarray, detected: np.ndarray, tol_samples: float) -> tuple[int, int, int]:
    """One-to-one matching of detections to reference beats; returns ``(TP, FP, FN)``."""
    reference, detected = np.sort(reference), np.sort(detected)
    used = np.zeros(detected.size, dtype=bool)
    tp = 0
    for r in reference:
        if detected.size == 0:
            break
        j = int(np.searchsorted(detected, r))
        candidates = [k for k in (j - 1, j) if 0 <= k < detected.size and not used[k]]
        candidates = [k for k in candidates if abs(detected[k] - r) <= tol_samples]
        if candidates:
            best = min(candidates, key=lambda k: abs(detected[k] - r))
            used[best] = True
            tp += 1
    return tp, int(detected.size - tp), int(reference.size - tp)


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def score_beats(
    reference: np.ndarray, signal: np.ndarray, fs: float, tol_ms: float, trim_s: float = 1.0
) -> dict[str, float]:
    """Detect beats in ``signal`` and score them against annotated ``reference`` beats.

    The first and last ``trim_s`` seconds are ignored (transform edge effects).
    """
    lo, hi = int(trim_s * fs), signal.size - int(trim_s * fs)
    detected = detect_beats(signal, fs)
    ref = reference[(reference >= lo) & (reference < hi)]
    det = detected[(detected >= lo) & (detected < hi)]
    return precision_recall_f1(*match_beats(ref, det, tol_ms / 1000 * fs))
