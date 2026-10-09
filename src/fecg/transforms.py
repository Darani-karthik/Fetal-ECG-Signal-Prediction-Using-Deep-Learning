"""Continuous Wavelet Transform (CWT) scalograms and their (approximate) inverse.

A 1-D ECG becomes a 2-D complex image ``W[scale, time]``. Rows are ordered by
increasing frequency (row 0 = ``fmin``), so with ``origin="lower"`` plots read like
a conventional spectrogram.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pywt
from scipy.signal import butter, sosfiltfilt


@dataclass(frozen=True)
class CWTConfig:
    fs: float = 250.0
    wavelet: str = "cmor1.5-1.0"
    n_scales: int = 64
    fmin: float = 1.0
    fmax: float = 100.0

    @property
    def freqs(self) -> np.ndarray:
        """Analysis frequencies in Hz, ascending, log-spaced."""
        return np.geomspace(self.fmin, self.fmax, self.n_scales)

    @property
    def scales(self) -> np.ndarray:
        """Wavelet scales (in samples) matching ``freqs``."""
        return pywt.central_frequency(self.wavelet) * self.fs / self.freqs


def cwt(x: np.ndarray, cfg: CWTConfig) -> np.ndarray:
    """Complex scalogram of a 1-D signal, shape ``(n_scales, len(x))``, complex64."""
    coefs, _ = pywt.cwt(np.asarray(x, dtype=np.float64), cfg.scales, cfg.wavelet, method="fft")
    return coefs.astype(np.complex64)


def synthesis_weights(cfg: CWTConfig) -> np.ndarray:
    """Per-scale weights ``kappa / sqrt(scale)`` of the inverse transform.

    Uses the delta-function reconstruction of Torrence & Compo (1998): for log-spaced
    scales, ``x(t) ~ kappa * sum_j Re W(s_j, t) / sqrt(s_j)``. The constant ``kappa``
    (which absorbs the wavelet admissibility constant and the log spacing) is calibrated
    numerically, so the round trip ``icwt(cwt(x))`` has unit gain inside the band.
    """
    return _kappa(cfg) / np.sqrt(cfg.scales)


def icwt(coefs: np.ndarray, cfg: CWTConfig) -> np.ndarray:
    """Reconstruct a 1-D signal from a complex scalogram ``(n_scales, T)``."""
    w = synthesis_weights(cfg)
    return np.einsum("s,st->t", w, np.real(coefs), optimize=True)


@lru_cache(maxsize=16)
def _kappa(cfg: CWTConfig) -> float:
    rng = np.random.default_rng(0)
    n = int(cfg.fs * 80)
    noise = rng.standard_normal(n)
    # band-limited noise strictly inside the analysis band
    lo, hi = cfg.fmin * 2.0, min(cfg.fmax / 2.0, 0.45 * cfg.fs)
    sos = butter(4, [lo, hi], btype="bandpass", fs=cfg.fs, output="sos")
    x = sosfiltfilt(sos, noise)
    edge = int(cfg.fs * 10)  # drop transform edge effects
    raw = np.einsum("s,st->t", 1.0 / np.sqrt(cfg.scales), np.real(cwt(x, cfg)))
    x, raw = x[edge:-edge], raw[edge:-edge]
    return float(np.dot(x, raw) / np.dot(raw, raw))
