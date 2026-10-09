"""Scalogram-domain and time-domain reconstruction metrics."""

from __future__ import annotations

import numpy as np


def scalogram_errors(pred: np.ndarray, true: np.ndarray, mag_norm: float) -> dict[str, float]:
    """MSE / RMSE / MAE between predicted and true fetal scalogram magnitudes.

    Both complex scalograms are converted to magnitude images and scaled to [0, 1] with
    ``mag_norm`` (a high percentile of the training-set magnitudes), as for an 8-bit image.
    """
    p = np.clip(np.abs(pred) / mag_norm, 0, 1)
    t = np.clip(np.abs(true) / mag_norm, 0, 1)
    err = p - t
    mse = float(np.mean(err**2))
    return {"mse": mse, "rmse": mse**0.5, "mae": float(np.mean(np.abs(err)))}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x - x.mean(), y - y.mean()
    denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    """SNR of ``estimate`` w.r.t. ``reference`` after optimal (least-squares) rescaling.

    ``10*log10( ||ref||^2 / ||ref - a*est||^2 )`` with ``a = <ref, est> / <est, est>``, so a
    pure gain mismatch is not penalised for either the input lead or the network output.
    """
    a = np.dot(reference, estimate) / max(np.dot(estimate, estimate), 1e-12)
    residual = reference - a * estimate
    return float(10 * np.log10(np.dot(reference, reference) / max(np.dot(residual, residual), 1e-12)))


def windowed_correlation(x: np.ndarray, y: np.ndarray, window: int, hop: int) -> np.ndarray:
    """Pearson correlation inside sliding windows (flat windows are skipped)."""
    out = [
        pearson(x[s : s + window], y[s : s + window])
        for s in range(0, x.size - window + 1, hop)
        if x[s : s + window].std() > 0 and y[s : s + window].std() > 0
    ]
    return np.asarray(out)
