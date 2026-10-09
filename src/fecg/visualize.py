"""Figures: scalogram comparison, waveform comparison, beat detection and training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _extent(n_time: int, fs: float, freqs: np.ndarray) -> list[float]:
    return [0.0, n_time / fs, 0, len(freqs)]


def _freq_ticks(ax, freqs: np.ndarray) -> None:
    idx = [int(np.argmin(np.abs(freqs - f))) for f in (1, 5, 10, 25, 50, 100) if freqs[0] <= f <= freqs[-1]]
    ax.set_yticks(idx)
    ax.set_yticklabels([f"{freqs[i]:.0f}" for i in idx])


def plot_scalograms(
    abd_cwt: list[np.ndarray],
    true_cwt: list[np.ndarray],
    pred_cwt: list[np.ndarray],
    titles: list[str],
    freqs: np.ndarray,
    fs: float,
    mag_norm: float,
    path: str | Path,
) -> None:
    """One row per example: abdominal input | true fetal | predicted fetal | absolute error."""
    rows = len(abd_cwt)
    fig, axes = plt.subplots(rows, 4, figsize=(15, 3.4 * rows), squeeze=False)
    for r in range(rows):
        inp = np.abs(abd_cwt[r][0])
        true = np.clip(np.abs(true_cwt[r]) / mag_norm, 0, 1)
        pred = np.clip(np.abs(pred_cwt[r]) / mag_norm, 0, 1)
        panels = [
            (inp / np.percentile(inp, 99.5), "Abdominal input (lead 1)", "jet"),
            (true, "True fetal scalogram", "jet"),
            (pred, "Predicted fetal scalogram", "jet"),
            (np.abs(pred - true), "Absolute error", "inferno"),
        ]
        for c, (img, name, cmap) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(
                np.clip(img, 0, 1), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1,
                extent=_extent(img.shape[1], fs, freqs),
            )
            _freq_ticks(ax, freqs)
            ax.set_title(name if r == 0 else "")
            ax.set_xlabel("Time (s)")
            if c == 0:
                ax.set_ylabel(f"{titles[r]}\nFrequency (Hz)")
            if c == 3:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_waveforms(
    true: list[np.ndarray], pred: list[np.ndarray], titles: list[str], fs: float, path: str | Path
) -> None:
    fig, axes = plt.subplots(len(true), 1, figsize=(12, 3.0 * len(true)), squeeze=False)
    for ax, t, p, title in zip(axes[:, 0], true, pred, titles):
        time = np.arange(t.size) / fs
        ax.plot(time, t, color="tab:green", lw=1.2, label="True fetal ECG (scalp)")
        ax.plot(time, p, color="tab:red", lw=1.2, label="Predicted fetal ECG")
        ax.fill_between(time, 0, p - t, color="gray", alpha=0.3, label="Error")
        ax.set_title(title)
        ax.set_ylabel("Amplitude (normalised)")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_beats(
    abd: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    reference_beats: np.ndarray,
    detected_beats: np.ndarray,
    fs: float,
    start_s: float,
    duration_s: float,
    path: str | Path,
) -> None:
    """Raw abdominal lead, true and predicted fetal ECG with annotated vs detected beats."""
    a, b = int(start_s * fs), int((start_s + duration_s) * fs)
    time = np.arange(a, b) / fs
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time, abd[a:b], color="tab:blue", lw=0.9)
    axes[0].set_title("Abdominal lead 1 (maternal ECG dominates)")
    axes[1].plot(time, true[a:b], color="tab:green", lw=1.0)
    axes[1].set_title("Direct fetal scalp ECG (ground truth) with annotated beats")
    axes[2].plot(time, pred[a:b], color="tab:red", lw=1.0)
    axes[2].set_title("U-Net prediction with detected beats")
    for beats, ax, color in ((reference_beats, axes[1], "k"), (reference_beats, axes[2], "k")):
        for s in beats[(beats >= a) & (beats < b)]:
            ax.axvline(s / fs, color=color, ls=":", lw=1)
    d = detected_beats[(detected_beats >= a) & (detected_beats < b)]
    axes[2].plot(d / fs, pred[d], "v", color="darkred", ms=7, label="detected beat")
    axes[2].plot([], [], color="k", ls=":", label="annotated beat")
    axes[2].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylabel("Norm. amp.")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_prediction(
    abd: np.ndarray, pred: np.ndarray, beats: np.ndarray, fs: float, start_s: float, duration_s: float,
    path: str | Path,
) -> None:
    """Raw abdominal lead and the extracted fetal ECG with detected beats (no reference needed)."""
    a = int(min(start_s, max(0.0, pred.size / fs - duration_s)) * fs)
    b = min(pred.size, a + int(duration_s * fs))
    time = np.arange(a, b) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    axes[0].plot(time, abd[a:b], color="tab:blue", lw=0.9)
    axes[0].set_title("Abdominal lead 1 (mixture of maternal and fetal ECG)")
    axes[1].plot(time, pred[a:b], color="tab:red", lw=1.0)
    d = beats[(beats >= a) & (beats < b)]
    axes[1].plot(d / fs, pred[d], "v", color="darkred", ms=7, label="detected fetal beat")
    axes[1].set_title("Extracted fetal ECG")
    axes[1].legend(loc="upper right", fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set_ylabel("Norm. amp.")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_history(history_path: str | Path, path: str | Path) -> None:
    history = json.loads(Path(history_path).read_text())
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, key, title in zip(axes, ("loss", "coef", "signal"), ("Total loss", "Scalogram MSE", "Waveform MSE")):
        ax.plot(epochs, [h["train"][key] for h in history], label="train")
        ax.plot(epochs, [h["val"][key] for h in history], label="validation")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
