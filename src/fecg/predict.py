"""Extract the fetal ECG from any EDF recording with abdominal leads (no reference needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from fecg.beats import detect_beats
from fecg.data.edf import read_edf
from fecg.data.preprocess import condition_signals
from fecg.inference import load_checkpoint, predict_scalogram, reconstruct_signal
from fecg.transforms import cwt
from fecg.utils import cwt_config, get_device


def predict_edf(
    edf_path: str | Path,
    checkpoint: str | Path,
    device: torch.device | None = None,
    channels: list[str] | None = None,
) -> dict:
    """Run the trained model on the abdominal leads of ``edf_path``.

    Returns the fetal ECG (normalised units, 250 Hz by default), the detected fetal beat
    sample indices, the mean fetal heart rate and the conditioned abdominal leads.
    """
    device = device or get_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    cfg, stats = ckpt["config"], ckpt["stats"]
    d, c = cfg["data"], cfg["cwt"]

    rec = read_edf(edf_path)
    channels = channels or d["abdominal_channels"]
    if len(channels) != cfg["model"]["in_channels"] // 2:
        raise ValueError(f"the model expects {cfg['model']['in_channels'] // 2} abdominal channels")
    abd = condition_signals(
        np.stack([rec.channel(ch) for ch in channels]), rec.fs, d["fs"], tuple(d["bandpass"])
    )

    cwt_cfg = cwt_config(cfg)
    abd_cwt = np.stack([cwt(ch, cwt_cfg) for ch in abd])
    scalogram = predict_scalogram(model, abd_cwt, c["window"], c["stride"], stats["target_scale"], device)
    fetal = reconstruct_signal(scalogram, cwt_cfg)

    beats = detect_beats(fetal, float(d["fs"]))
    rr = np.diff(beats) / float(d["fs"])
    return {
        "fs": float(d["fs"]),
        "fetal_ecg": fetal,
        "beats": beats,
        "heart_rate_bpm": float(60.0 / np.median(rr)) if rr.size else float("nan"),
        "abdominal": abd,
    }
