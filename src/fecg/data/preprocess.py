"""Turn raw EDF recordings into aligned (abdominal scalogram -> fetal scalogram) pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from fecg.data.annotations import read_beat_annotations
from fecg.data.edf import read_edf
from fecg.transforms import CWTConfig, cwt
from fecg.utils import cwt_config


def condition_signals(x: np.ndarray, fs_raw: float, fs: float, band: tuple[float, float]) -> np.ndarray:
    """Band-pass, down-sample and z-score each channel of ``x`` (channels x samples)."""
    sos = butter(4, band, btype="bandpass", fs=fs_raw, output="sos")
    x = sosfiltfilt(sos, x, axis=-1)
    if fs != fs_raw:
        ratio = fs_raw / fs
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(f"fs_raw/fs must be an integer, got {ratio}")
        x = resample_poly(x, up=1, down=int(round(ratio)), axis=-1)
    x = x - np.median(x, axis=-1, keepdims=True)
    return x / x.std(axis=-1, keepdims=True)


def preprocess_record(name: str, cfg: dict, out_root: str | Path) -> Path:
    """Preprocess one record and write its arrays under ``<out_root>/<name>/``."""
    d = cfg["data"]
    raw_dir = Path(d["raw_dir"])
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)

    rec = read_edf(raw_dir / f"{name}.edf")
    assert rec.fs == d["fs_raw"], f"{name}: expected {d['fs_raw']} Hz, got {rec.fs}"
    abd = np.stack([rec.channel(c) for c in d["abdominal_channels"]])
    fetal = rec.channel(d["fetal_channel"])[None]

    band = tuple(d["bandpass"])
    abd = condition_signals(abd, d["fs_raw"], d["fs"], band)
    fetal = condition_signals(fetal, d["fs_raw"], d["fs"], band)[0]

    cwt_cfg: CWTConfig = cwt_config(cfg)
    abd_cwt = np.stack([cwt(ch, cwt_cfg) for ch in abd])  # (4, S, T) complex64
    fetal_cwt = cwt(fetal, cwt_cfg)  # (S, T) complex64

    beats = read_beat_annotations(raw_dir / f"{name}.edf.qrs")
    beats = np.round(beats * d["fs"] / d["fs_raw"]).astype(np.int64)
    beats = beats[(beats >= 0) & (beats < fetal.size)]

    np.save(out / "abd.npy", abd.astype(np.float32))
    np.save(out / "fetal.npy", fetal.astype(np.float32))
    np.save(out / "abd_cwt.npy", abd_cwt)
    np.save(out / "fetal_cwt.npy", fetal_cwt)
    np.save(out / "beats.npy", beats)
    (out / "meta.json").write_text(
        json.dumps({"record": name, "fs": d["fs"], "n_samples": int(fetal.size), "n_beats": int(beats.size)})
    )
    return out


def fit_stats(cfg: dict) -> dict:
    """Fit the fetal-scalogram normalisation constants on the training subjects only.

    * ``target_scale`` - std of the real/imaginary coefficients (the network target is divided by it)
    * ``mag_norm``     - 99.5th percentile of ``|W|``, maps magnitudes to [0, 1] for image metrics

    Written to ``<processed_dir>/<data.stats_file>`` (default ``stats.json``).
    """
    out_root = Path(cfg["data"]["processed_dir"])
    parts = [np.load(out_root / r / "fetal_cwt.npy", mmap_mode="r") for r in cfg["data"]["split"]["train"]]
    values = np.concatenate([np.concatenate([p.real.ravel(), p.imag.ravel()]) for p in parts])
    magnitudes = np.concatenate([np.abs(p).ravel() for p in parts])
    stats = {
        "target_scale": float(values.std()),
        "mag_norm": float(np.percentile(magnitudes, 99.5)),
    }
    (out_root / cfg["data"].get("stats_file", "stats.json")).write_text(json.dumps(stats))
    print(f"target_scale = {stats['target_scale']:.4f}, mag_norm = {stats['mag_norm']:.4f}")
    return stats


def preprocess_all(cfg: dict) -> dict:
    """Preprocess every record of the split, then fit the normalisation statistics."""
    out_root = Path(cfg["data"]["processed_dir"])
    split = cfg["data"]["split"]
    for name in [r for key in ("train", "val", "test") for r in split[key]]:
        print(f"preprocessing {name} ...")
        preprocess_record(name, cfg, out_root)
    return fit_stats(cfg)
