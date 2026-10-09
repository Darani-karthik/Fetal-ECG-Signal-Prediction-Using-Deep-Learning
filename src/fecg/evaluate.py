"""Evaluate a trained model on held-out recordings and write metrics + figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fecg import visualize as viz
from fecg.beats import detect_beats, score_beats
from fecg.data.dataset import load_stats
from fecg.inference import load_checkpoint, predict_scalogram, reconstruct_signal
from fecg.metrics import pearson, scalogram_errors, snr_db, windowed_correlation
from fecg.transforms import icwt
from fecg.utils import cwt_config, get_device


def evaluate_record(model, name: str, cfg: dict, stats: dict, device: torch.device) -> tuple[dict, dict]:
    """Metrics and the arrays needed for plotting for one held-out record."""
    d, c = cfg["data"], cfg["cwt"]
    fs = float(d["fs"])
    rec = Path(d["processed_dir"]) / name
    abd_cwt = np.load(rec / "abd_cwt.npy", mmap_mode="r")
    true_cwt = np.load(rec / "fetal_cwt.npy")
    abd, fetal = np.load(rec / "abd.npy"), np.load(rec / "fetal.npy")
    beats = np.load(rec / "beats.npy")
    cwt_cfg = cwt_config(cfg)

    pred_cwt = predict_scalogram(model, abd_cwt, c["window"], c["stride"], stats["target_scale"], device)
    pred = reconstruct_signal(pred_cwt, cwt_cfg)
    oracle = icwt(true_cwt, cwt_cfg)  # best any scalogram model could do through the transform

    trim = int(1.0 * fs)  # ignore transform edge effects at both ends
    core = slice(trim, fetal.size - trim)
    tol = cfg["evaluate"]["beat_tolerance_ms"]

    # raw abdominal baseline: best single lead (selected by correlation with the reference)
    lead_corr = [abs(pearson(a[core], fetal[core])) for a in abd]
    best_lead = int(np.argmax(lead_corr))
    snr_in = snr_db(fetal[core], abd[best_lead][core])
    snr_out = snr_db(fetal[core], pred[core])
    win_corr = windowed_correlation(fetal[core], pred[core], c["window"], c["window"])

    lead_beats = [score_beats(beats, a, fs, tol) for a in abd]
    metrics = {
        "record": name,
        "scalogram": scalogram_errors(pred_cwt, true_cwt, stats["mag_norm"]),
        "waveform": {
            "correlation": pearson(fetal[core], pred[core]),
            "correlation_window_median": float(np.median(win_corr)),
            "correlation_window_min": float(win_corr.min()),
            "rmse": float(np.sqrt(np.mean((fetal[core] - pred[core]) ** 2))),
            "snr_in_db": snr_in,
            "snr_out_db": snr_out,
            "snr_gain_db": snr_out - snr_in,
            "transform_ceiling_correlation": pearson(fetal[core], oracle[core]),
        },
        "beats": {
            "unet": score_beats(beats, pred, fs, tol),
            "direct_fetal_ecg_detector": score_beats(beats, fetal, fs, tol),
            "best_raw_abdominal_lead": max(lead_beats, key=lambda m: m["f1"]),
        },
    }
    arrays = {
        "abd_cwt": abd_cwt, "true_cwt": true_cwt, "pred_cwt": pred_cwt,
        "abd": abd, "fetal": fetal, "pred": pred, "beats": beats,
        "win_corr_idx": _window_choice(fetal, pred, c["window"], trim),
    }
    return metrics, arrays


def _window_choice(fetal: np.ndarray, pred: np.ndarray, window: int, trim: int) -> dict[str, int]:
    """Start sample of the median-correlation and the worst-correlation window (no cherry-picking)."""
    starts = np.arange(trim, fetal.size - trim - window, window)
    corr = np.array([pearson(fetal[s : s + window], pred[s : s + window]) for s in starts])
    order = np.argsort(corr)
    return {"worst": int(starts[order[0]]), "median": int(starts[order[len(order) // 2]])}


def _aggregate(per_record: list[dict]) -> dict:
    def mean_of(path: tuple[str, ...]) -> float:
        vals = []
        for m in per_record:
            for key in path:
                m = m[key]
            vals.append(m)
        return float(np.mean(vals))

    return {
        "scalogram": {k: mean_of(("scalogram", k)) for k in ("mse", "rmse", "mae")},
        "waveform": {k: mean_of(("waveform", k)) for k in per_record[0]["waveform"]},
        "beats": {
            who: {k: mean_of(("beats", who, k)) for k in ("precision", "recall", "f1")}
            for who in per_record[0]["beats"]
        },
    }


def format_markdown(result: dict) -> str:
    m = result["aggregate"]
    lines = [
        f"Test subject(s): {', '.join(result['test_records'])}  (never seen during training)",
        "",
        "| Domain | Metric | Value |",
        "|---|---|---|",
    ]
    for k in ("mse", "rmse", "mae"):
        lines.append(f"| Scalogram | {k.upper()} | {m['scalogram'][k]:.4f} |")
    w = m["waveform"]
    lines += [
        f"| Waveform | Pearson correlation | {w['correlation']:.3f} |",
        f"| Waveform | Median windowed correlation (2 s) | {w['correlation_window_median']:.3f} |",
        f"| Waveform | SNR, best raw abdominal lead -> U-Net | {w['snr_in_db']:.2f} -> {w['snr_out_db']:.2f} dB |",
        f"| Waveform | SNR gain | {w['snr_gain_db']:+.2f} dB |",
        f"| Waveform | Correlation ceiling of the CWT/ICWT transform | {w['transform_ceiling_correlation']:.3f} |",
    ]
    lines += ["", "| Beat detection (+-50 ms) | Precision | Recall | F1 |", "|---|---|---|---|"]
    names = {
        "unet": "U-Net output",
        "direct_fetal_ecg_detector": "Direct scalp ECG (detector sanity check)",
        "best_raw_abdominal_lead": "Best raw abdominal lead (baseline)",
    }
    for key, label in names.items():
        b = m["beats"][key]
        lines.append(f"| {label} | {b['precision']:.3f} | {b['recall']:.3f} | {b['f1']:.3f} |")
    return "\n".join(lines) + "\n"


def evaluate(
    cfg: dict, checkpoint: str | Path, device: torch.device | None = None, make_figures: bool = True
) -> dict:
    """Score ``checkpoint`` on ``data.split.test``; writes metrics (and, optionally, figures)."""
    device = device or get_device()
    model, _ = load_checkpoint(checkpoint, device)
    stats = load_stats(cfg["data"]["processed_dir"], cfg["data"].get("stats_file", "stats.json"))
    out_dir = Path(cfg["evaluate"]["results_dir"])
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fs, window = float(cfg["data"]["fs"]), cfg["cwt"]["window"]
    freqs = cwt_config(cfg).freqs
    per_record = []
    for name in cfg["data"]["split"]["test"]:
        metrics, a = evaluate_record(model, name, cfg, stats, device)
        per_record.append(metrics)
        if not make_figures:
            continue

        picks = [("median-correlation window", a["win_corr_idx"]["median"]), ("worst window", a["win_corr_idx"]["worst"])]
        sl = [slice(s, s + window) for _, s in picks]
        titles = [f"{name}: {label}" for label, _ in picks]
        viz.plot_scalograms(
            [np.asarray(a["abd_cwt"][:, :, s]) for s in sl], [a["true_cwt"][:, s] for s in sl],
            [a["pred_cwt"][:, s] for s in sl], titles, freqs, fs, stats["mag_norm"],
            fig_dir / f"scalograms_{name}.png",
        )
        viz.plot_waveforms([a["fetal"][s] for s in sl], [a["pred"][s] for s in sl], titles, fs, fig_dir / f"waveforms_{name}.png")
        start = a["win_corr_idx"]["median"] / fs
        viz.plot_beats(
            a["abd"][0], a["fetal"], a["pred"], a["beats"], detect_beats(a["pred"], fs),
            fs, start_s=start, duration_s=8.0, path=fig_dir / f"beats_{name}.png",
        )

    result = {
        "test_records": cfg["data"]["split"]["test"],
        "checkpoint": str(checkpoint),
        "per_record": per_record,
        "aggregate": _aggregate(per_record),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (out_dir / "metrics.md").write_text(format_markdown(result), encoding="utf-8")
    history = Path(checkpoint).parent / "history.json"
    if make_figures and history.exists():
        viz.plot_history(history, fig_dir / "training_curves.png")
    print(format_markdown(result))
    return result
