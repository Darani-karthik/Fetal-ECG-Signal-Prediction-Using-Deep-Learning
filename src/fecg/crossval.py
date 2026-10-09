"""Leave-one-subject-out cross-validation over all recordings.

With only five subjects a single train/test split gives a noisy estimate, so every
subject takes a turn as the unseen test subject; the next subject (cyclically) is the
validation subject used for checkpoint selection and the remaining three are used to train.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from fecg.data.preprocess import fit_stats
from fecg.engine import train
from fecg.evaluate import evaluate
from fecg.utils import get_device

_SUMMARY_KEYS = [
    ("waveform", "correlation", "Waveform correlation"),
    ("waveform", "correlation_window_median", "Median 2 s windowed correlation"),
    ("waveform", "snr_gain_db", "SNR gain (dB)"),
    ("scalogram", "rmse", "Scalogram RMSE"),
    ("scalogram", "mae", "Scalogram MAE"),
]


def fold_splits(records: list[str]) -> list[dict[str, list[str]]]:
    n = len(records)
    return [
        {
            "train": [r for j, r in enumerate(records) if j not in (k, (k + 1) % n)],
            "val": [records[(k + 1) % n]],
            "test": [records[k]],
        }
        for k in range(n)
    ]


def cross_validate(cfg: dict, out_dir: str | Path, device: torch.device | None = None) -> dict:
    device = device or get_device()
    out_dir = Path(out_dir)
    split = cfg["data"]["split"]
    records = sorted({r for key in ("train", "val", "test") for r in split[key]})

    folds = []
    for k, fold_split in enumerate(fold_splits(records)):
        print(f"\n=== fold {k + 1}/{len(records)}: test={fold_split['test']} val={fold_split['val']} ===")
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["data"]["split"] = fold_split
        fold_cfg["data"]["stats_file"] = f"stats_fold{k}.json"
        fold_cfg["train"]["checkpoint_dir"] = str(out_dir / f"fold{k}" / "checkpoint")
        fold_cfg["evaluate"]["results_dir"] = str(out_dir / f"fold{k}")
        fit_stats(fold_cfg)
        best = train(fold_cfg, device)
        folds.append(evaluate(fold_cfg, best, device, make_figures=False))

    summary = _summarise(folds)
    (out_dir / "cv_metrics.json").write_text(json.dumps({"folds": folds, "summary": summary}, indent=2))
    (out_dir / "cv_metrics.md").write_text(format_markdown(folds, summary), encoding="utf-8")
    print(format_markdown(folds, summary))
    return summary


def _summarise(folds: list[dict]) -> dict:
    def stat(values: list[float]) -> dict[str, float]:
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    summary: dict = {}
    for group, key, _ in _SUMMARY_KEYS:
        summary[f"{group}.{key}"] = stat([f["aggregate"][group][key] for f in folds])
    for who in folds[0]["aggregate"]["beats"]:
        for key in ("precision", "recall", "f1"):
            summary[f"beats.{who}.{key}"] = stat([f["aggregate"]["beats"][who][key] for f in folds])
    return summary


def format_markdown(folds: list[dict], summary: dict) -> str:
    lines = [
        "| Test subject | Correlation | Median win. corr. | SNR gain (dB) | Beat F1 (U-Net) | Beat F1 (best raw lead) |",
        "|---|---|---|---|---|---|",
    ]
    for f in folds:
        a = f["aggregate"]
        lines.append(
            f"| {f['test_records'][0]} | {a['waveform']['correlation']:.3f} "
            f"| {a['waveform']['correlation_window_median']:.3f} | {a['waveform']['snr_gain_db']:+.2f} "
            f"| {a['beats']['unet']['f1']:.3f} | {a['beats']['best_raw_abdominal_lead']['f1']:.3f} |"
        )

    def ms(key: str, fmt: str = ".3f") -> str:
        return f"{summary[key]['mean']:{fmt}} ± {summary[key]['std']:{fmt.replace('+', '')}}"

    lines.append(
        f"| **mean ± std** | **{ms('waveform.correlation')}** | **{ms('waveform.correlation_window_median')}** "
        f"| **{ms('waveform.snr_gain_db', '+.2f')}** | **{ms('beats.unet.f1')}** "
        f"| **{ms('beats.best_raw_abdominal_lead.f1')}** |"
    )
    lines += [
        "",
        "| Beat detection (mean ± std over subjects) | Precision | Recall | F1 |",
        "|---|---|---|---|",
    ]
    names = {
        "unet": "U-Net output",
        "direct_fetal_ecg_detector": "Direct scalp ECG (detector sanity check)",
        "best_raw_abdominal_lead": "Best raw abdominal lead (baseline)",
    }
    for key, label in names.items():
        lines.append(
            f"| {label} | {ms(f'beats.{key}.precision')} | {ms(f'beats.{key}.recall')} | {ms(f'beats.{key}.f1')} |"
        )
    return "\n".join(lines) + "\n"
