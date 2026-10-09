"""End-to-end smoke test on tiny synthetic recordings (no dataset download needed)."""

import json
from pathlib import Path

import numpy as np

from fecg.data.preprocess import preprocess_all
from fecg.engine import train
from fecg.evaluate import evaluate
from fecg.utils import load_config
from helpers import make_synthetic_record


def test_preprocess_train_evaluate(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for i, name in enumerate(["a", "b", "c"]):
        make_synthetic_record(raw, name, seconds=40, seed=i)

    cfg = load_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    cfg["data"].update(
        raw_dir=str(raw), processed_dir=str(tmp_path / "proc"),
        split={"train": ["a"], "val": ["b"], "test": ["c"]},
    )
    cfg["cwt"].update(n_scales=32, window=256, stride=128)
    cfg["model"].update(base_channels=4, depth=2)
    cfg["train"].update(epochs=2, batch_size=8, amp=False, checkpoint_dir=str(tmp_path / "ckpt"))
    cfg["evaluate"]["results_dir"] = str(tmp_path / "results")

    preprocess_all(cfg)
    proc = tmp_path / "proc" / "a"
    assert np.load(proc / "abd_cwt.npy", mmap_mode="r").shape == (4, 32, 40 * 250)
    assert np.load(proc / "beats.npy").size > 100
    assert json.loads((tmp_path / "proc" / "stats.json").read_text())["target_scale"] > 0

    import torch

    best = train(cfg, torch.device("cpu"))
    assert best.exists() and (tmp_path / "ckpt" / "history.json").exists()

    result = evaluate(cfg, best, torch.device("cpu"))
    assert 0 <= result["aggregate"]["beats"]["unet"]["f1"] <= 1
    assert (tmp_path / "results" / "metrics.json").exists()
    assert (tmp_path / "results" / "figures" / "scalograms_c.png").exists()

    # inference on a raw EDF file without any fetal reference
    from fecg.predict import predict_edf

    out = predict_edf(raw / "c.edf", best, torch.device("cpu"))
    assert out["fetal_ecg"].shape == (40 * 250,) and out["beats"].ndim == 1
    assert out["abdominal"].shape == (4, 40 * 250)


def test_cross_validation_runs_every_subject_as_test(tmp_path):
    import torch

    from fecg.crossval import cross_validate, fold_splits

    splits = fold_splits(["a", "b", "c", "d", "e"])
    assert sorted(s["test"][0] for s in splits) == ["a", "b", "c", "d", "e"]
    for s in splits:  # no subject appears in two roles
        roles = s["train"] + s["val"] + s["test"]
        assert len(roles) == len(set(roles)) == 5 and len(s["train"]) == 3

    raw = tmp_path / "raw"
    raw.mkdir()
    for i, name in enumerate(["a", "b", "c"]):
        make_synthetic_record(raw, name, seconds=30, seed=i)
    cfg = load_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    cfg["data"].update(
        raw_dir=str(raw), processed_dir=str(tmp_path / "proc"),
        split={"train": ["a"], "val": ["b"], "test": ["c"]},
    )
    cfg["cwt"].update(n_scales=16, window=128, stride=128)
    cfg["model"].update(base_channels=4, depth=2)
    cfg["train"].update(epochs=1, batch_size=8, amp=False)
    preprocess_all(cfg)
    summary = cross_validate(cfg, tmp_path / "cv", torch.device("cpu"))
    assert (tmp_path / "cv" / "cv_metrics.json").exists()
    assert 0 <= summary["beats.unet.f1"]["mean"] <= 1
