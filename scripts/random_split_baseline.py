"""Optional diagnostic: train/test on a *random split of overlapping windows*.

The main evaluation holds out whole subjects. A random window split is what many image
pipelines do by default, but here neighbouring windows overlap by 75 % and come from the same
five recordings, so near-duplicates of every test window sit in the training set. This script
quantifies how much that inflates the scores; it is NOT a valid estimate of performance on new
subjects. Metrics are per window (a random split has no continuous test signal to detect beats in).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from fecg.data.dataset import ScalogramDataset, load_stats
from fecg.engine import _run_epoch
from fecg.losses import FetalScalogramLoss, synthesize
from fecg.metrics import pearson, scalogram_errors
from fecg.models import UNet
from fecg.utils import cwt_config, get_device, load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--out", default="results/random_window_split.json")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    seed_everything(cfg["seed"])
    device = get_device(args.device)
    d, c, m, t = cfg["data"], cfg["cwt"], cfg["model"], cfg["train"]
    records = [r for key in ("train", "val", "test") for r in d["split"][key]]

    stats = load_stats(d["processed_dir"], d.get("stats_file", "stats.json"))
    full = ScalogramDataset(d["processed_dir"], records, c["window"], c["stride"], augment=False)
    perm = np.random.default_rng(cfg["seed"]).permutation(len(full))
    n_test, n_val = int(0.1 * len(full)), int(0.1 * len(full))
    test_idx, val_idx, train_idx = perm[:n_test], perm[n_test : n_test + n_val], perm[n_test + n_val :]
    print(f"windows: train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}  (random split)")

    kw = dict(batch_size=t["batch_size"], num_workers=0, pin_memory=device.type == "cuda")
    train_loader = DataLoader(Subset(full, train_idx), shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(Subset(full, val_idx), **kw)
    test_loader = DataLoader(Subset(full, test_idx), **kw)

    model = UNet(m["in_channels"], m["out_channels"], m["base_channels"], m["depth"], m["dropout"]).to(device)
    loss_fn = FetalScalogramLoss(
        cwt_config(cfg), stats["target_scale"], t["mag_weight"], t["signal_weight"], t["corr_weight"]
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=t["lr"], epochs=t["epochs"], steps_per_epoch=len(train_loader), pct_start=0.15
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(t["amp"]) and device.type == "cuda")

    best, best_state = float("inf"), None
    for epoch in range(1, t["epochs"] + 1):
        tr = _run_epoch(model, train_loader, loss_fn, device, opt, sched, scaler)
        va = _run_epoch(model, val_loader, loss_fn, device)
        print(f"epoch {epoch:02d}  train {tr['loss']:.4f}  val {va['loss']:.4f}")
        if va["loss"] < best:
            best, best_state = va["loss"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    model.eval()
    mag_err, corrs = [], []
    with torch.no_grad():
        for x, y, sig in test_loader:
            pred = model(x.to(device)).float()
            wave = synthesize(pred, loss_fn.weights).cpu().numpy()
            pred_c = (pred[:, 0] + 1j * pred[:, 1]).cpu().numpy() * stats["target_scale"]
            true_c = (y[:, 0] + 1j * y[:, 1]).numpy() * stats["target_scale"]
            for i in range(x.size(0)):
                mag_err.append(scalogram_errors(pred_c[i], true_c[i], stats["mag_norm"]))
                corrs.append(pearson(sig[i].numpy(), wave[i]))
    result = {
        "split": "random windows (leaky - for comparison only)",
        "test_windows": int(len(test_idx)),
        "scalogram": {k: float(np.mean([e[k] for e in mag_err])) for k in ("mse", "rmse", "mae")},
        "waveform_correlation_mean": float(np.mean(corrs)),
        "waveform_correlation_median": float(np.median(corrs)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
