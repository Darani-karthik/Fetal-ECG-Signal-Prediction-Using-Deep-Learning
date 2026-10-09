"""Training loop: AdamW + one-cycle schedule, mixed precision, best-on-validation checkpointing."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fecg.data.dataset import ScalogramDataset, load_stats
from fecg.losses import FetalScalogramLoss
from fecg.models import UNet
from fecg.utils import cwt_config, get_device, seed_everything


def _run_epoch(model, loader, loss_fn, device, optimizer=None, scheduler=None, scaler=None, desc=""):
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    n = 0
    for x, y, sig in tqdm(loader, desc=desc, leave=False):
        x, y, sig = x.to(device), y.to(device), sig.to(device)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=scaler is not None and scaler.is_enabled()):
                pred = model(x)
            loss, parts = loss_fn(pred, y, sig)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        bs = x.size(0)
        n += bs
        for k, v in {"loss": float(loss.detach()), **parts}.items():
            totals[k] = totals.get(k, 0.0) + v * bs
    return {k: v / n for k, v in totals.items()}


def train(cfg: dict, device: torch.device | None = None) -> Path:
    """Train the U-Net described by ``cfg`` and return the path of the best checkpoint."""
    seed_everything(cfg["seed"])
    device = device or get_device()
    d, c, m, t = cfg["data"], cfg["cwt"], cfg["model"], cfg["train"]

    stats_file = d.get("stats_file", "stats.json")
    train_ds = ScalogramDataset(
        d["processed_dir"], d["split"]["train"], c["window"], c["stride"], bool(t["augment"]), stats_file
    )
    val_ds = ScalogramDataset(
        d["processed_dir"], d["split"]["val"], c["window"], c["stride"], stats_file=stats_file
    )
    loader_kw = dict(batch_size=t["batch_size"], num_workers=t["num_workers"], pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    print(f"device={device}  train windows={len(train_ds)}  val windows={len(val_ds)}")

    model = UNet(m["in_channels"], m["out_channels"], m["base_channels"], m["depth"], m["dropout"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {n_params / 1e6:.2f} M")

    loss_fn = FetalScalogramLoss(
        cwt_config(cfg), train_ds.target_scale, t["mag_weight"], t["signal_weight"], t["corr_weight"]
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=t["lr"], epochs=t["epochs"], steps_per_epoch=len(train_loader), pct_start=0.15
    )
    scaler = torch.amp.GradScaler(device.type, enabled=bool(t["amp"]) and device.type == "cuda")

    ckpt_dir = Path(t["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "unet_best.pt"
    history, best_val = [], float("inf")

    for epoch in range(1, t["epochs"] + 1):
        t0 = time.time()
        tr = _run_epoch(model, train_loader, loss_fn, device, optimizer, scheduler, scaler, f"train {epoch}")
        va = _run_epoch(model, val_loader, loss_fn, device, desc=f"val {epoch}")
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(
            f"epoch {epoch:02d}/{t['epochs']}  train {tr['loss']:.4f}  val {va['loss']:.4f} "
            f"(coef {va['coef']:.4f} sig {va['signal']:.4f} corr {1 - va['corr']:.3f})  {time.time() - t0:.0f}s"
        )
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "stats": load_stats(d["processed_dir"], stats_file),
                    "epoch": epoch,
                    "val_loss": best_val,
                },
                best_path,
            )

    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best val loss {best_val:.4f} -> {best_path}")
    return best_path
