"""Run a trained U-Net over a whole recording and rebuild the 1-D fetal ECG."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from fecg.data.dataset import build_features, window_starts
from fecg.transforms import CWTConfig, icwt


@torch.no_grad()
def predict_scalogram(
    model: nn.Module,
    abd_cwt: np.ndarray,
    window: int,
    stride: int,
    target_scale: float,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Predict the fetal complex scalogram ``(S, T)`` for abdominal scalograms ``(4, S, T)``.

    Overlapping window predictions are blended with a Hann taper, which removes seams at
    window borders where the network has the least context.
    """
    model.eval()
    n_scales, n_samples = abd_cwt.shape[1], abd_cwt.shape[2]
    acc = np.zeros((2, n_scales, n_samples), dtype=np.float32)
    weight = np.zeros(n_samples, dtype=np.float32)
    taper = np.hanning(window + 2)[1:-1].astype(np.float32)  # strictly positive

    starts = window_starts(n_samples, window, stride)
    for b in range(0, len(starts), batch_size):
        chunk = starts[b : b + batch_size]
        x = np.stack([build_features(np.asarray(abd_cwt[:, :, s : s + window])) for s in chunk])
        out = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
        for s, o in zip(chunk, out):
            acc[:, :, s : s + window] += o * taper
            weight[s : s + window] += taper

    ri = acc / weight
    return ((ri[0] + 1j * ri[1]) * target_scale).astype(np.complex64)


def reconstruct_signal(scalogram: np.ndarray, cwt_cfg: CWTConfig) -> np.ndarray:
    """Inverse CWT of a complex scalogram -> 1-D fetal ECG (normalised units)."""
    return icwt(scalogram, cwt_cfg).astype(np.float32)


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the model stored in a checkpoint written by :mod:`fecg.engine`."""
    from fecg.models import UNet

    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = ckpt["config"]["model"]
    model = UNet(m["in_channels"], m["out_channels"], m["base_channels"], m["depth"], m.get("dropout", 0.0)).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt
