"""Training objective: scalogram regression + magnitude + time-domain consistency."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from fecg.transforms import CWTConfig, synthesis_weights


def magnitude(ri: torch.Tensor) -> torch.Tensor:
    """|W| from a (B, 2, S, W) real/imaginary tensor."""
    return torch.sqrt(ri[:, 0] ** 2 + ri[:, 1] ** 2 + 1e-8)


def synthesize(ri: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Differentiable inverse CWT of a (B, 2, S, W) tensor -> (B, W) signal."""
    return torch.einsum("s,bsw->bw", weights, ri[:, 0])


def correlation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``1 - Pearson`` per window, averaged over the batch (scale-invariant)."""
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)
    corr = (x * y).sum(1) / (x.norm(dim=1) * y.norm(dim=1) + 1e-6)
    return (1.0 - corr).mean()


class FetalScalogramLoss(nn.Module):
    """``MSE(Re,Im) + a*MSE(|W|) + b*MSE(signal) + c*(1 - corr(signal))``.

    The first term is the plain scalogram regression described in the report. The
    magnitude term sharpens the visible scalogram. The signal terms act on the waveform
    obtained with the inverse CWT (a fixed linear map, hence differentiable) so the network
    is optimised for what is ultimately recovered; the correlation term is invariant to the
    amplitude of the fetal ECG, which cannot be inferred reliably from the abdomen.
    """

    def __init__(
        self,
        cwt_cfg: CWTConfig,
        target_scale: float,
        mag_weight: float,
        signal_weight: float,
        corr_weight: float = 0.0,
    ) -> None:
        super().__init__()
        w = synthesis_weights(cwt_cfg) * target_scale  # output is in normalised units
        self.register_buffer("weights", torch.as_tensor(w, dtype=torch.float32))
        self.mag_weight = mag_weight
        self.signal_weight = signal_weight
        self.corr_weight = corr_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor, signal: torch.Tensor):
        pred, target = pred.float(), target.float()
        parts = {"coef": F.mse_loss(pred, target)}
        parts["mag"] = F.mse_loss(magnitude(pred), magnitude(target))
        sig_pred = synthesize(pred, self.weights)
        parts["signal"] = F.mse_loss(sig_pred, signal)
        parts["corr"] = correlation_loss(sig_pred, signal)
        total = (
            parts["coef"]
            + self.mag_weight * parts["mag"]
            + self.signal_weight * parts["signal"]
            + self.corr_weight * parts["corr"]
        )
        return total, {k: float(v.detach()) for k, v in parts.items()}
