"""PyTorch dataset of (abdominal scalogram window, fetal scalogram window) pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def build_features(abd_cwt: np.ndarray) -> np.ndarray:
    """Abdominal complex scalograms ``(4, S, W)`` -> network input ``(8, S, W)``.

    Channels 0-3: log-compressed magnitude of each lead ("how much energy").
    Channels 4-7: phase of each lead scaled to [-1, 1] ("timing and alignment").
    """
    magnitude = np.log1p(np.abs(abd_cwt))
    phase = np.angle(abd_cwt) / np.pi
    return np.concatenate([magnitude, phase], axis=0).astype(np.float32)


def build_target(fetal_cwt: np.ndarray, target_scale: float) -> np.ndarray:
    """Fetal complex scalogram ``(S, W)`` -> normalised real/imaginary image ``(2, S, W)``."""
    return (np.stack([fetal_cwt.real, fetal_cwt.imag]) / target_scale).astype(np.float32)


def window_starts(n_samples: int, window: int, stride: int) -> list[int]:
    """Window start indices covering the whole record (the last window ends at the end)."""
    starts = list(range(0, n_samples - window + 1, stride))
    if starts[-1] != n_samples - window:
        starts.append(n_samples - window)
    return starts


def load_stats(processed_dir: str | Path, stats_file: str = "stats.json") -> dict:
    return json.loads((Path(processed_dir) / stats_file).read_text())


class ScalogramDataset(Dataset):
    """Sliding windows over preprocessed records (arrays are memory-mapped).

    With ``augment=True`` every sample is randomly perturbed in a way that keeps the
    (input, target) pair physically consistent. Because the CWT is linear this is done on
    the complex coefficients: (i) abdominal leads are shuffled, (ii) each lead gets a random
    gain (electrode contact and fetal position vary between subjects) and (iii) the polarity
    of the whole pair is flipped with probability 0.5.
    """

    def __init__(
        self,
        processed_dir: str | Path,
        records: list[str],
        window: int,
        stride: int,
        augment: bool = False,
        stats_file: str = "stats.json",
    ) -> None:
        self.root = Path(processed_dir)
        self.window = window
        self.augment = augment
        self._rng: np.random.Generator | None = None
        self.target_scale = load_stats(self.root, stats_file)["target_scale"]
        self.abd, self.fetal_cwt, self.fetal, self.index = [], [], [], []
        for r_idx, name in enumerate(records):
            rec = self.root / name
            self.abd.append(np.load(rec / "abd_cwt.npy", mmap_mode="r"))
            self.fetal_cwt.append(np.load(rec / "fetal_cwt.npy", mmap_mode="r"))
            self.fetal.append(np.load(rec / "fetal.npy", mmap_mode="r"))
            n = self.fetal[-1].shape[0]
            self.index += [(r_idx, s) for s in window_starts(n, window, stride)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, s = self.index[i]
        sl = slice(s, s + self.window)
        abd = np.asarray(self.abd[r][:, :, sl])
        fetal_cwt = np.asarray(self.fetal_cwt[r][:, sl])
        sig = np.array(self.fetal[r][sl], dtype=np.float32)  # copy: memmap slices are read-only
        if self.augment:
            abd, fetal_cwt, sig = self._augment(abd, fetal_cwt, sig)
        x = build_features(abd)
        y = build_target(fetal_cwt, self.target_scale)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(sig)

    def _augment(self, abd: np.ndarray, fetal_cwt: np.ndarray, sig: np.ndarray):
        if self._rng is None:  # created lazily so every DataLoader worker gets its own stream
            self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
        rng = self._rng
        abd = abd[rng.permutation(abd.shape[0])]
        abd = abd * rng.lognormal(0.0, 0.3, size=(abd.shape[0], 1, 1)).astype(np.float32)
        if rng.random() < 0.5:
            abd, fetal_cwt, sig = -abd, -fetal_cwt, -sig
        return abd, fetal_cwt, sig
