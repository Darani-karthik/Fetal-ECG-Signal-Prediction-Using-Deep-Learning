"""Shared helpers: config loading, seeding, device selection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

from fecg.transforms import CWTConfig


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Load a YAML config; ``overrides`` are ``section.key=value`` strings (value parsed as YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for part in parents:
            node = node[part]
        if leaf not in node:
            raise KeyError(f"unknown config key: {key}")
        node[leaf] = _parse_value(value)
    return cfg


def _parse_value(text: str):
    """YAML value, additionally accepting floats such as ``1e-4`` (YAML 1.1 wants ``1.0e-4``)."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def cwt_config(cfg: dict) -> CWTConfig:
    c = cfg["cwt"]
    return CWTConfig(
        fs=float(cfg["data"]["fs"]),
        wavelet=c["wavelet"],
        n_scales=int(c["n_scales"]),
        fmin=float(c["fmin"]),
        fmax=float(c["fmax"]),
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
