"""Minimal, dependency-free reader for European Data Format (EDF) files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EDFRecording:
    """Signals of one EDF file, in physical units (mV for adfecgdb)."""

    labels: list[str]
    fs: float
    signals: np.ndarray  # (n_channels, n_samples), float64

    def channel(self, label: str) -> np.ndarray:
        try:
            return self.signals[self.labels.index(label)]
        except ValueError as exc:
            raise KeyError(f"channel {label!r} not in {self.labels}") from exc


def read_edf(path: str | Path) -> EDFRecording:
    """Read an EDF file whose channels share one sampling rate."""
    raw = Path(path).read_bytes()
    if len(raw) < 256:
        raise ValueError(f"{path}: too short to be an EDF file")

    header_bytes = int(raw[184:192])
    n_records = int(raw[236:244])
    record_duration = float(raw[244:252])
    n_signals = int(raw[252:256])

    def text_fields(offset: int, width: int) -> tuple[list[str], int]:
        fields = [
            raw[offset + i * width : offset + (i + 1) * width].decode("ascii", "replace").strip()
            for i in range(n_signals)
        ]
        return fields, offset + n_signals * width

    pos = 256
    labels, pos = text_fields(pos, 16)
    _, pos = text_fields(pos, 80)  # transducer type
    _, pos = text_fields(pos, 8)  # physical dimension
    phys_min, pos = text_fields(pos, 8)
    phys_max, pos = text_fields(pos, 8)
    dig_min, pos = text_fields(pos, 8)
    dig_max, pos = text_fields(pos, 8)
    _, pos = text_fields(pos, 80)  # prefiltering
    samples_per_record, pos = text_fields(pos, 8)

    spr = np.array([int(s) for s in samples_per_record])
    record_len = int(spr.sum())

    data = np.frombuffer(raw, dtype="<i2", offset=header_bytes)
    if data.size < n_records * record_len:
        raise ValueError(f"{path}: truncated ({data.size} of {n_records * record_len} samples)")
    data = data[: n_records * record_len].reshape(n_records, record_len)

    pmin, pmax = np.array(phys_min, dtype=float), np.array(phys_max, dtype=float)
    dmin, dmax = np.array(dig_min, dtype=float), np.array(dig_max, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = (pmax - pmin) / (dmax - dmin)
    offset = pmin - dmin * gain

    # EDF+ files carry an "EDF Annotations" pseudo-channel; it is not a waveform.
    offsets = np.concatenate([[0], np.cumsum(spr)])
    keep = [i for i, name in enumerate(labels) if name != "EDF Annotations"]
    if len({int(spr[i]) for i in keep}) != 1:
        raise ValueError("waveform channels with different sampling rates are not supported")

    channels = [
        data[:, offsets[i] : offsets[i + 1]].reshape(-1).astype(np.float64) * gain[i] + offset[i]
        for i in keep
    ]
    return EDFRecording(
        labels=[labels[i] for i in keep],
        fs=int(spr[keep[0]]) / record_duration,
        signals=np.stack(channels),
    )
