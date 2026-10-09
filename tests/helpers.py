"""Writers for tiny synthetic EDF / WFDB-annotation files and a synthetic mother+fetus recording."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_edf(path: Path, labels: list[str], signals: np.ndarray, fs: int, record_s: int = 5) -> None:
    """Write an EDF+ file (int16, +-3276.8 uV range) with an 'EDF Annotations' pseudo-channel."""
    n_ch, n = signals.shape
    n_rec = n // (fs * record_s)
    spr = fs * record_s
    all_labels = labels + ["EDF Annotations"]
    ns = len(all_labels)
    gain = 6553.6 / 65535  # EDF spec: physical = (digital - dmin) * gain + pmin
    digital = np.clip(np.round((signals + 3276.8) / gain - 32768), -32768, 32767).astype("<i2")

    def field(values, width):
        return b"".join(str(v).encode("ascii").ljust(width)[:width] for v in values)

    header = (
        b"0".ljust(8) + b"synthetic".ljust(80) + b"synthetic".ljust(80) + b"01.01.00" + b"00.00.00"
        + str(256 * (ns + 1)).encode().ljust(8) + b"EDF+C".ljust(44)
        + str(n_rec).encode().ljust(8) + str(record_s).encode().ljust(8) + str(ns).encode().ljust(4)
    )
    sig_header = (
        field(all_labels, 16) + field([""] * ns, 80) + field(["uV"] * (ns - 1) + [""], 8)
        + field([-3276.8] * (ns - 1) + [0], 8) + field([3276.8] * (ns - 1) + [1], 8)
        + field([-32768] * ns, 8) + field([32767] * ns, 8) + field([""] * ns, 80)
        + field([spr] * (ns - 1) + [20], 8) + field([""] * ns, 32)
    )
    body = bytearray()
    for r in range(n_rec):
        for c in range(n_ch):
            body += digital[c, r * spr : (r + 1) * spr].tobytes()
        body += b"\x00" * 40  # annotation channel: 20 int16 samples
    Path(path).write_bytes(header + sig_header + bytes(body))


def write_qrs(path: Path, beats: np.ndarray) -> None:
    """Write beats as a WFDB annotation file, mimicking the layout of the PhysioNet files."""
    w: list[int] = []
    note = b"## time resolution: 1000"
    w += [(22 << 10) | 0, (63 << 10) | len(note)]
    w += list(struct.unpack(f"<{len(note) // 2}H", note[: len(note) // 2 * 2]))
    w += [(59 << 10), 0xFFFF, 0xFFFF, (0 << 10) | 1]  # SKIP of -1, then a dummy annotation at t=0
    t = 0
    for b in beats:
        delta = int(b) - t
        if delta > 1023:
            w += [(59 << 10), delta >> 16, delta & 0xFFFF, (1 << 10) | 0]
        else:
            w += [(1 << 10) | delta]
        t = int(b)
    w += [0]
    Path(path).write_bytes(struct.pack(f"<{len(w)}H", *w))


def pulse_train(n: int, beats: np.ndarray, width: float, amp: float) -> np.ndarray:
    t = np.arange(n)[:, None]
    return amp * np.exp(-0.5 * ((t - beats[None]) / width) ** 2).sum(1)


def make_synthetic_record(raw_dir: Path, name: str, seconds: int = 60, fs: int = 1000, seed: int = 0) -> np.ndarray:
    """Write ``<name>.edf`` and ``<name>.edf.qrs``: big slow maternal beats + small fast fetal beats."""
    rng = np.random.default_rng(seed)
    n = seconds * fs
    maternal = np.cumsum(rng.normal(450, 20, int(seconds * 1.6))).astype(int) + 100
    fetal = np.cumsum(rng.normal(230, 10, int(seconds * 4.5))).astype(int) + 60
    maternal, fetal = maternal[maternal < n - 100], fetal[fetal < n - 100]
    m = pulse_train(n, maternal, 12, 120.0)
    f = pulse_train(n, fetal, 8, 25.0)
    mix = np.stack([g * m + h * f + rng.normal(0, 3, n) for g, h in [(1.0, 1.0), (0.7, 1.2), (1.3, 0.8), (0.9, 1.1)]])
    direct = 4 * f + rng.normal(0, 3, n)
    write_edf(
        raw_dir / f"{name}.edf",
        ["Direct_1", "Abdomen_1", "Abdomen_2", "Abdomen_3", "Abdomen_4"],
        np.vstack([direct, mix]), fs,
    )
    write_qrs(raw_dir / f"{name}.edf.qrs", fetal)
    return fetal
