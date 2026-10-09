"""Download the PhysioNet Abdominal and Direct Fetal ECG Database (adfecgdb)."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

BASE_URL = "https://physionet.org/files/adfecgdb/1.0.0"
DEFAULT_RECORDS = ("r01", "r04", "r07", "r08", "r10")


def _fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checksums(path: Path) -> dict[str, str]:
    sums = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            sums[parts[1]] = parts[0]
    return sums


def download_adfecgdb(
    out_dir: str | Path, records: tuple[str, ...] | list[str] = DEFAULT_RECORDS
) -> list[Path]:
    """Download ``<record>.edf`` and ``<record>.edf.qrs`` and verify their SHA-256 sums."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sums_path = out_dir / "SHA256SUMS.txt"
    _fetch(f"{BASE_URL}/SHA256SUMS.txt", sums_path)
    checksums = _read_checksums(sums_path)

    files = []
    for rec in records:
        for name in (f"{rec}.edf", f"{rec}.edf.qrs"):
            dest = out_dir / name
            if not (dest.exists() and _sha256(dest) == checksums.get(name)):
                print(f"downloading {name} ...")
                _fetch(f"{BASE_URL}/{name}", dest)
            if name in checksums and _sha256(dest) != checksums[name]:
                dest.unlink()
                raise RuntimeError(f"checksum mismatch for {name}; file removed, retry the download")
            files.append(dest)
    return files
