"""Reader for PhysioNet (WFDB) binary beat-annotation files such as ``r01.edf.qrs``."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_SKIP, _NUM, _SUB, _CHN, _AUX = 59, 60, 61, 62, 63
_BEAT_CODES = range(1, 14)  # NORMAL (1) ... UNKNOWN (13): WFDB beat annotation types


def read_beat_annotations(path: str | Path) -> np.ndarray:
    """Return the sample index of every annotated beat.

    Format (WFDB "MIT" annotation files): a stream of little-endian 16-bit words,
    each holding a 6-bit annotation code (high bits) and a 10-bit time increment
    (low bits). Codes 59-63 are housekeeping records (signed 32-bit skips and
    attributes); non-beat annotations such as notes are ignored.
    """
    words = np.frombuffer(Path(path).read_bytes(), dtype="<u2")
    beats: list[int] = []
    t, i = 0, 0
    while i < words.size:
        word = int(words[i])
        code, interval = word >> 10, word & 0x3FF
        i += 1
        if code == 0 and interval == 0:  # end-of-file marker
            break
        if code == _SKIP:
            skip = (int(words[i]) << 16) | int(words[i + 1])
            t += skip - (1 << 32) if skip >= 1 << 31 else skip  # signed 32-bit
            i += 2
        elif code in (_NUM, _SUB, _CHN):
            continue  # attribute of an annotation, carries no time
        elif code == _AUX:
            i += (interval + 1) // 2  # skip the padded auxiliary string
        else:
            t += interval
            if code in _BEAT_CODES:
                beats.append(t)
    return np.asarray(beats, dtype=np.int64)
