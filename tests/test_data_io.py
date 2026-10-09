import numpy as np

from fecg.data.annotations import read_beat_annotations
from fecg.data.edf import read_edf
from helpers import write_edf, write_qrs


def test_read_edf_roundtrip_and_drops_annotation_channel(tmp_path):
    rng = np.random.default_rng(0)
    signals = rng.normal(0, 50, (3, 10_000))
    write_edf(tmp_path / "x.edf", ["Direct_1", "Abdomen_1", "Abdomen_2"], signals, fs=1000)
    rec = read_edf(tmp_path / "x.edf")
    assert rec.labels == ["Direct_1", "Abdomen_1", "Abdomen_2"]
    assert rec.fs == 1000
    assert rec.signals.shape == (3, 10_000)
    np.testing.assert_allclose(rec.signals, signals, atol=0.06)  # ~0.1 uV quantisation step
    np.testing.assert_array_equal(rec.channel("Abdomen_1"), rec.signals[1])


def test_read_beat_annotations_handles_skips_and_notes(tmp_path):
    beats = np.array([183, 651, 1118, 5000, 5460, 70_000, 70_470])  # includes >1023 gaps
    write_qrs(tmp_path / "x.qrs", beats)
    np.testing.assert_array_equal(read_beat_annotations(tmp_path / "x.qrs"), beats)


def test_config_overrides(tmp_path):
    import pytest

    from fecg.utils import load_config

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("train:\n  epochs: 20\ndata:\n  bandpass: [2.0, 100.0]\n")
    cfg = load_config(cfg_path, ["train.epochs=3", "data.bandpass=[0.5,100]"])
    assert cfg["train"]["epochs"] == 3 and cfg["data"]["bandpass"] == [0.5, 100]
    assert load_config(cfg_path, ["train.epochs=1e-4"])["train"]["epochs"] == 1e-4  # not a string
    with pytest.raises(KeyError):
        load_config(cfg_path, ["train.typo=1"])
