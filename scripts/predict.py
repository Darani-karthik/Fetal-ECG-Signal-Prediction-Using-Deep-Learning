"""Extract the fetal ECG from an EDF recording with 4 abdominal leads."""

import argparse
from pathlib import Path

import numpy as np

from fecg.predict import predict_edf
from fecg.utils import get_device
from fecg.visualize import plot_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", required=True, help="input recording, e.g. data/raw/r10.edf")
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pt")
    parser.add_argument("--channels", nargs=4, default=None, metavar="NAME", help="the 4 abdominal channel names")
    parser.add_argument("--out-dir", default="results/predictions")
    parser.add_argument("--plot-start", type=float, default=60.0, help="start (s) of the 8 s preview plot")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out = predict_edf(args.edf, args.checkpoint, get_device(args.device), args.channels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.edf).stem
    np.save(out_dir / f"{stem}_fetal_ecg.npy", out["fetal_ecg"])
    np.savetxt(out_dir / f"{stem}_beats_s.csv", out["beats"] / out["fs"], fmt="%.3f", header="beat_time_s")
    plot_prediction(out["abdominal"][0], out["fetal_ecg"], out["beats"], out["fs"], args.plot_start, 8.0,
                    out_dir / f"{stem}_prediction.png")
    print(f"{len(out['beats'])} fetal beats, median fetal heart rate {out['heart_rate_bpm']:.1f} bpm")
    print(f"saved to {out_dir}/")


if __name__ == "__main__":
    main()
