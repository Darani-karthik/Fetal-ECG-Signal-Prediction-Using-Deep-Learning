"""Evaluate a trained checkpoint on the held-out test subject(s); writes results/."""

import argparse

from fecg.evaluate import evaluate
from fecg.utils import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.epochs=5 model.dropout=0.1")
    parser.add_argument("--checkpoint", default="checkpoints/unet_best.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config, args.set), args.checkpoint, get_device(args.device))


if __name__ == "__main__":
    main()
