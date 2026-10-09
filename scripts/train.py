"""Train the scalogram U-Net."""

import argparse

from fecg.engine import train
from fecg.utils import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.epochs=5 model.dropout=0.1")
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda (default: cuda if available)")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    train(cfg, get_device(args.device))


if __name__ == "__main__":
    main()
