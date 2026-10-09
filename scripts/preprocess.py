"""Band-pass, resample, normalise and compute CWT scalograms for all records."""

import argparse

from fecg.data.preprocess import preprocess_all
from fecg.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.epochs=5 model.dropout=0.1")
    args = parser.parse_args()
    preprocess_all(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
