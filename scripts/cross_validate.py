"""Leave-one-subject-out cross-validation (5 folds); writes results/cv/."""

import argparse

from fecg.crossval import cross_validate
from fecg.utils import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. train.epochs=5 model.dropout=0.1")
    parser.add_argument("--out-dir", default="results/cv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    cross_validate(cfg, args.out_dir, get_device(args.device))


if __name__ == "__main__":
    main()
