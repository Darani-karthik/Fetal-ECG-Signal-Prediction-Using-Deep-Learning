"""Download the PhysioNet adfecgdb recordings into data/raw (checksum-verified)."""

import argparse

from fecg.data.download import DEFAULT_RECORDS, download_adfecgdb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--records", nargs="+", default=list(DEFAULT_RECORDS))
    args = parser.parse_args()

    files = download_adfecgdb(args.out_dir, args.records)
    print(f"ok: {len(files)} files verified in {args.out_dir}")


if __name__ == "__main__":
    main()
