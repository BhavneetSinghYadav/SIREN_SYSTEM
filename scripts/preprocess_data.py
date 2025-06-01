#!/usr/bin/env python
"""scripts/preprocess_data.py
Run this once to transform the raw Kaggle CSV files into cleaned, padded,
and z‑scored versions the rest of the SIREN pipeline expects.

Example
-------
$ python scripts/preprocess_data.py \
        --data_dir /kaggle/input/cmi-detect-behavior-with-sensor-data \
        --out_dir  /kaggle/working/clean_data
"""

import argparse
from pathlib import Path

from src.data.preprocessing import run_preprocessing


def main():
    parser = argparse.ArgumentParser(description="Pre‑process raw BFRB sensor dataset")
    parser.add_argument("--data_dir", required=True,
                        help="Folder containing raw train.csv / test.csv from Kaggle")
    parser.add_argument("--out_dir",  required=True,
                        help="Destination folder for train_processed.csv / test_processed.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    run_preprocessing(data_dir=data_dir, out_dir=out_dir)


if __name__ == "__main__":
    main()
