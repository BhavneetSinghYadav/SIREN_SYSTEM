#!/usr/bin/env python
"""
scripts/preprocess_data.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

CLI wrapper for **multimodal** preprocessing (IMU + Thermopile + ToF).

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
    p = argparse.ArgumentParser(description="Pre-process multimodal BFRB dataset")
    p.add_argument("--data_dir", required=True,
                   help="Folder with raw train.csv / test.csv from Kaggle")
    p.add_argument("--out_dir",  required=True,
                   help="Destination folder for *_processed.csv & label_map.json")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    run_preprocessing(data_dir=data_dir, out_dir=out_dir)
    print("[preprocess_data] complete.")


if __name__ == "__main__":
    main()
