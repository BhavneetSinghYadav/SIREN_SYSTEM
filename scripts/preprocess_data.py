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
from src.data.preprocessing import run_preprocessing, SEQ_LEN


def main():
    p = argparse.ArgumentParser(description="Pre-process multimodal BFRB dataset")
    p.add_argument("--data_dir", required=True,
                   help="Folder with raw train.csv / test.csv from Kaggle")
    p.add_argument("--out_dir",  required=True,
                   help="Destination folder for *_processed.csv & label_map.json")
    p.add_argument("--frames", type=int, default=SEQ_LEN,
                   help="Target number of frames after resampling")
    p.add_argument("--window", type=int, default=None,
                   help="Length of sliding windows (default: frames)")
    p.add_argument("--stride", type=int, default=None,
                   help="Stride between windows (default: window size)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    run_preprocessing(
        data_dir=data_dir,
        out_dir=out_dir,
        n_frames=args.frames,
        window_size=args.window,
        window_stride=args.stride,
    )
    print("[preprocess_data] complete.")


if __name__ == "__main__":
    main()
