"""
src/data/preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~~

One-shot dataset cleaner for the SIREN project.

• Reads raw train/test CSVs
• Cleans NaNs / Infs  → 0.0
• Z-scores IMU columns per sequence
• Tail-crops or symmetric-pads to fixed length
• Encodes gesture strings → int IDs
• Writes cleaned CSVs + label_map.json

Run via scripts/preprocess_data.py
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

# ------------------------------ config -------------------------------
IMU_COLUMNS = [
    "acc_x", "acc_y", "acc_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
]
SEQ_LEN = 200
PAD_VALUE = 0.0


# ------------------------- helper functions --------------------------
def _pad_or_crop(arr: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    """Tail-crop if long, symmetric-pad if short."""
    n = arr.shape[0]
    if n == target_len:
        return arr
    if n > target_len:
        return arr[-target_len:]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(arr, ((left, right), (0, 0)),
                  mode="constant", constant_values=PAD_VALUE)


def _zscore(x: np.ndarray, eps=1e-6) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    std = np.clip(x.std(axis=0, keepdims=True), eps, None)
    return (x - mu) / std


def clean_split(raw_csv: Path,
                out_csv: Path,
                label_map: dict[str, int] | None,
                is_train: bool):
    """Process one split (train or test)."""
    df = pd.read_csv(raw_csv)

    # keep relevant columns
    keep = ["sequence_id", "sequence_counter", *IMU_COLUMNS]
    if is_train:
        keep.append("gesture")
    df = df[keep]

    # gesture → int mapping
    if is_train:
        if label_map is None:       # first time
            unique = sorted(df["gesture"].unique())
            label_map = {g: i for i, g in enumerate(unique)}
        df["gesture_id"] = df["gesture"].map(label_map)

    processed_rows = []

    for seq_id, grp in df.groupby("sequence_id"):
        grp = grp.sort_values("sequence_counter")
        imu = grp[IMU_COLUMNS].to_numpy(np.float32)
        imu = np.nan_to_num(imu, nan=0.0, posinf=0.0, neginf=0.0)
        imu = _zscore(imu)
        imu = _pad_or_crop(imu)

        proc = pd.DataFrame(
            imu, columns=[f"imu_{c}" for c in IMU_COLUMNS])
        proc.insert(0, "sequence_id", seq_id)
        if is_train:
            gid = label_map[grp["gesture"].iloc[0]]
            proc.insert(1, "gesture_id", gid)
        processed_rows.append(proc)

    out_df = pd.concat(processed_rows, ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    return label_map


# ---------------------- public entry-point ---------------------------
def run_preprocessing(data_dir: str | Path,
                      out_dir: str | Path):
    """
    Parameters
    ----------
    data_dir : folder containing raw train/test CSVs.
    out_dir  : destination for processed CSVs + label_map.json
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map: dict[str, int] | None = None

    label_map = clean_split(
        raw_csv=data_dir / "train.csv",
        out_csv=out_dir / "train_processed.csv",
        label_map=label_map,
        is_train=True)

    _ = clean_split(
        raw_csv=data_dir / "test.csv",
        out_csv=out_dir / "test_processed.csv",
        label_map=label_map,
        is_train=False)

    # save mapping for loader or inference
    with open(out_dir / "label_map.json", "w") as jf:
        json.dump(label_map, jf, indent=2)

    print(f"[preprocess] done → files written to {out_dir.resolve()}")


# ----------------------------- CLI -----------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess BFRB dataset")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    run_preprocessing(args.data_dir, args.out_dir)
# Data normalization and alignment
