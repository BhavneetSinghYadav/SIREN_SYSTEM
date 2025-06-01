"""
src/data/preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Multimodal dataset cleaner for the SIREN project (v0.3).

Upgrades from the previous IMU‑only version:
  • Supports **Thermopile** (5 cols) & **ToF** (5×64 = 320 cols)
  • Handles ToF missing value -1 → 0
  • Z‑scores **all** sensor columns per sequence
  • Pads / crops to fixed SEQ_LEN
  • Writes *_processed.csv with **full 332‑D** frame vectors + gesture_id

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
THERMO_COLUMNS = [f"thm_{i}" for i in range(1, 6)]  # 5 temps
TOF_COLUMNS    = [f"tof_{s}_v{v}" for s in range(1, 6) for v in range(64)]  # 320

ALL_COLUMNS = IMU_COLUMNS + THERMO_COLUMNS + TOF_COLUMNS

SEQ_LEN   = 200
PAD_VALUE = 0.0
MISSING_TOF = -1  # sentinel in raw csv

# ------------------------- helper functions --------------------------

def _pad_or_crop(arr: np.ndarray, target_len: int = SEQ_LEN) -> np.ndarray:
    n = arr.shape[0]
    if n == target_len:
        return arr
    if n > target_len:
        return arr[-target_len:]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(arr, ((left, right), (0, 0)), mode="constant", constant_values=PAD_VALUE)


def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    std = np.clip(x.std(axis=0, keepdims=True), eps, None)
    return (x - mu) / std


def clean_split(raw_csv: Path, out_csv: Path, label_map: dict[str, int] | None, is_train: bool):
    """Process one split (train or test)."""
    df = pd.read_csv(raw_csv)

    keep = ["sequence_id", "sequence_counter", *ALL_COLUMNS]
    if is_train:
        keep.append("gesture")
    df = df[keep]

    # gesture → int mapping
    if is_train:
        if label_map is None:
            unique = sorted(df["gesture"].unique())
            label_map = {g: i for i, g in enumerate(unique)}
        df["gesture_id"] = df["gesture"].map(label_map)

    processed_rows = []

    for seq_id, grp in df.groupby("sequence_id"):
        grp = grp.sort_values("sequence_counter")
        X = grp[ALL_COLUMNS].to_numpy(np.float32)  # (T, 332)

        # --- clean ---
        # 1) missing ToF -1 → 0
        X[:, len(IMU_COLUMNS)+len(THERMO_COLUMNS):][X[:, len(IMU_COLUMNS)+len(THERMO_COLUMNS):] == MISSING_TOF] = 0.0
        # 2) nan/inf → 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # 3) per-sequence z‑score
        X = _zscore(X)

        # 4) pad/crop
        X = _pad_or_crop(X)

        proc = pd.DataFrame(X, columns=ALL_COLUMNS)
        proc.insert(0, "sequence_id", seq_id)
        if is_train:
            gid = label_map[grp["gesture"].iloc[0]]
            proc.insert(1, "gesture_id", gid)
        processed_rows.append(proc)

    out_df = pd.concat(processed_rows, ignore_index=True)
    out_df.to_csv(out_csv, index=False)
    return label_map

# ---------------------- public entry-point ---------------------------

def run_preprocessing(data_dir: str | Path, out_dir: str | Path):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map: dict[str, int] | None = None

    label_map = clean_split(
        raw_csv=data_dir / "train.csv",
        out_csv=out_dir / "train_processed.csv",
        label_map=label_map,
        is_train=True,
    )

    _ = clean_split(
        raw_csv=data_dir / "test.csv",
        out_csv=out_dir / "test_processed.csv",
        label_map=label_map,
        is_train=False,
    )

    with open(out_dir / "label_map.json", "w") as jf:
        json.dump(label_map, jf, indent=2)

    print(f"[preprocess] done → files written to {out_dir.resolve()}")


# ----------------------------- CLI -----------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Preprocess multimodal BFRB dataset")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()
    run_preprocessing(args.data_dir, args.out_dir)
