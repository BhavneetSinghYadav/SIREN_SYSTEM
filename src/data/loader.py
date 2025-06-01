"""
src/data/loader.py
~~~~~~~~~~~~~~~~~~

Robust dataloader for processed multimodal SIREN dataset.
Now:

✓  Derives Thermo / ToF start-indices **from the CSV header** (no hard-coding).
✓  Allows easy ablation via flags   use_imu / use_thermo / use_tof.
✓  Emits clear diagnostic if expected columns are missing / re-ordered.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

try:
    import torch; TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

# ------------------------------------------------------------------ default col names
IMU_COLUMNS    = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
THERMO_COLUMNS = [f"thm_{i}"              for i in range(1, 6)]          # 5
TOF_COLUMNS    = [f"tof_{s}_v{v}" for s in range(1, 6) for v in range(64)]  # 320
ALL_COLUMNS    = IMU_COLUMNS + THERMO_COLUMNS + TOF_COLUMNS              # 332

SEQ_LEN = 200

# ------------------------------------------------------------------ helper
def _assert_cols(df: pd.DataFrame, cols: List[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[loader] CSV missing expected columns: {missing[:5]}...")

# ------------------------------------------------------------------ dataset
class SequenceDataset:
    """
    Parameters
    ----------
    csv_path : path to *_processed.csv
    mode     : 'train' | 'test'
    use_imu / use_thermo / use_tof : bool – sensor-subset ablation
    """

    def __init__(self,
                 csv_path: str | Path,
                 mode: str = "train",
                 use_torch: bool = True,
                 use_imu: bool = True,
                 use_thermo: bool = True,
                 use_tof: bool = True):

        assert mode in {"train", "test"}
        self.mode      = mode
        self.use_torch = TORCH_AVAILABLE and use_torch

        df = pd.read_csv(csv_path)

        # sanity: ensure expected columns exist
        _assert_cols(df, ALL_COLUMNS)

        # dynamic slice indices (robust to re-ordering in preprocessing)
        thermo_start = df.columns.get_loc("thm_1")  - (1 if mode == "train" else 0) - 1
        tof_start    = df.columns.get_loc("tof_1_v0")- (1 if mode == "train" else 0) - 1

        self.THERMO_START = thermo_start
        self.TOF_START    = tof_start

        sensor_cols: List[str] = []
        if use_imu:     sensor_cols += IMU_COLUMNS
        if use_thermo:  sensor_cols += THERMO_COLUMNS
        if use_tof:     sensor_cols += TOF_COLUMNS

        self.sensor_cols = sensor_cols
        self.in_channels = len(sensor_cols)

        cols = ["sequence_id", *sensor_cols]
        if mode == "train":
            cols.insert(1, "gesture_id")
        df = df[cols]

        grouped: Dict[str, np.ndarray] = {}
        labels:  Dict[str, int] = {}

        for seq_id, grp in df.groupby("sequence_id"):
            mat = grp[sensor_cols].to_numpy(np.float32)
            assert mat.shape[0] == SEQ_LEN, "preprocess guarantees length"
            grouped[seq_id] = mat
            if mode == "train":
                labels[seq_id] = int(grp["gesture_id"].iloc[0])

        self.seq_ids = sorted(grouped.keys())
        self.X = grouped
        self.y = labels if mode == "train" else None

    # ------------------ pytorch helpers ------------------
    def __len__(self): return len(self.seq_ids)

    def __getitem__(self, idx: int) -> Tuple:
        sid = self.seq_ids[idx]
        x   = self.X[sid]
        if self.use_torch:
            x = torch.tensor(x, dtype=torch.float32)
        return (x, self.y.get(sid) if self.y else None, sid)

# ------------------------------------------------------------------ factory
def get_dataset(data_dir: str | Path,
                split: str = "train",
                **kwargs) -> SequenceDataset:
    csv = "train_processed.csv" if split == "train" else "test_processed.csv"
    return SequenceDataset(Path(data_dir) / csv, mode=split, **kwargs)

# ------------------------------------------------------------------ debug
if __name__ == "__main__":
    ds = get_dataset("../../data/clean", split="train", use_torch=False,
                     use_thermo=True, use_tof=False)  # ablation example
    print("Loaded", len(ds), "| frame shape:", ds[0][0].shape, "| channels:", ds.in_channels)
