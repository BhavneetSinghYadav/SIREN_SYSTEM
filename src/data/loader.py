"""
src/data/loader.py
~~~~~~~~~~~~~~~~~~

Dataloader for **processed multimodal** SIREN dataset (IMU + Thermopile + ToF).
The processed CSVs contain:
  • sequence_id  (always)
  • gesture_id   (train only)
  • 332 sensor columns in the fixed order  IMU → Thermo → ToF.

This loader returns the full `(SEQ_LEN, 332)` frame tensor so that downstream
symbolic extractors (thermo / ToF) can slice directly.  The CNN encoder will
sub‑select the first 7 channels (IMU) unless you choose to feed extra streams.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

# ---------- CONFIG ----------------------------------------------------
IMU_COLUMNS = [
    "acc_x", "acc_y", "acc_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
]
THERMO_COLUMNS = [f"thm_{i}" for i in range(1, 6)]
TOF_COLUMNS    = [f"tof_{s}_v{v}" for s in range(1, 6) for v in range(64)]
ALL_COLUMNS    = IMU_COLUMNS + THERMO_COLUMNS + TOF_COLUMNS  # 332 cols

SEQ_LEN = 200  # guaranteed by preprocessing

# handy index positions for feature extractors
THERMO_START = len(IMU_COLUMNS)                # 7
TOF_START    = len(IMU_COLUMNS) + len(THERMO_COLUMNS)  # 12

# ----------------------------------------------------------------------
class SequenceDataset:
    """PyTorch‑style dataset yielding (sensor_matrix, label, seq_id)."""

    def __init__(self,
                 csv_path: str | Path,
                 mode: str = "train",
                 use_torch: bool = True):

        assert mode in {"train", "test"}
        self.mode = mode
        self.use_torch = TORCH_AVAILABLE and use_torch

        df = pd.read_csv(csv_path)

        cols = ["sequence_id", *ALL_COLUMNS]
        if mode == "train":
            cols.insert(1, "gesture_id")
        df = df[cols]

        grouped: Dict[str, np.ndarray] = {}
        labels: Dict[str, int] = {}

        for seq_id, grp in df.groupby("sequence_id"):
            mat = grp[ALL_COLUMNS].to_numpy(np.float32)  # (T, 332)
            assert mat.shape[0] == SEQ_LEN, "preprocess guarantees length"
            grouped[seq_id] = mat
            if mode == "train":
                labels[seq_id] = int(grp["gesture_id"].iloc[0])

        self.seq_ids = sorted(grouped.keys())
        self.X = grouped
        self.y = labels if mode == "train" else None

    # ‑‑‑ PyTorch helpers ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> Tuple:
        sid = self.seq_ids[idx]
        x = self.X[sid]  # (T, 332)
        if self.use_torch:
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "train":
            return x, self.y[sid], sid
        return x, None, sid


# convenience factory --------------------------------------------------

def get_dataset(data_dir: str | Path,
                split: str = "train",
                **kwargs) -> SequenceDataset:
    csv = "train_processed.csv" if split == "train" else "test_processed.csv"
    return SequenceDataset(Path(data_dir) / csv, mode=split, **kwargs)


# debug ---------------------------------------------------------------
if __name__ == "__main__":
    ds = get_dataset("../../data/clean", split="train", use_torch=False)
    print("Loaded", len(ds), "sequences | sample shape:", ds[0][0].shape)
