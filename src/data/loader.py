"""
src/data/loader.py
~~~~~~~~~~~~~~~~~~

Dataloader for pre-processed SIREN dataset.
"""

from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

# ---------- CONFIG ----------------------------------------------------
IMU_COLUMNS = [
    "imu_acc_x", "imu_acc_y", "imu_acc_z",
    "imu_rot_w", "imu_rot_x", "imu_rot_y", "imu_rot_z",
]
SEQ_LEN = 200  # sequences already padded in preprocessing
# ----------------------------------------------------------------------


class SequenceDataset:
    """
    Returns
    -------
    x : torch.Tensor | np.ndarray   (SEQ_LEN, 7)
    y : int | None                  gesture_id
    seq_id : str
    """

    def __init__(self,
                 csv_path: str | Path,
                 mode: str = "train",
                 use_torch: bool = True):

        assert mode in {"train", "test"}
        self.mode = mode
        self.use_torch = TORCH_AVAILABLE and use_torch

        df = pd.read_csv(csv_path)

        # expected columns already: sequence_id, gesture_id (train), imu_*
        keep = ["sequence_id", *IMU_COLUMNS]
        if mode == "train":
            keep.insert(1, "gesture_id")
        df = df[keep]

        grouped: Dict[str, np.ndarray] = {}
        labels: Dict[str, int] = {}

        for seq_id, grp in df.groupby("sequence_id"):
            imu = grp[IMU_COLUMNS].to_numpy(np.float32)
            assert imu.shape[0] == SEQ_LEN, "preprocess step guarantees length"
            grouped[seq_id] = imu
            if mode == "train":
                labels[seq_id] = int(grp["gesture_id"].iloc[0])

        self.seq_ids = sorted(grouped.keys())
        self.X = grouped
        self.y = labels if mode == "train" else None

    # -------------------- PyTorch helpers -------------------
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> Tuple:
        sid = self.seq_ids[idx]
        x = self.X[sid]
        if self.use_torch:
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "train":
            return x, self.y[sid], sid  # y is plain int
        return x, None, sid


# ------------- convenience factory -------------------------
def get_dataset(data_dir: str | Path,
                split: str = "train",
                **kwargs) -> SequenceDataset:
    csv_file = "train_processed.csv" if split == "train" else "test_processed.csv"
    return SequenceDataset(Path(data_dir) / csv_file, mode=split, **kwargs)


# ---------------------- debug ------------------------------
if __name__ == "__main__":
    ds = get_dataset("../../data/clean", split="train", use_torch=False)
    print("Loaded", len(ds), "sequences → sample shape:", ds[0][0].shape)
