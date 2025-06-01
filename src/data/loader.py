"""
src/data/loader.py
~~~~~~~~~~~~~~~~~~

Stable dataloader for SIREN.

Key improvements (v0.2)
-----------------------
• Encodes string gesture labels -> int IDs (self.label2idx)
• Returns labels as plain int; collate_fn tensor-ises later
• Sorts rows by `sequence_counter`
• Works for both train / test splits without crash
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

# ───────────────────────── Config ──────────────────────────
IMU_COLUMNS = [
    "acc_x", "acc_y", "acc_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
]
DEFAULT_SEQ_LEN = 200
PAD_VALUE = 0.0


# ───────────────────── Helper functions ────────────────────
def _pad_or_crop(arr: np.ndarray,
                 target_len: int = DEFAULT_SEQ_LEN,
                 pad_value: float = PAD_VALUE) -> np.ndarray:
    """Tail-crop or symmetric-pad to fixed length."""
    length = arr.shape[0]
    if length == target_len:
        return arr
    if length > target_len:
        return arr[-target_len:]
    pad_total = target_len - length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(arr, ((pad_left, pad_right), (0, 0)),
                  mode="constant",
                  constant_values=pad_value)


def _zscore(seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = seq.mean(axis=0, keepdims=True)
    std = seq.std(axis=0, keepdims=True) + eps
    return (seq - mean) / std


# ──────────────────────── Dataset ──────────────────────────
class SequenceDataset:
    """
    Sequence-level dataset.

    Returns
    -------
    x : torch.Tensor | np.ndarray  shape (T,F)
    y : int | None                 gesture id
    seq_id : str
    """

    # class-wide label mapping (shared by train & test)
    label2idx: Dict[str, int] = {}

    def __init__(self,
                 csv_path: str | Path,
                 mode: str = "train",
                 seq_len: int = DEFAULT_SEQ_LEN,
                 use_torch: bool = True):

        assert mode in {"train", "test"}
        self.mode = mode
        self.seq_len = seq_len
        self.use_torch = TORCH_AVAILABLE and use_torch

        df = pd.read_csv(csv_path)

        # keep only IMU + id + gesture
        cols = ["sequence_id", "sequence_counter", *IMU_COLUMNS]
        if mode == "train":
            cols.append("gesture")
        df = df[cols]

        # create label map once
        if mode == "train" and not SequenceDataset.label2idx:
            unique = sorted(df["gesture"].unique())
            SequenceDataset.label2idx = {g: i for i, g in enumerate(unique)}

        grouped = {}
        labels = {}

        for seq_id, grp in df.groupby("sequence_id"):
            grp = grp.sort_values("sequence_counter")
            imu = grp[IMU_COLUMNS].to_numpy(np.float32)
            imu = _zscore(imu)
            imu = _pad_or_crop(imu, seq_len)
            grouped[seq_id] = imu
            if mode == "train":
                gstr = grp["gesture"].iloc[0]
                labels[seq_id] = SequenceDataset.label2idx[gstr]

        self.seq_ids = sorted(grouped.keys())
        self.X = grouped
        self.y = labels if mode == "train" else None

    # ------------------- PyTorch helpers --------------------
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> Tuple:
        sid = self.seq_ids[idx]
        x = self.X[sid]
        if self.use_torch:
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "train":
            y = self.y[sid]          # plain int
            return x, y, sid
        return x, None, sid


# ───────────────────── Convenience factory ──────────────────
def get_dataset(data_dir: str | Path,
                split: str = "train",
                **kwargs) -> SequenceDataset:
    csv_file = "train.csv" if split == "train" else "test.csv"
    return SequenceDataset(Path(data_dir) / csv_file,
                           mode=split, **kwargs)


# ------------------------ Debug run -------------------------
if __name__ == "__main__":
    ds = get_dataset("../../data/raw", split="train", use_torch=False)
    print("N =", len(ds), " first sample ", ds[0][:2])
