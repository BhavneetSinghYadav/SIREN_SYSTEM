"""
src/data/loader.py
~~~~~~~~~~~~~~~~~~

Data-loading utilities for the SIREN project.

• Loads `train.csv` / `test.csv`
• Groups rows by `sequence_id`
• Normalises IMU values
• Returns (sequence_tensor, label) pairs via a PyTorch-style Dataset

Author: Bhavya (BhavneetSinghYadav) – 2025
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import numpy as np

# Optional PyTorch import (fallback to NumPy arrays if unavailable)
try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
IMU_COLUMNS = [
    "acc_x", "acc_y", "acc_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
]
DEFAULT_SEQ_LEN = 200           # Fixed length after padding / cropping
PAD_VALUE        = 0.0


# ──────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────
def _pad_or_crop(arr: np.ndarray,
                 target_len: int = DEFAULT_SEQ_LEN,
                 pad_value: float = PAD_VALUE) -> np.ndarray:
    """
    Pad with `pad_value` or crop to fixed length along axis=0.

    • If the sequence is longer than `target_len`, we KEEP the **last**
      `target_len` timesteps (gesture likely near the end of each sequence).
    • If shorter, we pad *symmetrically* (same as before).
    """
    length = arr.shape[0]

    # Exact match ── nothing to do
    if length == target_len:
        return arr

    # CROP ── keep tail
    if length > target_len:
        return arr[-target_len:]            # last N frames

    # PAD ── fill both sides
    pad_total = target_len - length
    pad_left  = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(arr, ((pad_left, pad_right), (0, 0)),
                  mode="constant",
                  constant_values=pad_value)


def _zscore(seq: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Simple z-score normalisation along the time axis."""
    mean = seq.mean(axis=0, keepdims=True)
    std  = seq.std(axis=0, keepdims=True) + eps
    return (seq - mean) / std


# ──────────────────────────────────────────────────────────────────────────
# Core dataset
# ──────────────────────────────────────────────────────────────────────────
class SequenceDataset:
    """
    Minimal sequence-level dataset (PyTorch-compatible).

    Parameters
    ----------
    csv_path : str | Path
        Path to train.csv or test.csv.
    mode : {"train", "test"}
        Determines whether labels are returned.
    seq_len : int
        Fixed length for all sequences after padding / cropping.
    use_torch : bool
        If True (and PyTorch available) → returns torch.Tensor objects.

    Yields
    ------
    X : (seq_len, num_features) array/torch.Tensor
    y : int | None
    seq_id : str
    """
    def __init__(self,
                 csv_path: str | Path,
                 mode: str = "train",
                 seq_len: int = DEFAULT_SEQ_LEN,
                 use_torch: bool = True):

        assert mode in {"train", "test"}
        self.csv_path  = Path(csv_path)
        self.mode      = mode
        self.seq_len   = seq_len
        self.use_torch = TORCH_AVAILABLE and use_torch

        print(f"[loader] reading {self.csv_path.name} …")
        df = pd.read_csv(self.csv_path)

        # Keep only IMU columns for v0
        cols_required = ["sequence_id", *IMU_COLUMNS]
        if mode == "train":
            cols_required.append("gesture")
        df = df[cols_required]

        # Group rows by sequence_id → NumPy array (timesteps, features)
        grouped: Dict[str, np.ndarray] = {}
        labels:  Dict[str, int]        = {}

        for seq_id, grp in df.groupby("sequence_id"):
            imu_seq = grp[IMU_COLUMNS].to_numpy(dtype=np.float32)
            imu_seq = _zscore(imu_seq)               # normalise per-sequence
            imu_seq = _pad_or_crop(imu_seq, seq_len) # fixed length
            grouped[seq_id] = imu_seq
            if mode == "train":
                # gesture column is constant within a sequence
                labels[seq_id]  = grp["gesture"].iloc[0]

        self.seq_ids = sorted(grouped.keys())
        self.X       = grouped
        self.y       = labels if mode == "train" else None

    # PyTorch-style helpers ------------------------------------------------
    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> Tuple:
        seq_id = self.seq_ids[idx]
        x = self.X[seq_id]                         # (seq_len, features)
        if self.use_torch:
            x = torch.tensor(x, dtype=torch.float32)

        if self.mode == "train":
            y = self.y[seq_id]
            if self.use_torch:
                y = torch.tensor(y, dtype=torch.long)
            return x, y, seq_id
        return x, None, seq_id


# ──────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────
def get_dataset(data_dir: str | Path,
                split: str = "train",
                **kwargs) -> SequenceDataset:
    """
    Return SequenceDataset for 'train' or 'test'.

    Examples
    --------
    >>> train_ds = get_dataset("../data/raw", split="train")
    >>> x, y, sid = train_ds[0]
    """
    csv_file = "train.csv" if split == "train" else "test.csv"
    csv_path = Path(data_dir) / csv_file
    return SequenceDataset(csv_path, mode=split, **kwargs)


# ──────────────────────────────────────────────────────────────────────────
# Debug CLI
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick sanity check from command line:
    # $ python -m src.data.loader ../data/raw train
    import sys
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../../data/raw")
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    ds = get_dataset(root, split=split, use_torch=False)
    print(f"Loaded {len(ds)} sequences ⇢ shape[0]: {ds[0][0].shape}")
# Data loading logic
