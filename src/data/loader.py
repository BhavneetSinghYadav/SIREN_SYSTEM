"""Robust dataloader for processed multimodal SIREN dataset.

v1.3 – 2025‑06‑02
-----------------
* **Auto‑detects** column order from CSV header (no manual offsets)
* **Ablation flags** – `use_imu / use_thermo / use_tof` – determine `sensor_cols`
* **Shape assertion** ensures `(SEQ_LEN, in_channels)` after slicing
* Compatible public API: `SequenceDataset` + `get_dataset()` unchanged
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ModuleNotFoundError:  # unit‑test mode without torch
    TORCH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Column definitions (names must match those emitted by preprocessing script)
# ---------------------------------------------------------------------------
IMU_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "rot_w",
    "rot_x",
    "rot_y",
    "rot_z",
]
THERMO_COLUMNS = [f"thm_{i}" for i in range(1, 6)]  # 5
TOF_COLUMNS = [f"tof_{s}_v{v}" for s in range(1, 6) for v in range(64)]  # 320

ALL_COLUMNS: List[str] = IMU_COLUMNS + THERMO_COLUMNS + TOF_COLUMNS  # 332
SEQ_LEN = 200  # guaranteed by preprocessing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_cols(df: pd.DataFrame, required: List[str]) -> None:
    """Ensure all required columns exist in dataframe."""

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[loader] CSV missing expected columns: {missing[:5]} …")


def _select_sensor_cols(
    *,
    imu: bool = True,
    thermo: bool = True,
    tof: bool = True,
) -> List[str]:
    cols: List[str] = []
    if imu:
        cols += IMU_COLUMNS
    if thermo:
        cols += THERMO_COLUMNS
    if tof:
        cols += TOF_COLUMNS
    return cols


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class SequenceDataset:
    """Return `(sequence_tensor, label_or_None, sequence_id)` for each item."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        mode: str = "train",
        use_torch: bool = True,
        use_imu: bool = True,
        use_thermo: bool = True,
        use_tof: bool = True,
    ) -> None:
        assert mode in {"train", "test"}, "mode must be 'train' or 'test'"
        self.mode = mode
        self.use_torch = TORCH_AVAILABLE and use_torch

        df = pd.read_csv(csv_path)
        _assert_cols(df, ALL_COLUMNS)

        # ---------------- sensor selection --------------------
        self.sensor_cols: List[str] = _select_sensor_cols(
            imu=use_imu, thermo=use_thermo, tof=use_tof
        )
        self.in_channels: int = len(self.sensor_cols)

        # ---------------- column subset -----------------------
        cols: List[str] = ["sequence_id", *self.sensor_cols]
        if mode == "train":
            cols.insert(1, "gesture_id")  # keep label right after id for grouping
        df = df[cols]

        # ---------------- group by sequence -------------------
        self.X: Dict[str, np.ndarray] = {}
        self.y: Dict[str, int] | None = {} if mode == "train" else None
        for seq_id, grp in df.groupby("sequence_id"):
            mat = grp[self.sensor_cols].to_numpy(np.float32)
            assert (
                mat.shape == (SEQ_LEN, self.in_channels)
            ), f"shape mismatch {mat.shape} for seq {seq_id}"
            self.X[seq_id] = mat
            if mode == "train":
                self.y[seq_id] = int(grp["gesture_id"].iloc[0])  # type: ignore[arg-type]

        self.seq_ids: List[str] = sorted(self.X.keys())

    # ---------------- torch helpers ---------------------------
    def __len__(self) -> int:  # noqa: D401 – trivial
        return len(self.seq_ids)

    def __getitem__(self, idx: int) -> Tuple:
        sid = self.seq_ids[idx]
        x = self.X[sid]
        if self.use_torch:
            x = torch.tensor(x, dtype=torch.float32)  # type: ignore[assignment]
        label = self.y[sid] if self.mode == "train" else None  # type: ignore[index]
        return x, label, sid


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def get_dataset(
    data_dir: str | Path,
    *,
    split: str = "train",
    **kwargs,
) -> SequenceDataset:
    csv_name = "train_processed.csv" if split == "train" else "test_processed.csv"
    return SequenceDataset(Path(data_dir) / csv_name, mode=split, **kwargs)


# ---------------------------------------------------------------------------
# Debug CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ds = get_dataset(
        "../../data/clean",
        split="train",
        use_torch=False,
        use_thermo=True,
        use_tof=False,
    )
    print(
        "Loaded", len(ds), "| frame shape:", ds[0][0].shape, "| channels:", ds.in_channels
    )
