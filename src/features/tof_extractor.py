"""
src/features/tof_extractor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic extractor for **Time‑of‑Flight (ToF)** distance grids (5 sensors × 64 pixels).
The raw competition data provides per‑frame columns:
    tof_[1‑5]_v[0‑63]
with values 0‑254 or ‑1 (no reflection).

This extractor summarises the 320‑D grid *per frame* into a **4‑D symbolic
vector per sequence** that captures overall hand‑to‑head proximity and signal
quality:

    0. mean_dist        – mean of valid ToF readings (scaled 0‑1)
    1. std_dist         – standard deviation of valid readings
    2. valid_ratio      – fraction of pixels that returned >0 (non‑missing)
    3. spatial_entropy  – Shannon entropy over the 16‑bin histogram of distances

The output is suitable for concatenation into the FusionNet symbolic vector.
"""

from __future__ import annotations

from typing import List, Sequence
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_SENSORS = 5        # ToF sensors on Helios
PIXELS_PER = 64        # 8×8 grid per sensor → 64 pixels
TOTAL_PIX  = NUM_SENSORS * PIXELS_PER  # 320 features per frame
MISSING_VAL = -1       # dataset uses -1 for "no reflection"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _spatial_entropy(valid_vals: np.ndarray, bins: int = 16) -> float:
    """Return Shannon entropy of a 1‑D array of distances (0‑254)."""
    if valid_vals.size == 0:
        return 0.0
    hist, _ = np.histogram(valid_vals, bins=bins, range=(0, 255), density=True)
    # avoid log(0)
    hist = hist[hist > 0]
    return float(-(hist * np.log2(hist)).sum())


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------
class ToFExtractor:
    """Compute a 4‑D summary vector from a sequence's ToF grid values."""

    def __init__(self, start_idx: int):
        """
        Parameters
        ----------
        start_idx : int
            Column index *within the full feature array* where tof_1_v0 begins.
            (All ToF columns are assumed contiguous and ordered row‑wise.
            For the current dataset this is len(IMU_COLUMNS)+len(THERMO_COLUMNS).)
        """
        self.start = start_idx
        self.end   = start_idx + TOTAL_PIX

    # ------------------------------------------------------------------
    def extract(self, seq: np.ndarray) -> np.ndarray:
        """Return 4‑element feature vector for a (T, F) sequence array."""
        # slice out ToF part  → shape (T, 320)
        tof_seq = seq[:, self.start:self.end].astype(np.float32)
        tof_seq = tof_seq.reshape(-1)  # flatten all frames

        # mask missing values
        valid_mask = tof_seq != MISSING_VAL
        valid_vals = tof_seq[valid_mask]

        if valid_vals.size == 0:             # fully missing → zeros
            return np.zeros(4, dtype=np.float32)

        mean_d  = valid_vals.mean() / 255.0   # scale 0‑1
        std_d   = valid_vals.std()  / 255.0
        ratio   = valid_vals.size / tof_seq.size
        ent     = _spatial_entropy(valid_vals)

        return np.array([mean_d, std_d, ratio, ent], dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def dim() -> int:
        """Dimensionality of the extracted feature vector."""
        return 4


# ---------------------------------------------------------------------------
# Smoke‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dummy = np.random.randint(0, 255, size=(200, TOTAL_PIX), dtype=np.int16)
    dummy[np.random.rand(*dummy.shape) < 0.1] = MISSING_VAL  # inject missing
    # prepend zeros to simulate IMU/thermo columns
    seq_full = np.concatenate([np.zeros((200, 20)), dummy], axis=1)

    extractor = ToFExtractor(start_idx=20)
    vec = extractor.extract(seq_full)
    print("ToF feature vector:", vec, "shape:", vec.shape)
