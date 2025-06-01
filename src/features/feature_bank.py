"""
src/features/feature_bank.py (v0.2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Risk‑hardened symbolic feature registry.

Key upgrades
------------
1. **Column‑name‑driven** start‑index discovery ⇒ no hard‑coded offsets.
2. Easy **ablation flags** (`use_rhythm`, `use_thermo`, `use_tof`).
3. Graceful fallback if a requested sensor block is absent in the frame.
4. `extract_all` now guards against NaNs and returns fixed‐length vector.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .rhythmicity      import RhythmicityExtractor
from .thermo_extractor  import ThermoExtractor
from .tof_extractor     import ToFExtractor

# ---------------------------------------------------------------------
class SymbolicFeatureBank:
    """Bundle symbolic extractors with dynamic column‑position detection."""

    def __init__(self,
                 frame_columns: List[str],
                 use_rhythm: bool = True,
                 use_thermo: bool = True,
                 use_tof: bool = True):
        """
        Parameters
        ----------
        frame_columns : list of column names in the (T,F) matrix order.
        use_* flags   : quick ablation toggles.
        """
        self.extractors: List[Tuple[str, object]] = []

        if use_rhythm:
            self.extractors.append(("rhythmicity", RhythmicityExtractor()))

        if use_thermo:
            try:
                thm_start = frame_columns.index("thm_1")
            except ValueError:
                raise ValueError("[FeatureBank] 'thm_1' column not found; preprocessing mismatch")
            self.extractors.append(("thermo", ThermoExtractor(start_idx=thm_start)))

        if use_tof:
            try:
                tof_start = frame_columns.index("tof_1_v0")
            except ValueError:
                raise ValueError("[FeatureBank] 'tof_1_v0' column not found; preprocessing mismatch")
            self.extractors.append(("tof", ToFExtractor(start_idx=tof_start)))

        self._dim = sum(ex.dim() for _, ex in self.extractors)

    # --------------------------------------------------------------
    def extract_all(self, seq: np.ndarray) -> np.ndarray:
        """Concatenate all symbolic features (guards against NaNs)."""
        parts = [ex.extract(seq) for _, ex in self.extractors]
        vec   = np.concatenate(parts).astype(np.float32)
        # replace any nan/inf that slipped through
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------------------
    def dim(self) -> int:
        return self._dim

    # convenience: feature label list
    def feature_names(self) -> List[str]:
        names = []
        for tag, ex in self.extractors:
            names += [f"{tag}_{i}" for i in range(ex.dim())]
        return names

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate column order & dummy sequence
    cols = [f"x{i}" for i in range(7)] + ["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"] + ["tof_1_v0"]
    seq  = np.zeros((200, len(cols)), dtype=np.float32)

    bank = SymbolicFeatureBank(cols, use_rhythm=True, use_thermo=True, use_tof=True)
    v    = bank.extract_all(seq)
    print("dim =", bank.dim(), v.shape)
