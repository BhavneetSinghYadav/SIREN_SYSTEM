"""
src/features/feature_bank.py  (v0.3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unified symbolic‑feature registry used by FusionNet.

Supported extractors (all ON by default):
* **Rhythmicity**              – 1‑D
* **Thermopile means**         – 5‑D
* **ToF summary**              – 4‑D
* **Tremor entropy**           – 1‑D
* **Posture anchor (Euler)**   – 3‑D
→ Total dim = 14 when all enabled.

Risk‑mitigation:
  • Column‑name discovery (no hard‑coded offsets)
  • Optional ablation flags (use_*)
  • NaN / Inf sanitation before return
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .rhythmicity       import RhythmicityExtractor
from .thermo_extractor  import ThermoExtractor
from .tof_extractor     import ToFExtractor
from .tremor_entropy    import TremorEntropyExtractor
from .posture_anchor    import PostureAnchorExtractor

# ---------------------------------------------------------------------
class SymbolicFeatureBank:
    """Bundle symbolic extractors with dynamic column‑position lookup."""

    def __init__(self,
                 frame_columns: List[str],
                 use_rhythm:  bool = True,
                 use_thermo:  bool = True,
                 use_tof:     bool = True,
                 use_tremor:  bool = True,
                 use_posture: bool = True):

        self.extractors: List[Tuple[str, object]] = []
        cols = frame_columns  # alias

        # ---- helper for lookup ----
        def idx_of(col: str) -> int:
            try:
                return cols.index(col)
            except ValueError:
                raise ValueError(f"[FeatureBank] column '{col}' not found in frame")

        # ---------------------------
        if use_rhythm:
            self.extractors.append(("rhythmicity", RhythmicityExtractor()))

        if use_thermo:
            self.extractors.append(("thermo", ThermoExtractor(start_idx=idx_of("thm_1"))))

        if use_tof:
            self.extractors.append(("tof", ToFExtractor(start_idx=idx_of("tof_1_v0"))))

        if use_tremor:
            self.extractors.append(("tremor", TremorEntropyExtractor(acc_start_idx=idx_of("acc_x"))))

        if use_posture:
            self.extractors.append(("posture", PostureAnchorExtractor(rot_start_idx=idx_of("rot_w"))))

        self._dim = sum(ex.dim() for _, ex in self.extractors)

    # --------------------------------------------------------------
    def extract_all(self, seq: np.ndarray) -> np.ndarray:
        parts = [ex.extract(seq) for _, ex in self.extractors]
        vec   = np.concatenate(parts).astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------------------
    def dim(self) -> int:
        return self._dim

    # human‑readable labels ---------------------------------------
    def feature_names(self) -> List[str]:
        names: List[str] = []
        for tag, ex in self.extractors:
            names += [f"{tag}_{i}" for i in range(ex.dim())]
        return names

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # minimal sanity check on dummy frame columns
    cols = (
        ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"] +
        [f"thm_{i}" for i in range(1, 6)] + ["tof_1_v0"]
    )
    seq  = np.zeros((200, len(cols)), dtype=np.float32)
    bank = SymbolicFeatureBank(cols)
    v    = bank.extract_all(seq)
    print("dim =", bank.dim(), v.shape, "feature names:", bank.feature_names())
