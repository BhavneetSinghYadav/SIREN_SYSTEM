"""SymbolicFeatureBank (v0.4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hosts lightweight, on‑the‑fly symbolic feature extractors for a single
sensor sequence **(SEQ_LEN, C)**. Public API is unchanged but now:

* Gracefully **auto‑disables** an extractor if its required columns are
  absent (e.g. ToF disabled in dataset) – logs `warnings.warn` once.
* Offers `auto_disable_missing` flag to turn this behaviour on/off.
* Adds `__len__` alias for `dim()` so `len(bank)` works.
* Keeps *zero‑fill sanitation* for NaN/Inf.

Total dimensionality becomes dynamic – call `dim()` after creation.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple

import numpy as np

from .posture_anchor import PostureAnchorExtractor
from .rhythmicity import RhythmicityExtractor
from .thermo_extractor import ThermoExtractor
from .tof_extractor import ToFExtractor
from .tremor_entropy import TremorEntropyExtractor

__all__ = ["SymbolicFeatureBank"]

# ---------------------------------------------------------------------------


class SymbolicFeatureBank:
    """Bundle symbolic extractors with dynamic column‑position lookup.

    Parameters
    ----------
    frame_columns : list[str]
        Column order used in the *sequence matrix* (same as the loader).
    auto_disable_missing : bool, default True
        If *True*, silently drops an extractor whose anchor column is not
        present – instead of raising. Useful when running ablation.
    use_* flags : bool
        Enable/disable individual extractors irrespective of column
        availability.
    """

    def __init__(
        self,
        frame_columns: List[str],
        *,
        auto_disable_missing: bool = True,
        use_rhythm: bool = True,
        use_thermo: bool = True,
        use_tof: bool = True,
        use_tremor: bool = True,
        use_posture: bool = True,
    ) -> None:
        self.extractors: List[Tuple[str, object]] = []
        cols = frame_columns  # alias within closure

        # helper --------------------------------------------------
        def _maybe_idx(col: str) -> int | None:
            try:
                return cols.index(col)
            except ValueError:
                return None

        def _add(tag: str, extractor_builder, anchor_col: str | None):
            if anchor_col is None:
                if auto_disable_missing:
                    warnings.warn(
                        f"[FeatureBank] missing column for extractor '{tag}' – disabled.",
                        RuntimeWarning,
                    )
                    return
                raise ValueError(f"[FeatureBank] column '{anchor_col}' not found but required")
            self.extractors.append((tag, extractor_builder()))

        # ---------------------------------------------------------
        if use_rhythm:
            self.extractors.append(("rhythmicity", RhythmicityExtractor()))

        if use_thermo:
            _add("thermo", lambda: ThermoExtractor(start_idx=_maybe_idx("thm_1")), _maybe_idx("thm_1"))

        if use_tof:
            _add("tof", lambda: ToFExtractor(start_idx=_maybe_idx("tof_1_v0")), _maybe_idx("tof_1_v0"))

        if use_tremor:
            _add(
                "tremor",
                lambda: TremorEntropyExtractor(acc_start_idx=_maybe_idx("acc_x")),
                _maybe_idx("acc_x"),
            )

        if use_posture:
            _add(
                "posture",
                lambda: PostureAnchorExtractor(rot_start_idx=_maybe_idx("rot_w")),
                _maybe_idx("rot_w"),
            )

        self._dim: int = sum(ex.dim() for _, ex in self.extractors)

    # --------------------------------------------------------------
    def extract_all(self, seq: np.ndarray) -> np.ndarray:
        parts = [ex.extract(seq) for _, ex in self.extractors]
        vec = np.concatenate(parts, dtype=np.float32) if parts else np.empty(0, dtype=np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    # --------------------------------------------------------------
    def dim(self) -> int:
        return self._dim

    __len__ = dim  # alias

    # -------------------------- names ---------------------------
    def feature_names(self) -> List[str]:
        names: List[str] = []
        for tag, ex in self.extractors:
            names += [f"{tag}_{i}" for i in range(ex.dim())]
        return names


# ---------------------------------------------------------------------------
# Debug sanity
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dummy_cols = [
        "acc_x",
        "acc_y",
        "acc_z",
        "rot_w",
        "rot_x",
        "rot_y",
        "rot_z",
        *[f"thm_{i}" for i in range(1, 6)],
        "tof_1_v0",
    ]
    seq = np.zeros((200, len(dummy_cols)), dtype=np.float32)
    bank = SymbolicFeatureBank(dummy_cols)
    vec = bank.extract_all(seq)
    print("dim", bank.dim(), vec.shape, "feature names", bank.feature_names())
