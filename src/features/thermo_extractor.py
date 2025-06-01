"""
src/features/thermo_extractor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic extractor for the *five* thermopile sensors (thm_1 … thm_5).
Each frame has exactly 5 temperature readings in °C.  For gesture-level
symbolics we summarise **per‑sensor mean temperature** over the entire
sequence, yielding a **5‑dimensional** vector:

    [ mean_thm1 , mean_thm2 , … , mean_thm5 ]

Rationale:
  • Keeps spatial separation of the five thermopile diodes
  • Compact (dim=5) yet informative for proximity / skin‑contact heat

The extractor is intentionally simple; you can later swap it for something
richer (mean+std, time‑derivative, etc.) without changing its public API.
"""

from __future__ import annotations

import numpy as np

NUM_THM = 5              # thm_1 … thm_5


class ThermoExtractor:
    """Return a 5‑D mean‑temperature vector for a sequence."""

    def __init__(self, start_idx: int):
        """
        Parameters
        ----------
        start_idx : int
            Column index where `thm_1` begins inside the full (T,F) matrix.
        """
        self.start = start_idx
        self.end   = start_idx + NUM_THM

    # --------------------------------------------------------------
    def extract(self, seq: np.ndarray) -> np.ndarray:
        """Compute per‑sensor mean over time.

        Returns
        -------
        np.ndarray shape (5,)  dtype float32
        """
        thm_seq = seq[:, self.start:self.end].astype(np.float32)  # (T,5)
        return thm_seq.mean(axis=0, dtype=np.float32)

    # --------------------------------------------------------------
    @staticmethod
    def dim() -> int:
        return NUM_THM


# Smoke‑test ----------------------------------------------------------
if __name__ == "__main__":
    dummy = np.random.uniform(28, 36, size=(200, NUM_THM)).astype(np.float32)
    seq_full = np.concatenate([np.zeros((200, 7)), dummy], axis=1)

    ext = ThermoExtractor(start_idx=7)
    vec = ext.extract(seq_full)
    print("Thermo vector:", vec, vec.shape)
