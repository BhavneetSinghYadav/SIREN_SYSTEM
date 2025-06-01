"""
src/features/rhythmicity.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic Feature #1 – Rhythmicity Index
---------------------------------------
<… doc-string unchanged …>
"""

from __future__ import annotations
from typing import List, Tuple

import numpy as np

# Optional GPU FFT
try:
    import torch
    _TORCH_OK = True
except ModuleNotFoundError:
    _TORCH_OK = False
# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _fft_power(sig: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return freq (Hz) & normalised power spectrum."""
    n = _next_pow2(len(sig))
    if _TORCH_OK:
        x = torch.tensor(sig, dtype=torch.float32, device="cpu")
        fft = torch.fft.rfft(x, n=n)
        pwr = (fft.real ** 2 + fft.imag ** 2).cpu().numpy()
    else:
        pwr = np.abs(np.fft.rfft(sig, n=n)) ** 2
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    return freq, pwr / (pwr.sum() + 1e-9)


def _spectral_entropy(pwr: np.ndarray, eps: float = 1e-9) -> float:
    ent = -np.sum(pwr * np.log(pwr + eps))
    return float(ent / np.log(len(pwr) + eps))         # 0‒1
# ──────────────────────────────────────────────────────────────
class RhythmicityExtractor:
    """Return a 3-element vector describing periodicity of a sequence."""

    _FEAT_NAMES = ["dom_freq_hz", "spectral_entropy", "rhythmicity_score"]

    def __init__(self, sample_rate: float = 50.0, band: Tuple[float, float] = (0.5, 6.0)):
        self.fs = sample_rate
        self.f_lo, self.f_hi = band

    # public ---------------------------------------------------
    def extract(self, seq: np.ndarray | list[np.ndarray]) -> np.ndarray:
        """
        seq : (T,F)  or list/array of such matrices.
        Returns  (3,) or (N,3) np.float32
        """
        if isinstance(seq, np.ndarray) and seq.ndim == 2:
            return self._feat(seq)
        return np.vstack([self._feat(s) for s in seq]).astype(np.float32)

    # required by SymbolicFeatureBank -------------------------
    def dim(self) -> int:        # <-- added ✔
        return len(self._FEAT_NAMES)

    # convenience ---------------------------------------------
    @classmethod
    def get_feature_names(cls) -> List[str]:
        return cls._FEAT_NAMES.copy()

    # internal -------------------------------------------------
    def _feat(self, seq: np.ndarray) -> np.ndarray:
        acc_mag = np.linalg.norm(seq[:, :3], axis=1)
        f, p = _fft_power(acc_mag, self.fs)

        mask = (f >= self.f_lo) & (f <= self.f_hi)
        f, p = f[mask], p[mask]

        dom_freq = float(f[int(np.argmax(p))])
        entropy  = _spectral_entropy(p)
        rhythmic = 1.0 - entropy
        return np.array([dom_freq, entropy, rhythmic], dtype=np.float32)


# smoke-test -----------------------------------------------------------
if __name__ == "__main__":
    fs = 50
    t  = np.arange(0, 4, 1 / fs)
    sig = np.sin(2 * np.pi * 2.0 * t)
    seq = np.tile(sig[:, None], (1, 7)).astype(np.float32)

    ext = RhythmicityExtractor(fs)
    print(ext.extract(seq))        # should print 3-element vector
