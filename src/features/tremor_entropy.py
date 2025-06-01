"""
src/features/tremor_entropy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tremor‑entropy extractor
------------------------
A single scalar that quantifies *micro‑vibration irregularity* in the IMU
accelerometer signal.  High values ⇒ jittery, chaotic motion;  low values ⇒
smooth / periodic motion.

Algorithm (FFT‑entropy)
^^^^^^^^^^^^^^^^^^^^^^^
1. Concatenate the three accelerometer axes  → 1‑D vector of length 3*T.
2. Apply Hamming window, zero‑pad to next power‑of‑two, take FFT.
3. Convert to power spectrum (magnitude²).
4. Normalise the spectrum to a probability vector  p.
5. Shannon entropy  H = −Σ p * log₂ p  (eps‑guarded).
6. Scale to 0‑1 by dividing by  log₂(N/2)  (max entropy for real FFT).

Returns a *float32*, always finite.
"""

from __future__ import annotations
import numpy as np

ACC_IDXS = [0, 1, 2]  # acc_x, acc_y, acc_z are first three IMU columns

class TremorEntropyExtractor:
    """Compute FFT‑entropy of concatenated accelerometer signal."""

    def __init__(self, acc_start_idx: int = 0):
        # index of acc_x inside full feature vector (0 for processed CSV)
        self.acc_start = acc_start_idx
        self.acc_end   = acc_start_idx + 3  # x,y,z

    # ------------------------------------------------------------------
    def extract(self, seq: np.ndarray) -> np.ndarray:
        """Return a (1,) numpy array with the entropy scalar."""
        acc = seq[:, self.acc_start:self.acc_end].astype(np.float32)  # (T,3)
        flat = acc.reshape(-1)

        # Remove DC component
        flat = flat - flat.mean()
        if np.all(flat == 0):
            return np.array([0.0], dtype=np.float32)

        # zero‑pad to next power of two for efficient FFT
        n_orig = flat.size
        n_fft  = 1 << (n_orig - 1).bit_length()
        win    = np.hamming(n_orig)
        fft_in = flat * win
        fft_in = np.pad(fft_in, (0, n_fft - n_orig))

        spec   = np.abs(np.fft.rfft(fft_in))**2  # real‑valued power spectrum
        if spec.sum() == 0:
            return np.array([0.0], dtype=np.float32)
        p      = spec / spec.sum()
        entropy = -np.sum(p * np.log2(p + 1e-12))

        max_h  = np.log2(p.size)  # entropy upper bound
        norm_H = float(entropy / max_h)  # 0‑1 range
        return np.array([norm_H], dtype=np.float32)

    # ------------------------------------------------------------------
    @staticmethod
    def dim() -> int:
        return 1


# Smoke‑test -----------------------------------------------------------
if __name__ == "__main__":
    # random wobble
    T = 200
    dummy = np.random.randn(T, 332).astype(np.float32)
    ext = TremorEntropyExtractor(acc_start_idx=0)
    val = ext.extract(dummy)
    print("tremor_entropy:", val, val.shape)
# Analyze micro-shocks
