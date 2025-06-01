"""
src/features/rhythmicity.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Symbolic Feature #1 – Rhythmicity Index
---------------------------------------
Quantifies the degree of periodic / loop-like motion in a wrist-sensor
sequence.  Compulsive BFRB gestures are *hypothesised* to show stronger,
stable rhythmic components than short, deliberate, non-compulsive gestures.

Output
------
For every sequence we return a 3-element NumPy vector:

[  dominant_freq_hz ,
   spectral_entropy  ,
   rhythmicity_score ]

• dominant_freq_hz  – frequency (Hz) of strongest peak in the band-limited
  magnitude spectrum (0.5–6 Hz, typical human hand-motion range).

• spectral_entropy  – Shannon entropy of the normalised power spectrum
  (lower  ⇒ more periodic / tonally concentrated).

• rhythmicity_score – 1 − spectral_entropy, ranged 0–1 for convenience.

Author: Bhavya  (Bhavneet Singh Yadav) — 2025
"""

from __future__ import annotations
from pathlib import Path
from typing  import List, Tuple, Dict

import numpy as np

# Optional PyTorch import for GPU-accelerated FFT
try:
    import torch
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _next_pow_two(n: int) -> int:
    """Return next power of two ≥ n —for efficient FFT."""
    return 1 << (n - 1).bit_length()


def _compute_fft(signal_1d: np.ndarray, sample_rate: float = 50.0
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    signal_1d : (T,) float32
        Magnitude signal (e.g., |acc|) per time-step.
    sample_rate : float
        Samples per second.  50 Hz ≈ typical BNO080 output.

    Returns
    -------
    freq  : (K,) frequencies in Hz
    power : (K,) power spectrum (normalised)
    """
    n = _next_pow_two(len(signal_1d))
    if TORCH_AVAILABLE:
        x = torch.tensor(signal_1d, dtype=torch.float32, device="cpu")
        fft = torch.fft.rfft(x, n=n)
        power = (fft.real ** 2 + fft.imag ** 2).cpu().numpy()
    else:
        fft = np.fft.rfft(signal_1d, n=n)
        power = np.abs(fft) ** 2

    freq = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    power = power / (power.sum() + 1e-9)          # normalise to 1.0
    return freq, power


def _spectral_entropy(power: np.ndarray, eps: float = 1e-9) -> float:
    """Shannon entropy  (0 = pure tone, 1 = white-noise)."""
    entropy = -np.sum(power * (np.log(power + eps)))
    # divide by log(K) to bound 0–1
    return float(entropy / np.log(len(power) + eps))


# --------------------------------------------------------------------------- #
# Core class
# --------------------------------------------------------------------------- #
class RhythmicityExtractor:
    """
    Compute rhythmicity features for a single sequence or a batch.

    Usage
    -----
    >>> from rhythmicity import RhythmicityExtractor
    >>> extractor = RhythmicityExtractor()
    >>> feats = extractor.extract(sequence)         # (3,) vector
    >>> names = extractor.get_feature_names()
    """

    def __init__(self,
                 sample_rate: float = 50.0,
                 freq_band: Tuple[float, float] = (0.5, 6.0)):
        self.fs        = sample_rate
        self.f_low, self.f_high = freq_band

    # --------------------------------------------------------------------- #
    def _extract_one(self, seq: np.ndarray) -> np.ndarray:
        """
        seq : (T, F) array (IMU columns after normalization).
              We expect acc_x, acc_y, acc_z among features 0–2.

        Returns  (3,) float32  (dominant_freq_hz, spec_entropy, rhythm_score)
        """
        acc_mag = np.linalg.norm(seq[:, :3], axis=1)    # magnitude signal
        freq, power = _compute_fft(acc_mag, self.fs)

        # Band-limit
        mask = (freq >= self.f_low) & (freq <= self.f_high)
        freq_band = freq[mask]
        pwr_band  = power[mask]

        # Dominant frequency within band
        dom_idx = int(np.argmax(pwr_band))
        dom_freq = float(freq_band[dom_idx])

        # Spectral entropy
        entropy = _spectral_entropy(pwr_band)
        rhythmicity = 1.0 - entropy                    # invert for convenience

        return np.asarray([dom_freq, entropy, rhythmicity], dtype=np.float32)

    # --------------------------------------------------------------------- #
    def extract(self, sequences: np.ndarray | List[np.ndarray]
                ) -> np.ndarray:
        """
        sequences : array-like
            • If 2-D (T,F) → single sequence.
            • If list/3-D (N,T,F) → batch.

        Returns
        -------
        feats : (N,3) or (3,) NumPy array
        """
        if isinstance(sequences, np.ndarray) and sequences.ndim == 2:
            return self._extract_one(sequences)
        # Batch mode
        batch_feats = [self._extract_one(seq) for seq in sequences]
        return np.vstack(batch_feats)

    # --------------------------------------------------------------------- #
    @staticmethod
    def get_feature_names() -> List[str]:
        return ["dom_freq_hz", "spectral_entropy", "rhythmicity_score"]


# --------------------------------------------------------------------------- #
# CLI quick-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Minimal sanity test with white-noise vs sine-wave
    import matplotlib.pyplot as plt

    fs = 50
    t  = np.arange(0, 4, 1/fs)
    sine = np.sin(2 * np.pi * 2.0 * t)          # 2 Hz tone
    noise = np.random.randn(len(t))

    def to_seq(signal):
        # duplicate into fake (T,7) IMU channels
        return np.tile(signal[:, None], (1, 7)).astype(np.float32)

    ext = RhythmicityExtractor(sample_rate=fs)
    print("Sine → ", ext.extract(to_seq(sine)))
    print("Noise → ", ext.extract(to_seq(noise)))
# Extract rhythmicity features
