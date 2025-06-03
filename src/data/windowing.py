"""Utility helpers for segmenting a time-series into fixed windows."""

from __future__ import annotations

import numpy as np


def sliding_window(seq: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Return an array of shape ``(N, window, F)`` comprised of sliding windows.

    Parameters
    ----------
    seq : np.ndarray
        Input array of shape ``(T, F)``.
    window : int
        Length of each window.
    stride : int
        Step size between windows.

    Returns
    -------
    np.ndarray
        Array of windows.  If ``T < window`` the sequence is padded with the
        edge value so at least one window is produced.
    """

    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")

    t, f = seq.shape
    if t < window:
        pad = window - t
        seq = np.pad(seq, ((0, pad), (0, 0)), mode="edge")
        t = window

    starts = np.arange(0, t - window + 1, stride)
    if starts.size == 0:
        starts = np.array([0])

    windows = np.stack([seq[s : s + window] for s in starts])
    return windows


__all__ = ["sliding_window"]
