"""
src/features/posture_anchor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Posture‑anchor extractor
-----------------------
Derives a **3‑D posture embedding** `[pitch_mean, roll_mean, yaw_mean]` from the
quaternion orientation stream (`rot_w, rot_x, rot_y, rot_z`).  These statistics
encode the dominant device pose during a gesture.

* Quaternion → Euler conversion uses aerospace (ZYX) convention:
    yaw (ψ)   – rotation around Z
    pitch (θ) – rotation around Y
    roll (φ)  – rotation around X

Values are expressed in **radians** and normalised to roughly −1…1 by dividing
by π (so downstream network stays numerically stable).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------
# Quaternion → Euler (ZYX) helper
# ---------------------------------------------------------------------

def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """Vectorised quaternion (w,x,y,z) → (roll, pitch, yaw) in radians."""
    w, x, y, z = q.T  # each (N,)

    # roll (x‑axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    # pitch (y‑axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch = np.arcsin(t2)

    # yaw (z‑axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    return np.stack((roll, pitch, yaw), axis=1)  # (N,3)

# ---------------------------------------------------------------------
class PostureAnchorExtractor:
    """Return mean Euler angles (scaled) for the entire sequence."""

    def __init__(self, rot_start_idx: int):
        self.rot_start = rot_start_idx        # where rot_w starts
        self.rot_end   = rot_start_idx + 4

    # --------------------------------------------------------------
    def extract(self, seq: np.ndarray) -> np.ndarray:
        rot = seq[:, self.rot_start:self.rot_end].astype(np.float32)  # (T,4)
        eul = quat_to_euler(rot)                                     # (T,3)
        mean_angles = eul.mean(axis=0) / np.pi                       # normalise
        return mean_angles.astype(np.float32)                        # (3,)

    # --------------------------------------------------------------
    @staticmethod
    def dim() -> int:
        return 3

# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy seq with random quats (not unit‑norm, but fine for smoke test)
    T = 200
    seq = np.zeros((T, 332), dtype=np.float32)
    rand_q = np.random.randn(T, 4).astype(np.float32)
    rand_q /= np.linalg.norm(rand_q, axis=1, keepdims=True)
    seq[:, 3:7] = rand_q  # assuming rot_w starts at idx 3

    ext = PostureAnchorExtractor(rot_start_idx=3)
    vec = ext.extract(seq)
    print("posture vec:", vec, vec.shape)
# Detect posture fixation
