"""
src/models/cnn_encoder.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Minimal 1-D Convolutional encoder for IMU sequences.

• Input  : tensor  (B, T, F)   or  (T, F)
• Output : latent (B, 128)     or  (128,)   via .forward(return_logits=False)
• Optionally returns class-logits when return_logits=True
            logits shape  -> (B, n_classes)

All layers / parameters kept lightweight for training on Kaggle / Colab T4.

Author : Bhavya  (Bhavneet Singh Yadav) — 2025
"""

from __future__ import annotations
from typing import Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:   # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for cnn_encoder.py.\n"
        "Install with   pip install torch   (CPU build is fine for prototyping)."
    )


# ──────────────────────────────────────────────────────────────────────────
# Core network
# ──────────────────────────────────────────────────────────────────────────
class CNNEncoder(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Number of input features per timestep (e.g., 7 for IMU v0).
    n_classes : int
        Number of gesture classes for classification head.
    latent_dim : int
        Dimensionality of the latent embedding exposed to fusion nets.
    """

    def __init__(self,
                 in_channels: int = 7,
                 n_classes: int = 18,
                 latent_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim  = latent_dim
        self.n_classes   = n_classes

        # Backbone : Conv → BN → ReLU → Pool  (×3)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(output_size=1)   # → (B, latent_dim, 1)
        )

        self.classifier = nn.Linear(latent_dim, n_classes)

    # ---------------------------------------------- #
    def forward(self,
                x: torch.Tensor,
                return_logits: bool = False
                ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        x : Tensor
            (B, T, F)  or  (T, F) where F = in_channels.
        return_logits : bool
            If True  → returns (latent, logits)
            Else     → returns latent only.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)            # (1, T, F)

        # (B, T, F) → (B, F, T)  for Conv1d
        x = x.permute(0, 2, 1)

        latent = self.features(x).squeeze(-1)    # (B, latent_dim)
        logits = self.classifier(latent)         # (B, n_classes)

        if return_logits:
            return latent, logits
        return latent

    # ---------------------------------------------- #
    def get_output_dim(self) -> int:
        """Return latent vector dimensionality (for FusionNet)."""
        return self.latent_dim


# ──────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    B, T, F = 4, 200, 7
    dummy   = torch.randn(B, T, F)
    model   = CNNEncoder(in_channels=F, n_classes=18, latent_dim=128)
    lat, log = model(dummy, return_logits=True)
    print("latent shape :", lat.shape)   # (4, 128)
    print("logits shape :", log.shape)   # (4, 18)
# 1D CNN model
