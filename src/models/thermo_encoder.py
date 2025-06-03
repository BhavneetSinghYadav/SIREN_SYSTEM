"""
Thermopile stream encoder using a shallow Conv1D backbone and
attention pooling.  Suitable for drop-in use with ``FusionNet``.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for thermo_encoder.py.\n"
        "Install with   pip install torch   (CPU build is fine)."
    )


# ---------------------------------------------------------------------------
class _AttPool(nn.Module):
    """Simple attention pooling over the temporal dimension."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        w = self.score(x)                # (B, T, 1)
        w = F.softmax(w, dim=1)
        return torch.sum(w * x, dim=1)   # (B, D)


# ---------------------------------------------------------------------------
class ThermoEncoder(nn.Module):
    """Shallow Conv1D encoder with attention pooling."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 18,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.pool = _AttPool(latent_dim)
        self.classifier = nn.Linear(latent_dim, n_classes)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)

        h = self.conv(x.permute(0, 2, 1))  # (B, D, T)
        h = h.permute(0, 2, 1)             # (B, T, D)
        latent = self.pool(h)
        logits = self.classifier(latent)
        return (latent, logits) if return_logits else latent

    # ------------------------------------------------------------------
    def get_output_dim(self) -> int:
        return self.latent_dim


if __name__ == "__main__":  # pragma: no cover
    B, T, F = 4, 200, 1
    x = torch.randn(B, T, F)
    model = ThermoEncoder(in_channels=F)
    lat, log = model(x, return_logits=True)
    print("latent", lat.shape, "logits", log.shape)
