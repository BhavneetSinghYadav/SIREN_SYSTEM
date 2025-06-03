"""
IMU sequence encoder combining a temporal CNN with a bidirectional GRU.

* Input  : ``(B, T, F)`` or ``(T, F)``
* Output : latent vector ``(B, latent_dim)``

The interface mirrors :class:`CNNEncoder` so that it can be swapped into
:class:`FusionNet` without changes.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - torch missing
    raise ModuleNotFoundError(
        "PyTorch is required for imu_encoder.py.\n"
        "Install with   pip install torch   (CPU build is fine)."
    )


# ---------------------------------------------------------------------------
class IMUEncoder(nn.Module):
    """Temporal CNN + Bi-GRU encoder for IMU streams."""

    def __init__(
        self,
        in_channels: int = 7,
        n_classes: int = 18,
        latent_dim: int = 128,
        gru_layers: int = 1,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # simple temporal CNN backbone
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True),
        )

        # bidirectional GRU over conv features
        self.gru = nn.GRU(
            input_size=latent_dim,
            hidden_size=latent_dim // 2,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(latent_dim, n_classes)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # CNN expects (B, C, T)
        h = self.conv(x.permute(0, 2, 1))  # (B, latent_dim, T)
        h = h.permute(0, 2, 1)             # (B, T, latent_dim)

        _, h_n = self.gru(h)               # h_n: (layers*2, B, latent_dim//2)
        # take last layer's forward & backward states
        latent = torch.cat([h_n[-2], h_n[-1]], dim=1)
        logits = self.classifier(latent)

        return (latent, logits) if return_logits else latent

    # ------------------------------------------------------------------
    def get_output_dim(self) -> int:
        """Return dimensionality of the latent vector."""
        return self.latent_dim


if __name__ == "__main__":  # pragma: no cover - smoke test
    B, T, F = 4, 200, 7
    x = torch.randn(B, T, F)
    model = IMUEncoder(in_channels=F, n_classes=18, latent_dim=128)
    lat, log = model(x, return_logits=True)
    print("latent", lat.shape, "logits", log.shape)
