"""
Time-of-Flight (ToF) grid encoder using a lightweight Swin-style
Transformer backbone followed by a GRU.  Each frame is interpreted as
``in_channels`` × ``img_size`` × ``img_size`` and processed independently
before temporal aggregation with a GRU.
"""

from __future__ import annotations

from typing import Tuple

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover
    raise ModuleNotFoundError(
        "PyTorch is required for tof_encoder.py.\n"
        "Install with   pip install torch   (CPU build is fine)."
    )


# ---------------------------------------------------------------------------
class _SwinBlock(nn.Module):
    """Minimal Swin-style transformer block."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class _MiniSwin(nn.Module):
    """Very small patch-based transformer used as backbone."""

    def __init__(self, in_ch: int, img_size: int, patch: int, dim: int, depth: int, heads: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        num_patches = (img_size // patch) ** 2
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dim))
        self.blocks = nn.ModuleList([_SwinBlock(dim, heads) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)              # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)     # (B, N, dim)
        x = x + self.pos[:, : x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        return x.mean(dim=1)                 # (B, dim)


# ---------------------------------------------------------------------------
class ToFEncoder(nn.Module):
    """Swin-style ToF encoder followed by a GRU."""

    def __init__(
        self,
        in_channels: int = 5,
        img_size: int = 8,
        n_classes: int = 18,
        latent_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.in_channels = in_channels

        self.swin = _MiniSwin(in_channels, img_size, patch=2, dim=latent_dim, depth=depth, heads=heads)
        self.gru = nn.GRU(latent_dim, latent_dim, batch_first=True)
        self.classifier = nn.Linear(latent_dim, n_classes)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_logits: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(0)

        B, T, F = x.shape
        frame = x.view(B * T, self.in_channels, self.img_size, self.img_size)
        feat = self.swin(frame)                 # (B*T, latent_dim)
        feat = feat.view(B, T, self.latent_dim) # (B, T, latent_dim)

        _, h_n = self.gru(feat)
        latent = h_n[-1]
        logits = self.classifier(latent)
        return (latent, logits) if return_logits else latent

    # ------------------------------------------------------------------
    def get_output_dim(self) -> int:
        return self.latent_dim


if __name__ == "__main__":  # pragma: no cover
    B, T = 2, 10
    dummy = torch.randn(B, T, 5 * 8 * 8)
    model = ToFEncoder()
    out, log = model(dummy, return_logits=True)
    print("latent", out.shape, "logits", log.shape)
