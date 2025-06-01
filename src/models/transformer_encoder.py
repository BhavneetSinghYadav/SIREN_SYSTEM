# ────────────────────────────────────────────────────────────────
# src/models/transformer_encoder.py
# ----------------------------------------------------------------
"""
TransformerEncoder – sequence-level embedding for SIREN
-------------------------------------------------------
* Input : (B, T, F)  or  (T, F)
* Output: latent (B, latent_dim)      via .forward(return_logits=False)
          + optional logits (B, n_classes)
Interface identical to CNNEncoder so FusionNet & train script stay unchanged.
"""
from __future__ import annotations
from typing import Tuple

import math
import torch
import torch.nn as nn

# ----------------------------------------------------------------
class _PositionalEncoding(nn.Module):
    """Fixed sinusoidal PE (same as Vaswani et al.)."""
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)           # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[: x.size(1)].unsqueeze(0)


# ----------------------------------------------------------------
class TransformerEncoder(nn.Module):
    """
    Parameters
    ----------
    in_channels : int   • feature dim per timestep  (e.g. 332)
    n_classes   : int   • gesture count
    latent_dim  : int   • model dimension D (d_model) & output pooling size
    n_layers    : int   • transformer blocks
    n_heads     : int   • attention heads
    dropout     : float • dropout throughout
    """
    def __init__(self,
                 in_channels: int,
                 n_classes: int,
                 latent_dim: int = 128,
                 n_layers: int = 4,
                 n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes  = n_classes

        # 1) linear projection from raw features → d_model
        self.input_proj = nn.Linear(in_channels, latent_dim)

        # 2) positional encoding
        self.pos_enc = _PositionalEncoding(latent_dim)

        # 3) transformer encoder stack
        enc_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                               nhead=n_heads,
                                               dim_feedforward=latent_dim * 4,
                                               dropout=dropout,
                                               batch_first=True,
                                               norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # 4) sequence pooling (mean)   → latent vector
        self.pool = nn.AdaptiveAvgPool1d(1)      # (B, D, 1)

        # 5) classifier head
        self.classifier = nn.Linear(latent_dim, n_classes)

    # ------------------------------------------------------------
    def forward(self,
                x: torch.Tensor,
                return_logits: bool = False
                ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        x : (B,T,F) or (T,F)
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # project & encode
        h = self.input_proj(x)              # (B,T,D)
        h = self.pos_enc(h)
        h = self.encoder(h)                 # (B,T,D)

        # mean-pool over time
        latent = self.pool(h.transpose(1, 2)).squeeze(-1)   # (B,D)
        logits = self.classifier(latent)

        return (latent, logits) if return_logits else latent

    # ------------------------------------------------------------
    def get_output_dim(self) -> int:
        return self.latent_dim


# ----------------------------------------------------------------
if __name__ == "__main__":
    B, T, F = 4, 200, 332
    x = torch.randn(B, T, F)
    model = TransformerEncoder(in_channels=F, n_classes=18, latent_dim=128)
    lat, log = model(x, return_logits=True)
    print("latent:", lat.shape, "logits:", log.shape)
# ----------------------------------------------------------------
