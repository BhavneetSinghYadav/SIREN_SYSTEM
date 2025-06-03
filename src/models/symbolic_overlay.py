from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SymbolicOverlay(nn.Module):
    """Fuse a sensor's latent vector with its symbolic features."""

    def __init__(self, latent_dim: int, sym_dim: int, fusion_type: str = "concat") -> None:
        super().__init__()
        assert fusion_type in {"concat", "gated"}, "fusion_type must be 'concat' or 'gated'"
        self.latent_dim = latent_dim
        self.sym_dim = sym_dim
        self.fusion_type = fusion_type

        if fusion_type == "gated":
            self.gate_fc = nn.Sequential(
                nn.Linear(sym_dim, latent_dim),
                nn.Sigmoid(),
            )
            self._out_dim = latent_dim
        else:
            self.gate_fc = None
            self._out_dim = latent_dim + sym_dim

    # ------------------------------------------------------------------
    def forward(self, latent: torch.Tensor, sym: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "concat":
            return torch.cat([latent, sym.float()], dim=1)
        gate = self.gate_fc(sym.float())
        return latent * gate

    # ------------------------------------------------------------------
    def get_output_dim(self) -> int:
        return self._out_dim
