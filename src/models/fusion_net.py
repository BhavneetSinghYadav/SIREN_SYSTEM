"""
src/models/fusion_net.py  ·  v1.2
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fusion network that merges
  • sequence latent  (B, seq_dim)
  • symbolic vector  (B, sym_dim)

New in v1.2
-----------
* supports  ``fusion_type = {"concat","gated"}``
* auto-detects sym_dim from first batch if set to -1
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
class _MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        h = max(64, in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ──────────────────────────────────────────────────────────────
class FusionNet(nn.Module):
    def __init__(
        self,
        sequence_encoder: nn.Module,
        sym_dim: int = -1,                # -1 → infer at runtime
        n_classes: int = 18,
        fusion_type: str = "concat",      # "concat" | "gated"
    ):
        super().__init__()
        assert fusion_type in {"concat", "gated"}
        self.encoder      = sequence_encoder
        self.sym_dim_init = sym_dim
        self._sym_dim     = sym_dim       # may update after first forward
        self.fusion_type  = fusion_type

        self.seq_dim   = self.encoder.get_output_dim()
        fused_dim      = self.seq_dim if fusion_type == "gated" else self.seq_dim + sym_dim
        self.classifier = _MLPHead(fused_dim, n_classes)

        if fusion_type == "gated":
            # gate: symbolic → [0,1] sigmoid weights on latent dim
            self.gate_fc = nn.Sequential(
                nn.Linear(self._sym_dim if self._sym_dim > 0 else 1, self.seq_dim),
                nn.Sigmoid()
            )

    # ──────────────────────────────────────────────────────────
    def _maybe_init_gate(self, sym_dim: int, device, dtype):
        """Lazy-create gate_fc at first call when sym_dim was unknown."""
        if self.gate_fc[0].in_features == 1:       # placeholder
            self._sym_dim = sym_dim
            self.gate_fc = nn.Sequential(
                nn.Linear(sym_dim, self.seq_dim),
                nn.Sigmoid()
            ).to(device=device, dtype=dtype)

    # ──────────────────────────────────────────────────────────
    def forward(
        self,
        seq_tensor: torch.Tensor,            # (B,T,F)
        sym_tensor: Optional[torch.Tensor] = None,
        return_latent: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:

        latent = self.encoder(seq_tensor, return_logits=False)   # (B, seq_dim)

        # -------- symbolic handling ----------------------------------
        if sym_tensor is None:
            sym_tensor = torch.zeros(latent.size(0), 0,
                                     device=latent.device, dtype=latent.dtype)

        if sym_tensor.ndim == 1:        # (sym_dim,)  → (1,sym_dim)
            sym_tensor = sym_tensor.unsqueeze(0)

        if self.fusion_type == "concat":
            fused = torch.cat([latent, sym_tensor.float()], dim=1)

        else:  # gated
            if self._sym_dim_init == -1 and self.gate_fc[0].in_features == 1:
                self._maybe_init_gate(sym_tensor.size(1), latent.device, latent.dtype)

            gate = self.gate_fc(sym_tensor.float())          # (B, seq_dim)
            fused = latent * gate                            # element-wise gating

        logits = self.classifier(fused)

        return (fused, logits) if return_latent else logits

    # ----------------------------------------------------------------
    def get_output_dim(self) -> int:
        """Dimension of the vector fed into `classifier` (for downstream use)."""
        if self.fusion_type == "concat":
            return self.seq_dim + self._sym_dim
        return self.seq_dim
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from models.transformer_encoder import TransformerEncoder
    B, T, F, S = 3, 200, 332, 16
    seq = torch.randn(B, T, F)
    sym = torch.randn(B, S)

    enc = TransformerEncoder(in_channels=F, n_classes=18, latent_dim=128)
    net = FusionNet(enc, sym_dim=S, n_classes=18, fusion_type="gated")

    lat, log = net(seq, sym, return_latent=True)
    print("latent", lat.shape, "logits", log.shape)
