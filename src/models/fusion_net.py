"""
src/models/fusion_net.py
~~~~~~~~~~~~~~~~~~~~~~~~

Fusion network that merges:
  • Learned latent embedding from a sequence encoder   (e.g., CNNEncoder)
  • Symbolic feature vector (e.g., rhythmicity, posture, tremor_entropy)

Workflow
--------
latent_seq = sequence_encoder(seq)          # (B, seq_dim)
x_sym      = symbolic_vector                # (B, sym_dim)  (can be empty)
fusion     = [latent_seq | x_sym]           # concat
logits     = classifier(fusion)             # (B, n_classes)

Compatible with the SIREN project coding contract.

Author : Bhavya  — 2025
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Helper : simple MLP head
# ──────────────────────────────────────────────────────────────────────────
class _MLPHead(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden: int = 128,
                 n_classes: int = 18,
                 dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────
# FusionNet
# ──────────────────────────────────────────────────────────────────────────
class FusionNet(nn.Module):
    """
    Parameters
    ----------
    sequence_encoder : nn.Module
        Any module with  (latent, ...) = forward(seq, return_logits=False)
        and  .get_output_dim()
    sym_dim : int
        Dimensionality of the symbolic feature vector.
        If 0 ⇒ model treats input as IMU-only.
    n_classes : int
        Output gesture classes.
    """

    def __init__(self,
                 sequence_encoder: nn.Module,
                 sym_dim: int = 3,
                 n_classes: int = 18):
        super().__init__()

        self.encoder  = sequence_encoder
        self.sym_dim  = sym_dim
        self.seq_dim  = self.encoder.get_output_dim()
        self.fused_dim = self.seq_dim + sym_dim

        self.classifier = _MLPHead(
            in_dim=self.fused_dim,
            hidden=max(64, self.fused_dim // 2),
            n_classes=n_classes,
            dropout=0.3,
        )

    # ------------------------------------------------------------------ #
    def forward(self,
                seq_tensor: torch.Tensor,
                sym_tensor: Optional[torch.Tensor] = None,
                return_latent: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Parameters
        ----------
        seq_tensor : (B, T, F) or (T, F) torch.Tensor
        sym_tensor : (B, sym_dim) or (sym_dim,) or None
            Symbolic features *per sequence*.  If None or sym_dim == 0,
            an all-zeros placeholder is concatenated.
        return_latent : bool
            If True → returns (fused_latent, logits)
            else       returns logits only.

        Returns
        -------
        logits : (B, n_classes)
        or (latent, logits)
        """
        latent_seq = self.encoder(seq_tensor, return_logits=False)  # (B, seq_dim)

        # Ensure batch shape for sym features
        if self.sym_dim == 0 or sym_tensor is None:
            # create zero placeholder (no symbolic features)
            sym = torch.zeros(latent_seq.size(0), 0,
                              device=latent_seq.device, dtype=latent_seq.dtype)
        else:
            # Handle (sym_dim,) → (1, sym_dim)
            if sym_tensor.ndim == 1:
                sym_tensor = sym_tensor.unsqueeze(0)
            sym = sym_tensor.float()

        fused = torch.cat([latent_seq, sym], dim=1)                 # (B, fused_dim)
        logits = self.classifier(fused)

        if return_latent:
            return fused, logits
        return logits

    # ------------------------------------------------------------------ #
    def get_output_dim(self) -> int:
        """Return fused latent dimensionality."""
        return self.fused_dim


# ──────────────────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from cnn_encoder import CNNEncoder   # local relative import when run directly

    B, T, F = 4, 200, 7
    seq     = torch.randn(B, T, F)
    sym     = torch.randn(B, 3)          # rhythmicity, posture, tremor

    cnn   = CNNEncoder(in_channels=F, n_classes=18, latent_dim=128)
    model = FusionNet(cnn, sym_dim=3, n_classes=18)

    fused, logits = model(seq, sym, return_latent=True)
    print("fused shape :", fused.shape)   # (4, 131)
    print("logits shape:", logits.shape)  # (4, 18)
# Fusion model of symbolic and neural features
