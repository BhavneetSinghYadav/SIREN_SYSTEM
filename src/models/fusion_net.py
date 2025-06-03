"""
Fusion network that merges
  • sequence latent  (B, seq_dim)
  • symbolic vector  (B, sym_dim)

v1.3 – 2025‑06‑02
-----------------
* Robust lazy initialisation when `sym_dim == -1` for **both** `concat` and `gated` modes
* Handles encoder outputs shaped **(B, T, D)** by mean‑pooling (configurable)
* Adds defensive shape assertions & type hints
* Keeps full backward‑compatibility with `FusionNet` v1.2 public API
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

# ──────────────────────────────────────────────────────────────
class _MLPHead(nn.Module):
    """Two‑layer MLP classifier with BatchNorm + Dropout."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        return self.net(x)


# ──────────────────────────────────────────────────────────────
class FusionNet(nn.Module):
    """Merge a sequence encoder representation with a symbolic vector.

    Parameters
    ----------
        sequence_encoder : nn.Module | None
            Must implement `forward(seq) -> Tensor` **and** `get_output_dim()`.
            The forward may return either `(B, seq_dim)` or `(B, T, seq_dim)`.
            If ``None``, ``input_dim`` must be provided and ``FusionNet`` acts
            purely as a classifier over pre-fused sensor vectors.
    sym_dim : int, default ``-1``
        Dimensionality of the symbolic vector.  ``-1`` → infer on the first call.
    n_classes : int, default ``18``
        Number of output classes for classification.
    fusion_type : {"concat", "gated"}, default "concat"
        • **concat**: `[latent, sym]` → classifier
        • **gated**: `latent * σ(sym → gate)` → classifier
    pool : {"mean", "cls"}, default "mean"
        Pooling strategy if encoder returns a temporal tensor (B, T, D).
    """

    def __init__(
        self,
        sequence_encoder: nn.Module | None = None,
        *,
        input_dim: int | None = None,
        sym_dim: int = -1,
        n_classes: int = 18,
        fusion_type: str = "concat",
        pool: str = "mean",
    ) -> None:
        super().__init__()
        assert fusion_type in {"concat", "gated"}, "fusion_type must be 'concat' or 'gated'"
        assert pool in {"mean", "cls"}, "pool must be 'mean' or 'cls'"
        assert sequence_encoder is not None or input_dim is not None, "Provide encoder or input_dim"

        self.encoder = sequence_encoder
        self.pool = pool
        self.fusion_type = fusion_type

        if sequence_encoder is not None:
            self.seq_dim = sequence_encoder.get_output_dim()
            self.sym_dim_init = sym_dim  # user‑provided value (may be -1)
            self._sym_dim = max(1, sym_dim)

            if fusion_type == "gated":
                self.gate_fc = nn.Sequential(
                    nn.Linear(self._sym_dim, self.seq_dim),
                    nn.Sigmoid(),
                )
                fused_dim = self.seq_dim
            else:
                self.gate_fc = None
                fused_dim = self.seq_dim + self._sym_dim
        else:
            self.seq_dim = int(input_dim)
            self.sym_dim_init = 0
            self._sym_dim = 0
            self.gate_fc = None
            fused_dim = self.seq_dim

        self.classifier = _MLPHead(fused_dim, n_classes)

    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _do_pool(x: torch.Tensor, strategy: str = "mean") -> torch.Tensor:
        """Pool (B, T, D) → (B, D) if necessary."""
        if x.ndim == 3:
            if strategy == "mean":
                return x.mean(dim=1)
            if strategy == "cls":
                return x[:, 0]  # assume first token is [CLS]
        return x  # already (B, D)

    # ──────────────────────────────────────────────────────────
    def _maybe_refit_lazy_layers(self, sym_dim: int, device, dtype) -> None:
        """If sym_dim was -1 at init, rebuild layers with correct shape."""
        if sym_dim == self._sym_dim:  # nothing to do
            return

        # update stored dim
        self._sym_dim = sym_dim

        if self.fusion_type == "gated":
            self.gate_fc = nn.Sequential(
                nn.Linear(sym_dim, self.seq_dim),
                nn.Sigmoid(),
            ).to(device=device, dtype=dtype)
            # classifier unchanged (seq_dim)
        else:  # concat → classifier input dim changes
            fused_dim = self.seq_dim + sym_dim
            self.classifier = _MLPHead(fused_dim, self.classifier.net[-1].out_features).to(
                device=device,
                dtype=dtype,
            )

    # ──────────────────────────────────────────────────────────
    def forward(
        self,
        seq_tensor: torch.Tensor | None = None,
        sym_tensor: Optional[torch.Tensor] = None,
        *,
        sensor_latents: Optional[list[torch.Tensor]] = None,
        return_latent: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        seq_tensor : Tensor | None
            (B, T, C) or (B, C) depending on encoder. Required if ``sensor_latents``
            is ``None``.
        sym_tensor : Tensor | None
            (B, sym_dim) symbolic vector.  If ``None`` → zeros.
        sensor_latents : list[Tensor] | None
            Pre-fused latent vectors from multiple sensors. Overrides ``seq_tensor``
            if provided.
        return_latent : bool, default False
            Whether to also return fused latent representation.
        """

        if sensor_latents is not None:
            fused_vec = torch.cat(sensor_latents, dim=1)
            logits = self.classifier(fused_vec)
            return (fused_vec, logits) if return_latent else logits

        assert self.encoder is not None, "FusionNet initialized without encoder must use sensor_latents"

        # ---------------- encode sequence --------------------
        latent = self.encoder(seq_tensor, return_logits=False)  # (B, T, D) *or* (B, D)
        latent = self._do_pool(latent, self.pool)  # ensure (B, D)

        # ---------------- symbolic handling ------------------
        if sym_tensor is None:
            sym_tensor = torch.zeros(latent.size(0), 0, device=latent.device, dtype=latent.dtype)

        if sym_tensor.ndim == 1:  # (sym_dim,) → (1, sym_dim)
            sym_tensor = sym_tensor.unsqueeze(0)

        # Lazy resize if sym_dim was unknown at init
        if self.sym_dim_init == -1 and sym_tensor.size(1) != self._sym_dim:
            self._maybe_refit_lazy_layers(sym_tensor.size(1), latent.device, latent.dtype)

        # defensive shape checks
        assert sym_tensor.shape[0] == latent.shape[0], "Batch size mismatch between seq and symbolic"

        # ---------------- fusion -----------------------------
        if self.fusion_type == "concat":
            fused = torch.cat([latent, sym_tensor.float()], dim=1)
        else:  # gated
            gate = self.gate_fc(sym_tensor.float())  # (B, seq_dim)
            fused = latent * gate

        logits = self.classifier(fused)
        return (fused, logits) if return_latent else logits

    # ----------------------------------------------------------------
    def get_output_dim(self) -> int:  # noqa: D401 – simple access
        """Return vector dim fed into ``classifier`` (after fusion)."""
        if self.encoder is None:
            return self.seq_dim
        if self.fusion_type == "concat":
            return self.seq_dim + self._sym_dim
        return self.seq_dim


# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # quick sanity
    from models.transformer_encoder import TransformerEncoder

    B, T, F, S = 3, 200, 332, 16
    seq = torch.randn(B, T, F)
    sym = torch.randn(B, S)

    enc = TransformerEncoder(in_channels=F, n_classes=18, latent_dim=128)
    net = FusionNet(enc, sym_dim=S, n_classes=18, fusion_type="gated", pool="mean")

    fused, log = net(seq, sym, return_latent=True)
    print("fused", fused.shape, "logits", log.shape)
