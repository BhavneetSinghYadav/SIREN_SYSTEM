"""MetaFusionNet combines three sensor overlays into a global vector.

This module stacks three ``FusionNet`` instances – typically one for IMU,
Thermopile and ToF streams – and concatenates their latent vectors into a
single global representation. Two prediction heads operate on this vector:

* ``multi_head`` – deep multiclass classifier returning logits for all classes.
* ``binary_heads`` – one ``nn.Linear`` per class for independent binary
  detection. ``threshold`` controls the default probability cut-off.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .fusion_net import FusionNet, _MLPHead


class MetaFusionNet(nn.Module):
    """Fuse three sensor overlays and expose multi/binary heads."""

    def __init__(
        self,
        imu_net: FusionNet,
        thermo_net: FusionNet,
        tof_net: FusionNet,
        *,
        n_classes: int = 18,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.imu_net = imu_net
        self.thermo_net = thermo_net
        self.tof_net = tof_net
        self.threshold = threshold
        self.n_classes = n_classes

        fused_dim = (
            imu_net.get_output_dim()
            + thermo_net.get_output_dim()
            + tof_net.get_output_dim()
        )
        self.multi_head = _MLPHead(fused_dim, n_classes)
        self.binary_heads = nn.ModuleList(
            nn.Linear(fused_dim, 1) for _ in range(n_classes)
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        imu_seq: torch.Tensor,
        thermo_seq: torch.Tensor,
        tof_seq: torch.Tensor,
        sym_tensor: torch.Tensor,
        *,
        return_global: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return multiclass logits and binary logits (B, C)."""
        imu_lat, _ = self.imu_net(imu_seq, sym_tensor, return_latent=True)
        th_lat, _ = self.thermo_net(thermo_seq, sym_tensor, return_latent=True)
        tf_lat, _ = self.tof_net(tof_seq, sym_tensor, return_latent=True)

        global_vec = torch.cat([imu_lat, th_lat, tf_lat], dim=1)
        multi_logits = self.multi_head(global_vec)
        bin_logits = torch.cat([h(global_vec) for h in self.binary_heads], dim=1)

        if return_global:
            return global_vec, multi_logits, bin_logits
        return multi_logits, bin_logits

    # ------------------------------------------------------------------
    def predict_binary(self, bin_logits: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid + threshold to binary logits → (B, C) bool tensor."""
        probs = torch.sigmoid(bin_logits)
        return probs >= self.threshold

    # ------------------------------------------------------------------
    def get_global_dim(self) -> int:
        return self.multi_head.net[0].in_features
