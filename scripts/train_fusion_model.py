"""Train SIREN multimodal fusion model

Supports:
    • Encoders   : CNN / Transformer
    • Fusion     : concat / gated
    • Sensors    : IMU (+Thermo / +ToF) – configurable
    • Pooling    : mean | cls for Transformer outputs (B, T, D)

Usage example
-------------
python scripts/train_fusion_model.py \
    --data-dir data/clean \
    --model-type transformer \
    --fusion gated \
    --epochs 5 --batch 32
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data import loader as dl  # ALL_COLUMNS
from src.data.loader import get_dataset
from src.evaluation.metrics import RunningScore
from src.features.feature_bank import SymbolicFeatureBank
from src.models.cnn_encoder import CNNEncoder
from src.models.transformer_encoder import TransformerEncoder
from src.models.fusion_net import FusionNet
from src.models.symbolic_overlay import SymbolicOverlay

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # fallback – tqdm not installed
    tqdm = lambda x, **k: x  # type: ignore

# ────────────────────────────────────────────────────────────────────────────────
SEED = 777


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Collate fn ─ stacks torch sequences and builds symbolic numpy → torch tensor
# ----------------------------------------------------------------------------

def collate_fn_multi(
    batch,
    banks: dict,
    *,
    use_imu: bool = True,
    use_thermo: bool = True,
    use_tof: bool = True,
) -> Tuple[dict, dict, torch.Tensor]:
    """Return per-sensor tensors and symbolic vectors."""
    seqs, labels, _ = zip(*batch)
    seqs_t = torch.stack(seqs)  # (B, T, F)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    idx = 0
    seq_parts = {}
    if use_imu:
        seq_parts["imu"] = seqs_t[..., idx : idx + len(dl.IMU_COLUMNS)]
        idx += len(dl.IMU_COLUMNS)
    if use_thermo:
        seq_parts["thermo"] = seqs_t[..., idx : idx + len(dl.THERMO_COLUMNS)]
        idx += len(dl.THERMO_COLUMNS)
    if use_tof:
        seq_parts["tof"] = seqs_t[..., idx : idx + len(dl.TOF_COLUMNS)]

    sym_parts = {}
    for name, bank in banks.items():
        sym_np = np.vstack([bank.extract_all(s.cpu().numpy()) for s in seqs_t])
        sym_parts[name] = torch.from_numpy(sym_np).float()

    return seq_parts, sym_parts, labels_t


# ----------------------------------------------------------------------------
# Build encoder backbone
# ----------------------------------------------------------------------------

def build_encoder(model_type: str, in_ch: int, num_classes: int) -> nn.Module:
    if model_type == "cnn":
        return CNNEncoder(in_channels=in_ch, n_classes=num_classes, latent_dim=128)
    if model_type == "transformer":
        return TransformerEncoder(
            in_channels=in_ch,
            n_classes=num_classes,
            latent_dim=128,
            n_layers=4,
            n_heads=4,
        )
    raise ValueError(f"[train] unknown model_type '{model_type}'")


# ----------------------------------------------------------------------------
# Main train routine
# ----------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    data_root = Path(args.data_dir)
    with open(data_root / "label_map.json") as jf:
        num_classes = len(json.load(jf))
    print(f"[INFO] num_classes = {num_classes}")

    # ---------------- dataset ----------------
    ds_full = get_dataset(
        data_root,
        split="train",
        use_torch=True,
        use_thermo=not args.no_thermo,
        use_tof=not args.no_tof,
        use_imu=True,
    )
    train_len = int(0.9 * len(ds_full))
    ds_train, ds_val = random_split(ds_full, [train_len, len(ds_full) - train_len])

    # Use dataset's active sensor columns so symbolic extraction aligns
    # with the actual feature ordering produced by `SequenceDataset`.
    frame_cols = ds_full.sensor_cols

    banks: dict = {
        "imu": SymbolicFeatureBank(
            frame_cols,
            use_rhythm=True,
            use_thermo=False,
            use_tof=False,
            use_tremor=True,
            use_posture=True,
        )
    }
    if not args.no_thermo:
        banks["thermo"] = SymbolicFeatureBank(
            frame_cols,
            use_rhythm=False,
            use_thermo=True,
            use_tof=False,
            use_tremor=False,
            use_posture=False,
        )
    if not args.no_tof:
        banks["tof"] = SymbolicFeatureBank(
            frame_cols,
            use_rhythm=False,
            use_thermo=False,
            use_tof=True,
            use_tremor=False,
            use_posture=False,
        )

    common_dl = dict(
        batch_size=args.batch,
        collate_fn=lambda b: collate_fn_multi(
            b,
            banks,
            use_imu=True,
            use_thermo=not args.no_thermo,
            use_tof=not args.no_tof,
        ),
        num_workers=0,
        pin_memory=True,
    )
    train_loader = DataLoader(ds_train, shuffle=True, **common_dl)
    val_loader = DataLoader(ds_val, shuffle=False, **common_dl)

    # ---------------- model -------------------
    enc_imu = build_encoder(args.model_type, len(dl.IMU_COLUMNS), num_classes)
    overlay_imu = SymbolicOverlay(enc_imu.get_output_dim(), banks["imu"].dim(), args.fusion)

    enc_thermo = overlay_thermo = None
    if not args.no_thermo:
        enc_thermo = build_encoder(args.model_type, len(dl.THERMO_COLUMNS), num_classes)
        overlay_thermo = SymbolicOverlay(enc_thermo.get_output_dim(), banks["thermo"].dim(), args.fusion)

    enc_tof = overlay_tof = None
    if not args.no_tof:
        enc_tof = build_encoder(args.model_type, len(dl.TOF_COLUMNS), num_classes)
        overlay_tof = SymbolicOverlay(enc_tof.get_output_dim(), banks["tof"].dim(), args.fusion)

    fusion_dim = overlay_imu.get_output_dim()
    if overlay_thermo is not None:
        fusion_dim += overlay_thermo.get_output_dim()
    if overlay_tof is not None:
        fusion_dim += overlay_tof.get_output_dim()

    model = FusionNet(
        None,
        input_dim=fusion_dim,
        n_classes=num_classes,
    ).to(device)
    encoders = {
        "imu": enc_imu.to(device),
    }
    overlays = {
        "imu": overlay_imu.to(device),
    }
    if enc_thermo is not None:
        encoders["thermo"] = enc_thermo.to(device)
        overlays["thermo"] = overlay_thermo.to(device)
    if enc_tof is not None:
        encoders["tof"] = enc_tof.to(device)
        overlays["tof"] = overlay_tof.to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    # ---------------- loop --------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for b, (seq_parts, sym_parts, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:02d}")):
            y = y.to(device)
            fused_list = []

            lat = encoders["imu"](seq_parts["imu"].to(device))
            fused_list.append(overlays["imu"](lat, sym_parts["imu"].to(device)))

            if not args.no_thermo:
                lat = encoders["thermo"](seq_parts["thermo"].to(device))
                fused_list.append(overlays["thermo"](lat, sym_parts["thermo"].to(device)))

            if not args.no_tof:
                lat = encoders["tof"](seq_parts["tof"].to(device))
                fused_list.append(overlays["tof"](lat, sym_parts["tof"].to(device)))

            if epoch == 1 and b == 0:
                shapes = [f.shape for f in fused_list]
                print("[DEBUG] fused shapes", shapes, "labels", y.shape)

            optim.zero_grad()
            logits = model(sensor_latents=fused_list)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * y.size(0)
        train_loss = loss_sum / train_len

        # ---- validation ----
        model.eval()
        rs = RunningScore(num_classes)
        with torch.no_grad():
            for seq_parts, sym_parts, y in val_loader:
                fused_list = []
                lat = encoders["imu"](seq_parts["imu"].to(device))
                fused_list.append(overlays["imu"](lat, sym_parts["imu"].to(device)))

                if not args.no_thermo:
                    lat = encoders["thermo"](seq_parts["thermo"].to(device))
                    fused_list.append(overlays["thermo"](lat, sym_parts["thermo"].to(device)))

                if not args.no_tof:
                    lat = encoders["tof"](seq_parts["tof"].to(device))
                    fused_list.append(overlays["tof"](lat, sym_parts["tof"].to(device)))

                preds = model(sensor_latents=fused_list).argmax(1)
                rs.update(preds.cpu().numpy(), y.numpy())
        rep = rs.report()
        f1, acc = rep["macro_f1"], rep["accuracy"]
        print(
            f"Epoch {epoch:02d} | loss={train_loss:.4f} | val_macro_f1={f1:.4f} | acc={acc:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "best_fusion.pt")
            print("  ↳ New best model saved.")

    print(f"[DONE] best val macro-F1 = {best_f1:.4f}")


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SIREN multimodal fusion model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-type", choices=["cnn", "transformer"], default="transformer")
    parser.add_argument("--fusion", choices=["concat", "gated"], default="gated")
    parser.add_argument("--pool", choices=["mean", "cls"], default="mean", help="Pooling for Transformer outputs")

    # sensor toggles (on by default)
    parser.add_argument("--no-thermo", action="store_true", help="Disable Thermopile stream")
    parser.add_argument("--no-tof", action="store_true", help="Disable ToF stream")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)

    args = parser.parse_args()
    train(args)
