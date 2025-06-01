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

def collate_fn(batch, bank: SymbolicFeatureBank) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, labels, _ = zip(*batch)  # list[(T, F)]
    seqs_t = torch.stack(seqs)  # (B, T, F)
    labels_t = torch.as_tensor(labels, dtype=torch.long)

    # symbolic extraction (CPU numpy for feature-bank)
    sym_np = np.vstack([bank.extract_all(s.cpu().numpy()) for s in seqs_t])
    sym_t = torch.from_numpy(sym_np).float()
    return seqs_t, sym_t, labels_t


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

    frame_cols = dl.ALL_COLUMNS  # after sensor toggles, loader will subset internally
    sym_bank = SymbolicFeatureBank(frame_cols)
    print(f"[INFO] symbolic dim  = {sym_bank.dim()}")

    common_dl = dict(
        batch_size=args.batch,
        collate_fn=lambda b: collate_fn(b, sym_bank),
        num_workers=0,
        pin_memory=True,
    )
    train_loader = DataLoader(ds_train, shuffle=True, **common_dl)
    val_loader = DataLoader(ds_val, shuffle=False, **common_dl)

    # ---------------- model -------------------
    encoder = build_encoder(args.model_type, len(frame_cols), num_classes)
    model = FusionNet(
        encoder,
        sym_dim=sym_bank.dim(),
        n_classes=num_classes,
        fusion_type=args.fusion,
        pool=args.pool,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    # ---------------- loop --------------------
    for epoch in range(1, args.epochs + 1):
        # ---- train phase ----
        model.train()
        loss_sum = 0.0
        for b, (seqs, syms, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:02d}")):
            seqs, syms, y = seqs.to(device), syms.to(device), y.to(device)

            # one-time shape sanity print
            if epoch == 1 and b == 0:
                print("[DEBUG] seqs", seqs.shape, "syms", syms.shape, "labels", y.shape)

            optim.zero_grad()
            logits = model(seqs, syms)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * seqs.size(0)
        train_loss = loss_sum / train_len

        # ---- validation ----
        model.eval()
        rs = RunningScore(num_classes)
        with torch.no_grad():
            for seqs, syms, y in val_loader:
                preds = model(seqs.to(device), syms.to(device)).argmax(1)
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
