"""Train MetaFusionNet with joint multiclass and binary heads."""

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

from src.data import loader as dl
from src.data.loader import get_dataset
from src.evaluation.metrics import RunningScore
from src.features.feature_bank import SymbolicFeatureBank
from src.models.cnn_encoder import CNNEncoder
from src.models.transformer_encoder import TransformerEncoder
from src.models.fusion_net import FusionNet
from src.models.meta_fusion_net import MetaFusionNet

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover
    tqdm = lambda x, **k: x  # type: ignore

SEED = 777


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Collate
# ------------------------------------------------------------

def collate_fn(batch, bank: SymbolicFeatureBank) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, labels, _ = zip(*batch)
    seqs_t = torch.stack(seqs)  # (B, T, F)
    labels_t = torch.as_tensor(labels, dtype=torch.long)
    sym_np = np.vstack([bank.extract_all(s.cpu().numpy()) for s in seqs_t])
    sym_t = torch.from_numpy(sym_np).float()

    imu = seqs_t[..., : len(dl.IMU_COLUMNS)]
    thm = seqs_t[..., len(dl.IMU_COLUMNS) : len(dl.IMU_COLUMNS) + len(dl.THERMO_COLUMNS)]
    tof = seqs_t[..., len(dl.IMU_COLUMNS) + len(dl.THERMO_COLUMNS) :]
    return imu, thm, tof, sym_t, labels_t


# ------------------------------------------------------------
# Build encoder helper
# ------------------------------------------------------------

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
    raise ValueError(f"unknown model_type {model_type}")


# ------------------------------------------------------------
# Training routine
# ------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    data_root = Path(args.data_dir)
    with open(data_root / "label_map.json") as jf:
        num_classes = len(json.load(jf))
    print(f"[INFO] num_classes = {num_classes}")

    ds_full = get_dataset(
        data_root,
        split="train",
        use_torch=True,
        use_thermo=True,
        use_tof=True,
        use_imu=True,
    )
    train_len = int(0.9 * len(ds_full))
    ds_train, ds_val = random_split(ds_full, [train_len, len(ds_full) - train_len])

    frame_cols = ds_full.sensor_cols
    sym_bank = SymbolicFeatureBank(frame_cols)
    print(f"[INFO] symbolic dim = {sym_bank.dim()}")

    common_dl = dict(
        batch_size=args.batch,
        collate_fn=lambda b: collate_fn(b, sym_bank),
        num_workers=0,
        pin_memory=True,
    )
    train_loader = DataLoader(ds_train, shuffle=True, **common_dl)
    val_loader = DataLoader(ds_val, shuffle=False, **common_dl)

    # models per sensor
    imu_enc = build_encoder(args.model_type, len(dl.IMU_COLUMNS), num_classes)
    thm_enc = build_encoder(args.model_type, len(dl.THERMO_COLUMNS), num_classes)
    tof_enc = build_encoder(args.model_type, len(dl.TOF_COLUMNS), num_classes)

    imu_net = FusionNet(imu_enc, sym_dim=sym_bank.dim(), n_classes=num_classes, fusion_type=args.fusion, pool=args.pool)
    thm_net = FusionNet(thm_enc, sym_dim=sym_bank.dim(), n_classes=num_classes, fusion_type=args.fusion, pool=args.pool)
    tof_net = FusionNet(tof_enc, sym_dim=sym_bank.dim(), n_classes=num_classes, fusion_type=args.fusion, pool=args.pool)

    model = MetaFusionNet(imu_net, thm_net, tof_net, n_classes=num_classes, threshold=args.threshold).to(device)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            imu, thm, tof, sym, y = [t.to(device) for t in batch]

            optim.zero_grad()
            mc_logits, bin_logits = model(imu, thm, tof, sym)
            loss_ce = ce(mc_logits, y)
            y_onehot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()
            loss_bce = bce(bin_logits, y_onehot)
            loss = loss_ce + loss_bce
            loss.backward()
            optim.step()
            loss_sum += loss.item() * y.size(0)
        train_loss = loss_sum / train_len

        model.eval()
        rs = RunningScore(num_classes)
        with torch.no_grad():
            for batch in val_loader:
                imu, thm, tof, sym, y = [t.to(device) for t in batch]
                mc_logits, _ = model(imu, thm, tof, sym)
                preds = mc_logits.argmax(1)
                rs.update(preds.cpu().numpy(), y.cpu().numpy())
        rep = rs.report()
        f1, acc = rep["macro_f1"], rep["accuracy"]
        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | val_macro_f1={f1:.4f} | acc={acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "best_meta_fusion.pt")
            print("  ↳ New best model saved.")

    print(f"[DONE] best val macro-F1 = {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MetaFusionNet")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-type", choices=["cnn", "transformer"], default="transformer")
    parser.add_argument("--fusion", choices=["concat", "gated"], default="gated")
    parser.add_argument("--pool", choices=["mean", "cls"], default="mean")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary head threshold")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args)
