#!/usr/bin/env python
# scripts/train_fusion_model.py
# ---------------------------------------------------------------
# Train SIREN fusion model on processed IMU + Thermo + ToF data
# ---------------------------------------------------------------

import argparse, json, random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.loader  import get_dataset, THERMO_START, TOF_START
from src.features.feature_bank import SymbolicFeatureBank
from src.models.cnn_encoder     import CNNEncoder
from src.models.fusion_net      import FusionNet
from src.evaluation.metrics     import RunningScore

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x, **k: x

# ------------------------------------------------------------------
def set_seed(seed: int = 777):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ------------------------------------------------------------------
def collate_fn(batch, bank: SymbolicFeatureBank) -> Tuple:
    seqs, labels, _ = zip(*batch)          # list[(T,F)]
    seqs   = torch.stack(seqs)             # (B,T,F)
    labels = torch.as_tensor(labels, dtype=torch.long)

    sym_np = np.vstack([bank.extract_all(s.cpu().numpy()) for s in seqs])
    sym_t  = torch.as_tensor(sym_np, dtype=torch.float32)
    return seqs, sym_t, labels

# ------------------------------------------------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    data_root = Path(args.data_dir)

    # num_classes
    with open(data_root / "label_map.json") as jf:
        num_classes = len(json.load(jf))
    print(f"[INFO] num_classes = {num_classes}")

    # dataset ----------------------------------------------------------------
    ds_full = get_dataset(data_root, split="train", use_torch=True)
    train_len = int(0.9 * len(ds_full))
    ds_train, ds_val = random_split(ds_full, [train_len, len(ds_full)-train_len])

    # symbolic feature bank (dim=10)
    sym_bank = SymbolicFeatureBank(THERMO_START, TOF_START)
    print(f"[INFO] symbolic dim = {sym_bank.dim()}")

    common_dl = dict(
        batch_size=args.batch,
        collate_fn=lambda b: collate_fn(b, sym_bank),
        num_workers=0,
        pin_memory=True,
    )
    train_loader = DataLoader(ds_train, shuffle=True,  **common_dl)
    val_loader   = DataLoader(ds_val,   shuffle=False, **common_dl)

    # model ------------------------------------------------------------------
    IN_CHANNELS = 332                     # full multimodal frame dim
    cnn   = CNNEncoder(in_channels=IN_CHANNELS,
                       n_classes=num_classes,
                       latent_dim=128)
    model = FusionNet(cnn,
                      sym_dim=sym_bank.dim(),
                      n_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optim     = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    out_dir = Path("artifacts"); out_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for seqs, syms, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            seqs, syms, y = seqs.to(device), syms.to(device), y.to(device)
            optim.zero_grad()
            logits = model(seqs, syms)
            loss   = criterion(logits, y)
            loss.backward(); optim.step()
            loss_sum += loss.item() * seqs.size(0)

        train_loss = loss_sum / train_len

        # ---------------- validation ----------------
        model.eval(); rs = RunningScore(num_classes)
        with torch.no_grad():
            for seqs, syms, y in val_loader:
                preds = torch.argmax(model(seqs.to(device), syms.to(device)), 1)
                rs.update(preds.cpu().numpy(), y.numpy())

        rep = rs.report(); f1, acc = rep["macro_f1"], rep["accuracy"]
        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} "
              f"| val_macro_f1={f1:.4f} | acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "best_fusion.pt")
            print("  ↳ New best model saved.")

    print(f"[DONE] best val macro-F1 = {best_f1:.4f}")

# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train SIREN multimodal fusion model")
    p.add_argument("--data-dir", required=True,
                   help="Folder with *_processed.csv & label_map.json")
    p.add_argument("--epochs",   type=int, default=40)
    p.add_argument("--batch",    type=int, default=64)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--seed",     type=int, default=777)
    args = p.parse_args()
    train(args)
