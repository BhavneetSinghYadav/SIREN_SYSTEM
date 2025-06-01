#!/usr/bin/env python
# scripts/train_fusion_model.py
# ---------------------------------------------------------------
# Train SIREN fusion model on pre-processed data
# ---------------------------------------------------------------

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.loader          import get_dataset
from src.features.rhythmicity import RhythmicityExtractor
from src.models.cnn_encoder   import CNNEncoder
from src.models.fusion_net    import FusionNet
from src.evaluation.metrics   import RunningScore

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x, **k: x  # fallback

# ---------------- utils ---------------------------------------
def set_seed(seed: int = 777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch, sym_extractor: RhythmicityExtractor) -> Tuple:
    seqs, labels, _ = zip(*batch)              # list of tensors / ints
    seqs   = torch.stack(seqs)                 # (B,T,F)
    labels = torch.as_tensor(labels, dtype=torch.long)

    sym_np = np.vstack([sym_extractor.extract(s.cpu().numpy()) for s in seqs])
    sym_t  = torch.as_tensor(sym_np, dtype=torch.float32)
    return seqs, sym_t, labels


# ---------------- main train loop -----------------------------
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)

    data_root = Path(args.data_dir)

    # num_classes from label_map.json
    with open(data_root / "label_map.json") as jf:
        label_map = json.load(jf)
    num_classes = len(label_map)
    print(f"[INFO] num_classes = {num_classes}")

    # dataset
    ds_full = get_dataset(data_root, split="train", use_torch=True)
    train_len = int(0.9 * len(ds_full))
    ds_train, ds_val = random_split(ds_full, [train_len, len(ds_full) - train_len])

    sym_ext = RhythmicityExtractor()

    common_dl_kwargs = dict(
        batch_size=args.batch,
        collate_fn=lambda b: collate_fn(b, sym_ext),
        num_workers=0,                # safer on Kaggle
        pin_memory=True,
    )

    train_loader = DataLoader(ds_train, shuffle=True, **common_dl_kwargs)
    val_loader   = DataLoader(ds_val,   shuffle=False, **common_dl_kwargs)

    # model
    cnn   = CNNEncoder(in_channels=7, n_classes=num_classes, latent_dim=128)
    model = FusionNet(cnn, sym_dim=3, n_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optim     = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = 0.0
    out_dir = Path("artifacts"); out_dir.mkdir(exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        loss_sum = 0.0
        for seqs, syms, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            seqs, syms, y = seqs.to(device), syms.to(device), y.to(device)
            optim.zero_grad()
            logits = model(seqs, syms)
            loss   = criterion(logits, y)
            loss.backward()
            optim.step()
            loss_sum += loss.item() * seqs.size(0)

        train_loss = loss_sum / train_len

        # --- validate ---
        model.eval()
        rs = RunningScore()
        with torch.no_grad():
            for seqs, syms, y in val_loader:
                seqs, syms = seqs.to(device), syms.to(device)
                preds = torch.argmax(model(seqs, syms), 1).cpu().numpy()
                rs.update(preds, y.numpy())

        rep = rs.report()
        f1, acc = rep["macro_f1"], rep["accuracy"]
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"| val_macro_f1={f1:.4f} | acc={acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out_dir / "best_fusion.pt")
            print("  ↳ New best model saved.")

    print(f"[DONE] best val macro-F1 = {best_f1:.4f}")


# -------------------------- CLI ------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train SIREN fusion model")
    p.add_argument("--data-dir", required=True,
                   help="Folder containing train_processed.csv & label_map.json")
    p.add_argument("--epochs",   type=int, default=6)
    p.add_argument("--batch",    type=int, default=64)
    p.add_argument("--lr",       type=float, default=3e-4)
    p.add_argument("--seed",     type=int, default=777)
    args = p.parse_args()
    train(args)
