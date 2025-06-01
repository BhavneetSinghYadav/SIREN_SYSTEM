#!/usr/bin/env python
# scripts/train_fusion_model.py
# ---------------------------------------------------------------
# End-to-end training script for the SIREN Fusion model.
#
# • Loads train.csv from --data-dir
# • Builds SequenceDataset  (IMU only)
# • Extracts Rhythmicity features on-the-fly
# • Trains FusionNet (CNN encoder + symbolic vector)
# • Tracks macro-F1 on hold-out validation set
# ---------------------------------------------------------------

import argparse
from pathlib import Path
import random
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import f1_score

# --- SIREN internal imports ------------------------------------
from src.data.loader          import get_dataset
from src.features.rhythmicity import RhythmicityExtractor
from src.models.cnn_encoder   import CNNEncoder
from src.models.fusion_net    import FusionNet
# ---------------------------------------------------------------


# ─────────────────────────────────────────────────────────────── #
def set_seed(seed: int = 777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────── #
def collate_fn(batch, sym_extractor):
    """Custom collate: stack sequences; build symbolic vectors."""
    seqs, labels, _ = zip(*batch)                     # each seq is (T,F)
    seqs   = torch.stack(seqs)                        # (B,T,F)
    labels = torch.tensor(labels, dtype=torch.long)

    # Symbolic features (batch)
    sym_np  = np.vstack([sym_extractor.extract(seq.numpy()) for seq in seqs])
    sym_t   = torch.tensor(sym_np, dtype=torch.float32)
    return seqs, sym_t, labels


# ─────────────────────────────────────────────────────────────── #
def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # 1. Dataset ------------------------------------------------
    ds_full = get_dataset(args.data_dir, split="train", use_torch=True)
    train_sz = int(0.9 * len(ds_full))
    val_sz   = len(ds_full) - train_sz
    ds_train, ds_val = random_split(ds_full, [train_sz, val_sz])

    sym_ext = RhythmicityExtractor()

    train_loader = DataLoader(ds_train, batch_size=args.batch,
                              shuffle=True,  num_workers=2,
                              collate_fn=lambda b: collate_fn(b, sym_ext))
    val_loader   = DataLoader(ds_val,   batch_size=args.batch,
                              shuffle=False, num_workers=2,
                              collate_fn=lambda b: collate_fn(b, sym_ext))

    # 2. Model --------------------------------------------------
    cnn   = CNNEncoder(in_channels=7,
                       n_classes=args.num_classes,
                       latent_dim=128)
    model = FusionNet(cnn,
                      sym_dim=3,
                      n_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_path = 0.0, Path("artifacts")
    best_path.mkdir(exist_ok=True, parents=True)

    # 3. Train --------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for seqs, syms, labels in train_loader:
            seqs, syms, labels = seqs.to(device), syms.to(device), labels.to(device)

            opt.zero_grad()
            logits = model(seqs, syms)
            loss   = criterion(logits, labels)
            loss.backward()
            opt.step()
            run_loss += loss.item() * seqs.size(0)

        train_loss = run_loss / len(ds_train)

        # 4. Validation ---------------------------------------
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for seqs, syms, labels in val_loader:
                seqs, syms = seqs.to(device), syms.to(device)
                logits = model(seqs, syms)
                y_hat  = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(y_hat)
                gts.extend(labels.numpy())

        f1 = f1_score(gts, preds, average="macro")
        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | val_f1={f1:.4f}")

        # 5. Checkpoint --------------------------------------
        if f1 > best_f1:
            best_f1 = f1
            save_fp = best_path / "best_fusion.pt"
            torch.save(model.state_dict(), save_fp)
            print(f"  ↳ New best F1!  Model saved to {save_fp}")

    print(f"[DONE] best val macro-F1 = {best_f1:.4f}")


# ─────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SIREN Fusion model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path containing train.csv")
    parser.add_argument("--epochs",   type=int, default=6)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--seed",     type=int, default=777)
    args = parser.parse_args()

    train(args)
# Train the full model
