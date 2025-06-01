#!/usr/bin/env python
# scripts/train_fusion_model.py
# ---------------------------------------------------------------
# End-to-end training script for the SIREN Fusion model.
# ---------------------------------------------------------------

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# --- SIREN internal imports ------------------------------------
from src.data.loader            import get_dataset
from src.features.rhythmicity   import RhythmicityExtractor
from src.models.cnn_encoder     import CNNEncoder
from src.models.fusion_net      import FusionNet
from src.evaluation.metrics     import RunningScore, classification_report  # ✱ NEW
# ---------------------------------------------------------------

try:
    from tqdm import tqdm  # progress bar (optional)
except ModuleNotFoundError:                                # ✱ NEW
    tqdm = lambda x, **k: x                                # fallback: identity


# ─────────────────────────────────────────────────────────────── #
def set_seed(seed: int = 777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────── #
def collate_fn(batch, sym_extractor: RhythmicityExtractor) -> Tuple:
    """
    Stacks sequence tensors and symbolic vectors into a mini-batch.
    """
    seqs, labels, _ = zip(*batch)               # each seq is (T,F) torch.Tensor
    seqs   = torch.stack(seqs)                  # (B,T,F)
    labels = torch.as_tensor(labels, dtype=torch.long)

    # Symbolic features – keep on CPU, we'll send to GPU only if needed
    sym_np = np.vstack([sym_extractor.extract(seq.cpu().numpy()) for seq in seqs])
    sym_t  = torch.as_tensor(sym_np, dtype=torch.float32)
    return seqs, sym_t, labels


# ─────────────────────────────────────────────────────────────── #
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # 1. Dataset ------------------------------------------------
    ds_full = get_dataset(args.data_dir, split="train", use_torch=True)
    train_len = int(0.9 * len(ds_full))
    val_len   = len(ds_full) - train_len
    ds_train, ds_val = random_split(ds_full, [train_len, val_len])

    sym_ext = RhythmicityExtractor()

    train_loader = DataLoader(
        ds_train, batch_size=args.batch, shuffle=True, num_workers=2,
        collate_fn=lambda b: collate_fn(b, sym_ext)
    )
    val_loader   = DataLoader(
        ds_val,   batch_size=args.batch, shuffle=False, num_workers=2,
        collate_fn=lambda b: collate_fn(b, sym_ext)
    )

    # 2. Model --------------------------------------------------
    cnn   = CNNEncoder(in_channels=7, n_classes=args.num_classes, latent_dim=128)
    model = FusionNet(cnn, sym_dim=3, n_classes=args.num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_dir = 0.0, Path("artifacts")
    best_dir.mkdir(parents=True, exist_ok=True)

    # 3. Train --------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for seqs, syms, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
            seqs, syms, y = seqs.to(device), syms.to(device), y.to(device)

            opt.zero_grad()
            logits = model(seqs, syms)
            loss   = criterion(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item() * seqs.size(0)

        train_loss = running_loss / train_len

        # ---------- Validation ----------
        model.eval()
        rs = RunningScore()
        with torch.no_grad():
            for seqs, syms, y in val_loader:
                seqs, syms = seqs.to(device), syms.to(device)
                logits = model(seqs, syms)
                preds  = torch.argmax(logits, 1).cpu().numpy()
                rs.update(preds, y.numpy())

        report = rs.report()
        f1  = report["macro_f1"]
        acc = report["accuracy"]
        print(f"Epoch {epoch:02d} | loss={train_loss:.4f} "
              f"| val_macro_f1={f1:.4f} | acc={acc:.4f}")

        # Optional: quick look at worst classes
        if epoch == args.epochs or f1 > best_f1:
            worst = sorted(report["per_class_f1"].items(), key=lambda x: x[1])[:3]
            print("  ↳ lowest per-class F1:", worst)

        # ---------- Checkpoint ----------
        if f1 > best_f1:
            best_f1 = f1
            save_fp = best_dir / "best_fusion.pt"
            torch.save(model.state_dict(), save_fp)
            print(f"  ↳ New best! saved to {save_fp}")

    print(f"[DONE] best val macro-F1 = {best_f1:.4f}")


# ─────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SIREN Fusion model")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing train.csv")
    parser.add_argument("--epochs",   type=int, default=6)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--seed",     type=int, default=777)
    args = parser.parse_args()

    train(args)
