#!/usr/bin/env python
"""Run inference with a trained FusionNet model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.loader import get_dataset
from src.features.feature_bank import SymbolicFeatureBank
from src.models.cnn_encoder import CNNEncoder
from src.models.transformer_encoder import TransformerEncoder
from src.models.fusion_net import FusionNet


# ---------------------------------------------------------------------------
def collate_fn(batch, bank: SymbolicFeatureBank) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    seqs, _, ids = zip(*batch)
    seqs_t = torch.stack(seqs)
    sym_np = np.vstack([bank.extract_all(s.cpu().numpy()) for s in seqs_t])
    sym_t = torch.from_numpy(sym_np).float()
    return seqs_t, sym_t, list(ids)


# ---------------------------------------------------------------------------
def build_encoder(model_type: str, in_ch: int, num_classes: int) -> torch.nn.Module:
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
    raise ValueError(f"unknown model_type '{model_type}'")


# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_dir)

    with open(data_root / "label_map.json") as jf:
        num_classes = len(json.load(jf))

    ds = get_dataset(
        data_root,
        split="test",
        use_imu=True,
        use_thermo=not args.no_thermo,
        use_tof=not args.no_tof,
    )
    bank = SymbolicFeatureBank(ds.sensor_cols)

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(b, bank),
    )

    encoder = build_encoder(args.model_type, len(ds.sensor_cols), num_classes)
    model = FusionNet(
        encoder,
        sym_dim=bank.dim(),
        n_classes=num_classes,
        fusion_type=args.fusion,
        pool=args.pool,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    preds = []
    seq_ids = []
    with torch.no_grad():
        for seqs, syms, ids in loader:
            out = model(seqs.to(device), syms.to(device))
            pred = out.argmax(1).cpu().numpy()
            preds.extend(pred.tolist())
            seq_ids.extend(ids)

    df = pd.DataFrame({"sequence_id": seq_ids, "gesture_id": preds})
    df.to_csv(args.out_csv, index=False)
    print(f"[inference] wrote predictions → {args.out_csv}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run inference with a trained model")
    p.add_argument("--data-dir", required=True, help="Folder with *_processed.csv")
    p.add_argument("--model-path", required=True, help="Path to saved model.pt")
    p.add_argument("--out-csv", required=True, help="Destination CSV for predictions")
    p.add_argument("--model-type", choices=["cnn", "transformer"], default="transformer")
    p.add_argument("--fusion", choices=["concat", "gated"], default="gated")
    p.add_argument("--pool", choices=["mean", "cls"], default="mean")
    p.add_argument("--no-thermo", action="store_true")
    p.add_argument("--no-tof", action="store_true")
    p.add_argument("--batch", type=int, default=64)
    run(p.parse_args())
