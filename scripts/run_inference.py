<<<<<<< codex/implement-inference-script-with-predictions-and-voting
#!/usr/bin/env python
"""Run inference with a trained FusionNet model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.loader import get_dataset
from src.features.feature_bank import SymbolicFeatureBank
from src.models.cnn_encoder import CNNEncoder
from src.models.transformer_encoder import TransformerEncoder
from src.models.fusion_net import FusionNet
from src.submission.format_output import kaggle_submission_df


# ---------------------------------------------------------------------------
# Helpers from training script
# ---------------------------------------------------------------------------

def build_encoder(
    model_type: str, in_ch: int, num_classes: int
) -> torch.nn.Module:
    if model_type == "cnn":
        return CNNEncoder(
            in_channels=in_ch,
            n_classes=num_classes,
            latent_dim=128,
        )
    if model_type == "transformer":
        return TransformerEncoder(
            in_channels=in_ch,
            n_classes=num_classes,
            latent_dim=128,
            n_layers=4,
            n_heads=4,
        )
    raise ValueError(f"[inference] unknown model_type '{model_type}'")


def collate_fn(batch, bank: SymbolicFeatureBank):
    seqs, _, ids = zip(*batch)
    seqs_t = torch.stack(seqs)
    feats = [bank.extract_all(s.cpu().numpy()) for s in seqs_t]
    sym_np = np.vstack(feats)
    syms_t = torch.from_numpy(sym_np).float()
    return seqs_t, syms_t, list(ids)


def rule_override(
    seq_batch: torch.Tensor, sensor_cols: List[str], thresh: float = -0.3
) -> torch.BoolTensor:
    """Return mask where thermo indicates no body contact."""
    if "thm_1" not in sensor_cols:
        return torch.zeros(seq_batch.size(0), dtype=torch.bool)
    idx = sensor_cols.index("thm_1")
    temps = seq_batch[:, :, idx: idx + 5].mean(dim=(1, 2))
    return temps < thresh


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_dir)

    with open(data_root / "label_map.json") as jf:
        label_map = json.load(jf)
    num_classes = len(label_map)
    inv_label = {v: k for k, v in label_map.items()}

    ds = get_dataset(
        data_root,
        split="test",
        use_torch=True,
        use_thermo=not args.no_thermo,
        use_tof=not args.no_tof,
        use_imu=True,
    )
    bank = SymbolicFeatureBank(ds.sensor_cols)
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, bank),
        num_workers=0,
    )

    encoder = build_encoder(args.model_type, ds.in_channels, num_classes)
    model = FusionNet(
        encoder,
        sym_dim=bank.dim(),
        n_classes=num_classes,
        fusion_type=args.fusion,
        pool=args.pool,
    ).to(device)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    seq_ids_all: List[str] = []
    pred_labels: List[int] = []

    with torch.no_grad():
        for seqs, syms, ids in loader:
            seqs = seqs.to(device)
            syms = syms.to(device)
            logits = model(seqs, syms)
            probs = torch.softmax(logits, dim=1)
            confs, top_cls = probs.max(1)
            prob_none = probs[:, 0] if probs.size(1) > 1 else 1 - confs

            # ----- binary vote -----
            gesture_score = (1 - prob_none) * args.binary_weight
            none_score = prob_none * args.binary_weight

            # ----- rule-based override -----
            if args.rules:
                mask = rule_override(seqs, ds.sensor_cols)
                none_score[mask] = 1.0
                gesture_score[mask] = 0.0

            # ----- combine with multi-class confidence -----
            class_score = confs * args.multi_weight
            final_preds = []
            pairs = zip(gesture_score, none_score, class_score, top_cls)
            for gs, ns, cs, cls_idx in pairs:
                if ns >= gs + cs or cs < args.threshold:
                    final_preds.append(0)
                else:
                    final_preds.append(int(cls_idx.item()))

            seq_ids_all.extend(ids)
            pred_labels.extend(final_preds)

    df = kaggle_submission_df(seq_ids_all, pred_labels, inv_label)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[INFO] wrote {len(df)} predictions to {args.out_csv}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIREN inference")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-path", required=True, help="Path to saved model.pt")
    parser.add_argument("--out-csv", default="submission.csv", help="Destination CSV for predictions")
    parser.add_argument(
        "--model-type",
        choices=["cnn", "transformer"],
        default="transformer",
    )
    parser.add_argument(
        "--fusion",
        choices=["concat", "gated"],
        default="gated",
    )
    parser.add_argument("--pool", choices=["mean", "cls"], default="mean")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--binary-weight",
        type=float,
        default=0.5,
        help="Vote weight for binary detector",
    )
    parser.add_argument(
        "--multi-weight",
        type=float,
        default=0.5,
        help="Vote weight for multi-class classifier",
    )
    parser.add_argument(
        "--rules", action="store_true", help="Enable rule-based override"
    )
    parser.add_argument("--no-thermo", action="store_true")
    parser.add_argument("--no-tof", action="store_true")
    args = parser.parse_args()
    run(args)


