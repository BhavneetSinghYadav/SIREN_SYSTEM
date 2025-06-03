# SIREN: Symbolic Inference of Repetitive Embodied Narratives

This repository implements **SIREN** – a multimodal gesture recognition system
combining learnable sensor encoders and lightweight symbolic features.  The
project targets IMU, Thermopile and ToF streams and follows a two‑stage design:

1. *Sensor–wise encoders* produce latent vectors for each modality.
2. *Symbolic overlays* fuse handcrafted features with the learned latents.
3. *Meta fusion* merges the three streams and feeds both a multiclass head and
   per‑class binary heads.

An optional ensemble decoder further mixes binary and multiclass confidence
scores together with rule based overrides.

## 1. Preprocessing

The raw dataset (as provided on Kaggle) contains `train.csv` and `test.csv`.
Run the preprocessing script to clean each sequence, normalise sensor streams
and optionally segment into sliding windows:

```bash
python scripts/preprocess_data.py \
    --data_dir /path/to/raw_kaggle_data \
    --out_dir  /path/to/clean_data \
    --frames   200               # resample length
    --window   200               # window size (optional)
    --stride   200               # stride between windows
```

The script performs per‑stream z‑scoring, linear resampling to the target frame
count and emits `train_processed.csv`, `test_processed.csv` and
`label_map.json` in the chosen output directory.

## 2. Training

`train_fusion_model.py` trains a single FusionNet that joins one sequence
encoder with its symbolic overlay.  Sensors can be toggled on/off and either a
CNN or Transformer backbone may be selected.

```bash
python scripts/train_fusion_model.py \
    --data-dir /path/to/clean_data \
    --model-type transformer \
    --fusion gated \
    --epochs 40 \
    --batch 64
```

Weights are written to `artifacts/best_fusion.pt`.

`train_meta_fusion.py` builds on top by combining three FusionNet instances
into a `MetaFusionNet` with both multiclass and binary heads:

```bash
python scripts/train_meta_fusion.py \
    --data-dir /path/to/clean_data \
    --model-type transformer \
    --fusion gated \
    --epochs 40
```

The best model is stored as `artifacts/best_meta_fusion.pt`.

## 3. Inference

Generate predictions on the test split with `run_inference.py`.
The script loads a saved FusionNet model and optionally applies simple rule
based overrides on the thermopile stream:

```bash
python scripts/run_inference.py \
    --data-dir /path/to/clean_data \
    --model-path artifacts/best_fusion.pt \
    --out-csv predictions.csv \
    --binary-weight 0.5 \
    --multi-weight 0.5 \
    --rules              # enable rule based mask
```

The resulting CSV contains `sequence_id` and `gesture_id` columns ready for
Kaggle submission.

Meta‑fusion models can be served in a similar way once an inference helper is
added.
