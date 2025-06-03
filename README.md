# SIREN: Symbolic Inference of Repetitive Embodied Narratives

This repository demonstrates a multimodal gesture recognition pipeline using the
**FusionNet** architecture. The network merges sequence embeddings from a CNN or
Transformer encoder with symbolic features extracted on the fly.

## 1. Preprocessing

The raw dataset (as provided on Kaggle) contains `train.csv` and `test.csv`.
Run the preprocessing script to clean, pad and z-score each sequence:

```bash
python scripts/preprocess_data.py \
    --data_dir /path/to/raw_kaggle_data \
    --out_dir  /path/to/clean_data
```

This writes `train_processed.csv`, `test_processed.csv` and
`label_map.json` to the specified output directory.

## 2. Training

Training uses the new FusionNet architecture and computes symbolic features
on the fly.

```bash
python scripts/train_fusion_model.py \
    --data-dir /path/to/clean_data \
    --model-type transformer \
    --fusion gated \
    --epochs 40 \
    --batch 64
```

The best model weights are saved to `artifacts/best_fusion.pt`.

## 3. Inference

Generate predictions on the test split with `run_inference.py`:

```bash
python scripts/run_inference.py \
    --data-dir /path/to/clean_data \
    --model-path artifacts/best_fusion.pt \
    --out-csv predictions.csv
```

The resulting CSV contains `sequence_id` and `gesture_id` columns
ready for submission.
