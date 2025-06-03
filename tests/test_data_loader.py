import numpy as np
import pandas as pd
from pathlib import Path

from src.data import preprocessing as pp
from src.data.loader import SequenceDataset, ALL_COLUMNS, SEQ_LEN


def test_zscore_normalisation():
    rng = np.random.RandomState(0)
    x = rng.rand(50, 3).astype(np.float32)
    z = pp._zscore(x)
    assert z.shape == x.shape
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(z.std(axis=0), 1, atol=1e-6)


def test_pad_or_crop():
    rng = np.random.RandomState(1)
    short = rng.rand(150, 4).astype(np.float32)
    long = rng.rand(250, 4).astype(np.float32)
    assert pp._pad_or_crop(short).shape == (SEQ_LEN, 4)
    assert pp._pad_or_crop(long).shape == (SEQ_LEN, 4)


def test_sequence_dataset_shapes(tmp_path):
    rows = []
    for sid in ["A", "B"]:
        for i in range(SEQ_LEN):
            frame = np.random.randn(len(ALL_COLUMNS)).astype(np.float32)
            rows.append([sid, 0, *frame])
    df = pd.DataFrame(rows, columns=["sequence_id", "gesture_id", *ALL_COLUMNS])
    csv = tmp_path / "train_processed.csv"
    df.to_csv(csv, index=False)

    ds = SequenceDataset(csv, mode="train", use_torch=False, use_imu=True, use_thermo=False, use_tof=False)
    assert len(ds) == 2
    x, label, sid = ds[0]
    assert x.shape == (SEQ_LEN, 7)
    assert isinstance(label, int)
    assert sid in {"A", "B"}

