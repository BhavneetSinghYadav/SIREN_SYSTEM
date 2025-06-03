import numpy as np

from src.features.thermo_extractor import ThermoExtractor
from src.features.tof_extractor import ToFExtractor, TOTAL_PIX
from src.features.rhythmicity import RhythmicityExtractor


def test_thermo_extractor_mean():
    data = np.full((200, 5), 30.0, dtype=np.float32)
    seq = np.concatenate([np.zeros((200, 7)), data], axis=1)
    ext = ThermoExtractor(start_idx=7)
    vec = ext.extract(seq)
    assert vec.shape == (5,)
    assert np.allclose(vec, 30.0)


def test_tof_extractor_shape_and_values():
    const_val = 100
    grid = np.full((200, TOTAL_PIX), const_val, dtype=np.int16)
    grid[0, 0] = -1  # inject missing pixel
    seq = np.concatenate([np.zeros((200, 12)), grid], axis=1)
    ext = ToFExtractor(start_idx=12)
    vec = ext.extract(seq)
    assert vec.shape == (4,)
    expected_mean = const_val / 255.0
    assert np.isclose(vec[0], expected_mean, atol=1e-6)
    assert vec[2] < 1.0  # valid ratio less than 1 due to missing pixel


def test_rhythmicity_extractor_frequency():
    fs = 50
    t = np.arange(0, 4, 1 / fs)
    sig = np.sin(2 * np.pi * 2.0 * t)
    seq = np.tile(sig[:, None], (1, 7)).astype(np.float32)
    ext = RhythmicityExtractor(sample_rate=fs)
    vec = ext.extract(seq)
    assert vec.shape == (3,)
    assert 0.0 < vec[0] < fs / 2

