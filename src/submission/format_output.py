"""Helpers for Kaggle submission formatting."""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

__all__ = ["kaggle_submission_df"]


def kaggle_submission_df(
    seq_ids: Iterable[str],
    gesture_ids: Iterable[int],
    id_to_label: Mapping[int, str] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame in the required submission format."""
    if id_to_label is not None:
        gestures = [id_to_label[int(g)] for g in gesture_ids]
        return pd.DataFrame({"sequence_id": list(seq_ids), "gesture": gestures})
    return pd.DataFrame({"sequence_id": list(seq_ids), "gesture_id": list(gesture_ids)})
