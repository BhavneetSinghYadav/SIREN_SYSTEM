"""
src/evaluation/metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Reusable metric helpers for the SIREN project.

• classification_report  – accuracy + macro/micro F1 + per-class F1
• confusion_df           – tidy pandas DataFrame confusion matrix
• plot_confusion         – optional heat-map visual
• RunningScore           – incremental batch accumulator

Author: Bhavya — 2025
"""

from __future__ import annotations
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)

try:
    import matplotlib.pyplot as plt
    MATPLOT_AVAILABLE = True
except ModuleNotFoundError:
    MATPLOT_AVAILABLE = False


# ──────────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────────
def classification_report(
    y_true: List[Any] | np.ndarray,
    y_pred: List[Any] | np.ndarray,
    labels: List[Any] | None = None,
) -> Dict[str, Any]:
    """Return accuracy, macro/micro F1 and per-class F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc       = accuracy_score(y_true, y_pred)
    f1_macro  = f1_score(y_true, y_pred, labels=labels,
                         average="macro", zero_division=0)
    f1_micro  = f1_score(y_true, y_pred, labels=labels,
                         average="micro", zero_division=0)
    f1_vector = f1_score(y_true, y_pred, labels=labels,
                         average=None, zero_division=0)

    return {
        "accuracy":     float(acc),
        "macro_f1":     float(f1_macro),
        "micro_f1":     float(f1_micro),
        "per_class_f1": {int(lbl): float(v) for lbl, v in zip(labels, f1_vector)},
    }


# ──────────────────────────────────────────────────────────────
# Confusion-matrix helpers
# ──────────────────────────────────────────────────────────────
def confusion_df(
    y_true: List[Any] | np.ndarray,
    y_pred: List[Any] | np.ndarray,
    labels: List[Any] | None = None,
) -> pd.DataFrame:
    """Return confusion matrix as a pandas DataFrame."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def plot_confusion(
    y_true: List[Any] | np.ndarray,
    y_pred: List[Any] | np.ndarray,
    labels: List[Any] | None = None,
    normalize: bool = False,
    cmap: str = "Blues",
    figsize: tuple[int, int] = (6, 5),
):
    """Render a heat-map confusion matrix (needs matplotlib)."""
    if not MATPLOT_AVAILABLE:
        print("[metrics] matplotlib not installed – skipping confusion plot.")
        return

    df = confusion_df(y_true, y_pred, labels)
    cm = df.to_numpy(dtype=np.float32)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar(fraction=0.046)
    tick_marks = np.arange(len(df))
    plt.xticks(tick_marks, df.columns, rotation=45, ha="right", fontsize=8)
    plt.yticks(tick_marks, df.index, fontsize=8)

    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=7)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────
# Incremental accumulator
# ──────────────────────────────────────────────────────────────
class RunningScore:
    """
    Incrementally accumulate predictions and targets,
    then compute the same report as `classification_report`.

    Parameters
    ----------
    num_classes : int | None
        If provided, forces the label set to {0 … num_classes-1}.
        This ensures F1 vectors have fixed length even when some
        classes are absent in a given epoch.
    """

    def __init__(self, num_classes: int | None = None):
        self.num_classes = num_classes
        self._y_true: List[int] = []
        self._y_pred: List[int] = []

    # ----------------------------------------------------------
    def update(self,
               preds: np.ndarray | List[int],
               targets: np.ndarray | List[int]):
        self._y_pred.extend(np.asarray(preds).tolist())
        self._y_true.extend(np.asarray(targets).tolist())

    def reset(self):
        self._y_true.clear()
        self._y_pred.clear()

    def report(self) -> Dict[str, Any]:
        labels = list(range(self.num_classes)) if self.num_classes else None
        return classification_report(self._y_true, self._y_pred, labels)


# ──────────────────────────────────────────────────────────────
# Smoke-test
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    y_t = [0, 1, 1, 2, 2, 2]
    y_p = [0, 1, 0, 2, 1, 2]
    print(classification_report(y_t, y_p))
    plot_confusion(y_t, y_p)
