"""
src/evaluation/metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Reusable metric helpers for the SIREN project.

• classification_report  – accuracy + macro/micro F1 + per-class F1
• confusion_df           – tidy pandas DataFrame confusion matrix
• plot_confusion         – optional heatmap visual
• RunningScore           – incremental batch accumulator (train/val loops)

Author: Bhavya  — 2025
"""

from __future__ import annotations
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

try:
    import matplotlib.pyplot as plt
    MATPLOT_AVAILABLE = True
except ModuleNotFoundError:
    MATPLOT_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Core report
# --------------------------------------------------------------------------- #
def classification_report(
    y_true: List[Any] | np.ndarray,
    y_pred: List[Any] | np.ndarray,
    labels: List[Any] | None = None,
) -> Dict[str, Any]:
    """
    Returns dict with:
      • accuracy
      • macro_f1
      • micro_f1
      • per_class_f1 {label: f1}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))

    acc  = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro")
    f1_micro = f1_score(y_true, y_pred, labels=labels, average="micro")

    f1_per_class = f1_score(
        y_true, y_pred, labels=labels, average=None,
    )
    per_class = {lbl: float(score) for lbl, score in zip(labels, f1_per_class)}

    return {
        "accuracy":     float(acc),
        "macro_f1":     float(f1_macro),
        "micro_f1":     float(f1_micro),
        "per_class_f1": per_class,
    }


# --------------------------------------------------------------------------- #
# Confusion matrix helpers
# --------------------------------------------------------------------------- #
def confusion_df(
    y_true: List[Any] | np.ndarray,
    y_pred: List[Any] | np.ndarray,
    labels: List[Any] | None = None,
) -> pd.DataFrame:
    """Return confusion matrix as DataFrame (labels as both index & columns)."""
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
    figsize: tuple = (6, 5),
):
    """
    Simple heatmap plot.  If matplotlib absent, prints a notice and returns None.
    """
    if not MATPLOT_AVAILABLE:
        print("[metrics] matplotlib not installed – skipping confusion plot.")
        return

    df = confusion_df(y_true, y_pred, labels)
    cm = df.to_numpy().astype(np.float32)
    if normalize:
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)

    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar(fraction=0.046)
    tick_marks = np.arange(len(df))
    plt.xticks(tick_marks, df.columns, rotation=45, ha="right")
    plt.yticks(tick_marks, df.index)

    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}",
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Incremental accumulator (avoids storing full arrays)
# --------------------------------------------------------------------------- #
class RunningScore:
    """
    Accumulate preds & targets over batches, then compute metrics.

    Usage
    -----
    rs = RunningScore()
    for logits, labels in loop:
        preds = logits.argmax(1).cpu().numpy()
        rs.update(preds, labels.cpu().numpy())
    report = rs.report()
    """
    def __init__(self):
        self._y_true: List[int] = []
        self._y_pred: List[int] = []

    def update(self, preds: np.ndarray | List[int], targets: np.ndarray | List[int]):
        self._y_pred.extend(list(preds))
        self._y_true.extend(list(targets))

    def reset(self):
        self._y_pred.clear()
        self._y_true.clear()

    def report(self, labels: List[Any] | None = None) -> Dict[str, Any]:
        return classification_report(self._y_true, self._y_pred, labels)


# --------------------------------------------------------------------------- #
# Smoke-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    y_t = [0, 1, 1, 2, 2, 2]
    y_p = [0, 1, 0, 2, 1, 2]
    print(classification_report(y_t, y_p))
    plot_confusion(y_t, y_p)
# Evaluation metrics
