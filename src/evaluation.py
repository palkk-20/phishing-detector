"""Shared evaluation helpers for both training scripts."""
from time import perf_counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Metrics with class 1 = phishing as the positive class."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "confusion_matrix": cm.tolist(),  # rows=true, cols=predicted, [0,1]
        "test_samples": int(len(y_true)),
    }
    if y_proba is not None:
        out["brier_score"] = float(brier_score_loss(y_true, y_proba))
    return out


def mean_latency_ms(fn, items, n=200, seed=42) -> float:
    """Average wall-clock time (ms) of fn(item) over up to n random items."""
    items = list(items)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n, len(items)), replace=False)
    start = perf_counter()
    for i in idx:
        fn(items[i])
    return (perf_counter() - start) / len(idx) * 1000.0
