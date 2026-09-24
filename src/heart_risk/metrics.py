"""Metrics for an imbalanced yes/no problem (about 8.6% positives).

Accuracy is reported only to show why it misleads: predicting "No" for everyone scores 91%.
The main ranking metric is PR-AUC (average precision), whose no-skill level is the prevalence.
"""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)


def threshold_max_f1(y, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def threshold_for_recall(y, proba, target=0.80) -> float:
    """Highest threshold that still catches at least `target` of the real cases (a screening setting)."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    ok = np.where(recall[:-1] >= target)[0]
    return float(thresholds[ok[-1]])


def summary(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "pr_auc": round(average_precision_score(y, proba), 4),
        "brier": round(brier_score_loss(y, proba), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "accuracy": round(float((pred == np.asarray(y)).mean()), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
