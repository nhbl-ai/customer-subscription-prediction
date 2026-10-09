"""Classification metrics plus a fixed contact-budget comparison."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def select_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def evaluate(y, scores, threshold, contact_fraction):
    labels = np.asarray(y)
    scores = np.asarray(scores)
    prediction = (scores >= threshold).astype(int)
    count = max(1, int(np.ceil(len(labels) * contact_fraction)))
    # Stable sorting makes ties deterministic. Ties have no meaningful ranking.
    chosen = np.argsort(-scores, kind="stable")[:count]
    top_precision = float(labels[chosen].mean())
    base_rate = float(labels.mean())
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "f1": float(f1_score(labels, prediction, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, prediction, labels=[0, 1]).tolist(),
        "contact_count": count,
        "precision_at_contact_budget": top_precision,
        "positive_rate": base_rate,
        "expected_random_precision": base_rate,
        "lift_at_contact_budget": top_precision / base_rate,
    }
