import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, log_loss,
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


def calibration_metrics(y, scores):
    """Probability losses; neither alone proves calibration or future validity."""
    labels = np.asarray(y)
    scores = np.asarray(scores)
    return {
        "brier_score": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, np.column_stack([1 - scores, scores]), labels=[0, 1])),
        "mean_score": float(scores.mean()),
        "positive_rate": float(labels.mean()),
    }


def economic_metrics(y, scores, threshold, call_cost, net_value_per_subscription):
    """Historical label-based utility proxy, not causal incremental revenue."""
    selected = np.asarray(scores) >= threshold
    subscriptions = int(np.asarray(y)[selected].sum())
    calls = int(selected.sum())
    return {
        "threshold": float(threshold), "calls": calls,
        "true_positive_count": subscriptions,
        "utility_proxy": float(subscriptions * net_value_per_subscription - calls * call_cost),
    }


def select_economic_threshold(y, scores, call_cost, net_value_per_subscription, contact_fraction):
    """Maximize validation utility with a budget, whole-score ties and a no-call option."""
    labels = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or not len(labels) or len(labels) != len(scores):
        raise ValueError("Labels and scores must be nonempty, aligned 1D arrays.")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores.")
    if not (np.isfinite(call_cost) and call_cost >= 0
            and np.isfinite(net_value_per_subscription) and net_value_per_subscription > 0):
        raise ValueError("Call cost must be nonnegative and net subscription value positive.")
    if not 0 < contact_fraction <= 1:
        raise ValueError("contact_fraction must be in (0, 1].")
    order = np.argsort(-scores, kind="stable")
    ranked_scores = scores[order]
    # A threshold cannot split tied scores. Evaluate only the end of each tie block.
    ends = np.r_[np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]), len(scores) - 1]
    budget = int(np.ceil(len(labels) * contact_fraction))
    cumulative_yes = np.cumsum(labels[order])
    best = economic_metrics(
        labels, scores, np.nextafter(scores.max(), np.inf), call_cost, net_value_per_subscription,
    )
    for end in ends:
        calls = int(end + 1)
        if calls > budget:
            break
        positives = int(cumulative_yes[end])
        utility = positives * net_value_per_subscription - calls * call_cost
        # Strict comparison keeps fewer calls when utility ties, including zero profit.
        if utility > best["utility_proxy"]:
            best = {
                "threshold": float(ranked_scores[end]), "calls": calls,
                "true_positive_count": positives, "utility_proxy": float(utility),
            }
    return {**best, "validation_call_budget": budget}
