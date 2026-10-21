"""Validation-only plots, calibration bins and descriptive error analysis."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


def reliability_table(y, scores, bins=10):
    """Fixed-width probability bins with counts; omit empty bins."""
    labels = np.asarray(y)
    scores = np.asarray(scores)
    membership = np.minimum((scores * bins).astype(int), bins - 1)
    rows = []
    for number in range(bins):
        chosen = membership == number
        if not chosen.any():
            continue
        rows.append({
            "bin": number + 1, "lower": number / bins, "upper": (number + 1) / bins,
            "rows": int(chosen.sum()), "mean_score": float(scores[chosen].mean()),
            "positive_rate": float(labels[chosen].mean()),
        })
    return pd.DataFrame(rows)


def error_summary(y, scores, threshold):
    """Support one-class groups and report undefined rates as None."""
    labels = np.asarray(y)
    scores = np.asarray(scores)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "rows": len(labels), "yes_count": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
        "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
        "roc_auc": float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None,
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "mean_score": float(scores.mean()),
    }


def group_errors(features, y, scores, threshold):
    """Analyze demographic and campaign groups without fitting new models."""
    groups = {
        column: features[column].astype("string").fillna("<missing>")
        for column in ("job", "education", "contact", "month", "poutcome")
        if column in features
    }
    if "age" in features:
        groups["age_band"] = pd.cut(
            features["age"], [-np.inf, 29, 39, 49, 59, np.inf],
            labels=["<=29", "30-39", "40-49", "50-59", ">=60"],
        ).astype("string").fillna("<missing>")
    if "previous" in features:
        groups["previous_contact"] = features["previous"].gt(0).map({True: "yes", False: "no"})
    labels = np.asarray(y)
    scores = np.asarray(scores)
    rows = []
    for column, values in groups.items():
        for category in sorted(values.unique()):
            chosen = values.eq(category).to_numpy(dtype=bool)
            summary = error_summary(labels[chosen], scores[chosen], threshold)
            rows.append({
                "group_column": column, "group": category,
                "small_group_under_100": summary["rows"] < 100, **summary,
            })
    return pd.DataFrame(rows)


def temporal_errors(features, y, scores, threshold, count, contact_fraction):
    """Contiguous source-row blocks; they are not equal calendar intervals."""
    labels = np.asarray(y)
    scores = np.asarray(scores)
    rows = []
    for number, positions in enumerate(np.array_split(np.arange(len(features)), count), 1):
        if not len(positions):
            continue
        summary = error_summary(labels[positions], scores[positions], threshold)
        budget = max(1, int(np.ceil(len(positions) * contact_fraction)))
        top = positions[np.argsort(-scores[positions], kind="stable")[:budget]]
        top_precision = float(labels[top].mean())
        rows.append({
            "block": number, "first_source_row": int(features.index[positions[0]]),
            "last_source_row": int(features.index[positions[-1]]), **summary,
            "precision_at_contact_budget": top_precision,
            "lift_at_contact_budget": top_precision / summary["positive_rate"] if summary["positive_rate"] else None,
        })
    return pd.DataFrame(rows)


def save_plots(report_dir, y, comparison_scores, calibration_tables, importance, temporal):
    """Write static figures with the noninteractive backend."""
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(report_dir / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, scores in comparison_scores.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        ax.step(recall, precision, where="post", label=f"{name} (AP={ap:.3f})", alpha=0.85)
    ax.axhline(np.asarray(y).mean(), color="black", linestyle="--", label="Random baseline")
    ax.set(xlabel="Recall", ylabel="Precision", title="Validation precision-recall", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "precision_recall.png", dpi=150)
    plt.close(fig)

    fig, (ax, counts) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    for name, table in calibration_tables.items():
        ax.plot(table.mean_score, table.positive_rate, marker="o", label=name)
        counts.plot(table.mean_score, table.rows, marker="o", label=name)
    ax.set(xlabel="Mean predicted score", ylabel="Observed positive rate", title="Validation reliability", xlim=(0, 1), ylim=(0, 1))
    counts.set(xlabel="Mean predicted score", ylabel="Rows per bin", title="Bin support (fixed-width bins)", yscale="log")
    ax.legend()
    counts.legend()
    fig.tight_layout()
    fig.savefig(report_dir / "calibration.png", dpi=150)
    plt.close(fig)

    table = importance.sort_values("importance_mean").tail(15)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(table.feature, table.importance_mean, xerr=table.importance_std, capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Decrease in validation average precision (mean +/- SD)", title="Permutation importance of selected model")
    fig.tight_layout()
    fig.savefig(report_dir / "permutation_importance.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for column in ("positive_rate", "precision", "recall", "mean_score"):
        axes[0].plot(temporal.block, temporal[column], marker="o", label=column)
    axes[1].plot(temporal.block, temporal.lift_at_contact_budget, marker="o")
    axes[1].axhline(1, color="black", linestyle="--")
    axes[0].set(xlabel="Contiguous validation row block", ylabel="Rate / mean score", title="Validation changes across row order", ylim=(0, 1))
    axes[1].set(xlabel="Contiguous validation row block", ylabel="Lift at contact budget", title="Ranking vs each block's base rate")
    for ax in axes:
        ax.set_xticks(temporal.block)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(report_dir / "temporal_validation.png", dpi=150)
    plt.close(fig)
