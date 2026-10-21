import numpy as np
import pandas as pd


def column_profile(frame):
    """Count literal unknown separately from pandas missing values."""
    return pd.DataFrame({
        "dtype": frame.dtypes.astype(str),
        "unique_count": frame.nunique(dropna=False),
        "missing_count": frame.isna().sum(),
        "unknown_count": frame.eq("unknown").sum(),
    }).assign(
        missing_pct=lambda t: 100 * t.missing_count / len(frame),
        unknown_pct=lambda t: 100 * t.unknown_count / len(frame),
    ).rename_axis("column")


def split_profile(raw, splits):
    rows = []
    for name, (features, labels) in splits.items():
        part = raw.loc[features.index]
        rows.append({
            "split": name, "rows": len(part),
            "first_source_row": int(part.index.min()),
            "last_source_row": int(part.index.max()),
            "yes_count": int(labels.sum()), "yes_pct": 100 * labels.mean(),
            "missing_cells": int(part.isna().sum().sum()),
            "duplicate_full_rows": int(part.duplicated().sum()),
            "pdays_999_pct": 100 * part.pdays.eq(999).mean(),
            "previous_zero_pct": 100 * part.previous.eq(0).mean(),
        })
    return pd.DataFrame(rows).set_index("split")


def category_rates(series, labels):
    """Train-only associations. Counts accompany rates to expose small groups."""
    table = pd.DataFrame({"category": series, "yes": labels}).groupby(
        "category", observed=True, dropna=False
    )["yes"].agg(rows="size", yes_count="sum", yes_rate="mean")
    table["yes_pct"] = 100 * table.pop("yes_rate")
    return table.sort_values("rows", ascending=False)


def ordered_blocks(features, labels, count=10):
    """Contiguous row blocks, not reconstructed calendar dates."""
    rows = []
    for block, positions in enumerate(np.array_split(np.arange(len(features)), count), 1):
        if not len(positions):
            continue
        part = features.iloc[positions]
        rows.append({
            "block": block, "first_source_row": int(part.index.min()),
            "last_source_row": int(part.index.max()), "rows": len(part),
            "yes_pct": 100 * labels.iloc[positions].mean(),
            "age_median": part.age.median(),
            "campaign_median": part.campaign.median(),
            "euribor3m_median": part.euribor3m.median(),
            "nr_employed_median": part["nr.employed"].median(),
            "pdays_999_pct": 100 * part.pdays.eq(999).mean(),
        })
    return pd.DataFrame(rows).set_index("block")


def categorical_shift(splits):
    """Feature-only total variation distance (0..1) relative to train."""
    train = splits["train"][0]
    rows = []
    for column in train.select_dtypes(include=["object", "string"]):
        reference = train[column].value_counts(normalize=True, dropna=False)
        for name, (features, _) in splits.items():
            if name == "train":
                continue
            current = features[column].value_counts(normalize=True, dropna=False)
            categories = reference.index.union(current.index)
            distance = 0.5 * (
                reference.reindex(categories, fill_value=0)
                - current.reindex(categories, fill_value=0)
            ).abs().sum()
            unseen = features[column].isin(current.index.difference(reference.index))
            rows.append({
                "column": column, "split": name,
                "total_variation": distance, "unseen_category_pct": 100 * unseen.mean(),
            })
    return pd.DataFrame(rows).sort_values("total_variation", ascending=False)
