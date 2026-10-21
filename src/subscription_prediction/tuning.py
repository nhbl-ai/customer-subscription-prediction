"""Select hyperparameters using only expanding, ordered folds inside train."""

import json

import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from .modeling import build_pipeline


def temporal_folds(X, y, config):
    """Validate class support and export boundaries before any model is fitted."""
    if config["split_strategy"] != "temporal":
        raise ValueError("Milestone 3 requires temporal splitting and an ordered CSV.")
    splitter = TimeSeriesSplit(n_splits=config["cv_splits"], gap=config["cv_gap"])
    folds = list(splitter.split(X))
    rows = []
    for number, (train, validation) in enumerate(folds, 1):
        if train[-1] >= validation[0]:
            raise ValueError("A temporal fold must train strictly before validation.")
        for name, positions in (("train", train), ("validation", validation)):
            labels = y.iloc[positions]
            if labels.nunique() != 2:
                raise ValueError(f"Fold {number} {name} must contain both labels.")
            rows.append({
                "fold": number, "subset": name, "rows": len(positions),
                "first_source_row": int(X.index[positions[0]]),
                "last_source_row": int(X.index[positions[-1]]),
                "yes_count": int(labels.sum()), "positive_rate": float(labels.mean()),
            })
    return folds, pd.DataFrame(rows)


def tune_pipeline(X, y, estimator, grid, folds):
    """Fit preprocessing anew in each fold; refit the winner on all train rows."""
    search = GridSearchCV(
        build_pipeline(X, estimator), param_grid=grid, cv=folds,
        scoring="average_precision", refit=True, n_jobs=1, error_score="raise",
    )
    search.fit(X, y)
    results = search.cv_results_
    rows = []
    for candidate, params in enumerate(results["params"]):
        row = {
            "params": json.dumps(params, sort_keys=True),
            "mean_cv_average_precision": float(results["mean_test_score"][candidate]),
            "std_cv_average_precision": float(results["std_test_score"][candidate]),
            "rank": int(results["rank_test_score"][candidate]),
        }
        for fold in range(len(folds)):
            row[f"fold_{fold + 1}_average_precision"] = float(
                results[f"split{fold}_test_score"][candidate]
            )
        rows.append(row)
    return search.best_estimator_, search.best_params_, pd.DataFrame(rows)
