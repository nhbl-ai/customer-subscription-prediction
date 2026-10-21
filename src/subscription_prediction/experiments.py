"""Milestone 3: train-only tuning, validation selection, separately requested test."""

import argparse
import hashlib
import json
import logging
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from .data import load_data, read_config, split_data
from .diagnostics import group_errors, reliability_table, save_plots, temporal_errors
from .evaluation import (
    calibration_metrics, economic_metrics, evaluate, select_economic_threshold, select_threshold,
)
from .modeling import build_pipeline, candidate_models
from .tuning import temporal_folds, tune_pipeline

LOGGER = logging.getLogger(__name__)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
    )


def validate_experiment_config(config):
    """Reject ambiguous setups before fitting or writing outputs."""
    if config["split_strategy"] != "temporal":
        raise ValueError("Milestone 3 requires split_strategy=temporal.")
    for key, minimum in (("cv_splits", 2), ("cv_gap", 0), ("permutation_repeats", 1),
                         ("calibration_bins", 2), ("time_blocks", 1)):
        value = config.get(key)
        if type(value) is not int or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}.")
    if "duration" not in config["drop_columns"]:
        raise ValueError("duration must be excluded before pre-call experiments.")
    if not config.get("feature_sets") or not all(
        isinstance(columns, list) for columns in config["feature_sets"].values()
    ):
        raise ValueError("feature_sets must map names to lists of extra columns to drop.")
    if set(config.get("parameter_grids", {})) != set(candidate_models(config)):
        raise ValueError("parameter_grids must configure dummy, logistic_regression and random_forest.")
    scenarios = config.get("economic_scenarios", [])
    names = [scenario["name"] for scenario in scenarios]
    if len(names) != len(set(names)) or config.get("primary_economic_scenario") not in names:
        raise ValueError("Economic scenario names must be unique and include the primary scenario.")
    for scenario in scenarios:
        select_economic_threshold(
            [0, 1], [0.1, 0.9], scenario["call_cost"],
            scenario["net_value_per_subscription"], config["contact_fraction"],
        )
    if config["artifact_dir"] == config["report_dir"]:
        raise ValueError("Use separate artifact and report directories.")


def run_experiments(config_path):
    """Choose and freeze a model using train/validation; do not score test here."""
    config = read_config(config_path)
    validate_experiment_config(config)
    X, y = load_data(config)
    splits = split_data(X, y, config)
    X_train, y_train = splits["train"]
    X_validation, y_validation = splits["validation"]
    folds, fold_table = temporal_folds(X_train, y_train, config)
    report_dir = config["report_dir"]
    artifact_dir = config["artifact_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fold_table.to_csv(report_dir / "cv_folds.csv", index=False)
    candidates, comparisons, cv_tables = {}, [], []
    best_params = {}
    feature_columns = {}
    for variant, dropped in config["feature_sets"].items():
        missing = set(dropped) - set(X_train.columns)
        if missing:
            raise ValueError(f"Unknown columns in {variant}: {sorted(missing)}")
        train_features = X_train.drop(columns=dropped)
        if not len(train_features.columns):
            raise ValueError(f"Feature set {variant} removes every column.")
        feature_columns[variant] = train_features.columns.tolist()
        validation_features = X_validation[train_features.columns]
        for name, estimator in candidate_models(config).items():
            key = f"{variant}/{name}"
            LOGGER.info("Tuning %s with %s temporal folds", key, len(folds))
            pipeline, params, table = tune_pipeline(
                train_features, y_train, estimator, config["parameter_grids"][name], folds,
            )
            scores = pipeline.predict_proba(validation_features)[:, 1]
            metrics = evaluate(
                y_validation, scores, select_threshold(y_validation, scores), config["contact_fraction"],
            )
            candidates[key] = {
                "pipeline": pipeline, "scores": scores,
                "columns": train_features.columns.tolist(), "metrics": metrics,
            }
            best_params[key] = params
            comparisons.append({"candidate": key, "variant": variant, "model": name, "stage": "tuned", **metrics})
            cv_tables.append(table.assign(candidate=key, variant=variant, model=name))
    cv_results = pd.concat(cv_tables, ignore_index=True)
    cv_results.to_csv(report_dir / "cv_results.csv", index=False)
    # Refit the original baseline on the first configured feature set, for a same-split reference.
    reference_variant = next(iter(config["feature_sets"]))
    reference_columns = feature_columns[reference_variant]
    for name, estimator in candidate_models(config).items():
        pipeline = build_pipeline(X_train[reference_columns], estimator)
        pipeline.fit(X_train[reference_columns], y_train)
        scores = pipeline.predict_proba(X_validation[reference_columns])[:, 1]
        metrics = evaluate(y_validation, scores, select_threshold(y_validation, scores), config["contact_fraction"])
        comparisons.append({
            "candidate": f"baseline/{name}", "variant": reference_variant, "model": name,
            "stage": "baseline_reference", **metrics,
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(report_dir / "validation_comparison.csv", index=False)
    selected = max(candidates, key=lambda key: candidates[key]["metrics"]["average_precision"])
    chosen = candidates[selected]
    columns = chosen["columns"]
    raw = chosen["pipeline"]
    LOGGER.info("Selected %s by raw validation AP; fitting temporal sigmoid calibration", selected)
    calibrated = CalibratedClassifierCV(
        estimator=clone(raw), method="sigmoid", cv=folds, ensemble=True, n_jobs=1,
    )
    calibrated.fit(X_train[columns], y_train)
    score_options = {
        "raw": chosen["scores"],
        "temporal_sigmoid": calibrated.predict_proba(X_validation[columns])[:, 1],
    }
    calibration = {
        name: {**calibration_metrics(y_validation, scores),
               "average_precision": float(average_precision_score(y_validation, scores))}
        for name, scores in score_options.items()
    }
    representation = min(calibration, key=lambda name: calibration[name]["brier_score"])
    model = raw if representation == "raw" else calibrated
    scores = score_options[representation]
    pd.DataFrame(calibration).T.to_csv(report_dir / "calibration_metrics.csv", index_label="representation")
    calibration_tables = {
        name: reliability_table(y_validation, values, config["calibration_bins"])
        for name, values in score_options.items()
    }
    pd.concat([
        table.assign(representation=name) for name, table in calibration_tables.items()
    ], ignore_index=True).to_csv(report_dir / "calibration_bins.csv", index=False)
    economics = []
    for scenario in config["economic_scenarios"]:
        choice = select_economic_threshold(
            y_validation, scores, scenario["call_cost"],
            scenario["net_value_per_subscription"], config["contact_fraction"],
        )
        economics.append({**scenario, **choice})
    economic_table = pd.DataFrame(economics)
    economic_table.to_csv(report_dir / "economic_scenarios.csv", index=False)
    primary = next(row for row in economics if row["name"] == config["primary_economic_scenario"])
    threshold = primary["threshold"]
    f1_threshold = select_threshold(y_validation, scores)
    threshold_comparison = []
    for rule, value in (("primary_economic_scenario", threshold), ("maximum_validation_F1", f1_threshold)):
        threshold_comparison.append({
            "rule": rule,
            **evaluate(y_validation, scores, value, config["contact_fraction"]),
            **economic_metrics(y_validation, scores, value, primary["call_cost"], primary["net_value_per_subscription"]),
        })
    pd.DataFrame(threshold_comparison).to_csv(report_dir / "threshold_comparison.csv", index=False)
    LOGGER.info("Permutation importance on validation, %s repeats", config["permutation_repeats"])
    permuted = permutation_importance(
        model, X_validation[columns], y_validation, scoring="average_precision",
        n_repeats=config["permutation_repeats"], random_state=config["random_state"], n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": columns, "importance_mean": permuted.importances_mean,
        "importance_std": permuted.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(report_dir / "permutation_importance.csv", index=False)
    groups = group_errors(X_validation, y_validation, scores, threshold)
    groups.to_csv(report_dir / "group_errors.csv", index=False)
    temporal = temporal_errors(
        X_validation, y_validation, scores, threshold, config["time_blocks"], config["contact_fraction"],
    )
    temporal.to_csv(report_dir / "temporal_validation.csv", index=False)
    prediction = (scores >= threshold).astype(int)
    pd.DataFrame({
        "source_row": X_validation.index, "actual": y_validation.to_numpy(),
        "subscription_score": scores, "predicted": prediction,
        "error_type": np.select(
            [(y_validation.to_numpy() == 1) & (prediction == 0),
             (y_validation.to_numpy() == 0) & (prediction == 1)], ["FN", "FP"], default="correct",
        ),
    }).to_csv(report_dir / "validation_predictions.csv", index=False)
    plot_scores = {key: value["scores"] for key, value in candidates.items()
                   if not key.endswith("/dummy")}
    plot_scores[f"selected/{representation}"] = scores
    save_plots(report_dir, y_validation, plot_scores, calibration_tables, importance, temporal)
    bundle = {
        "pipeline": model, "threshold": threshold, "model_name": selected,
        "feature_columns": columns,
        "numeric_columns": X_train[columns].select_dtypes(include="number").columns.tolist(),
        "positive_label": config["positive_label"], "negative_label": config["negative_label"],
        "sklearn_version": sklearn.__version__, "score_representation": representation,
        "economic_scenario": primary,
    }
    model_path = artifact_dir / "model.joblib"
    joblib.dump(bundle, model_path)
    report = {
        "selected_model": selected, "score_representation": representation,
        "selection_metric": "raw_validation_average_precision",
        "calibration_selection_metric": "validation_brier_score",
        "threshold_rule": "maximum validation utility proxy within contact budget",
        "primary_economic_scenario": primary,
        "threshold": threshold, "f1_reference_threshold": f1_threshold,
        "validation": evaluate(y_validation, scores, threshold, config["contact_fraction"]),
        "calibration": calibration, "best_params": best_params,
        "feature_sets": feature_columns,
        "split_sizes": {name: len(pair[0]) for name, pair in splits.items()},
        "train_positive_rate": float(y_train.mean()),
        "cv_splits": config["cv_splits"], "cv_gap": config["cv_gap"],
        "test_evaluated": False,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
        "limitations": [
            "Test was previously viewed in baseline/EDA; it is not an untouched independent holdout.",
            "CSV row order represents chronology, not equal-duration periods; customer IDs are absent.",
            "Feature availability remains conditional; ablation does not verify release-time availability.",
            "Calibration and model selection reuse validation; these scores are exploratory.",
            "Temporal sigmoid is an ensemble of fold models, so differences include ensembling.",
            "Utility is a historical-label proxy, not causal incremental profit; scenario units must match.",
            "The validation budget constraint is not guaranteed by a fixed threshold on future data.",
            "Calibration on historical data does not establish valid probabilities under distribution shift.",
        ],
    }
    write_json(report_dir / "metrics.json", report)
    # Freeze every decision and provenance before the separate test command can run.
    write_json(report_dir / "selection.json", {
        "selected_model": selected, "score_representation": representation,
        "threshold": threshold, "primary_economic_scenario": primary,
        "data_sha256": file_sha256(config["data_path"]),
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(model_path),
        "validation_report_sha256": file_sha256(report_dir / "metrics.json"),
    })
    LOGGER.info("Frozen %s (%s), threshold=%.6f; test not scored", selected, representation, threshold)
    return report


def evaluate_frozen_test(config_path):
    """Read the saved selection; never refit, retune or select using test."""
    config = read_config(config_path)
    validate_experiment_config(config)
    report_dir = config["report_dir"]
    selection_path = report_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Run validation experiments first to freeze selection.json.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_path = config["artifact_dir"] / "model.joblib"
    expected = {
        "data_sha256": file_sha256(config["data_path"]),
        "config_sha256": file_sha256(config_path),
        "model_sha256": file_sha256(model_path),
        "validation_report_sha256": file_sha256(report_dir / "metrics.json"),
    }
    if any(selection[key] != value for key, value in expected.items()):
        raise ValueError("Data, configuration, model or validation report changed after selection was frozen.")
    selection_hash = file_sha256(selection_path)
    destination = report_dir / "test_metrics.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("selection_sha256") == selection_hash:
            LOGGER.info("This frozen selection already has a test report; returning it without rescoring.")
            return existing
    X, y = load_data(config)
    X_test, y_test = split_data(X, y, config)["test"]
    bundle = joblib.load(model_path)
    scores = bundle["pipeline"].predict_proba(X_test[bundle["feature_columns"]])[:, 1]
    scenario = selection["primary_economic_scenario"]
    threshold = selection["threshold"]
    result = {
        "selection_sha256": selection_hash,
        "selected_model": selection["selected_model"],
        "score_representation": selection["score_representation"],
        "test": evaluate(y_test, scores, threshold, config["contact_fraction"]),
        "calibration": calibration_metrics(y_test, scores),
        "economic_scenario": scenario,
        "economics": economic_metrics(
            y_test, scores, threshold, scenario["call_cost"], scenario["net_value_per_subscription"],
        ),
        "note": "Previously viewed test; fixed decisions from validation. No test-based reselection.",
    }
    write_json(destination, result)
    pd.DataFrame({
        "source_row": X_test.index, "actual": y_test.to_numpy(),
        "subscription_score": scores, "predicted": (scores >= threshold).astype(int),
    }).to_csv(report_dir / "test_predictions.csv", index=False)
    LOGGER.info("Scored saved model on test: AP=%.4f", result["test"]["average_precision"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone3.json")
    parser.add_argument("--evaluate-test", action="store_true", help="Score only the already frozen model on test.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.evaluate_test:
            evaluate_frozen_test(args.config)
        else:
            run_experiments(args.config)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
