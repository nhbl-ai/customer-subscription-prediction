"""Train candidates; choose model/threshold on validation; evaluate test once."""

import argparse
import json
import logging
import platform

import joblib
import pandas as pd
import sklearn

from .data import load_data, read_config, split_data
from .evaluation import evaluate, select_threshold
from .modeling import build_pipeline, candidate_models

LOGGER = logging.getLogger(__name__)


def train(config_path):
    config = read_config(config_path)
    X, y = load_data(config)
    splits = split_data(X, y, config)
    X_train, y_train = splits["train"]
    X_validation, y_validation = splits["validation"]
    results = {}
    fitted = {}
    for name, estimator in candidate_models(config).items():
        LOGGER.info("Training %s on %s rows", name, len(X_train))
        pipeline = build_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        scores = pipeline.predict_proba(X_validation)[:, 1]
        threshold = select_threshold(y_validation, scores)
        results[name] = evaluate(y_validation, scores, threshold, config["contact_fraction"])
        fitted[name] = pipeline
    selected = max(results, key=lambda name: results[name]["average_precision"])
    pipeline = fitted[selected]
    threshold = results[selected]["threshold"]
    X_test, y_test = splits["test"]
    test_scores = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_scores, threshold, config["contact_fraction"])
    for key in ("artifact_dir", "report_dir"):
        config[key].mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipeline, "threshold": threshold, "model_name": selected,
        "feature_columns": X_train.columns.tolist(),
        "numeric_columns": X_train.select_dtypes(include="number").columns.tolist(),
        "positive_label": config["positive_label"], "negative_label": config["negative_label"],
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, config["artifact_dir"] / "model.joblib")
    report = {
        "selected_model": selected,
        "selection_metric": "validation_average_precision",
        "threshold_rule": "maximum validation F1",
        "split_strategy": config["split_strategy"],
        "split_sizes": {name: len(pair[0]) for name, pair in splits.items()},
        "validation": results, "test": test_metrics,
        "config": {key: str(value) if hasattr(value, "__fspath__") else value for key, value in config.items()},
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__},
    }
    (config["report_dir"] / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(results).T.to_csv(config["report_dir"] / "validation_comparison.csv", index_label="model")
    pd.DataFrame({
        "source_row": X_test.index, "actual": y_test.to_numpy(),
        "subscription_score": test_scores, "predicted": (test_scores >= threshold).astype(int),
    }).to_csv(config["report_dir"] / "test_predictions.csv", index=False)
    LOGGER.info("Selected %s | test F1=%.3f | test AP=%.3f", selected, test_metrics["f1"], test_metrics["average_precision"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        train(args.config)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
