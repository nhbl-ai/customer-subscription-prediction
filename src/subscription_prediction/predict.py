"""Score a CSV with the saved training pipeline and rank customers."""

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(model_path, input_path, output_path, separator=";", top_k=None):
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    # Load only artifacts you trust: joblib uses Python pickle internally.
    bundle = joblib.load(model_path)
    frame = pd.read_csv(input_path, sep=separator)
    if frame.empty:
        raise ValueError("Input CSV has no rows.")
    columns = bundle["feature_columns"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    features = frame[columns].copy()
    for column in bundle["numeric_columns"]:
        features[column] = pd.to_numeric(features[column], errors="raise")
    scores = bundle["pipeline"].predict_proba(features)[:, 1]
    result = frame.copy()
    result.insert(0, "source_row", frame.index)
    result["subscription_score"] = scores
    result["predicted_subscription"] = [
        bundle["positive_label"] if score >= bundle["threshold"] else bundle["negative_label"]
        for score in scores
    ]
    result = result.sort_values("subscription_score", ascending=False, kind="stable")
    result.insert(0, "priority_rank", range(1, len(result) + 1))
    if top_k is not None:
        result = result.head(top_k)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="artifacts/model.joblib")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="reports/customer_priorities.csv")
    parser.add_argument("--separator", default=";")
    parser.add_argument("--top-k", type=int)
    args = parser.parse_args()
    try:
        result = predict(args.model, args.input, args.output, args.separator, args.top_k)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Saved {len(result)} ranked customers to {args.output}")


if __name__ == "__main__":
    main()
