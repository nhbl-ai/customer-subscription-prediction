"""Publish aggregate evidence from a verified frozen experiment, without refitting."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from subscription_prediction.data import read_config


ROOT = Path(__file__).resolve().parents[1]
TABLES = (
    "validation_comparison.csv", "cv_folds.csv", "cv_results.csv",
    "calibration_metrics.csv", "calibration_bins.csv", "economic_scenarios.csv",
    "threshold_comparison.csv", "permutation_importance.csv",
    "group_errors.csv", "temporal_validation.csv",
)
FIGURES = ("precision_recall.png", "calibration.png", "permutation_importance.png", "temporal_validation.png")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(config_path, destination):
    config_path = Path(config_path).resolve()
    destination = Path(destination).resolve()
    config = read_config(config_path)
    source = config["report_dir"]
    selection_path = source / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    test_path = source / "test_metrics.json"
    test = json.loads(test_path.read_text(encoding="utf-8"))
    originals = {
        "data_sha256": config["data_path"], "config_sha256": config_path,
        "model_sha256": config["artifact_dir"] / "model.joblib",
        "validation_report_sha256": source / "metrics.json",
    }
    if any(sha256(path) != selection[key] for key, path in originals.items()):
        raise ValueError("Frozen experiment provenance changed; refusing to publish mismatched results.")
    if test["selection_sha256"] != sha256(selection_path):
        raise ValueError("Test report does not belong to this frozen selection.")
    # Check every input before writing the public snapshot.
    for name in (*TABLES, *FIGURES):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    # Keep all decisions/metrics but replace machine-specific resolved paths.
    metrics["config"] = json.loads(config_path.read_text(encoding="utf-8"))
    destination.mkdir(parents=True, exist_ok=True)
    published = {"metrics.json": metrics, "test_metrics.json": test, "selection.json": selection}
    for name, content in published.items():
        if name == "selection.json":
            # test_metrics.json refers to the hash of these exact original bytes.
            shutil.copyfile(selection_path, destination / name)
            continue
        (destination / name).write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name in (*TABLES, *FIGURES):
        shutil.copyfile(source / name, destination / name)
    manifest = {
        "description": "Aggregate portfolio snapshot; no raw data, row predictions or serialized model.",
        "dataset": "UCI Bank Marketing / bank-additional-full.csv",
        "dataset_doi": "https://doi.org/10.24432/C5K306",
        "dataset_license": "CC BY 4.0",
        "selection_sha256": sha256(selection_path),
        "original_validation_report_sha256": selection["validation_report_sha256"],
        "note": "Public metrics.json uses portable config paths, so its hash differs from the frozen original.",
        "files": {name: sha256(destination / name) for name in (*published, *TABLES, *FIGURES)},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/milestone3.json"))
    parser.add_argument("--output", default=str(ROOT / "docs/results"))
    args = parser.parse_args()
    export(args.config, args.output)
    print(f"Exported verified aggregate evidence to {args.output}")


if __name__ == "__main__":
    main()
