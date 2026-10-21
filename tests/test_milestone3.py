"""Behavioral checks for chronological isolation, threshold ties and frozen test scoring."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from subscription_prediction.data import load_data, read_config, split_data
from subscription_prediction.diagnostics import error_summary, reliability_table
from subscription_prediction.evaluation import economic_metrics, select_economic_threshold
from subscription_prediction.experiments import evaluate_frozen_test, run_experiments
from subscription_prediction.tuning import temporal_folds, tune_pipeline
from tools.export_portfolio import export


ROOT = Path(__file__).resolve().parents[1]


class EconomicThresholdTests(unittest.TestCase):
    def test_matches_exhaustive_thresholds_and_includes_no_calls(self):
        random = np.random.default_rng(42)
        for _ in range(20):
            labels = random.integers(0, 2, 30)
            scores = random.integers(0, 5, 30) / 4
            choice = select_economic_threshold(labels, scores, 2, 7, 0.4)
            thresholds = np.r_[np.nextafter(scores.max(), np.inf), np.unique(scores)]
            possible = [economic_metrics(labels, scores, value, 2, 7) for value in thresholds]
            possible = [row for row in possible if row["calls"] <= 12]
            optimum = max(possible, key=lambda row: (row["utility_proxy"], -row["calls"]))
            self.assertEqual(choice["utility_proxy"], optimum["utility_proxy"])
            self.assertEqual(choice["calls"], optimum["calls"])

    def test_cannot_split_a_tied_score_to_fit_budget(self):
        choice = select_economic_threshold([1, 1, 0, 0], [0.8, 0.8, 0.1, 0.1], 1, 10, 0.25)
        self.assertEqual(choice["calls"], 0)
        self.assertGreater(choice["threshold"], 0.8)

    def test_unprofitable_contacts_choose_no_calls(self):
        choice = select_economic_threshold([0, 1], [0.9, 0.1], 20, 10, 1)
        self.assertEqual(choice["calls"], 0)
        self.assertEqual(choice["utility_proxy"], 0)

    def test_invalid_economic_inputs_raise(self):
        for cost, value, budget in ((-1, 10, 0.2), (1, 0, 0.2), (1, 10, 0), (np.nan, 10, 0.2)):
            with self.assertRaises(ValueError):
                select_economic_threshold([0, 1], [0.1, 0.9], cost, value, budget)


class TemporalAndDiagnosticsTests(unittest.TestCase):
    def test_expanding_folds_are_ordered_and_respect_gap(self):
        features = pd.DataFrame({"age": np.arange(80)}, index=np.arange(100, 180))
        labels = pd.Series(np.tile([0, 1], 40), index=features.index)
        folds, table = temporal_folds(features, labels, {"split_strategy": "temporal", "cv_splits": 3, "cv_gap": 2})
        self.assertEqual(len(table), 6)
        for train, validation in folds:
            self.assertEqual(validation[0] - train[-1] - 1, 2)
            self.assertLess(features.index[train[-1]], features.index[validation[0]])
        self.assertTrue(set(folds[0][0]).issubset(folds[1][0]))

    def test_single_class_fold_fails_before_training(self):
        features = pd.DataFrame({"age": np.arange(20)})
        labels = pd.Series([0] * 10 + [1] * 10)
        with self.assertRaisesRegex(ValueError, "both labels"):
            temporal_folds(features, labels, {"split_strategy": "temporal", "cv_splits": 2, "cv_gap": 0})

    def test_one_class_groups_have_undefined_rates(self):
        row = error_summary([0, 0], [0.1, 0.9], 0.5)
        self.assertIsNone(row["recall"])
        self.assertIsNone(row["average_precision"])
        self.assertIsNone(row["roc_auc"])
        self.assertEqual(row["fp"], 1)

    def test_calibration_bins_cover_boundaries(self):
        table = reliability_table([0, 0, 1, 1], [0, 0.1, 0.9, 1], bins=10)
        self.assertEqual(table.rows.sum(), 4)
        self.assertEqual(table.iloc[-1]["rows"], 2)


class ExperimentIntegrationTests(unittest.TestCase):
    def test_train_validation_only_then_test_without_refitting_and_hash_guard(self):
        runtime = ROOT / "reports" / "test_runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            config_path = root / "configs" / "experiment.json"
            config = json.loads((ROOT / "configs" / "milestone3.json").read_text(encoding="utf-8"))
            config.update({
                "data_path": "sample.csv", "artifact_dir": "artifacts", "report_dir": "reports",
                "cv_splits": 2, "cv_gap": 1, "random_forest_estimators": 3,
                "feature_sets": {"sample": []}, "permutation_repeats": 1,
                "parameter_grids": {
                    "dummy": {}, "logistic_regression": {"classifier__C": [1]},
                    "random_forest": {"classifier__max_depth": [2]},
                },
            })
            config_path.write_text(json.dumps(config), encoding="utf-8")
            sample = pd.DataFrame({
                "age": np.tile([25, 35, 45, 55], 40),
                "job": np.tile(["a", "b"], 80), "duration": np.arange(160),
                "y": np.tile(["no", "yes"], 80),
            })
            sample.to_csv(root / "sample.csv", sep=";", index=False)
            loaded = read_config(config_path)
            X, y = load_data(loaded)
            splits = split_data(X, y, loaded)
            # Poison test labels: validation experiments must not access them.
            poisoned = {**splits, "test": (splits["test"][0], object())}
            with patch("subscription_prediction.experiments.split_data", return_value=poisoned), \
                 patch("subscription_prediction.experiments.tune_pipeline", wraps=tune_pipeline) as tuned:
                report = run_experiments(config_path)
            for call in tuned.call_args_list:
                self.assertLess(call.args[0].index.max(), splits["validation"][0].index.min())
            self.assertFalse(report["test_evaluated"])
            self.assertNotIn("test", report)
            self.assertFalse((root / "reports" / "test_metrics.json").exists())
            with patch.object(Pipeline, "fit", side_effect=AssertionError("Test must never fit")):
                result = evaluate_frozen_test(config_path)
            self.assertIn("test", result)
            with patch("subscription_prediction.experiments.load_data", side_effect=AssertionError("No repeat scoring")):
                self.assertEqual(evaluate_frozen_test(config_path), result)
            public = root / "public"
            with patch.object(Pipeline, "fit", side_effect=AssertionError("Export must never fit")):
                manifest = export(config_path, public)
            self.assertIn("precision_recall.png", manifest["files"])
            for name, expected_hash in manifest["files"].items():
                self.assertEqual(hashlib.sha256((public / name).read_bytes()).hexdigest(), expected_hash)
            self.assertEqual(manifest["files"]["selection.json"], result["selection_sha256"])
            snapshot = json.loads((public / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(snapshot["config"]["data_path"], "sample.csv")
            self.assertFalse((public / "test_predictions.csv").exists())
            test_report = root / "reports/test_metrics.json"
            mismatched = {**result, "selection_sha256": "different-selection"}
            test_report.write_text(json.dumps(mismatched), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not belong"):
                export(config_path, root / "rejected")
            self.assertFalse((root / "rejected").exists())
            test_report.write_text(json.dumps(result), encoding="utf-8")
            sample.loc[159, "age"] = 80
            sample.to_csv(root / "sample.csv", sep=";", index=False)
            with self.assertRaisesRegex(ValueError, "changed after selection"):
                evaluate_frozen_test(config_path)
            with self.assertRaisesRegex(ValueError, "provenance changed"):
                export(config_path, root / "changed-data")


if __name__ == "__main__":
    unittest.main()
