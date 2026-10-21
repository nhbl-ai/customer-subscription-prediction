"""Exercise the public training/inference workflow without downloading UCI data."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from subscription_prediction.predict import predict
from subscription_prediction.train import train


ROOT = Path(__file__).resolve().parents[1]


class PublicWorkflowTests(unittest.TestCase):
    def test_training_artifact_and_ranked_csv_with_unseen_category(self):
        runtime = ROOT / "reports" / "test_runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            config = json.loads((ROOT / "configs/default.json").read_text(encoding="utf-8"))
            config.update(data_path="sample.csv", artifact_dir="artifacts", report_dir="reports",
                          random_forest_estimators=3, random_forest_max_depth=2)
            config_path = root / "configs/default.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            sample = pd.DataFrame({
                "age": np.tile([20, 30, 40, 50], 60),
                "job": np.tile(["admin.", "technician"], 120),
                # A post-call feature that must never enter the artifact.
                "duration": np.tile([0, 1000], 120),
                "y": np.tile(["no", "yes"], 120),
            })
            sample.loc[168:, "age"] += 100
            sample.to_csv(root / "sample.csv", sep=";", index=False)
            report = train(config_path)
            bundle = joblib.load(root / "artifacts/model.joblib")
            self.assertNotIn("duration", bundle["feature_columns"])
            self.assertEqual(report["split_sizes"], {"train": 168, "validation": 36, "test": 36})
            # The scaler is fitted on train alone, including a shifted validation age.
            scaler = bundle["pipeline"].named_steps["preprocess"].named_transformers_["numeric"].named_steps["scale"]
            self.assertAlmostEqual(scaler.mean_[0], sample.age.iloc[:168].mean())
            customers = pd.DataFrame({"customer_id": ["demo-A", "demo-B", "demo-C"],
                                      "age": [25, 35, 35], "job": ["unseen-job"] * 3})
            input_path = root / "customers.csv"
            output_path = root / "nested/priorities.csv"
            customers.to_csv(input_path, sep=";", index=False)
            ranked = predict(root / "artifacts/model.joblib", input_path, output_path, top_k=2)
            self.assertEqual(len(ranked), 2)
            self.assertEqual(ranked.priority_rank.tolist(), [1, 2])
            self.assertTrue(ranked.subscription_score.is_monotonic_decreasing)
            self.assertTrue(ranked.subscription_score.between(0, 1).all())
            for row in ranked.itertuples():
                self.assertEqual(row.customer_id, customers.iloc[row.source_row].customer_id)
                expected = "yes" if row.subscription_score >= bundle["threshold"] else "no"
                self.assertEqual(row.predicted_subscription, expected)
            self.assertEqual(len(pd.read_csv(output_path)), 2)
            pd.DataFrame({"job": ["admin."]}).to_csv(input_path, sep=";", index=False)
            with self.assertRaisesRegex(ValueError, "Missing feature columns"):
                predict(root / "artifacts/model.joblib", input_path, output_path)
            customers.iloc[:0].to_csv(input_path, sep=";", index=False)
            with self.assertRaisesRegex(ValueError, "no rows"):
                predict(root / "artifacts/model.joblib", input_path, output_path)
            with self.assertRaisesRegex(ValueError, "positive integer"):
                predict(root / "artifacts/model.joblib", input_path, output_path, top_k=0)


if __name__ == "__main__":
    unittest.main()
