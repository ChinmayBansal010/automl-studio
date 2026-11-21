import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main


class TrainingTests(unittest.TestCase):
    def run_training_case(self, frame, task, target, model_name="Decision Tree"):
        request = main.Train(
            dataset_id="dataset-test",
            target=target,
            task=task,
            mode="manual",
            model_name=model_name,
            selected_features=["feature"],
            test_size=0.2,
            random_state=42,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(main, "ROOT", Path(temp_dir)), patch.object(main, "get_df", return_value=frame):
                result = main.run_training(request, "training-test-user")
        main.models.clear()
        return result

    def test_classification_returns_string_labels_and_labeled_confusion_matrix(self):
        frame = pd.DataFrame({
            "feature": np.arange(80, dtype=float),
            "target": ["class-a" if i % 2 == 0 else "class-b" for i in range(80)],
        })
        result = self.run_training_case(frame, "classification", "target")
        self.assertEqual(result["task"], "classification")
        self.assertEqual(len(result["results"][0]["actual"]), result["test_rows"])
        self.assertTrue(set(result["results"][0]["actual"]).issubset({"class-a", "class-b"}))
        self.assertTrue(set(result["results"][0]["predictions"]).issubset({"class-a", "class-b"}))
        matrix = result["best_evaluation"]["confusion_matrix"]
        self.assertEqual(matrix["labels"], ["class-a", "class-b"])
        self.assertEqual(len(matrix["matrix"]), 2)
        self.assertTrue(all(len(row) == 2 for row in matrix["matrix"]))

    def test_regression_returns_finite_numeric_actual_and_predictions(self):
        x = np.arange(80, dtype=float)
        frame = pd.DataFrame({"feature": x, "target": 2.5 * x + 3.0})
        result = self.run_training_case(frame, "regression", "target")
        self.assertEqual(result["task"], "regression")
        self.assertTrue(all(np.isfinite(result["results"][0]["actual"])))
        self.assertTrue(all(np.isfinite(result["results"][0]["predictions"])))
        self.assertIn("r2", result["best_evaluation"]["metrics"])
        self.assertIn("mae", result["best_evaluation"]["metrics"])

    def test_candidate_pools_are_task_specific(self):
        classification = main.candidates("classification")
        regression = main.candidates("regression")
        self.assertIn("Logistic Regression", classification)
        self.assertNotIn("Logistic Regression", regression)
        self.assertIn("Ridge Regression", regression)
        self.assertNotIn("Ridge Regression", classification)

    def test_multiple_jobs_can_be_queued_without_parallel_training(self):
        class FakeExecutor:
            def __init__(self):
                self.submitted = []

            def submit(self, function, *args):
                self.submitted.append((function, args))

        request = main.Train(
            dataset_id="dataset-test",
            target="target",
            task="classification",
            mode="manual",
            model_name="Decision Tree",
            selected_features=["feature"],
        )
        fake_executor = FakeExecutor()
        with patch.object(main, "TRAINING_EXECUTOR", fake_executor), patch.dict(main.TRAINING_JOBS, {}, clear=True):
            first = main.train(request, "queue-user")
            second = main.train(request, "queue-user")
            self.assertEqual(first["status"], "queued")
            self.assertEqual(second["status"], "queued")
            self.assertEqual(len(fake_executor.submitted), 2)
            self.assertEqual(main.training_job_status(first["job_id"], "queue-user")["status"], "queued")
            with self.assertRaises(HTTPException) as denied:
                main.training_job_status(first["job_id"], "another-user")
            self.assertEqual(denied.exception.status_code, 404)

    def test_regression_rejects_categorical_target(self):
        frame = pd.DataFrame({
            "feature": np.arange(20, dtype=float),
            "target": ["low" if i % 2 == 0 else "high" for i in range(20)],
        })
        request = main.Train(
            dataset_id="dataset-test",
            target="target",
            task="regression",
            mode="manual",
            model_name="Decision Tree",
            selected_features=["feature"],
        )
        with patch.object(main, "get_df", return_value=frame):
            with self.assertRaises(HTTPException) as error:
                main.run_training(request, "training-test-user")
        self.assertEqual(error.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
