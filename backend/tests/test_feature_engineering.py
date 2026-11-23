import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main


class FeatureEngineeringTests(unittest.TestCase):
    def run_operation(self, frame, payload):
        dataset_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(main, "ROOT", Path(temp_dir)), patch.object(main, "get_df", return_value=frame):
                return main.feature_engineering(dataset_id, payload, "feature-test-user")

    def test_one_hot_encodes_only_the_selected_column(self):
        frame = pd.DataFrame({
            "city": ["Delhi", "Mumbai", "Delhi", "Pune"],
            "segment": ["A", "B", "A", "C"],
            "amount": [10, 20, 30, 40],
        })
        result = self.run_operation(frame, {"operation": "one_hot_encode", "columns": ["city"]})
        self.assertIn("city_Delhi", result["types"]["numeric"] + result["types"]["categorical"])
        self.assertIn("segment", [column["column"] for column in result["columns_info"]])
        self.assertIn("amount", [column["column"] for column in result["columns_info"]])
        self.assertNotIn("city", [column["column"] for column in result["columns_info"]])

    def test_standardize_changes_only_selected_numeric_columns(self):
        frame = pd.DataFrame({"selected": [1.0, 2.0, 3.0], "untouched": [10.0, 20.0, 30.0]})
        dataset_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(main, "ROOT", root), patch.object(main, "get_df", return_value=frame):
                main.feature_engineering(dataset_id, {"operation": "standardize", "columns": ["selected"]}, "feature-test-user")
            saved = pd.read_csv(root / "users" / __import__("hashlib").sha256(b"feature-test-user").hexdigest() / "datasets" / (dataset_id + ".csv"))
        self.assertAlmostEqual(float(saved["selected"].mean()), 0.0, places=6)
        self.assertEqual(saved["untouched"].tolist(), [10.0, 20.0, 30.0])

    def test_one_hot_encoding_all_columns_in_a_single_column_dataset_is_allowed(self):
        frame = pd.DataFrame({"kind": ["x", "y", "x"]})
        result = self.run_operation(frame, {"operation": "one_hot_encode", "columns": ["kind"]})
        names = [column["column"] for column in result["columns_info"]]
        self.assertIn("kind_x", names)
        self.assertIn("kind_y", names)

    def test_numeric_transform_rejects_categorical_selection(self):
        frame = pd.DataFrame({"category": ["a", "b", "a"]})
        with self.assertRaises(HTTPException) as error:
            self.run_operation(frame, {"operation": "standardize", "columns": ["category"]})
        self.assertEqual(error.exception.status_code, 400)

    def test_pca_uses_only_selected_numeric_columns(self):
        frame = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "untouched": ["x", "y", "x", "y"],
        })
        result = self.run_operation(frame, {"operation": "pca", "columns": ["a", "b"], "components": 1})
        names = [column["column"] for column in result["columns_info"]]
        self.assertIn("PC1", names)
        self.assertIn("untouched", names)
        self.assertNotIn("a", names)
        self.assertNotIn("b", names)


if __name__ == "__main__":
    unittest.main()
