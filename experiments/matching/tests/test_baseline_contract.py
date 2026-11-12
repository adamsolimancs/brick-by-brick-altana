"""Compare experiment group membership with the existing real Spark baseline."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.matching import prepare_record  # noqa: E402


class BaselineContractTests(unittest.TestCase):
    def test_same_equivalence_classes_as_existing_gold(self):
        path = ROOT / "lib" / "gold.py"
        spec = importlib.util.spec_from_file_location("existing_gold_baseline", path)
        gold = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gold)
        os.environ["PYSPARK_PYTHON"] = sys.executable
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("matching-baseline-contract")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        try:
            rows = [
                ("A", 1, "Acme LLC", "123 Main St", "Testville", "NY", "US"),
                ("B", 1, " ACME\u00a0LLC ", "123 Main St", "Testville", "NY", "us"),
                ("A", 2, "Acme LLC", "123 Main Street", "Testville", "NY", "US"),
                ("A", 3, "Acme LLC", "128 Main St", "Testville", "NY", "US"),
                ("B", 2, "Other Company", "123 Main St", "Testville", "NY", "US"),
            ]
            frame = spark.createDataFrame(
                rows,
                "provider string, provider_id int, company string, street_address string, city string, state string, country string",
            )
            actual = gold.prepare_gold_input(frame).collect()
            observed = {}
            expected = {}
            for row in actual:
                self.assertEqual(row.gold_input_reasons, [])
                identity = (row.provider, row.provider_id)
                observed.setdefault(
                    prepare_record(row.asDict())["baseline_id"], set()
                ).add(identity)
                expected.setdefault(row.entity_id, set()).add(identity)
            self.assertEqual(
                {frozenset(g) for g in observed.values()},
                {frozenset(g) for g in expected.values()},
            )
        finally:
            spark.stop()


if __name__ == "__main__":
    unittest.main()
