"""The publication pipeline's row accounting view, run on real Spark plans."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyspark
from pyspark.sql import DataFrameReader, SparkSession

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
SILVER = "addressresolution.silver"


def _identity_decorator(*args, **kwargs):
    return lambda function: function


def _load():
    with patch.object(
        pyspark,
        "pipelines",
        SimpleNamespace(
            materialized_view=_identity_decorator,
            expect_all_or_fail=_identity_decorator,
        ),
        create=True,
    ):
        spec = importlib.util.spec_from_file_location(
            "address_pipeline_accounting", PIPELINE / "address_pipeline_accounting.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class PipelineAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("pipeline-accounting")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls.module = _load()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def frames(self, dedup_rows):
        audit = self.spark.createDataFrame(
            [(10, 6, 4, 5, 2, 2, 1, 0)],
            "source_rows long, provider_a_rows long, provider_b_rows long, "
            "cleaned_rows long, translated_rows long, incomplete_cleaning_rows long, "
            "incomplete_translation_rows long, review_rows long",
        )
        incomplete = self.spark.createDataFrame(
            [("", "1 Main St"), ("Boston", "")],
            "city string, street_address string",
        )
        dedup = self.spark.createDataFrame(
            dedup_rows,
            "deduplication_status string, is_canonical boolean, is_duplicate boolean, "
            "needs_match_review boolean, deduplication_run_id string, "
            "gold_hold_reason string, filled_field string",
        )
        return {
            f"{SILVER}.dlq_routing_audit": audit,
            f"{SILVER}.dlq_incomplete_v2": incomplete,
            f"{SILVER}.addresses_deduplication": dedup,
        }

    def accounting(self, dedup_rows):
        frames = self.frames(dedup_rows)
        with (
            patch.object(self.module, "spark", self.spark, create=True),
            patch.object(
                DataFrameReader, "table", side_effect=lambda name: frames[name]
            ),
        ):
            return self.module.address_pipeline_accounting()

    def test_counts_every_route_from_source_to_gold(self):
        hold = "same_company_address_conflict"
        result = self.accounting(
            [
                ("canonical", True, False, True, "run1", None, None),
                ("duplicate", False, True, True, "run1", None, None),
                ("singleton", True, False, False, "run1", None, None),
                ("singleton", True, False, True, "run1", hold, None),
                ("canonical", True, False, True, "run1", hold, None),
                ("duplicate", False, True, False, "run1", hold, None),
                ("input_rejected", False, False, True, "run1", None, None),
                # An incomplete row completed from its Gold record: it is one
                # of the 3 incomplete rows, not an extra deduplication input.
                ("duplicate", False, True, False, "run1", None, "city"),
            ]
        )
        rows = result.collect()
        self.assertEqual(len(rows), 1)
        row = rows[0].asDict()
        self.assertEqual(row["source_rows"], 10)
        self.assertEqual(row["incomplete_rows"], 3)
        self.assertEqual(row["incomplete_filled_rows"], 1)
        self.assertEqual(row["incomplete_parked_rows"], 2)
        self.assertEqual(row["incomplete_city_blank_rows"], 1)
        self.assertEqual(row["incomplete_street_blank_rows"], 1)
        self.assertEqual(row["incomplete_other_rows"], 0)
        self.assertEqual(row["translation_review_rows"], 0)
        self.assertEqual(row["deduplication_input_rows"], 7)
        self.assertEqual(row["gold_rows"], 2)
        self.assertEqual(row["duplicate_rows"], 2)
        self.assertEqual(row["same_company_conflict_rows"], 3)
        self.assertEqual(row["same_company_conflict_entities"], 2)
        self.assertEqual(row["rejected_rows"], 1)
        self.assertEqual(row["gold_rows_needing_match_review"], 1)
        self.assertEqual(row["duplicate_rows_needing_match_review"], 1)
        self.assertEqual(row["deduplication_run_id"], "run1")
        checks = result.selectExpr(*self.module.EXPECTATIONS.values()).first()
        self.assertTrue(all(checks), checks)

    def test_reconciliation_fails_when_a_row_is_lost(self):
        result = self.accounting(
            [
                ("canonical", True, False, False, "run1", None, None),
                ("duplicate", False, True, False, "run1", None, None),
            ]
        )
        checks = result.selectExpr(*self.module.EXPECTATIONS.values()).first()
        self.assertFalse(checks[0], "silver routes must not reconcile with 2 of 7 rows")


if __name__ == "__main__":
    unittest.main()
