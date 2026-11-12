"""Publication views: the same-company hold, its DLQ, and what reaches Gold."""

import importlib.util
import os
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyspark
from pyspark.sql import DataFrameReader, SparkSession

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
SILVER = "addressresolution.silver"
MATCHING = "addressresolution.matching"
REVIEW = "array<struct<other_record_id:string,decision:string,fs_match_probability:double,reason_codes:array<string>>>"
RECORDS = f"""
    deduplication_run_id string, record_id string, provider string,
    provider_id string, company string, street_address string, city string,
    state string, country string, created_at timestamp, updated_at timestamp,
    name_key string, street_key string, city_key string, region_key string,
    entity_id string, canonical_record_id string, is_canonical boolean,
    is_duplicate boolean, match_review_evidence {REVIEW}
"""


def _identity_decorator(*args, **kwargs):
    return lambda function: function


def _load(name):
    with patch.object(
        pyspark,
        "pipelines",
        SimpleNamespace(
            materialized_view=_identity_decorator,
            expect_or_fail=_identity_decorator,
        ),
        create=True,
    ):
        spec = importlib.util.spec_from_file_location(name, PIPELINE / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def record(
    record_id, entity, company, street, review=(), canonical=None, city="Boston"
):
    return (
        "run2",
        record_id,
        "provider_a",
        record_id,
        company,
        street,
        city,
        "MA",
        "US",
        None,
        None,
        company.lower(),
        street.lower(),
        city.lower(),
        "ma",
        entity,
        canonical or record_id,
        canonical is None,
        canonical is not None,
        [(other, "review_conflict", 0.99, ["numeric_conflict"]) for other in review],
    )


class PublicationOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("publication-outputs")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")
        cls.modules = {
            name: _load(name)
            for name in (
                "addresses_deduplication",
                "dlq_same_company_conflict",
                "addresses_deduplicated",
            )
        }
        rows = [
            # Same company, two streets: both entities are held.
            record("a1", "A", "Acme", "12 Main St", review=["b1"]),
            record("a2", "A", "Acme", "12 Main St", canonical="a1"),
            record("b1", "B", "Acme", "99 Other Rd", review=["a1"]),
            # Same address, different company (shared building): stays in Gold.
            record("c1", "C", "Zeta", "5 Elm St", review=["d1"]),
            record("d1", "D", "Yotta", "5 Elm St", review=["c1"]),
            record("e1", "E", "Solo", "1 Lone Ln"),
        ]
        runs = cls.spark.createDataFrame(
            [
                ("run1", "COMPLETED", datetime(2026, 1, 1)),
                ("run2", "COMPLETED", datetime(2026, 1, 2)),
            ],
            "deduplication_run_id string, status string, completed_at timestamp",
        )
        frames = {
            f"{MATCHING}.resolution_runs": runs,
            f"{MATCHING}.resolved_records": cls.spark.createDataFrame(rows, RECORDS),
        }

        def run(name):
            module = cls.modules[name]
            with (
                patch.object(module, "spark", cls.spark, create=True),
                patch.object(
                    DataFrameReader, "table", side_effect=lambda table: frames[table]
                ),
            ):
                return getattr(module, name)()

        frames[f"{SILVER}.addresses_deduplication"] = run("addresses_deduplication")
        cls.silver = {
            row["record_id"]: row
            for row in frames[f"{SILVER}.addresses_deduplication"].collect()
        }
        cls.dlq = {
            row["record_id"]: row for row in run("dlq_same_company_conflict").collect()
        }
        cls.gold = run("addresses_deduplicated").collect()

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_same_company_conflict_holds_both_entities(self):
        held = {key for key, row in self.silver.items() if row["gold_hold_reason"]}
        self.assertEqual(held, {"a1", "a2", "b1"})
        conflict = self.silver["a1"]["same_company_conflicts"][0]
        self.assertEqual(conflict["other_record_id"], "b1")
        self.assertEqual(conflict["other_street_address"], "99 Other Rd")
        self.assertEqual(conflict["differing_fields"], ["street_address"])

    def test_shared_building_review_is_not_a_hold(self):
        for key in ("c1", "d1", "e1"):
            self.assertIsNone(self.silver[key]["gold_hold_reason"])
            self.assertEqual(self.silver[key]["same_company_conflicts"], [])

    def test_dlq_lists_every_record_of_a_held_entity(self):
        self.assertEqual(set(self.dlq), {"a1", "a2", "b1"})
        self.assertEqual(
            self.dlq["a1"]["dlq_reasons"],
            ["street_address:differs_from_same_company_record"],
        )
        self.assertEqual(
            self.dlq["a2"]["dlq_reasons"], ["entity:member_has_same_company_conflict"]
        )

    def test_remaining_incomplete_queue_drops_rows_completed_from_gold(self):
        module = _load("dlq_incomplete_v2")
        frames = {
            f"{SILVER}.dlq_incomplete": self.spark.createDataFrame(
                [("provider_b", 7, ""), ("provider_b", 8, ""), ("provider_a", 7, "")],
                "provider string, provider_id int, city string",
            ),
            f"{SILVER}.addresses_deduplication": self.spark.createDataFrame(
                [("provider_b", "7", "city"), ("provider_b", "8", None)],
                "provider string, provider_id string, filled_field string",
            ),
        }
        with (
            patch.object(module, "spark", self.spark, create=True),
            patch.object(
                DataFrameReader, "table", side_effect=lambda table: frames[table]
            ),
        ):
            rows = module.dlq_incomplete_v2().collect()
        self.assertEqual(
            sorted((row["provider"], row["provider_id"]) for row in rows),
            [("provider_a", 7), ("provider_b", 8)],
        )

    def test_gold_publishes_only_canonical_records_not_on_hold(self):
        self.assertEqual(
            sorted(row["provider_id"] for row in self.gold), ["c1", "d1", "e1"]
        )


if __name__ == "__main__":
    unittest.main()
