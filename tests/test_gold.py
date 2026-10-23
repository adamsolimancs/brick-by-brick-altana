"""Execute real Spark plans without Databricks services or model calls."""

import os
import sys
import unittest

from pyspark.sql import SparkSession

from lib.gold import build_gold, prepare_gold_input


class GoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("gold-integrity-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.ansi.enabled", "true")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def frame(self, records):
        return self.spark.createDataFrame(
            [(*r, None, None, False) for r in records],
            "provider string, provider_id int, company string, street_address string, "
            "city string, state string, country string, created_at date, updated_at date, country_inferred boolean",
        )

    def test_routing_and_exact_entity_boundaries(self):
        def row(provider, key, company="Acme Ltd", street="12 Main St", country="US"):
            return provider, key, company, street, "Boston", "MA", country

        source = self.frame(
            [
                row("A", 1),
                row("A", 1),  # duplicate must not inflate support
                row("B", 1, company=" ACME\u00a0Ltd "),  # composite source ID
                row("A", 2, company="Other Ltd"),  # shared address != company
                row("A", 3, street="12 Main St Apt 2"),  # unit preserved
                row("A", 4, street="99 Main St"),  # other company location
                row("A", 5, company="北京"),
                row("A", 6, street=None),
                row("A", None),
                row("A", 7, country="USA"),
                row("A", 8),
                row("A", 8, street="777 Main St"),
            ]
        )
        prepared = prepare_gold_input(source).cache()
        try:
            self.assertEqual(prepared.count(), 12)
            rejected = prepared.where("size(gold_input_reasons) > 0").collect()
            self.assertEqual(len(rejected), 6)
            conflicts = [r for r in rejected if r.provider_id == 8]
            self.assertEqual(len(conflicts), 2)
            self.assertTrue(
                all(
                    "source_id:conflicting_records" in r.gold_input_reasons
                    for r in conflicts
                )
            )
            result = build_gold(prepared).collect()
            self.assertEqual(len(result), 4)
            merged = next(r for r in result if r.source_record_count == 2)
            self.assertEqual(merged.input_row_count, 3)
            self.assertEqual(merged.provider_count, 2)
            self.assertEqual(len(merged.source_records), 2)
            self.assertEqual(len(merged.source_evidence), 2)
            self.assertEqual(len(merged.candidate_records), 2)
            self.assertEqual(
                merged.company, " ACME\u00a0Ltd "
            )  # deterministic lexical tie
            self.assertEqual(merged.selection_support_fraction, 0.5)
            self.assertEqual(merged.company_location_count, 3)
            self.assertIn("company_has_multiple_addresses", merged.review_reasons)
            self.assertTrue(all(r.needs_review for r in result))
            self.assertEqual(sum(r.input_row_count for r in result) + len(rejected), 12)
            # Partition order must not change IDs or selected records.
            reversed_result = build_gold(
                prepare_gold_input(source.repartition(3))
            ).collect()
            self.assertEqual(
                sorted((r.entity_id, r.company, r.street_address) for r in result),
                sorted(
                    (r.entity_id, r.company, r.street_address) for r in reversed_result
                ),
            )
        finally:
            prepared.unpersist()

    def test_empty_and_all_invalid_inputs(self):
        for records in ([], [("A", None, None, None, None, None, None)]):
            prepared = prepare_gold_input(self.frame(records))
            self.assertEqual(build_gold(prepared).count(), 0)
            self.assertEqual(prepared.count(), len(records))


if __name__ == "__main__":
    unittest.main()
