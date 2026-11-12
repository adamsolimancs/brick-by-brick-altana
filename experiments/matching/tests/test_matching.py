"""Real candidate SQL and fixed-model inference; no local model fitting."""

import sys
import unittest
from pathlib import Path

import duckdb
from splink import DuckDBAPI, Linker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.matching import (  # noqa: E402
    build_candidates,
    comparison_summary,
    model_settings,
    prepare_record,
    register_records,
    suggestions,
)


def record(provider_id, company="Acme LLC", street="123 Main St Suite 4", **changes):
    return {
        "provider": "fixture",
        "provider_id": str(provider_id),
        "company": company,
        "street_address": street,
        "city": "Testville",
        "state": "New York",
        "country": "US",
        "input_route": "clean",
        **changes,
    }


class MatchingTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect(":memory:")

    def tearDown(self):
        self.con.close()

    def test_normalization_preserves_location_identity(self):
        original = prepare_record(record(1))
        variant = prepare_record(
            record(2, company="ACME", street="123 Main Street #4", state="NY")
        )
        self.assertEqual(original["street_key"], variant["street_key"])
        self.assertEqual(original["region_key"], variant["region_key"])
        self.assertEqual(original["name_key"], variant["name_key"])
        self.assertNotEqual(original["baseline_id"], variant["baseline_id"])
        self.assertEqual(original["house_key"], "123")
        self.assertEqual(original["unit_key"], "4")
        self.assertEqual(
            prepare_record(record(3, street="64A Test Rd"))["house_key"], "64a"
        )
        self.assertEqual(
            prepare_record(record(4, street="64 Test Rd"))["house_key"], "64"
        )
        self.assertNotEqual(
            prepare_record(record(5, street="Test Rd 64"))["parser_status"],
            "leading_house_number",
        )

    def test_exact_baseline_and_rescue_passes(self):
        rows = [
            record(1),
            record(2, company=" ACME\u00a0LLC "),
            record(3, company="Acne LLC", street="123 Main Street #4"),
            record(4, state="New Jersey", city="Elsewhere"),
            record(5, provider="other", country="CA"),
        ]
        prepared = register_records(self.con, rows)
        audit, _ = build_candidates(self.con)
        self.assertEqual(audit[0]["raw_pair_count"], 1)
        found = self.con.execute(
            "SELECT id_l,id_r,blocking_passes FROM candidates"
        ).fetchall()
        pair = tuple(sorted((prepared[0]["unique_id"], prepared[3]["unique_id"])))
        self.assertTrue(
            any(
                (a, b) == pair and "country_name_rescue" in passes
                for a, b, passes in found
            )
        )
        self.assertTrue(
            all(prepared[4]["unique_id"] not in (a, b) for a, b, _ in found)
        )
        self.assertEqual(len(found), len({(a, b) for a, b, _ in found}))
        self.assertTrue(all(a < b for a, b, _ in found))

    def test_budget_fails_before_pair_expansion(self):
        register_records(self.con, [record(i) for i in range(5)])
        with self.assertRaisesRegex(ValueError, "budget"):
            build_candidates(self.con, max_pairs=2)
        self.assertNotIn(
            "candidates", [r[0] for r in self.con.execute("SHOW TABLES").fetchall()]
        )

    def test_duplicate_source_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate source"):
            register_records(self.con, [record(1), record(1, street="999 Test Rd")])

    def test_inference_guards_and_baseline_conservation(self):
        records = [
            record(1),
            record(2),
            record(3, company="ACME", street="123 Main Street #4", state="NY"),
            record(4, street="128 Main St Suite 4"),
            record(5, street="123 Main St Suite 9"),
            record(6, company="Different Company"),
            record(7, street="123 Main St"),
        ]
        prepared = register_records(self.con, records)
        _, rules = build_candidates(self.con)
        settings = model_settings(rules, 0.001).create_settings_dict("duckdb")
        # These fixed distributions test inference plumbing, not learned quality.
        for comparison in settings["comparisons"]:
            levels = [
                level
                for level in comparison["comparison_levels"]
                if not level.get("is_null_level")
            ]
            count = len(levels)
            for index, level in enumerate(levels):
                level["m_probability"] = 0.9 if index == 0 else 0.1 / (count - 1)
                level["u_probability"] = 0.001 if index == 0 else 0.999 / (count - 1)
        api = DuckDBAPI(self.con)
        linker = Linker(api.register("records"), settings)
        predicted = linker.inference.predict(threshold_match_probability=0)
        self.con.execute(
            f"CREATE TABLE predictions AS SELECT * FROM {predicted.physical_name}"
        )
        suggestions(self.con, 0.001)
        rows = (
            self.con.execute("SELECT * FROM decisions").fetch_arrow_table().to_pylist()
        )
        ids = {i + 1: r["unique_id"] for i, r in enumerate(prepared)}

        def pair(a, b):
            return next(
                r
                for r in rows
                if {r["unique_id_l"], r["unique_id_r"]} == {ids[a], ids[b]}
            )

        self.assertEqual(pair(1, 2)["decision"], "exact_baseline")
        self.assertEqual(pair(1, 3)["decision"], "proposed_match")
        self.assertEqual(pair(1, 4)["decision"], "review_conflict")
        self.assertEqual(pair(1, 5)["decision"], "review_conflict")
        self.assertNotEqual(pair(1, 6)["decision"], "proposed_match")
        self.assertFalse(pair(1, 7)["numeric_conflict"])
        self.assertTrue(pair(1, 7)["unit_missing_one_side"])
        self.assertEqual(pair(1, 7)["decision"], "review_uncertain")
        self.assertFalse(any(r["approved_for_merge"] for r in rows))
        self.assertTrue(
            all(
                r["probability_prior_4x"]
                >= r["probability_prior_2x"]
                >= r["match_probability"]
                for r in rows
            )
        )
        summary = comparison_summary(self.con, len(prepared))
        self.assertEqual(summary["exact_baseline_pairs"], 1)
        self.assertEqual(summary["baseline_pairs_retained"], 1)
        self.assertIsNone(summary["precision"])


if __name__ == "__main__":
    unittest.main()
