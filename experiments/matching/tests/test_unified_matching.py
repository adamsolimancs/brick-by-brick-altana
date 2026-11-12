"""Real Splink saved-pair inference and Spark decision checks; fitting runs remotely."""

import math
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pyarrow as pa
from splink import DuckDBAPI, Linker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.unified_matching import (  # noqa: E402
    ADDRESS_NON_MATCH,
    route_address_conflicts,
    add_decisions,
    model_settings,
    predict_saved_pairs,
    register_jaro_winkler,
    regularize_model,
    iteration_summary,
)


def fixed_settings():
    settings = model_settings(0.001)
    for comparison in settings["comparisons"]:
        levels = [
            v for v in comparison["comparison_levels"] if not v.get("is_null_level")
        ]
        for i, level in enumerate(levels):
            level["m_probability"] = 0.9 if i == 0 else 0.1 / (len(levels) - 1)
            level["u_probability"] = 0.001 if i == 0 else 0.999 / (len(levels) - 1)
    return settings


class UnifiedInferenceTests(unittest.TestCase):
    def test_parameter_history_rows_are_not_counted_as_em_iterations(self):
        history = [SimpleNamespace(iteration=i) for i in range(8) for _ in range(24)]
        summary = iteration_summary(history)
        self.assertEqual(summary["history_records"], 192)
        self.assertEqual(summary["em_iterations"], 7)
        self.assertFalse(summary["iteration_cap_reached"])
        capped = [SimpleNamespace(iteration=i) for i in range(101) for _ in range(2)]
        self.assertTrue(iteration_summary(capped)["iteration_cap_reached"])
        with self.assertRaises(ValueError):
            iteration_summary([])

    def setUp(self):
        self.con = duckdb.connect(":memory:")
        register_jaro_winkler(self.con)
        rows = [
            dict(
                unique_id="a",
                company="acme",
                street_name="main st",
                city="東京",
                country="US",
            ),
            dict(
                unique_id="b",
                company="acme",
                street_name="main st",
                city="東京",
                country="US",
            ),
            dict(
                unique_id="c",
                company="acme",
                street_name="main st",
                city="東亰",
                country="JP",
            ),
            dict(
                unique_id="d",
                company="other",
                street_name=None,
                city="other",
                country="US",
            ),
        ]
        self.con.register("input_arrow", pa.Table.from_pylist(rows))
        self.con.execute("CREATE TABLE clean_records AS SELECT * FROM input_arrow")
        self.api = DuckDBAPI(self.con)
        self.linker = Linker(self.api.register("clean_records"), fixed_settings())

    def tearDown(self):
        self.con.close()

    def test_exact_saved_pair_set_includes_pair_outside_country_blocks(self):
        self.con.execute(
            "CREATE TABLE saved_id_pairs AS SELECT '0' match_key,'a' join_key_l,'b' join_key_r UNION ALL SELECT '0','a','c' UNION ALL SELECT '0','a','d'"
        )
        predict_saved_pairs(self.con, self.linker)
        rows = (
            self.con.execute("SELECT * FROM unified_predictions")
            .fetch_arrow_table()
            .to_pylist()
        )
        self.assertEqual(
            {(r["unique_id_l"], r["unique_id_r"]) for r in rows},
            {("a", "b"), ("a", "c"), ("a", "d")},
        )
        exact = next(r for r in rows if r["unique_id_r"] == "b")
        unicode = next(r for r in rows if r["unique_id_r"] == "c")
        missing = next(r for r in rows if r["unique_id_r"] == "d")
        self.assertEqual(exact["gamma_company"], 4)
        self.assertEqual(exact["gamma_street_name"], 4)
        self.assertEqual(exact["gamma_city"], 3)
        self.assertEqual(unicode["gamma_city"], 0)
        self.assertEqual(missing["gamma_street_name"], -1)
        self.assertLess(exact["mw_tf_adj_company"], 0)
        for row in rows:
            weights = sum(v for k, v in row.items() if k.startswith("mw_"))
            self.assertAlmostEqual(
                row["match_weight"], math.log2(0.001 / 0.999) + weights
            )
            self.assertTrue(0 <= row["match_probability"] <= 1)

    def test_invalid_missing_duplicate_and_over_budget_candidates_fail(self):
        for query, limit in [
            ("SELECT '0' match_key,'a' join_key_l,'z' join_key_r", 10),
            ("SELECT '0' match_key,'b' join_key_l,'a' join_key_r", 10),
            (
                "SELECT '0' match_key,'a' join_key_l,'b' join_key_r UNION ALL SELECT '0','a','b'",
                10,
            ),
            (
                "SELECT '0' match_key,'a' join_key_l,'b' join_key_r UNION ALL SELECT '0','a','c'",
                1,
            ),
        ]:
            self.con.execute("CREATE OR REPLACE TABLE saved_id_pairs AS " + query)
            with self.subTest(query=query), self.assertRaises(ValueError):
                predict_saved_pairs(self.con, self.linker, max_pairs=limit)

    def test_zero_and_unobserved_parameters_are_explicit_positive_distributions(self):
        model = fixed_settings()
        levels = [
            v
            for v in model["comparisons"][0]["comparison_levels"]
            if not v.get("is_null_level")
        ]
        levels[0]["m_probability"] = 0
        del levels[1]["u_probability"]
        result, metadata = regularize_model(model)
        self.assertEqual(len(metadata["unobserved_parameters"]), 1)
        self.assertTrue(metadata["floored_parameters"])
        for comparison in result["comparisons"]:
            for parameter in ["m_probability", "u_probability"]:
                values = [
                    v[parameter]
                    for v in comparison["comparison_levels"]
                    if not v.get("is_null_level")
                ]
                self.assertAlmostEqual(sum(values), 1)
                self.assertTrue(all(0 < p < 1 for p in values))
        for level in levels:
            level.pop("m_probability", None)
        with self.assertRaisesRegex(ValueError, "Entire comparison"):
            regularize_model(model)


class UnifiedDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession

        os.environ["PYSPARK_PYTHON"] = sys.executable
        worker_path = os.pathsep.join(
            filter(
                None,
                [
                    str(Path(__file__).resolve().parents[1]),
                    os.environ.get("PYTHONPATH"),
                ],
            )
        )
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("unified-fs-guards")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.executorEnv.PYTHONPATH", worker_path)
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_pair_country_correction_requires_supplied_country_and_address_agreement(
        self,
    ):
        base = {
            "country_l": "TR",
            "country_r": "US",
            "state_l": "US-UT",
            "state_r": "Utah",
            "fs_country_inferred_l": True,
            "fs_country_inferred_r": False,
            "country_conflict": True,
            "region_conflict": True,
            "house_conflict": False,
            "unit_conflict": False,
            "numeric_conflict": False,
            "parsed_address_conflict": False,
            "parsed_address_review_required": False,
            "unit_missing_one_side": False,
            "source_exact_baseline": False,
            "fs_match_probability": 0.999,
            "fs_match_weight": 10.0,
            **{
                f"{field}_jaro_winkler_similarity": 1.0
                for field in ("company", "street_name", "city")
            },
        }
        cases = [
            ("correct", {}, "US"),
            ("supplied", {"fs_country_inferred_l": False}, None),
            ("both_inferred", {"fs_country_inferred_r": True}, None),
            ("suite", {"unit_conflict": True}, None),
            ("building", {"parsed_address_conflict": True}, None),
            ("region", {"state_r": "Oregon"}, None),
            ("weak", {"fs_match_probability": 0.98}, None),
            (
                "vote_us",
                {"id_l": "ambiguous", "state_l": "CA", "state_r": "California"},
                None,
            ),
            (
                "vote_ca",
                {
                    "id_l": "ambiguous",
                    "country_r": "CA",
                    "state_l": "CA",
                    "state_r": "CA",
                },
                None,
            ),
        ]
        rows = [
            dict(base, id_l=label, id_r=label + "_donor") | changes
            for label, changes, _ in cases
        ]
        scored = add_decisions(
            self.spark.createDataFrame(rows),
            {"prior": 0.001, "unobserved_parameters": []},
            pair_geography=True,
        )
        by_donor = {row.id_r: row for row in scored.collect()}
        for label, _, expected in cases:
            row = by_donor[label + "_donor"]
            self.assertEqual(row.fs_corrected_country, expected)
            self.assertEqual(
                row.fs_decision, "proposed_match" if expected else "review_conflict"
            )
            self.assertTrue(row.country_conflict)
            self.assertIsNone(row.fs_human_decision)
            self.assertEqual(
                row.fs_merge_authority, "model_proposal" if expected else "none"
            )

    def test_number_conflicts_close_only_when_the_parsed_evidence_is_clear(self):
        base = dict(
            fs_decision="review_conflict",
            fs_match_probability=0.99999,
            company_jaro_winkler_similarity=0.97,
            street_name_jaro_winkler_similarity=0.98,
            city_jaro_winkler_similarity=1.0,
            country_l="TR",
            country_r="TR",
            street_address_l="27 Hançer Mission, Suite 215",
            street_address_r="277 Haçer Mission, Suite 215",
            street_parse_warnings_l=[],
            street_parse_warnings_r=[],
            parsed_house_numbers_l=["27"],
            parsed_house_numbers_r=["277"],
            parsed_units_l=["215"],
            parsed_units_r=["215"],
            parsed_buildings_l=["a"],
            parsed_buildings_r=["a"],
            parsed_floors_l=["1"],
            parsed_floors_r=["1"],
            house_conflict=True,
            unit_conflict=False,
            numeric_conflict=True,
            parsed_house_numbers_conflict=True,
            parsed_units_conflict=False,
        )
        cases = {
            "omitted_digit": ({}, "review_uncertain", ["possible_house_number_typo"]),
            "different_house": (
                {"parsed_house_numbers_r": ["29"]},
                ADDRESS_NON_MATCH,
                ["different_house_numbers"],
            ),
            "different_company": (
                {"company_jaro_winkler_similarity": 0.55},
                ADDRESS_NON_MATCH,
                ["different_house_numbers"],
            ),
            "range": (
                {"parsed_house_numbers_r": ["27-29"]},
                "review_conflict",
                ["ambiguous_address_number_evidence"],
            ),
            "legacy_house_flag": (
                {"parsed_house_numbers_r": ["027"]},
                "review_conflict",
                ["ambiguous_address_number_evidence"],
            ),
            "slash_typo": (
                {
                    "parsed_house_numbers_l": ["93/17"],
                    "parsed_house_numbers_r": ["93/7"],
                },
                "review_uncertain",
                ["possible_house_number_typo"],
            ),
            "different_explicit_suites": (
                {
                    "street_address_l": "Studio 7 George Junctions, Suite 105",
                    "street_address_r": "Estúdio 7 George Junctions, Suite 109",
                    "parsed_house_numbers_l": [],
                    "parsed_house_numbers_r": [],
                    "parsed_units_l": ["7", "105"],
                    "parsed_units_r": ["7", "109"],
                    "street_parse_warnings_l": ["multiple_units"],
                    "street_parse_warnings_r": ["multiple_units"],
                },
                ADDRESS_NON_MATCH,
                ["different_suite_numbers"],
            ),
            "ambiguous_parse": (
                {"street_parse_warnings_r": ["ambiguous_trailing_numbers_retained"]},
                "review_conflict",
                ["ambiguous_address_number_evidence"],
            ),
            "exact_baseline": ({"fs_decision": "exact_baseline"}, "exact_baseline", []),
        }
        rows = [
            dict(base, label=label) | values for label, (values, _, _) in cases.items()
        ]
        result = route_address_conflicts(self.spark.createDataFrame(rows))
        for row in result.collect():
            with self.subTest(case=row.label):
                self.assertEqual(
                    (row.fs_decision, row.fs_address_conflict_reasons),
                    cases[row.label][1:],
                )

    def test_high_probability_cannot_override_component_or_unknown_evidence(self):
        base = dict(
            house_conflict=False,
            unit_conflict=False,
            numeric_conflict=False,
            region_conflict=False,
            country_conflict=False,
            parsed_address_conflict=False,
            parsed_address_review_required=False,
            unit_missing_one_side=False,
            source_exact_baseline=False,
            company_jaro_winkler_similarity=1.0,
            street_name_jaro_winkler_similarity=1.0,
            city_jaro_winkler_similarity=1.0,
            fs_gamma_company=4,
            fs_gamma_street_name=4,
            fs_gamma_city=3,
            decision="historical",
            match_probability=0.5,
        )
        cases = [
            ("proposed", {}, "proposed_match"),
            ("house", {"house_conflict": True}, "review_conflict"),
            ("suite", {"unit_conflict": True}, "review_conflict"),
            ("country", {"country_conflict": True}, "review_conflict"),
            ("region", {"region_conflict": True}, "review_conflict"),
            ("new_building", {"parsed_address_conflict": True}, "review_conflict"),
            ("partial", {"parsed_address_review_required": True}, "review_uncertain"),
            ("missing_suite", {"unit_missing_one_side": True}, "review_uncertain"),
            (
                "missing_road",
                {"street_name_jaro_winkler_similarity": None},
                "review_uncertain",
            ),
            ("unsupported", {"fs_gamma_street_name": 2}, "review_uncertain"),
            ("baseline", {"source_exact_baseline": True}, "exact_baseline"),
            ("nonmatch", {"fs_match_probability": 0.001}, "likely_nonmatch"),
        ]
        rows = []
        for label, changes, _ in cases:
            row = {**base, "label": label, "fs_match_probability": 0.999999, **changes}
            row["fs_match_weight"] = math.log2(
                row["fs_match_probability"] / (1 - row["fs_match_probability"])
            )
            rows.append(row)
        frame = self.spark.createDataFrame(rows)
        training = {
            "prior": 0.001,
            "unobserved_parameters": [{"comparison": "street_name", "gamma": 2}],
        }
        result = add_decisions(frame, training)
        observed = {r["label"]: r.asDict() for r in result.collect()}
        for label, _, expected in cases:
            self.assertEqual(observed[label]["fs_decision"], expected)
            self.assertIsNone(observed[label]["fs_human_decision"])
            self.assertEqual(
                observed[label]["fs_merge_authority"],
                {
                    "exact_baseline": "exact_rule",
                    "proposed_match": "model_proposal",
                }.get(expected, "none"),
            )
            self.assertGreaterEqual(
                observed[label]["fs_probability_prior_4x"],
                observed[label]["fs_probability_prior_2x"],
            )
            self.assertGreaterEqual(
                observed[label]["fs_probability_prior_2x"],
                observed[label]["fs_match_probability"],
            )
        self.assertEqual(result.select(*frame.columns).exceptAll(frame).count(), 0)
        capped = add_decisions(
            frame, {**training, "training_sessions": [{"iteration_cap_reached": True}]}
        )
        self.assertEqual(
            capped.where("label='proposed'").first()["fs_decision"], "review_uncertain"
        )

    def test_original_text_agreement_proposes_despite_inconsistent_translation(self):
        base = dict(
            house_conflict=False,
            unit_conflict=False,
            numeric_conflict=False,
            region_conflict=False,
            country_conflict=False,
            parsed_address_conflict=False,
            parsed_address_review_required=False,
            unit_missing_one_side=False,
            source_exact_baseline=False,
            company_jaro_winkler_similarity=1.0,
            street_name_jaro_winkler_similarity=1.0,
            city_jaro_winkler_similarity=1.0,
            original_company_agreement=False,
            original_street_name_agreement=False,
            original_city_agreement=False,
            fs_gamma_company=4,
            fs_gamma_street_name=4,
            fs_gamma_city=3,
            fs_match_probability=0.008,
            decision="historical",
            match_probability=0.5,
        )
        cases = [
            (
                "street_rendered_twice",
                {
                    "street_name_jaro_winkler_similarity": 0.68,
                    "city_jaro_winkler_similarity": 0.91,
                    "original_street_name_agreement": True,
                },
                "proposed_match",
                True,
            ),
            (
                "company_rendered_twice",
                {
                    "company_jaro_winkler_similarity": 0.67,
                    "original_company_agreement": True,
                    "fs_match_probability": 0.992,
                },
                "proposed_match",
                True,
            ),
            (
                "all_originals",
                {
                    "company_jaro_winkler_similarity": 0.2,
                    "street_name_jaro_winkler_similarity": 0.2,
                    "city_jaro_winkler_similarity": 0.2,
                    "original_company_agreement": True,
                    "original_street_name_agreement": True,
                    "original_city_agreement": True,
                },
                "proposed_match",
                True,
            ),
            (
                "conflict_still_wins",
                {"original_street_name_agreement": True, "house_conflict": True},
                "review_conflict",
                False,
            ),
            (
                "review_still_wins",
                {
                    "original_street_name_agreement": True,
                    "parsed_address_review_required": True,
                },
                "review_uncertain",
                False,
            ),
            (
                "city_not_covered",
                {
                    "street_name_jaro_winkler_similarity": 0.68,
                    "city_jaro_winkler_similarity": 0.6,
                    "original_street_name_agreement": True,
                },
                "likely_nonmatch",
                False,
            ),
            (
                "company_needs_near_exact",
                {
                    "company_jaro_winkler_similarity": 0.9,
                    "original_street_name_agreement": True,
                    "fs_match_probability": 0.5,
                },
                "review_uncertain",
                False,
            ),
            ("no_original_evidence", {}, "likely_nonmatch", False),
        ]
        rows = []
        for label, changes, _, _ in cases:
            row = {**base, "label": label, **changes}
            row["fs_match_weight"] = math.log2(
                row["fs_match_probability"] / (1 - row["fs_match_probability"])
            )
            rows.append(row)
        result = add_decisions(
            self.spark.createDataFrame(rows),
            {"prior": 0.001, "unobserved_parameters": []},
        )
        observed = {r["label"]: r.asDict() for r in result.collect()}
        for label, _, decision, agreement in cases:
            with self.subTest(label=label):
                self.assertEqual(observed[label]["fs_decision"], decision)
                self.assertEqual(
                    observed[label]["fs_original_text_agreement"], agreement
                )
                self.assertEqual(
                    observed[label]["fs_merge_authority"],
                    "model_proposal" if decision == "proposed_match" else "none",
                )


if __name__ == "__main__":
    unittest.main()
