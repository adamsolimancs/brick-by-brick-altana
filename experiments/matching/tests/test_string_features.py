"""Real Spark scoring of saved pairs: additive metrics, Unicode, and guards."""

import json
import os
import sys
import unittest
from pathlib import Path

import duckdb
from pyspark.sql import SparkSession, functions as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.matching import prepare_record  # noqa: E402
from lib.string_features import (  # noqa: E402
    COMPARISONS,
    RECORD_COLUMNS,
    prepare_comparison_records,
    add_string_features,
    attach_comparison_fields,
    invalid_source_condition,
)


def record(i, **changes):
    return prepare_record(
        {
            "provider": "fixture",
            "provider_id": str(i),
            "company": "Acme LLC",
            "street_address": "123 Main St Suite 4",
            "city": "Zürich",
            "state": "NY",
            "country": "US",
            **changes,
        }
    )


class StringFeaturesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.previous_pythonpath = os.environ.get("PYTHONPATH")
        # Local Spark does not distribute driver sys.path like Databricks does.
        os.environ["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                [str(Path(__file__).resolve().parents[1]), cls.previous_pythonpath],
            )
        )
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("candidate-string-features")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        if cls.previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = cls.previous_pythonpath

    def test_reparsed_streets_score_names_and_preserve_new_component_conflicts(self):
        source = [
            record(10, street_address="123 Main St Suite 4, Floor 12"),
            record(11, street_address="124 Main Street Suite 9 - Floor 15"),
            record(12, street_address="Wang Road Building F Unit 4", country="CN"),
            record(13, street_address="Wang Road J Building Unit 4", country="CN"),
            record(14, street_address="85/18 Baker Highway", country="AU"),
            record(15, street_address="90/18 Baker Highway", country="AU"),
            record(16, street_address="Gehringerweg 47-26", country="DE"),
            record(17, street_address="Gehringerweg 48-30", country="DE"),
            record(18, street_address="123 Route 66 Apt 4 Floor 12"),
            record(19, street_address="124 Route 66 Unit 5 Floor 13"),
        ]
        names = [
            "chained",
            "buildings",
            "slash_units",
            "trailing_house",
            "numbered_road",
        ]
        pairs = self.spark.createDataFrame(
            [
                (
                    name,
                    a["unique_id"],
                    b["unique_id"],
                    a["content_hash"],
                    b["content_hash"],
                    False,
                    False,
                    "existing_decision",
                )
                for name, a, b in zip(names, source[::2], source[1::2])
            ],
            "label string,id_l string,id_r string,content_hash_l string,content_hash_r string,house_conflict boolean,unit_conflict boolean,decision string",
        )
        scored = add_string_features(
            attach_comparison_fields(pairs, self.spark.createDataFrame(source))
        )
        rows = {r["label"]: r.asDict() for r in scored.collect()}
        for row in rows.values():
            self.assertEqual(row["street_name_levenshtein_distance"], 0)
            self.assertEqual(row["street_name_jaro_winkler_similarity"], 1)
            self.assertTrue(row["parsed_address_conflict"])
            self.assertFalse(row["house_conflict"])
            self.assertFalse(row["unit_conflict"])
            self.assertEqual(row["decision"], "existing_decision")
        self.assertEqual(rows["chained"]["comparison_street_name_l"], "main st")
        for component in ("house_numbers", "units", "floors"):
            self.assertTrue(rows["chained"][f"parsed_{component}_conflict"])
        self.assertTrue(rows["buildings"]["parsed_buildings_conflict"])
        self.assertFalse(rows["buildings"]["parsed_units_conflict"])
        self.assertTrue(rows["slash_units"]["parsed_units_conflict"])
        self.assertFalse(rows["slash_units"]["parsed_house_numbers_conflict"])
        self.assertTrue(rows["trailing_house"]["parsed_house_numbers_conflict"])
        self.assertEqual(rows["numbered_road"]["comparison_street_name_l"], "route 66")
        self.assertEqual(scored.select(*pairs.columns).exceptAll(pairs).count(), 0)

    def test_partial_component_sets_require_review_without_a_false_contradiction(self):
        sources = [
            record(30, street_address="123 Main St Suite 541, Suite 970"),
            record(31, street_address="123 Main Street Suite 541"),
        ]
        pairs = self.spark.createDataFrame(
            [
                (
                    sources[0]["unique_id"],
                    sources[1]["unique_id"],
                    sources[0]["content_hash"],
                    sources[1]["content_hash"],
                )
            ],
            "id_l string,id_r string,content_hash_l string,content_hash_r string",
        )
        result = (
            attach_comparison_fields(pairs, self.spark.createDataFrame(sources))
            .first()
            .asDict()
        )
        self.assertFalse(result["parsed_units_conflict"])
        self.assertTrue(result["parsed_units_partial_agreement"])
        self.assertFalse(result["parsed_address_conflict"])
        self.assertTrue(result["parsed_address_review_required"])

    def test_plural_road_name_does_not_create_false_suite_review(self):
        sources = [
            record(i, street_address="Flat 77 Example Flats", country="GB")
            for i in (40, 41)
        ]
        pairs = self.spark.createDataFrame(
            [
                (
                    sources[0]["unique_id"],
                    sources[1]["unique_id"],
                    sources[0]["content_hash"],
                    sources[1]["content_hash"],
                    False,
                    "review_uncertain",
                )
            ],
            "id_l string,id_r string,content_hash_l string,content_hash_r string,unit_conflict boolean,decision string",
        )
        result = (
            add_string_features(
                attach_comparison_fields(pairs, self.spark.createDataFrame(sources))
            )
            .first()
            .asDict()
        )
        for side in ("l", "r"):
            self.assertEqual(result[f"parsed_units_{side}"], ["77"])
            self.assertEqual(result[f"comparison_street_name_{side}"], "example flats")
        self.assertFalse(result["parsed_address_conflict"])
        self.assertFalse(result["parsed_address_review_required"])
        self.assertEqual(result["street_name_jaro_winkler_similarity"], 1)
        self.assertFalse(result["unit_conflict"])
        self.assertEqual(result["decision"], "review_uncertain")

    def test_end_to_end_metrics_leave_building_suite_and_original_evidence_intact(self):
        records = [
            record(1),
            record(2, street_address="124 Main Street Suite 9"),
            # Zürich/Zurich is folded on purpose; Zurick keeps the one-edit typo.
            record(
                3,
                company="Acne LLC",
                street_address="123 Maim Street #4",
                city="Zurick",
            ),
        ]
        prepared = self.spark.createDataFrame(records)
        pairs = self.spark.createDataFrame(
            [
                (
                    "building_and_suite",
                    records[0]["unique_id"],
                    records[1]["unique_id"],
                    records[0]["content_hash"],
                    records[1]["content_hash"],
                    True,
                    True,
                    "review_conflict",
                    0.97,
                ),
                (
                    "typos",
                    records[0]["unique_id"],
                    records[2]["unique_id"],
                    records[0]["content_hash"],
                    records[2]["content_hash"],
                    False,
                    False,
                    "review_embedding_candidate",
                    0.92,
                ),
            ],
            "label string, id_l string, id_r string, content_hash_l string, content_hash_r string, house_conflict boolean, unit_conflict boolean, decision string, retrieval_score double",
        )
        enriched = attach_comparison_fields(pairs, prepared)
        self.assertEqual(enriched.where(invalid_source_condition()).count(), 0)
        scored = add_string_features(enriched)
        observed = {r["label"]: r.asDict() for r in scored.collect()}
        conflict = observed["building_and_suite"]
        self.assertEqual(conflict["street_name_levenshtein_distance"], 0)
        self.assertEqual(conflict["street_name_jaro_winkler_similarity"], 1)
        self.assertTrue(conflict["house_conflict"])
        self.assertTrue(conflict["unit_conflict"])
        self.assertEqual(conflict["decision"], "review_conflict")
        typo = observed["typos"]
        for name in COMPARISONS:
            self.assertEqual(typo[f"{name}_levenshtein_distance"], 1)
        self.assertAlmostEqual(typo["company_levenshtein_similarity"], 0.75)
        self.assertAlmostEqual(typo["street_name_levenshtein_similarity"], 1 - 1 / 7)
        self.assertAlmostEqual(typo["city_levenshtein_similarity"], 1 - 1 / 6)
        self.assertEqual(scored.select(*pairs.columns).exceptAll(pairs).count(), 0)
        # Validate the old ASCII Jaro-Winkler convention alongside the new field.
        with duckdb.connect(":memory:") as connection:
            expected = connection.execute(
                "SELECT jaro_winkler_similarity('acme','acne')"
            ).fetchone()[0]
        self.assertAlmostEqual(typo["company_jaro_winkler_similarity"], expected)

    def test_missing_keys_stay_unknown_and_unicode_uses_character_lengths(self):
        frame = self.spark.createDataFrame(
            [
                ("exact", "東京", "東京"),
                ("unicode", "東京", "東亰"),
                ("empty", "", ""),
                ("null", None, "acme"),
                ("blank", "   ", "acme"),
            ],
            "label string, left string, right string",
        )
        for name in COMPARISONS:
            frame = frame.withColumn(f"comparison_{name}_l", F.col("left")).withColumn(
                f"comparison_{name}_r", F.col("right")
            )
        result = {r["label"]: r.asDict() for r in add_string_features(frame).collect()}
        for name in COMPARISONS:
            self.assertEqual(result["exact"][f"{name}_levenshtein_distance"], 0)
            self.assertEqual(result["exact"][f"{name}_levenshtein_similarity"], 1)
            self.assertEqual(result["unicode"][f"{name}_levenshtein_distance"], 1)
            self.assertEqual(result["unicode"][f"{name}_levenshtein_similarity"], 0.5)
            for label in ("empty", "null", "blank"):
                for metric in (
                    "levenshtein_distance",
                    "levenshtein_similarity",
                    "jaro_winkler_similarity",
                ):
                    self.assertIsNone(result[label][f"{name}_{metric}"])

    def test_missing_or_stale_source_record_is_detected_without_dropping_pair(self):
        source = record(1)
        records = self.spark.createDataFrame([source])
        pairs = self.spark.createDataFrame(
            [
                (source["unique_id"], "missing", "stale", "missing"),
            ],
            "id_l string, id_r string, content_hash_l string, content_hash_r string",
        )
        enriched = attach_comparison_fields(pairs, records)
        self.assertEqual(enriched.count(), 1)
        self.assertEqual(enriched.where(invalid_source_condition()).count(), 1)

    def pair_of(self, left, right, **columns):
        pairs = self.spark.createDataFrame(
            [
                (
                    left["unique_id"],
                    right["unique_id"],
                    left["content_hash"],
                    right["content_hash"],
                )
            ],
            "id_l string,id_r string,content_hash_l string,content_hash_r string",
        )
        records = self.spark.createDataFrame([left, right])
        return attach_comparison_fields(pairs, records).first().asDict()

    def test_identical_multi_unit_parses_do_not_require_review(self):
        same = self.pair_of(
            record(40, street_address="4613 Hoffman Orchard Suite 883, Apt 187"),
            record(41, street_address="4613 Hoffman Orchard Suite 883 Apt 187"),
        )
        self.assertEqual(sorted(same["parsed_units_l"]), ["187", "883"])
        self.assertEqual(same["parsed_units_l"], same["parsed_units_r"])
        self.assertFalse(same["parsed_address_review_required"])
        self.assertFalse(same["parsed_address_conflict"])
        different = self.pair_of(
            record(42, street_address="4613 Hoffman Orchard Suite 883, Apt 187"),
            record(43, street_address="4613 Hoffman Orchard Suite 883, Apt 188"),
        )
        self.assertTrue(different["parsed_units_partial_agreement"])
        self.assertTrue(different["parsed_address_review_required"])
        one_sided = self.pair_of(
            record(44, street_address="4613 Hoffman Orchard Suite 883, Apt 187"),
            record(45, street_address="4613 Hoffman Orchard Suite 883"),
        )
        self.assertTrue(one_sided["parsed_address_review_required"])

    def test_original_text_agreement_is_only_claimed_when_both_sides_have_it(self):
        def evidence(**originals):
            return json.dumps({f"{k}_original": v for k, v in originals.items()})

        translated = self.pair_of(
            record(
                50,
                company="Viewwin Network Co. Ltd.",
                street_address="867 Mori Street Building D",
                city="Machida-shi",
                country="JP",
                translation_evidence_json=evidence(
                    company="维旺明网络有限公司 S.A.",
                    street_address="867 森 Street Building D",
                    city="町田市",
                ),
            ),
            record(
                51,
                company="ViewNet Information Technology Co., Ltd.",
                street_address="867 Forest St. - Building D",
                city="Machida City",
                country="JP",
                translation_evidence_json=evidence(
                    company="维旺明网络有限公司",
                    street_address="867 森 Street Building D",
                    city="町田市",
                ),
            ),
        )
        self.assertEqual(translated["original_name_key_l"], "维旺明网络")
        self.assertEqual(translated["original_road_key_l"], "森 street")
        self.assertEqual(translated["original_city_key_l"], "町田市")
        self.assertTrue(translated["original_company_agreement"])
        self.assertTrue(translated["original_street_name_agreement"])
        self.assertTrue(translated["original_city_agreement"])
        self.assertEqual(translated["comparison_street_name_l"], "mori street")
        self.assertEqual(translated["comparison_street_name_r"], "forest st")
        one_sided = self.pair_of(
            record(
                52,
                company="Saito University",
                street_address="281 Satō Street",
                city="Yokohama",
                country="JP",
                translation_evidence_json=evidence(
                    company="斎藤 University", street_address="281 佐藤 Street"
                ),
            ),
            record(
                53,
                company="Saito University",
                street_address="281 Sato St.",
                city="Yokohama",
                country="JP",
                translation_evidence_json=evidence(company="斎藤 University"),
            ),
        )
        self.assertTrue(one_sided["original_company_agreement"])
        self.assertFalse(one_sided["original_street_name_agreement"])
        self.assertFalse(one_sided["original_city_agreement"])
        self.assertIsNone(one_sided["original_city_key_l"])
        self.assertIsNone(one_sided["original_road_key_r"])
        unchanged = self.pair_of(
            record(54, translation_evidence_json=evidence(company="Acme LLC")),
            record(55),
        )
        self.assertIsNone(unchanged["original_name_key_l"])
        self.assertFalse(unchanged["original_company_agreement"])
        without_evidence = self.pair_of(record(56), record(57))
        self.assertFalse(without_evidence["original_company_agreement"])

    def test_notebook_record_columns_are_enough_to_prepare_records(self):
        source = record(
            60,
            company="Viewwin Network Co. Ltd.",
            translation_evidence_json=json.dumps(
                {"company_original": "维旺明网络有限公司 S.A."}
            ),
        )
        records = self.spark.createDataFrame([source]).select(*RECORD_COLUMNS)
        prepared = prepare_comparison_records(records).first().asDict()
        self.assertEqual(prepared["original_name_key"], "维旺明网络")
        self.assertEqual(prepared["street_component_road_key"], "main st")


if __name__ == "__main__":
    unittest.main()
