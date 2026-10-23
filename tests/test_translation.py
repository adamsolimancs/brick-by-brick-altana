"""Run with python -m unittest discover -s tests (requires PySpark 3.5+).

Exercise Spark transformations with controlled model responses. No Databricks
endpoint is invoked; decorators alone are stubbed for the local Spark runtime.
"""

import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import pyspark
    from pyspark.sql import SparkSession, functions as F, types as T
except ImportError as exc:
    raise unittest.SkipTest("PySpark is required for translation tests") from exc


registered_expectations = {}


def _expect_all(rules):
    def decorate(function):
        function._test_expectations = dict(rules)
        return function

    return decorate


def _decorator(**kwargs):
    def register(function):
        # Capture constraints at registration time, so reversed decorator order
        # cannot pass unnoticed as it did with identity-only stubs.
        registered_expectations[kwargs["name"]] = dict(
            getattr(function, "_test_expectations", {})
        )
        return function

    return register


spec = importlib.util.spec_from_file_location(
    "translation", Path(__file__).resolve().parents[1] / "translation.py"
)
translation = importlib.util.module_from_spec(spec)
with patch.object(
    pyspark,
    "pipelines",
    SimpleNamespace(
        expect_all=_expect_all,
        materialized_view=_decorator,
    ),
    create=True,
):
    spec.loader.exec_module(translation)


class TranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("translation-integrity-tests")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "1")
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_shared_dataset_registers_translation_expectations(self):
        rules = registered_expectations[
            "addressresolution.silver.providers_translation_results"
        ]
        self.assertEqual(
            rules.get("translation_checks_passed"), "NOT translation_needs_review"
        )
        for field in ("street", "city", "state"):
            self.assertIn(f"no_residual_non_latin_{field}", rules)

    def test_preparation_uses_whole_field_country_specific_lookups(self):
        cases = [
            ("cn", " 北京 ", "广东省", " CHN ", "Beijing", "Guangdong"),
            ("kr", "서울", "경기도", "KR", "Seoul", "Gyeonggi"),
            ("jp", "板橋区", "東京都", "Japan", "Itabashi-ku", "Tokyo"),
            ("wrong_country", "北京", "北京市", "Japan", "北京", "北京市"),
            ("unknown", "北京", "北京市", None, "北京", "北京市"),
            ("partial", "北京市附近", None, "China", "北京市附近", None),
            ("null", None, None, "China", None, None),
        ]
        source = self.spark.createDataFrame(
            [
                (key, "北京 Street 01", city, state, country)
                for key, city, state, country, _, _ in cases
            ],
            "id STRING, street_address STRING, city STRING, state STRING, country STRING",
        )
        rows = {r.id: r for r in translation._prepare_translation(source).collect()}
        for key, city, state, country, expected_city, expected_state in cases:
            with self.subTest(case=key):
                row = rows[key]
                self.assertEqual((row.city, row.state), (expected_city, expected_state))
                self.assertEqual((row.city_original, row.state_original), (city, state))
                self.assertEqual(row.street_address, "北京 Street 01")
                self.assertEqual(row.country, country)

    def test_chinese_building_identifiers_are_preserved_and_failures_quarantined(self):
        source = "刘街B座, Suite 654"
        cases = [
            ("valid", source, "Liu Street Building B, Suite 654", False),
            ("lost", source, "Liu Street, Suite 654", True),
            ("changed", source, "Liu Street Building C, Suite 654", True),
            ("unlabeled", source, "Liu Street B, Suite 654", True),
            ("wrong_role", source, "Liu Street Unit B, Suite 654", True),
            ("block", source, "Liu Street Block B, Suite 654", False),
            ("case_alias", "张路g座", "Zhang Road Bldg. G", False),
            ("space", "张路G\u3000座", "Zhang Road Building G", False),
            ("compound", "刘街B-02座", "Liu Street Building B-02", False),
            ("numeric", "刘街02座", "Liu Street Building 02", False),
            ("mixed_label", "刘街 Building B 座", "Liu Street Building B", False),
            (
                "mixed_compound_label",
                "刘街 Building B-02 座",
                "Liu Street Building B-02",
                False,
            ),
            (
                "mixed_slash_label",
                "刘街 Building B/02 座",
                "Liu Street Building B/02",
                False,
            ),
            (
                "mixed_numeric_label",
                "刘街 Building 02-03 座",
                "Liu Street Building 02-03",
                False,
            ),
            (
                "mixed_compound_duplicate_lost",
                "刘街 Building B-02 座、B-02座",
                "Liu Street Building B-02",
                True,
            ),
            (
                "mixed_compound_partial_added",
                "刘街 Building B-02 座",
                "Liu Street Building B Building B-02",
                True,
            ),
            (
                "building_added",
                "458 岡田 Street",
                "458 Okada Street Building G",
                True,
            ),
            ("unit_added", source, "Liu Street Building B, Suite 654, Unit C", True),
            ("suite_added", "458 岡田 Street", "458 Okada Street Suite C", True),
            ("apartment_added", "458 岡田 Street", "458 Okada Street Apt. C", True),
            ("floor_added", "458 岡田 Street", "458 Okada Street Floor C", True),
            (
                "numeric_label_added",
                "458 岡田 Street",
                "Okada Street Unit 458",
                True,
            ),
            (
                "russian_building",
                "алл. Механизаторов, д. 674 к. 6/4",
                "674 Mekhanizatorov Alley Building 6/4",
                False,
            ),
            (
                "russian_structure",
                "алл. Дальняя, д. 2 стр. 73",
                "2 Dalnyaya Alley Building 73",
                False,
            ),
            (
                "russian_building_changed",
                "алл. Механизаторов, д. 674 к. 6/4",
                "6/4 Mekhanizatorov Alley Building 674",
                True,
            ),
            (
                "russian_building_full_marker",
                "ул. Дальняя, д. 2 корпус 73",
                "2 Dalnyaya Street Building 73",
                False,
            ),
            (
                "ambiguous_russian_marker",
                "ул. Дальняя, к. G",
                "Dalnyaya Street Building G",
                True,
            ),
            ("duplicate_lost", "刘街B座、B座", "Liu Street Building B", True),
            (
                "mixed_labels",
                "刘街B座, Building G, Suite 654",
                "Building G Building B Liu Street Suite 654",
                False,
            ),
            (
                "english_building_lost",
                "刘街B座, Building G, Suite 654",
                "Liu Street Building B Suite 654",
                True,
            ),
        ]
        df = self.spark.createDataFrame(
            [
                (key, original, original, (response, None))
                for key, original, response, _ in cases
            ],
            "id STRING, street_address STRING, street_address_original STRING, "
            "__response STRUCT<response:STRING,errorMessage:STRING>",
        )
        results = translation._apply_ai_response(
            df, "street_address", "__response", result_field="response"
        ).withColumns(
            {
                f"{field}_translation_issues": F.array().cast("array<string>")
                for field in ("city", "state")
            }
        )
        results = translation._add_translation_review_flags(results)
        rows = {row.id: row for row in results.collect()}
        for key, original, response, failed in cases:
            with self.subTest(case=key):
                row = rows[key]
                self.assertEqual(row.street_address, original if failed else response)
                self.assertEqual(row.street_address_original, original)
                self.assertEqual(row.street_address_ai_response, response)
                self.assertEqual(row.translation_needs_review, failed)
                self.assertEqual(
                    row.street_address_translation_status,
                    "invalid_response" if failed else "ai_translated",
                )
                expected_reasons = (
                    ["street_address:address_labels_changed"] if failed else []
                )
                if key == "mixed_compound_duplicate_lost":
                    expected_reasons.insert(0, "street_address:numbers_changed")
                self.assertEqual(row.translation_review_reasons, expected_reasons)

        stored_results = self.spark.createDataFrame(list(rows.values()), results.schema)
        with (
            patch.object(translation, "spark", self.spark, create=True),
            patch.object(
                self.spark.read.__class__, "table", return_value=stored_results
            ),
        ):
            accepted = {row.id for row in translation.providers_translated().collect()}
            review = {
                row.id for row in translation.providers_translation_review().collect()
            }
        self.assertEqual(accepted, {key for key, _, _, failed in cases if not failed})
        self.assertEqual(review, {key for key, _, _, failed in cases if failed})

    def test_acceptance_fallback_and_diagnostics(self):
        source = "07851 عمير Stravenue Suite 750 Building G"
        valid = "07851 Umair Stravenue Suite 750 Building G"
        cases = [
            (
                "no_letters",
                "刘路",
                "刘路",
                "...",
                None,
                ["no_letters"],
                "invalid_response",
            ),
            ("valid", source, source, valid, None, [], "ai_translated"),
            (
                "trimmed",
                source,
                source,
                "\u3000" + valid + "\u00a0",
                None,
                [],
                "ai_translated",
            ),
            (
                "reordered",
                source,
                source,
                "Building G Suite 750 07851 Umair Stravenue",
                None,
                [],
                "ai_translated",
            ),
            (
                "aliases",
                source,
                source,
                "07851 Umair Stravenue Ste. 750 Bldg. G",
                None,
                [],
                "ai_translated",
            ),
            (
                "zero_lost",
                source,
                source,
                valid.replace("07851", "7851"),
                None,
                ["numbers_changed"],
                "invalid_response",
            ),
            (
                "number_lost",
                "10 刘路 Suite 10",
                "10 刘路 Suite 10",
                "10 Liu Road",
                None,
                ["numbers_changed", "address_labels_changed"],
                "invalid_response",
            ),
            (
                "number_added",
                "刘路",
                "刘路",
                "99 Liu Road",
                None,
                ["numbers_changed"],
                "invalid_response",
            ),
            (
                "compound_changed",
                "701-3 刘路",
                "701-3 刘路",
                "3-701 Liu Road",
                None,
                ["numbers_changed"],
                "invalid_response",
            ),
            (
                "label_lost",
                source,
                source,
                valid.replace("Building G", ""),
                None,
                ["address_labels_changed"],
                "invalid_response",
            ),
            (
                "label_changed",
                source,
                source,
                valid.replace("Building G", "Building H"),
                None,
                ["address_labels_changed"],
                "invalid_response",
            ),
            (
                "non_latin",
                "刘路",
                "刘路",
                "刘 Road",
                None,
                ["non_latin_response"],
                "invalid_response",
            ),
            (
                "quoted",
                "刘路",
                "刘路",
                '"Liu Road"',
                None,
                ["unexpected_format"],
                "invalid_response",
            ),
            *[
                (
                    key,
                    "458 岡田 Street",
                    "458 岡田 Street",
                    response,
                    None,
                    ["unexpected_format"],
                    "invalid_response",
                )
                for key, response in (
                    ("single_quoted", "'458 Okada Street'"),
                    ("curly_single_quoted", "‘458 Okada Street’"),
                    ("english_prefix", "English translation: 458 Okada Street"),
                    ("translation_prefix", "Translation: 458 Okada Street"),
                    (
                        "translated_address_prefix",
                        "Translated address: 458 Okada Street",
                    ),
                    (
                        "prefix_case_and_spaces",
                        "ENGLISH\u3000TRANSLATION : 458 Okada Street",
                    ),
                )
            ],
            (
                "internal_apostrophe",
                "458 西安 Road",
                "458 西安 Road",
                "458 Xi'an Road",
                None,
                [],
                "ai_translated",
            ),
            (
                "internal_curly_apostrophe",
                "458 西安 Road",
                "458 西安 Road",
                "458 Xi’an Road",
                None,
                [],
                "ai_translated",
            ),
            (
                "translation_street_name",
                "458 翻译 Street",
                "458 翻译 Street",
                "458 Translation Street",
                None,
                [],
                "ai_translated",
            ),
            (
                "multiline",
                "刘路",
                "刘路",
                "Liu\nRoad",
                None,
                ["unexpected_format"],
                "invalid_response",
            ),
            (
                "fenced",
                "刘路",
                "刘路",
                "\x60\x60\x60Liu Road\x60\x60\x60",
                None,
                ["unexpected_format"],
                "invalid_response",
            ),
            (
                "null_response",
                "刘路",
                "刘路",
                None,
                None,
                ["empty_response"],
                "invalid_response",
            ),
            (
                "blank",
                "刘路",
                "刘路",
                "\u3000 \t\u00a0",
                None,
                ["empty_response"],
                "invalid_response",
            ),
            (
                "endpoint_error",
                "刘路",
                "刘路",
                None,
                "endpoint unavailable",
                ["ai_error"],
                "ai_error",
            ),
            ("null_input", None, None, None, None, [], "unchanged"),
            (
                "latin_input",
                "10 Main Street",
                "10 Main Street",
                None,
                None,
                [],
                "unchanged",
            ),
            ("lookup", "Tokyo", "東京都", None, None, [], "lookup"),
        ]
        # Exercise both explicitly configured success-field contracts.
        for result_field in ("result", "response"):
            with self.subTest(result_field=result_field):
                schema = T.StructType(
                    [
                        T.StructField("id", T.StringType()),
                        T.StructField("street_address", T.StringType()),
                        T.StructField("street_address_original", T.StringType()),
                        T.StructField(
                            "__response",
                            T.StructType(
                                [
                                    T.StructField(result_field, T.StringType()),
                                    T.StructField("errorMessage", T.StringType()),
                                ]
                            ),
                        ),
                    ]
                )
                df = self.spark.createDataFrame(
                    [
                        (key, original, saved, (response, error))
                        for key, original, saved, response, error, _, _ in cases
                    ],
                    schema,
                )
                output = translation._apply_ai_response(
                    df, "street_address", "__response", result_field=result_field
                )
                self.assertNotIn("__response", output.columns)
                rows = {row.id: row for row in output.collect()}
                self.assertEqual(len(rows), len(cases))
                for key, original, saved, response, error, issues, status in cases:
                    row = rows[key]
                    with self.subTest(case=key):
                        self.assertEqual(row.street_address_translation_issues, issues)
                        self.assertEqual(row.street_address_translation_status, status)
                        self.assertEqual(row.street_address_ai_error, error)
                        self.assertEqual(row.street_address_ai_response, response)
                        self.assertEqual(row.street_address_original, saved)
                        expected = (
                            response.strip() if status == "ai_translated" else original
                        )
                        self.assertEqual(row.street_address, expected)

    def test_ai_translation_builds_plan_without_schema_analysis(self):
        source = self.spark.createDataFrame(
            [("key", "street_address", "prompt")],
            "request_key STRING, field_name STRING, request_prompt STRING",
        )

        def forbid_schema_analysis(_df):
            raise AssertionError("Pipeline queries must not inspect DataFrame.schema")

        for result_field in ("response", "result"):
            with self.subTest(result_field=result_field):
                response = F.struct(
                    F.lit("Liu Road").alias(result_field),
                    F.lit(None).cast("string").alias("errorMessage"),
                )
                with (
                    patch.object(F, "expr", return_value=response),
                    patch.object(
                        type(source), "schema", property(forbid_schema_analysis)
                    ),
                ):
                    output = translation._infer_requests(source)
                row = output.collect()[0]
                self.assertEqual(row.ai_result[result_field], "Liu Road")
                self.assertIsNone(row.ai_result.errorMessage)
                self.assertEqual(row.request_key, "key")

    def test_distinct_requests_preserve_context_and_invalidate_changed_configuration(
        self,
    ):
        source = self.spark.createDataFrame(
            [
                ("刘路", "未知城", "未知城", "China"),
                ("刘路", "未知城", "未知城", "China"),
                ("刘路", "未知城", None, "Japan"),
                ("刘路", None, None, None),
                ("刘路", None, None, None),
                ("10 Main Street", "北京", "北京市", "China"),
                (None, None, None, None),
            ],
            "street_address STRING, city STRING, state STRING, country STRING",
        )
        prepared = translation._prepare_translation(source)
        requests = translation._translation_requests(prepared).collect()
        self.assertEqual(len(requests), 6)
        self.assertEqual(sum(r.source_field_count for r in requests), 10)
        self.assertEqual(len({r.request_key for r in requests}), 6)
        self.assertFalse(
            any(r.input_text in ("北京", "北京市", "10 Main Street") for r in requests)
        )
        by_context = {(r.country_context, r.field_name): r for r in requests}
        self.assertEqual(
            by_context[("unknown", "street_address")].source_field_count, 2
        )
        self.assertNotEqual(
            by_context[("China", "city")].request_key,
            by_context[("China", "state")].request_key,
        )
        self.assertNotEqual(
            by_context[("China", "city")].request_key,
            by_context[("Japan", "city")].request_key,
        )
        original_keys = {r.request_key for r in requests}
        for setting, value in (
            ("AI_MODEL", "different-model"),
            ("PROMPT_VERSION", "v2"),
            ("TRANSLATION_PROMPT", "Different instructions: "),
        ):
            with (
                self.subTest(setting=setting),
                patch.object(translation, setting, value),
            ):
                changed = translation._translation_requests(
                    translation._prepare_translation(source)
                )
                self.assertTrue(
                    original_keys.isdisjoint(r.request_key for r in changed.collect())
                )
        empty = translation._translation_requests(
            translation._prepare_translation(source.limit(0))
        )
        self.assertEqual(empty.collect(), [])

    def test_shared_responses_preserve_duplicate_rows_and_missing_results_fail_closed(
        self,
    ):
        source = self.spark.createDataFrame(
            [
                ("same", "刘路", None, None, None),
                ("same", "刘路", None, None, None),
                ("missing", "未知路", None, None, "CN"),
                ("plain", "Main Street", None, None, "US"),
            ],
            "id STRING, street_address STRING, city STRING, state STRING, country STRING",
        )
        prepared = translation._prepare_translation(source)
        requests = translation._translation_requests(prepared)
        self.assertEqual(requests.count(), 2)
        raw = requests.where(F.col("input_text") == "刘路").withColumn(
            "ai_result",
            F.struct(
                F.lit("Liu Road").alias(translation.AI_RESPONSE_FIELD),
                F.lit(None).cast("string").alias("errorMessage"),
            ),
        )
        rows = translation._validate_translation(prepared, raw).collect()
        self.assertEqual(len(rows), 4)
        duplicates = [r for r in rows if r.id == "same"]
        self.assertEqual(len(duplicates), 2)
        self.assertTrue(
            all(
                r.street_address == "Liu Road" and not r.translation_needs_review
                for r in duplicates
            )
        )
        missing = next(r for r in rows if r.id == "missing")
        self.assertEqual(missing.street_address, "未知路")
        self.assertEqual(
            missing.translation_review_reasons, ["street_address:empty_response"]
        )
        self.assertFalse(
            next(r for r in rows if r.id == "plain").translation_needs_review
        )
        self.assertFalse(any(name.startswith("__") for name in rows[0].asDict()))

    def test_pipeline_preserves_rows_and_builds_review_flags(self):
        source = self.spark.createDataFrame(
            [
                ("good", "458 岡田 Street", "板橋区", "東京都", "Japan"),
                ("bad", "07851 عمير Stravenue", "راتبville", "SC", "SA"),
                ("quality_bad", "10 刘路", "板橋区", "東京都", "Japan"),
                ("format_bad", "459 岡田 Street", "板橋区", "東京都", "Japan"),
                ("label_bad", "460 岡田 Street", "板橋区", "東京都", "Japan"),
                ("compound_good", "刘街 Building B-02 座", "北京", "北京市", "China"),
                ("plain", "10 Main Street", None, None, "US"),
            ],
            ["id", "street_address", "city", "state", "country"],
        )

        def fake_inference(requests):
            response = (
                F.when(F.col("input_text") == "458 岡田 Street", "458 Okada Street")
                .when(F.col("input_text") == "10 刘路", "99 Liu Road")
                .when(
                    F.col("input_text") == "459 岡田 Street",
                    "English translation: 459 Okada Street",
                )
                .when(
                    F.col("input_text") == "460 岡田 Street",
                    "460 Okada Street Building G",
                )
                .when(
                    F.col("input_text") == "刘街 Building B-02 座",
                    "Liu Street Building B-02",
                )
                .when(F.col("input_text") == "راتبville", "Ratibville")
            )
            error = F.when(
                F.col("input_text") == "07851 عمير Stravenue", "endpoint unavailable"
            )
            return requests.withColumn(
                "ai_result",
                F.struct(
                    response.alias(translation.AI_RESPONSE_FIELD),
                    error.alias("errorMessage"),
                ),
            )

        # Execute the registered pipeline stages with actual materialization
        # boundaries. Only the Databricks endpoint invocation is substituted.
        tables = {"addressresolution.silver.providers_merged": source}
        with (
            patch.object(translation, "spark", self.spark, create=True),
            patch.object(
                self.spark.read.__class__,
                "table",
                side_effect=lambda name: tables[name],
            ),
            patch.object(translation, "_infer_requests", side_effect=fake_inference),
        ):
            for stage in ("prepared", "requests", "raw", "results"):
                result_df = getattr(translation, f"providers_translation_{stage}")()
                stored_rows = result_df.collect()
                tables[f"addressresolution.silver.providers_translation_{stage}"] = (
                    self.spark.createDataFrame(stored_rows, result_df.schema)
                )
        rows = {row.id: row for row in stored_rows}
        self.assertEqual(
            set(rows),
            {
                "good",
                "bad",
                "plain",
                "quality_bad",
                "format_bad",
                "label_bad",
                "compound_good",
            },
        )
        self.assertEqual(
            rows["good"].full_address, "458 Okada Street, Itabashi-ku, Tokyo, Japan"
        )
        self.assertEqual(rows["good"].city_original, "板橋区")
        self.assertEqual(rows["good"].city_translation_status, "lookup")
        self.assertFalse(rows["good"].translation_needs_review)
        self.assertEqual(rows["good"].translation_review_reasons, [])
        self.assertTrue(rows["bad"].translation_needs_review)
        self.assertEqual(
            rows["bad"].translation_review_reasons, ["street_address:ai_error"]
        )
        self.assertEqual(rows["bad"].street_address, "07851 عمير Stravenue")
        self.assertEqual(rows["bad"].city, "Ratibville")
        self.assertFalse(rows["plain"].translation_needs_review)
        self.assertEqual(rows["plain"].full_address, "10 Main Street, US")
        self.assertTrue(rows["quality_bad"].translation_needs_review)
        self.assertEqual(
            rows["quality_bad"].translation_review_reasons,
            ["street_address:numbers_changed"],
        )
        self.assertIsNone(rows["quality_bad"].street_address_ai_error)
        self.assertEqual(rows["quality_bad"].street_address, "10 刘路")
        for key, issue in (
            ("format_bad", "unexpected_format"),
            ("label_bad", "address_labels_changed"),
        ):
            with self.subTest(case=key):
                self.assertTrue(rows[key].translation_needs_review)
                self.assertEqual(
                    rows[key].translation_review_reasons, [f"street_address:{issue}"]
                )
                number = "459" if key == "format_bad" else "460"
                self.assertEqual(rows[key].street_address, f"{number} 岡田 Street")
                self.assertEqual(
                    rows[key].full_address,
                    f"{number} 岡田 Street, Itabashi-ku, Tokyo, Japan",
                )
        self.assertFalse(rows["compound_good"].translation_needs_review)
        self.assertEqual(rows["compound_good"].translation_review_reasons, [])
        self.assertEqual(
            rows["compound_good"].full_address,
            "Liu Street Building B-02, Beijing, Beijing, China",
        )

        # Simulate the stored shared stage, so both branches reuse its responses.
        stored_results = self.spark.createDataFrame(stored_rows, result_df.schema)
        with (
            patch.object(translation, "spark", self.spark, create=True),
            patch.object(
                self.spark.read.__class__, "table", return_value=stored_results
            ) as read_table,
            patch.object(
                translation,
                "_infer_requests",
                side_effect=AssertionError("Output branches must not invoke AI"),
            ),
        ):
            accepted = {
                row.id: row for row in translation.providers_translated().collect()
            }
            review = {
                row.id: row
                for row in translation.providers_translation_review().collect()
            }
        self.assertEqual(set(accepted), {"good", "plain", "compound_good"})
        self.assertEqual(set(review), {"bad", "quality_bad", "format_bad", "label_bad"})
        self.assertTrue(set(accepted).isdisjoint(review))
        self.assertEqual(accepted | review, rows)
        self.assertEqual(read_table.call_count, 2)
        for call in read_table.call_args_list:
            self.assertEqual(
                call.args,
                ("addressresolution.silver.providers_translation_results",),
            )

    def test_routing_preserves_duplicates_and_handles_empty_or_unknown_flags(self):
        cases = [
            [],
            [("accepted", False, "preserved"), ("accepted", False, "preserved")],
            [("failed", True, "endpoint error"), ("unknown", None, "missing flag")],
        ]
        schema = "id STRING, translation_needs_review BOOLEAN, evidence STRING"
        for source_rows in cases:
            with self.subTest(source_rows=source_rows):
                source = self.spark.createDataFrame(source_rows, schema)
                with (
                    patch.object(translation, "spark", self.spark, create=True),
                    patch.object(
                        self.spark.read.__class__, "table", return_value=source
                    ),
                ):
                    accepted = translation.providers_translated().collect()
                    review = translation.providers_translation_review().collect()
                self.assertEqual(
                    Counter(tuple(row) for row in accepted + review),
                    Counter(source_rows),
                )
                self.assertTrue(
                    all(row.translation_needs_review is False for row in accepted)
                )
                self.assertTrue(
                    all(row.translation_needs_review is not False for row in review)
                )


if __name__ == "__main__":
    unittest.main()
