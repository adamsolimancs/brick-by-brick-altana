"""Model-labeled pair benchmark: blinded prompts, strict parsing, weighted scoring."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.llm_labeling import (  # noqa: E402
    LABELS,
    MERGED_DECISIONS,
    POSITIVE_LABEL,
    evaluate,
    insert_statements,
    pair_prompt,
    parse_response,
    response_schema,
    stratum_quotas,
)


def record(**changes):
    base = {
        "provider": "B",
        "provider_id": "2276",
        "company": "Fujii Foods Co., Ltd.",
        "street_address": "aleja Ludowa 446",
        "city": "Zielona Góra",
        "state": "Zachodniopomorskie",
        "country": "PL",
        "country_inferred": "false",
        "translation_evidence_json": json.dumps(
            {
                "company_original": "有限会社藤井食品",
                "street_address_original": "aleja Ludowa 446",
                "company_translation_status": "ai_translated",
                "street_address_translation_status": "unchanged",
            }
        ),
    }
    return {**base, **changes}


def pair(**changes):
    base = {
        "id_l": "aaa",
        "id_r": "bbb",
        "l": record(),
        "r": record(provider="A", provider_id="91", company="Fujii Foods Ltd"),
        "fs_decision": "proposed_match",
        "fs_match_probability": 0.998765,
        "candidate_origin": "blocked",
    }
    return {**base, **changes}


class SchemaAndParsingTests(unittest.TestCase):
    def test_response_schema_is_strict_and_enumerates_project_labels(self):
        schema = response_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(sorted(schema["required"]), sorted(schema["properties"]))
        self.assertEqual(tuple(schema["properties"]["label"]["enum"]), LABELS)
        self.assertIn("same_company_same_location", LABELS)
        self.assertIn("insufficient_evidence", LABELS)

    def test_parse_response_accepts_valid_and_rejects_unknown_labels(self):
        parsed = parse_response(
            json.dumps(
                {
                    "label": "different_location",
                    "confidence": 0.8,
                    "reason": "Different house numbers on the same street.",
                }
            )
        )
        self.assertEqual(parsed["label"], "different_location")
        for bad in (
            {"label": "maybe", "confidence": 0.5, "reason": "x"},
            {"label": "different_location", "confidence": 1.5, "reason": "x"},
            {"label": "different_location", "reason": "x"},
            "not json",
        ):
            with self.assertRaises(ValueError):
                parse_response(bad if isinstance(bad, str) else json.dumps(bad))


class PromptTests(unittest.TestCase):
    def test_prompt_shows_both_records_and_original_text_but_no_scores(self):
        text, _ = pair_prompt(pair())
        for value in (
            "Fujii Foods Co., Ltd.",
            "Fujii Foods Ltd",
            "aleja Ludowa 446",
            "有限会社藤井食品",
            "Zielona Góra",
            "PL",
        ):
            self.assertIn(value, text)
        for leak in ("proposed_match", "0.998", "blocked", "probability", "decision"):
            self.assertNotIn(leak, text)

    def test_orientation_is_seeded_and_varies_across_pairs(self):
        first, swapped_first = pair_prompt(pair())
        again, swapped_again = pair_prompt(pair())
        self.assertEqual((first, swapped_first), (again, swapped_again))
        swaps = {pair_prompt(pair(id_l=f"id{i}", id_r=f"id{i}z"))[1] for i in range(40)}
        self.assertEqual(swaps, {True, False})
        if swapped_first:
            self.assertLess(
                text_index(first, "Fujii Foods Ltd"), text_index(first, "Co., Ltd.")
            )
        else:
            self.assertLess(
                text_index(first, "Co., Ltd."), text_index(first, "Fujii Foods Ltd")
            )

    def test_inferred_country_is_marked_as_a_pipeline_guess(self):
        text, _ = pair_prompt(pair(l=record(country_inferred="true")))
        self.assertIn("inferred", text.lower())


def text_index(text, needle):
    index = text.find(needle)
    assert index >= 0, needle
    return index


class SamplingTests(unittest.TestCase):
    def test_quotas_cover_every_stratum_and_sum_to_size(self):
        populations = {
            ("blocked", "exact_baseline"): 5548,
            ("blocked", "proposed_match"): 6198,
            ("embedding", "proposed_match"): 3,
            ("embedding", "review_conflict"): 367878,
            ("unblocked", "never_compared"): 20,
        }
        quotas = stratum_quotas(populations, 100)
        self.assertEqual(sum(quotas.values()), 100)
        self.assertEqual(quotas[("embedding", "proposed_match")], 3)
        self.assertTrue(all(0 < quotas[k] <= populations[k] for k in populations))
        self.assertGreater(
            quotas[("blocked", "proposed_match")], quotas[("blocked", "exact_baseline")]
        )


class EvaluationTests(unittest.TestCase):
    def labeled(self, decision, label, origin="blocked", population=10, sample=5):
        return {
            "candidate_origin": origin,
            "fs_decision": decision,
            "label": label,
            "population_n": population,
            "sample_n": sample,
            "id_l": "x",
            "id_r": "y",
        }

    def test_weighted_precision_and_recall_of_pipeline_merges(self):
        rows = [
            # proposed matches: 3 agree, 1 model says different (weight 2 each)
            *[self.labeled("proposed_match", POSITIVE_LABEL) for _ in range(3)],
            self.labeled("proposed_match", "different_location"),
            # review bucket with a missed duplicate (weight 100 each)
            self.labeled("review_conflict", POSITIVE_LABEL, population=1000, sample=10),
            self.labeled(
                "review_conflict", "different_company", population=1000, sample=10
            ),
            # insufficient evidence is reported, not scored
            self.labeled("proposed_match", "insufficient_evidence"),
        ]
        result = evaluate(rows)
        self.assertEqual(result["scored_pairs"], 6)
        self.assertEqual(result["unscored_insufficient_evidence"], 1)
        self.assertAlmostEqual(result["unweighted"]["precision"], 3 / 4)
        self.assertAlmostEqual(result["unweighted"]["recall"], 3 / 4)
        self.assertAlmostEqual(result["weighted"]["precision"], 3 / 4)
        self.assertAlmostEqual(result["weighted"]["recall"], 6 / (6 + 100))
        self.assertEqual(len(result["disagreements"]), 2)
        self.assertEqual(
            {d["kind"] for d in result["disagreements"]},
            {"merged_but_model_disagrees", "not_merged_but_model_says_same"},
        )
        self.assertIn("proposed_match", MERGED_DECISIONS)


class InsertTests(unittest.TestCase):
    def test_insert_statements_escape_quotes_and_chunk_rows(self):
        rows = [
            {"a": "it's", "b": 1.5, "c": None, "d": True},
            {"a": "plain", "b": 2, "c": "x", "d": False},
            {"a": "third", "b": 3, "c": "y", "d": True},
        ]
        statements = insert_statements("cat.sch.tbl", rows, chunk=2)
        self.assertEqual(len(statements), 2)
        self.assertTrue(
            statements[0].startswith("INSERT INTO cat.sch.tbl (a, b, c, d) VALUES")
        )
        self.assertIn("'it\\'s'", statements[0])
        self.assertIn("NULL", statements[0])
        self.assertIn("true", statements[0])
        self.assertIn("'third'", statements[1])

    def test_insert_statements_escape_backslashes(self):
        # Backslash escapes the next character in Databricks SQL, so an
        # unescaped backslash before a quote would close the literal early.
        statements = insert_statements("cat.sch.tbl", [{"a": "x\\' OR 1=1 --"}])
        self.assertIn("('x\\\\\\' OR 1=1 --')", statements[0])


if __name__ == "__main__":
    unittest.main()


class OrchestrationTests(unittest.TestCase):
    def test_label_pairs_retries_invalid_json_once_then_fails_loudly(self):
        from lib.llm_labeling import label_pairs

        calls = []

        def judge(system, user, schema):
            calls.append(user)
            if len(calls) == 1:
                return "garbage"
            return json.dumps(
                {"label": "different_company", "confidence": 0.9, "reason": "ok"}
            )

        rows = label_pairs([pair()], judge)
        self.assertEqual(len(calls), 2)
        self.assertEqual(rows[0]["label"], "different_company")
        self.assertIn("orientation_swapped", rows[0])
        self.assertEqual(rows[0]["raw_response"], calls and rows[0]["raw_response"])

        def always_bad(system, user, schema):
            return "garbage"

        with self.assertRaises(ValueError):
            label_pairs([pair()], always_bad)

    def test_sample_sql_is_parameterized_and_validates_identifiers(self):
        from lib.llm_labeling import sample_sql

        quotas = {("blocked", "proposed_match"): 5, ("unblocked", "never_compared"): 2}
        sql = sample_sql("addressresolution.matching", "run_1", "src_1", quotas)
        self.assertIn("'blocked'", sql)
        self.assertIn("'proposed_match'", sql)
        self.assertIn("never_compared", sql)
        self.assertIn("sha2(", sql)
        self.assertIn("run_1", sql)
        with self.assertRaises(ValueError):
            sample_sql("addressresolution.matching", "bad id; drop", "src_1", quotas)


class ReportTests(unittest.TestCase):
    def test_report_markdown_states_metrics_and_lists_disagreements(self):
        from lib.llm_labeling import report_markdown

        result = evaluate(
            [
                {
                    "candidate_origin": "blocked",
                    "fs_decision": "proposed_match",
                    "label": "different_location",
                    "population_n": 10,
                    "sample_n": 5,
                    "id_l": "x",
                    "id_r": "y",
                    "reason": "House numbers differ.",
                },
                {
                    "candidate_origin": "blocked",
                    "fs_decision": "exact_baseline",
                    "label": POSITIVE_LABEL,
                    "population_n": 10,
                    "sample_n": 5,
                    "id_l": "p",
                    "id_r": "q",
                    "reason": "Identical.",
                },
            ]
        )
        text = report_markdown(result, {"model": "gpt-6.1-sol", "size": 2})
        self.assertIn("gpt-6.1-sol", text)
        self.assertIn("Precision", text)
        self.assertIn("50.0%", text)
        self.assertIn("House numbers differ.", text)
        self.assertIn("not ground truth", text)

    def test_strata_sql_shares_the_candidate_definition_with_sampling(self):
        from lib.llm_labeling import sample_sql, strata_sql

        strata = strata_sql("addressresolution.matching", "run_1", "src_1")
        sample = sample_sql(
            "addressresolution.matching",
            "run_1",
            "src_1",
            {("blocked", "proposed_match"): 1},
        )
        self.assertIn("never_compared", strata)
        self.assertIn("GROUP BY", strata)
        self.assertEqual(
            strata.split("), candidates AS")[0], sample.split("), candidates AS")[0]
        )
