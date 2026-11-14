"""Address judge requests and responses for same-company pairs."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import address_judge  # noqa: E402
from lib.matching import prepare_record  # noqa: E402
from lib.resolution import JUDGE_THRESHOLD, pair_key  # noqa: E402


def source(provider_id, street, **overrides):
    row = dict(
        provider="provider_a",
        provider_id=provider_id,
        company="Acme Ltd",
        street_address=street,
        city="Ko Samui",
        state="Mukdahan",
        country="TH",
    )
    return prepare_record(dict(row, **overrides))


class AddressJudgeTests(unittest.TestCase):
    def setUp(self):
        self.left = source("1", "63/44 Thiwar Road Floor 25")
        self.right = source(
            "2",
            "63/44 Theewor Road, floor 25",
            translation_evidence_json=json.dumps(
                {"street_address_original": "63/44 ถนนธีวร, floor 25"}
            ),
        )

    def test_request_is_the_same_whichever_record_comes_first(self):
        (forward,) = address_judge.requests([(self.left, self.right)])
        (backward,) = address_judge.requests([(self.right, self.left)])
        self.assertEqual(forward, backward)
        self.assertEqual(
            (forward["id_l"], forward["id_r"]),
            pair_key(self.left["unique_id"], self.right["unique_id"]),
        )
        self.assertEqual(
            forward["request_key"], address_judge.request_key(forward["prompt"])
        )

    def test_prompt_shows_address_text_and_the_original_only(self):
        text = address_judge.prompt(self.left, self.right)
        self.assertIn("63/44 Thiwar Road Floor 25", text)
        self.assertIn("63/44 ถนนธีวร, floor 25", text)
        self.assertEqual(text.count("street_address_original"), 1)
        # The model never sees the company, ids or the pipeline's scores.
        for hidden in ("Acme", "provider", self.left["unique_id"]):
            self.assertNotIn(hidden, text)

    def test_request_key_changes_with_the_request(self):
        other = source("3", "63/44 Thewor Road, floor 25")
        (first,) = address_judge.requests([(self.left, self.right)])
        (second,) = address_judge.requests([(self.left, other)])
        self.assertNotEqual(first["request_key"], second["request_key"])

    def test_response_parsing_accepts_only_a_complete_judgment(self):
        good = address_judge.parse_response(
            '{"same_address": true, "confidence": 0.95, "reason": " Same name. "}'
        )
        self.assertEqual(
            good,
            dict(
                same_address=True,
                confidence=0.95,
                reason="Same name.",
                model=address_judge.MODEL,
            ),
        )
        for bad in (
            None,
            "",
            "yes",
            "[]",
            '{"same_address": "true", "confidence": 0.9}',
            '{"same_address": true}',
            '{"same_address": true, "confidence": 1.4}',
            '{"same_address": true, "confidence": true}',
        ):
            self.assertIsNone(address_judge.parse_response(bad), bad)

    def test_unusable_responses_give_no_judgment(self):
        rows = [
            dict(
                id_l="b",
                id_r="a",
                response='{"same_address": false, "confidence": 0.9}',
            ),
            dict(id_l="c", id_r="d", response=""),
        ]
        result = address_judge.judgments(rows)
        self.assertEqual(list(result), [("a", "b")])
        self.assertFalse(result["a", "b"]["same_address"])

    def test_query_names_the_model_and_keeps_failures_as_rows(self):
        sql = address_judge.ai_query_sql("prompt")
        self.assertIn(f"ai_query('{address_judge.MODEL}', prompt", sql)
        self.assertIn("failOnError => false", sql)
        self.assertIn("responseFormat", sql)
        self.assertGreaterEqual(JUDGE_THRESHOLD, 0.9)


class StreetOnlyRequestTests(unittest.TestCase):
    def test_street_only_request_hides_the_city_and_has_its_own_key(self):
        left = source("1", "57 Mus'af Creek")
        right = source("2", "57 Musaeef Creek", city="")
        (full,) = address_judge.requests([(left, right)])
        (street,) = address_judge.requests([(left, right)], street_only=True)
        self.assertNotEqual(full["request_key"], street["request_key"])
        self.assertIn(address_judge.STREET_ONLY, street["prompt"])
        self.assertNotIn("Ko Samui", street["prompt"])
        self.assertNotIn(address_judge.STREET_ONLY, full["prompt"])


if __name__ == "__main__":
    unittest.main()
