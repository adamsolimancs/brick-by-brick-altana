"""Regression tests for false transitive merges and lost deduplication evidence."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.resolution import resolve_records  # noqa: E402


def record(record_id, baseline=None, **overrides):
    return dict(
        unique_id=record_id,
        baseline_id=baseline or record_id,
        company="Acme",
        street_address="12 Main St",
        city="Boston",
        state="MA",
        country="US",
        country_inferred="false",
        **overrides,
    )


def pair(left, right, decision="proposed_match", **overrides):
    values = dict(
        id_l=left,
        id_r=right,
        fs_decision=decision,
        fs_match_probability=0.999,
        candidate_origin="blocked",
        fs_guard_conflict=False,
        fs_guard_review=False,
        fs_uses_untrained_level=False,
        company_jaro_winkler_similarity=0.95,
        street_name_jaro_winkler_similarity=0.95,
        city_jaro_winkler_similarity=1.0,
    )
    return dict(values, **overrides)


class ResolutionTests(unittest.TestCase):
    def test_terminal_address_conflict_does_not_trigger_match_review(self):
        result = resolve_records(
            [record("a"), record("b")],
            [
                pair("a", "b", "non_match_address_conflict", fs_guard_conflict=True),
            ],
        )
        self.assertEqual(len(result), 2)
        self.assertTrue(all(row["is_canonical"] for row in result))
        self.assertFalse(any(row["needs_match_review"] for row in result))

    def test_incompatible_third_record_is_not_absorbed_by_transitivity(self):
        result = resolve_records(
            [record(i) for i in "abc"],
            [
                pair("a", "b"),
                pair("b", "c"),
                pair("a", "c", "review_conflict", fs_guard_conflict=True),
            ],
        )
        rows = {row["record_id"]: row for row in result}
        self.assertEqual(rows["a"]["entity_id"], rows["b"]["entity_id"])
        self.assertNotEqual(rows["b"]["entity_id"], rows["c"]["entity_id"])
        self.assertTrue(rows["c"]["needs_match_review"])
        self.assertTrue(
            any(
                e["decision"] == "review_cluster_consistency"
                for e in rows["b"]["match_review_evidence"]
            )
        )

    def test_exact_groups_require_complete_direct_evidence(self):
        records = [record(i, "same") for i in "abc"]
        with self.assertRaisesRegex(ValueError, "missing scored pair"):
            resolve_records(records, [pair("a", "b", "exact_baseline")])
        rows = resolve_records(
            records,
            [
                pair("a", "b", "exact_baseline"),
                pair("a", "c", "exact_baseline"),
                pair("b", "c", "exact_baseline"),
            ],
        )
        self.assertEqual(sum(r["is_duplicate"] for r in rows), 2)
        self.assertTrue(all(r["deduplication_evidence"] for r in rows))
        self.assertEqual({r["merge_authority"] for r in rows}, {"exact_rule"})

    def test_every_duplicate_has_direct_canonical_proof_and_stage(self):
        rows = resolve_records(
            [record("a"), record("b")], [pair("a", "b", candidate_origin="embedding")]
        )
        duplicate = next(row for row in rows if row["is_duplicate"])
        self.assertEqual(
            duplicate["deduplication_evidence"][0]["other_record_id"],
            duplicate["canonical_record_id"],
        )
        self.assertEqual(duplicate["deduplication_steps"], ["fellegi_sunter_hnsw"])
        self.assertEqual(duplicate["merge_authority"], "model_proposal_policy")
        self.assertIn("0.99", duplicate["deduplication_reasons"][0])

    def test_below_threshold_and_guarded_pairs_are_not_merged(self):
        for change in (
            {"fs_match_probability": 0.98},
            {"fs_guard_conflict": True},
            {"fs_guard_review": True},
            {"fs_uses_untrained_level": True},
            {"company_jaro_winkler_similarity": 0.5},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "safeguards"),
            ):
                resolve_records([record("a"), record("b")], [pair("a", "b", **change)])
        rows = resolve_records(
            [record("a"), record("b")],
            [pair("a", "b", "review_uncertain", fs_match_probability=0.9)],
        )
        self.assertTrue(
            all(row["is_canonical"] and row["needs_match_review"] for row in rows)
        )

    def test_selection_and_group_ids_are_deterministic(self):
        records = [record("a"), record("b")]
        records[0]["country_inferred"] = "true"
        pairs = [
            pair(
                "a", "b", fs_corrected_country="US", fs_country_corrected_record_id="a"
            )
        ]
        first = sorted(resolve_records(records, pairs), key=lambda r: r["record_id"])
        second = sorted(
            resolve_records(list(reversed(records)), pairs),
            key=lambda r: r["record_id"],
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0]["canonical_record_id"], "b")
        self.assertIn("inferred country", first[0]["deduplication_reasons"][0])

    def test_singletons_are_conserved(self):
        rows = resolve_records([record(i) for i in "abc"], [])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["deduplication_status"] == "singleton" for row in rows))
        self.assertTrue(all(row["merge_authority"] == "none" for row in rows))

    def test_unknown_country_provenance_does_not_outrank_supplied_country(self):
        records = [record("a"), record("b")]
        records[0]["country_inferred"] = None
        rows = resolve_records(records, [pair("a", "b")])
        self.assertTrue(all(row["canonical_record_id"] == "b" for row in rows))

    def test_original_text_agreement_is_a_distinct_accepted_proof(self):
        agreed = pair(
            "a",
            "b",
            fs_match_probability=0.008,
            street_name_jaro_winkler_similarity=0.68,
            company_jaro_winkler_similarity=1.0,
            fs_original_text_agreement=True,
            original_company_agreement=False,
            original_street_name_agreement=True,
            original_city_agreement=False,
        )
        rows = resolve_records([record("a"), record("b")], [agreed])
        duplicate = next(row for row in rows if row["is_duplicate"])
        self.assertEqual(duplicate["deduplication_steps"], ["original_text_agreement"])
        self.assertEqual(duplicate["merge_authority"], "model_proposal_policy")
        self.assertIn("before translation", duplicate["deduplication_reasons"][0])
        for change in (
            {"original_street_name_agreement": False},
            {"company_jaro_winkler_similarity": 0.9},
            {"fs_guard_review": True},
            {"fs_original_text_agreement": False},
        ):
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(ValueError, "safeguards"),
            ):
                resolve_records([record("a"), record("b")], [dict(agreed, **change)])


if __name__ == "__main__":
    unittest.main()
