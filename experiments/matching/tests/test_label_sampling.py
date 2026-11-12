"""Allocation, score blinding, side mapping, and validation grouping."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.label_sampling import allocation, make_exports, stratum_quotas  # noqa: E402


class LabelSamplingTests(unittest.TestCase):
    def test_allocation_covers_cells_and_respects_small_populations(self):
        populations = {"a": 2, "b": 100, "c": 100}
        selected = allocation(populations, 15)
        self.assertEqual(sum(selected.values()), 15)
        self.assertEqual(selected["a"], 2)
        self.assertTrue(all(1 <= selected[k] <= populations[k] for k in selected))
        with self.assertRaises(ValueError):
            allocation(populations, 2)
        with self.assertRaises(ValueError):
            allocation(populations, 203)

    def test_group_and_stratum_budgets_reconstruct_the_population(self):
        strata = [
            dict(
                candidate_origin="blocked",
                decision="proposed_match",
                sample_focus=f"cell{i}",
                population_n=n,
            )
            for i, n in enumerate((1, 100, 300))
        ]
        strata += [
            dict(
                candidate_origin="embedding",
                decision="review_conflict",
                sample_focus="cell0",
                population_n=1000,
            )
        ]
        quotas = stratum_quotas(strata, 30)
        self.assertEqual(sum(quotas.values()), 30)
        self.assertEqual(quotas[("blocked", "proposed_match", "cell0")], 1)
        reconstructed = sum(
            quotas[(r["candidate_origin"], r["decision"], r["sample_focus"])]
            * int(r["population_n"])
            / quotas[(r["candidate_origin"], r["decision"], r["sample_focus"])]
            for r in strata
        )
        self.assertAlmostEqual(reconstructed, 1401)

    def test_blind_templates_keep_scores_out_and_preserve_side_identity(self):
        rows = []
        for i in range(2):
            row = dict(
                id_l="a",
                id_r=f"b{i}",
                content_hash_l="hash-a",
                content_hash_r=f"hash-b{i}",
                candidate_origin="embedding",
                decision="review_conflict",
                sample_focus="company_jw_only",
                company_l="Company A",
                company_r=f"Company B{i}",
                street_address_l="123 Main St",
                street_address_r="124 Main St",
                city_l="A City",
                city_r="B City",
                state_l="NY",
                state_r="NJ",
                country_l="US",
                country_r="US",
                company_jaro_winkler_similarity=0.9,
                company_levenshtein_similarity=0.6,
                translation_evidence_json_l=json.dumps(
                    {"company_original": "株式会社"}
                ),
                translation_evidence_json_r="{}",
            )
            for field in ("company", "street_name", "city"):
                row[f"comparison_{field}_l"] = f"{field} a"
                row[f"comparison_{field}_r"] = f"{field} b{i}"
            rows.append(row)
        strata = [
            dict(
                candidate_origin="embedding",
                decision="review_conflict",
                sample_focus="company_jw_only",
                population_n=100,
            )
        ]
        exported = make_exports(
            rows,
            strata,
            {("embedding", "review_conflict", "company_jw_only"): 2},
            "source",
        )
        self.assertEqual(len(exported["fields"]), 6)
        self.assertEqual(len({r["evaluation_group"] for r in exported["analysis"]}), 1)
        self.assertAlmostEqual(
            sum(r["sampling_weight"] for r in exported["analysis"]), 100
        )
        for pair in exported["pairs"]:
            self.assertEqual(pair["overall_label"], "")
            self.assertNotIn("decision", pair)
            self.assertNotIn("company_jaro_winkler_similarity", pair)
            self.assertIn("株式会社", pair["record_a"] + pair["record_b"])
        for field in exported["fields"]:
            self.assertEqual(field["label"], "")
            metadata = next(
                r for r in exported["analysis"] if r["pair_id"] == field["pair_id"]
            )
            side = "l" if metadata["display_a_unique_id"] == metadata["id_l"] else "r"
            self.assertEqual(
                field["value_a"], metadata[f"comparison_{field['field']}_{side}"]
            )


if __name__ == "__main__":
    unittest.main()
