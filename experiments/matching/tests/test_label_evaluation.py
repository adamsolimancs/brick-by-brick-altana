"""Evaluation contracts, weighting, ties, leakage, and unfilled templates."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.label_evaluation import (  # noqa: E402
    confusion,
    evaluate,
    grouped_folds,
    join_labels,
    select_threshold,
    weighted_auc,
)


def fixture():
    fields, analysis = [], []
    for i in range(12):
        ref = f"P{i:03d}"
        item = dict(
            pair_ref=ref,
            evaluation_group=f"g{i // 2}",
            sampling_weight=2,
            sample_focus="company_jw_only",
        )
        for field in ("company", "street_name", "city"):
            fields.append(
                dict(
                    pair_ref=ref,
                    pair_id=f"id{i}",
                    field=field,
                    value_a=f"a{i}",
                    value_b=f"b{i}",
                    label="",
                    reviewer="",
                )
            )
            item[field + "_jaro_winkler_similarity"] = 0.9 if i % 2 else 0.1
            item[field + "_levenshtein_similarity"] = 0.1 if i % 2 else 0.9
        analysis.append(item)
    return dict(fields=fields, analysis=analysis)


def labeled(sample):
    rows = copy.deepcopy(sample["fields"])
    for row in rows:
        row["label"] = "same" if int(row["pair_ref"][1:]) % 2 else "different"
        row["reviewer"] = "Independent test reviewer"
    return rows


class LabelEvaluationTests(unittest.TestCase):
    def test_model_reference_and_unweighted_mode_are_explicit(self):
        sample = fixture()
        labels = labeled(sample)
        for row in labels:
            row["reviewer_type"] = "model"
            row["human_verified"] = "false"
        result = evaluate(sample, labels, n_folds=3, unweighted=True)
        self.assertEqual(result["status"], "model_review_pilot")
        self.assertEqual(result["human_verified_judgments"], 0)
        self.assertEqual(result["reviewer_types"], {"model": 36})
        self.assertEqual(result["weighting"], "unweighted_labeled_subset")
        self.assertEqual(
            result["fields"]["company"]["metrics"]["jaro_winkler"][
                "weighted_out_of_fold"
            ]["tp"],
            6,
        )

    def test_blank_sample_produces_no_thresholds_or_accuracy(self):
        sample = fixture()
        result = evaluate(sample, sample["fields"])
        self.assertEqual(result["status"], "labels_required")
        self.assertEqual(result["label_counts"]["company"], {"blank": 12})
        for field in result["fields"].values():
            for metric in field["metrics"].values():
                self.assertIsNone(metric["weighted_auc_all_labeled"])
                self.assertIsNone(metric["weighted_out_of_fold"])
                self.assertIsNone(metric["development_threshold_all_labels"])

    def test_identity_source_values_reviewers_and_adjudication_validated(self):
        sample = fixture()
        for name, value in [
            ("pair_id", "stale"),
            ("value_a", "changed"),
            ("label", "maybe"),
            ("reviewer", ""),
        ]:
            rows = labeled(sample)
            rows[0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                join_labels(rows, sample)
        rows = labeled(sample)
        with self.assertRaisesRegex(ValueError, "Duplicate judgment"):
            join_labels(rows + [rows[0]], sample)

    def test_groups_are_deterministic_balanced_and_shared_across_fields(self):
        sample = fixture()
        folds = grouped_folds(sample["analysis"], 3)
        self.assertEqual(set(folds.values()), {0, 1, 2})
        self.assertEqual(folds, grouped_folds(list(reversed(sample["analysis"])), 3))
        self.assertEqual(len(folds), 6)

    def test_threshold_selection_uses_weights_and_ties_atomically(self):
        rows = [
            dict(label="same", weight=1, score=0.9),
            dict(label="different", weight=9, score=0.9),
            dict(label="same", weight=10, score=0.5),
        ]
        self.assertIsNone(select_threshold(rows, "score", 0.9))
        self.assertEqual(select_threshold(rows, "score", 0.5), 0.5)
        stats = confusion(rows, "score", 0.5)
        self.assertAlmostEqual(stats["precision"], 11 / 20)
        self.assertAlmostEqual(stats["recall"], 1)
        self.assertEqual(
            weighted_auc(
                [
                    dict(label="same", weight=1, s=0.8),
                    dict(label="different", weight=9, s=0.8),
                ],
                "s",
            ),
            0.5,
        )

    def test_bad_metric_abstains_instead_of_dropping_test_cases(self):
        sample = fixture()
        result = evaluate(sample, labeled(sample), n_folds=3)
        for field in result["fields"].values():
            jw = field["metrics"]["jaro_winkler"]
            lev = field["metrics"]["levenshtein"]
            self.assertEqual(jw["out_of_fold_n"], 12)
            self.assertEqual(lev["out_of_fold_n"], 12)
            self.assertEqual(jw["weighted_out_of_fold"]["recall"], 1)
            self.assertEqual(jw["weighted_out_of_fold"]["precision"], 1)
            self.assertEqual(lev["weighted_out_of_fold"]["recall"], 0)
            self.assertIsNone(lev["weighted_out_of_fold"]["precision"])
            self.assertEqual(jw["weighted_auc_all_labeled"], 1)
            self.assertEqual(lev["weighted_auc_all_labeled"], 0)

    def test_heldout_labels_cannot_change_own_fold_threshold(self):
        sample = fixture()
        labels = labeled(sample)
        first = evaluate(sample, labels, n_folds=3)
        folds = grouped_folds(sample["analysis"], 3)
        groups = {r["pair_ref"]: r["evaluation_group"] for r in sample["analysis"]}
        for row in labels:
            if folds[groups[row["pair_ref"]]] == 0:
                row["label"] = "different" if row["label"] == "same" else "same"
        second = evaluate(sample, labels, n_folds=3)
        for field in first["fields"]:
            before = first["fields"][field]["fold_details"][0]
            after = second["fields"][field]["fold_details"][0]
            self.assertEqual(before["thresholds"], after["thresholds"])

    def test_uncertain_is_excluded_and_missing_rows_count_as_blank(self):
        sample = fixture()
        rows = labeled(sample)[1:]
        rows[0]["label"] = "uncertain"
        result = evaluate(sample, rows, n_folds=3)
        self.assertEqual(result["label_counts"]["company"]["blank"], 1)
        self.assertEqual(result["label_counts"]["street_name"]["uncertain"], 1)
        self.assertEqual(result["fields"]["street_name"]["binary_labels"], 11)


if __name__ == "__main__":
    unittest.main()
