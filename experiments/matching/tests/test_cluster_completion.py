"""Real Spark + Splink completion through final canonical selection."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.cluster_completion import (
    missing_cluster_pairs,
    completion_features,
    score_completion,
)
from lib.matching import prepare_record
from lib.string_features import prepare_comparison_records
from lib.unified_matching import add_decisions, route_address_conflicts
from lib.resolution import resolve_records
from test_unified_matching import fixed_settings


class CompletionPlanTests(unittest.TestCase):
    def test_only_missing_edges_and_no_unrelated_records(self):
        missing, skipped = missing_cluster_pairs(
            "abcd", [("a", "b"), ("b", "c")], [("a", "b"), ("b", "c")]
        )
        self.assertEqual(missing, [("a", "c")])
        self.assertEqual(skipped, [])
        # An existing negative a-c must never be rescored.
        self.assertEqual(
            missing_cluster_pairs(
                "abc", [("a", "b"), ("b", "c")], [("a", "b"), ("b", "c"), ("a", "c")]
            )[0],
            [],
        )

    def test_whole_component_budget_and_determinism(self):
        edges = [("a", "b"), ("b", "c")]
        for kwargs, reason in [
            ({"max_group_size": 2}, "group_size_budget"),
            ({"max_new_pairs": 0}, "new_pair_budget"),
        ]:
            missing, skipped = missing_cluster_pairs("abc", edges, edges, **kwargs)
            self.assertEqual(missing, [])
            self.assertEqual(skipped[0]["reason"], reason)
        self.assertEqual(
            missing_cluster_pairs("cba", list(reversed(edges)), edges),
            missing_cluster_pairs("abc", edges, edges),
        )
        with self.assertRaisesRegex(ValueError, "unknown record"):
            missing_cluster_pairs("ab", [("a", "z")], [])


class CompletionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyspark.sql import SparkSession

        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
        os.environ["PYSPARK_PYTHON"] = sys.executable
        cls.previous_pythonpath = os.environ.get("PYTHONPATH")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            filter(
                None,
                (
                    str(Path(__file__).resolve().parents[1]),
                    cls.previous_pythonpath,
                ),
            )
        )
        cls.spark = (
            SparkSession.builder.master("local[2]")
            .appName("cluster-completion-test")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.executorEnv.PYTHONPATH", os.environ["PYTHONPATH"])
            .getOrCreate()
        )
        cls.spark.sparkContext.setLogLevel("ERROR")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()
        if cls.previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = cls.previous_pythonpath

    def test_missing_edge_scored_with_same_model_and_conflict_cannot_merge(self):
        from pyspark.sql import functions as F

        # Three variants plus unrelated records: full population is needed for TF.
        rows = [
            prepare_record(
                dict(
                    provider="A",
                    provider_id=str(i),
                    company="Acme",
                    street_address=street,
                    city="Boston",
                    state="MA",
                    country="US",
                    country_inferred="false",
                )
            )
            for i, street in enumerate(
                ["12 Main St", "12 Main Street", "12 Main st.", "99 Main St"]
            )
        ]
        for row, key in zip(rows, "abcd"):
            row["unique_id"] = key
        for i in range(20):
            row = prepare_record(
                dict(
                    provider="B",
                    provider_id=str(i),
                    company=f"Other {i}",
                    street_address=f"{i + 100} Other Road",
                    city="Austin",
                    state="TX",
                    country="US",
                    country_inferred="false",
                )
            )
            row["unique_id"] = f"z{i:02}"
            rows.append(row)
        prepared = self.spark.createDataFrame(rows)
        components = prepare_comparison_records(prepared)
        # Materialize UDF result once to model the saved component table.
        components = self.spark.createDataFrame(components.collect(), components.schema)
        missing, _ = missing_cluster_pairs(
            [r["unique_id"] for r in rows],
            [("a", "b"), ("b", "c")],
            [("a", "b"), ("b", "c")],
        )
        self.assertEqual(missing, [("a", "c")])
        ids = [("a", "b"), ("b", "c"), *missing, ("a", "d")]
        features = completion_features(
            self.spark.createDataFrame(ids, "id_l string,id_r string"),
            prepared,
            components,
        )
        predictions = score_completion(
            self.spark, prepared, components, ids, fixed_settings()
        )
        training = dict(prior=0.001, unobserved_parameters=[], training_sessions=[])
        scored = route_address_conflicts(
            add_decisions(
                features.join(predictions, ["id_l", "id_r"]),
                training,
                pair_geography=True,
            )
        )
        observed = [r.asDict(recursive=True) for r in scored.collect()]
        by_pair = {(r["id_l"], r["id_r"]): r for r in observed}
        self.assertEqual(by_pair["a", "c"]["fs_decision"], "proposed_match")
        self.assertNotEqual(by_pair["a", "d"]["fs_decision"], "proposed_match")
        # Exact variants in this fixture need their actual exact decision.
        for r in rows:
            r["baseline_id"] = r["unique_id"]
        before = resolve_records(rows, [by_pair["a", "b"], by_pair["b", "c"]])
        after = resolve_records(rows, observed)
        self.assertEqual(sum(r["is_duplicate"] for r in before), 1)
        self.assertEqual(sum(r["is_duplicate"] for r in after), 2)
        removed = [r for r in after if r["is_duplicate"]]
        self.assertTrue(all(r["deduplication_evidence"] for r in removed))
        self.assertTrue(
            all(
                "fellegi_sunter_cluster_completion" in r["deduplication_steps"]
                for r in removed
            )
        )
        self.assertTrue(next(r for r in after if r["record_id"] == "d")["is_canonical"])
        # Check Unicode-aware model levels match the saved string comparator.
        from lib.unified_matching import THRESHOLDS, expected_gamma

        for field in THRESHOLDS:
            self.assertEqual(
                scored.where(
                    ~F.col("fs_gamma_" + field).eqNullSafe(expected_gamma(field))
                ).count(),
                0,
            )


if __name__ == "__main__":
    unittest.main()
