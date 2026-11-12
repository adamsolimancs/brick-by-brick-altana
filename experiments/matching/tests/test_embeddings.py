"""Real FAISS HNSW + DuckDB retrieval, exclusion, guards, and recall diagnostics."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import duckdb
import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.embeddings import (  # noqa: E402
    build_index,
    embedding_text,
    encode_texts,
    exact_search_audit,
    finalize_candidates,
    initialize_retrieval,
    json_text,
    normalized_vectors,
    retrieve,
)
from lib.matching import register_records  # noqa: E402


def record(i, **changes):
    return {
        "provider": "fixture",
        "provider_id": str(i),
        "company": "Acme LLC",
        "street_address": "123 Main St Suite 4",
        "city": "Testville",
        "state": "New York",
        "country": "US",
        "input_route": "clean",
        **changes,
    }


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.con = duckdb.connect(":memory:")
        self.con.execute("CREATE TABLE known_pairs(id_l VARCHAR,id_r VARCHAR)")
        initialize_retrieval(self.con)

    def tearDown(self):
        self.con.close()

    def test_hnsw_to_review_candidates_excludes_every_old_pair(self):
        prepared = register_records(
            self.con,
            [
                record(1),
                record(2),
                record(3, company="Acme Incorporated"),
                record(4, street_address="128 Main St Suite 4"),
                record(5, street_address="123 Main St Suite 9"),
                record(6, country="CA"),
                record(7, street_address="123 Main St"),
            ],
        )
        ids = [r["unique_id"] for r in prepared]
        old_pairs = [tuple(sorted((ids[0], ids[1]))), tuple(sorted((ids[1], ids[2])))]
        self.con.executemany("INSERT INTO known_pairs VALUES (?,?)", old_pairs)
        vectors = normalized_vectors(
            np.random.default_rng(3).normal(size=(len(ids), 16))
        )
        index = build_index(vectors)
        for view in ("record", "address"):
            audit = retrieve(
                self.con, ids, vectors, index, view, top_k=6, search_k=6, min_cosine=-1
            )
            self.assertEqual(audit["known_neighbor_hits"], 4)
            recall = exact_search_audit(
                self.con, ids, vectors, index, view, top_k=6, min_cosine=-1
            )
            self.assertEqual(recall["novel_neighbor_recall_at_k"], 1)
        counts = finalize_candidates(self.con)
        candidates = (
            self.con.execute("SELECT * FROM embedding_candidates")
            .fetch_arrow_table()
            .to_pylist()
        )
        self.assertEqual(sum(counts.values()), 19)  # C(7,2) minus the 2 known pairs
        self.assertEqual(
            len(candidates), len({(r["id_l"], r["id_r"]) for r in candidates})
        )
        self.assertTrue(
            all(
                r["id_l"] < r["id_r"] and not r["approved_for_merge"]
                for r in candidates
            )
        )
        self.assertTrue(
            all(r["retrieval_views"] == "address,record" for r in candidates)
        )
        self.assertFalse(set(old_pairs) & {(r["id_l"], r["id_r"]) for r in candidates})

        def pair(a, b):
            return next(
                r
                for r in candidates
                if {r["id_l"], r["id_r"]} == {ids[a - 1], ids[b - 1]}
            )

        self.assertEqual(pair(1, 3)["decision"], "review_embedding_candidate")
        for a, b, flag in (
            (1, 4, "house_conflict"),
            (1, 5, "unit_conflict"),
            (1, 6, "country_conflict"),
        ):
            self.assertTrue(pair(a, b)[flag])
            self.assertEqual(pair(a, b)["decision"], "review_conflict")
        self.assertTrue(pair(1, 7)["unit_missing_one_side"])
        self.assertEqual(pair(1, 7)["score_type"], "cosine_similarity_not_probability")

    def test_search_budget_crowding_is_measured_against_exact_novel_neighbors(self):
        ids = [str(i) for i in range(8)]
        vectors = normalized_vectors([[1, i / 20] for i in range(8)])
        # Query 0's nearest neighbor is already scored. A too-small overfetch
        # budget loses unseen neighbors; the audit must reveal that loss.
        self.con.execute("INSERT INTO known_pairs VALUES ('0','1')")
        index = build_index(vectors)
        audit = retrieve(
            self.con, ids, vectors, index, "record", top_k=1, search_k=1, min_cosine=-1
        )
        recall = exact_search_audit(
            self.con, ids, vectors, index, "record", top_k=1, min_cosine=-1
        )
        self.assertGreater(audit["queries_below_top_k"], 0)
        self.assertLess(recall["novel_neighbor_recall_at_k"], 1)

    def test_partial_recall_audit_is_json_serializable(self):
        ids = [str(i) for i in range(8)]
        vectors = normalized_vectors([[1, i / 20] for i in range(8)])
        self.con.execute("INSERT INTO known_pairs VALUES ('0','1')")
        index = build_index(vectors)
        retrieve(
            self.con, ids, vectors, index, "record", top_k=2, search_k=2, min_cosine=-1
        )
        audit = exact_search_audit(
            self.con, ids, vectors, index, "record", top_k=2, min_cosine=-1
        )
        self.assertLess(audit["novel_neighbor_recall_at_k"], 1)
        # Real partial hits previously promoted the count to np.int64 and broke
        # the final Delta completion manifest after all candidate tables wrote.
        json.dumps(audit)
        decoded = json.loads(
            json_text({"audit": audit, "count": np.int64(3), "flag": np.bool_(True)})
        )
        self.assertEqual(decoded["audit"], audit)
        self.assertEqual(decoded["count"], 3)
        self.assertIs(decoded["flag"], True)
        with self.assertRaises(ValueError):
            json_text({"invalid_score": float("nan")})

    def test_ties_small_corpus_empty_outputs_and_saved_index(self):
        prepared = register_records(self.con, [record(1), record(2), record(3)])
        ids = [r["unique_id"] for r in prepared]
        vectors = normalized_vectors(np.ones((3, 4)))
        index = build_index(vectors)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "index.faiss")
            faiss.write_index(index, path)
            index = faiss.read_index(path)
        retrieve(self.con, ids, vectors, index, "address", top_k=1, search_k=100)
        audit = exact_search_audit(self.con, ids, vectors, index, "address", top_k=1)
        self.assertEqual(audit["novel_neighbor_recall_at_k"], 1)
        self.con.execute("DELETE FROM embedding_passes")
        self.assertEqual(finalize_candidates(self.con), {})
        self.assertGreater(
            len(self.con.execute("SELECT * FROM embedding_candidates").description), 10
        )

    def test_candidate_budget_fails_and_bad_vectors_are_rejected(self):
        vectors = np.eye(3, dtype="float32")
        index = build_index(vectors)
        with self.assertRaisesRegex(ValueError, "budget"):
            retrieve(
                self.con,
                ["a", "b", "c"],
                vectors,
                index,
                "record",
                top_k=2,
                search_k=2,
                min_cosine=-1,
                max_pairs=1,
            )
        for bad in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], []):
            with self.assertRaises(ValueError):
                normalized_vectors(bad)

    def test_text_views_keep_numbers_units_unicode_and_refuse_truncation(self):
        from lib.matching import prepare_record

        prepared = prepare_record(
            record(1, company="Компания", street_address="123 Main St Suite 9")
        )
        self.assertIn("компания", embedding_text(prepared, "record"))
        self.assertNotIn("компания", embedding_text(prepared, "address"))
        self.assertIn("123 main st unit 9", embedding_text(prepared, "address"))
        with self.assertRaises(ValueError):
            embedding_text(prepared, "invalid")

        class ShortModel:
            max_seq_length = 2

            def tokenizer(self, batch, **kwargs):
                self.tokenizer_kwargs = kwargs
                return {"input_ids": [[1, 2, 3] for _ in batch]}

            def encode(self, *args, **kwargs):
                raise AssertionError("Truncated inputs must never reach inference")

        model = ShortModel()
        with self.assertRaisesRegex(ValueError, "token limit"):
            encode_texts(model, ["too many tokens"])
        self.assertFalse(model.tokenizer_kwargs["truncation"])


if __name__ == "__main__":
    unittest.main()
