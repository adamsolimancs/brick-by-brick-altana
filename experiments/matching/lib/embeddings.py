"""Bounded, review-only embedding retrieval over a completed matching snapshot.

The encoder runs locally on job compute. Only model files are downloaded; address
text is not sent to an external inference API. Cosines are not probabilities.
"""

import json

import numpy as np
import pyarrow as pa

from lib.matching import digest

VERSION = "multilingual_hnsw_v1"
MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
DIMENSION = 384
VIEWS = ("record", "address")
MODEL_KEY = digest([VERSION, MODEL, REVISION, DIMENSION])


def embedding_text(record, view):
    """Keep building/unit digits and Unicode; never abbreviate or truncate text."""
    address = " | ".join(
        f"{label}: {record[key]}"
        for label, key in (
            ("street", "street_key"),
            ("city", "city_key"),
            ("region", "region_key"),
            ("country", "country_key"),
        )
    )
    if view == "address":
        return address
    if view == "record":
        return f"company: {record['name_key']} | {address}"
    raise ValueError(f"Unknown embedding view: {view}")


def normalized_vectors(vectors):
    array = np.asarray(vectors, dtype="float32")
    if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
        raise ValueError("Embeddings must be a nonempty finite 2D array")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero embeddings cannot be searched with cosine similarity")
    return np.ascontiguousarray(array / norms)


def encode_texts(model, texts, batch_size=64):
    """Fail on model truncation rather than silently losing address/unit evidence."""
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        tokens = model.tokenizer(batch, truncation=False, padding=False)["input_ids"]
        if any(len(ids) > model.max_seq_length for ids in tokens):
            raise ValueError(
                f"Embedding text exceeds model token limit {model.max_seq_length}; "
                "no text was silently truncated. Inspect the input snapshot."
            )
    return normalized_vectors(
        model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
    )


def build_index(vectors, m=32, ef_construction=200, ef_search=200):
    import faiss

    vectors = normalized_vectors(vectors)
    index = faiss.IndexHNSWFlat(vectors.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    # Stable row ordering and single-thread construction make rebuilds reproducible.
    faiss.omp_set_num_threads(1)
    index.add(vectors)
    faiss.omp_set_num_threads(4)
    return index


def initialize_retrieval(connection):
    """Caller registers records and known_pairs(id_l,id_r) from the same base run."""
    connection.execute("""CREATE TABLE embedding_passes (
        id_l VARCHAR, id_r VARCHAR, view VARCHAR, cosine DOUBLE,
        neighbor_rank BIGINT, query_id VARCHAR
    )""")


def retrieve(
    connection,
    ids,
    vectors,
    index,
    view,
    top_k=20,
    search_k=100,
    min_cosine=0.70,
    max_pairs=2_000_000,
    batch_size=256,
):
    """Overfetch before excluding *all* old candidates, then cap new neighbors.

    The search budget is explicit. Shortfalls are audited rather than represented
    as exhaustive recall; the exact-search audit also measures novel-neighbor loss.
    """
    if len(ids) != len(vectors) or len(ids) != len(set(ids)):
        raise ValueError("Vector rows must map one-to-one to unique source IDs")
    if not 1 <= top_k <= search_k or not -1 <= min_cosine <= 1:
        raise ValueError("Invalid neighbor budget or cosine threshold")
    if index.ntotal != len(ids):
        raise ValueError("HNSW index and source snapshot sizes differ")
    count = min(len(ids), search_k + 1)
    index.hnsw.efSearch = max(index.hnsw.efSearch, count)
    audit = {
        "view": view,
        "queries": len(ids),
        "known_neighbor_hits": 0,
        "selected_directed_neighbors": 0,
        "queries_below_top_k": 0,
        "search_budget_saturated_queries": 0,
    }
    for offset in range(0, len(ids), batch_size):
        batch = normalized_vectors(vectors[offset : offset + batch_size])
        scores, labels = index.search(batch, count)
        rows = []
        for local, (neighbors, similarities) in enumerate(zip(labels, scores)):
            query = offset + local
            eligible = 0
            for rank, (neighbor, score) in enumerate(zip(neighbors, similarities), 1):
                if neighbor < 0 or neighbor == query:
                    continue
                if score < min_cosine:
                    continue
                eligible += 1
                a, b = sorted((ids[query], ids[int(neighbor)]))
                rows.append((a, b, ids[query], float(np.clip(score, -1, 1)), rank))
            if count < len(ids) and eligible >= search_k:
                audit["search_budget_saturated_queries"] += 1
        schema = pa.schema(
            [
                ("id_l", pa.string()),
                ("id_r", pa.string()),
                ("query_id", pa.string()),
                ("cosine", pa.float64()),
                ("neighbor_rank", pa.int64()),
            ]
        )
        connection.register(
            "neighbor_batch",
            pa.Table.from_pylist(
                [dict(zip(schema.names, row)) for row in rows], schema=schema
            ),
        )
        audit["known_neighbor_hits"] += connection.execute("""
            SELECT count(*) FROM neighbor_batch JOIN known_pairs USING (id_l,id_r)
        """).fetchone()[0]
        connection.execute("""CREATE OR REPLACE TEMP TABLE novel_batch AS
            SELECT n.*, row_number() OVER (
                PARTITION BY query_id ORDER BY cosine DESC, id_l, id_r
            ) new_rank FROM neighbor_batch n
            ANTI JOIN known_pairs k USING (id_l,id_r)
        """)
        selected, covered = connection.execute(
            """
            SELECT count(*), count(DISTINCT query_id) FROM novel_batch WHERE new_rank <= ?
        """,
            [top_k],
        ).fetchone()
        full = connection.execute(
            """
            SELECT count(*) FROM (SELECT query_id FROM novel_batch
                GROUP BY query_id HAVING count(*) >= ?)
        """,
            [top_k],
        ).fetchone()[0]
        audit["selected_directed_neighbors"] += selected
        audit["queries_below_top_k"] += len(batch) - full
        assert covered <= len(batch)
        existing = connection.execute(
            "SELECT count(*) FROM embedding_passes"
        ).fetchone()[0]
        if existing + selected > max_pairs:
            raise ValueError(
                "Embedding candidate-pass budget exceeded; no silent sampling"
            )
        connection.execute(
            """INSERT INTO embedding_passes
            SELECT id_l,id_r,?,cosine,neighbor_rank,query_id
            FROM novel_batch WHERE new_rank <= ?""",
            [view, top_k],
        )
    return audit


def exact_search_audit(
    connection,
    ids,
    vectors,
    index,
    view,
    top_k=20,
    min_cosine=0.70,
    sample_size=128,
    seed=20261030,
):
    """Tie-aware geometric recall after old-pair exclusion, not matching accuracy."""
    vectors = normalized_vectors(vectors)
    rng = np.random.default_rng(seed)
    sample = sorted(rng.choice(len(ids), min(sample_size, len(ids)), replace=False))
    positions = {key: i for i, key in enumerate(ids)}
    credit = expected = 0
    for query in sample:
        scores = vectors @ vectors[query]
        scores[query] = -np.inf
        known = connection.execute(
            """SELECT CASE WHEN id_l=? THEN id_r ELSE id_l END
            FROM known_pairs WHERE id_l=? OR id_r=?""",
            [ids[query]] * 3,
        ).fetchall()
        for (key,) in known:
            scores[positions[key]] = -np.inf
        eligible = scores[scores >= min_cosine]
        count = min(top_k, len(eligible))
        if not count:
            continue
        cutoff = np.partition(eligible, -count)[-count]
        retrieved = connection.execute(
            """SELECT id_l,id_r FROM embedding_passes
            WHERE view=? AND query_id=?""",
            [view, ids[query]],
        ).fetchall()
        neighbors = {b if a == ids[query] else a for a, b in retrieved}
        hits = sum(1 for key in neighbors if scores[positions[key]] >= cutoff - 1e-5)
        credit += min(count, hits)
        expected += count
    return {
        "view": view,
        "sample_queries": len(sample),
        "exact_novel_neighbors": expected,
        "retrieved_exact_novel_neighbors": credit,
        "novel_neighbor_recall_at_k": credit / expected if expected else None,
        "tie_tolerance": 1e-5,
        "seed": seed,
        "interpretation": "vector retrieval recall only; true-match recall unknown",
    }


def finalize_candidates(connection):
    connection.execute("""CREATE TABLE embedding_candidates AS
        WITH pairs AS (
            SELECT id_l,id_r,
                string_agg(DISTINCT view, ',' ORDER BY view) retrieval_views,
                max(CASE WHEN view='record' THEN cosine END) record_cosine,
                max(CASE WHEN view='address' THEN cosine END) address_cosine,
                max(cosine) retrieval_score, min(neighbor_rank) best_neighbor_rank
            FROM embedding_passes GROUP BY id_l,id_r
        ), evidence AS (
            SELECT p.*, l.content_hash content_hash_l, r.content_hash content_hash_r,
                l.provider provider_l, r.provider provider_r,
                l.provider_id provider_id_l, r.provider_id provider_id_r,
                l.company company_l, r.company company_r,
                l.street_address street_address_l, r.street_address street_address_r,
                l.city city_l, r.city city_r, l.state state_l, r.state state_r,
                l.country country_l, r.country country_r,
                l.input_route input_route_l, r.input_route input_route_r,
                l.country_key <> r.country_key country_conflict,
                l.region_key <> r.region_key region_conflict,
                l.house_key <> '' AND r.house_key <> '' AND l.house_key <> r.house_key house_conflict,
                l.unit_key <> '' AND r.unit_key <> '' AND l.unit_key <> r.unit_key unit_conflict,
                (l.unit_key = '') <> (r.unit_key = '') unit_missing_one_side,
                l.number_signature <> r.number_signature numeric_conflict,
                jaro_winkler_similarity(l.name_key,r.name_key) name_similarity,
                jaro_winkler_similarity(l.street_key,r.street_key) street_similarity,
                jaro_winkler_similarity(l.city_key,r.city_key) city_similarity
            FROM pairs p JOIN records l ON p.id_l=l.unique_id
                JOIN records r ON p.id_r=r.unique_id
        ) SELECT *, CASE WHEN country_conflict OR region_conflict OR house_conflict
            OR unit_conflict OR numeric_conflict THEN 'review_conflict'
            ELSE 'review_embedding_candidate' END decision,
            'cosine_similarity_not_probability' score_type, false approved_for_merge
        FROM evidence
    """)
    invalid = connection.execute("""SELECT count(*) FROM embedding_candidates
        WHERE id_l >= id_r OR approved_for_merge OR NOT isfinite(retrieval_score)
    """).fetchone()[0]
    overlap = connection.execute("""SELECT count(*) FROM embedding_candidates
        JOIN known_pairs USING(id_l,id_r)""").fetchone()[0]
    if invalid or overlap:
        raise AssertionError("New candidate identity/exclusion contract failed")
    return dict(
        connection.execute("""SELECT decision,count(*) FROM embedding_candidates
        GROUP BY decision""").fetchall()
    )


def json_text(value):
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported manifest value: {type(item).__name__}")

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, default=scalar, allow_nan=False
    )
