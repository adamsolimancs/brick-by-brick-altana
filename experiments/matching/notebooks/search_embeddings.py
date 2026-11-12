# Databricks notebook source
# MAGIC %md
# MAGIC # HNSW candidate expansion
# MAGIC Index every embedded record. Exclude every previously scored pair, retain
# MAGIC conflicts for review, and never turn cosine similarities into probabilities.

# COMMAND ----------

import hashlib
import importlib.metadata
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import faiss
import mlflow
import numpy as np
import pyarrow as pa
from pyspark.sql import functions as F

dbutils.widgets.text("library_root", "")
library_root = dbutils.widgets.get("library_root")
sys.path.insert(0, library_root)
from lib.embedding_runtime import append_result, require_fresh_outputs, settings  # noqa: E402
from lib.embeddings import (  # noqa: E402
    DIMENSION,
    MODEL_KEY,
    VIEWS,
    build_index,
    exact_search_audit,
    finalize_candidates,
    initialize_retrieval,
    json_text,
    normalized_vectors,
    retrieve,
)
from lib.matching import digest  # noqa: E402

config = settings(dbutils)
prefix = config["prefix"]
run_id = config["embedding_run_id"]
input_embedding_run_id = config["source_embedding_run_id"] or run_id
require_fresh_outputs(
    spark,
    [
        f"{prefix}.{table}"
        for table in (
            "embedding_candidate_passes",
            "embedding_candidates",
            "embedding_review_sample",
        )
    ],
    run_id,
)
stages = (
    spark.read.table(f"{prefix}.embedding_runs")
    .where(
        (F.col("embedding_run_id") == input_embedding_run_id)
        & (F.col("status") == "EMBEDDED")
    )
    .collect()
)
if len(stages) != 1 or stages[0]["status"] != "EMBEDDED":
    raise ValueError(
        "Exactly one successful embedding stage is required; start a new run if repairing"
    )
manifest = json.loads(stages[0]["manifest_json"])
ignored = {"embedding_run_id", "source_embedding_run_id"}
if {k: v for k, v in manifest["config"].items() if k not in ignored} != {
    k: v for k, v in config.items() if k not in ignored
} or manifest["model_key"] != MODEL_KEY:
    raise ValueError("Embedding/search configuration or model changed between tasks")
manifest["source_embedding_run_id"] = input_embedding_run_id
manifest["embedding_run_id"] = run_id
manifest["config"] = config
source_run_id = manifest["source_run_id"]
source = spark.read.table(f"{prefix}.prepared_records").where(
    F.col("experiment_run_id") == source_run_id
)
if source.limit(config["max_records"] + 1).count() != manifest["accepted_records"]:
    raise ValueError("Source snapshot changed or exceeds the driver budget")
records = [row.asDict() for row in source.orderBy("unique_id").collect()]
ids = [r["unique_id"] for r in records]
if (
    len(ids) != len(set(ids))
    or digest(
        sorted(
            (r["unique_id"], r["content_hash"], r["translation_evidence_json"])
            for r in records
        )
    )
    != manifest["source_input_fingerprint"]
):
    raise ValueError("Source identities or fingerprint changed between tasks")
con = duckdb.connect(":memory:")
con.execute("SET memory_limit='2GB'")
con.register("record_input", pa.Table.from_pylist(records))
con.execute("CREATE TABLE records AS SELECT * FROM record_input")
con.execute("CREATE TABLE known_pairs(id_l VARCHAR, id_r VARCHAR)")
old = (
    spark.read.table(f"{prefix}.pair_scores")
    .where(F.col("experiment_run_id") == source_run_id)
    .select(F.col("unique_id_l").alias("id_l"), F.col("unique_id_r").alias("id_r"))
)
if old.limit(config["max_candidate_pairs"] + 1).count() > config["max_candidate_pairs"]:
    raise ValueError("Prior candidate table exceeds the experiment budget")
batch = []
for row in old.toLocalIterator():
    batch.append(row.asDict())
    if len(batch) == 25_000:
        con.register("known_batch", pa.Table.from_pylist(batch))
        con.execute("INSERT INTO known_pairs SELECT * FROM known_batch")
        batch = []
if batch:
    con.register("known_batch", pa.Table.from_pylist(batch))
    con.execute("INSERT INTO known_pairs SELECT * FROM known_batch")
known_count = con.execute("SELECT count(*) FROM known_pairs").fetchone()[0]
invalid_known = con.execute("""SELECT count(*) FROM known_pairs k
    LEFT JOIN records l ON k.id_l=l.unique_id LEFT JOIN records r ON k.id_r=r.unique_id
    WHERE k.id_l >= k.id_r OR l.unique_id IS NULL OR r.unique_id IS NULL""").fetchone()[
    0
]
if (
    invalid_known
    or known_count != manifest["source_candidate_pairs"]
    or known_count
    != con.execute(
        "SELECT count(*) FROM (SELECT DISTINCT id_l,id_r FROM known_pairs)"
    ).fetchone()[0]
):
    raise ValueError("Prior pairs do not belong to this source snapshot")
initialize_retrieval(con)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(
    experiment_id=config["experiment_id"], run_name="embedding_hnsw"
) as run:
    manifest["mlflow_run_id"] = run.info.run_id
    manifest["hnsw"] = {
        "M": 32,
        "ef_construction": 200,
        "ef_search": max(200, config["search_k"] + 1),
    }
    manifest["retrieval_audit"] = []
    manifest["exact_search_audit"] = []
    embedded = spark.read.table(f"{prefix}.embedding_records").where(
        F.col("embedding_run_id") == input_embedding_run_id
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "index_ids.json").write_text(json_text(ids))
        for view in VIEWS:
            frame = embedded.where(F.col("view") == view).orderBy("unique_id")
            vectors = np.empty((len(ids), DIMENSION), dtype="float32")
            position = 0
            for row in frame.toLocalIterator():
                if (
                    position >= len(ids)
                    or row["unique_id"] != ids[position]
                    or row["content_hash"] != records[position]["content_hash"]
                ):
                    raise ValueError(
                        "Embedding identity/content does not match the source snapshot"
                    )
                vectors[position] = row["embedding"]
                position += 1
            if position != len(ids):
                raise ValueError("Missing vectors: every accepted row must be indexed")
            vectors = normalized_vectors(vectors)
            index = build_index(
                vectors,
                **{
                    "m": manifest["hnsw"]["M"],
                    "ef_construction": 200,
                    "ef_search": manifest["hnsw"]["ef_search"],
                },
            )
            manifest["retrieval_audit"].append(
                retrieve(
                    con,
                    ids,
                    vectors,
                    index,
                    view,
                    top_k=config["top_k"],
                    search_k=config["search_k"],
                    min_cosine=config["min_cosine"],
                    max_pairs=config["max_candidate_pairs"],
                )
            )
            manifest["exact_search_audit"].append(
                exact_search_audit(
                    con,
                    ids,
                    vectors,
                    index,
                    view,
                    top_k=config["top_k"],
                    min_cosine=config["min_cosine"],
                )
            )
            # Catch invalid audit/manifest values before appending any results.
            json_text(manifest)
            faiss.write_index(index, str(root / f"{view}.faiss"))
        mlflow.log_artifacts(directory, "hnsw_indexes")
    counts = finalize_candidates(con)
    for table, query in {
        "embedding_candidate_passes": "SELECT * FROM embedding_passes",
        "embedding_candidates": "SELECT * FROM embedding_candidates",
        "embedding_review_sample": """SELECT * EXCLUDE(review_rank) FROM (
            SELECT *,row_number() OVER (PARTITION BY decision,retrieval_views,input_route_l,input_route_r
                ORDER BY hash(id_l || id_r)) review_rank FROM embedding_candidates
        ) WHERE review_rank<=30""",
    }.items():
        append_result(
            spark,
            con.execute(query).fetch_arrow_table(),
            f"{prefix}.{table}",
            run_id,
            source_run_id,
        )
    if spark.read.table(f"{prefix}.embedding_candidates").where(
        F.col("embedding_run_id") == run_id
    ).count() != sum(counts.values()):
        raise AssertionError("Persisted candidate count differs from retrieval output")
    manifest.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "known_pairs_excluded": known_count,
            "candidate_counts": counts,
            "new_candidate_pairs": sum(counts.values()),
            "dependencies": {
                p: importlib.metadata.version(p)
                for p in (
                    "faiss-cpu",
                    "numpy",
                    "sentence-transformers",
                    "torch",
                    "transformers",
                    "duckdb",
                    "mlflow",
                )
            },
            "precision": None,
            "true_match_recall": None,
            "approved_merges": 0,
            "notes": [
                "Cosines are retrieval similarities, not calibrated match probabilities",
                "All old scored pairs excluded, including low-probability and conflict pairs",
                "No country/region blocking; discrepancies remain explicit review flags",
                "Finite neighbor budgets and model quality can miss true pairs",
            ],
        }
    )
    source_hash = hashlib.sha256()
    source_paths = [
        *Path(library_root).glob("lib/*.py"),
        Path(library_root) / "notebooks/embed_matching.py",
        Path(library_root) / "notebooks/search_embeddings.py",
    ]
    for path in sorted(source_paths):
        source_hash.update(path.read_bytes())
        mlflow.log_text(path.read_text(), f"source/{path.name}")
    manifest["library_code_sha256"] = source_hash.hexdigest()
    manifest_json = json_text(manifest)
    mlflow.log_dict(manifest, "embedding_manifest.json")
    mlflow.log_metrics(
        {"new_candidate_pairs": sum(counts.values()), "indexed_records": len(ids)}
    )
    for audit in manifest["exact_search_audit"]:
        if audit["novel_neighbor_recall_at_k"] is not None:
            mlflow.log_metric(
                f"{audit['view']}_novel_neighbor_recall_at_k",
                audit["novel_neighbor_recall_at_k"],
            )
    spark.createDataFrame(
        [(run_id, source_run_id, "COMPLETED", manifest_json)],
        "embedding_run_id string, source_run_id string, status string, manifest_json string",
    ).write.format("delta").mode("append").saveAsTable(f"{prefix}.embedding_runs")
con.close()
dbutils.jobs.taskValues.set(key="embedding_run_id", value=run_id)
dbutils.notebook.exit(manifest_json)
