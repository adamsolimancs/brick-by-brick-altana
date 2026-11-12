# Databricks notebook source
# MAGIC %md
# MAGIC # Embed every accepted row in a completed matching snapshot
# MAGIC Runs on job compute with a pinned multilingual model. Cache keys include
# MAGIC the model revision, text-format version, and exact input text hash.

# COMMAND ----------

import json
import sys
from datetime import datetime, timezone

from pyspark.sql import functions as F

dbutils.widgets.text("library_root", "")
sys.path.insert(0, dbutils.widgets.get("library_root"))
from lib.embedding_runtime import require_fresh_outputs, settings  # noqa: E402
from lib.embeddings import (  # noqa: E402
    DIMENSION,
    MODEL,
    MODEL_KEY,
    REVISION,
    VERSION,
    VIEWS,
    embedding_text,
    encode_texts,
    json_text,
)
from lib.matching import digest  # noqa: E402

config = settings(dbutils)
prefix = config["prefix"]
run_id = config["embedding_run_id"]
started = datetime.now(timezone.utc).isoformat()
require_fresh_outputs(spark, [f"{prefix}.embedding_records"], run_id)
completed = spark.read.table(f"{prefix}.runs").where(F.col("status") == "COMPLETED")
if config["source_run_id"] != "latest":
    completed = completed.where(F.col("experiment_run_id") == config["source_run_id"])
selected = (
    completed.orderBy(
        F.get_json_object("manifest_json", "$.completed_at").desc(),
        F.col("experiment_run_id").desc(),
    )
    .limit(1)
    .collect()
)
if not selected:
    raise ValueError("Run the existing exact/Fellegi-Sunter job successfully first")
source_run_id = selected[0]["experiment_run_id"]
source_manifest = json.loads(selected[0]["manifest_json"])
source = spark.read.table(f"{prefix}.prepared_records").where(
    F.col("experiment_run_id") == source_run_id
)
count = source.limit(config["max_records"] + 1).count()
if not 2 <= count <= config["max_records"]:
    raise ValueError(
        "Snapshot exceeds bounded embedding capacity; no rows sampled or truncated"
    )
records = [r.asDict() for r in source.orderBy("unique_id").collect()]
if len({r["unique_id"] for r in records}) != count:
    raise ValueError("Duplicate source identities in the completed snapshot")
fingerprint = digest(
    sorted(
        (r["unique_id"], r["content_hash"], r["translation_evidence_json"])
        for r in records
    )
)
if fingerprint != source_manifest["input_fingerprint"]:
    raise ValueError(
        "Prepared records no longer match the completed source fingerprint"
    )

# COMMAND ----------

spark.sql(f"""CREATE TABLE IF NOT EXISTS {prefix}.embedding_runs (
    embedding_run_id STRING, source_run_id STRING, status STRING, manifest_json STRING
) USING DELTA""")
if (
    spark.read.table(f"{prefix}.embedding_runs")
    .where(F.col("embedding_run_id") == run_id)
    .limit(1)
    .count()
):
    raise ValueError(
        "This run ID already exists; start a fresh job run instead of repairing"
    )
spark.sql(f"""CREATE TABLE IF NOT EXISTS {prefix}.embedding_cache (
    model_key STRING, text_hash STRING, embedding ARRAY<FLOAT>
) USING DELTA""")
text_rows = []
for record in records:
    for view in VIEWS:
        text = embedding_text(record, view)
        text_rows.append(
            (record["unique_id"], record["content_hash"], view, text, digest(text))
        )
texts = spark.createDataFrame(
    text_rows,
    "unique_id string, content_hash string, view string, embedding_text string, text_hash string",
)
cache = spark.read.table(f"{prefix}.embedding_cache").where(
    F.col("model_key") == MODEL_KEY
)
missing = (
    texts.select("text_hash", "embedding_text")
    .distinct()
    .join(cache.select("text_hash").distinct(), "text_hash", "left_anti")
    .orderBy("text_hash")
    .collect()
)
if missing:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(4)
    model = SentenceTransformer(
        MODEL, revision=REVISION, device="cpu", trust_remote_code=False
    )
    # Downloads model weights only. Inference is local; no address text leaves compute.
    for offset in range(0, len(missing), 4096):
        batch = missing[offset : offset + 4096]
        vectors = encode_texts(model, [row["embedding_text"] for row in batch])
        if vectors.shape != (len(batch), DIMENSION):
            raise ValueError("Unexpected embedding shape for the pinned model")
        spark.createDataFrame(
            [
                (MODEL_KEY, row["text_hash"], vector.tolist())
                for row, vector in zip(batch, vectors)
            ],
            "model_key string, text_hash string, embedding array<float>",
        ).write.format("delta").mode("append").saveAsTable(f"{prefix}.embedding_cache")

# Re-read the cache after writes; deduplicate for safe reuse across concurrent owners.
cache = (
    spark.read.table(f"{prefix}.embedding_cache")
    .where(F.col("model_key") == MODEL_KEY)
    .dropDuplicates(["text_hash"])
)
embedded = (
    texts.join(cache, "text_hash", "inner")
    .withColumn("source_run_id", F.lit(source_run_id))
    .withColumn("embedding_run_id", F.lit(run_id))
)
embedded.write.format("delta").mode("append").saveAsTable(f"{prefix}.embedding_records")
saved = spark.read.table(f"{prefix}.embedding_records").where(
    F.col("embedding_run_id") == run_id
)
if (
    saved.count() != count * len(VIEWS)
    or saved.where(F.size("embedding") != DIMENSION).limit(1).count()
):
    raise AssertionError("Embedding row conservation/dimension check failed")
manifest = {
    "version": VERSION,
    "embedding_run_id": run_id,
    "source_run_id": source_run_id,
    "started_at": started,
    "embedded_at": datetime.now(timezone.utc).isoformat(),
    "model": MODEL,
    "model_revision": REVISION,
    "model_key": MODEL_KEY,
    "dimension": DIMENSION,
    "views": list(VIEWS),
    "accepted_records": count,
    "source_input_fingerprint": source_manifest["input_fingerprint"],
    "source_snapshot_rows": source_manifest["summary"]["input_snapshot_rows"],
    "source_rejected_rows": source_manifest["summary"]["rejected_input_rows"],
    "source_candidate_pairs": source_manifest["summary"]["candidate_pairs"],
    "new_texts_encoded": len(missing),
    "distinct_texts": texts.select("text_hash").distinct().count(),
    "config": config,
}
spark.createDataFrame(
    [(run_id, source_run_id, "EMBEDDED", json_text(manifest))],
    "embedding_run_id string, source_run_id string, status string, manifest_json string",
).write.format("delta").mode("append").saveAsTable(f"{prefix}.embedding_runs")
dbutils.jobs.taskValues.set(key="embedding_run_id", value=run_id)
dbutils.notebook.exit(json_text(manifest))
