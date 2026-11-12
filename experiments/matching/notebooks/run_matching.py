# Databricks notebook source
# MAGIC %md
# MAGIC # First company/location matching experiment
# MAGIC Unsupervised probabilities and proposed links require review. This job
# MAGIC never updates production silver/gold records or approves fuzzy merges.

# COMMAND ----------

import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import mlflow
import pyarrow as pa
from pyspark.sql import Window, functions as F, types as T

for name, default in {
    "output_catalog": "addressresolution",
    "output_schema": "matching_experiments_aes10130",
    "library_root": "",
    "experiment_id": "",
    "max_records": "100000",
    "max_candidate_pairs": "2000000",
}.items():
    dbutils.widgets.text(name, default)

library_root = dbutils.widgets.get("library_root")
sys.path.insert(0, library_root)
from lib.matching import (  # noqa: E402
    FIELDS,
    MERGE_THRESHOLD,
    REJECT_THRESHOLD,
    SEED,
    VERSION,
    build_candidates,
    comparison_summary,
    digest,
    fit_model,
    register_records,
    suggestions,
)

catalog = dbutils.widgets.get("output_catalog")
schema = dbutils.widgets.get("output_schema")
if catalog != "addressresolution" or not (
    schema == "matching" or re.fullmatch(r"matching_experiments_[a-zA-Z0-9_]+", schema)
):
    raise ValueError(
        "Experiment writes must stay in addressresolution.matching_experiments_<owner>"
    )
prefix = f"{catalog}.{schema}"
max_records = int(dbutils.widgets.get("max_records"))
max_pairs = int(dbutils.widgets.get("max_candidate_pairs"))
if not 2 <= max_records <= 100_000 or not 1 <= max_pairs <= 2_000_000:
    raise ValueError(
        "This driver-bounded experiment supports at most 100,000 records and 2,000,000 candidate-pass rows"
    )

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {prefix}")
mlflow.set_registry_uri("databricks-uc")
experiment_id = dbutils.widgets.get("experiment_id")
started = datetime.now(timezone.utc).isoformat()

# COMMAND ----------


def source_projection(df, route):
    business = [
        "provider",
        "provider_id",
        *FIELDS,
        "created_at",
        "updated_at",
        "country_inferred",
    ]
    evidence = [
        c
        for c in df.columns
        if c.endswith(("_original", "_translation_status", "_ai_model"))
    ]
    evidence += [
        c
        for c in (
            "translation_route",
            "translation_review_reasons",
            "remaining_dlq_reasons",
        )
        if c in df.columns
    ]
    return df.select(
        *[F.col(c).cast("string").alias(c) for c in business],
        F.lit(route).alias("input_route"),
        F.to_json(F.struct(*[F.col(c) for c in business])).alias("source_record_json"),
        F.to_json(F.struct(*[F.col(c) for c in evidence])).alias(
            "translation_evidence_json"
        )
        if evidence
        else F.lit("{}").alias("translation_evidence_json"),
    )


def append_arrow(connection, query, table, run_id):
    """Write explicit scalar schemas, including empty/all-null outputs, in batches."""
    arrow = connection.execute(query).fetch_arrow_table()
    fields = []
    for field in arrow.schema:
        if pa.types.is_boolean(field.type):
            dtype = T.BooleanType()
        elif pa.types.is_integer(field.type):
            dtype = T.LongType()
        elif pa.types.is_floating(field.type) or pa.types.is_decimal(field.type):
            dtype = T.DoubleType()
        elif (
            pa.types.is_string(field.type)
            or pa.types.is_large_string(field.type)
            or pa.types.is_null(field.type)
        ):
            dtype = T.StringType()
        else:
            raise TypeError(f"Unexpected result type for {field.name}: {field.type}")
        fields.append(T.StructField(field.name, dtype, True))
    target_schema = T.StructType(fields)
    batches = arrow.to_batches(max_chunksize=25_000)
    if not batches:
        batches = [None]
    for batch in batches:
        rows = batch.to_pylist() if batch is not None else []
        # Normalize Arrow numeric widths for Spark's strict explicit schema.
        for row in rows:
            for field in fields:
                if row[field.name] is not None and isinstance(
                    field.dataType, T.DoubleType
                ):
                    row[field.name] = float(row[field.name])
        frame = spark.createDataFrame(rows, target_schema).withColumn(
            "experiment_run_id", F.lit(run_id)
        )
        frame.write.format("delta").mode("append").saveAsTable(f"{prefix}.{table}")


# Human labels are intentionally independent of all refreshable experiment outputs.
spark.sql(f"""CREATE TABLE IF NOT EXISTS {prefix}.match_labels (
    unique_id_l STRING, unique_id_r STRING, content_hash_l STRING, content_hash_r STRING,
    label STRING, reviewer STRING, reviewed_at TIMESTAMP, notes STRING
) USING DELTA COMMENT 'Persistent reviewed company/location pair judgments; never overwritten by an experiment'""")

# COMMAND ----------

with mlflow.start_run(experiment_id=experiment_id, run_name=VERSION) as run:
    run_id = run.info.run_id
    clean = spark.read.table("addressresolution.silver.cleaned_1_0")
    translated = spark.read.table("addressresolution.silver.providers_translated")
    results = spark.read.table(
        "addressresolution.silver.providers_translation_results"
    ).where(F.col("translation_route") == "accepted")
    # Validate accepted-output/source parity before taking original text from results.
    identity_business = ["provider", "provider_id", *FIELDS]
    left = translated.select(*identity_business)
    right = results.select(*identity_business)
    if left.exceptAll(right).limit(1).count() or right.exceptAll(left).limit(1).count():
        raise ValueError(
            "Accepted translation output and evidence snapshots disagree; refresh silver consistently first"
        )

    source = source_projection(clean, "clean").unionByName(
        source_projection(results, "translated")
    )
    if source.limit(max_records + 1).count() > max_records:
        raise ValueError(
            "Input exceeds driver-bound experiment budget; no sampling or truncation performed"
        )
    source = source.withColumn(
        "source_identity_count",
        F.count("*").over(Window.partitionBy("provider", "provider_id")),
    )
    reasons = [
        F.when(
            F.col(c).isNull()
            | (F.length(F.regexp_replace(F.col(c), r"[\s\p{Z}]+", "")) == 0),
            F.lit(f"{c}:missing"),
        )
        for c in ["provider", "provider_id", *FIELDS]
    ]
    reasons.append(
        F.when(F.col("source_identity_count") > 1, F.lit("source_identity:duplicate"))
    )
    source = source.withColumn(
        "input_reasons",
        F.concat_ws(",", F.filter(F.array(*reasons), lambda x: x.isNotNull())),
    )
    source.withColumn("experiment_run_id", F.lit(run_id)).write.format("delta").mode(
        "append"
    ).saveAsTable(f"{prefix}.input_snapshots")
    snapshot = spark.read.table(f"{prefix}.input_snapshots").where(
        F.col("experiment_run_id") == run_id
    )
    snapshot_count = snapshot.count()
    if snapshot_count > max_records:
        raise ValueError(
            "Materialized snapshot exceeds input limit; it was retained but will not be collected"
        )
    accepted = snapshot.where(F.col("input_reasons") == "")
    rejected_count = snapshot.where(F.col("input_reasons") != "").count()
    rows = [
        row.asDict() for row in accepted.orderBy("provider", "provider_id").collect()
    ]
    if len(rows) + rejected_count != snapshot_count:
        raise AssertionError("Input conservation failed")

    connection = duckdb.connect(":memory:")
    # A hard local-memory bound; larger deployments need distributed execution.
    connection.execute("SET memory_limit='2GB'")
    prepared = register_records(connection, rows)
    blocking_audit, rules = build_candidates(connection, max_pairs)
    baseline_pairs = connection.execute(
        "SELECT count(*) FROM candidate_passes WHERE blocking_pass='exact_baseline'"
    ).fetchone()[0]
    linker, model, training = fit_model(
        connection, rules, baseline_pairs, len(prepared)
    )
    suggestions(connection, training["prior"])
    summary = comparison_summary(connection, len(prepared))
    summary.update(
        {"input_snapshot_rows": snapshot_count, "rejected_input_rows": rejected_count}
    )

    # All outputs are run-keyed append-only experiments, not approved entity merges.
    append_arrow(connection, "SELECT * FROM records", "prepared_records", run_id)
    append_arrow(
        connection, "SELECT * FROM candidate_passes", "candidate_passes", run_id
    )
    append_arrow(connection, "SELECT * FROM decisions", "pair_scores", run_id)
    append_arrow(
        connection,
        "SELECT baseline_id, count(*) record_count, count(DISTINCT provider) provider_count FROM records GROUP BY baseline_id",
        "exact_clusters",
        run_id,
    )
    append_arrow(
        connection,
        "SELECT decision, count(*) pair_count FROM decisions GROUP BY decision",
        "comparison_counts",
        run_id,
    )
    connection.register("audit_arrow", pa.Table.from_pylist(blocking_audit))
    append_arrow(connection, "SELECT * FROM audit_arrow", "blocking_audit", run_id)
    # Deterministic review sample across outcome and provider/script-route slices.
    append_arrow(
        connection,
        """SELECT * EXCLUDE (review_rank) FROM (
        SELECT *, row_number() OVER (
            PARTITION BY decision, input_route_l, input_route_r
            ORDER BY hash(unique_id_l || unique_id_r)
        ) review_rank FROM decisions
    ) WHERE review_rank <= 30""",
        "review_sample",
        run_id,
    )

    source_hash = hashlib.sha256()
    source_files = sorted(
        [
            *Path(library_root).glob("lib/*.py"),
            *Path(library_root).glob("altana_matching_components/*.py"),
        ]
    )
    for path in source_files:
        source_hash.update(path.read_bytes())
        mlflow.log_text(path.read_text(), f"source/{path.relative_to(library_root)}")
    manifest = {
        "version": VERSION,
        "experiment_run_id": run_id,
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source_tables": [
            "addressresolution.silver.cleaned_1_0",
            "addressresolution.silver.providers_translated",
            "addressresolution.silver.providers_translation_results",
        ],
        "input_fingerprint": digest(
            sorted(
                (r["unique_id"], r["content_hash"], r["translation_evidence_json"])
                for r in prepared
            )
        ),
        "library_code_sha256": source_hash.hexdigest(),
        "dependencies": {
            p: importlib.metadata.version(p)
            for p in ["splink", "duckdb", "mlflow", "pyarrow"]
        },
        "seed": SEED,
        "blocking_audit": blocking_audit,
        "training": training,
        "thresholds": {"proposed": MERGE_THRESHOLD, "reject": REJECT_THRESHOLD},
        "summary": summary,
        "notes": [
            "No reviewed match labels found during initial discovery",
            "No probability calibration or live precision/recall measurement",
            "No fuzzy clustering or production merges",
            "Baseline group membership uses original case/whitespace-normalized company plus all address fields",
            "Model probabilities depend on an unvalidated prior; 2x/4x prior sensitivity retained per pair",
        ],
    }
    mlflow.log_dict(model, "splink_model.json")
    mlflow.log_dict(manifest, "manifest.json")
    mlflow.log_dict(summary, "comparison_summary.json")
    mlflow.log_metrics(
        {k: v for k, v in summary.items() if isinstance(v, (float, int))}
    )
    mlflow.log_params(
        {
            "version": VERSION,
            "input_fingerprint": manifest["input_fingerprint"],
            "prior": training["prior"],
            "merge_threshold": MERGE_THRESHOLD,
            "reject_threshold": REJECT_THRESHOLD,
        }
    )
    spark.createDataFrame(
        [(run_id, json.dumps(manifest, sort_keys=True), "COMPLETED")],
        "experiment_run_id string, manifest_json string, status string",
    ).write.format("delta").mode("append").saveAsTable(f"{prefix}.runs")
    connection.close()

# COMMAND ----------

display(
    spark.read.table(f"{prefix}.comparison_counts").where(
        F.col("experiment_run_id") == run_id
    )
)
dbutils.jobs.taskValues.set(key="experiment_run_id", value=run_id)
dbutils.notebook.exit(
    json.dumps(
        {"experiment_run_id": run_id, "output_schema": prefix, "summary": summary}
    )
)
