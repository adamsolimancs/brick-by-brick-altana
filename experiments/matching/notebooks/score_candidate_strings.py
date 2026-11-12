# Databricks notebook source
# MAGIC %md
# MAGIC # Compare saved candidate text with Levenshtein and Jaro–Winkler
# MAGIC Reuses existing pairs and their normalized source snapshot. Adds evidence;
# MAGIC preserves all existing decisions and conflict checks. No model/index tasks.

# COMMAND ----------

import hashlib
import importlib.metadata
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from pyspark.sql import Window, functions as F

defaults = {
    "output_catalog": "addressresolution",
    "output_schema": "matching_experiments_aes10130",
    "library_root": "",
    "experiment_id": "",
    "input_embedding_run_id": "latest",
    "string_scoring_run_id": "",
    "max_candidate_pairs": "2000000",
}
for name, default in defaults.items():
    dbutils.widgets.text(name, default)
config = {name: dbutils.widgets.get(name) for name in defaults}
if config["output_catalog"] != "addressresolution" or not (
    config["output_schema"] == "matching"
    or re.fullmatch(r"matching_experiments_[a-zA-Z0-9_]+", config["output_schema"])
):
    raise ValueError(
        "Writes must stay in addressresolution.matching_experiments_<owner>"
    )
run_id = config["string_scoring_run_id"]
if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", run_id):
    raise ValueError("A unique string_scoring_run_id is required")
max_pairs = int(config["max_candidate_pairs"])
if not 1 <= max_pairs <= 2_000_000:
    raise ValueError("Candidate limit must be between 1 and 2,000,000")
prefix = f"{config['output_catalog']}.{config['output_schema']}"
sys.path.insert(0, config["library_root"])
from lib.string_features import (  # noqa: E402
    COMPARISONS,
    VERSION,
    add_string_features,
    RECORD_COLUMNS,
    attach_comparison_fields,
    invalid_source_condition,
    prepare_comparison_records,
)
from altana_matching_components import VERSION as COMPONENT_VERSION  # noqa: E402

started = datetime.now(timezone.utc).isoformat()
for table in (
    "string_pair_scores",
    "string_review_sample",
    "string_scoring_runs",
    "string_record_components",
):
    target = f"{prefix}.{table}"
    if (
        spark.catalog.tableExists(target)
        and spark.read.table(target)
        .where(F.col("string_scoring_run_id") == run_id)
        .limit(1)
        .count()
    ):
        raise ValueError("This run has existing outputs; start a fresh job run")

completed = spark.read.table(f"{prefix}.embedding_runs").where(
    F.col("status") == "COMPLETED"
)
if config["input_embedding_run_id"] != "latest":
    completed = completed.where(
        F.col("embedding_run_id") == config["input_embedding_run_id"]
    )
selected = (
    completed.orderBy(
        F.get_json_object("manifest_json", "$.completed_at").desc(),
        F.col("embedding_run_id").desc(),
    )
    .limit(1)
    .collect()
)
if not selected:
    raise ValueError("A completed embedding candidate run is required")
input_embedding_run_id = selected[0]["embedding_run_id"]
embedding_manifest = json.loads(selected[0]["manifest_json"])
source_run_id = embedding_manifest["source_run_id"]

embedding_pairs = spark.read.table(f"{prefix}.embedding_candidates").where(
    F.col("embedding_run_id") == input_embedding_run_id
)
blocked_pairs = (
    spark.read.table(f"{prefix}.pair_scores")
    .where(F.col("experiment_run_id") == source_run_id)
    .withColumnRenamed("unique_id_l", "id_l")
    .withColumnRenamed("unique_id_r", "id_r")
    .withColumn("source_run_id", F.lit(source_run_id))
)
embedding_count = embedding_pairs.count()
blocked_count = blocked_pairs.count()
if (
    embedding_count != embedding_manifest["new_candidate_pairs"]
    or blocked_count != embedding_manifest["source_candidate_pairs"]
    or embedding_count + blocked_count > max_pairs
):
    raise ValueError("Saved candidate counts changed or exceed the scoring budget")
if (
    embedding_pairs.where(~F.col("source_run_id").eqNullSafe(F.lit(source_run_id)))
    .limit(1)
    .count()
):
    raise ValueError("Embedding candidates reference a different matching snapshot")
pairs = embedding_pairs.withColumn("candidate_origin", F.lit("embedding")).unionByName(
    blocked_pairs.withColumn("candidate_origin", F.lit("blocked")),
    allowMissingColumns=True,
)
input_columns = pairs.columns
if (
    pairs.where(
        F.col("id_l").isNull()
        | F.col("id_r").isNull()
        | (F.col("id_l") >= F.col("id_r"))
    )
    .limit(1)
    .count()
):
    raise ValueError("Invalid saved unordered pair identity")
if pairs.groupBy("id_l", "id_r").count().where(F.col("count") != 1).limit(1).count():
    raise ValueError("Saved candidate sets contain duplicate or overlapping pairs")
records = spark.read.table(f"{prefix}.prepared_records").where(
    F.col("experiment_run_id") == source_run_id
)
record_count = records.count()
if (
    record_count != embedding_manifest["accepted_records"]
    or records.select("unique_id").distinct().count() != record_count
):
    raise ValueError(
        "Prepared source snapshot changed or contains duplicate identities"
    )
record_columns = list(RECORD_COLUMNS)
components = (
    prepare_comparison_records(records.select(*record_columns))
    .withColumn("string_scoring_run_id", F.lit(run_id))
    .withColumn("source_run_id", F.lit(source_run_id))
)
components.write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(f"{prefix}.string_record_components")
component_records = spark.read.table(f"{prefix}.string_record_components").where(
    F.col("string_scoring_run_id") == run_id
)
if (
    component_records.count() != record_count
    or component_records.select("unique_id").distinct().count() != record_count
):
    raise AssertionError("Parsed source-record conservation failed")
if (
    component_records.select(*record_columns)
    .exceptAll(records.select(*record_columns))
    .limit(1)
    .count()
):
    raise AssertionError("Street parsing changed an original snapshot field")
component_summary = (
    component_records.agg(
        F.count("*").alias("records"),
        F.count(
            F.when(~F.col("road_key").eqNullSafe(F.col("street_component_road_key")), 1)
        ).alias("changed_road_keys"),
        F.count(F.when(F.length("street_component_road_key") == 0, 1)).alias(
            "missing_road_names"
        ),
        *[
            F.count(F.when(F.size("street_component_" + name) > 0, 1)).alias(
                "records_with_" + name
            )
            for name in ("house_numbers", "units", "buildings", "floors")
        ],
        F.count(
            F.when(
                F.array_contains(
                    "street_component_warnings", "unresolved_component_marker_retained"
                ),
                1,
            )
        ).alias("unresolved_component_markers"),
        F.count(
            F.when(
                F.array_contains(
                    "street_component_warnings", "ambiguous_trailing_numbers_retained"
                ),
                1,
            )
        ).alias("ambiguous_trailing_numbers_retained"),
    )
    .first()
    .asDict()
)
component_country_counts = [
    row.asDict()
    for row in component_records.groupBy("country")
    .agg(
        F.count("*").alias("records"),
        F.count(
            F.when(~F.col("road_key").eqNullSafe(F.col("street_component_road_key")), 1)
        ).alias("changed_road_keys"),
    )
    .orderBy("country")
    .collect()
]
enriched = attach_comparison_fields(pairs, component_records)
if enriched.where(invalid_source_condition()).limit(1).count():
    raise ValueError(
        "Candidate source identity/content hash does not match the saved snapshot"
    )
enriched = enriched.drop(
    *[
        f"_prepared_{field}_{side}"
        for side in ("l", "r")
        for field in ("record_id", "content_hash")
    ]
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(experiment_id=config["experiment_id"], run_name=VERSION) as run:
    scored = (
        add_string_features(enriched)
        .withColumn("string_scoring_run_id", F.lit(run_id))
        .withColumn("input_embedding_run_id", F.lit(input_embedding_run_id))
        .withColumn("string_feature_version", F.lit(VERSION))
    )
    scored.write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(f"{prefix}.string_pair_scores")
    saved = spark.read.table(f"{prefix}.string_pair_scores").where(
        F.col("string_scoring_run_id") == run_id
    )
    total = embedding_count + blocked_count
    if (
        saved.count() != total
        or saved.select("id_l", "id_r").distinct().count() != total
    ):
        raise AssertionError("Scored pair conservation failed")
    # Exact multiset comparison includes original probabilities, similarities,
    # decisions, and every original conflict flag. No scorer can override them.
    if (
        saved.select(*input_columns)
        .exceptAll(pairs.select(*input_columns))
        .limit(1)
        .count()
    ):
        raise AssertionError(
            "An existing score, decision, guard, or evidence field changed"
        )
    feature_summary = {}
    for name in COMPARISONS:
        distance = F.col(f"{name}_levenshtein_distance")
        lev = F.col(f"{name}_levenshtein_similarity")
        jw = F.col(f"{name}_jaro_winkler_similarity")
        left, right = F.col(f"comparison_{name}_l"), F.col(f"comparison_{name}_r")
        known = (F.length(F.trim(left)) > 0) & (F.length(F.trim(right)) > 0)
        invalid = (
            (known & (distance.isNull() | lev.isNull() | jw.isNull()))
            | (distance < 0)
            | (distance > F.greatest(F.length(left), F.length(right)))
            | F.isnan(lev)
            | (lev < 0)
            | (lev > 1)
            | F.isnan(jw)
            | (jw < 0)
            | (jw > 1)
            | (
                ~F.coalesce(known, F.lit(False))
                & (distance.isNotNull() | lev.isNotNull() | jw.isNotNull())
            )
        )
        summary = (
            saved.agg(
                F.count(F.when(invalid, 1)).alias("invalid_rows"),
                F.count(F.when(distance.isNull(), 1)).alias("unknown_rows"),
                F.count(F.when(distance == 0, 1)).alias("exact_text_pairs"),
                F.avg(lev).alias("mean_levenshtein_similarity"),
                F.avg(jw).alias("mean_jaro_winkler_similarity"),
            )
            .first()
            .asDict()
        )
        if summary["invalid_rows"]:
            raise AssertionError(
                f"Invalid string comparison outputs: {name}: {summary}"
            )
        feature_summary[name] = summary
    review = (
        saved.withColumn(
            "_review_rank",
            F.row_number().over(
                Window.partitionBy("candidate_origin", "decision").orderBy(
                    F.sha2(F.concat("id_l", "id_r"), 256)
                )
            ),
        )
        .where(F.col("_review_rank") <= 30)
        .drop("_review_rank")
    )
    review.write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(f"{prefix}.string_review_sample")
    decision_counts = [
        row.asDict()
        for row in saved.groupBy("candidate_origin", "decision").count().collect()
    ]
    source_paths = [
        Path(config["library_root"]) / "lib/string_features.py",
        Path(config["library_root"]) / "altana_matching_components/__init__.py",
        Path(config["library_root"]) / "lib/matching.py",
        Path(config["library_root"]) / "notebooks/score_candidate_strings.py",
    ]
    code_hash = hashlib.sha256()
    for path in source_paths:
        code_hash.update(path.read_bytes())
        mlflow.log_text(
            path.read_text(), f"source/{path.relative_to(config['library_root'])}"
        )
    manifest = {
        "version": VERSION,
        "string_scoring_run_id": run_id,
        "input_embedding_run_id": input_embedding_run_id,
        "source_run_id": source_run_id,
        "source_input_fingerprint": embedding_manifest["source_input_fingerprint"],
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "scored_pairs": total,
        "embedding_pairs": embedding_count,
        "blocked_pairs": blocked_count,
        "decision_counts": decision_counts,
        "feature_summary": feature_summary,
        "source_code_sha256": code_hash.hexdigest(),
        "mlflow_run_id": run.info.run_id,
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("rapidfuzz", "pandas", "pyarrow", "mlflow")
        },
        "spark_version": spark.version,
        "config": config,
        "embeddings_rebuilt": False,
        "hnsw_indexes_rebuilt": False,
        "existing_fields_preserved": True,
        "input_pair_set_preserved": True,
        "comparison_keys": COMPARISONS,
        "street_component_version": COMPONENT_VERSION,
        "street_component_summary": component_summary,
        "street_component_country_counts": component_country_counts,
        "parsed_address_conflict_pairs": saved.where(
            F.col("parsed_address_conflict")
        ).count(),
        "parsed_address_review_pairs": saved.where(
            F.col("parsed_address_review_required")
        ).count(),
        "parsed_partial_agreement_pairs": saved.where(
            F.col("parsed_house_numbers_partial_agreement")
            | F.col("parsed_units_partial_agreement")
            | F.col("parsed_buildings_partial_agreement")
            | F.col("parsed_floors_partial_agreement")
        ).count(),
        "changed_street_name_pairs": saved.where(
            ~F.col("comparison_street_name_l").eqNullSafe(
                F.col("previous_street_name_l")
            )
            | ~F.col("comparison_street_name_r").eqNullSafe(
                F.col("previous_street_name_r")
            )
        ).count(),
        "normalization": "1 - levenshtein / max(character lengths); empty/null keys are unknown",
        "notes": [
            "Additive comparison evidence only; original decisions and conflict flags preserved",
            "Street-name comparisons reparse accepted raw street text; previous road keys and original guards are retained",
            "Parsed house, unit, building, and floor values and conflict flags are additional review evidence, never automatic merge decisions",
            "No new acceptance threshold, probability calibration, or automatic merge",
        ],
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False)
    mlflow.log_dict(manifest, "string_features_manifest.json")
    mlflow.log_metrics(
        {
            "scored_pairs": total,
            "embedding_pairs": embedding_count,
            "blocked_pairs": blocked_count,
        }
    )
    spark.createDataFrame(
        [(run_id, input_embedding_run_id, source_run_id, "COMPLETED", payload)],
        "string_scoring_run_id string, input_embedding_run_id string, source_run_id string, status string, manifest_json string",
    ).write.format("delta").mode("append").saveAsTable(f"{prefix}.string_scoring_runs")
dbutils.jobs.taskValues.set(key="string_scoring_run_id", value=run_id)
dbutils.notebook.exit(payload)
