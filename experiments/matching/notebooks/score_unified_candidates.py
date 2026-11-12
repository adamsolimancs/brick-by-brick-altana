# Databricks notebook source
# MAGIC %md
# MAGIC # Unified cleaned-field Fellegi–Sunter
# MAGIC Fits Splink on the saved record snapshot, then scores exactly the saved
# MAGIC blocked + HNSW candidate union. Existing evidence remains versioned intact.

# COMMAND ----------

import hashlib
import importlib.metadata
import json
import math
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import mlflow
import pyarrow as pa
from pyspark.sql import Window, functions as F

defaults = {
    "output_catalog": "addressresolution",
    "output_schema": "matching_experiments_aes10130",
    "artifact_volume": "unified_scoring_artifacts",
    "library_root": "",
    "experiment_id": "",
    "input_string_scoring_run_id": "956878669358056",
    "unified_run_id": "",
    "max_records": "100000",
    "max_candidate_pairs": "2000000",
}
for name, default in defaults.items():
    dbutils.widgets.text(name, default)
config = {name: dbutils.widgets.get(name) for name in defaults}
if config["output_catalog"] != "addressresolution" or not (
    config["output_schema"] == "matching"
    or re.fullmatch(r"matching_experiments_[A-Za-z0-9_]+", config["output_schema"])
):
    raise ValueError("Writes must remain in the isolated matching experiment schema")
if not re.fullmatch(r"[A-Za-z0-9_]+", config["artifact_volume"]):
    raise ValueError("Invalid volume name")
for name in ("input_string_scoring_run_id", "unified_run_id"):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", config[name]):
        raise ValueError(f"Invalid {name}")
max_records, max_pairs = int(config["max_records"]), int(config["max_candidate_pairs"])
if not 2 <= max_records <= 100_000 or not 1 <= max_pairs <= 2_000_000:
    raise ValueError("Invalid bounded scoring budgets")
prefix = f"{config['output_catalog']}.{config['output_schema']}"
run_id, input_run = config["unified_run_id"], config["input_string_scoring_run_id"]
sys.path.insert(0, config["library_root"])
from lib.unified_matching import (  # noqa: E402
    ADDRESS_NON_MATCH,
    VERSION,
    THRESHOLDS,
    add_decisions,
    expected_gamma,
    fit_cleaned_model,
    merge_authority,
    predict_saved_pairs,
    register_jaro_winkler,
    route_address_conflicts,
)

started = datetime.now(timezone.utc).isoformat()
tables = (
    "unified_pair_scores",
    "unified_review_sample",
    "unified_models",
    "unified_decision_transitions",
    "unified_runs",
)
for table in tables:
    if (
        spark.catalog.tableExists(f"{prefix}.{table}")
        and spark.read.table(f"{prefix}.{table}")
        .where(F.col("unified_run_id") == run_id)
        .limit(1)
        .count()
    ):
        raise ValueError(
            "Run ID has previous outputs; start a new run instead of overwriting"
        )
markers = (
    spark.read.table(f"{prefix}.string_scoring_runs")
    .where(
        (F.col("string_scoring_run_id") == input_run) & (F.col("status") == "COMPLETED")
    )
    .collect()
)
if len(markers) != 1:
    raise ValueError("Exactly one completed cleaned string-scoring run is required")
source = json.loads(markers[0]["manifest_json"])
if source.get("street_component_version") not in {
    "street_components_v3",
    "street_components_v4",
    "street_components_v5",
    "street_components_v6",
}:
    raise ValueError("This scorer requires validated v3 to v6 street components")
source_run = source["source_run_id"]
pairs = spark.read.table(f"{prefix}.string_pair_scores").where(
    F.col("string_scoring_run_id") == input_run
)
input_columns = pairs.columns
count = pairs.count()
if count != source["scored_pairs"] or count > max_pairs:
    raise ValueError("Saved candidate count changed or exceeds budget")
if (
    pairs.where(
        F.col("id_l").isNull()
        | F.col("id_r").isNull()
        | (F.col("id_l") >= F.col("id_r"))
    )
    .limit(1)
    .count()
):
    raise ValueError("Invalid unordered saved candidate IDs")
if pairs.select("id_l", "id_r").distinct().count() != count:
    raise ValueError("Duplicate saved candidate pairs")
origin_counts = {
    r["candidate_origin"]: r["count"]
    for r in pairs.groupBy("candidate_origin").count().collect()
}
if origin_counts != {
    "blocked": source["blocked_pairs"],
    "embedding": source["embedding_pairs"],
}:
    raise ValueError("Blocked/HNSW union changed")
components = spark.read.table(f"{prefix}.string_record_components").where(
    F.col("string_scoring_run_id") == input_run
)
prepared = spark.read.table(f"{prefix}.prepared_records").where(
    F.col("experiment_run_id") == source_run
)
record_count = components.count()
if (
    record_count != source["street_component_summary"]["records"]
    or record_count > max_records
    or components.select("unique_id").distinct().count() != record_count
    or prepared.count() != record_count
    or prepared.select("unique_id").distinct().count() != record_count
):
    raise ValueError("Record snapshot changed or exceeds budget")
record_keys = (
    components.select(
        "unique_id", "content_hash", "name_key", "city_key", "street_component_road_key"
    )
    .alias("c")
    .join(
        prepared.select(
            "unique_id",
            "content_hash",
            "country_key",
            "baseline_id",
            "country_inferred",
        ).alias("p"),
        F.col("c.unique_id") == F.col("p.unique_id"),
        "left",
    )
)
if (
    record_keys.where(
        F.col("p.unique_id").isNull()
        | ~F.col("c.content_hash").eqNullSafe(F.col("p.content_hash"))
    )
    .limit(1)
    .count()
):
    raise ValueError("Source component identity/hash mismatch")
records = record_keys.select(
    F.col("c.unique_id").alias("unique_id"),
    F.col("c.content_hash").alias("content_hash"),
    "name_key",
    "city_key",
    "street_component_road_key",
    "country_key",
    "baseline_id",
    F.col("country_inferred").cast("boolean").alias("country_inferred"),
)
enriched = pairs
temporary = []
for side in ("l", "r"):
    names = {name: f"_record_{name}_{side}" for name in records.columns}
    lookup = records.select(
        *[F.col(name).alias(alias) for name, alias in names.items()]
    )
    enriched = enriched.join(
        lookup, F.col(f"id_{side}") == F.col(names["unique_id"]), "left"
    )
    bad = F.col(names["unique_id"]).isNull() | ~F.col(names["content_hash"]).eqNullSafe(
        F.col(f"content_hash_{side}")
    )
    for field, key in {
        "company": "name_key",
        "street_name": "street_component_road_key",
        "city": "city_key",
    }.items():
        bad = bad | ~F.col(names[key]).eqNullSafe(F.col(f"comparison_{field}_{side}"))
    if enriched.where(bad).limit(1).count():
        raise ValueError("Saved comparison fields differ from their record snapshot")
    temporary.extend(names.values())
enriched = (
    enriched.withColumn(
        "source_exact_baseline",
        F.col("_record_baseline_id_l") == F.col("_record_baseline_id_r"),
    )
    .withColumn("fs_country_inferred_l", F.col("_record_country_inferred_l"))
    .withColumn("fs_country_inferred_r", F.col("_record_country_inferred_r"))
    .drop(*temporary)
)
baseline_count = enriched.where("source_exact_baseline").count()
global_baseline = int(
    prepared.groupBy("baseline_id")
    .count()
    .agg(F.sum(F.col("count") * (F.col("count") - 1) / 2))
    .first()[0]
)
if baseline_count != global_baseline:
    raise AssertionError("Saved candidates do not preserve the full exact baseline")

artifact_root = Path(
    f"/Volumes/{config['output_catalog']}/{config['output_schema']}/{config['artifact_volume']}/{run_id}"
)
artifact_root.mkdir(parents=True, exist_ok=False)
id_path = artifact_root / "candidate_ids"
pairs.select("id_l", "id_r").write.mode("error").parquet(str(id_path))
model_records = (
    records.select(
        "unique_id",
        *[
            F.when(F.length(F.col(column)) > 0, F.col(column)).alias(name)
            for name, column in {
                "company": "name_key",
                "street_name": "street_component_road_key",
                "city": "city_key",
                "country": "country_key",
            }.items()
        ],
    )
    .orderBy("unique_id")
    .toPandas()
)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
with mlflow.start_run(experiment_id=config["experiment_id"], run_name=VERSION) as mlrun:
    with tempfile.TemporaryDirectory(prefix="altana-unified-") as temporary_directory:
        connection = duckdb.connect(":memory:")
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET threads=4")
        connection.execute(
            "SET temp_directory=" + "'" + temporary_directory.replace("'", "''") + "'"
        )
        register_jaro_winkler(connection)
        connection.register(
            "records_arrow", pa.Table.from_pandas(model_records, preserve_index=False)
        )
        connection.execute(
            "CREATE TABLE clean_records AS SELECT * FROM records_arrow ORDER BY unique_id"
        )
        connection.unregister("records_arrow")
        del model_records
        connection.execute(
            "CREATE TABLE saved_id_pairs AS SELECT '0' AS match_key,id_l AS join_key_l,id_r AS join_key_r FROM read_parquet(?)",
            [str(id_path / "*.parquet")],
        )
        if (
            connection.execute("SELECT count(*) FROM saved_id_pairs").fetchone()[0]
            != count
        ):
            raise AssertionError("Exported candidate IDs changed")
        linker, model, training = fit_cleaned_model(
            connection, baseline_count, record_count
        )
        predict_saved_pairs(connection, linker, max_pairs)
        available = [
            c[0]
            for c in connection.execute(
                "SELECT * FROM unified_predictions LIMIT 0"
            ).description
        ]
        output_columns = [
            "unique_id_l AS id_l",
            "unique_id_r AS id_r",
            "match_weight AS fs_match_weight",
            "match_probability AS fs_match_probability",
        ]
        output_columns += [
            f'"{name}" AS "fs_{name}"'
            for name in available
            if re.fullmatch(r"(?:gamma|mw|tf)_[A-Za-z0-9_]+", name)
        ]
        prediction_path = artifact_root / "predictions.parquet"
        connection.execute(
            f"COPY (SELECT {','.join(output_columns)} FROM unified_predictions) TO '{str(prediction_path)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        connection.close()
    model_payload = json.dumps(
        model, sort_keys=True, ensure_ascii=False, allow_nan=False
    )
    (artifact_root / "model.json").write_text(model_payload)
    mlflow.log_dict(model, "unified_model.json")
    mlflow.log_dict(training, "training_manifest.json")
    predictions = spark.read.parquet(str(prediction_path))
    if (
        predictions.count() != count
        or predictions.select("id_l", "id_r").distinct().count() != count
    ):
        raise AssertionError("Exported predictions changed candidate identities")
    if (
        pairs.select("id_l", "id_r")
        .exceptAll(predictions.select("id_l", "id_r"))
        .limit(1)
        .count()
    ):
        raise AssertionError("Saved candidates missing predictions")
    scored = (
        route_address_conflicts(
            add_decisions(
                enriched.join(predictions, ["id_l", "id_r"], "inner"),
                training,
                pair_geography=True,
            )
        )
        .withColumn("unified_run_id", F.lit(run_id))
        .withColumn("unified_version", F.lit(VERSION))
    )
    invalid_gamma = F.lit(False)
    for name in THRESHOLDS:
        invalid_gamma = invalid_gamma | ~F.col("fs_gamma_" + name).eqNullSafe(
            expected_gamma(name)
        )
    if scored.where(invalid_gamma).limit(1).count():
        raise AssertionError(
            "Splink comparison levels disagree with saved RapidFuzz features"
        )
    invalid_score = (
        F.col("fs_match_probability").isNull()
        | F.isnan("fs_match_probability")
        | (F.col("fs_match_probability") < 0)
        | (F.col("fs_match_probability") > 1)
        | F.col("fs_match_weight").isNull()
        | F.isnan("fs_match_weight")
        | (F.abs(F.col("fs_match_weight")) > 10000)
    )
    if scored.where(invalid_score).limit(1).count():
        raise AssertionError("Invalid unified Fellegi-Sunter probabilities or weights")
    log_prior = math.log2(training["prior"] / (1 - training["prior"]))
    component_weight = F.lit(log_prior)
    for name in predictions.columns:
        if name.startswith("fs_mw_"):
            component_weight = component_weight + F.col(name)
    if (
        scored.where(F.abs(component_weight - F.col("fs_match_weight")) > 1e-8)
        .limit(1)
        .count()
    ):
        raise AssertionError(
            "Fellegi-Sunter evidence weights do not reconstruct the score"
        )
    scored.write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(f"{prefix}.unified_pair_scores")
    saved = spark.read.table(f"{prefix}.unified_pair_scores").where(
        F.col("unified_run_id") == run_id
    )
    if (
        saved.count() != count
        or saved.select("id_l", "id_r").distinct().count() != count
        or saved.select(*input_columns)
        .exceptAll(pairs.select(*input_columns))
        .limit(1)
        .count()
    ):
        raise AssertionError(
            "Pair conservation or preservation of original evidence failed"
        )
    if (
        saved.where(
            (F.col("fs_decision") == "proposed_match")
            & (
                F.col("fs_guard_conflict")
                | F.col("fs_guard_review")
                | (
                    (F.col("fs_match_probability") < 0.99)
                    & ~F.coalesce(F.col("fs_original_text_agreement"), F.lit(False))
                )
            )
        )
        .limit(1)
        .count()
    ):
        raise AssertionError("A proposed match bypassed component/model safeguards")
    # Scoring records no human decision, and merge authority must follow the
    # decision exactly: exact_rule / model_proposal merge, everything else is
    # review-only. The resolver enforces the same contract from its side.
    if (
        saved.where("fs_human_decision IS NOT NULL").limit(1).count()
        or saved.where(
            F.col("fs_merge_authority") != merge_authority(F.col("fs_decision"))
        )
        .limit(1)
        .count()
        or saved.where("fs_decision='exact_baseline'").count() != baseline_count
    ):
        raise AssertionError("Exact baseline or merge-authority contract changed")
    decision_counts = [
        r.asDict()
        for r in saved.groupBy("candidate_origin", "fs_decision")
        .count()
        .orderBy("candidate_origin", "fs_decision")
        .collect()
    ]
    transitions = (
        saved.groupBy("candidate_origin", "decision", "fs_decision")
        .count()
        .withColumn("unified_run_id", F.lit(run_id))
    )
    transitions.write.format("delta").mode("append").saveAsTable(
        f"{prefix}.unified_decision_transitions"
    )
    review = (
        saved.where(F.col("fs_decision") != ADDRESS_NON_MATCH)
        .withColumn(
            "_review_rank",
            F.row_number().over(
                Window.partitionBy("candidate_origin", "fs_decision").orderBy(
                    F.sha2(F.concat("id_l", "id_r"), 256)
                )
            ),
        )
        .where(F.col("_review_rank") <= 30)
        .drop("_review_rank")
    )
    review.write.format("delta").mode("append").option(
        "mergeSchema", "true"
    ).saveAsTable(f"{prefix}.unified_review_sample")
    spark.createDataFrame(
        [(run_id, model_payload, json.dumps(training, sort_keys=True))],
        "unified_run_id string,model_json string,training_json string",
    ).write.format("delta").mode("append").saveAsTable(f"{prefix}.unified_models")
    code_hash = hashlib.sha256()
    for relative in (
        "lib/unified_matching.py",
        "altana_matching_components/__init__.py",
        "notebooks/score_unified_candidates.py",
    ):
        path = Path(config["library_root"]) / relative
        code_hash.update(path.read_bytes())
        mlflow.log_text(path.read_text(), "source/" + relative)
    manifest = {
        "version": VERSION,
        "country_corrections_by_origin": [
            row.asDict()
            for row in saved.where("fs_corrected_country IS NOT NULL")
            .groupBy("candidate_origin", "fs_decision")
            .count()
            .orderBy("candidate_origin", "fs_decision")
            .collect()
        ],
        "unified_run_id": run_id,
        "source_string_scoring_run_id": input_run,
        "source_matching_run_id": source_run,
        "input_embedding_run_id": source["input_embedding_run_id"],
        "source_input_fingerprint": source["source_input_fingerprint"],
        "street_component_version": source["street_component_version"],
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "records": record_count,
        "scored_pairs": count,
        "exact_baseline_pairs": baseline_count,
        "origin_counts": origin_counts,
        "decision_counts": decision_counts,
        "training": training,
        "thresholds": THRESHOLDS,
        "acceptance_probability": 0.99,
        "minimum_field_jaro_winkler": 0.85,
        "calibration_status": "unvalidated_unsupervised",
        "approved_merges": 0,
        "probability_threshold_099_pairs": saved.where(
            F.col("fs_match_probability") >= 0.99
        ).count(),
        "original_text_agreement_proposals": saved.where(
            (F.col("fs_decision") == "proposed_match")
            & F.col("fs_original_text_agreement")
        ).count(),
        "original_fields_preserved": True,
        "all_saved_candidates_scored": True,
        "comparison_gamma_mismatches": 0,
        "score_weight_reconstruction_mismatches": 0,
        "embeddings_rebuilt": False,
        "hnsw_indexes_rebuilt": False,
        "source_code_sha256": code_hash.hexdigest(),
        "mlflow_run_id": mlrun.info.run_id,
        "artifact_root": str(artifact_root),
        "config": config,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("splink", "duckdb", "rapidfuzz", "pandas", "pyarrow", "mlflow")
        },
        "notes": [
            "One comparison per field; company exact-match term-frequency adjustment enabled",
            "u sampled from the full record population, not the HNSW-selected candidate distribution",
            "Two unsupervised EM sessions on country/company and country/clean street",
            "All saved pairs registered through Splink's public register_blocked_pairs_for_predict API",
            "Historical probabilities/decisions are preserved; use fs_* for current scores",
            "Conflict, partial/missing component, unsupported parameter, and iteration-cap checks remain separate from the probabilistic model",
            "No entity-level human ground truth or probability calibration; model-reviewed field labels were not used for fitting",
        ],
    }
    payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, allow_nan=False)
    (artifact_root / "manifest.json").write_text(payload)
    mlflow.log_dict(manifest, "unified_manifest.json")
    mlflow.log_metrics(
        {
            "scored_pairs": count,
            "records": record_count,
            "exact_baseline_pairs": baseline_count,
        }
    )
    spark.createDataFrame(
        [(run_id, input_run, "COMPLETED", payload)],
        "unified_run_id string,input_string_scoring_run_id string,status string,manifest_json string",
    ).write.format("delta").mode("append").saveAsTable(f"{prefix}.unified_runs")
dbutils.jobs.taskValues.set(key="unified_run_id", value=run_id)
dbutils.notebook.exit(payload)
