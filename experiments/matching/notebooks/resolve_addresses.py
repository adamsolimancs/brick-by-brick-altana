# Databricks notebook source
"""Resolve one explicitly completed scoring run and stage the two business outputs."""

import json
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import Window, functions as F

for name, default in {
    "library_root": "",
    "output_catalog": "addressresolution",
    "output_schema": "matching",
    "unified_run_id": "",
    "deduplication_run_id": "",
    "silver_workflow_run_id": "manual",
}.items():
    dbutils.widgets.text(name, default)
config = {
    name: dbutils.widgets.get(name)
    for name in (
        "library_root",
        "output_catalog",
        "output_schema",
        "unified_run_id",
        "deduplication_run_id",
        "silver_workflow_run_id",
    )
}
if (
    config["output_catalog"] != "addressresolution"
    or config["output_schema"] != "matching"
):
    raise ValueError("Business resolution uses addressresolution.matching staging")
for name in ("unified_run_id", "deduplication_run_id"):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", config[name]):
        raise ValueError(f"Invalid {name}")
sys.path.insert(0, config["library_root"])
from lib.resolution import (  # noqa: E402
    MAPPING_SCHEMA,
    REVIEW_DECISIONS,
    SELECTION_RULE,
    VERSION,
    attach_incomplete,
    fill_judgment_candidates,
    judgment_candidates,
    resolve_records,
    settle_same_company,
)
from lib import address_judge  # noqa: E402
from lib.matching import FIELDS, digest  # noqa: E402
from lib.unified_matching import ADDRESS_NON_MATCH  # noqa: E402
from lib.cluster_completion import (  # noqa: E402
    completion_features,
    missing_cluster_pairs,
    score_completion,
)
from lib.unified_matching import (  # noqa: E402
    add_decisions,
    route_address_conflicts,
    expected_gamma,
    THRESHOLDS,
)  # noqa: E402

prefix = f"{config['output_catalog']}.{config['output_schema']}"
run_id = config["deduplication_run_id"]
score_id = config["unified_run_id"]
for name in (
    "resolved_records",
    "resolution_runs",
    "resolution_pair_scores",
    "resolution_completion_audit",
):
    if (
        spark.catalog.tableExists(f"{prefix}.{name}")
        and spark.read.table(f"{prefix}.{name}")
        .where(F.col("deduplication_run_id") == run_id)
        .limit(1)
        .count()
    ):
        raise ValueError(
            "Partial or completed resolution already exists. Start a fresh workflow run."
        )
completed = (
    spark.read.table(f"{prefix}.unified_runs")
    .where((F.col("unified_run_id") == score_id) & (F.col("status") == "COMPLETED"))
    .collect()
)
if len(completed) != 1:
    raise ValueError("Exactly one completed scoring run is required")
manifest = json.loads(completed[0]["manifest_json"])
source_run = manifest["source_matching_run_id"]
prepared = spark.read.table(f"{prefix}.prepared_records").where(
    F.col("experiment_run_id") == source_run
)
if prepared.count() != manifest["records"] or manifest["records"] > 50_000:
    raise ValueError(
        "Source snapshot changed or exceeds the bounded resolution capacity"
    )
pairs = spark.read.table(f"{prefix}.unified_pair_scores").where(
    F.col("unified_run_id") == score_id
)
if pairs.count() != manifest["scored_pairs"]:
    raise ValueError("Completed scoring pair count changed")
records = [row.asDict() for row in prepared.collect()]
if (
    digest(
        sorted(
            (
                record["unique_id"],
                record["content_hash"],
                record["translation_evidence_json"],
            )
            for record in records
        )
    )
    != manifest["source_input_fingerprint"]
):
    raise ValueError("Resolution source fingerprint differs from the scoring snapshot")

# Expand only small connected groups of already passing pairs. Existing scored
# nonmatches stay in the known set and are never reinterpreted as missing evidence.
known = [(r.id_l, r.id_r) for r in pairs.select("id_l", "id_r").toLocalIterator()]
proposals = [
    (r.id_l, r.id_r)
    for r in pairs.where(F.col("fs_decision").isin("exact_baseline", "proposed_match"))
    .select("id_l", "id_r")
    .collect()
]
missing, skipped = missing_cluster_pairs(
    [r["unique_id"] for r in records],
    proposals,
    known,
    max_new_pairs=min(100_000, max(0, 2_000_000 - len(known))),
)
if missing:
    models = (
        spark.read.table(f"{prefix}.unified_models")
        .where(F.col("unified_run_id") == score_id)
        .collect()
    )
    if len(models) != 1:
        raise ValueError("Completion requires exactly one saved scoring model")
    model, training = (
        json.loads(models[0].model_json),
        json.loads(models[0].training_json),
    )
    components = spark.read.table(f"{prefix}.string_record_components").where(
        F.col("string_scoring_run_id") == manifest["source_string_scoring_run_id"]
    )
    if components.count() != len(records) or components.select(
        "unique_id"
    ).distinct().count() != len(records):
        raise ValueError("Completion component snapshot changed")
    new_ids = spark.createDataFrame(missing, "id_l string,id_r string")
    features = completion_features(new_ids, prepared, components)
    predictions = score_completion(spark, prepared, components, missing, model)
    # Match the persisted scorer's exact numeric types, including integer gamma
    # widths, so runs with/without completion share one stable Delta schema.
    for field in predictions.schema:
        if field.name in pairs.columns:
            predictions = predictions.withColumn(
                field.name, F.col(field.name).cast(pairs.schema[field.name].dataType)
            )
    added = features.join(predictions, ["id_l", "id_r"], "inner")
    # Geography uses donor consensus across ALL pairs, including new evidence.
    # Reapply decisions to the union so a new conflicting donor cannot bypass it.
    pairs = route_address_conflicts(
        add_decisions(
            pairs.unionByName(added, allowMissingColumns=True),
            training,
            pair_geography=True,
        )
    )
    invalid = F.lit(False)
    for name in THRESHOLDS:
        invalid = invalid | ~F.col("fs_gamma_" + name).eqNullSafe(expected_gamma(name))
    if (
        pairs.where(
            invalid
            | F.col("fs_match_probability").isNull()
            | F.isnan("fs_match_probability")
            | (F.col("fs_match_probability") < 0)
            | (F.col("fs_match_probability") > 1)
        )
        .limit(1)
        .count()
    ):
        raise AssertionError("Invalid completion score or comparison level")
pairs.withColumn("unified_run_id", F.lit(score_id)).withColumn(
    "unified_version", F.lit(manifest["version"])
).withColumn("deduplication_run_id", F.lit(run_id)).write.format("delta").mode(
    "append"
).option("mergeSchema", "true").saveAsTable(f"{prefix}.resolution_pair_scores")
pairs = spark.read.table(f"{prefix}.resolution_pair_scores").where(
    F.col("deduplication_run_id") == run_id
)
if pairs.count() != len(known) + len(missing) or pairs.select(
    "id_l", "id_r"
).distinct().count() != len(known) + len(missing):
    raise AssertionError("Completion pair conservation failed")
spark.createDataFrame(
    [
        (
            run_id,
            json.dumps(
                {
                    "original_pairs": len(known),
                    "added_pairs": len(missing),
                    "max_group_size": 100,
                    "max_new_pairs": 100_000,
                    "skipped_components": skipped,
                    "model_run_id": score_id,
                    "completion_code_sha256": hashlib.sha256(
                        (
                            Path(config["library_root"]) / "lib/cluster_completion.py"
                        ).read_bytes()
                    ).hexdigest(),
                }
            ),
        )
    ],
    "deduplication_run_id string,manifest_json string",
).write.format("delta").mode("append").saveAsTable(
    f"{prefix}.resolution_completion_audit"
)
relevant_pairs = pairs.where(
    F.col("fs_decision").isin("exact_baseline", "proposed_match")
    | (
        F.col("fs_decision").isin(*REVIEW_DECISIONS)
        & (F.col("fs_match_probability") > 0.80)
    )
)
mapping = resolve_records(records, [row.asDict() for row in relevant_pairs.collect()])
incomplete_columns = [
    "provider",
    "provider_id",
    *FIELDS,
    "created_at",
    "updated_at",
    "country_inferred",
    "routed_from",
]
incomplete = [
    row.asDict()
    for row in spark.read.table("addressresolution.silver.dlq_incomplete")
    .select(*[F.col(c).cast("string") for c in incomplete_columns], "dlq_reasons")
    .collect()
]
by_incomplete_identity = {
    (row["provider"], row["provider_id"]): row for row in incomplete
}
# Same-company review pairs with matching numbers whose words no text rule
# reconciles (romanization, translation) go to the address judge model. Answers
# are kept by request, so a rerun reuses them and only new requests are sent.
# A failed or empty answer is not stored: the pair stays on hold and is asked
# again on the next run.
# A blank-city row whose street no rule finds is judged the same way against
# the company's records with the same numbers, on the street alone.
judge_requests = address_judge.requests(
    judgment_candidates(mapping, records)
) + address_judge.requests(
    fill_judgment_candidates(records, incomplete), street_only=True
)
judgments = {}
if judge_requests:
    judgment_table = f"{prefix}.same_company_address_judgments"
    asked = spark.createDataFrame(
        judge_requests, "id_l string, id_r string, request_key string, prompt string"
    )
    unique_requests = asked.select("request_key", "prompt").distinct()
    if spark.catalog.tableExists(judgment_table):
        unique_requests = unique_requests.join(
            spark.read.table(judgment_table).select("request_key"),
            "request_key",
            "left_anti",
        )
    (
        unique_requests.withColumn(
            "ai_result", F.expr(address_judge.ai_query_sql("prompt"))
        )
        .select(
            "request_key",
            "prompt",
            F.lit(address_judge.MODEL).alias("model"),
            F.lit(address_judge.PROMPT_VERSION).alias("prompt_version"),
            F.col("ai_result.result").alias("response"),
            F.current_timestamp().alias("judged_at"),
            F.lit(run_id).alias("first_deduplication_run_id"),
        )
        .where(F.length(F.trim("response")) > 0)
        .write.format("delta")
        .mode("append")
        .saveAsTable(judgment_table)
    )
    answers = spark.read.table(judgment_table).select("request_key", "response")
    if answers.groupBy("request_key").count().where("count > 1").limit(1).count():
        raise AssertionError("Conflicting address judgments for one request")
    judgments = address_judge.judgments(
        row.asDict() for row in asked.join(answers, "request_key", "inner").collect()
    )
# Settle the same-company review pairs: the same place written twice is merged
# (by text rule, or by a judgment at 0.90 or above), two clearly different
# addresses are released to Gold as separate records. Everything else, and
# every pair with a different number, stays on hold.
mapping = settle_same_company(mapping, records, judgments)
# Fold incomplete Silver rows into the one resolved entity that can complete
# them. They join as duplicates, so the Gold record never changes; rows without
# a unique fit stay parked in dlq_incomplete.
mapping, filled = attach_incomplete(mapping, records, incomplete, judgments)
filled_rows = []
for completed in filled:
    source = by_incomplete_identity[completed["provider"], completed["provider_id"]]
    values = dict(
        completed,
        experiment_run_id=source_run,
        input_route="incomplete_filled",
        source_record_json=json.dumps(source, ensure_ascii=False, sort_keys=True),
        translation_evidence_json="{}",
        source_identity_count=1,
        input_reasons="",
    )
    filled_rows.append(tuple(values.get(field.name) for field in prepared.schema))
filled_count = len(filled_rows)
prepared = prepared.unionByName(spark.createDataFrame(filled_rows, prepared.schema))
resolved = (
    prepared.withColumnRenamed("unique_id", "record_id")
    .join(spark.createDataFrame(mapping, MAPPING_SCHEMA), "record_id", "inner")
    .withColumn("input_reasons", F.lit(""))
)
reason_type = "array<struct<reason_code:string,pair_count:bigint>>"
if "fs_address_conflict_reasons" in pairs.columns:
    non_matches = (
        pairs.where(F.col("fs_decision") == ADDRESS_NON_MATCH)
        .select(
            F.explode(F.array("id_l", "id_r")).alias("record_id"),
            "fs_address_conflict_reasons",
        )
        .select(
            "record_id", F.explode("fs_address_conflict_reasons").alias("reason_code")
        )
        .groupBy("record_id", "reason_code")
        .agg(F.count("*").alias("pair_count"))
        .groupBy("record_id")
        .agg(
            F.sort_array(F.collect_list(F.struct("reason_code", "pair_count"))).alias(
                "address_non_match_reasons"
            )
        )
    )
    resolved = resolved.join(non_matches, "record_id", "left")
else:
    resolved = resolved.withColumn(
        "address_non_match_reasons", F.lit(None).cast(reason_type)
    )
resolved = resolved.withColumn(
    "address_non_match_reasons",
    F.coalesce("address_non_match_reasons", F.array().cast(reason_type)),
)
snapshot = spark.read.table(f"{prefix}.input_snapshots").where(
    F.col("experiment_run_id") == source_run
)
rejected = snapshot.where(F.col("input_reasons") != "")
rejected_count = rejected.count()
if rejected_count:
    # Preserve every rejected observation, including identical repeated source IDs.
    identity = [
        "provider",
        "provider_id",
        "source_record_json",
        "translation_evidence_json",
        "input_route",
    ]
    rejected = (
        rejected.withColumn(
            "_occurrence",
            F.row_number().over(Window.partitionBy(*identity).orderBy("input_reasons")),
        )
        .withColumn(
            "record_id",
            F.concat(
                F.lit("rejected_"),
                F.sha2(F.to_json(F.struct(*identity, "_occurrence")), 256),
            ),
        )
        .drop("_occurrence")
        .withColumn("is_duplicate", F.lit(False))
        .withColumn("is_canonical", F.lit(False))
        .withColumn("deduplication_status", F.lit("input_rejected"))
        .withColumn(
            "deduplication_reasons",
            F.array(
                F.concat(
                    F.lit("Input excluded from resolution. "), F.col("input_reasons")
                )
            ),
        )
        .withColumn("needs_match_review", F.lit(True))
    )
    resolved = resolved.unionByName(rejected, allowMissingColumns=True)
resolved = (
    resolved.withColumn("country_inferred", F.col("country_inferred").cast("boolean"))
    .withColumn("created_at", F.try_to_timestamp("created_at"))
    .withColumn("updated_at", F.try_to_timestamp("updated_at"))
    .withColumn("deduplication_run_id", F.lit(run_id))
    .withColumn("scoring_run_id", F.lit(score_id))
    .withColumn("source_snapshot_run_id", F.lit(source_run))
    .withColumn("resolution_rule_version", F.lit(VERSION))
    .withColumn("canonical_selection_rule", F.lit(SELECTION_RULE))
    .withColumn("scoring_model_version", F.lit(manifest["version"]))
    .withColumn("match_probability_calibration", F.lit(manifest["calibration_status"]))
    .withColumn("silver_workflow_run_id", F.lit(config["silver_workflow_run_id"]))
    .withColumn("resolved_at", F.current_timestamp())
)
resolved.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
    f"{prefix}.resolved_records"
)
saved = spark.read.table(f"{prefix}.resolved_records").where(
    F.col("deduplication_run_id") == run_id
)
summary = (
    saved.agg(
        F.count("*").alias("silver_count"),
        F.countDistinct("record_id").alias("unique_records"),
        F.count_if("is_canonical").alias("gold_count"),
        F.count_if("is_duplicate").alias("duplicate_count"),
        F.countDistinct("entity_id").alias("entities"),
        F.countDistinct(
            F.when(
                F.expr(
                    "is_canonical AND provider IS NOT NULL AND provider_id IS NOT NULL"
                ),
                F.struct("provider", "provider_id"),
            )
        ).alias("unique_canonical_provider_keys"),
    )
    .first()
    .asDict()
)
if (
    summary["silver_count"] != snapshot.count() + filled_count
    or summary["unique_records"] != summary["silver_count"]
    or summary["gold_count"] + summary["duplicate_count"]
    != manifest["records"] + filled_count
    or saved.where("filled_field IS NOT NULL AND NOT is_duplicate").limit(1).count()
    or saved.where("filled_field IS NOT NULL").count() != filled_count
    or summary["gold_count"] != summary["entities"]
    or summary["gold_count"] != summary["unique_canonical_provider_keys"]
):
    raise AssertionError("Record conservation or canonical uniqueness failed")
canonical_ids = saved.where("is_canonical").select(
    F.col("record_id").alias("canonical_record_id")
)
if (
    saved.where("is_duplicate")
    .join(canonical_ids, "canonical_record_id", "left_anti")
    .limit(1)
    .count()
):
    raise AssertionError("Duplicate row has no canonical record")
if saved.where("is_duplicate AND size(deduplication_evidence) = 0").limit(1).count():
    raise AssertionError("A removed duplicate is missing direct match evidence")
spark.createDataFrame(
    [
        (
            run_id,
            score_id,
            datetime.now(timezone.utc),
            "COMPLETED",
            summary["silver_count"],
            summary["gold_count"],
            summary["duplicate_count"],
            rejected_count,
            filled_count,
        )
    ],
    "deduplication_run_id string, scoring_run_id string, completed_at timestamp, status string, "
    "silver_count long, gold_count long, duplicate_count long, rejected_count long, "
    "incomplete_filled_count long",
).write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(
    f"{prefix}.resolution_runs"
)
dbutils.notebook.exit(
    json.dumps(
        dict(
            summary,
            deduplication_run_id=run_id,
            incomplete_filled_count=filled_count,
        )
    )
)
