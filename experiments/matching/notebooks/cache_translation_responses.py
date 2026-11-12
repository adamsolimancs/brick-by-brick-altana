# Databricks notebook source
"""Checkpoint committed inference responses outside the refreshable Silver pipeline.

Run before refresh to bootstrap/protect old responses and after ALL_DONE to retain
responses even when downstream cleaning fails. Never invoke an endpoint here.
"""

from pyspark.sql import functions as F
from delta.tables import DeltaTable

CACHE = "addressresolution.default.translation_response_cache"


def cache_key():
    # Contract shared with translation._cached_inference; source counts and
    # validation/glossary rules deliberately do not invalidate raw responses.
    return F.sha2(
        F.to_json(
            F.struct(
                "request_key",
                "model",
                "model_parameters",
                "prompt_version",
                "field_name",
                "input_text",
                "country_context",
                "request_prompt",
            )
        ),
        256,
    )


spark.sql(f"""CREATE TABLE IF NOT EXISTS {CACHE} (
    cache_key STRING, ai_result STRUCT<result: STRING,errorMessage: STRING>,
    cached_at TIMESTAMP, source_table STRING
) USING DELTA""")
for source in (
    "addressresolution.silver.providers_translation_raw",
    "addressresolution.silver.providers_translation_escalation_raw",
):
    if not spark.catalog.tableExists(source):
        continue
    rows = (
        spark.read.table(source)
        .where(
            F.col("ai_result.errorMessage").isNull()
            & (F.length(F.trim("ai_result.result")) > 0)
        )
        .select(cache_key().alias("cache_key"), "ai_result")
        .distinct()
    )
    if rows.groupBy("cache_key").count().where("count > 1").limit(1).count():
        raise ValueError("Raw inference contains conflicting responses for one request")
    # First committed successful response wins. Cache misses use exactly these
    # raw responses on subsequent runs; validation is always evaluated afresh.
    (
        DeltaTable.forName(spark, CACHE)
        .alias("cache")
        .merge(
            rows.withColumn("cached_at", F.current_timestamp())
            .withColumn("source_table", F.lit(source))
            .alias("new"),
            "cache.cache_key = new.cache_key",
        )
        .whenNotMatchedInsertAll()
        .execute()
    )
