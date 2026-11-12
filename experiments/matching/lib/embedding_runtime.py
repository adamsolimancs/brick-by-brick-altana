"""Databricks I/O for the isolated embedding experiment (no production writes)."""

import re

import pyarrow as pa


def settings(dbutils):
    defaults = {
        "output_catalog": "addressresolution",
        "output_schema": "matching_experiments_aes10130",
        "source_run_id": "latest",
        "source_embedding_run_id": "",
        "embedding_run_id": "",
        "experiment_id": "",
        "max_records": "50000",
        "max_candidate_pairs": "2000000",
        "top_k": "20",
        "search_k": "100",
        "min_cosine": "0.70",
    }
    for name, default in defaults.items():
        dbutils.widgets.text(name, default)
    result = {name: dbutils.widgets.get(name) for name in defaults}
    if result["output_catalog"] != "addressresolution" or not (
        result["output_schema"] == "matching"
        or re.fullmatch(r"matching_experiments_[a-zA-Z0-9_]+", result["output_schema"])
    ):
        raise ValueError(
            "Writes must stay in addressresolution.matching_experiments_<owner>"
        )
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", result["embedding_run_id"]):
        raise ValueError(
            "A unique embedding_run_id is required (the Databricks job run ID)"
        )
    result["prefix"] = f"{result['output_catalog']}.{result['output_schema']}"
    for name in ("max_records", "max_candidate_pairs", "top_k", "search_k"):
        result[name] = int(result[name])
    result["min_cosine"] = float(result["min_cosine"])
    if not (
        2 <= result["max_records"] <= 50_000
        and 1 <= result["max_candidate_pairs"] <= 2_000_000
        and 1 <= result["top_k"] <= result["search_k"] <= 500
        and -1 <= result["min_cosine"] <= 1
    ):
        raise ValueError("Invalid embedding experiment budgets")
    return result


def require_fresh_outputs(spark, tables, embedding_run_id):
    """Fail before any append if a task is repaired after writing partial results."""
    from pyspark.sql import functions as F

    for table in tables:
        if (
            spark.catalog.tableExists(table)
            and spark.read.table(table)
            .where(F.col("embedding_run_id") == embedding_run_id)
            .limit(1)
            .count()
        ):
            raise ValueError(
                "This run already has outputs; start a fresh job run instead of repairing"
            )


def append_result(spark, arrow, table, embedding_run_id, source_run_id):
    """Bounded scalar result writes, with explicit schemas even when empty."""
    from pyspark.sql import functions as F, types as T

    fields = []
    for field in arrow.schema:
        if pa.types.is_boolean(field.type):
            dtype = T.BooleanType()
        elif pa.types.is_integer(field.type):
            dtype = T.LongType()
        elif pa.types.is_floating(field.type):
            dtype = T.DoubleType()
        elif pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            dtype = T.StringType()
        else:
            raise TypeError(f"Unsupported output type {field}")
        fields.append(T.StructField(field.name, dtype))
    for batch in arrow.to_batches(max_chunksize=25_000) or [None]:
        frame = spark.createDataFrame(
            batch.to_pylist() if batch else [], T.StructType(fields)
        )
        (
            frame.withColumn("embedding_run_id", F.lit(embedding_run_id))
            .withColumn("source_run_id", F.lit(source_run_id))
            .write.format("delta")
            .mode("append")
            .saveAsTable(table)
        )
