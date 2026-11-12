"""Bounded candidate completion; existing negative evidence is never overwritten."""

from collections import defaultdict
from itertools import combinations


def missing_cluster_pairs(
    record_ids,
    accepted_pairs,
    known_pairs,
    *,
    max_group_size=100,
    max_new_pairs=100_000,
):
    """Complete small connected proposal groups, not the whole record population.

    Components define where to LOOK, never permission to merge. Skip a whole
    component when its budget cannot be satisfied; resolution remains conservative.
    """
    if max_group_size < 2 or max_new_pairs < 0:
        raise ValueError("Invalid cluster completion budget")
    parent = {key: key for key in record_ids}

    def root(key):
        if key not in parent:
            raise ValueError("Completion pair references an unknown record")
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for left, right in accepted_pairs:
        a, b = sorted((root(left), root(right)))
        parent[b] = a
    groups = defaultdict(list)
    for key in sorted(parent):
        groups[root(key)].append(key)
    known = set(known_pairs)
    result, skipped = [], []
    for members in sorted(groups.values()):
        if len(members) < 3:
            continue
        if len(members) > max_group_size:
            skipped.append(dict(record_ids=members, reason="group_size_budget"))
            continue
        missing = [pair for pair in combinations(members, 2) if pair not in known]
        if len(result) + len(missing) > max_new_pairs:
            skipped.append(dict(record_ids=members, reason="new_pair_budget"))
            continue
        result.extend(missing)
    return result, skipped


def completion_features(pair_ids, prepared, components):
    """Construct the same source, legacy guards and parsed comparisons as retrieval."""
    from pyspark.sql import functions as F
    from lib.string_features import (
        add_string_features,
        attach_comparison_fields,
        invalid_source_condition,
    )

    result = pair_ids
    fields = (
        "content_hash",
        "company",
        "street_address",
        "city",
        "state",
        "country",
        "country_key",
        "region_key",
        "house_key",
        "unit_key",
        "number_signature",
        "baseline_id",
        "country_inferred",
    )
    for side in ("l", "r"):
        lookup = prepared.select(
            F.col("unique_id").alias(f"_id_{side}"),
            *[F.col(f).alias(f"{f}_{side}") for f in fields],
        )
        result = result.join(
            lookup, F.col(f"id_{side}") == F.col(f"_id_{side}"), "left"
        )
    for field, flag in (
        ("country_key", "country_conflict"),
        ("region_key", "region_conflict"),
        ("number_signature", "numeric_conflict"),
    ):
        result = result.withColumn(flag, F.col(f"{field}_l") != F.col(f"{field}_r"))
    for field, flag in (("house_key", "house_conflict"), ("unit_key", "unit_conflict")):
        result = result.withColumn(
            flag,
            (F.col(f"{field}_l") != "")
            & (F.col(f"{field}_r") != "")
            & (F.col(f"{field}_l") != F.col(f"{field}_r")),
        )
    result = (
        result.withColumn(
            "unit_missing_one_side",
            (F.col("unit_key_l") == "") != (F.col("unit_key_r") == ""),
        )
        .withColumn(
            "source_exact_baseline", F.col("baseline_id_l") == F.col("baseline_id_r")
        )
        .withColumn("candidate_origin", F.lit("cluster_completion"))
    )
    for side in ("l", "r"):
        result = result.withColumn(
            f"fs_country_inferred_{side}",
            F.col(f"country_inferred_{side}").cast("boolean"),
        )
    result = attach_comparison_fields(result, components)
    if result.where(invalid_source_condition()).limit(1).count():
        raise ValueError("Completion source identity/hash mismatch")
    return add_string_features(result)


def score_completion(
    spark, prepared, components, pair_ids, model, *, max_pairs=100_000
):
    """Use the saved model and FULL population, preserving term frequencies."""
    import re
    import duckdb
    import pyarrow as pa
    from splink import DuckDBAPI, Linker
    from lib.unified_matching import predict_saved_pairs, register_jaro_winkler

    snapshot_fields = ("unique_id", "content_hash", "name_key", "city_key")
    if (
        components.select(*snapshot_fields)
        .exceptAll(prepared.select(*snapshot_fields))
        .limit(1)
        .count()
        or prepared.select(*snapshot_fields)
        .exceptAll(components.select(*snapshot_fields))
        .limit(1)
        .count()
    ):
        raise ValueError(
            "Completion component population differs from the fitted snapshot"
        )
    keys = components.select(
        "unique_id", "name_key", "city_key", "street_component_road_key"
    ).join(prepared.select("unique_id", "country_key"), "unique_id")
    records = []
    for row in keys.collect():
        records.append(
            dict(
                unique_id=row.unique_id,
                company=row.name_key or None,
                street_name=row.street_component_road_key or None,
                city=row.city_key or None,
                country=row.country_key or None,
            )
        )
    with duckdb.connect(":memory:") as connection:
        connection.execute("SET memory_limit='2GB'")
        connection.execute("SET threads=4")
        register_jaro_winkler(connection)
        connection.register("records_arrow", pa.Table.from_pylist(records))
        connection.execute(
            "CREATE TABLE clean_records AS SELECT * FROM records_arrow ORDER BY unique_id"
        )
        connection.register(
            "pairs_arrow",
            pa.table(
                {
                    "join_key_l": [p[0] for p in pair_ids],
                    "join_key_r": [p[1] for p in pair_ids],
                }
            ),
        )
        connection.execute(
            "CREATE TABLE saved_id_pairs AS SELECT '0' match_key,* FROM pairs_arrow"
        )
        api = DuckDBAPI(connection)
        linker = Linker(api.register("clean_records"), model)
        predict_saved_pairs(connection, linker, max_pairs)
        names = [
            c[0]
            for c in connection.execute(
                "SELECT * FROM unified_predictions LIMIT 0"
            ).description
        ]
        projection = [
            "unique_id_l AS id_l",
            "unique_id_r AS id_r",
            "match_weight AS fs_match_weight",
            "match_probability AS fs_match_probability",
        ]
        projection += [
            f'"{name}" AS "fs_{name}"'
            for name in names
            if re.fullmatch(r"(?:gamma|mw|tf)_[A-Za-z0-9_]+", name)
        ]
        predictions = connection.execute(
            "SELECT " + ",".join(projection) + " FROM unified_predictions"
        ).fetch_arrow_table()

    def spark_type(field):
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            return "string"
        if pa.types.is_integer(field.type):
            return "long"
        return "double"

    schema = ",".join(
        f"{field.name} {spark_type(field)}" for field in predictions.schema
    )
    # Explicit schema preserves all-null optional evidence columns.
    rows = [
        {
            k: (
                float(v)
                if v is not None and spark_type(predictions.schema.field(k)) == "double"
                else v
            )
            for k, v in row.items()
        }
        for row in predictions.to_pylist()
    ]
    return spark.createDataFrame(rows, schema)
