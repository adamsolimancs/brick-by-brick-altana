"""Add text-comparison evidence to saved pairs without changing match decisions."""

from pyspark.sql import functions as F

VERSION = "levenshtein_jaro_winkler_v6"
COMPARISONS = {
    "company": "name_key",
    "street_name": "street_component_road_key",
    "city": "city_key",
}
# Keys of the text before translation, when a field was translated. Two
# records whose originals agree are the same text rendered differently.
ORIGINAL_KEYS = {
    "company": "original_name_key",
    "street_name": "original_road_key",
    "city": "original_city_key",
}
COMPONENTS = ("house_numbers", "units", "buildings", "floors")
# What prepare_comparison_records needs from prepared_records; the notebook
# selects exactly this so the saved component table carries nothing extra.
RECORD_COLUMNS = (
    "unique_id",
    "content_hash",
    "company",
    "street_address",
    "city",
    "country",
    "name_key",
    "city_key",
    "road_key",
    "house_key",
    "unit_key",
    "translation_evidence_json",
)


def prepare_comparison_records(records):
    """Parse each source record before joining to its many candidate pairs."""
    import pandas as pd

    from altana_matching_components import VERSION as COMPONENT_VERSION

    fields = ("road_key", *COMPONENTS, "actions", "warnings")
    schema = "road_key string," + ",".join(
        f"{name} array<string>" for name in fields[1:]
    )

    @F.pandas_udf(schema)
    def components(streets: pd.Series, countries: pd.Series) -> pd.DataFrame:
        from altana_matching_components import parse_street_components

        return pd.DataFrame(
            [
                parse_street_components(
                    None if pd.isna(s) else s, None if pd.isna(c) else c
                )
                for s, c in zip(streets, countries)
            ],
            columns=fields,
        )

    @F.pandas_udf("name string, road string, city string")
    def originals(
        evidence: pd.Series,
        companies: pd.Series,
        streets: pd.Series,
        cities: pd.Series,
        countries: pd.Series,
    ) -> pd.DataFrame:
        import json

        from altana_matching_components import (
            company_name_key,
            parse_street_components,
            text_key,
        )

        def original(blob, field, current):
            value = json.loads(blob).get(f"{field}_original") if blob else None
            return value if value and value != current else None

        rows = []
        for blob, company, street, city, country in zip(
            evidence, companies, streets, cities, countries
        ):
            blob = None if pd.isna(blob) else blob
            company = original(blob, "company", company)
            street = original(blob, "street_address", street)
            city = original(blob, "city", city)
            rows.append(
                (
                    company_name_key(company) if company else None,
                    parse_street_components(
                        street, None if pd.isna(country) else country
                    )["road_key"]
                    if street
                    else None,
                    text_key(city) if city else None,
                )
            )
        return pd.DataFrame(rows, columns=["name", "road", "city"])

    evidence = (
        F.col("translation_evidence_json")
        if "translation_evidence_json" in records.columns
        else F.lit(None).cast("string")
    )
    result = records.withColumn(
        "_street_components", components("street_address", "country")
    ).withColumn(
        "_originals",
        originals(evidence, "company", "street_address", "city", "country"),
    )
    for name in fields:
        result = result.withColumn(
            "street_component_" + name, F.col("_street_components." + name)
        )
    for field, key in ORIGINAL_KEYS.items():
        part = {"company": "name", "street_name": "road", "city": "city"}[field]
        result = result.withColumn(key, F.col("_originals." + part))
    return (
        result.drop("_street_components", "_originals")
        .withColumn("street_component_version", F.lit(COMPONENT_VERSION))
        .withColumn("comparison_record_version", F.lit(VERSION))
    )


def attach_comparison_fields(pairs, records):
    """Left joins preserve input pairs; callers must check the source/hash evidence."""
    if "street_component_road_key" not in records.columns:
        records = prepare_comparison_records(records)
    result = pairs
    for side in ("l", "r"):
        lookup = records.select(
            F.col("unique_id").alias(f"_prepared_record_id_{side}"),
            F.col("content_hash").alias(f"_prepared_content_hash_{side}"),
            *[
                F.col(key).alias(f"comparison_{name}_{side}")
                for name, key in COMPARISONS.items()
            ],
            F.col("house_key").alias(f"comparison_building_{side}"),
            F.col("unit_key").alias(f"comparison_suite_{side}"),
            F.col("road_key").alias(f"previous_street_name_{side}"),
            *[
                F.col(f"street_component_{name}").alias(f"parsed_{name}_{side}")
                for name in COMPONENTS
            ],
            F.col("street_component_actions").alias(f"street_parse_actions_{side}"),
            F.col("street_component_warnings").alias(f"street_parse_warnings_{side}"),
            *[
                (
                    F.col(key) if key in records.columns else F.lit(None).cast("string")
                ).alias(f"{key}_{side}")
                for key in ORIGINAL_KEYS.values()
            ],
        )
        result = result.join(
            lookup, F.col(f"id_{side}") == F.col(f"_prepared_record_id_{side}"), "left"
        )
    for field, key in ORIGINAL_KEYS.items():
        left, right = F.col(f"{key}_l"), F.col(f"{key}_r")
        result = result.withColumn(
            f"original_{field}_agreement",
            left.isNotNull() & right.isNotNull() & (left == right),
        )
    any_conflict = F.lit(False)
    review_required = F.lit(False)
    # A warning that both sides raise over the same parsed values ("Suite 883,
    # Apt 187" on each record) describes a shared ambiguity, not a difference.
    ignorable_warnings = [F.lit("road_number_tokens_retained")]
    for component in COMPONENTS:
        left, right = F.col(f"parsed_{component}_l"), F.col(f"parsed_{component}_r")
        has_left = F.coalesce(F.size(left), F.lit(0)) > 0
        has_right = F.coalesce(F.size(right), F.lit(0)) > 0
        both = has_left & has_right
        equal = F.array_sort(left).eqNullSafe(F.array_sort(right))
        overlap = F.coalesce(F.arrays_overlap(left, right), F.lit(False))
        conflict = both & ~overlap
        partial = both & overlap & ~equal
        result = (
            result.withColumn(f"parsed_{component}_conflict", conflict)
            .withColumn(f"parsed_{component}_missing_one_side", has_left != has_right)
            .withColumn(f"parsed_{component}_partial_agreement", partial)
        )
        any_conflict = any_conflict | F.col(f"parsed_{component}_conflict")
        review_required = review_required | conflict | partial | (has_left != has_right)
        ignorable_warnings.append(F.when(both & equal, "multiple_" + component))
    ignorable = F.filter(
        F.array(*ignorable_warnings), lambda warning: warning.isNotNull()
    )
    for side in ("l", "r"):
        review_required = review_required | (
            F.size(F.array_except(F.col(f"street_parse_warnings_{side}"), ignorable))
            > 0
        )
    return result.withColumn("parsed_address_conflict", any_conflict).withColumn(
        "parsed_address_review_required", review_required
    )


def invalid_source_condition():
    result = F.lit(False)
    for side in ("l", "r"):
        result = (
            result
            | F.col(f"_prepared_record_id_{side}").isNull()
            | ~F.col(f"content_hash_{side}").eqNullSafe(
                F.col(f"_prepared_content_hash_{side}")
            )
        )
    return result


def add_string_features(pairs):
    """Spark-native Levenshtein plus Arrow-batched RapidFuzz Jaro–Winkler.

    Empty/null comparison keys are unknown, including when both sides are empty.
    Existing score, decision, building, and suite columns are left untouched.
    """
    import pandas as pd

    @F.pandas_udf("double")
    def jaro_winkler(left: pd.Series, right: pd.Series) -> pd.Series:
        from rapidfuzz.distance import JaroWinkler

        return pd.Series(
            [
                None
                if pd.isna(a) or pd.isna(b) or not a.strip() or not b.strip()
                else JaroWinkler.normalized_similarity(a, b, prefix_weight=0.1)
                for a, b in zip(left, right)
            ],
            dtype="float64",
        )

    result = pairs
    for name in COMPARISONS:
        left, right = F.col(f"comparison_{name}_l"), F.col(f"comparison_{name}_r")
        known = (F.length(F.trim(left)) > 0) & (F.length(F.trim(right)) > 0)
        distance = f"{name}_levenshtein_distance"
        result = result.withColumn(distance, F.when(known, F.levenshtein(left, right)))
        result = result.withColumn(
            f"{name}_levenshtein_similarity",
            F.when(
                known,
                F.lit(1.0)
                - F.col(distance)
                / F.greatest(F.length(left), F.length(right), F.lit(1)),
            ),
        ).withColumn(f"{name}_jaro_winkler_similarity", jaro_winkler(left, right))
    return result
