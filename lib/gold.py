"""Conservative entity/address baseline; no AI calls or cross-product matching.

Exact case/whitespace-normalized company AND every address component must agree.
Entity IDs identify provisional entity/address clusters, not global legal entities.
"""

from functools import reduce

from pyspark.sql import DataFrame, Window, functions as F

FIELDS = ("company", "street_address", "city", "state", "country")
RULE_VERSION = "exact_entity_address_v1"
NON_LATIN = r"[^\p{IsLatin}\p{IsCommon}\p{IsInherited}]"


def normalize(column):
    return F.lower(F.trim(F.regexp_replace(column, r"[\s\p{Z}]+", " ")))


def prepare_gold_input(df: DataFrame) -> DataFrame:
    """Retain every input row, flag quality and conflicting source identities."""
    normalized = df
    for field in FIELDS:
        normalized = normalized.withColumn(f"_{field}", normalize(F.col(field)))
    normalized = normalized.withColumn(
        "entity_id",
        F.sha2(F.to_json(F.struct(*[F.col(f"_{f}") for f in FIELDS])), 256),
    )
    bounds = {
        "company": (2, 100),
        "street_address": (4, 200),
        "city": (2, 50),
        "state": (2, 50),
    }
    reasons = []
    for field in ("provider", "provider_id"):
        value = normalize(F.col(field).cast("string"))
        reasons.append(
            F.when(value.isNull() | (value == ""), F.lit(f"{field}:missing"))
        )
    for field in FIELDS:
        value = F.col(f"_{field}")
        reasons.append(
            F.when(value.isNull() | (value == ""), F.lit(f"{field}:missing"))
        )
        if field in bounds:
            low, high = bounds[field]
            reasons.append(
                F.when(
                    value.isNotNull()
                    & (~F.length(value).between(low, high) | value.rlike(NON_LATIN)),
                    F.lit(f"{field}:invalid_text"),
                )
            )
    reasons.append(
        F.when(
            F.col("_country").isNotNull() & ~F.col("_country").rlike("^[a-z]{2}$"),
            F.lit("country:expected_alpha2"),
        )
    )
    # Do not let one source ID contribute to multiple entities. Count signatures
    # before filtering invalid content so conflicting malformed rows also count.
    identities = normalized.groupBy("provider", "provider_id").agg(
        F.countDistinct("entity_id").alias("_identity_variants")
    )
    normalized = normalized.join(identities, ["provider", "provider_id"], "left")
    reasons.append(
        F.when(F.col("_identity_variants") > 1, F.lit("source_id:conflicting_records"))
    )
    return normalized.withColumn(
        "gold_input_reasons", F.filter(F.array(*reasons), lambda x: x.isNotNull())
    )


def build_gold(prepared: DataFrame) -> DataFrame:
    """One provisional entity/address cluster, with deterministic whole-row selection.

    Retain punctuation, accents, house/unit numbers and full company names in keys.
    Never merge by address alone, strip legal suffixes, or chain fuzzy matches.
    """
    accepted = prepared.where(F.size("gold_input_reasons") == 0)
    # Support is by distinct source ID, not raw row frequency or date recency.
    variants = accepted.groupBy("entity_id", *FIELDS).agg(
        F.countDistinct(F.struct("provider", "provider_id")).alias("variant_support")
    )
    ranking = Window.partitionBy("entity_id").orderBy(
        F.desc("variant_support"), *[F.col(f).asc() for f in FIELDS]
    )
    winner = variants.withColumn("_rank", F.row_number().over(ranking)).where(
        "_rank = 1"
    )
    evidence = accepted.groupBy("entity_id").agg(
        F.sort_array(F.collect_set(F.struct("provider", "provider_id"))).alias(
            "source_records"
        ),
        F.countDistinct("provider").alias("provider_count"),
        F.countDistinct(F.struct("provider", "provider_id")).alias(
            "source_record_count"
        ),
        F.count("*").alias("input_row_count"),
        F.sort_array(
            F.collect_set(
                F.struct(
                    "provider",
                    "provider_id",
                    *FIELDS,
                    "created_at",
                    "updated_at",
                    "country_inferred",
                )
            )
        ).alias("source_evidence"),
        F.min("created_at").alias("earliest_source_created_at"),
        F.max("updated_at").alias("latest_source_updated_at"),
        F.sort_array(F.collect_set(F.struct(*FIELDS))).alias("candidate_records"),
    )
    # Same normalized company + country at another address is review evidence,
    # not a license to combine branches or choose one business location.
    locations = accepted.groupBy("_company", "_country").agg(
        F.countDistinct("entity_id").alias("company_location_count")
    )
    keys = accepted.select("entity_id", "_company", "_country").distinct()
    result = (
        winner.drop("_rank")
        .join(evidence, "entity_id")
        .join(keys, "entity_id")
        .join(locations, ["_company", "_country"])
    )
    winner_fields = reduce(
        lambda a, b: a & b, [F.col(f"a.{f}") == F.col(f"w.{f}") for f in FIELDS]
    )
    provenance = (
        accepted.alias("a")
        .join(
            winner.alias("w"),
            (F.col("a.entity_id") == F.col("w.entity_id")) & winner_fields,
        )
        .groupBy(F.col("a.entity_id").alias("entity_id"))
        .agg(
            F.sort_array(F.collect_set(F.struct("a.provider", "a.provider_id"))).alias(
                "selected_sources"
            )
        )
    )
    return (
        result.join(provenance, "entity_id")
        .withColumn(
            "selection_support_fraction",
            F.col("variant_support") / F.col("source_record_count"),
        )
        .withColumn("match_rule_score", F.lit(1.0))
        .withColumn("rule_version", F.lit(RULE_VERSION))
        .withColumn(
            "selection_rule", F.lit("distinct_source_support_then_lexical_whole_record")
        )
        .withColumn(
            "resolution_status",
            F.when(F.col("source_record_count") > 1, "exact_agreement").otherwise(
                "provisional_singleton"
            ),
        )
        .withColumn("human_review_status", F.lit("pending"))
        .withColumn("needs_review", F.lit(True))
        .withColumn(
            "review_reasons",
            F.filter(
                F.array(
                    F.lit("unvalidated_exact_match_baseline"),
                    F.when(
                        F.col("source_record_count") == 1,
                        F.lit("uncorroborated_source"),
                    ),
                    F.when(
                        F.col("company_location_count") > 1,
                        F.lit("company_has_multiple_addresses"),
                    ),
                ),
                lambda x: x.isNotNull(),
            ),
        )
        .drop("_company", "_country")
    )
