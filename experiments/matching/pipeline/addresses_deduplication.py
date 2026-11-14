"""All source addresses, with duplicate decisions and the evidence for each merge."""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

HOLD_REASON = "same_company_address_conflict"
# Review decision the resolver writes on a same-company pair it settled as two
# clearly different addresses (lib/resolution.py). Such a pair holds nothing.
DIFFERENT_ADDRESS = "same_company_different_address"
# Address parts compared between two same-company records, as (label, key).
ADDRESS_PARTS = (
    ("street_address", "street_key"),
    ("city", "city_key"),
    ("state", "region_key"),
)


def same_company_conflicts(records):
    """Review pairs whose two records carry the same company name key.

    The matcher refused to merge these pairs because the addresses disagree or
    could not be verified. With the same company on both sides, that is either
    one company at two addresses or two companies sharing a name, and nothing
    in the data says which. Pairs the resolver already settled are not here:
    the same place was merged into one entity, and two clearly different
    addresses carry the DIFFERENT_ADDRESS decision. One row per (record,
    conflicting record).
    """
    other = records.select(
        F.col("record_id").alias("other_id"),
        F.col("entity_id").alias("other_entity_id"),
        F.col("name_key").alias("other_name_key"),
        F.col("provider").alias("other_provider"),
        F.col("provider_id").alias("other_provider_id"),
        F.col("company").alias("other_company"),
        F.col("street_address").alias("other_street_address"),
        F.col("city").alias("other_city"),
        F.col("state").alias("other_state"),
        *[F.col(key).alias(f"other_{key}") for _, key in ADDRESS_PARTS],
    )
    differing = F.filter(
        F.array(
            *[
                F.when(~F.col(key).eqNullSafe(F.col(f"other_{key}")), F.lit(label))
                for label, key in ADDRESS_PARTS
            ]
        ),
        lambda label: label.isNotNull(),
    )
    return (
        records.select(
            "record_id",
            "entity_id",
            "name_key",
            *[key for _, key in ADDRESS_PARTS],
            F.explode("match_review_evidence").alias("review"),
        )
        .join(other, F.col("review.other_record_id") == F.col("other_id"), "inner")
        .where(
            (F.col("name_key") != "")
            & (F.col("name_key") == F.col("other_name_key"))
            & (F.col("entity_id") != F.col("other_entity_id"))
            & (F.col("review.decision") != DIFFERENT_ADDRESS)
        )
        .select(
            "record_id",
            "entity_id",
            F.struct(
                F.col("other_id").alias("other_record_id"),
                "other_provider",
                "other_provider_id",
                "other_entity_id",
                "other_company",
                "other_street_address",
                "other_city",
                "other_state",
                F.col("review.decision").alias("decision"),
                F.col("review.fs_match_probability").alias("fs_match_probability"),
                F.col("review.reason_codes").alias("reason_codes"),
                differing.alias("differing_fields"),
            ).alias("conflict"),
        )
    )


@dp.materialized_view(
    name="addressresolution.silver.addresses_deduplication",
    comment=(
        "One row per source observation with canonical identity, duplicate "
        "decision, scoring stage and direct match evidence. gold_hold_reason is "
        "set on every record of an entity that is held back from Gold; "
        "filled_field marks incomplete source rows completed from their Gold record."
    ),
)
@dp.expect_or_fail("record_identity", "record_id IS NOT NULL")
@dp.expect_or_fail(
    "duplicate_has_evidence",
    "NOT is_duplicate OR (canonical_record_id IS NOT NULL AND size(deduplication_evidence) > 0)",
)
def addresses_deduplication():
    completed = (
        spark.read.table("addressresolution.matching.resolution_runs")
        .where(F.col("status") == "COMPLETED")
        .orderBy(F.desc("completed_at"), F.desc("deduplication_run_id"))
        .limit(1)
    )
    records = spark.read.table("addressresolution.matching.resolved_records").join(
        completed.select("deduplication_run_id"), "deduplication_run_id", "inner"
    )
    provenance = (
        records.where("entity_id IS NOT NULL")
        .groupBy("entity_id")
        .agg(
            F.countDistinct("provider").alias("provider_count"),
            F.sort_array(F.collect_set(F.struct("provider", "provider_id"))).alias(
                "source_records"
            ),
        )
    )
    conflicts = same_company_conflicts(records)
    per_record = conflicts.groupBy("record_id").agg(
        F.sort_array(F.collect_list("conflict")).alias("same_company_conflicts")
    )
    # A conflict on any member puts the whole entity on hold: its Gold row
    # would otherwise assert an address the pipeline could not settle.
    held = (
        conflicts.select("entity_id")
        .distinct()
        .withColumn("gold_hold_reason", F.lit(HOLD_REASON))
    )
    return (
        records.join(provenance, "entity_id", "left")
        .join(per_record, "record_id", "left")
        .join(held, "entity_id", "left")
        .select(
            *records.columns,
            "provider_count",
            "source_records",
            F.coalesce(
                "same_company_conflicts",
                F.array().cast(per_record.schema["same_company_conflicts"].dataType),
            ).alias("same_company_conflicts"),
            "gold_hold_reason",
        )
    )
