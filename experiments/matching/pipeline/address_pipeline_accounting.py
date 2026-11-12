"""Where every source row ended up: one row of counts from bronze to Gold.

dlq_routing_audit already proves the Silver routes reconcile. This view carries
that accounting through deduplication, so a dashboard or a run check can show in
one place how many rows reached Gold, how many were folded into another record,
how many are held from Gold because the same company appears at an address
the matcher would not merge (dlq_same_company_conflict), how many wait for a
human (match review), and how many are parked because a field is blank. The
update fails unless every source row is accounted for.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

SILVER = "addressresolution.silver"
EXPECTATIONS = {
    # Filled rows sit in dlq_incomplete and in deduplication, so count them once.
    "silver_routes_reconcile": (
        "source_rows = deduplication_input_rows + incomplete_rows"
        " + translation_review_rows"
    ),
    "deduplication_conserves_rows": (
        "deduplication_input_rows + incomplete_filled_rows = gold_rows"
        " + duplicate_rows + same_company_conflict_rows + rejected_rows"
    ),
    "incomplete_cleanup_reconciles": (
        "incomplete_rows = incomplete_filled_rows + incomplete_parked_rows"
    ),
}


def _blank(column):
    return F.col(column).isNull() | (F.col(column) == "")


@dp.materialized_view(
    name=f"{SILVER}.address_pipeline_accounting",
    comment=(
        "One row of counts for the latest run: source rows, rows parked as "
        "incomplete or in translation review, incomplete rows completed from a "
        "Gold record, rows into deduplication, Gold rows, duplicates folded into "
        "a Gold row, rows held in dlq_same_company_conflict, and rows flagged for "
        "match review. "
        "The update fails unless every source row is accounted for."
    ),
)
@dp.expect_all_or_fail(EXPECTATIONS)
def address_pipeline_accounting():
    audit = spark.read.table(f"{SILVER}.dlq_routing_audit").select(
        "source_rows",
        "provider_a_rows",
        "provider_b_rows",
        "cleaned_rows",
        "translated_rows",
        (
            F.col("incomplete_cleaning_rows") + F.col("incomplete_translation_rows")
        ).alias("incomplete_rows"),
        F.col("review_rows").alias("translation_review_rows"),
    )
    # What is still parked after the Gold-level cleanup, by what is blank.
    incomplete = spark.read.table(f"{SILVER}.dlq_incomplete_v2").agg(
        F.count(F.lit(1)).alias("incomplete_parked_rows"),
        F.count_if(_blank("city")).alias("incomplete_city_blank_rows"),
        F.count_if(_blank("street_address") & ~_blank("city")).alias(
            "incomplete_street_blank_rows"
        ),
        F.count_if(~_blank("city") & ~_blank("street_address")).alias(
            "incomplete_other_rows"
        ),
    )
    held = F.col("gold_hold_reason").isNotNull()
    filled = F.col("filled_field").isNotNull()
    deduplication = spark.read.table(f"{SILVER}.addresses_deduplication").agg(
        F.count_if(~filled).alias("deduplication_input_rows"),
        F.count_if(filled).alias("incomplete_filled_rows"),
        F.count_if(F.col("is_canonical") & ~held).alias("gold_rows"),
        F.count_if(F.col("is_duplicate") & ~held).alias("duplicate_rows"),
        F.count_if(held).alias("same_company_conflict_rows"),
        F.count_if(held & F.col("is_canonical")).alias(
            "same_company_conflict_entities"
        ),
        F.count_if(F.col("deduplication_status") == "input_rejected").alias(
            "rejected_rows"
        ),
        F.count_if(F.col("is_canonical") & ~held & F.col("needs_match_review")).alias(
            "gold_rows_needing_match_review"
        ),
        F.count_if(F.col("is_duplicate") & ~held & F.col("needs_match_review")).alias(
            "duplicate_rows_needing_match_review"
        ),
        F.max("deduplication_run_id").alias("deduplication_run_id"),
    )
    return audit.crossJoin(incomplete).crossJoin(deduplication)
