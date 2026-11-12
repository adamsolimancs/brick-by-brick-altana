"""Records held back from Gold: same company name, address not settled.

Each row is a source record of an entity that addresses_deduplication put on
hold because the matcher found another record with the same company name but
would not merge the two (street, city, region or unit disagree, or a parse
could not be verified). That is either one company at two addresses, where one
true address still has to be chosen, or two companies that share a name. Until
a rule or a reviewer decides, neither side is published to Gold.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

DLQ_REASONS = """
    CASE
        WHEN size(same_company_conflicts) = 0
            THEN array('entity:member_has_same_company_conflict')
        ELSE array_sort(array_distinct(flatten(transform(
            same_company_conflicts,
            conflict -> CASE
                WHEN size(conflict.differing_fields) = 0
                    THEN array('address:same_company_match_unverified')
                ELSE transform(
                    conflict.differing_fields,
                    field -> concat(field, ':differs_from_same_company_record')
                )
            END
        ))))
    END
"""


@dp.materialized_view(
    name="addressresolution.silver.dlq_same_company_conflict",
    comment=(
        "Dead letter queue of source records held back from Gold because another "
        "record names the same company at an address the matcher would not merge "
        "with theirs: one company at two addresses, or two companies sharing a "
        "name. same_company_conflicts lists the other records; rows with an empty "
        "list are members of a held entity. Nothing here is in Gold."
    ),
)
@dp.expect_or_fail("held_from_gold", "gold_hold_reason IS NOT NULL")
def dlq_same_company_conflict():
    return (
        spark.read.table("addressresolution.silver.addresses_deduplication")
        .where("gold_hold_reason IS NOT NULL")
        .select(
            "provider",
            "provider_id",
            "company",
            "street_address",
            "city",
            "state",
            "country",
            "created_at",
            "updated_at",
            F.expr(DLQ_REASONS).alias("dlq_reasons"),
            "same_company_conflicts",
            "record_id",
            "entity_id",
            "canonical_record_id",
            "is_canonical",
            "source_records",
            "gold_hold_reason",
            "deduplication_run_id",
        )
    )
