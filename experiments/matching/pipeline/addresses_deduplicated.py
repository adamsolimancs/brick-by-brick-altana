"""Business address records and source metadata.

One row per resolved company/location. Entities on hold (gold_hold_reason, see
dlq_same_company_conflict) are left out until their address is settled.
"""

from pyspark import pipelines as dp


@dp.materialized_view(
    name="addressresolution.gold.addresses_deduplicated",
    comment="Company and location addresses with provider identity and business metadata.",
    schema="""
        company STRING,
        street_address STRING,
        city STRING,
        state STRING,
        country STRING,
        provider STRING NOT NULL,
        provider_id STRING NOT NULL,
        created_at TIMESTAMP,
        updated_at TIMESTAMP,
        CONSTRAINT addresses_deduplicated_pk PRIMARY KEY (provider, provider_id)
    """,
)
@dp.expect_or_fail(
    "business_fields",
    "company IS NOT NULL AND street_address IS NOT NULL AND city IS NOT NULL AND country IS NOT NULL",
)
def addresses_deduplicated():
    return (
        spark.read.table("addressresolution.silver.addresses_deduplication")
        .where("is_canonical AND gold_hold_reason IS NULL")
        .select(
            "company",
            "street_address",
            "city",
            "state",
            "country",
            "provider",
            "provider_id",
            "created_at",
            "updated_at",
        )
    )
