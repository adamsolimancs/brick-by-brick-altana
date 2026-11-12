"""dlq_incomplete after the Gold-level cleanup: the rows still unresolved.

Silver parks incomplete rows in dlq_incomplete so they do not hold up good
data. Silver runs before resolution, so it cannot know which of them a
resolved record can later complete. This view is that later step: it drops the
rows the resolver completed and folded into a Gold record (filled_field in
addresses_deduplication). What is left is the real open queue.
"""

from pyspark import pipelines as dp
from pyspark.sql import functions as F

SILVER = "addressresolution.silver"


@dp.materialized_view(
    name=f"{SILVER}.dlq_incomplete_v2",
    comment=(
        "Incomplete source rows still unresolved after resolution: dlq_incomplete "
        "minus the rows completed from a Gold record and folded into it (see "
        "filled_field in addresses_deduplication). Use this, not dlq_incomplete, "
        "as the open queue; dlq_incomplete is the Silver routing stage."
    ),
)
def dlq_incomplete_v2():
    resolved = (
        spark.read.table(f"{SILVER}.addresses_deduplication")
        .where("filled_field IS NOT NULL")
        .select(
            F.col("provider").alias("resolved_provider"),
            F.col("provider_id").cast("string").alias("resolved_provider_id"),
        )
    )
    incomplete = spark.read.table(f"{SILVER}.dlq_incomplete")
    return incomplete.join(
        resolved,
        (F.col("provider") == F.col("resolved_provider"))
        & (F.col("provider_id").cast("string") == F.col("resolved_provider_id")),
        "left_anti",
    )
