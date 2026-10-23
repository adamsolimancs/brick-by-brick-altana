"""Lakeflow source: clean fast path -> provisional gold, independent of AI.

Deploy alongside lib/gold.py with the repository root on Python's import path.
Use a separate pipeline from translation to isolate update failures and latency.
"""

from pyspark import pipelines as dp

from lib.gold import build_gold, prepare_gold_input


@dp.materialized_view(name="addressresolution.silver.gold_input_1_0")
def gold_input_1_0():
    return prepare_gold_input(spark.read.table("addressresolution.silver.cleaned_1_0"))


@dp.materialized_view(name="addressresolution.gold.input_review_1_0")
def gold_input_review_1_0():
    return spark.read.table("addressresolution.silver.gold_input_1_0").where(
        "size(gold_input_reasons) > 0"
    )


@dp.materialized_view(
    name="addressresolution.gold.entities_1_0",
    comment="Provisional exact company/address clusters with source evidence; pending review.",
)
def entities_1_0():
    return build_gold(spark.read.table("addressresolution.silver.gold_input_1_0"))
