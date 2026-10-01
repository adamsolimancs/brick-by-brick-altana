# Databricks notebook source
# Clean Provider B feed: bronze -> silver

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StringType

df_b = spark.table("addressresolution.bronze.provider_b_feed")

# COMMAND ----------

# Trim leading/trailing whitespace on every string column, then turn empty strings into nulls.
# F.trim only strips ASCII spaces, so use a regex to also catch tabs, newlines, and NBSPs.
string_cols = [f.name for f in df_b.schema.fields if isinstance(f.dataType, StringType)]

df_b = df_b.select(
    *[
        F.nullif(F.regexp_replace(F.col(c), r"^[\s ]+|[\s ]+$", ""), F.lit("")).alias(c)
        if c in string_cols
        else F.col(c)
        for c in df_b.columns
    ]
)

# COMMAND ----------

# Diagnostic (no changes): how street_address splits by script, to inform the title-casing decision.
# Latin-only includes accented Latin (e.g. "Raik-Weiß-Ring"), not just ASCII.
has_latin = F.col("street_address").rlike(r"\p{IsLatin}")
has_other = F.col("street_address").rlike(r"[^\p{IsLatin}\p{IsCommon}\p{IsInherited}]")

display(
    df_b.withColumn(
        "script",
        F.when(F.col("street_address").isNull(), "null")
        .when(has_latin & ~has_other, "latin_only")
        .when(has_latin & has_other, "mixed")
        .otherwise("non_latin_only"),
    )
    .groupBy("script")
    .count()
)

# COMMAND ----------

# Rename columns to the silver schema.
df_b = df_b.withColumnsRenamed(
    {
        "name": "company",
        "locality": "city",
        "admin_region": "region",
    }
)

df_b.printSchema()

# COMMAND ----------

df_b.write.mode("overwrite").saveAsTable("addressresolution.silver.provider_b_feed")
