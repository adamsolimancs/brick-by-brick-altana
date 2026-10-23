# Databricks notebook source
# Clean Provider B feed: bronze -> silver

# COMMAND ----------

import sys

# Shared team Git folder; lets this notebook import lib/ even when run from outside the folder.
REPO_ROOT = "/Workspace/Users/jl13852@nyu.edu/brick-by-brick-altana"
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from lib.clean_functions import script_class, trim_strings  # noqa: E402

df_b = spark.table("addressresolution.bronze.provider_b_feed")

# COMMAND ----------

# Trim leading/trailing whitespace on every string column, then turn empty strings into nulls.
df_b = trim_strings(df_b)

# COMMAND ----------

# Diagnostic (no changes): how street_address splits by script, to inform the title-casing decision.
display(df_b.groupBy(script_class(df_b.street_address).alias("script")).count())

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
