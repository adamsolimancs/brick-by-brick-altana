"""
Entity Resolution Pipeline — Bronze → Silver cleaning, merge, and DLQ split.

Reads from the raw bronze tables (provider_a_raw, provider_b_raw) ingested by
the separate Bronze File Ingestion Pipeline, not the original feed tables.
Raw tables have string IDs and string dates (plus ingestion metadata columns);
the cleaning step casts them to the canonical silver schema types.

  1. Provider A cleaning  → addressresolution.silver.provider_a_clean
  2. Provider B cleaning  → addressresolution.silver.provider_b_clean
  3. Merge + DLQ routing  → addressresolution.silver.cleaned_1_0 (clean rows)
                            addressresolution.silver.dlq_translate
                            addressresolution.silver.dlq_incomplete
                            addressresolution.silver.dlq_routing_audit

The merge routes each row to the dead letter queue (DLQ) when any of the
following holds for the cleaned values:
  - null or blank company, street_address, city, state, or country
  - non-Latin text in company, street_address, city, or state
  - abnormally short or long company, street_address, city, or state
    (bounds: company 2-100, street_address 4-200, city 2-50, state 2-50;
    city matches the existing city_length_flag thresholds)
  - null, duplicate, or unparseable provider IDs within each provider
    (all rows of a duplicate (provider, provider_id) group go to the DLQ)
  - invalid country codes (anything but an ISO 3166-1 alpha-2 code)
  - dates whose non-null raw value failed tolerant parsing

ID/date parsing is tolerant (try_cast / try_to_date): malformed values become
NULL plus a date_parse_failed flag instead of failing the pipeline, and reach
the DLQ through the rules above.

DLQ rows are split by what can repair them, using the reasons above:
  - dlq_translate: any non_latin reason, whatever else fails. translation.py
    reads it, so imperfect rows never block clean ones.
  - dlq_incomplete: rows only the source or a person can repair (blank,
    abnormal-length, or invalid values). It holds the DLQ rows with no
    non_latin reason, plus translated rows that still fail one of these rules
    (routed_from tells the two apart, and dlq_reasons is recomputed for
    translated rows).
dlq_routing_audit counts every route and fails the update unless the counts
reconcile. cleaned_1_0 feeds downstream matching. providers_merged was replaced
by this split on 2026-10-21, and dlq_1_0 by dlq_translate and dlq_incomplete on
2026-10-24.

Shared cleaning utilities are imported from jl13852's workspace.
"""

import sys

sys.path.insert(0, "/Workspace/Users/jl13852@nyu.edu")

from address_cleaning_utils import (
    normalize_whitespace,
    add_address_helper_columns,
    infer_country_from_state,
    map_country_to_iso,
)
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import lit, col, length, when
from pyspark.sql.window import Window
from pyspark import pipelines as dp


# Canonical silver column order (17 columns, identical for both providers).
SILVER_COLS = [
    "provider_id",
    "provider",
    "company",
    "street_address",
    "city",
    "state",
    "country",
    "created_at",
    "updated_at",
    "full_address",
    "street_address_is_null",
    "city_is_null",
    "country_is_null",
    "country_inferred",
    "has_missing_address",
    "city_length_flag",
    "country_length_flag",
]

# Provider clean tables additionally carry date_parse_failed (set by the
# cleaning step and consumed by DLQ routing; dropped from the cleaned_1_0 and
# dlq_1_0 outputs, which project SILVER_COLS).
CLEAN_COLS = SILVER_COLS + ["date_parse_failed"]


def clean_provider_feed(
    df,
    provider_label,
    id_col,
    whitespace_cols,
    column_renames=None,
    date_col=None,
    map_to_iso=False,
):
    """Clean a provider feed into the canonical silver schema.

    Parameters
    ----------
    provider_label : str
        "A" or "B" — value for the ``provider`` column.
    id_col : str
        Source column name for the provider ID (renamed to ``provider_id``).
    whitespace_cols : list[str]
        Columns to normalize whitespace on.
    column_renames : dict, optional
        Mapping of source column names to canonical names.
    date_col : str, optional
        Source column with a string date to cast to DATE.  The raw string is
        dropped and replaced with the DATE cast; ``created_at`` is set to NULL.
        If None, ``created_at`` must already exist as DATE and ``updated_at``
        is set to NULL.
    map_to_iso : bool, default False
        If True, map country full names to ISO 3166-1 alpha-2 codes.
    """
    # Step 1: Normalize whitespace.
    df = normalize_whitespace(df, whitespace_cols)

    # Step 2: Casing SKIPPED — initcap corrupts CJK+Latin mixed addresses.

    # Step 3: Rename columns, add provider label, rename ID, add helpers.
    if column_renames:
        df = df.withColumnsRenamed(column_renames)
    df = df.withColumn("provider", lit(provider_label))
    df = df.withColumnRenamed(id_col, "provider_id")
    # Tolerant ID parse: unparseable raw strings become NULL (the merge routes
    # them to the DLQ) instead of failing the pipeline.
    df = df.withColumn("provider_id", F.expr("try_cast(provider_id AS INT)"))
    df = add_address_helper_columns(
        df, "street_address", "city", "state", "country", date_col=date_col
    )

    # Step 3b: Infer missing countries from state (mode lookup).
    df = infer_country_from_state(
        df,
        "state",
        "country",
        "provider_id",
        apply_casing=False,
        street_col="street_address",
        city_col="city",
    )

    # Step 3c: Map country full names to ISO alpha-2 (Provider A only).
    if map_to_iso:
        df = map_country_to_iso(
            df, "country", street_col="street_address", city_col="city"
        )

    # Step 3d: Flag abnormally short/long city and country values.
    # Short cities (< 2 chars) are likely data errors; long cities (> 50 chars)
    # catch AI hallucinations, misplaced addresses, or country names in the
    # city field. Country should be ISO alpha-2 (exactly 2 chars); anything
    # shorter or longer indicates an unmapped or corrupt value.
    df = df.withColumn(
        "city_length_flag",
        when(length(col("city")) < 2, lit("too_short"))
        .when(length(col("city")) > 50, lit("too_long"))
        .otherwise(lit(None)),
    )
    df = df.withColumn(
        "country_length_flag",
        when(length(col("country")) < 2, lit("too_short"))
        .when(length(col("country")) > 3, lit("too_long"))
        .otherwise(lit(None)),
    )

    # Step 3e: Flag dates whose non-null raw value failed tolerant parsing.
    # try_to_date (helper) and try_cast return NULL for malformed values, so
    # parsing never fails the pipeline; the flag lets the merge route these
    # rows to the dead letter queue instead.
    if date_col:
        # Provider A: the helper already made a try_to_date copy of the raw string.
        df = df.withColumn(
            "date_parse_failed",
            col(date_col).isNotNull() & col(f"{date_col}_date").isNull(),
        )
    else:
        # Provider B: raw created_at is a string in the raw tables.
        df = df.withColumn(
            "date_parse_failed",
            col("created_at").isNotNull()
            & F.expr("try_cast(created_at AS DATE)").isNull(),
        )

    # Harmonize timestamps so both providers share the same canonical schema.
    if date_col:
        # Provider A: drop raw string, use DATE cast, add NULL created_at.
        df = df.drop(date_col)
        df = df.withColumnRenamed(f"{date_col}_date", date_col)
        df = df.withColumn("created_at", lit(None).cast("date"))
    else:
        # Provider B: cast created_at (string in raw tables) to DATE; add NULL updated_at.
        df = df.withColumn("created_at", F.expr("try_cast(created_at AS DATE)"))
        df = df.withColumn("updated_at", lit(None).cast("date"))

    return df.select(*CLEAN_COLS)


# ---------------------------------------------------------------------------
# 1. Provider A — Bronze → Silver
# ---------------------------------------------------------------------------


@dp.materialized_view(
    name="addressresolution.silver.provider_a_clean",
    comment=(
        "Cleaned Provider A raw: whitespace-normalized, country inferred "
        "from state and mapped to ISO alpha-2, helper columns added. "
        "Casing skipped (14% mixed-script). ID cast string to int from raw table."
    ),
)
def provider_a_clean():
    """Clean the Provider A bronze raw table into the canonical silver schema."""
    return clean_provider_feed(
        spark.read.table("addressresolution.bronze.provider_a_raw"),
        provider_label="A",
        id_col="provider_a_id",
        whitespace_cols=[
            "company",
            "street_address",
            "city",
            "state",
            "country",
            "updated_at",
        ],
        date_col="updated_at",
        map_to_iso=True,
    )


# ---------------------------------------------------------------------------
# 2. Provider B — Bronze → Silver
# ---------------------------------------------------------------------------


@dp.materialized_view(
    name="addressresolution.silver.provider_b_clean",
    comment=(
        "Cleaned Provider B raw: columns renamed to canonical schema, "
        "whitespace-normalized, country inferred from state, helper columns "
        "added. Casing skipped (15% mixed-script). ID cast string to int, "
        "created_at cast string to date from raw table."
    ),
)
def provider_b_clean():
    """Clean the Provider B bronze raw table into the canonical silver schema."""
    return clean_provider_feed(
        spark.read.table("addressresolution.bronze.provider_b_raw"),
        provider_label="B",
        id_col="provider_b_id",
        whitespace_cols=[
            "name",
            "street_address",
            "locality",
            "admin_region",
            "country",
        ],
        column_renames={
            "name": "company",
            "locality": "city",
            "admin_region": "state",
        },
    )


# ---------------------------------------------------------------------------
# 3. Merge providers and route imperfect rows to a dead letter queue
# ---------------------------------------------------------------------------

# Dead letter queue routing rules, evaluated on cleaned values at merge time.
DLQ_REQUIRED_FIELDS = ("company", "street_address", "city", "state", "country")
DLQ_NON_LATIN_FIELDS = ("company", "street_address", "city", "state")
DLQ_MIN_LENGTHS = {"company": 2, "street_address": 4, "city": 2, "state": 2}
DLQ_MAX_LENGTHS = {"company": 100, "street_address": 200, "city": 50, "state": 50}
# Same script classification as translation.py: Latin script plus Common and
# Inherited (digits, punctuation, whitespace, combining accents) are clean;
# any other character marks the value as non-Latin.
NON_LATIN_PATTERN = r"[^\p{IsLatin}\p{IsCommon}\p{IsInherited}]"
# Valid ISO 3166-1 alpha-2 country code: exactly two uppercase Latin letters.
# After country inference and ISO mapping, anything else is invalid.
COUNTRY_CODE_PATTERN = r"^[A-Z]{2}$"


def _dlq_reasons() -> Column:
    """One string per failed cleaning rule, prefixed with the field name.

    Reasons follow translation.py's review-reason style, e.g.
    ``company:too_short`` or ``street_address:non_latin``. Null or blank
    values only trigger ``null_or_blank`` (length checks are null-safe).
    Also flags null/unparseable or duplicate provider IDs, invalid country
    codes, and dates that failed tolerant parsing.
    """
    reasons = []
    for field in DLQ_REQUIRED_FIELDS:
        value = F.col(field)
        reasons.append(
            F.when(
                value.isNull() | (F.trim(value) == ""),
                F.lit(f"{field}:null_or_blank"),
            )
        )
    for field in DLQ_NON_LATIN_FIELDS:
        value = F.col(field)
        reasons.append(
            F.when(
                F.coalesce(value.rlike(NON_LATIN_PATTERN), F.lit(False)),
                F.lit(f"{field}:non_latin"),
            )
        )
    for field, min_len in DLQ_MIN_LENGTHS.items():
        max_len = DLQ_MAX_LENGTHS[field]
        value = F.col(field)
        reasons.append(
            F.when(
                value.isNotNull() & (F.length(value) < min_len),
                F.lit(f"{field}:too_short"),
            )
        )
        reasons.append(
            F.when(
                value.isNotNull() & (F.length(value) > max_len),
                F.lit(f"{field}:too_long"),
            )
        )
    # Tolerant ID parsing leaves null/unparseable provider IDs as NULL.
    reasons.append(
        F.when(
            F.col("provider_id").isNull(),
            F.lit("provider_id:null_or_invalid"),
        )
    )
    # Duplicate (provider, provider_id) pairs break the composite key.
    reasons.append(
        F.when(
            F.col("provider_id_duplicate"),
            F.lit("provider_id:duplicate"),
        )
    )
    # Country must survive cleaning as a valid ISO 3166-1 alpha-2 code;
    # null/blank is already covered by the required-field rule above.
    country = F.col("country")
    reasons.append(
        F.when(
            country.isNotNull()
            & ((F.trim(country) == "") | ~country.rlike(COUNTRY_CODE_PATTERN)),
            F.lit("country:invalid_code"),
        )
    )
    # A non-null raw date that failed tolerant parsing.
    reasons.append(
        F.when(
            F.col("date_parse_failed") & (F.col("provider") == "A"),
            F.lit("updated_at:invalid_date"),
        ).when(
            F.col("date_parse_failed") & (F.col("provider") == "B"),
            F.lit("created_at:invalid_date"),
        )
    )
    return F.filter(F.array(*reasons), lambda reason: reason.isNotNull())


def _merged_with_dlq_reasons() -> DataFrame:
    """Both cleaned provider tables unioned, with dlq_reasons for routing."""
    df_a = spark.read.table("addressresolution.silver.provider_a_clean")
    df_b = spark.read.table("addressresolution.silver.provider_b_clean")
    df = df_a.unionByName(df_b)
    # Null/duplicate ID check within each provider: all rows of a duplicate
    # (provider, provider_id) group are routed to the DLQ so cleaned_1_0
    # keeps a unique composite key. Null IDs share one partition and are
    # already flagged provider_id:null_or_invalid.
    df = df.withColumn(
        "provider_id_duplicate",
        F.count(F.lit(1)).over(Window.partitionBy("provider", "provider_id")) > 1,
    )
    return df.withColumn("dlq_reasons", _dlq_reasons())


@dp.materialized_view(
    name="addressresolution.silver.cleaned_1_0",
    comment=(
        "Merged provider rows passing all cleaning rules: no null/blank "
        "required fields, no non-Latin text in company/street_address/city/"
        "state, and lengths within bounds. Replaces providers_merged for "
        "clean rows. Composite PK: (provider, provider_id)."
    ),
)
def cleaned_1_0():
    """Clean rows only; imperfect rows are routed to dlq_1_0 instead."""
    return (
        _merged_with_dlq_reasons()
        .where(F.size(F.col("dlq_reasons")) == 0)
        .select(*SILVER_COLS)
    )


def _needs_translation() -> Column:
    """True when translation can repair at least one of the row's reasons."""
    return F.exists(F.col("dlq_reasons"), lambda reason: reason.endswith(":non_latin"))


@dp.materialized_view(
    name="addressresolution.silver.dlq_translate",
    comment=(
        "Dead letter queue rows with non-Latin text in company, street_address, "
        "city, or state; they may fail other cleaning rules too. dlq_reasons lists "
        "each failed rule. The translation step reads this table."
    ),
)
def dlq_translate():
    """Route DLQ rows that translation can repair, with their reasons."""
    return (
        _merged_with_dlq_reasons()
        .where((F.size(F.col("dlq_reasons")) > 0) & _needs_translation())
        .select(*SILVER_COLS, "dlq_reasons")
    )


@dp.materialized_view(
    name="addressresolution.silver.dlq_incomplete",
    comment=(
        "Dead letter queue rows that only the source or a person can repair: "
        "blank, abnormal-length, or invalid values. routed_from is 'cleaning' for "
        "rows without non-Latin text, or 'translation' for rows that translated "
        "cleanly but still fail a rule (their dlq_reasons are recomputed after "
        "translation; originals are in providers_translation_results)."
    ),
)
def dlq_incomplete():
    """Rows needing source fixes, from cleaning and from translation."""
    from_cleaning = (
        _merged_with_dlq_reasons()
        .where((F.size(F.col("dlq_reasons")) > 0) & ~_needs_translation())
        .select(*SILVER_COLS, "dlq_reasons", lit("cleaning").alias("routed_from"))
    )
    from_translation = (
        spark.read.table("addressresolution.silver.providers_translation_results")
        .where(col("translation_route").eqNullSafe(lit("incomplete")))
        .select(
            *SILVER_COLS,
            col("remaining_dlq_reasons").alias("dlq_reasons"),
            lit("translation").alias("routed_from"),
        )
    )
    return from_cleaning.unionByName(from_translation)


def _row_count(table: str, alias: str, condition: Column = None) -> DataFrame:
    df = spark.read.table(table)
    if condition is not None:
        df = df.where(condition)
    return df.agg(F.count(lit(1)).alias(alias))


@dp.materialized_view(
    name="addressresolution.silver.dlq_routing_audit",
    comment=(
        "One row of route counts: provider rows into cleaned_1_0, dlq_translate, "
        "and dlq_incomplete, then translated rows into providers_translated, "
        "providers_translation_review, and dlq_incomplete. The update fails "
        "unless every row lands in exactly one route."
    ),
)
@dp.expect_all_or_fail(
    {
        "cleaning_routes_reconcile": (
            "source_rows = cleaned_rows + translate_rows + incomplete_cleaning_rows"
        ),
        "translation_routes_reconcile": (
            "translate_rows = translation_result_rows"
            " AND translation_result_rows = translated_rows + review_rows"
            " + incomplete_translation_rows"
        ),
    }
)
def dlq_routing_audit():
    """Count each route so a lost or duplicated row fails the update."""
    silver = "addressresolution.silver"
    incomplete = f"{silver}.dlq_incomplete"
    counts = [
        _row_count(f"{silver}.provider_a_clean", "provider_a_rows"),
        _row_count(f"{silver}.provider_b_clean", "provider_b_rows"),
        _row_count(f"{silver}.cleaned_1_0", "cleaned_rows"),
        _row_count(f"{silver}.dlq_translate", "translate_rows"),
        _row_count(
            incomplete, "incomplete_cleaning_rows", col("routed_from") == "cleaning"
        ),
        _row_count(
            f"{silver}.providers_translation_results", "translation_result_rows"
        ),
        _row_count(f"{silver}.providers_translated", "translated_rows"),
        _row_count(f"{silver}.providers_translation_review", "review_rows"),
        _row_count(
            incomplete,
            "incomplete_translation_rows",
            col("routed_from") == "translation",
        ),
    ]
    audit = counts[0]
    for count in counts[1:]:
        audit = audit.crossJoin(count)
    return audit.withColumn(
        "source_rows", col("provider_a_rows") + col("provider_b_rows")
    )
