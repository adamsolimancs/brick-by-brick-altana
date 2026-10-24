"""
Shared address-cleaning utilities for Provider A and Provider B datasets.

All functions operate on Spark DataFrames and accept column names as arguments
so they work regardless of schema differences between providers.
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# 1. Whitespace normalization
# ---------------------------------------------------------------------------

def normalize_whitespace(df, columns):
    """Trim, collapse internal whitespace, and nullify empty strings.

    Replaces every run of whitespace (ASCII, NBSP, and other Unicode spaces
    incl. ideographic space \u3000) with a single space,
    strips leading/trailing whitespace, and converts empty strings to NULL.

    Parameters
    ----------
    df : DataFrame
    columns : list[str]
        Column names to normalize.
    """
    _WS = r"[\s\u00A0\u2000-\u200A\u202F\u205F\u3000\uFEFF]+"
    for c in columns:
        df = df.withColumn(
            c,
            F.nullif(F.trim(F.regexp_replace(F.col(c), _WS, " ")), F.lit("")),
        )
    return df


# ---------------------------------------------------------------------------
# 2. Casing standardization
# ---------------------------------------------------------------------------

def standardize_casing(df, columns):
    """Apply Title Case (initcap) to the given columns.

    .. note::
       ``initcap`` capitalizes the first character of every space-separated
       token and lowercases the rest.  For CJK+Latin mixed tokens such as
       "刘街B座" it will incorrectly lowercase the B to "刘街b座" because the
       token starts with a CJK character that has no case.  Both Provider A
       (~14 % mixed) and Provider B (~15 % mixed) are affected.  Call this
       only on columns where the trade-off is acceptable.
    """
    for c in columns:
        df = df.withColumn(c, F.initcap(F.col(c)))
    return df


# ---------------------------------------------------------------------------
# 3. Helper columns (full_address, null flags, completeness flag, date cast)
# ---------------------------------------------------------------------------

def add_address_helper_columns(
    df,
    street_col="street_address",
    city_col="city",
    region_col="region",
    country_col="country",
    date_col=None,
    date_col_out=None,
):
    """Add full_address, per-field null flags, has_missing_address, and an
    optional DATE-cast column.

    Parameters
    ----------
    df : DataFrame
    street_col, city_col, region_col, country_col : str
        Column names used to build ``full_address`` and null flags.
    date_col : str, optional
        If given, a ``try_to_date`` copy is created.
    date_col_out : str, optional
        Name for the date-cast column (defaults to ``date_col + "_date"``).
    """
    df = df.withColumn(
        "full_address",
        F.concat_ws(", ", F.col(street_col), F.col(city_col), F.col(region_col), F.col(country_col)),
    )
    df = df.withColumn(f"{street_col}_is_null", F.col(street_col).isNull())
    df = df.withColumn(f"{city_col}_is_null", F.col(city_col).isNull())
    df = df.withColumn(f"{country_col}_is_null", F.col(country_col).isNull())
    df = df.withColumn(
        "has_missing_address",
        F.col(street_col).isNull() | F.col(city_col).isNull() | F.col(country_col).isNull(),
    )
    if date_col is not None:
        out = date_col_out or f"{date_col}_date"
        df = df.withColumn(out, F.try_to_date(F.col(date_col), "yyyy-MM-dd"))
    return df


# ---------------------------------------------------------------------------
# 4. Country inference from state/region
# ---------------------------------------------------------------------------

def infer_country_from_state(df, state_col, country_col, id_col, apply_casing=True, street_col=None, city_col=None):
    """Fill NULL ``country_col`` values using the most frequent country per
    state/region (mode lookup built from non-null rows).

    Adds a boolean ``country_inferred`` flag.

    Parameters
    ----------
    apply_casing : bool, default True
        If True, Title-case ``country_col`` after inference (for full-name
        countries like "United States").  Set False for ISO codes ("US").
    street_col, city_col : str, optional
        If provided, recalculate has_missing_address after inference.
    """
    # Secondary sort on country gives a deterministic tiebreak when counts tie.
    w = Window.partitionBy(state_col).orderBy(F.desc("cnt"), country_col)
    state_country_map = (
        df.filter(F.col(country_col).isNotNull())
        .groupBy(state_col, country_col)
        .agg(F.sum(F.lit(1)).alias("cnt"))
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .select(state_col, F.col(country_col).alias("inferred_country"))
    )

    df = (
        df.join(state_country_map, on=state_col, how="left")
        .withColumn("country_inferred", F.col(country_col).isNull() & F.col("inferred_country").isNotNull())
        .withColumn(
            country_col,
            F.when(F.col(country_col).isNull(), F.col("inferred_country")).otherwise(F.col(country_col)),
        )
    )
    if apply_casing:
        df = df.withColumn(country_col, F.initcap(F.col(country_col)))
    df = df.drop("inferred_country")

    # Recalculate null flags to reflect post-inference state
    if f"{country_col}_is_null" in df.columns:
        df = df.withColumn(f"{country_col}_is_null", F.col(country_col).isNull())
    if street_col and city_col and "has_missing_address" in df.columns:
        df = df.withColumn(
            "has_missing_address",
            F.col(street_col).isNull() | F.col(city_col).isNull() | F.col(country_col).isNull(),
        )
    return df


# ---------------------------------------------------------------------------
# 5. Before / after comparison
# ---------------------------------------------------------------------------

def compare_before_after(df_clean, df_raw, id_col, compare_cols):
    """Return rows where any ``compare_cols`` value differs between clean and
    raw. Null-safe: a value that is NULL on exactly one side counts as changed
    (e.g. NULL -> inferred country, empty-string -> NULL).

    Parameters
    ----------
    df_clean, df_raw : DataFrame
        Must share ``id_col``.  Raw should contain the original (pre-clean)
        values.
    compare_cols : list[str]
        Columns present in *both* DataFrames to compare.
    """
    diff_filter = None
    for c in compare_cols:
        cond = ~df_raw[c].eqNullSafe(df_clean[c])  # null-safe "changed" check
        diff_filter = cond if diff_filter is None else diff_filter | cond

    changed = df_clean.join(df_raw, on=id_col, how="inner").select(
        F.col(id_col),
        *[df_raw[c].alias(f"{c}_raw") for c in compare_cols],
        *[df_clean[c].alias(f"{c}_clean") for c in compare_cols],
    )
    if diff_filter is not None:
        changed = changed.filter(diff_filter)
    return changed


# ---------------------------------------------------------------------------
# 6. Null / missing summary
# ---------------------------------------------------------------------------

def missing_summary(df, flag_cols):
    """Aggregate boolean flag columns into a single-row summary.

    Parameters
    ----------
    flag_cols : list[str]
        Boolean columns to sum (cast to int).
    """
    return df.agg(
        *[F.sum(F.col(c).cast("int")).alias(c) for c in flag_cols],
    )


# ---------------------------------------------------------------------------
# 7. Country name to ISO code mapping
# ---------------------------------------------------------------------------

_COUNTRY_ISO = {
    "United States": "US",
    "United Kingdom": "GB",
    "Germany": "DE",
    "France": "FR",
    "Canada": "CA",
    "Japan": "JP",
    "China": "CN",
    "Brazil": "BR",
    "Australia": "AU",
    "Spain": "ES",
    "Italy": "IT",
    "Mexico": "MX",
    "India": "IN",
    "Russia": "RU",
    "South Korea": "KR",
    "Turkey": "TR",
    "Netherlands": "NL",
    "Poland": "PL",
    "Thailand": "TH",
    "Saudi Arabia": "SA",
}

def map_country_to_iso(df, country_col, street_col=None, city_col=None):
    """Map full country names to ISO 3166-1 alpha-2 codes.

    Unmapped values become NULL (indicates bad data).  Call after country
    inference so inferred values are also mapped.

    Parameters
    ----------
    street_col, city_col : str, optional
        If provided, recalculate null flags after mapping.
    """
    mapping = F.create_map(*[F.lit(x) for k, v in _COUNTRY_ISO.items() for x in (k, v)])
    df = df.withColumn(country_col, mapping[F.col(country_col)])
    # Recalculate null flags to reflect post-mapping state
    if f"{country_col}_is_null" in df.columns:
        df = df.withColumn(f"{country_col}_is_null", F.col(country_col).isNull())
    if street_col and city_col and "has_missing_address" in df.columns:
        df = df.withColumn(
            "has_missing_address",
            F.col(street_col).isNull() | F.col(city_col).isNull() | F.col(country_col).isNull(),
        )
    return df