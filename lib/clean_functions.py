"""Shared cleaning functions for provider feeds (bronze -> silver).

Everything here must be provider-agnostic: take a DataFrame or Column and
return one, with no hard-coded table or column names, so the same function
works for any of the N provider feeds.

This is a plain Python module, not a Databricks notebook. Import it from a
notebook in the Git folder with:

    from lib.clean_functions import trim_strings
"""

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Java regex: \s is ASCII whitespace only, so add \p{Z} to also catch NBSP,
# the ideographic space (U+3000), and other Unicode separators.
_WS = r"[\s\p{Z}]"


def string_columns(df: DataFrame) -> list[str]:
    """Return the names of all string-typed columns in df."""
    return [f.name for f in df.schema.fields if isinstance(f.dataType, StringType)]


def trim_strings(df: DataFrame, cols: list[str] | None = None) -> DataFrame:
    """Strip leading/trailing whitespace and turn empty strings into nulls.

    Applies to `cols`, or to every string column when `cols` is None. Uses a
    regex because F.trim only strips ASCII spaces, missing tabs, newlines,
    and NBSPs.
    """
    targets = set(string_columns(df) if cols is None else cols)
    return df.select(
        *[
            F.nullif(
                F.regexp_replace(F.col(c), rf"^{_WS}+|{_WS}+$", ""), F.lit("")
            ).alias(c)
            if c in targets
            else F.col(c)
            for c in df.columns
        ]
    )


def collapse_whitespace(col: Column) -> Column:
    """Replace each internal run of whitespace (incl. newlines) with one space."""
    return F.regexp_replace(col, rf"{_WS}+", " ")


def script_class(col: Column) -> Column:
    """Classify a string by writing system.

    Returns one of "null", "latin_only", "mixed", "non_latin_only", or
    "no_letters". Latin includes accented Latin (e.g. "Raik-Weiß-Ring");
    digits and punctuation (Common/Inherited scripts) count as neither.
    """
    has_latin = col.rlike(r"\p{IsLatin}")
    has_other = col.rlike(r"[^\p{IsLatin}\p{IsCommon}\p{IsInherited}]")
    return (
        F.when(col.isNull(), "null")
        .when(has_latin & ~has_other, "latin_only")
        .when(has_latin & has_other, "mixed")
        .when(has_other, "non_latin_only")
        .otherwise("no_letters")
    )
