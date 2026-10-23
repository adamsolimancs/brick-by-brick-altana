# Clean fast path and provisional gold

The supplied `/Users/adoma/Downloads/provider_cleaning.py` was reviewed on
2026-10-21. It defines **`addressresolution.silver.cleaned_1_0`**, not `clean_1_0`.
Its complementary filters split the union of the cleaned providers into clean
and quarantined snapshots. This is a sound way to remove AI from gold's data
dependencies, but does not isolate failures when translation shares the same
pipeline update. Use a separate gold pipeline reading the committed clean table.

## Cleaning review findings

1. **High: malformed IDs/dates can still stop the clean path** (lines 106,
   107–109, 149). The explicit integer/date casts run before routing. Under ANSI
   mode malformed/overflowing input can throw; with permissive casts a null ID
   is currently allowed into `cleaned_1_0`. Retain original IDs as strings and
   raw date strings, parse dates with `try_cast`, and add explicit parse-failure
   reasons. Inspect `add_address_helper_columns` too: its external source was
   not supplied, so its date parsing cannot be verified here. Do not merely
   replace a cast and lose the original value needed for repair.
2. **High: country flags are not routing rules** (lines 134–138, 223–265).
   Nonblank values such as `United States` or `??` can pass to clean even when
   flagged. The comment says exactly two characters, the flag allows three,
   and neither is checked by routing. Define one country-code contract and
   validate against an authoritative code mapping, not length alone. The gold
   guard below requires two Latin letters; this is syntax validation, not ISO
   membership validation.
3. **High: the advertised composite primary key is unenforced** (line 283).
   Neither null source IDs nor duplicate/conflicting `(provider, provider_id)`
   rows are checked. Gold must not arbitrarily collapse conflicting identities.
   The new gold input stage quarantines every row of a conflicting identity and
   counts distinct sources separately from physical duplicate rows.
4. **High: the local translation source still reads `providers_merged`**
   (`translation.py`, preparation function). The new cleaner's comments do not
   redirect that dependency. Translation also only repairs street/city/state,
   not company, missing fields or abnormal lengths. Reading the entire DLQ and
   trusting `providers_translated` would falsely promote structural failures.
   Route eligible translation work separately and reapply **all** cleaning rules
   after repair before any future merge into the gold input. Gold v1 deliberately
   has no dependency on translated or DLQ rows. Existing translation code was
   left intact because its local edits predate this work.
5. **Medium: country inference needs an uncertainty trail** (lines 111–115).
   State-only inference can be ambiguous across countries. The external lookup
   and its tie handling could not be inspected. Keep `country_inferred` and
   record lookup confidence/provenance before treating inferred country as fact.
6. **Low: blank values can receive duplicate reasons** (lines 231–263).
   Empty strings trigger both `null_or_blank` and `too_short`, despite the
   docstring. Exclude trimmed blanks from length checks. The DLQ is a current
   materialized snapshot, not an append-only retry queue; manual corrections
   need a separate durable table.

Databricks documents [tolerant date parsing with try_cast](https://docs.databricks.com/aws/en/sql/language-manual/functions/to_date).
The unavailable shared cleaning utilities prevent a complete cleaner execution
test; the supplied Downloads file has not been modified.

## Gold v1 contract

`gold.py` and `lib/gold.py` define these pipeline outputs:

| Table | Purpose |
| --- | --- |
| `addressresolution.silver.gold_input_1_0` | Every clean-table input row, normalized keys and defensive rejection reasons. |
| `addressresolution.gold.input_review_1_0` | Missing/invalid content or conflicting source identities. |
| `addressresolution.gold.entities_1_0` | One provisional exact company/address cluster with source evidence. |

This is an intentionally conservative **entity/address** baseline. It is not
global company resolution: branches, spelling variants, abbreviations, differing
state conventions and translated records remain separate. Case and Unicode
whitespace are normalized only for comparison. Punctuation, accents, house/unit
numbers and legal suffixes are retained. Different companies sharing an address
do not merge. No all-pairs join, fuzzy transitive clustering or AI inference runs.

The selected whole source record has the most distinct source-ID support, with
a deterministic lexical tie break. No components from different addresses are
spliced together; `selected_sources` supplies provenance for every selected
component. Original alternatives remain in `candidate_records`. `source_records`
contains composite identifiers, `input_row_count` includes physical duplicates,
and `source_record_count` counts distinct identities. `source_evidence` also keeps
each source's original components, dates and `country_inferred` flag. Source dates are summarized
but never compared as if creation and update timestamps were equivalent.

`match_rule_score = 1.0` means the exact rule matched, **not** an estimated
probability. `selection_support_fraction` is observed source support, not
accuracy. All clusters retain `human_review_status = pending` and `needs_review`
until the rule has been evaluated on labeled matches. Same company/country at
multiple addresses produces an extra review reason without combining locations.
IDs are deterministic hashes of normalized content and change when content
changes; they are not permanent business identifiers across corrections.

## Build in Databricks

1. First materialize `cleaned_1_0` from the new cleaner, including its provider
   dependencies. Verify its actual schema and counts; the older run visible in
   the browser referenced `providers_merged` and does not establish current state.
2. Copy `gold.py` and `lib/gold.py` together into a workspace folder with this
   layout, and set that root as the pipeline's import root. Add **only `gold.py`**
   as the source of a separate serverless Lakeflow pipeline. Avoid a folder glob
   that would also discover `translation.py` and schedule AI. The pipeline
   identity needs read access to silver and create/write access to gold.
3. Validate, then run this gold pipeline. It reads the already committed silver
   snapshot and performs no model calls. Schedule it after the clean update,
   independently of the translation update.
4. Run the checks in `gold_checks.sql`. Acceptance requires no null/duplicate
   entity IDs and conservation of every input row across gold and review.

Local source creation and tests do not deploy anything. Live materialization is
blocked here: Databricks displays “Automated browser control detected” and asks
for manual use or Genie One MCP; no Databricks connector/CLI is configured.

## Local verification

With PySpark 3.5+ and Java 17:

```sh
python -m unittest discover -s tests -p 'test_gold.py' -v
ruff check gold.py lib/gold.py tests/test_gold.py
```

Tests execute Spark transformations for duplicate support, cross-provider IDs,
source-ID conflicts, missing values, Unicode whitespace, distinct companies at
one address, different units/locations, reproducible selection, empty input and
row conservation. They do not verify the external cleaner or live permissions.

Verified locally on 2026-10-21 using PySpark 3.5.3 and Java 17: both Spark tests
passed (8.5 seconds), scoped Ruff checks passed, and Python compilation passed.
The Spark test runner needed local socket access and an explicitly aligned
worker Python; no Databricks endpoint or AI service was called.
