# Translation diagnostics and review routing

translation.py reads the cleaned, merged provider table. It defines six
materialized views in addressresolution.silver:

| Dataset | Contents |
| --- | --- |
| providers_translation_prepared | Original fields, deterministic lookups, and versioned request keys. |
| providers_translation_requests | One row per distinct AI request, with source_field_count showing how many fields share it. |
| providers_translation_raw | Stored request metadata and raw ai_result structs; the only AI stage. |
| providers_translation_results | Every input row with translation results and diagnostics; no AI calls. |
| providers_translated | Only rows passing translation checks; use this for downstream processing. |
| providers_translation_review | Current failed rows with the evidence needed for human review. |

The AI calls occur only in providers_translation_raw. Every downstream stage reads
stored tables by name; no validation or output branch invokes inference.

## Request reuse and refresh behavior

Previously, inference already skipped fields resolved by deterministic lookup or
containing only Latin text. However, two records with the same remaining field,
country context, and text each requested their own translation. Now they share a
stored response. Provider rows, including exact duplicates and unrelated columns,
are preserved by left joins back to the prepared records.

Request keys include field name, exact input text, country context, model,
PROMPT_VERSION, and the complete emitted prompt. Country context uses the original
string, with null mapped to `unknown` as in the original prompt. Country aliases
such as `CN` and `China` remain distinct requests; text is not aggressively
normalized because punctuation and number spelling matter. Changing the model,
prompt text, or prompt version changes the keys. A missing response fails
validation and retains the original field, rather than silently dropping a row.

Raw results are a **materialized snapshot, not an append-only inference cache**.
A refresh of the raw stage may recompute inference, and interrupted inference does
not guarantee committed partial progress. Spark/service retries can repeat calls;
this is not an exactly-once billing guarantee. Errors are retained for review and
are not automatically retried selectively. A separate persistent cache/worker is
needed for durable per-request retries across full refreshes; it cannot safely be
implemented by reading and writing the same dataset inside a pipeline definition.

No manual small-batch loop or forced repartitioning is introduced. Databricks
[recommends letting AI Functions manage parallelization and scaling](https://docs.databricks.com/aws/en/large-language-models/ai-functions#use-ai-functions-in-production-workflows).
Smaller batches alone are not a demonstrated throughput fix for this workload.

## Run in the workspace

1. Replace the existing translation source with `translation.py`; do not add a
   second copy defining the same outputs. No additional Python module is required.
2. Validate the pipeline, then refresh the new preparation and request stages
   first if you want to inspect the request reduction before paying for inference.
   Include any upstream tables that need updating in that selection.
3. Refresh the raw stage and downstream results/branches using the prepared inputs.
   Monitor AI Query completed/failed inference metrics and the update stop reason.
4. After raw responses have successfully committed, validation-only edits can use
   **Refresh selection** on providers_translation_results, providers_translated,
   and providers_translation_review. Keep preparation/requests/raw outside that
   selection to reuse their snapshots. A normal whole-pipeline refresh can still
   refresh raw inference. If inputs/model/prompt change, refresh all affected stages
   together; do not mix snapshots from different input/configuration versions.

Databricks documents [selective updates](https://docs.databricks.com/aws/en/ldp/updates).
Run these read-only queries in a separate SQL editor after the respective stages
finish (not inside the dataset functions):

```sql
-- Potential field-level calls before deduplication versus distinct requests.
SELECT field_name,
       SUM(source_field_count) AS eligible_source_fields,
       COUNT(*) AS distinct_ai_requests,
       SUM(source_field_count) - COUNT(*) AS repeated_requests_avoided
FROM addressresolution.silver.providers_translation_requests
GROUP BY field_name;

-- The live workspace exposes success text as ai_result.result.
SELECT COUNT(*) AS stored_requests,
       COUNT_IF(ai_result.errorMessage IS NOT NULL) AS endpoint_errors,
       COUNT_IF(ai_result.result IS NULL OR TRIM(ai_result.result) = '') AS empty_responses
FROM addressresolution.silver.providers_translation_raw;

-- With all stages refreshed against the same inputs, these must match.
SELECT 'prepared' AS stage, COUNT(*) AS rows FROM addressresolution.silver.providers_translation_prepared
UNION ALL
SELECT 'results', COUNT(*) FROM addressresolution.silver.providers_translation_results;
```

The request count is not a prediction of HTTP requests or billing: internal
batching/retries are service/runtime dependent. Check progress in the
[AI Query profile](https://docs.databricks.com/aws/en/large-language-models/ai-functions#monitor-ai-functions-progress),
not only the pipeline's rows-read counter. Live endpoint throughput and the cause
of the previous stop cannot be verified locally.

For each of street_address, city, and state, it retains:

| Column suffix | Meaning |
| --- | --- |
| _original | Cleaned, merged input before deterministic lookup or AI translation. |
| _ai_response | Raw model text, including invalid output; null when AI was skipped or returned no text. |
| _ai_error | Nullable endpoint error message. Null means no endpoint error was reported, including when AI was skipped. |
| _translation_status | unchanged, lookup, ai_translated, ai_error, or invalid_response. |
| _translation_issues | Array of failure reasons; empty for fields without an issue. |

AI responses are trimmed and checked for empty output, remaining non-Latin
script, absence of letters, added, lost, or changed explicit English building/unit
labels or supported native building identifiers, and obvious formatting problems
(multiline text, code fences, surrounding single/double quotes, or prefixes such
as `English translation:` and `Translated address:`). Internal apostrophes in
place names such as `Xi'an` remain valid.

Numeric comparisons ignore ordering but preserve duplicate counts, leading zeros,
and compound identifiers such as 701-3 and 6/4. Label comparisons accept
common abbreviations such as Ste./Suite and Bldg./Building, with Block/Blk
accepted as building labels. For example, `刘街B座, Suite 654` must retain
building B as an explicit English building/block label; removing B, changing it
to C, or moving it into a unit label routes the row to review. Chinese suffix
identifiers and explicit English building identifiers are checked together,
including duplicate counts and compound identifiers. A mixed marker such as
`Building B-02 座` counts once. Russian `корпус`/`строение` markers and their
abbreviations also count as buildings; `к.` is recognized only after a `д.` house
number, as in the supplied feeds. The meanings of `корпус` and `строение` follow
the [UPU address terminology](https://www.upu.int/UPU/media/upu/documents/PostCode/Thorougfare-Types-and-Abbreviations.pdf).
Label comparisons run in both directions, even when a source category is empty,
so introducing `Building G` or `Unit C` also routes a row to review. These checks
are conservative: unsupported native labels or legitimate reformattings can
require review. They cannot prove that the translation is correct or that all
source-language components survived.

Failed responses leave the field's input intact. Row-level
translation_needs_review is true if any field has an issue, and
translation_review_reasons identifies the field and reason, for example
street_address:numbers_changed. Expectations monitor these failures while
retaining the rows in the shared stage. The accepted branch filters failures out,
and the review branch retains them. Null input fields are left alone; this step
does not perform upstream completeness checks.

## Review branch

A Delta quarantine dataset is a practical dead-letter equivalent for a batch
pipeline. The two output branches filter the stored results using the same flag:

- Accepted rows: translation_needs_review is explicitly false.
- Review rows: translation_needs_review is true (or unexpectedly null).

Every result goes to exactly one output branch. A row with any failed field is
held for review as a whole, even if its other fields translated successfully.
The all-results stage preserves evidence for auditing and retries.

providers_translation_review is a current snapshot, not an append-only queue
history. A later successful translation removes the row from that snapshot.

Keep human review decisions and corrections in a separate persistent Delta table
keyed by stable provider/source record identifiers. Writing corrections directly
to a pipeline-managed materialized view would allow a refresh to overwrite them.
Include reviewer, review status, correction, and the relevant input/model/prompt
version when implementing that workflow. Distinguish retryable endpoint failures
from translations requiring human judgment.

Databricks documents this as the
[quarantine pattern](https://docs.databricks.com/aws/en/ldp/expectation-patterns#quarantine-invalid-data).
failOnError => false captures row-level endpoint errors; query/configuration
errors can still fail the entire update.

## Local validation

### AI response contract

`AI_RESPONSE_FIELD` in `translation.py` retains the live workspace setting `result`, with
`errorMessage` carrying endpoint errors, as described in the
[Databricks ai_query reference](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query).
The helper uses explicit field expressions so Spark can resolve them during
normal pipeline planning. It does not inspect `df.schema` inside a dataset
definition.

Before the first workspace run, inspect the return schema in a separate
Databricks notebook outside the pipeline definition:

```python
probe = spark.sql("""
    SELECT ai_query(
        'system.ai.qwen35-122b-a10b', 'Return only OK.', failOnError => false
    ) AS ai_result
""")
print(probe.schema["ai_result"].dataType.simpleString())
```

This requests schema metadata without collecting rows or performing inference.
If a different runtime names the success field `response`, set `AI_RESPONSE_FIELD = "response"` before
running the pipeline. A mismatched contract fails normal Spark analysis rather
than silently discarding responses. Tests cover both configured field names and
reject schema inspection while constructing the translation plan.

### Running tests

With PySpark 3.5+ and a compatible Java runtime:

    python -m unittest discover -s tests -v
    ruff check translation.py tests/test_translation.py

Tests execute real Spark transformations with controlled inference responses;
they do not call a Databricks model service or validate live workspace access.
