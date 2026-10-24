# Translation diagnostics and review routing

translation.py reads `dlq_translate`, the dead letter queue rows with non-Latin
text (see [DLQ routes](#dlq-routes)). It defines eight materialized views in
addressresolution.silver:

| Dataset | Contents |
| --- | --- |
| providers_translation_prepared | Original fields, deterministic lookups, and versioned request keys. |
| providers_translation_requests | One row per distinct AI request, with source_field_count showing how many fields share it. |
| providers_translation_raw | Stored request metadata and raw ai_result structs from the primary model. |
| providers_translation_escalation_requests | Requests whose primary response failed the checks, with primary_issues. |
| providers_translation_escalation_raw | Stored escalation responses from the larger model. |
| providers_translation_results | Every input row with translation results, translation_route, and diagnostics; no AI calls. |
| providers_translated | Only rows passing translation checks and every cleaning rule; use this for downstream processing. |
| providers_translation_review | Current rows where a translation failed, with the evidence needed for human review. |

The AI calls occur only in the two raw stages. Every downstream stage reads
stored tables by name; no validation or output branch invokes inference.

## DLQ routes

provider_cleaning.py splits the rows failing a cleaning rule by what can repair
them, using the dlq_reasons it already computes:

| Dataset | Rows |
| --- | --- |
| cleaned_1_0 | Rows passing every cleaning rule. |
| dlq_translate | Rows with any `:non_latin` reason, whatever else fails. Translation reads it. |
| dlq_incomplete | Rows only the source or a person can repair: blank, abnormal-length, or invalid values. `routed_from = 'cleaning'` rows have no non-Latin text; `routed_from = 'translation'` rows translated cleanly but still fail a rule, and carry their recomputed reasons and translated values. |
| dlq_routing_audit | One row of counts per route. Its expectations fail the update unless every row lands in exactly one route. |

A row with non-Latin text and a blank city goes to translation first, then to
dlq_incomplete, so it neither skips translation nor loses its missing-field
reason. dlq_1_0 was replaced by this split on 2026-10-24; together, dlq_translate
and the cleaning rows of dlq_incomplete hold the same rows.

## Request reuse and refresh behavior

Previously, inference already skipped fields resolved by deterministic lookup or
containing only Latin text. However, two records with the same remaining field,
country context, and text each requested their own translation. Now they share a
stored response. Provider rows, including exact duplicates and unrelated columns,
are preserved by left joins back to the prepared records.

Request keys include field name, exact input text, country context, model,
PROMPT_VERSION, and the complete emitted prompt. Country context uses the trimmed
country string when it looks like a country name or code (letters, spaces, and
`.,'()-`, at most 60 characters); null or anything else, including
instruction-like text, becomes `unknown`. Every prompt tells the model to treat the
field text as data, not as instructions. Country aliases
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
| _ai_model | Model whose response is shown; null when no response was stored. |

AI responses are trimmed and checked for empty output, remaining non-Latin
script, absence of letters, added, lost, or changed explicit English building/unit
labels or supported native labels and building identifiers, and obvious formatting problems
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
The Hindi and Thai labels in `NATIVE_ADDRESS_LABELS` (`सुइट`/`สวีท` Suite,
`अपार्टमेंट`/`อพาร์ตเมนต์` Apartment, `ชั้น` Floor, `อาคาร` Building, and others) count
as their English categories, and a Hindi letter name after a label counts as that
letter (`बिल्डिंग एच` is Building H). Without them, a correct `Suite 612` looked
like an added label.
Label comparisons run in both directions, even when a source category is empty,
so introducing `Building G` or `Unit C` also routes a row to review. These checks
are conservative: unsupported native labels or legitimate reformattings can
require review. They cannot prove that the translation is correct or that all
source-language components survived.

Failed responses leave the field's input intact. Row-level
translation_needs_review is true if any field has an issue, and
translation_review_reasons identifies the field and reason, for example
street_address:numbers_changed. Expectations monitor these failures while
retaining the rows in the shared stage.

Rows arrive from the DLQ because they failed at least one cleaning rule, and
translation only repairs non-Latin text. So after translation, the null/blank
and length rules of provider_cleaning.py are reapplied to company,
street_address, city, and state, and DLQ reasons for other fields (provider_id,
country, dates) carry over unchanged. These remaining_dlq_reasons join
translation_review_reasons in the same `field:reason` format, for example
`company:too_short` or `provider_id:duplicate`, so a row is accepted only once
every reason it was quarantined for is resolved. A remaining reason in a field
that translation changed, such as an answer too long for the bounds, is a
translation failure: it joins translation_review_reasons. Any other remaining
reason, such as a blank city, is not, and routes the row to dlq_incomplete. The
length bounds are copied in
CLEANING_LENGTH_BOUNDS; keep them in sync with provider_cleaning.py.
providers_translated also declares the company rules with expect_all_or_fail:
routing already excludes those rows, so a violation means a routing bug and fails
the update instead of publishing the row.

## Prompts

Each field has its own prompt (`TRANSLATION_PROMPTS`), and a country group can
override it where its conventions differ: Chinese, Korean, and Russian street addresses,
and Russian city and state names. Every other group, and rows with no group, use
the field's default prompt. One shared address prompt carried every field's rules
to every field. The 12B model then applied street rules to city values, rendering
the village prefix in `д. Курильск` as "Building 1 Kurilsk", and rendered Russian
`д.` house numbers as "Building". Specialized prompts keep each request short and
let a prompt show worked examples. The row's group is kept in
`translation_country_group`; the request key hashes the emitted prompt, so moving a
country to another group re-requests its fields.

The Russian city prompt drops the settlement-type prefix (`г.`, `д.`, `к.`, `п.`,
`с.`, `ст.`, `клх`), matching the place lookup (`к. Нерчинск` → `Nerchinsk`).

The `english` group (US, GB, AU, and CA values in `translation_country_groups`)
has its own street and city prompts. Non-Latin text in these rows is English
spelled out phonetically in Hindi or Thai (`6311 सारा प्लाजा सूट 034` is
`6311 Sara Plaza Suite 034`). The default prompts translated its meaning or
swapped in a famous place (`पश्चिम वनेसापोर्ट` became "West Vancouver" instead of
"West Vanessaport"), so these prompts ask for the original English spelling and
list the Hindi and Thai address labels. Thai addresses from Thailand are not in
the group and keep the default prompts.

## Deterministic lookups

City and state lookups join the `translation_place_lookup` and
`translation_country_groups` tables in addressresolution.default. Before joining,
each lookup is reduced to one row per key, so a duplicated table row can never
duplicate provider rows:

- Country values are compared trimmed and lowercased. A value listed under two
  different country groups is ambiguous and skips deterministic lookup.
- The state field prefers a `province` entry over a `city` entry; the city field
  uses `city` entries only.
- Blank or non-Latin English names are ignored, and a key with conflicting English
  names is skipped. Skipped values fall through to AI translation.

## Escalation

Requests whose primary response (databricks-gemma-3-12b) fails the checks are
stored in providers_translation_escalation_requests and sent once more, with the
same prompt, to `ESCALATION_MODEL` (system.ai.qwen35-122b-a10b). Its response
replaces the primary one only if the primary fails and the escalation passes the
same checks; otherwise the primary response stays, so review shows it. The
`_ai_model` column records the chosen model. In a test on the 155 requests Gemma
failed on 2026-10-24, Qwen passed 83, including Japanese company names Gemma
left in kanji. It is slow (about 7 minutes for those requests), which is why it
only sees failures.

The escalation model returns an empty result whenever `max_tokens` is set, so it
runs with temperature 0 only (`ESCALATION_TEMPERATURE`). Validation-only edits
reselect between stored responses without new inference: refresh
providers_translation_results and its branches as described above.

## Review branch

A Delta quarantine dataset is a practical dead-letter equivalent for a batch
pipeline. translation_route sends every result to exactly one place:

- `accepted` (providers_translated): no translation issue and no remaining
  cleaning failure.
- `review` (providers_translation_review): translation_needs_review is true. A
  null or unknown route also goes to review.
- `incomplete` (dlq_incomplete): translation passed, but a cleaning failure that
  translation cannot repair remains.

A row with any failed field is held for review as a whole, even if its other
fields translated successfully. The all-results stage preserves evidence for
auditing and retries.

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

Requests use `temperature` 0 (`AI_TEMPERATURE`) so a refresh reproduces the same
answers, and `max_tokens` 128 (`AI_MAX_TOKENS`). The longest valid response seen is
about 40 tokens; the cap stops runaway answers, such as a region name returned as
an invented full address. A capped answer exceeds the cleaning length bounds and
goes to review. Both settings are part of the request key.

Company names are checked for address leakage using street-level terms only
(street, road, building, unit, floor, …). Area words such as County, District,
Town, and Village are not leakage: organization names contain 县/区/町/村, so
`洋县 Regional Medical Center` → "Yang County Regional Medical Center" is correct.

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
        'databricks-gemma-3-12b', 'Return only OK.', failOnError => false
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
