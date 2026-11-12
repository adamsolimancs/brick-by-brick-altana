# Levenshtein and Jaro–Winkler on saved candidates

**Address Candidate String Features** is a separate serverless STANDARD job.
It reads existing candidate pairs and their saved normalized source records.
It has no embedding, model fitting, or HNSW tasks and generates no new pairs.

The input is a completed `embedding_runs` entry, defaulting to the latest
completion. The job scores both its saved `embedding_candidates` and the original
matching run's `pair_scores`. The sets must be disjoint. The original tables are
read-only inputs; results append to separate string-feature tables.

## Completed run — October 31, 2026

The current cleanup result is the November 1 v3 run below. This section records
the original v1 result for reproducibility.

[Successful Databricks run](https://dbc-031efa75-388b.cloud.databricks.com/jobs/950913600552982/runs/95648575728701?o=7474651042679944).
String scoring run ID: `95648575728701`.
Input embedding run: `6082159854758`; source matching run:
`ea2277b0d4444cee8d744ba5277b523e`.
MLflow run: `95fba3ac8a8246b1b54e947a9d72150c`.

| Saved pair origin | Scored pairs |
| --- | ---: |
| HNSW embedding candidates | 984,646 |
| Original blocked candidates | 37,139 |
| Total | **1,021,785** |

All nine added metric columns were computed for every pair. This particular
snapshot had no empty/unknown comparison keys and no invalid metric rows.
The complete input pair set and every original evidence field were preserved,
including original Jaro–Winkler scores, probabilities, decisions, and conflict
flags. Independent read-only checks confirmed no duplicate/missing pairs,
invalid pair orientations, changed building/suite guards or decisions, or
approved merges. The stored Levenshtein distances and normalized similarities
also matched independent recomputation.

No embeddings or HNSW indexes were rebuilt. The run executed only
`score_saved_pairs`: 129 seconds of task execution, about 5 minutes 25 seconds
including STANDARD provisioning. All 15 local matching tests, Ruff checks, and
strict workspace bundle validation passed. The local runtime code hash matches
the successful run's manifest.

Use `95648575728701` for `:string_scoring_run_id` to reproduce v1. Existing
decision counts remain unchanged; the new features introduce no acceptance rule.

## Completed cleanup — November 1, 2026

[Successful v3 Databricks run](https://dbc-031efa75-388b.cloud.databricks.com/jobs/950913600552982/runs/956878669358056?o=7474651042679944).
Use **`956878669358056`** for current string-scoring queries. This run reuses
embedding run `6082159854758` and matching snapshot
`ea2277b0d4444cee8d744ba5277b523e`; it rebuilds no embeddings or HNSW indexes.
MLflow run: `2ca9bbbb674248f1a68c892ddcfa20fe`.

| Audit result | Count |
| --- | ---: |
| Accepted source records parsed | 33,677 |
| Records with changed street comparison keys | 15,732 (46.7%) |
| Saved candidate pairs preserved | 1,021,785 |
| Pairs with changed street comparison inputs | 583,862 |
| Records with separate building values | 4,931 |
| Records with separate unit values | 12,761 |
| Records with separate floor values | 2,607 |
| Ambiguous trailing-number formats retained | 933 |
| Records with no remaining road name | 2 |

The two empty road names came from unit-only strings (`Studio 48` and
`Studio 78k`). Their 275 candidate pairs have unknown street similarities rather
than exact matches. All nine metrics passed range/null checks. The 46 retained
component-word warnings include legitimate road names ending in Flat or Block,
so that count is not an address-error estimate.

Independent full-pair SQL checks confirmed zero missing/extra pairs, changed
source identities/hashes, changed old decisions/guards/probabilities/cosines,
changed company/city comparison values or metrics, or altered copies of the old
street keys. No exact-baseline pair gained a disjoint component conflict.
The local code hash matches the completed manifest:
`43148471cb675460766e9ec377bf3a13f27c63ed5c7b570fec3b851e9b012a63`.
All 36 local tests, Ruff, and strict bundle validation passed. Task execution was
129 seconds, plus 195 seconds of STANDARD setup.

Compared with old house/unit/numeric flags, 46,035 pairs gain additional parsed
component conflict evidence. One historical `proposed_match` has a disjoint
component flag, and 792 have the broader review-required flag (including missing
components, partial information, and parser ambiguity). These counts do **not**
establish false matches. For example, translations can label the same number as
a house, building, or apartment differently. Original evidence remains available
for adjudication; old decisions and probabilities remain historical.

An initial v2 audit run (`720218095856775`) is retained for traceability and is
superseded by v3. The full-snapshot audit motivated tighter trailing-number rules,
preservation of numbered Russian road names, and separate partial-agreement
flags. Always filter by the v3 run ID above when using the current cleanup.

The reusable `scripts/verify_street_cleanup.py` checks the completed run against
v1, confirms the deployed code hash, and reevaluates fixed reference labels.
Private verification/evaluation outputs are under
`data/labeling/outputs/01a18f67-e1af-7aa0-9eff-6eb8393dddda/component_cleanup/verified_956878669358056/`.
The same 150 model-generated labels were retained; only compared input strings
and saved scores were refreshed after source identities/hashes were verified.
See [LABELING.md](LABELING.md) for the updated pilot interpretation.

## Run

From this directory, with the explicitly selected workspace profile:

```bash
databricks bundle validate --strict -t dev --profile <selected-profile>
databricks bundle deploy -t dev --profile <selected-profile>
databricks bundle run candidate_string_features -t dev --profile <selected-profile> \
  --params input_embedding_run_id=6082159854758
```

The selected embedding run fixes the exact source matching run and snapshot.
The job verifies source identities, content hashes, saved counts, unique unordered
pair identities, and absence of overlap between the two candidate sets.
`max_candidate_pairs` defaults to 2,000,000 and fails rather than truncating.
The new `string_scoring_run_id` is the Databricks job run ID.

## Added fields

Each comparison has three new fields:

| Field prefix | Saved normalized input | Added metrics |
| --- | --- | --- |
| `company` | `name_key` | `company_levenshtein_distance`, `company_levenshtein_similarity`, `company_jaro_winkler_similarity` |
| `street_name` | `street_component_road_key` (v3; `road_key` in v1) | `street_name_levenshtein_distance`, `street_name_levenshtein_similarity`, `street_name_jaro_winkler_similarity` |
| `city` | `city_key` | `city_levenshtein_distance`, `city_levenshtein_similarity`, `city_jaro_winkler_similarity` |

`comparison_<field>_l` and `_r` retain the exact strings compared.
`comparison_building_l/r` and `comparison_suite_l/r` retain the **legacy** snapshot
number and suite keys for compatibility. In v3, use the additional
`parsed_house_numbers_l/r`, `parsed_units_l/r`, `parsed_buildings_l/r`, and
`parsed_floors_l/r` arrays for the reparsed component evidence.

Raw edit distance uses Spark's native `functions.levenshtein` without a cutoff,
so every known pair gets its exact nonnegative distance rather than a `-1`
threshold sentinel. Normalized similarity is:

```text
1 - edit_distance / max(character_length_left, character_length_right)
```

It ranges from 0 to 1; higher means closer text. Empty, whitespace-only, and null
keys produce null metrics, including when both inputs are empty. Missing values
are unknown rather than exact agreements.

Jaro–Winkler uses pinned RapidFuzz 3.14.1 with prefix weight 0.1 in Arrow-batched
Spark pandas UDFs. These explicit new columns compare the same normalized strings
as Levenshtein. Legacy `name_similarity`, `street_similarity`, and
`city_similarity` remain unchanged. In particular, legacy `street_similarity`
compares the full normalized street; `street_name_*` compares the cleaned
street-name view in v3. This does not retrain the old Splink model.

## Street components v3

The old snapshot parser only removed a unit at the end of the address. Chained
components such as `Suite 368, Floor 12`, leading apartments, building letters,
and many trailing house-number formats consequently leaked into `road_key`.
Starting with `levenshtein_jaro_winkler_v3`, the string job reparses each accepted
raw street **once per record**, materializes its components, then joins them to
the saved candidate pairs. Neither snapshot fields nor old run outputs change.

The shared `altana_matching_components` package provides the parser and the same
normalization/alias functions already used by matching. Alias and normalization
semantics are unchanged. This separate package also avoids colliding with the
repository's other `lib` package on Spark workers.

The parser handles explicit unit, building/block, floor/level, and house labels;
repeated components; prefix flat/studio formats; country-scoped local labels;
Chinese reversed building letters; missing CamelCase separators before numeric
component labels; and recognized trailing house-number formats
in DE/NL/RU/KR/ES/IT/BR/MX/PL. Australian `85/18 Baker Highway` is separated into
unit `85`, house `18`, and road `baker highway`. Compound slash numbers elsewhere
are retained as a component without assuming Australian semantics.

Examples:

| Accepted street | Clean comparison key | Separate evidence |
| --- | --- | --- |
| `157 Bullock Place Suite 368, Floor 12` | `bullock place` | House 157; unit 368; floor 12 |
| `858 Bennett Knoll Suite 541, Suite 970` | `bennett knoll` | House 858; both units retained and flagged as multiple |
| `Guangzhou Road H Building, Suite 242` | `guangzhou road` | Building h; unit 242 |
| `Gehringerweg 47-26` | `gehringerweg` | House 47-26 |
| `1209 5th Avenue Suite 4` | `5th ave` | House 1209; unit 4; road ordinal retained |

Ambiguous bare numbers, unsupported formats, and unrecognized labels stay in the
text with warnings. `Route 66`, `7 Mile Road`, `Plein 1945`, and numbers within
Korean road names are preserved. This bounded parser is not a universal postal
validator. Korean administrative/locality text may remain in the road view;
truncations and mistranslations are not guessed away.

`previous_street_name_l/r` preserves the old comparison key.
`street_parse_actions_l/r` and `street_parse_warnings_l/r` explain extraction and
remaining ambiguities. Empty cleaned road names produce unknown similarities.

## Guards and decisions

Building, suite, one-sided missing-suite, numeric, region, and existing country
conflict flags are copied with the original decisions, probabilities, and cosine
scores. A full multiset comparison against the input verifies their preservation.
For example, `123 Main St Suite 4` and `124 Main St Suite 9` have identical street
names, but their original building/suite conflicts still hold.

V3 additionally exposes `parsed_<component>_conflict`,
`parsed_<component>_partial_agreement`, and
`parsed_<component>_missing_one_side`, where `<component>` is `house_numbers`,
`units`, `buildings`, or `floors`. Two known component sets flag a conflict only
when they are disjoint. Overlapping unequal sets (including subsets) require
review as partial information; their component hierarchy is not inferred.
Missing values are unknown, not conflicts. `parsed_address_conflict` combines
the four disjoint-conflict flags. `parsed_address_review_required` additionally
includes partial agreements, one-sided missing components, and parser warnings
other than retained road-name numbers or a `multiple_*` warning both records
raise over identical parsed values. Records also carry `original_name_key`,
`original_road_key` and `original_city_key` (the same keys computed on the text
before translation, NULL when a field was not translated), and pairs carry
`original_company_agreement`, `original_street_name_agreement` and
`original_city_agreement` when both sides have that key and it is equal.
These are review cues, not proof of
different entities. Historical decisions are preserved; no links are approved.

No new acceptance threshold or combined probability is introduced. The added
metrics are evidence for review or future labeled scoring. None is a calibrated
match probability.

## Outputs

All outputs stay in `addressresolution.matching_experiments_<owner>`.

| Table | Purpose |
| --- | --- |
| `string_pair_scores` | Existing pair evidence plus normalized comparison strings and nine new metrics |
| `string_record_components` | V3 per-record raw/legacy keys, clean road name, extracted component arrays, version, actions, and warnings |
| `string_review_sample` | Deterministic sample of up to 30 pairs per candidate origin and original decision |
| `string_scoring_runs` | Completion manifest, input run IDs, pair counts, feature summaries, dependencies, code hash, and MLflow run |

Filter outputs by a `string_scoring_run_id` with a `COMPLETED` marker.
`candidate_origin` distinguishes `embedding` from `blocked`; absent original
fields are null after union. Original blocked-pair IDs are named `id_l/id_r` in
the new output to match the embedding-pair convention. Both sets retain
`source_run_id`, while `input_embedding_run_id` records the selection context.
The original FS `experiment_run_id` and embedding `embedding_run_id` are retained
where available.

Both auto-optimization retries and ordinary task retries are disabled. Existing
output rows are checked before appending; start a new job run after failure.
Failed output rows without a completion marker must not be mixed into results.

## Review query

For a score-blind human-labeling pilot comparing the two metrics, see
[LABELING.md](LABELING.md). The pilot samples existing saved pairs and does not
rerun embeddings or string scoring.

Replace `:string_scoring_run_id` with a completed run from `string_scoring_runs`:

```sql
SELECT candidate_origin, id_l, id_r, decision,
       company_l, company_r, street_address_l, street_address_r,
       comparison_street_name_l, comparison_street_name_r,
       company_levenshtein_distance, company_levenshtein_similarity,
       company_jaro_winkler_similarity,
       street_name_levenshtein_distance, street_name_levenshtein_similarity,
       street_name_jaro_winkler_similarity,
       city_levenshtein_distance, city_levenshtein_similarity,
       city_jaro_winkler_similarity,
       house_conflict, unit_conflict, unit_missing_one_side,
       parsed_house_numbers_l, parsed_house_numbers_r,
       parsed_units_l, parsed_units_r, parsed_buildings_l, parsed_buildings_r,
       parsed_floors_l, parsed_floors_r,
       parsed_address_conflict, parsed_address_review_required,
       match_probability, record_cosine, address_cosine
FROM addressresolution.matching_experiments_aes10130.string_review_sample
WHERE string_scoring_run_id = :string_scoring_run_id
ORDER BY candidate_origin, decision, id_l, id_r;
```

## Validation

Local Spark tests execute the actual native Levenshtein expressions and
pandas-UDF Jaro–Winkler comparisons. They verify known edit distances, Unicode
character-length normalization, null/empty handling, stale/missing source
records, original-field preservation, and perfect street-name similarity despite
building/suite conflicts. Component fixtures cover repeated/leading units,
country-scoped house numbers, floor/building order, ambiguous numbers, and
numbered roads. Live validation checks record/pair conservation, all original
evidence, and metric ranges before writing a completion marker.

Primary references:
[Spark Levenshtein](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/api/pyspark.sql.functions.levenshtein.html),
[Spark pandas UDFs](https://spark.apache.org/docs/latest/api/python/tutorial/sql/arrow_pandas.html),
[RapidFuzz Jaro–Winkler](https://rapidfuzz.github.io/RapidFuzz/Usage/distance/JaroWinkler.html).
