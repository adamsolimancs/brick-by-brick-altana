# First matching experiment

The deployed end-to-end refresh is documented in [Address scoring and
deduplication](DEDUPLICATION.md). Run **Silver → Address Deduplication** for the
combined workflow. It publishes the source-level audit table
`addressresolution.silver.addresses_deduplication` and the canonical output
`addressresolution.gold.addresses_deduplicated`.

An isolated, manually triggered Databricks job comparing the existing exact
company/location equivalence rule with multi-pass blocking and unsupervised
Fellegi-Sunter scoring using Splink 5.0.0 / DuckDB 1.4.1.

The job reads committed silver data and writes only to
`addressresolution.matching_experiments_<developer>`. It has no schedule and
does not change silver/gold pipelines, approve fuzzy links, or call LLMs.

For additional candidates outside all existing blocked/scored pairs, see the
separate [embedding + HNSW job](EMBEDDINGS.md). It embeds the same completed input
snapshot and produces review candidates with cosine similarities.

To add Levenshtein and Jaro–Winkler comparisons to saved candidates without
repeating embeddings or HNSW retrieval, use the
[string-feature job](STRING_FEATURES.md).

For a new Fellegi–Sunter model over the **entire saved blocked + HNSW union**,
using cleaned street names and the component safeguards, see
[unified scoring](UNIFIED_SCORING.md). Current columns use the `fs_*` prefix;
the historical blocked-only results remain intact.
See the [full pipeline diagram](PIPELINE.md) for current pair counts and data flow.
A frontier model can label a stratified sample as a second opinion on the
merge decisions; see [LLM_LABELING.md](LLM_LABELING.md).
The [HNSW guard audit](HNSW_GUARD_AUDIT.md) explains the one-pair result,
inferred-country issues, and a locally tested parser fix awaiting deployment.

## Run

From this directory, using the explicitly selected workspace profile:

```bash
databricks bundle validate --strict -t dev --profile aes10130@nyu.edu
databricks bundle deploy -t dev --profile aes10130@nyu.edu
databricks bundle run matching_experiment -t dev --profile aes10130@nyu.edu
```

The dev bundle creates the original serverless performance-optimized matching
job, separate STANDARD embedding and string-feature jobs, and one shared MLflow
experiment.
Performance-optimized mode is for the interactive matching experiment; use
STANDARD if this becomes an unattended batch workflow. The environment version
comes from the current official CLI template and dependencies are pinned in the
job resource. This is a bounded single-node experiment, not a claim that driver
collection scales to millions of records. It fails above 100,000 input records or
2,000,000 candidate-pass rows; it does not silently truncate or sample the inputs.

## Input and reproducibility

- `addressresolution.silver.cleaned_1_0` supplies accepted clean records.
- `providers_translated` is checked against accepted
  `providers_translation_results` records before their original text and
  translation diagnostics are retained. A disagreement stops the experiment.
- The union is materialized to a run-keyed Delta snapshot before fitting.
  Duplicate/missing source identities and missing required business fields are
  retained with input rejection reasons and excluded from matching.
- Composite source identity is `(provider, provider_id)`. IDs are not compared
  across providers as if they described the same source observation.
- The source snapshot, derived records, model JSON, exact prior assumption,
  dependency versions, code hash, input fingerprint, and thresholds are retained.
  Library source is also logged to MLflow. Runs can be compared without changing
  prior results. Only `runs.status = 'COMPLETED'` identifies fully written outputs;
  failed runs may retain useful partial snapshots and an MLflow failure record.

## Blocking

Candidate pairs are unique and unordered. Both within-provider and
cross-provider pairs are included. The union of five passes is evaluated:

1. Original case/Unicode-whitespace-normalized company and every address field,
   preserving the exact baseline's pair coverage.
2. Country + normalized region + city + parsed leading building number.
3. Country + normalized region + a six-character company comparison prefix.
4. Country + company comparison name, rescuing region/city inconsistencies.
5. Country + city + an eight-character street comparison prefix.

Empty keys do not create wildcard blocks. Block sizes and potential pair counts
are checked before expansion. The experiment stops on oversized blocks rather
than silently dropping candidates. State/province aliases are country-scoped,
with a fallback so a known full name and its code still agree under a country
that has no table (`Maryland` and `MD` filed under TR); street aliases apply
only to US/CA/GB/AU comparison views. The company comparison name drops
legal-entity forms at either end (Inc, Ltd, Co., Ltd., GmbH, S.A., Sp. z o.o.,
株式会社, 주식회사, ООО) as a comparison view only. Raw values are retained.

The parser deliberately handles only leading building numbers and explicit
trailing units. Other international formats remain unparsed. Number tokens are
retained as an additional conservative guard. Comparison keys fold accents on
Latin letters only (Satō and Sato, Ürümqi and Urumqi); marks in Thai, Devanagari
and other scripts are kept. There is no geocoding, external enrichment,
libpostal deployment, or transliteration of source text in this version.
Country is a blocking constraint; wrong inferred countries can still miss true
pairs. This and prefix-block misses need evaluation with reviewed labels.

## Probabilistic model

Three comparison groups use exact and Jaro-Winkler levels: company comparison
name (with term-frequency adjustment), normalized street, and locality. Building,
unit, numeric, and region disagreements remain explicit decision guards rather
than being independently counted again as extra probabilistic evidence.

Splink estimates non-match `u` distributions from at most one million sampled
pairs with a recorded seed. It estimates match `m` distributions in two EM
sessions, blocking on country/name and country/street respectively. The comparison
used for a training block is excluded by Splink from that session's estimation;
the other session supplies its estimates. EM training is capped at 200,000 pairs
per session. Fewer than ten available pairs stops training. Unobserved/untrained
levels and any library-default use are recorded in the run manifest.

The global prior uses the observed exact-pair rate, floored at one possible pair.
That is an explicit assumption, not a measured duplicate rate or a proven lower
bound. Per-pair probabilities under 2x and 4x priors are retained as sensitivity
checks. Fuzzy probabilities are uncalibrated, and correlated address fields can
make them overconfident. None is presented as measured accuracy.

Outcomes:

- `exact_baseline`: original exact rule agrees; remains separately identifiable.
- `proposed_match`: probability at least 0.99, strong name/street/locality
  evidence, and no explicit numeric/unit/region conflict or missing unit on only
  one side; or agreement of the text before translation on the fields where
  the translated renderings differ. This is a reviewable suggestion, not an
  approved merge.
- `review_conflict`: explicit building, numeric, suite, or region disagreement.
- `review_uncertain`: ambiguous probability or incomplete unit information.
- `likely_nonmatch`: probability at most 0.10 and no earlier review condition.

Missing units are unknown, not conflicts. The existing baseline's
`match_rule_score = 1.0` denotes rule agreement and is not compared numerically
with model probabilities. No transitive fuzzy clustering is performed.

## Outputs

All derived datasets carry `experiment_run_id`; append-only runs are isolated.

| Table | Purpose |
| --- | --- |
| `input_snapshots` | Source records, original/translation evidence, input rejections |
| `prepared_records` | Accepted original records and versioned comparison fields |
| `candidate_passes` | Unordered source pairs and every generating pass |
| `blocking_audit` | Candidate-pass counts and largest block sizes |
| `pair_scores` | Pair evidence, component weights, probabilities, guards, outcomes |
| `exact_clusters` | Exact baseline groups with source/provider counts |
| `comparison_counts` | Baseline, proposed, rejected, and review pair counts |
| `review_sample` | Deterministic sample stratified by outcome and input routes |
| `runs` | Completion manifest, model assumptions, summary metrics |
| `match_labels` | Persistent human labels, independent of experiment outputs |

The model and manifest are also logged to MLflow. The notebook's final output
returns the run ID, output schema, and summary. Use the matching job's run page
to see execution status, notebook output, and the linked MLflow run.

## Evaluation limits and next step

Initial discovery on October 29 found 25,969 clean rows and 7,708 accepted
translated rows, no overlapping IDs, and no reviewed matching labels. Actual
experiment counts come from its own materialized snapshot.

The comparison measures candidate reduction, exact-baseline pair preservation,
additional suggested pairs, and conflict/review volume on the **same inputs**.
Exact-baseline coverage is a regression check, not true-match recall. Precision
and recall remain null until there is independently reviewed ground truth.
The synthetic unit fixtures validate behavior, not real matching accuracy.

Review `review_sample`, then populate `match_labels` with ordered source IDs,
both content hashes, reviewer, time, notes, and a label such as
`same_company_same_location`, `different_location`, `different_company`, or
`insufficient_evidence`. Audit pairs outside the current blocks too: candidate-only
labels cannot reveal missed matches. Keep related company/location variants
together when creating training and evaluation splits.

## Local validation

From the repository root, with the pinned libraries in the ignored `.venv`:

```bash
uv pip install --python .venv/bin/python splink==5.0.0 duckdb==1.4.1 \
  faiss-cpu==1.13.2 numpy==2.2.6 pandas==2.2.3 rapidfuzz==3.14.1
ruff check experiments/matching
.venv/bin/python -m unittest discover -s experiments/matching/tests -v
```

Tests execute candidate SQL, fixed-parameter Splink inference, conflict guards,
pair conservation, duplicate-ID rejection, and the real PySpark gold baseline's
equivalence contract. They do not fit a model locally. Model fitting and the
end-to-end comparison run in Databricks.
