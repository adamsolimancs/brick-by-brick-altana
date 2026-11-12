# Unified Fellegi–Sunter scoring on cleaned saved candidates

`unified_fellegi_sunter` fits a new Splink 5.0.0 / DuckDB 1.4.1 model and scores
the existing blocked + HNSW candidate union. It consumes completed string-scoring
run `956878669358056`, whose `street_components_v3` fields separate road names
from house, unit, building, and floor values. No embeddings, HNSW indexes,
blocking passes, or candidate generation are rerun.

The input contains 33,677 accepted snapshot records and 1,021,785 unique unordered
pairs: 37,139 blocked and 984,646 additional embedding candidates. All 5,544 exact
baseline pairs are included in inference and remain separately identifiable.

## Completed run — November 1, 2026

[Successful Databricks run](https://dbc-031efa75-388b.cloud.databricks.com/jobs/632401166051077/runs/525733446208790?o=7474651042679944).
Job: **[dev aes10130] Address Unified Fellegi-Sunter Scoring**.
Current `unified_run_id`: **`525733446208790`**;
model version: `unified_cleaned_jw_fs_v2`.
The [full pipeline diagram](PIPELINE.md) shows the current artifact flow and counts.

| Current decision | Blocked pairs | HNSW-only pairs | Total |
| --- | ---: | ---: | ---: |
| Exact baseline | 5,544 | 0 | 5,544 |
| Proposed fuzzy match | 6,192 | 1 | 6,193 |
| Review: conflict | 23,095 | 982,468 | 1,005,563 |
| Review: uncertain | 2,253 | 888 | 3,141 |
| Likely nonmatch | 55 | 1,289 | 1,344 |
| **All scored pairs** | **37,139** | **984,646** | **1,021,785** |

The 6,193 proposals are additional to the exact baseline, not 6,193 new pairs
relative to the historical model. Of the previous 7,037 blocked proposals, 6,179
remain proposed, 855 require uncertain review, one has a conflict, and two are
likely nonmatches. Thirteen other blocked pairs become proposed, plus the single
HNSW-only pair. All are suggestions rather than confirmed entity matches.

There are 22,189 raw model probabilities ≥0.99 across the whole union. The lower
proposal count reflects the independent address safeguards and minimum field
similarities. A raw high probability cannot bypass these checks.

The two EM sessions converged in **15** and **9** iterations. Every comparison
parameter was observed; no probability floors or fallback parameters were needed
in this run. Company exact-match term-frequency adjustment remained enabled.
The assumed global prior is `9.776858270292555e-06`, based on 5,544 exact pairs
among 567,053,326 possible pairs. Probability calibration remains unvalidated.

The job verified all three comparison gammas against saved JW features for every
pair, and reconstructed every match weight from the prior, field contributions,
and term-frequency adjustment. Independent read-only SQL checks found zero
missing/extra pairs, changed historical evidence, invalid probabilities/weights,
unsafe proposals, approved merges, or nonmonotone prior-sensitivity values.
All **41** local tests, Ruff checks, and strict bundle validation passed.

MLflow run: `e020f29673f14f94b08a0398c2f4b246`.
Source code SHA-256:
`609f4215d0c43bdbb49c6d143cc222174800e771f232e58485a0e2ca8a4b39ae`.
Task execution was 156 seconds plus 3 seconds of setup on serverless
PERFORMANCE_OPTIMIZED compute. The experiment's managed volume retains candidate
IDs, predictions, model, and manifest under this run ID.

The preliminary run `748494297034735` is superseded. Its probabilities were
valid, but its convergence guard counted parameter-history rows as iterations
and conservatively marked all non-exact cases for review. V2 uses each history
record's actual iteration number, with a regression test for that distinction.
Use the current run ID above for final decisions; historical outputs remain
available for traceability.

Private independent verification outputs are under
`data/labeling/outputs/01a18f67-e1af-7aa0-9eff-6eb8393dddda/unified_fs/verified_525733446208790/`.

## Model and fitting

There is one comparison per field, with mutually exclusive levels:

| Field | Comparison levels (first matching level wins) |
| --- | --- |
| Company | Missing; exact; JW ≥0.95; ≥0.85; ≥0.70; disagreement |
| Cleaned street name | Missing; exact; JW ≥0.95; ≥0.85; ≥0.70; disagreement |
| City | Missing; exact; JW ≥0.95; ≥0.85; disagreement |

Both training and inference use the same RapidFuzz 3.14.1 Unicode-aware
Jaro–Winkler implementation with prefix weight 0.1 as the saved string features.
Levenshtein remains available in the output as historical comparison evidence;
it receives no independent model weight. Company exact agreement has Splink's
term-frequency adjustment, using frequencies from all snapshot records.

Splink estimates `u` from a seeded sample targeting at most one million random
pairs from the **full record population**, rather than the embedding-selected
candidate distribution. Two EM sessions estimate `m`, using country + exact
company and country + exact cleaned street respectively. The comparison used for
each training block is excluded from that session's estimates. The blocks contain
21,206 and 65,873 pairs in this snapshot, within the explicit 200,000-pair budget
per session. The iteration limit is 100 with convergence tolerance 0.0001.

The global prior is the observed exact-baseline pair rate, floored at one possible
pair and capped at 0.1. This is an explicit assumption, not a measured duplicate
rate. The output includes 2x/4x prior sensitivity probabilities. No model-reviewed
field labels or human entity labels are used for training.

Unobserved comparison-level parameters receive an explicit probability floor of
1e-6; zero estimates receive the same floor. Each field's `m` and `u` distributions
are then normalized to sum to one. The final model and every adjustment are saved.
Pairs using an unobserved parameter cannot become proposed matches. Reaching the
EM iteration cap likewise forces review rather than permitting proposals.

## Exact candidate coverage

The public Splink `register_blocked_pairs_for_predict()` API registers the exact
saved candidate-ID table before `predict(threshold_match_probability=0)`. It does
not apply the training blocks to prediction. Country-crossing pairs and pairs
outside every original block still receive scores; guards handle conflicts later.

Only the bounded record table is collected to the driver. Candidate IDs and
numeric predictions use per-run Parquet files in the managed UC volume
`addressresolution.matching_experiments_aes10130.unified_scoring_artifacts`.
DuckDB uses a 2 GB memory limit, four threads, and ephemeral local spill storage.
The job stops above 100,000 records or 2,000,000 saved pairs. Outputs are isolated
by run ID and refuse overwrite/retry of a partial run.

## Decisions and safeguards

New output columns use `fs_*`; the existing `match_probability`, `decision`,
comparison metrics, source text, identities/hashes, cosine scores, and original
conflict flags remain unchanged.

Decision order:

1. `exact_baseline`: original five-field normalized exact rule agrees. Its FS
   probability is still computed; rule agreement is separate from calibrated truth.
2. `review_conflict`: any original house, suite, numeric, region, or country flag,
   or a disjoint parsed house/unit/building/floor component conflict.
3. `proposed_match`: FS probability ≥0.99; all three JW similarities ≥0.85; no
   conflict, partial/missing component review flag, parser ambiguity, unsupported
   comparison tier, or EM iteration-cap issue. Alternatively, original-text
   agreement (`fs_original_text_agreement`): translation rendered the same
   source text differently on the two records (森 Street as Mori Street and
   Forest St), so the text before translation is compared instead. Each of
   company, street name and city must either agree before translation
   (`original_company_agreement`, `original_street_name_agreement`,
   `original_city_agreement` from the string-feature stage) or meet the ≥0.85
   similarity; the company must agree before translation or reach ≥0.95; at
   least one field must rest on original text; and every conflict and review
   guard above still applies. The model probability is recorded but not
   required for these pairs.
4. `review_uncertain`: incomplete/ambiguous component or model evidence.
5. `likely_nonmatch`: probability ≤0.10 without an earlier condition.
6. Otherwise `review_uncertain`.

Missing components are unknown. Overlapping unequal component sets represent
partial agreement and require review, while disjoint known sets flag a conflict.
A parser warning that both records raise over the same parsed values (two
units on each side, `Suite 883, Apt 187`) is a shared ambiguity, not a
difference, and no longer forces review on its own; the first benchmark found
488 such pairs held back with identical units.
These checks are separate from the three probabilistic comparison weights.
Probabilities remain unsupervised and uncalibrated; correlated address fields
and retrieval selection can affect them.

Each pair also states what it authorizes. `fs_merge_authority` is `exact_rule`
for `exact_baseline`, `model_proposal` for `proposed_match`, and `none` for every
review or non-match decision. `fs_human_decision` is reserved for a reviewer's
verdict and is always NULL from scoring; nothing consumes it yet. The resolver
merges `exact_rule` and `model_proposal` pairs after re-checking the proposal
safeguards (see [DEDUPLICATION.md](DEDUPLICATION.md)). Runs before November 9,
2026 carry the retired column `fs_approved_for_merge`, which was always false
and did not describe what the resolver merged; it is NULL on newer rows.

## Outputs

| Table | Purpose |
| --- | --- |
| `unified_pair_scores` | All original input evidence plus current `fs_*` probabilities, weights, gammas, safeguards, and decisions |
| `unified_models` | Final model JSON and fitting/regularization manifest |
| `unified_decision_transitions` | Counts by candidate origin, historical decision, and current decision |
| `unified_review_sample` | Up to 30 deterministic review cases per origin/current decision |
| `unified_runs` | Completion marker, input provenance, counts, diagnostics, code hash, and MLflow reference |

Filter every output by a `unified_run_id` whose `unified_runs.status` is
`COMPLETED`. Use `fs_match_probability` and `fs_decision` for the current model.

```sql
SELECT candidate_origin, id_l, id_r, company_l, company_r,
       street_address_l, street_address_r, city_l, city_r,
       fs_match_probability, fs_match_weight, fs_decision,
       fs_mw_company, fs_mw_tf_adj_company, fs_mw_street_name, fs_mw_city,
       parsed_house_numbers_l, parsed_house_numbers_r,
       parsed_units_l, parsed_units_r, parsed_buildings_l, parsed_buildings_r,
       fs_guard_conflict, fs_guard_review,
       match_probability AS historical_probability,
       decision AS historical_decision
FROM addressresolution.matching_experiments_aes10130.unified_pair_scores
WHERE unified_run_id = '525733446208790'
  AND fs_decision = 'proposed_match'
ORDER BY candidate_origin, fs_match_probability DESC;
```

## Run and verification

From this bundle directory, with an explicitly selected profile:

```bash
databricks bundle validate --strict -t dev --profile <selected-profile>
databricks bundle deploy -t dev --profile <selected-profile>
databricks bundle run unified_fellegi_sunter -t dev --profile <selected-profile> \
  --params input_string_scoring_run_id=956878669358056
```

The job verifies source record hashes and comparison keys, exact pair/origin
coverage, preservation of every input column, all three comparison gammas against
saved JW scores, probability/weight validity, decomposition of each match weight,
exact-baseline coverage, and inability of any proposal to bypass safeguards.
`scripts/verify_unified_run.py` independently checks completed tables, model
distributions, deployed source hash, probabilities, original evidence, decisions,
and prior sensitivity through read-only SQL.

Primary references:
[Splink precomputed pairs](https://moj-analytical-services.github.io/splink/api_docs/table_management.html#register_blocked_pairs_for_predict),
[Splink training](https://moj-analytical-services.github.io/splink/api_docs/training.html),
[Splink term-frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html),
[UC volume file operations](https://docs.databricks.com/aws/en/volumes/volume-files).
