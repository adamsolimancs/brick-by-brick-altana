# First live matching experiment

Completed October 29, 2026, using the explicitly selected `aes10130@nyu.edu`
profile. [Successful Databricks run](https://dbc-031efa75-388b.cloud.databricks.com/jobs/785262215792809/runs/339731647991599?o=7474651042679944),
[MLflow experiment](https://dbc-031efa75-388b.cloud.databricks.com/ml/experiments/3205418160044522?w=7474651042679944).

Experiment run ID: `ea2277b0d4444cee8d744ba5277b523e`.
Outputs: `addressresolution.matching_experiments_aes10130`.
Input fingerprint: `e6b54ab5a7626a4e9afbe94f9c80a43b7d41785cf40952c86d6767f94998edf5`.

## Same-input comparison

Both methods used the same materialized snapshot of 33,677 records: 25,969
clean records and 7,708 accepted translated records. No input rows were rejected.

| Measure | Observed result |
| --- | ---: |
| Possible unordered pairs | 567,053,326 |
| Unique blocked pairs scored | 37,139 |
| Pair comparison reduction | 99.99345% |
| Exact-baseline groups, including singletons | 28,271 |
| Exact-baseline matching pairs | 5,544 |
| Exact-baseline pairs preserved | 5,544 / 5,544 |
| Additional proposed matching pairs | 7,037 |
| Proposals with exact normalized name/street/city agreement | 4,799 |
| Proposals involving a fuzzy comparison | 2,238 |
| Candidate pairs held for explicit location conflicts | 22,842 |
| Other uncertain/review pairs | 1,531 |
| Likely non-matching pairs | 185 |
| Approved fuzzy merges | 0 |

The reduction is a computational measure, not recall. The additional proposed
pairs are hypotheses, not confirmed duplicates, and pair counts are not counts
of additional resolved companies. There are no reviewed matching labels, so
precision and recall were not measured. No fuzzy clusters or canonical production
records were changed.

## Contribution of each blocking pass

The additional column counts unique pairs first discovered by that pass, in the
implemented order. A pair may appear in several passes, so raw counts overlap.

| Pass | Raw pairs | Additional unique pairs |
| --- | ---: | ---: |
| Original exact baseline | 5,544 | 5,544 |
| Country/region/city/building number | 14,964 | 13,356 |
| Country/region/company prefix | 17,761 | 7,702 |
| Country/company-name rescue | 21,206 | 9,006 |
| Country/city/street prefix | 23,043 | 1,531 |
| Total unique candidates | | 37,139 |

## Model and sensitivity

Splink 5.0.0 with DuckDB 1.4.1 fitted three comparison groups: company name,
street, and locality. The company comparison uses term-frequency adjustment.
Non-match probabilities were estimated with a seeded sample capped at one
million pairs. EM training had 21,206 available country/name pairs and 12,886
available country/street pairs. No comparison levels used untrained library
defaults in this run.

The assumed prior was approximately `0.00000977685827`, derived from the
observed exact-pair rate. It is not a measured true-duplicate probability.
Multiplying the prior by four would add 24 otherwise eligible proposals at the
0.99 threshold. This is a limited prior-sensitivity check, not calibration.

The original run manifest labeled its parameter-history row counts as
`iterations`; these were not EM iteration counts. That diagnostic key is corrected
to `parameter_history_records` for subsequent runs. This correction does not
change blocking, model fitting, scoring, thresholds, or the results above.

## Checks on saved outputs

Read-only SQL checks against the completed Delta outputs verified:

- 37,139 scored pairs, with no duplicates or invalid unordered-pair orientation.
- No null, NaN, or out-of-range probabilities.
- All 5,544 exact-baseline pairs retained.
- No proposed match violated the configured number, unit, region, similarity,
  or probability guards.
- No pair was marked approved for merging.

Conflict flags overlap: 5,431 pairs had parsed building-number disagreements,
8,961 had unit disagreements, 12,135 had other numeric-signature disagreements,
and 10,099 had region disagreements. These are signals, not reviewed labels.

Local validation passed six tests, including real PySpark equivalence against
the existing gold baseline and fixed-model Splink inference. Ruff checks,
Python compilation, and strict bundle validation passed. Model fitting and the
end-to-end comparison ran in Databricks; no model was fitted locally.

## Interpretation and next step

Most additional suggestions came from normalizing formatting, punctuation,
suffixes, and geographic aliases. Another 2,238 used fuzzy name, street, or city
comparisons and deserve particular scrutiny. Location-conflict guards prevented
the scorer from treating many superficially similar records as proposed matches.

Start reviewing the persisted `review_sample`, prioritizing fuzzy proposals and
potential branch/co-tenant cases. Populate the independent `match_labels` table
with content-versioned judgments. Also audit excluded pairs to measure blocking
misses; reviewing candidates alone cannot establish end-to-end recall. Use those
labels to evaluate the scorer and choose thresholds before approving merges.

See [README.md](README.md) for deployment and model assumptions, and
[checks.sql](checks.sql) for repeatable read-only verification queries. Detailed
source rows and review examples remain in Databricks rather than in this report.
