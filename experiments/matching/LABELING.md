# Compare string metrics with human labels

The November 1, 2026 pilot contains **300 existing candidate pairs** and **900
field judgments**: company name, street name, and city. It reads completed
string-scoring run `95648575728701` from
`addressresolution.matching_experiments_aes10130.string_pair_scores`.
No embeddings, indexes, candidates, probabilities, or decisions are rebuilt.

Private exports live under the gitignored directory
`data/labeling/outputs/01a18f67-e1af-7aa0-9eff-6eb8393dddda/`.
The workbook contains no comparator scores, old model decisions, or sample
strata. Those are kept in a separate analysis file.

## Labeling

Use **Field labels** in `candidate_pair_labeling.xlsx`. For each displayed
normalized value pair, enter `same`, `different`, or `uncertain`, your reviewer
name, and any notes. Label name/value agreement, rather than inferring entity
identity from equal text. Company names can agree without proving the same
legal company; street and city names can agree across different locations.
Street-name judgments exclude building and suite numbers. The full accepted
records and available original-language text provide context.

Read the workbook's **Guide** for the rubric. **Pair review** is optional and
captures the separate question of company/location identity, including building
and suite information. Keep pair references and source values unchanged.

Have two reviewers label P001–P025 independently in separate copies, discuss
disagreements, and settle the rubric before dividing the remaining pairs by pair
reference. Preserve independent labels when measuring reviewer agreement and
record adjudication separately. Use `uncertain` where evidence is insufficient.
Do not open `analysis_only_scores.csv` until labeling is finished.

## Sampling design

The saved population contains 1,021,785 pairs: 37,139 from blocking and 984,646
additional HNSW pairs. The pilot draws 170 blocked pairs and 130 HNSW pairs:

| Source | Prior decision (hidden from annotators) | Sample |
| --- | --- | ---: |
| Blocked | exact_baseline | 20 |
| Blocked | proposed_match | 50 |
| Blocked | review_conflict | 40 |
| Blocked | review_uncertain | 40 |
| Blocked | likely_nonmatch | 20 |
| HNSW | review_embedding_candidate | 70 |
| HNSW | review_conflict | 60 |

Within those groups, strata cover exact agreements, comparator ranking
disagreements, and other cases. A ranking disagreement means one algorithm
places a field in its own top-scoring decile while the other does not. Each
algorithm has its own field-specific cutoff; these are sampling controls, not
match thresholds. If several fields disagree, a seeded hash chooses one focus.
All 44 nonempty strata are represented; 208 sampled pairs have a disagreement
focus, 48 have all three fields exact, and 44 have neither focus.

Within each stratum, a seeded SHA-256 order selects pairs without replacement.
Display order and A/B orientation are shuffled independently and reproducibly.
The manifest records each stratum's population N and sample n. The analysis
file retains stable pair IDs, side mappings, scores, previous flags, inclusion
probability n/N, and sampling weight N/n. Weights reconstruct the full saved
candidate population. Raw sample percentages are not population estimates.

## After labeling

Join field labels to scores by `pair_ref` (or the full `pair_id` in the CSVs)
and field. Compare Jaro–Winkler with normalized Levenshtein similarity on the
same adjudicated cases, separately for company, street name, and city. Keep
uncertain labels separate and report their rate.

Choose each metric's threshold on development labels, then evaluate on held-out
labels, or use grouped cross-validation if the pilot is too small for a useful
holdout. `evaluation_group` keeps shared record IDs and identical normalized
company names together; review additional related-company groupings before
splitting. It is a leakage-prevention hint, not an inferred entity ID.
Use sampling weights for candidate-population precision/recall estimates and
report uncertainty and performance on comparator-disagreement cases. Compare
recall at a shared precision target or precision at a shared recall target;
using the same numeric threshold for both algorithms is not a fair comparison.
This pilot may justify a larger targeted sample before declaring a winner.

These labels evaluate field agreement within the existing candidate pool. They
do not measure matches missed by blocking/HNSW or establish overall entity-match
probabilities. Building/suite guards remain separate evidence. No model is
changed automatically by labeling.

### Run the local evaluation

After labeling and adjudication, pass the completed workbook (or field-label
CSV) to the evaluator. It reads the files without modifying them and needs no
Databricks connection:

```bash
.venv/bin/python experiments/matching/scripts/evaluate_field_labels.py \
  --labels /absolute/path/to/completed_candidate_pair_labeling.xlsx \
  --sample data/labeling/outputs/01a18f67-e1af-7aa0-9eff-6eb8393dddda/raw/sample.json \
  --output data/labeling/evaluation-a-new-directory \
  --target-precision 0.95 --folds 5
```

The precision target is an exploratory setting; change it to reflect the desired
operating point. Each metric gets its own threshold on training folds. The same
held-out cases are used for both metrics. When a metric cannot achieve the target
on training labels, it predicts no agreements for that fold; those test cases
are retained. Folds whose training data lack either class cannot be evaluated.
The report exposes test coverage and per-fold thresholds rather than silently
dropping failed fits.

Outputs are `evaluation.md` and `evaluation.json`, with label counts, weighted
and unweighted out-of-fold precision/recall/F1, weighted ROC AUC, comparator-focus
diagnostics, and input SHA-256 hashes. Full-label development thresholds are
identified separately from held-out results. No winner is chosen automatically,
no confidence intervals or significance claims are produced, and no production
rules change. Missing or uncertain labels may be selective; original sampling
weights alone do not correct label nonresponse.

The evaluator rejects changed compared values, stale pair IDs where supplied,
duplicate unadjudicated judgments, invalid labels, and labels without a reviewer.
A blank workbook produces `labels_required` with no accuracy or fitted
thresholds. The November 1 dry run confirmed **0/900 judgments entered**, so the
human-ground-truth comparison remains pending human labeling.

### Model-reviewed pilot — November 1, 2026

At the user's request, Codex reviewed P001–P050 (150 field judgments) directly
from the score-blind field/pair CSVs, using accepted records and available
original-language context. All labels and written explanations were finalized
and hashed before inspecting per-pair comparator scores. Labels are explicit
name/value judgments, not inferred legal-entity matches. No external verification
was performed and no human labels were substituted or overwritten.

The private `model_review_50/` output contains the model-label CSV, a review and
comparison workbook, label provenance, and evaluation JSON/Markdown. The labels
are tagged `reviewer_type=model`, `human_verified=false`; the evaluator preserves
these tags and reports `model_review_pilot`. The workbook can be read back by the
same evaluator. Use `--unweighted` for this descriptive subset comparison rather
than presenting it as accuracy for the million-pair population.

| Field | Same | Different | Uncertain |
| --- | ---: | ---: | ---: |
| Company | 24 | 25 | 1 |
| Street name | 23 | 24 | 3 |
| City | 23 | 26 | 1 |

In grouped five-fold evaluation with a 95% training precision target, the
held-out **agreement with model reference labels** was:

| Field | Metric | Precision | Recall | TP / FP / FN / TN |
| --- | --- | ---: | ---: | --- |
| Company | Jaro–Winkler | 100.0% | 95.8% | 23 / 0 / 1 / 25 |
| Company | Levenshtein | 100.0% | 83.3% | 20 / 0 / 4 / 25 |
| Street name | Jaro–Winkler | 95.5% | 91.3% | 21 / 1 / 2 / 23 |
| Street name | Levenshtein | 100.0% | 82.6% | 19 / 0 / 4 / 24 |
| City | Both metrics | 100.0% | 95.7% | 22 / 0 / 1 / 26 |

All 145 definite judgments received one held-out prediction per metric. The
five uncertain judgments were excluded. A 100% training precision sensitivity
check preserved the street/city results; company Jaro–Winkler recall fell to
91.7% with no false positives, still above Levenshtein's 83.3%.

This small batch supports retaining Jaro–Winkler for company-name comparisons
while investigating the street precision/recall tradeoff; city operating-point
results tie. It does not establish an optimal metric or calibrated accuracy.
No production comparisons, thresholds, guards, decisions, embeddings, or indexes
were changed. Cases worth a small human spot-check are the five uncertain
judgments: P011 street, P012 city, P017 company, P031 street, and P048 street.

The review also exposes parser limitations: some `street_name` keys still retain
building, suite, and floor tokens. The model judged the road-name component while
both existing metrics scored the full saved key. For example, P035 differs in
unit notation despite agreement on the road name. Cleaning that comparison
input is a separate improvement to evaluate before switching metrics.

### After street-component cleanup

String-scoring run `956878669358056` uses `street_components_v3`. The original
reference judgments above were fixed before cleanup. The verification script
confirmed pair identities and content hashes, projected those judgments onto the
cleaned comparison view, and reran the same grouped evaluation. Original model
labels, source data, and the initial workbook remain intact.

Company and city results are unchanged. On the 47 definite street judgments,
both metrics have **87.0% recall**; Jaro–Winkler has **100.0% precision** (20 TP,
0 FP, 3 FN, 24 TN) and Levenshtein has **87.0% precision** (20 TP, 3 FP, 3 FN,
21 TN). Those are held-out agreements with model labels, not validated production
accuracy. Cleaner comparison inputs need not monotonically improve every pilot
metric: removing house/unit numbers removes evidence that belongs in separate
guards, and the learned string thresholds change.

This supports Jaro–Winkler as the current primary comparator while retaining
Levenshtein for diagnostics and further labeled comparisons. It does not justify
counting the two as independent evidence for a single field in Fellegi–Sunter.
The updated report and source/label provenance are in the private
`component_cleanup/verified_956878669358056/evaluation/` output. See
[STRING_FEATURES.md](STRING_FEATURES.md) for full-snapshot cleanup and validation.

## Reproduce the raw sample

From the repository root, select an authorized Databricks profile explicitly:

```bash
.venv/bin/python experiments/matching/scripts/create_labeling_sample.py \
  --profile <selected-profile> --run-id 95648575728701 --size 300 \
  --output data/labeling/a-new-empty-directory
```

The script refuses an existing output directory. It exports score-blind
`field_labels.csv` and `pair_labels.csv`, separate `analysis_only_scores.csv`,
`manifest.json`, the exact `sampling.sql`, and private `sample.json` for workbook
authoring. Keep source data and human labels out of Git.

Validation covers stratum allocation, population weights, score-blind exports,
side mapping, original-language context, and validation groups. Live checks
confirm the completed source run, population count, quotas, and unique pairs.
The workbook was rendered on all three sheets and verified after export for
source alignment, empty label cells, dropdowns, frozen headers, and formula
caches. No human judgments have been prefilled.
