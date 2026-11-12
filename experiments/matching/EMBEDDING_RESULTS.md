# First completed embedding expansion

Completed October 30, 2026 (America/New_York), using the explicitly selected
`aes10130@nyu.edu` profile.
[Successful Databricks run](https://dbc-031efa75-388b.cloud.databricks.com/jobs/288992095267152/runs/6082159854758?o=7474651042679944).

- Job: **[dev aes10130] Address Embedding Candidates** (`288992095267152`).
- Embedding run ID: `6082159854758`.
- Source matching run: `ea2277b0d4444cee8d744ba5277b523e`.
- MLflow run: `190260af659e459eb151f0db8f7d633a`, in the
  [shared experiment](https://dbc-031efa75-388b.cloud.databricks.com/ml/experiments/3205418160044522?w=7474651042679944).
- Output schema: `addressresolution.matching_experiments_aes10130`.
- Source fingerprint: `e6b54ab5a7626a4e9afbe94f9c80a43b7d41785cf40952c86d6767f94998edf5`.

## Same-input expansion

The job used the same 33,677 accepted records as the
[original matching comparison](RESULTS.md), with no rejected input rows.
It indexed every accepted record in both full-record and address-only spaces.
It excluded all 37,139 previously scored pairs, including old uncertain,
conflict, and low-probability pairs.

| Measure | Observed result |
| --- | ---: |
| Accepted source records | 33,677 |
| Persisted record/view vectors | 67,354 |
| Distinct embedded texts | 45,673 |
| Previously scored pairs excluded | 37,139 |
| New directed full-record neighbor hits | 672,318 |
| New directed address-only neighbor hits | 670,819 |
| Unique new unordered candidate pairs | 984,646 |
| New pairs with explicit comparison conflicts | 974,240 |
| New pairs without those explicit conflicts | 10,406 |
| Persisted stratified review-sample pairs | 546 |
| Approved merges | 0 |

The initial retrieval settings were deliberately broad: up to 100 neighbors
searched per record per view, retaining up to 20 new neighbors above cosine
0.70 after old-pair exclusion. Most semantic neighbors disagreed on country,
region, building number, numeric tokens, or suite. These remain flagged
`review_conflict`, rather than being treated as additional matches.

The 10,406 remaining pairs are `review_embedding_candidate`, not verified
duplicates. Absence of an explicit conflict does not establish company or
location identity, and one-sided missing units remain visible as an additional
flag. Only **3** of these pairs also have name, street, and city Jaro–Winkler
similarities all at least 0.85 with no one-sided missing unit. That is a useful
inspection filter, not an accuracy measurement. The larger pool requires human
review and potentially labeled reranking before making match decisions.

## Approximate-search quality

Both FAISS HNSW indexes use `M=32`, `efConstruction=200`, and `efSearch=200`.
The pinned multilingual encoder produces 384-dimensional normalized vectors.
Model identity, revision, representation version, dependencies, and code hash
are retained in the run manifest. Both index binaries and their stable source-ID
map are stored under the MLflow run's `hnsw_indexes` artifact directory.

A seeded sample of 128 queries per view was compared with exhaustive exact
search over every vector. Old pairs were excluded before selecting the exact
top-20 eligible new neighbors; ties use tolerance `1e-5`.

| View | Exact new neighbors retrieved | Sampled vector recall@20 |
| --- | ---: | ---: |
| Company + address | 2,559 / 2,560 | 99.9609% |
| Address only | 2,540 / 2,560 | 99.2188% |

These percentages measure agreement with exact **vector search**, not matching
precision, accuracy, or recall of real duplicate entities. A semantic model can
retrieve a wrong company or different branch very accurately. Neither these
percentages, cosine scores, nor the original uncalibrated Fellegi–Sunter 0.99
scores substitute for independently reviewed labels.

There were 118 full-record queries and 144 address-only queries with fewer than
20 retained new neighbors. Finite search budgets, old-pair exclusion, and the
cosine cutoff can limit coverage. The audit measures their effect on the sampled
queries; it does not certify exhaustive coverage of every record.

## Saved-output verification

Read-only checks against the successful run's Delta outputs verified:

- Exactly 33,677 unique source records per view, with valid 384-dimensional,
  finite, normalized vectors.
- Exactly 984,646 unique unordered new pairs.
- Zero duplicate pairs, invalid pair orientations, or previously scored pairs.
- Zero null/NaN/out-of-range retrieval scores.
- Zero unflagged explicit conflicts and zero approved merges.
- One final `embedding_runs.status='COMPLETED'` marker and 546 review-sample pairs.
- The local runtime code hash matches the successful Databricks run.

Local validation passed all 12 tests, including real FAISS retrieval, real
candidate SQL, original matching regressions, and partial-recall JSON manifest
serialization. Ruff checks and strict workspace bundle validation passed. A real
encoder smoke test used synthetic text; the complete data workflow ran in
Databricks.

The first attempt encoded and cached all texts but failed when serializing a
NumPy audit count into the final completion manifest. Its outputs remain
uncompleted and must not be mixed into these results. The count conversion was
fixed, a regression test was added, and both tasks now disable serverless
auto-optimization retries and reject existing partial outputs before appending.
The successful rerun encoded **0 new texts**; its cached embedding stage took
49 seconds of execution, followed by 440 seconds for search, audit, and output
writes. Including STANDARD provisioning, the successful rerun took about
11 minutes 38 seconds. The initial uncached embedding task required about
21 minutes of execution.

## Start reviewing

Always filter by the completed run ID; the tables also retain prior/failed runs.
Inspect the three pairs passing the additional string-similarity heuristic first:

```sql
SELECT id_l, id_r,
       company_l, company_r, street_address_l, street_address_r,
       city_l, city_r, country_l, country_r,
       record_cosine, address_cosine,
       name_similarity, street_similarity, city_similarity
FROM addressresolution.matching_experiments_aes10130.embedding_candidates
WHERE embedding_run_id = '6082159854758'
  AND decision = 'review_embedding_candidate'
  AND NOT unit_missing_one_side
  AND name_similarity >= 0.85
  AND street_similarity >= 0.85
  AND city_similarity >= 0.85
ORDER BY record_cosine DESC NULLS LAST, id_l, id_r;
```

For broader inspection, use `embedding_review_sample` filtered to the same run,
including both review outcomes. Persist reviewed judgments in the independent
`match_labels` table using the stable IDs and both content hashes. Reviewing only
the strongest few examples cannot estimate overall precision or missed-match
recall.

See [EMBEDDINGS.md](EMBEDDINGS.md) for deployment and design details and
[embedding_checks.sql](embedding_checks.sql) for repeatable saved-output checks.
