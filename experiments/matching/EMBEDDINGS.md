# Embedding candidate expansion

`embedding_candidates` is a separate, manually triggered, two-task serverless
STANDARD job named **Address Embedding Candidates**. It reads a completed
exact/Fellegi–Sunter experiment and writes only to the same developer experiment
schema. It never approves links or changes silver/gold data.

See [the first completed run and review queries](EMBEDDING_RESULTS.md) for measured
candidate counts, vector-search recall, and interpretation limits.

The separate [string-feature job](STRING_FEATURES.md) adds Levenshtein alongside
Jaro–Winkler on these saved pairs, without recomputing vectors or HNSW results.

## Run

Select a workspace profile explicitly, then run from this directory:

```bash
databricks bundle validate --strict -t dev --profile <selected-profile>
databricks bundle deploy -t dev --profile <selected-profile>
databricks bundle run embedding_candidates -t dev --profile <selected-profile>
```

The default `source_run_id=latest` resolves the most recently completed original
matching experiment **once**, in the embedding task. The search task uses that
exact snapshot, even if another matching run finishes meanwhile. For a specific
comparison, pass its MLflow experiment run ID:

```bash
databricks bundle run embedding_candidates -t dev --profile <selected-profile> \
  --params source_run_id=ea2277b0d4444cee8d744ba5277b523e
```

The job run ID is the `embedding_run_id` shared by both tasks. Start a fresh job
run after failure. Existing output rows are checked before writing, and rerunning
the embedding stage with an existing stage marker is rejected, to prevent
duplicate append-only outputs. Both tasks explicitly disable serverless
auto-optimization retries as well as setting `max_retries=0`: serverless
auto-optimization otherwise adds retries beyond the normal task retry policy.
See the [official retry guidance](https://docs.databricks.com/aws/en/jobs/run-serverless-jobs).
Only `embedding_runs.status='COMPLETED'` identifies fully written search results.
Failed runs can retain cache entries, embeddings, and partial candidate outputs.

## Pipeline

1. **`embed_snapshot`** reads every accepted `prepared_records` row from the
   selected completed run. It uses that run's original input snapshot, rejection
   counts, stable source IDs, content hashes, and input fingerprint. Rejected
   source rows remain quarantined in the original `input_snapshots`; this job
   does not silently reinstate them or read a newer silver snapshot.
2. It creates two representations per row: normalized **company + full address**
   and **address alone**. Address-only retrieval helps when company-name spelling
   differs; full-record retrieval supplies a complementary view. Unicode,
   building numbers, and suite tokens are retained. Original source values remain
   available with each review pair. This reuses existing normalization without
   introducing a country, region, or prefix blocking constraint.
3. A revision-pinned multilingual Sentence Transformer computes 384-dimensional,
   normalized CPU embeddings. Cache keys include the exact text hash, model
   revision, and representation version. Repeated identical texts are encoded
   once, while every source row retains its own vector mapping. Model weights
   are downloaded from Hugging Face; input text is encoded on job compute rather
   than sent to an external inference endpoint. Inputs beyond the model token
   limit fail explicitly rather than losing trailing address/unit information.
4. **`search_hnsw`** builds two FAISS HNSW indexes containing all accepted rows,
   with `M=32`, `efConstruction=200`, and `efSearch>=200`. Stable ID order and
   single-thread index construction support reproducibility. Both index binaries
   and their row-to-source-ID map are saved in the linked MLflow run.
5. Each row retrieves up to `search_k=100` neighbors per view (plus self).
   Self-pairs and **every previously scored pair** are removed, including old
   uncertain/conflict/low-probability pairs. From the remaining neighbors with
   cosine at least `min_cosine=0.70`, at most `top_k=20` new neighbors per query
   per view are retained. Directional results are combined into unique unordered
   pairs, retaining which views found them. Search is global and includes both
   within-provider and cross-provider pairs.
6. New pairs retain company/street/city comparisons and country, region,
   building-number, numeric-token, unit-conflict, and one-sided missing-unit
   indicators. Explicit conflicts receive `review_conflict`; other pairs receive
   `review_embedding_candidate`. All have `approved_for_merge=false`.

The model is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`. Runtime dependencies are pinned in
`resources/embedding.job.yml`. Vectors and cache entries use `model_key` to avoid
mixing incompatible spaces. Changes to the representation or model must change
the representation version/model key.

## Outputs

All output tables live in `addressresolution.matching_experiments_<owner>`.

| Table | Purpose |
| --- | --- |
| `embedding_cache` | Reusable normalized vectors by model key and input text hash |
| `embedding_records` | Every accepted record, view, text, hash, and vector, keyed by embedding/source run |
| `embedding_candidate_passes` | New directed neighbors with retrieval view, cosine, and original neighbor rank |
| `embedding_candidates` | Unique new unordered pairs, source evidence, similarities, and review guards |
| `embedding_review_sample` | Deterministic sample stratified by review outcome, view, and input route |
| `embedding_runs` | Embedding-stage and final completion manifests, configuration, and diagnostics |

The MLflow run includes index artifacts, source libraries, dependency versions,
the source snapshot fingerprint, retrieval counts, and exact-search audit metrics.
Existing `match_labels` stays independent. A reviewed pair can use the same
stable IDs and content hashes whether originally discovered by blocking or HNSW.

## Interpretation and limits

Cosine similarity is **not a match probability**. The `0.70` cutoff and top-20
budget are initial retrieval settings, not thresholds calibrated against labels.
The output is a candidate pool for review or later labeled reranking. Address-only
neighbors can include unrelated companies in the same building; full-record
neighbors can include different branches of the same company. Numeric/unit guards
remain essential because semantic encoders can underweight small digit changes.
The existing Fellegi–Sunter 0.99 scores also remain uncalibrated.

Retrieval is bounded, approximate, and not exhaustive. Overfetching reduces
crowding by known pairs, but dense neighborhoods can still consume the search
budget. The run records query shortfalls, known-neighbor hits, and saturated
searches. A seeded sample of up to 128 queries per view is also compared with
**exact search over every vector**, excluding all old pairs before choosing the
exact top-k eligible new neighbors. Tie-aware `novel_neighbor_recall_at_k`
measures loss from HNSW **and** the finite overfetch budget. It measures geometric
retrieval quality, not recall of real duplicate entities. Review labels are still
needed to measure matching precision/recall and select retrieval settings.

The job fails above 50,000 accepted input records or 2,000,000 directed
candidate-pass rows. It does not sample/truncate input records to stay within
those limits. Encoding is batched, and indexes are in job memory; this deliberately
fits the existing approximately 34k-row experiment. It is not a distributed index
service or an implementation for millions of rows. Model downloads require job
compute access to Hugging Face. The initial full CPU encoding can take several
minutes; subsequent runs reuse text embeddings. No schedule is installed.

## Local validation

From the repository root:

```bash
uv pip install --python .venv/bin/python \
  sentence-transformers==5.1.2 transformers==4.57.1 torch==2.8.0 \
  faiss-cpu==1.13.2 numpy==2.2.6
ruff check experiments/matching
.venv/bin/python -m unittest discover -s experiments/matching/tests -v
```

Tests run real FAISS HNSW and DuckDB candidate SQL, exclusion of known pairs,
unordered-pair conservation, conflict guards, index save/load, tie-aware exact
retrieval auditing, partial-recall manifest serialization, budget failures, and
truncation protection. They use synthetic
vectors and source fixtures; they do not claim live entity-matching accuracy.

Primary implementation references:
[FAISS HNSW indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
[FAISS cosine metrics](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances),
[SentenceTransformer encoding](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html),
[model card](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
