# Company and address matching: hypotheses and research

Research date: October 28, 2026. This is a design proposal, not an implemented or benchmarked matcher.

## Recommendation

Start with exact matching, multiple blocking passes, explicit component comparisons, and an interpretable probabilistic scorer. Evaluate Jev and a supervised tabular classifier as challengers. Add embedding retrieval if it finds true matches that lexical blocking misses. Use an LLM or human for ambiguous evidence, then enforce consistency when forming clusters.

The target for the first matcher should follow Adam's stated intent: duplicate observations of the same company at the same location. Company identity, location identity, and address quality are separate questions. Do not make a strong company-name match compensate for an incompatible location.

## What was verified

Read-only CLI inspection used the explicitly selected `aes10130@nyu.edu` profile at `https://dbc-031efa75-388b.cloud.databricks.com`.

- `Silver Entity Resolution Pipeline` (`f6e1e8a3-2396-4c3e-bbac-ac2c4a3b2c58`) is a serverless, triggered Lakeflow pipeline. Its latest recorded update started October 24, 2026, at 21:30:08 UTC and completed.
- Exported `provider_cleaning.py` and `translation.py` were byte-identical to the local files when inspected.
- Live `addressresolution.silver.cleaned_1_0` exists. Its schema contains company, street_address, city, state, country, source identity, dates, and quality flags. It has no separate postal-code, coordinate, house-number, or unit columns. Some components may be embedded in street_address.
- The live translation source publishes accepted rows to `addressresolution.silver.providers_translated`. Its richer originals, requests, and validation evidence remain in `providers_translation_results` and related stages.
- Local `gold.py` / `lib/gold.py` implement a provisional exact company/address baseline. A live lookup of `addressresolution.gold.entities_1_0` returned that the table does not exist. This does not establish whether other matching work exists elsewhere.
- The local CSVs contain 17,557 provider A records and 17,628 provider B records. These are local file counts, not live table counts. No live data queries, model calls, training, pipeline runs, or deployments were performed.

The official project brief in this repo supplies synthetic data and excludes real addresses and actual geospatial data. External geocoding or real-world enrichment should therefore be outside this experiment. Its one-best-address-per-company presentation can be supported later with a company-level rollup while preserving separate company/location records underneath. Adam's requested location grain takes precedence for this proposal.

## 1. Matching hypotheses to test

### Separate the questions

Retain company evidence, location evidence, and a final probability for the specific duplicate target. Do not multiply independently estimated company and location probabilities: the evidence is related.

| Company evidence | Location evidence | Interpretation |
| --- | --- | --- |
| Strong | Strong | Candidate duplicate company/location observation |
| Strong | Conflicting | Possible branch, relocation, source error, or same-name company; preserve separately |
| Conflicting | Strong | Possible co-tenants; sharing an address alone does not justify a merge |
| Incomplete | Incomplete | Insufficient evidence; remain unresolved |

Company names alone cannot reliably establish franchise ownership, legal identity, or parent/subsidiary relationships. A company-level graph needs additional evidence and its own policy. If historical moves matter, maintain validity periods and company/location relationships instead of collapsing different addresses into one.

### Parse before fuzzy scoring

Add comparison fields for house/building number, street name, direction, unit, locality, administrative region, and country where the address format supports them. Preserve number suffixes, ranges, leading zeros, and original text. A parser failure is missing evidence, not disagreement. Avoid imposing a US street schema on every country.

Keep several name views: original, normalized full name, tokens, and optionally a legal-suffix-reduced comparison view. Removing LLC or Ltd must not erase potentially useful distinctions from the source. Expand abbreviations using country/context rules; `St` is not unconditionally `Street`.

Keep native-script originals alongside transliterated/translated views. Translation agreement is evidence, not proof that the original names or places are identical. Version transformations so false merges can be traced to the representation that created them.

### Generate candidates through several overlapping passes

Take the union of rules, then deduplicate unordered pairs using composite source identifiers:

1. Country + normalized region + normalized locality + house number, where present.
2. Country + normalized region + uncommon company token or name key.
3. Country + strongly matching name + street signature, allowing a missing or inconsistent region.
4. A rescue pass using character n-gram similarity, and later possibly multilingual embedding neighbors.

These are experimental rules, not a finished specification. State-only blocks are large and miss matches when state spelling/coding differs. Normalize `NY` and `New York` before blocking; scope region codes by country. An inferred country should not become unquestionable exclusion evidence. Provide a measured fallback for uncertain country/region values.

Monitor oversized brand and common-address blocks; refine them rather than silently dropping pairs. Record every rule that generated a pair. Candidate rules should cover both within-provider duplicates and cross-provider duplicates. Do not assume one-to-one linkage across feeds.

### Score agreements, conflicts, and missing values separately

| Feature group | Candidate evidence | Treatment |
| --- | --- | --- |
| Company | Exact full name, edit distance, token overlap, character n-grams, alias agreement | Require discriminating identity evidence; common brand tokens are weaker |
| Street | Parsed road exactness, token overlap, edit similarity, direction | Compare street separately from building/unit numbers |
| Building | Exact number/suffix, range compatibility, explicit conflict | A reliable conflict prevents automatic location merging |
| Unit | Exact, missing on one side, explicit conflict | Missing is unknown; two confidently different suites stay separate or receive review |
| Locality/region | Normalized agreement, alias/translation evidence | Correlated context; avoid counting derived evidence repeatedly |
| Country | Direct vs inferred agreement, conflict, uncertainty | Trusted conflict can exclude location matching; uncertain inference needs review/fallback |
| Provenance | Parser status, translation status, provider pair, raw/derived fields | Distinguish source evidence from transformations of that same evidence |

Illustrative example: `Acme LLC, 123 Main St, Suite 4` and `ACME, 123 Main Street #4` are plausible duplicates. Replacing `123` with `128` creates a material location conflict despite very high whole-string similarity. A number typo remains possible, but should need stronger evidence or review.

For exploration, an explicit weighted feature score is useful. Do not describe `0.9` on such a score as 90% correctness. Prefer learning evidence weights and validating probability calibration before enabling automatic fuzzy merges.

### Use two thresholds and an abstention state

For a validated duplicate probability `p`:

- `p >= T_merge`, sufficient observed evidence, and no structural conflict: automatically link.
- `p <= T_reject` with strong contrary evidence: reject this pair.
- Otherwise: review, optional specialist classifier, or LLM adjudication.

Choose thresholds from held-out labels and the cost of false merges. With sparse input, a low score can reflect insufficient information rather than a demonstrated non-match; retain that distinction. Also review model disagreement and cluster conflicts, even when one model reports high confidence. No numeric threshold in this proposal is an accuracy claim.

### Cluster with constraints

Pairwise scores do not guarantee valid clusters. A-B and B-C links can join A and C even when A and C have incompatible house numbers or suites.

Start with conservative exact anchors; attach candidates only with adequate identity/location evidence and no conflict with the cluster's known components. Send bridge edges, incompatible attachments, and unusually large clusters to review. An anchor check alone is insufficient if it ignores conflicts with other members. Treat unscored pairs as unknown. For ambiguous small groups, score the extra pairs needed to check consistency. Keep evidence edges and membership versions so decisions can be revised.

### Select a canonical address after matching

Rank whole observed address records using completeness, independent provider corroboration, transformation/validation quality, and applicable recency. Multiple IDs from one feed are not independent providers; copied feeds can also exaggerate support. Do not compare provider A's update date with provider B's creation date as equivalent freshness measurements.

Keep match probability separate from address-selection quality. Do not fabricate an address by taking individually popular components from different locations. Preserve losing candidates and component/source provenance. A single current best address can be a downstream view, with other valid company locations retained.

## 2. Established and modern techniques

### Probabilistic record linkage: Fellegi-Sunter / Splink

Fellegi-Sunter accumulates evidence using how likely each comparison outcome is among matches versus non-matches:

`match_weight = log2(prior_odds) + sum(log2(m_i / u_i))`

For an observed comparison level, `m_i` is its likelihood among true matches and `u_i` among non-matches. Splink implements this framework, model estimation, blocking, and diagnostics. Its term-frequency adjustments reduce the evidential value of common values. This makes it a useful first statistical baseline for the project's explainability requirement. [Model](https://moj-analytical-services.github.io/splink/topic_guides/theory/fellegi_sunter.html), [frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html).

Limitations: independence assumptions, correlated city/region/country/full-address features, and candidate-selection effects can make probabilities overconfident. Validate on the actual candidate population. A pair blocked on state does not acquire fresh independent evidence merely because state agrees.

**Integration:** estimate and inspect a model in a separate Databricks experiment/job. Splink supports Spark and DuckDB backends, but its published Spark example configures SparkContext and a similarity JAR. Lakeflow pipelines do not support JVM libraries, so do not assume that example is a drop-in dependency for this serverless pipeline. Test compatibility separately, use Spark-native comparison expressions where possible, and run backend-specific estimation outside declarative dataset definitions. A bounded DuckDB experiment on Databricks is a reasonable starting point at this sample size; it is not a claim of million-record capacity. [Splink backends](https://moj-analytical-services.github.io/splink/topic_guides/splink_fundamentals/backends/backends.html), [Spark example](https://moj-analytical-services.github.io/splink/demos/examples/spark/deduplicate_1k_synthetic.html), [pipeline dependencies](https://docs.databricks.com/aws/en/ldp/developer/external-dependencies).

### Supervised similarity classifiers and active learning

Train logistic regression, gradient-boosted trees, or another small classifier on explicit pair features. These are proposals to benchmark, not assumed winners. Human labels can focus on uncertain pairs, close non-matches, and model disagreements. Dedupe is an established library that learns matching behavior from human training examples. [Dedupe](https://github.com/dedupeio/dedupe).

**Integration:** store labels and features in Delta, train within Databricks, track experiments with MLflow, register the chosen artifact in Unity Catalog, and batch-score new candidate pairs. Separate model fitting/calibration from routine pipeline refreshes. Dedupe can serve as an experimental comparator; its Python workflow is not automatically a distributed Spark pipeline.

### Address-specific parsing: libpostal

Libpostal provides international parsing and normalization, including decomposition into labeled address components. It is useful preprocessing for component comparisons, not a complete company matcher or proof that an address exists. [Official repository](https://github.com/openvenues/libpostal).

**Integration:** evaluate parser outputs on this synthetic multilingual sample before adopting them. It includes a native C library and model assets; validate packaging/runtime support rather than assuming `pip install` is sufficient. A separate compatible parsing job can publish structured Delta features if the current serverless environment cannot host it. Keep original strings and parser/version evidence.

### Lexical retrieval, embeddings, and approximate neighbors

Character n-grams can retrieve typo variants; token-set MinHash can reduce candidate search by approximate Jaccard similarity. Spark documents MinHashLSH, but deployment still needs a runtime compatibility test. [Spark LSH](https://spark.apache.org/docs/latest/ml-features.html#locality-sensitive-hashing).

Embeddings are worth testing for multilingual names and representations with limited lexical overlap. The hypothesis is improved candidate recall. Semantic closeness alone does not establish an identical business location, especially when numbers differ. Compare name and address embeddings separately, retaining exact numerical features in the final matcher.

**Integration:** Databricks AI Search, formerly Vector Search, can index Delta data with managed or supplied embeddings and Delta Sync. Its documented ANN metric is L2; normalized vectors give the same ranking as cosine similarity. Retrieval scores remain distinct from calibrated duplicate probabilities. [AI Search](https://docs.databricks.com/aws/en/vector-search/vector-search).

Pgvector supports exact/approximate vector retrieval and cosine distance in Postgres. It is an indexing/storage choice, not a matching algorithm. Prefer testing retrieval within the existing Delta workflow first; introducing Postgres would add synchronization and another store without demonstrated matching benefit. [Pgvector](https://github.com/pgvector/pgvector).

### Transformer pair classifiers / cross-encoders

Ditto frames entity matching as classification of a pair of serialized records using a pretrained transformer. It represents the established approach of letting a model inspect both records jointly after candidate generation. [Ditto paper](https://www.vldb.org/pvldb/vol14/p50-li.pdf).

**Integration:** a later challenger trained/evaluated on project labels and batch-scored after blocking. It should earn its place by improving difficult multilingual/alias cases beyond the simpler baselines. General entity-matching benchmark results are not address-resolution accuracy guarantees.

### Jev as a specialist decision model

TypeSafe documents Jev's structured decision interface and an entity-alignment cookbook on 450 beer-product candidate pairs. That is direct evidence of a matching-oriented usage pattern, not a benchmark on company addresses. [Jev introduction](https://docs.typesafe.ai/introduction), [entity-alignment cookbook](https://docs.typesafe.ai/cookbooks/entity_alignment).

**Integration:** serialize the two source records, available parsed components, and provenance. Ask separate narrow questions about company compatibility, location compatibility, and explicit number/unit conflicts; optionally use a joint outcome with `same_company_same_location`, `possible_same_company_different_location`, `different_company`, and `insufficient_evidence`. Store the complete distribution and exact model/question version. Benchmark as a challenger or adjudication stage with controlled calls from a job; it is not assumed to be a built-in Databricks model service.

Jev's `confidence` is a function of its answer distribution, not a measured address-matching accuracy. Test calibration on held-out project labels before automatic merging. [Confidence semantics](https://docs.typesafe.ai/confidence).

### LLM adjudication and the 2025-2026 research direction

COLING 2025's ComEM work studies matching, comparison, and candidate selection together, motivated by inconsistencies in independent binary decisions. For this project, a useful experiment is presenting a small ambiguous candidate group and allowing “none / insufficient evidence,” while enforcing location constraints outside the model. [ComEM](https://aclanthology.org/2025.coling-main.8/).

CE-RAG4EM, revised October 15, 2026, studies blocking-aware batching of retrieval and generation. The transferable idea is sharing relevant context within candidate blocks. Its results do not establish performance on these synthetic addresses; external factual enrichment would also conflict with the exercise's data constraint. [CE-RAG4EM](https://arxiv.org/abs/2602.05708).

Research also explores transferring LLM explanations into smaller matchers. Treat that as a later option: model-generated labels/explanations need human auditing and must stay out of the independent test truth. [Explanation-based matching](https://arxiv.org/abs/2406.09330).

**Integration:** test Databricks `ai_classify` for fixed outcomes. Current documentation describes version 2.1 confidence scores and rationales; its response shape differs from 2.0. If richer field evidence or a pinned model is needed, evaluate structured `ai_query` output instead. Verify the selected runtime and response contract first. Model certainty does not replace task-level calibration. [ai_classify](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_classify), [ai_query](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_query).

## 3. Proposed Databricks workflow

```text
Existing bronze -> existing silver cleaning
                         |-> cleaned_1_0 ----------------------+
                         |-> translation -> providers_translated|
                                                              v
                         matching input with raw/derived provenance
                                      v
                         comparison fields -> candidate pairs
                                      v
                         pair features -> baseline/challenger scores
                                      |-> durable review/adjudication
                                      v
                         constrained cluster membership
                                      v
                         canonical company/location records
                                      v
                         optional company-level best-address view
```

Proposed datasets, not existing resources:

| Dataset | Purpose |
| --- | --- |
| `addressresolution.silver.matching_input_v1` | Clean and accepted translated business fields, joined to originals/translation evidence by source identity and matching input version |
| `addressresolution.silver.match_candidates_v1` | Unique unordered source pairs, blocking reasons, snapshot version |
| `addressresolution.silver.match_features_v1` | Component comparisons, missingness/conflicts, transformation provenance |
| `addressresolution.silver.match_scores_v1` | Score, probability if validated, outcome, model/rule/threshold versions |
| `addressresolution.gold.company_locations_v1` | Canonical records and explanation/provenance |
| `addressresolution.gold.cluster_members_v1` | Source-to-cluster membership, evidence edges, membership version |
| `addressresolution.gold.match_review_v1` | Current ambiguous pairs/groups, with full evidence |
| `addressresolution.gold.match_labels_v1` | Persistent human judgments and corrections, independently managed |

Use explicit schemas when combining clean and translated rows: their outputs have different additional columns. Assert that a source observation contributes once; carry raw fields and translation evidence by joining the correct input version, not just whatever current result happens to share an ID.

Keep feature generation and deterministic scoring in a separate matching pipeline/job reading committed silver outputs. Do not train models or make arbitrary external API calls inside declarative dataset functions. AI adjudication should have its own persisted request/result stage and failure routing so ordinary matching can progress without waiting for every difficult pair.

Reuse the translation workflow's request deduplication and audit pattern, but distinguish stored materialized snapshots from a durable cache. Repeated refreshes/retries may repeat inference. For durable reuse, maintain a separate job-managed request/result table keyed by canonical pair identity, input content/version, task, model, and prompt/question version. Human labels also belong outside refreshed pipeline outputs.

Start in batch. Later, process changed records against an existing candidate index, reevaluate affected clusters, and retain merge/split history. Stable assigned entity IDs and membership history are preferable to content hashes as permanent identities.

## 4. Evaluation and first experiments

1. Build a seed label set, perhaps 500-1,000 carefully selected pairs as an initial experiment, covering true matches, close non-matches, co-tenants, branches, number/suite conflicts, translation variants, and missing fields. This is a starting workload, not sufficient evidence for a rare-error precision guarantee. Keep a separate random sample representing the operational candidate mix.
2. Measure blocking recall independently: the fraction of labeled true matches retrieved by candidate generation. Audit excluded pairs and use controlled synthetic perturbations as a supplemental robustness test. Labels sampled only from current candidates cannot reveal missed candidate pairs.
3. Compare exact-only, rules plus fuzzy features, probabilistic linkage, a small supervised classifier, and Jev on the same frozen split. Add embedding candidates and a transformer/LLM challenger separately so each improvement can be attributed.
4. Split by underlying entity/company group where labels support it, keeping related addresses and variants together; add held-out provider/country tests when data permits. Avoid random pair splits that leak one company's variants into training and test sets. Separate training, calibration/threshold selection, and final evaluation.
5. Report precision and recall at proposed automation thresholds, calibration, review coverage/load, candidate counts/runtime, and model cost per resolved record. Use uncertainty intervals for high-precision claims and representative sampling/weighting when evaluation oversamples difficult cases.
6. Evaluate final clusters for false merges/splits and final selected-address quality, not only pairwise classification. Break results down by country, provider pair, script, translation status, and component conflicts. A confidently wrong edge can damage a large cluster.

The initial experiment should answer whether multi-pass blocking plus explicit address components and a probabilistic model provides reliable gains beyond exact matching. Jev, embeddings, and LLMs should be judged by additional correct resolutions at a fixed false-merge rate and review budget.

## Verification limits

This research checked live configuration/schema and source parity, inspected the local baseline and official project brief, and consulted primary documentation and research. It did not measure match accuracy, parser quality, model calibration, package compatibility on compute, or live table row counts. No production changes were made.
