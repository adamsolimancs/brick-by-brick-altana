# Project Setup

Adam Soliman, 9/26/26

Address-resolution pipeline: the system will ingest many data sources, normalize their schemas, identify records that refer to the same company, and select a defensible canonical address with source provenance.

## Proposed file structure

```text
brick-by-brick-altana/
├── databricks.yml
├── pyproject.toml
├── README.md
├── docs/
│   └── [Architecture, Data Flow, Roadmap, etc.]
├── notebooks/
│   ├── 01_explore_feeds.py
│   ├── 02_evaluate_matching.py
│   └── 03_review_golden_records.py
├── resources/
│   └── pipeline.yml
├── src/
│   ├── ingest/
│   ├── normalize/
│   ├── matching/
│   ├── resolution/
│   └── quality/
└── tests/
    ├── integration/
    └── unit/
```

The structure is intentionally modular:

- `src/ingest/` reads provider feeds and attaches source metadata.
- `src/normalize/` maps differing schemas into a common model and standardizes comparable values.
- `src/matching/` generates candidate pairs, computes match evidence, and groups matching records.
- `src/resolution/` selects the golden address and records why each value won (not just choosing the lastest address).
- `src/quality/` contains validation rules, metrics, and monitoring checks.
- `notebooks/` contains exploration and presentation workflows rather than core business logic.
- `resources/` contains Databricks job and pipeline configuration used by the bundle.
- `tests/` separates fast unit tests from Spark and Databricks integration tests.

## Data architecture

Use a medallion-style pipeline while retaining enough intermediate evidence to explain every resolution decision.

```text
Provider feeds
    ↓
Bronze: raw source records and ingestion metadata
    ↓
Silver: unified schema and normalized address components
    ↓
Candidate generation and blocking
    ↓
Match scoring and entity clustering
    ↓
Gold: canonical entities, selected addresses, and provenance
```

### Bronze

Bronze tables preserve provider values without destructive cleaning. Each record should include a provider identifier, source record identifier, ingestion timestamp, and source file or batch identifier. This layer makes ingestion reproducible and supports later audits.

### Silver

Silver tables map provider-specific fields into a common schema and add normalized fields for comparison. Normalization can cover casing, whitespace, punctuation, common abbreviations, and country or administrative-region codes. Most of the work will be done in this layer since it poses the most challenges.

### Matching

Matching should be split into explicit stages, roughly:

1. Generate plausible candidate pairs through blocking.
2. Calculate interpretable evidence such as name similarity, address-token overlap, locality agreement, and missing-value indicators.
3. Classify or **score** candidate pairs (machine learning step). I think it could be pretty interesting to use Jev for classifying since it's cheaper/faster than LLMs.
4. Cluster matching records into resolved entities.

Blocking is part of the initial architecture because comparing every record with every other record is quadratic and will not scale to millions of rows. The first implementation can use simple deterministic blocks and add more sophisticated strategies after inspecting the data.

### Gold

Gold tables contain one resolved record per entity. In addition to the selected address, retain:

- all contributing source record identifiers;
- the winning value for each component;
- conflicting candidate values;
- match and selection scores;
- the rule or model version used; and
- human-review status for ambiguous cases.

Entity matching and address legitimacy should remain separate decisions. Two companies can share a legitimate address, and a valid registered address might not be an operational business location.

## Databricks capabilities

### Data Engineering and Delta Lake

Use Spark DataFrames for pipeline transformations and Delta Lake tables for durable bronze, silver, and gold datasets. 
- I think that starting with Spark-compatible interfaces from the beginning can avoid rewriting code later, but we should confirm with Steve.

### Lakeflow Jobs

Use Lakeflow Jobs to orchestrate ingestion, normalization, matching, resolution, quality checks, and any model-training tasks. Job definitions should be version-controlled rather than configured only through the workspace UI.

### Lakeflow Spark Declarative Pipelines

Consider a Lakeflow Spark Declarative Pipeline for managed table dependencies, incremental processing, and data-quality expectations. Start with batch processing for the supplied datasets. The instructions say we should consider batch and stream processing, but we can focus on batch first.

### Declarative Automation Bundles

Use a Declarative Automation Bundle, rooted at `databricks.yml`, to version job, pipeline, and environment configuration with the application code. This provides repeatable development and deployment targets without committing generated bundle state.

### Databricks Git folders and GitHub

Connect the GitHub repository as a Databricks Git folder for interactive development. GitHub remains the source of truth for branches, pull requests, and review. Deployed jobs should eventually reference a known branch, tag, or commit so a run can be traced back to its exact source version.

### MLflow and machine learning

Begin with deterministic matching rules and an explainable baseline. Use MLflow when experiments introduce learned match scoring, threshold selection, or model comparison. Record training data versions, features, parameters, evaluation metrics, thresholds, and artifacts so results remain reproducible.

## Python files and notebooks

Use regular Python modules for reusable ingestion, normalization, matching, resolution, and quality logic. These modules are easier to test, review, import, and execute in jobs. Use notebooks for EDA, visual evaluation, threshold tuning, and stakeholder demonstrations. Production notebooks should remain thin: they should import tested functions from `src/`, supply runtime parameters, and display results rather than contain the main implementation.
