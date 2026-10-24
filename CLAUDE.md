# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Brick-by-Brick Address Resolution (Tech@NYU × Databricks × Altana): a pipeline that unifies multiple provider feeds of company addresses, identifies records referring to the same company, and selects a defensible canonical address with source provenance. The problem statement is `docs/Brick-by-Brick Use Case — Address Resolution [Student Copy].pdf`; the initial design plan is in `docs/SETUP.md`.

**The PDF is the source of truth** — it is the official project instructions. `docs/SETUP.md` is our own working plan; where it adds requirements the PDF doesn't state, treat them as proposals, not constraints, and defer to the PDF on any conflict.

## Databricks is live; this repo is a local dev copy

What's in the Databricks workspace (`addressresolution` Unity Catalog tables, pipelines, and workspace files) is the live project and the source of truth. This repo — and its GitHub remote — is only a local copy for personal development. Consequences:

- Code here may lag behind or diverge from what runs in the workspace. Don't assume a file here matches its live counterpart, and don't describe this repo's contents as the state of the live pipeline. When in doubt, read the workspace with the CLI.
- Changes made here only go live when pushed to the workspace with the Databricks CLI (see below). Committing or pushing to GitHub does not deploy anything.
- The team Git folder `/Workspace/Users/jl13852@nyu.edu/brick-by-brick-altana` is outdated; don't treat it as the live code.
- Table names, paths, and the data architecture below describe the live project's conventions, which code here is written against — not resources this repo owns.

**STRICT: never change file or system paths in source code.** Workspace paths (e.g. `/Workspace/Users/...`, `sys.path` entries), Unity Catalog table names, and file references in pipeline and notebook code are set for the Databricks workspace, which is the source of truth. They will look wrong or unresolvable from this local copy; that is expected. Do not "fix", relativize, or redirect them, even when a local run or a test fails because of them. Adapt local tooling and tests to the source layout instead.

**All data in this project is synthetic.** The provider feeds and every `addressresolution` table contain generated data, not real companies or personal information.

## Databricks CLI and agent skills

Work on the live project is done from local agents (Claude Code, Codex) driving the Databricks CLI, not from Genie Code in the workspace. Agents edit files on local disk; the CLI moves code to/from the workspace, runs pipelines/jobs/SQL, and inspects workspace state.

- **CLI**: installed via Homebrew (`brew install databricks/tap/databricks`, v1.22.1). Auth is OAuth through `databricks auth login --host https://dbc-031efa75-388b.cloud.databricks.com` (drop the `?o=...` suffix — zsh treats `?` as a glob). Check with `databricks auth profiles`.
- **Agent skills**: the skills from [databricks-solutions/ai-dev-kit](https://github.com/databricks-solutions/ai-dev-kit) now ship as official Databricks AI Tools, installed globally with `databricks aitools install --agents claude-code,codex --scope global` (the kit's own bundled skills are deprecated). This adds the `databricks` plugin with skills such as `databricks-core` (CLI/auth, load first), `databricks-pipelines`, `databricks-jobs`, `databricks-dbsql`, and `databricks-unity-catalog`. Prefer these skills over ad hoc CLI usage. Re-run the install command to update.

## Silver pipeline workflow (Databricks CLI)

`silver_pipeline/` is a local copy of the Lakeflow pipeline **Silver Entity Resolution Pipeline** (id `f6e1e8a3-2396-4c3e-bbac-ac2c4a3b2c58`), whose source lives at `/Workspace/Users/aes10130@nyu.edu/Silver Entity Resolution Pipeline` (not a Git folder). The CLI is authenticated via the default profile (`aes10130@nyu.edu`). Edit locally, then push and run:

```bash
# Pull the latest workspace version (do this first if anyone edited in the web UI)
databricks workspace export-dir "/Workspace/Users/aes10130@nyu.edu/Silver Entity Resolution Pipeline" silver_pipeline --overwrite

# Push local edits (add --watch to keep uploading on save)
databricks sync silver_pipeline "/Workspace/Users/aes10130@nyu.edu/Silver Entity Resolution Pipeline"

# Run the pipeline and check status/errors
databricks pipelines start-update f6e1e8a3-2396-4c3e-bbac-ac2c4a3b2c58
databricks pipelines get f6e1e8a3-2396-4c3e-bbac-ac2c4a3b2c58
databricks pipelines list-pipeline-events f6e1e8a3-2396-4c3e-bbac-ac2c4a3b2c58
```

A `SessionStart` hook (`.claude/settings.json`) runs `scripts/pull_workspace.sh` at the start of each local session, pulling the live workspace version into `silver_pipeline/`. Files edited locally since the last pull are skipped and reported instead of overwritten; the last-pulled copy is kept in `.databricks-pull/` (gitignored). Run the script manually to re-pull mid-session.

`sync` makes the workspace match the local folder, so it can overwrite edits made in the Databricks web editor; re-export before syncing if the workspace may have changed. Other pipelines in the workspace (Bronze File Ingestion, Data Quality) belong to teammates — don't sync or start them.

## Current state vs. planned layout

This test repo is early. There is no `pyproject.toml` or `databricks.yml` yet. What exists:

- `data/provider_a_feed.csv`, `data/provider_b_feed.csv` — raw sample feeds (~18.5k rows each).
- `lib/clean_functions.py` — shared, provider-agnostic cleaning functions (see below).
- `cleaning/clean_provider_b.py` — a Databricks notebook (bronze → silver for provider B).

`docs/SETUP.md` proposes a larger structure (`src/{ingest,normalize,matching,resolution,quality}`, `resources/pipeline.yml`, `tests/`, a bundle at `databricks.yml`). That is a proposal, not the current layout; shared code lives in `lib/` for now.

## Shared code (`lib/`)

Reusable logic goes in plain Python modules under `lib/` (not Databricks notebooks — no `# Databricks notebook source` header, or they can't be imported). Notebooks stay thin: they import from `lib/`, pick tables/columns, and display results.

- `lib/clean_functions.py` — cleaning functions shared across all provider feeds. Functions take a `DataFrame` or `Column` and return one, with no hard-coded table or column names, so they work for any of the N providers. Before writing cleaning logic inside a provider notebook, check here; if it would apply to another provider, add it here instead.
- Notebooks append the workspace folder containing `lib/` to `sys.path` before `from lib.clean_functions import ...`. Use `%load_ext autoreload` / `%autoreload 2` to pick up `lib/` edits without restarting Python.

## Running code

Code is written for Databricks (PySpark), not local execution. `cleaning/*.py` files use the Databricks notebook source format (`# Databricks notebook source` header, cells separated by `# COMMAND ----------`) and rely on notebook globals `spark` and `display`. Keep that format when editing them.

Linting uses ruff (`ruff check .`, `ruff format .`). `ruff.toml` declares the notebook globals (`spark`, `display`, `dbutils`) as builtins; otherwise defaults apply.

Tests in `tests/` run real local Spark plans with Databricks decorators, tables, and `ai_query` stubbed. They need PySpark 3.5 (the local JDK is 11, which PySpark 4 does not support) in the gitignored `.venv`:

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python "pyspark==3.5.*"
PYTHONPATH=. .venv/bin/python -m unittest discover -s tests -v
```

## Data architecture

Medallion pipeline in Unity Catalog, catalog `addressresolution`, schemas `bronze` / `silver` (/ `gold` planned), one table per provider (e.g. `addressresolution.bronze.provider_b_feed` → `addressresolution.silver.provider_b_feed`).

- **Bronze**: raw provider values, never destructively cleaned; should carry provider id, source record id, ingestion timestamp, and batch/file id.
- **Silver**: providers mapped to a common schema plus normalized comparison fields.
- **Matching**: explicit stages — blocking (candidate generation; never all-pairs), interpretable pairwise evidence, scoring/classification, clustering into entities.
- **Gold**: one record per resolved entity with contributing source ids, winning value per component, conflicting candidates, match/selection scores, rule/model version, and human-review status. Selection must be explainable — not just "latest wins".

Entity matching and address legitimacy are separate decisions: companies can legitimately share an address (office buildings), and one company may have multiple valid addresses.

## Provider feed differences

The feeds disagree in schema and conventions, which is the core of the silver-layer work:

| | Provider A | Provider B |
|---|---|---|
| id | `provider_a_id` | `provider_b_id` |
| company name | `company` | `name` |
| city | `city` | `locality` |
| region | `state` (full names, e.g. `Saskatchewan`) | `admin_region` (often codes, e.g. `CT`) |
| country | full name (`United Kingdom`) | ISO-style code (`US`, `PL`) |
| timestamp | `updated_at` | `created_at` |

Both have `street_address`, which can contain embedded newlines, apt/suite fragments, mixed casing, and non-Latin scripts. The provider B cleaner renames to `company`, `city`, `region`; provider A's `state` has not yet been aligned to `region`. Whitespace trimming (`lib.clean_functions.trim_strings`) uses a regex rather than `F.trim` because `F.trim` misses tabs, newlines, NBSPs, and ideographic spaces.

About 20% of `street_address` values in *both* feeds contain non-Latin script (CJK, Cyrillic, Hangul, Thai, Arabic, some Devanagari), roughly half of them mixed with Latin (e.g. `458 岡田 Street`). The non-Latin part is usually the street name, so don't strip or transliterate it destructively; build comparison fields that handle both feeds identically.
