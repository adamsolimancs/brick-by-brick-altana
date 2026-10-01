# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Brick-by-Brick Address Resolution (Tech@NYU × Databricks × Altana): a pipeline that unifies multiple provider feeds of company addresses, identifies records referring to the same company, and selects a defensible canonical address with source provenance. The problem statement is `docs/Brick-by-Brick Use Case — Address Resolution [Student Copy].pdf`; the initial design plan is in `docs/SETUP.md`.

**The PDF is the source of truth** — it is the official project instructions. `docs/SETUP.md` is our own working plan; where it adds requirements the PDF doesn't state, treat them as proposals, not constraints, and defer to the PDF on any conflict.

## Current state vs. planned layout

The repo is early. There is no `pyproject.toml`, `databricks.yml`, test suite, or `src/` yet — only:

- `data/provider_a_feed.csv`, `data/provider_b_feed.csv` — raw sample feeds (~18.5k rows each).
- `cleaning/clean_provider_b.py` — a Databricks notebook (bronze → silver for provider B).

`docs/SETUP.md` defines the target structure (`src/{ingest,normalize,matching,resolution,quality}`, thin `notebooks/`, `resources/pipeline.yml`, `tests/{unit,integration}`, a Databricks bundle rooted at `databricks.yml`). When adding new code, move toward that layout: reusable logic belongs in importable, testable modules under `src/`; notebooks should only import from `src/`, pass parameters, and display results.

## Running code

Code is written for Databricks (PySpark), not local execution. `cleaning/*.py` files use the Databricks notebook source format (`# Databricks notebook source` header, cells separated by `# COMMAND ----------`) and rely on notebook globals `spark` and `display`. Keep that format when editing them.

Linting uses ruff (`ruff check .`, `ruff format .`); there is no ruff config yet, so defaults apply.

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

Both have `street_address`, which can contain embedded newlines, apt/suite fragments, mixed casing, and non-Latin scripts. The provider B cleaner renames to `company`, `city`, `region`; provider A's `state` has not yet been aligned to `region`. Whitespace trimming uses a regex rather than `F.trim` because `F.trim` misses tabs, newlines, and NBSPs.
