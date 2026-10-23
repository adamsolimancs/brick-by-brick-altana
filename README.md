# Brick-by-Brick: Address Resolution

A Tech@NYU, Databricks, and Altana project for building a scalable address-resolution pipeline that unifies provider feeds, identifies duplicate entities, and produces defensible canonical addresses with provenance.

> **Note:** This is a test/scratch repo for experimenting with the project. It is **not** in sync with the live Databricks project (the team Git folder and Unity Catalog tables). Code here may lag behind, diverge from, or never reach the live workspace — treat the Databricks workspace as the authoritative version of the pipeline.

## Members

- **Students:** Dharshan Vetrivelan, JJ Lee, Adam Soliman
- **Industry Advisor:** Steven McDonald

## Repo layout

- `lib/` — shared Python modules imported by notebooks. `lib/clean_functions.py` holds provider-agnostic cleaning functions; add anything reusable across feeds there rather than inside a provider notebook.
- `cleaning/` — Databricks notebooks, one per provider (bronze → silver). Import shared code with `from lib.clean_functions import ...`.
- `docs/` — the use-case PDF (source of truth) and `SETUP.md` (our design proposal).
