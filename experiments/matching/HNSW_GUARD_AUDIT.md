# HNSW rejection audit — November 1, 2026

This is a read-only audit of unified scoring run `525733446208790`. The existing
Databricks job and saved decisions have not been changed. A narrow parser patch
is prepared and tested locally as `street_components_v4`.

## Why only one passed

| Stage | HNSW-only pairs remaining |
| --- | ---: |
| Saved candidates, excluding every original blocked/scored pair | 984,646 |
| Fellegi–Sunter probability at least 0.99 | 37 |
| Above 0.99 and all three field JW similarities at least 0.85 | 35 |
| Pass all geographic/address/review safeguards | 1 |

All 34 score-qualified pairs held for review have a country or region conflict.
Of the 35 score-qualified pairs, 24 have a country conflict and 24 have a region
conflict, with 14 in both groups. Three also have house-number conflicts. None
has a parsed building, floor, or suite conflict. Disabling building/suite checks
alone would not add a proposal under the existing scores and other rules.
The two high-probability pairs that fail the field threshold have company JW
about 0.552 and different explicit suite numbers; the safeguards are useful.

## Findings from source-level inspection

1. **Inferred countries are treated too confidently.** All 24 score-qualified
   country-conflict pairs have exactly one `country_inferred=true` record in the
   saved source snapshot. Their strong text agreement makes them worth reviewing;
   it does not establish which country is correct. These are the kinds of pairs
   country blocking misses and HNSW can recover. Preserve supplied versus inferred
   provenance and distinguish an uncertain inferred-country discrepancy from a
   conflict between two supplied countries. Do not globally overwrite countries
   using a similar address or a state abbreviation.
2. **Region comparisons are sometimes too literal for inconsistent metadata.**
   Eight of the ten same-country region-conflict pairs have full-name/abbreviation
   spellings of US states, but countries are TR or SA, where US aliases are not
   applied. Six have no additional address/review guard. One has a different house
   number and another has multiple unit markers. The other two pairs have genuinely
   different prefectures and houses. Cross-country state aliases are diagnostic
   evidence for review, not sufficient grounds to normalize away the discrepancy.
3. **A confirmed plural-label parsing bug affects 22 GB records.** A road ending
   in `Flats` was parsed as a `Flat` label plus invented unit `s`. The local patch
   requires a label boundary, while keeping compact numeric forms such as `Apt12`
   and explicit letter identifiers such as `Suite S`. The full snapshot was
   compared against saved v3 fields before the change: zero parity mismatches.
   The patch changes only those 22 records across all 33,677 records.
4. **Legacy house flags can describe the wrong component.** There are 223 HNSW
   pairs where the old house flag conflicts despite equal nonempty parsed house
   numbers, mostly Australian unit/house forms. The inspected cases still have
   different roads, units, or geography; this count is not a count of true matches.
   No HNSW numeric flag contradicted an exactly equal parsed street and all four
   component sets under the audit predicate. Old flags should eventually be
   reconciled with reliable new components, with fallback checks retained for
   unparsed text.
5. **Real structural conflicts should remain.** The high-scoring examples include
   different houses and streets despite close company/city names. Other samples
   have genuinely different suites or buildings. Shared broad semantics or a high
   uncalibrated model probability does not establish the same company/location.

## Parser patch counterfactual

The 22 affected records touch 1,617 saved HNSW pairs. With the fitted model and
prior fixed, recomputing their street comparison level and component safeguards:

- changes 92 street comparison levels;
- clears 232 false parsed-component conflict flags and 41 parsed-review flags;
- yields zero additional proposed matches, with maximum probability about 0.927762;
- leaves 1,614 pairs in conflict review and one in uncertain review;
- moves two uncertain-review pairs to likely nonmatch.

The counterfactual first reconstructs every affected pair's original street
weight, probability, parsed flags, and decision and asserts agreement with the
saved run. It changes only the street log-likelihood contribution and reparsed
component flags. This is not a model refit or a new Databricks scoring run.

## Validation and artifacts

44 matching tests passed, including real Spark scoring. Ruff passed. SQL,
source evidence, per-pair model review notes, and the parser counterfactual are
kept in the ignored local `data/labeling/outputs` audit directory. No remote
pipeline or saved decision was modified. Parser v4 is prepared locally and is
not yet deployed; current scoring results still use v3.

Prioritize country-inference confidence and region diagnostics before relaxing
building or suite safeguards. Resolve geographic uncertainty in the promising
pairs rather than globally discarding country checks. The model remains
uncalibrated and this review does not establish entity-level ground truth.
