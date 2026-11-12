# Model-labeled pair benchmark

A frontier model judges a stratified sample of scored pairs as an independent
second opinion on the pipeline's merge decisions. It is cheap enough to re-run
whenever the sample, the rubric, or the pipeline changes, and it is not ground
truth: the disagreement list it produces is what a person should review.

## How it works

- `scripts/label_pairs_with_llm.py` draws a seeded, stratified sample from one
  completed unified scoring run: every `(candidate_origin, fs_decision)` stratum,
  weighted toward proposed matches and the near-miss review buckets, plus an
  `unblocked / never_compared` stratum of same-country, same-city record pairs
  the pipeline never compared, which probes blocking misses.
- Each pair is one independent request to `gpt-6.1-sol` (OpenAI Responses API,
  reasoning effort `medium`) with a fixed system prompt and a strict JSON schema
  over the project's `match_labels` vocabulary: `same_company_same_location`,
  `different_location`, `different_company`, `insufficient_evidence`, plus a
  confidence and a one-sentence reason. The model sees both records' fields and
  any original-language text before translation. It never sees the pipeline's
  decision, probabilities, or similarities. A/B order is shuffled per pair.
- Every raw response is logged. Labels, the manifest (seed, quotas, population
  sizes), `report.md`, and `disagreements.csv` are written to a new local
  directory and appended to the Delta table
  `addressresolution.matching_experiments_aes10130.llm_pair_labels`
  (`reviewer = 'gpt-6.1-sol'`), which is never overwritten.
- `lib/llm_labeling.py` holds the prompt, schema, parsing, sampling SQL,
  evaluation, and report rendering; `tests/test_llm_labeling.py` covers them.

## Metrics

A pipeline merge is an `exact_baseline` or `proposed_match` pair. Precision is
the share of merged pairs the model also labels the same company at the same
location. Recall is the share of model-positive pairs the pipeline merged,
within the sampled strata. Weighted figures reweight each stratum by its
population/sample ratio; raw figures count sampled pairs.
`insufficient_evidence` labels are reported, not scored.

## Run

The API key comes from `OPENAI_API_KEY` or `--key-file` (a file holding only
the key, kept under the ignored `data/`). From the repository root:

```bash
.venv/bin/python experiments/matching/scripts/label_pairs_with_llm.py \
  --profile aes10130@nyu.edu --unified-run-id <unified_run_id> \
  --output data/labeling/llm/<name> --size 300 --dry-run   # sample only
.venv/bin/python experiments/matching/scripts/label_pairs_with_llm.py \
  --profile aes10130@nyu.edu --unified-run-id <unified_run_id> \
  --output data/labeling/llm/<name> --size 300 --key-file data/openai.key
.venv/bin/python experiments/matching/scripts/evaluate_llm_labels.py \
  --labels data/labeling/llm/<name>
```

## Limits

The model and the pipeline read the same text and can be fooled the same way,
so agreement is not proof. The never-compared stratum samples 20 of tens of
thousands of same-city pairs, so it can detect gross blocking misses, not rare
ones. The model's confidence is self-reported. Treat the report as a measured
second opinion and the disagreements as the human review queue.
