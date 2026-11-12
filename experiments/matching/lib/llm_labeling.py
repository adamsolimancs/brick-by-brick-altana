"""Model-labeled pair benchmark for the deduplication pipeline.

A frontier model judges sampled record pairs as an independent second opinion.
Each pair is one request with a fixed prompt and a strict JSON schema; the
model sees only the two records (and any original-language text), never the
pipeline's decision or scores. Labels use the project's ``match_labels``
vocabulary. Model labels are not ground truth: disagreements with the
pipeline are the review set for a person.
"""

import hashlib
import json
import math
from collections import defaultdict

from lib.label_sampling import allocation, sql_string

MODEL = "gpt-6.1-sol"
PROMPT_VERSION = "llm_pair_labels_v1"
SEED = "altana-llm-labels-20261109-v1"
LABELS = (
    "same_company_same_location",
    "different_location",
    "different_company",
    "insufficient_evidence",
)
POSITIVE_LABEL = "same_company_same_location"
UNSCORED_LABEL = "insufficient_evidence"
# Pipeline decisions that the resolver's policy merges (see DEDUPLICATION.md).
MERGED_DECISIONS = ("exact_baseline", "proposed_match")
# Sampling weight per (candidate_origin, fs_decision). Merges and the near-miss
# review buckets dominate because errors there cost the most; the never-compared
# stratum probes blocking misses outside every candidate pool.
STRATUM_WEIGHTS = {
    ("blocked", "exact_baseline"): 30,
    ("blocked", "proposed_match"): 70,
    ("blocked", "review_uncertain"): 30,
    ("blocked", "review_conflict"): 30,
    ("blocked", "non_match_address_conflict"): 20,
    ("blocked", "likely_nonmatch"): 10,
    ("embedding", "proposed_match"): 10,
    ("embedding", "review_uncertain"): 25,
    ("embedding", "review_conflict"): 35,
    ("embedding", "non_match_address_conflict"): 20,
    ("embedding", "likely_nonmatch"): 10,
    ("cluster_completion", "proposed_match"): 10,
    ("unblocked", "never_compared"): 20,
}
RECORD_FIELDS = ("company", "street_address", "city", "state", "country")

SYSTEM_PROMPT = """You are auditing a company-address deduplication system.
You will see two records of company addresses from data providers. Decide whether
they describe the same company at the same physical location.

Labels:
- same_company_same_location: the same legal/business entity at the same place
  (the same building and unit where units are given). Formatting, casing,
  punctuation, abbreviations, transliteration, translation, legal-suffix
  variants (Ltd vs Co., Ltd.) and region-name spellings do not make records
  different.
- different_location: the same company, or plausibly the same company, at a
  clearly different address: a different house number, street, unit, city or
  country that is not explained by formatting or translation.
- different_company: clearly different companies, even if they share an
  address (companies legitimately share buildings).
- insufficient_evidence: the records do not contain enough information to
  decide either way.

Judge only from the text shown. Do not assume two records are the same because
they look similar; do not assume they differ because of spelling. A missing
unit on one side is unknown, not a conflict. When a country is marked as
inferred, treat it as a guess by the data pipeline rather than a supplied fact.
Respond with the JSON object only."""


def response_schema():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "confidence", "reason"],
        "properties": {
            "label": {"type": "string", "enum": list(LABELS)},
            "confidence": {
                "type": "number",
                "description": "Your confidence in the label, between 0 and 1.",
            },
            "reason": {
                "type": "string",
                "description": "One sentence naming the decisive evidence.",
            },
        },
    }


def parse_response(text):
    try:
        payload = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Response is not JSON: {text!r}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "label",
        "confidence",
        "reason",
    }:
        raise ValueError(f"Response has unexpected keys: {payload!r}")
    if payload["label"] not in LABELS:
        raise ValueError(f"Unknown label: {payload['label']!r}")
    confidence = payload["confidence"]
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0 <= confidence <= 1
        or math.isnan(confidence)
    ):
        raise ValueError(f"Confidence outside [0, 1]: {confidence!r}")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ValueError("Reason must be a nonempty string")
    return {
        "label": payload["label"],
        "confidence": float(confidence),
        "reason": payload["reason"].strip(),
    }


def _originals(record):
    try:
        evidence = json.loads(record.get("translation_evidence_json") or "{}")
    except ValueError:
        evidence = {}
    originals = {}
    for field in RECORD_FIELDS:
        status = evidence.get(f"{field}_translation_status")
        original = evidence.get(f"{field}_original")
        if status and status != "unchanged" and original:
            originals[field] = original
    return originals


def record_view(record):
    """The blinded, human-readable description of one record."""
    originals = _originals(record)
    lines = [f"provider: {record.get('provider') or ''}"]
    for field in RECORD_FIELDS:
        value = record.get(field)
        value = "" if value is None else str(value)
        if field == "country" and str(record.get("country_inferred")).lower() == "true":
            value = f"{value} (inferred by the pipeline, not supplied)"
        lines.append(f"{field}: {value}")
        if field in originals:
            lines.append(
                f"{field} (original text before translation): {originals[field]}"
            )
    return "\n".join(lines)


def swap_orientation(pair, seed=SEED):
    digest = hashlib.sha256(f"{seed}|{pair['id_l']}|{pair['id_r']}".encode()).digest()
    return digest[0] % 2 == 1


def pair_prompt(pair, seed=SEED):
    """Return (user prompt, swapped). Only record text reaches the prompt."""
    swapped = swap_orientation(pair, seed)
    first, second = (pair["r"], pair["l"]) if swapped else (pair["l"], pair["r"])
    text = (
        "Do these two records describe the same company at the same location?\n\n"
        f"Record A\n{record_view(first)}\n\n"
        f"Record B\n{record_view(second)}\n"
    )
    return text, swapped


def stratum_quotas(populations, size):
    weights = {key: STRATUM_WEIGHTS.get(key, 10) for key in populations}
    return allocation(populations, size, weights)


def _weight(row):
    population, sample = int(row["population_n"]), int(row["sample_n"])
    if sample <= 0 or population < sample:
        raise ValueError("Invalid stratum population or sample size")
    return population / sample


def _metrics(rows, weight):
    tp = fp = fn = 0.0
    for row in rows:
        merged = row["fs_decision"] in MERGED_DECISIONS
        same = row["label"] == POSITIVE_LABEL
        w = weight(row)
        if merged and same:
            tp += w
        elif merged:
            fp += w
        elif same:
            fn += w
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
    }


def evaluate(rows):
    """Agreement between pipeline merges and model labels.

    Precision: of pairs the pipeline merges, the share the model also calls the
    same company at the same location. Recall: of pairs the model calls the
    same, the share the pipeline merged, within the sampled strata. Weighted
    figures use population/sample weights per stratum; raw figures count
    sampled pairs. ``insufficient_evidence`` labels are reported, not scored.
    """
    scored = [row for row in rows if row["label"] != UNSCORED_LABEL]
    counts = defaultdict(int)
    for row in rows:
        counts[(row["candidate_origin"], row["fs_decision"], row["label"])] += 1
    disagreements = []
    for row in scored:
        merged = row["fs_decision"] in MERGED_DECISIONS
        same = row["label"] == POSITIVE_LABEL
        if merged != same:
            disagreements.append(
                dict(
                    row,
                    kind="merged_but_model_disagrees"
                    if merged
                    else "not_merged_but_model_says_same",
                )
            )
    return {
        "labeled_pairs": len(rows),
        "scored_pairs": len(scored),
        "unscored_insufficient_evidence": len(rows) - len(scored),
        "unweighted": _metrics(scored, lambda row: 1.0),
        "weighted": _metrics(scored, _weight),
        "counts": [
            {
                "candidate_origin": origin,
                "fs_decision": decision,
                "label": label,
                "pairs": n,
            }
            for (origin, decision, label), n in sorted(counts.items())
        ],
        "disagreements": disagreements,
    }


def _sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return sql_string(value)


def insert_statements(table, rows, chunk=100):
    """INSERT ... VALUES statements with escaped literals, chunked for DBSQL."""
    if not rows:
        return []
    columns = list(rows[0])
    statements = []
    for start in range(0, len(rows), chunk):
        values = ", ".join(
            "(" + ", ".join(_sql_literal(row[column]) for column in columns) + ")"
            for row in rows[start : start + chunk]
        )
        statements.append(f"INSERT INTO {table} ({', '.join(columns)}) VALUES {values}")
    return statements


def label_pair(pair, judge, retries=1):
    """Judge one pair; retry an unparseable response once, then fail loudly."""
    prompt, swapped = pair_prompt(pair)
    last_error = None
    for _ in range(retries + 1):
        raw = judge(SYSTEM_PROMPT, prompt, response_schema())
        try:
            parsed = parse_response(raw)
        except ValueError as error:
            last_error = error
            continue
        return dict(
            id_l=pair["id_l"],
            id_r=pair["id_r"],
            candidate_origin=pair.get("candidate_origin"),
            fs_decision=pair.get("fs_decision"),
            population_n=pair.get("population_n"),
            sample_n=pair.get("sample_n"),
            content_hash_l=pair["l"].get("content_hash"),
            content_hash_r=pair["r"].get("content_hash"),
            orientation_swapped=swapped,
            prompt=prompt,
            raw_response=raw,
            **parsed,
        )
    raise ValueError(f"Pair {pair['id_l']}/{pair['id_r']}: {last_error}")


def label_pairs(pairs, judge, mapper=map):
    """Label every pair; ``mapper`` may be a thread pool's map for concurrency."""
    return list(mapper(lambda pair: label_pair(pair, judge), pairs))


_IDENTIFIER = r"[A-Za-z0-9_-]{1,128}"
RECORD_COLUMNS = (
    "provider",
    "provider_id",
    "company",
    "street_address",
    "city",
    "state",
    "country",
    "country_inferred",
    "translation_evidence_json",
    "content_hash",
)


def _validate(prefix, unified_run_id, source_run_id):
    import re

    for value in (unified_run_id, source_run_id):
        if not re.fullmatch(_IDENTIFIER, value):
            raise ValueError(f"Invalid run id: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.]+", prefix):
        raise ValueError("Invalid table prefix")


def candidate_cte(prefix, unified_run_id, source_run_id):
    """Scored pairs of one unified run plus never-compared same-city pairs.

    The ``unblocked``/``never_compared`` stratum pairs records that share a
    country and city key but were never scored together, probing blocking
    misses outside every candidate pool.
    """
    _validate(prefix, unified_run_id, source_run_id)
    return f"""WITH scored AS (
  SELECT candidate_origin, fs_decision, id_l, id_r
  FROM {prefix}.unified_pair_scores WHERE unified_run_id = {sql_string(unified_run_id)}
), records AS (
  SELECT unique_id, country_key, city_key, {", ".join(RECORD_COLUMNS)}
  FROM {prefix}.prepared_records WHERE experiment_run_id = {sql_string(source_run_id)}
), unblocked AS (
  SELECT 'unblocked' AS candidate_origin, 'never_compared' AS fs_decision,
         a.unique_id AS id_l, b.unique_id AS id_r
  FROM records a
  JOIN records b
    ON a.country_key = b.country_key AND a.city_key = b.city_key AND a.unique_id < b.unique_id
  LEFT ANTI JOIN scored s ON s.id_l = a.unique_id AND s.id_r = b.unique_id
  WHERE a.country_key <> '' AND a.city_key <> ''
), candidates AS (
  SELECT * FROM scored UNION ALL SELECT * FROM unblocked
)"""


def strata_sql(prefix, unified_run_id, source_run_id):
    """Population of every (candidate_origin, fs_decision) stratum."""
    return f"""{candidate_cte(prefix, unified_run_id, source_run_id)}
SELECT candidate_origin, fs_decision, count(*) AS population_n
FROM candidates GROUP BY candidate_origin, fs_decision
ORDER BY candidate_origin, fs_decision"""


def sample_sql(prefix, unified_run_id, source_run_id, quotas, seed=SEED):
    """Seeded stratified sample; ``quotas`` maps (origin, decision) to a size."""
    if not quotas or any(int(n) <= 0 for n in quotas.values()):
        raise ValueError("Every stratum quota must be positive")
    quota_rows = ", ".join(
        f"({sql_string(origin)}, {sql_string(decision)}, {int(n)})"
        for (origin, decision), n in sorted(quotas.items())
    )

    def side(alias, suffix):
        return ", ".join(
            f"{alias}.{column} AS {column}_{suffix}" for column in RECORD_COLUMNS
        )

    return f"""{candidate_cte(prefix, unified_run_id, source_run_id)}, ranked AS (
  SELECT *,
         row_number() OVER (PARTITION BY candidate_origin, fs_decision
                            ORDER BY sha2(concat_ws('|', id_l, id_r, {sql_string(seed)}), 256)) AS rn,
         count(*) OVER (PARTITION BY candidate_origin, fs_decision) AS population_n
  FROM candidates
), quotas AS (
  SELECT * FROM VALUES {quota_rows} AS q(candidate_origin, fs_decision, sample_n)
)
SELECT r.candidate_origin, r.fs_decision, r.id_l, r.id_r, r.population_n, q.sample_n,
       {side("l", "l")}, {side("rr", "r")}
FROM ranked r
JOIN quotas q ON q.candidate_origin = r.candidate_origin AND q.fs_decision = r.fs_decision
JOIN records l ON l.unique_id = r.id_l
JOIN records rr ON rr.unique_id = r.id_r
WHERE r.rn <= q.sample_n
ORDER BY r.candidate_origin, r.fs_decision, r.rn"""


def pair_from_row(row):
    """Rebuild the pair dict that ``pair_prompt`` expects from a sample row."""
    pair = {
        key: row[key]
        for key in (
            "candidate_origin",
            "fs_decision",
            "id_l",
            "id_r",
            "population_n",
            "sample_n",
        )
    }
    for side in ("l", "r"):
        pair[side] = {column: row[f"{column}_{side}"] for column in RECORD_COLUMNS}
    return pair


def _percent(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def report_markdown(result, manifest):
    """Human-readable summary; model labels are a second opinion, not truth."""
    lines = [
        "# Model-labeled pair benchmark",
        "",
        f"Model `{manifest.get('model')}` labeled {result['labeled_pairs']} sampled pairs "
        f"(prompt `{manifest.get('prompt_version', PROMPT_VERSION)}`, effort "
        f"`{manifest.get('reasoning_effort', 'medium')}`, scoring run "
        f"`{manifest.get('unified_run_id', '')}`). Labels are the model's judgment "
        "of the text alone and are not ground truth; the disagreements below are "
        "the pairs a person should look at.",
        "",
        "## Agreement with pipeline merges",
        "",
        "A pipeline merge is an `exact_baseline` or `proposed_match` pair. Precision: "
        "of merged pairs, the share the model also calls the same company at the same "
        "location. Recall: of pairs the model calls the same, the share the pipeline "
        "merged, within the sampled strata. Weighted figures reweight each stratum "
        "by population/sample.",
        "",
        "| Measure | Unweighted | Weighted |",
        "| --- | ---: | ---: |",
        f"| Precision | {_percent(result['unweighted']['precision'])} | "
        f"{_percent(result['weighted']['precision'])} |",
        f"| Recall | {_percent(result['unweighted']['recall'])} | "
        f"{_percent(result['weighted']['recall'])} |",
        "",
        f"Scored pairs: {result['scored_pairs']}. Insufficient-evidence labels "
        f"(reported, not scored): {result['unscored_insufficient_evidence']}.",
        "",
        "## Labels by stratum",
        "",
        "| Candidate origin | Pipeline decision | Model label | Pairs |",
        "| --- | --- | --- | ---: |",
    ]
    for row in result["counts"]:
        lines.append(
            f"| {row['candidate_origin']} | {row['fs_decision']} | {row['label']} | {row['pairs']} |"
        )
    lines += [
        "",
        f"## Disagreements ({len(result['disagreements'])})",
        "",
        "| Kind | Origin | Pipeline decision | Model label | Model reason | Pair |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in result["disagreements"]:
        reason = str(row.get("reason", "")).replace("|", "\\|")
        lines.append(
            f"| {row['kind']} | {row['candidate_origin']} | {row['fs_decision']} | "
            f"{row['label']} | {reason} | {row['id_l'][:8]}/{row['id_r'][:8]} |"
        )
    return "\n".join(lines) + "\n"
