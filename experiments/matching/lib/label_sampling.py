"""Reproducible, score-blind labeling exports from existing candidate pairs."""

import hashlib
import math
from collections import defaultdict

FIELDS = ("company", "street_name", "city")
VERSION = "field_comparator_labels_v1"
SEED = "altana-labels-20261101-v1"
GROUP_WEIGHTS = {
    ("blocked", "exact_baseline"): 20,
    ("blocked", "proposed_match"): 50,
    ("blocked", "review_conflict"): 40,
    ("blocked", "review_uncertain"): 40,
    ("blocked", "likely_nonmatch"): 20,
    ("embedding", "review_embedding_candidate"): 70,
    ("embedding", "review_conflict"): 60,
}


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def allocation(populations, total, weights=None, minimums=None):
    """Bounded largest-remainder allocation, with at least one per nonempty cell."""
    keys = sorted(populations)
    if any(populations[k] <= 0 for k in keys):
        raise ValueError("Stratum populations must be positive")
    weights = weights or {k: 1 for k in keys}
    minimums = minimums or {k: 1 for k in keys}
    if not sum(minimums.values()) <= total <= sum(populations.values()):
        raise ValueError("Sample size cannot cover every stratum or exceeds population")
    ideal = {k: total * weights[k] / sum(weights.values()) for k in keys}
    result = {
        k: min(populations[k], max(minimums[k], math.floor(ideal[k]))) for k in keys
    }
    while sum(result.values()) < total:
        available = [k for k in keys if result[k] < populations[k]]
        chosen = max(available, key=lambda k: (ideal[k] - result[k], k))
        result[chosen] += 1
    while sum(result.values()) > total:
        available = [k for k in keys if result[k] > minimums[k]]
        chosen = max(available, key=lambda k: (result[k] - ideal[k], k))
        result[chosen] -= 1
    return result


def stratum_quotas(strata, size):
    groups = defaultdict(dict)
    for row in strata:
        groups[(row["candidate_origin"], row["decision"])][row["sample_focus"]] = int(
            row["population_n"]
        )
    totals = {group: sum(cells.values()) for group, cells in groups.items()}
    group_counts = allocation(
        totals,
        size,
        {g: GROUP_WEIGHTS.get(g, 30) for g in groups},
        {g: len(cells) for g, cells in groups.items()},
    )
    result = {}
    for group, cells in groups.items():
        for focus, n in allocation(cells, group_counts[group]).items():
            result[(*group, focus)] = n
    return result


def sampling_cte(prefix, run_id, cutoffs, seed=SEED):
    """Discordance uses separate top-decile cutoffs, not a shared score threshold."""
    structures = []
    for name in FIELDS:
        jw = float(cutoffs[f"{name}_jw"])
        lev = float(cutoffs[f"{name}_lev"])
        if not (0 <= jw <= 1 and 0 <= lev <= 1):
            raise ValueError("Invalid comparator quantile")
        a, b = f"{name}_jaro_winkler_similarity", f"{name}_levenshtein_similarity"
        state = f"CASE WHEN {a} >= {jw!r} AND {b} < {lev!r} THEN '{name}_jw_only' WHEN {b} >= {lev!r} AND {a} < {jw!r} THEN '{name}_lev_only' END"
        structures.append(
            f"named_struct('priority',sha2(concat_ws('|',id_l,id_r,'{name}',{sql_string(seed)}),256),'focus',({state}))"
        )
    return f"""WITH base AS (
      SELECT * FROM {prefix}.string_pair_scores
      WHERE string_scoring_run_id={sql_string(run_id)}
    ), focused AS (
      SELECT *, CASE
        WHEN company_levenshtein_distance=0 AND street_name_levenshtein_distance=0 AND city_levenshtein_distance=0 THEN 'all_fields_exact'
        ELSE coalesce(try_element_at(array_sort(filter(array({",".join(structures)}), x -> x.focus IS NOT NULL)),1).focus,'no_top_decile_disagreement')
      END sample_focus FROM base
    )"""


def context(row, side):
    """Raw accepted values and original-script evidence, without model output."""
    lines = [
        str(row.get(f"company_{side}") or ""),
        str(row.get(f"street_address_{side}") or ""),
    ]
    lines.append(
        ", ".join(
            str(row.get(f"{key}_{side}") or "") for key in ("city", "state", "country")
        )
    )
    import json

    evidence = json.loads(row.get(f"translation_evidence_json_{side}") or "{}")
    originals = []
    for field in ("company", "street_address", "city", "state", "country"):
        value = evidence.get(f"{field}_original")
        if value and value != row.get(f"{field}_{side}"):
            originals.append(f"{field}: {value}")
    if originals:
        lines.append("Original text: " + "; ".join(originals))
    return "\n".join(lines)


def make_exports(rows, strata, quotas, source_run_id, seed=SEED):
    """Separate annotator inputs from scores/strata; no prefilled human labels."""
    populations = {
        (r["candidate_origin"], r["decision"], r["sample_focus"]): int(
            r["population_n"]
        )
        for r in strata
    }
    ordered = sorted(
        rows,
        key=lambda r: hashlib.sha256(
            f"{seed}|display|{r['id_l']}|{r['id_r']}".encode()
        ).hexdigest(),
    )
    pair_rows, field_rows, analysis = [], [], []
    for i, row in enumerate(ordered, 1):
        pair_id = hashlib.sha256(
            "|".join(
                str(row[k])
                for k in ("id_l", "id_r", "content_hash_l", "content_hash_r")
            ).encode()
        ).hexdigest()
        ref = f"P{i:03d}"
        flip = hashlib.sha256(f"{seed}|sides|{pair_id}".encode()).digest()[0] % 2 == 1
        a, b = ("r", "l") if flip else ("l", "r")
        pair_rows.append(
            {
                "pair_ref": ref,
                "pair_id": pair_id,
                "record_a": context(row, a),
                "record_b": context(row, b),
                "overall_label": "",
                "reviewer": "",
                "notes": "",
            }
        )
        for field in FIELDS:
            field_rows.append(
                {
                    "pair_ref": ref,
                    "pair_id": pair_id,
                    "field": field,
                    "value_a": row.get(f"comparison_{field}_{a}") or "",
                    "value_b": row.get(f"comparison_{field}_{b}") or "",
                    "label": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
        stratum = (row["candidate_origin"], row["decision"], row["sample_focus"])
        item = dict(row)
        item.update(
            {
                "pair_ref": ref,
                "pair_id": pair_id,
                "source_scoring_run_id": source_run_id,
                "display_a_unique_id": row[f"id_{a}"],
                "display_b_unique_id": row[f"id_{b}"],
                "stratum_population_n": populations[stratum],
                "stratum_sample_n": quotas[stratum],
                "inclusion_probability": quotas[stratum] / populations[stratum],
                "sampling_weight": populations[stratum] / quotas[stratum],
            }
        )
        analysis.append(item)
    # Connected components keep repeated records and identical company keys
    # together when constructing later validation folds. These are conservative
    # grouping hints, not inferred company/entity matches.
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for row in analysis:
        union("id:" + row["id_l"], "id:" + row["id_r"])
        for side in ("l", "r"):
            name = row.get(f"comparison_company_{side}")
            if name:
                union("id:" + row[f"id_{side}"], "company:" + name)
    for row in analysis:
        row["evaluation_group"] = hashlib.sha256(
            find("id:" + row["id_l"]).encode()
        ).hexdigest()[:16]
    return {"pairs": pair_rows, "fields": field_rows, "analysis": analysis}
