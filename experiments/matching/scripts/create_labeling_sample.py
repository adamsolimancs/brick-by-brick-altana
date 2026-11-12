"""Draw a blinded annotation sample using only saved string-feature outputs."""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.label_sampling import (  # noqa: E402
    FIELDS,
    SEED,
    VERSION,
    make_exports,
    sampling_cte,
    sql_string,
    stratum_quotas,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument(
        "--prefix", default="addressresolution.matching_experiments_aes10130"
    )
    args = parser.parse_args()
    if not re.fullmatch(
        r"addressresolution\.matching_experiments_[A-Za-z0-9_]+", args.prefix
    ):
        raise ValueError("Invalid experiment schema")
    if (
        not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.run_id)
        or not 50 <= args.size <= 1000
    ):
        raise ValueError("Invalid run ID or sample size (50–1000)")
    if args.output.exists():
        raise ValueError(
            "Output directory already exists; never overwrite human labels"
        )

    def query(sql):
        result = subprocess.run(
            [
                "databricks",
                "experimental",
                "aitools",
                "tools",
                "query",
                sql,
                "--profile",
                args.profile,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    completed = query(
        f"SELECT manifest_json FROM {args.prefix}.string_scoring_runs WHERE string_scoring_run_id={sql_string(args.run_id)} AND status='COMPLETED'"
    )
    if len(completed) != 1:
        raise ValueError("Exactly one completed string-scoring run is required")
    source_manifest = json.loads(completed[0]["manifest_json"])
    quantile_sql = ", ".join(
        f"percentile_approx({field}_{metric},0.9,10000) AS {field}_{short}"
        for field in FIELDS
        for short, metric in (
            ("jw", "jaro_winkler_similarity"),
            ("lev", "levenshtein_similarity"),
        )
    )
    cutoffs = {
        key: float(value)
        for key, value in query(
            f"SELECT {quantile_sql} FROM {args.prefix}.string_pair_scores WHERE string_scoring_run_id={sql_string(args.run_id)}"
        )[0].items()
    }
    cte = sampling_cte(args.prefix, args.run_id, cutoffs)
    strata = query(
        cte
        + " SELECT candidate_origin,decision,sample_focus,count(*) population_n FROM focused GROUP BY candidate_origin,decision,sample_focus"
    )
    if (
        sum(int(row["population_n"]) for row in strata)
        != source_manifest["scored_pairs"]
    ):
        raise ValueError("Input population differs from its completion manifest")
    quotas = stratum_quotas(strata, args.size)
    quota_values = ",".join(
        "(" + ",".join([*(sql_string(v) for v in key), str(n)]) + ")"
        for key, n in sorted(quotas.items())
    )
    columns = [
        "id_l",
        "id_r",
        "content_hash_l",
        "content_hash_r",
        "source_run_id",
        "candidate_origin",
        "decision",
        "sample_focus",
    ]
    columns += [
        f"{field}_{side}"
        for side in ("l", "r")
        for field in (
            "provider",
            "provider_id",
            "company",
            "street_address",
            "city",
            "state",
            "country",
            "input_route",
        )
    ]
    columns += [f"comparison_{field}_{side}" for field in FIELDS for side in ("l", "r")]
    columns += [
        f"{field}_{metric}"
        for field in FIELDS
        for metric in (
            "levenshtein_distance",
            "levenshtein_similarity",
            "jaro_winkler_similarity",
        )
    ]
    columns += [
        "house_conflict",
        "unit_conflict",
        "unit_missing_one_side",
        "numeric_conflict",
        "region_conflict",
        "country_conflict",
        "match_probability",
        "record_cosine",
        "address_cosine",
        "retrieval_score",
    ]
    select = ",".join("p." + c for c in columns)
    draw_sql = (
        cte
        + f""", quotas(candidate_origin,decision,sample_focus,sample_n) AS (VALUES {quota_values}),
      ranked AS (
        SELECT f.*,row_number() OVER (PARTITION BY f.candidate_origin,f.decision,f.sample_focus
          ORDER BY sha2(concat_ws('|',f.id_l,f.id_r,{sql_string(SEED)}),256)) sample_rank
        FROM focused f
      ), chosen AS (
        SELECT r.* FROM ranked r JOIN quotas q USING(candidate_origin,decision,sample_focus)
        WHERE r.sample_rank <= q.sample_n
      )
      SELECT {select}, l.translation_evidence_json translation_evidence_json_l,
        r.translation_evidence_json translation_evidence_json_r
      FROM chosen p
      JOIN {args.prefix}.prepared_records l ON l.experiment_run_id=p.source_run_id AND l.unique_id=p.id_l
      JOIN {args.prefix}.prepared_records r ON r.experiment_run_id=p.source_run_id AND r.unique_id=p.id_r
    """
    )
    rows = query(draw_sql)
    if (
        len(rows) != args.size
        or len({(r["id_l"], r["id_r"]) for r in rows}) != args.size
    ):
        raise AssertionError(
            "Sample conservation or unique unordered-pair check failed"
        )
    observed = Counter(
        (r["candidate_origin"], r["decision"], r["sample_focus"]) for r in rows
    )
    if observed != Counter(quotas):
        raise AssertionError("Sample allocation differs from planned quotas")
    exports = make_exports(rows, strata, quotas, args.run_id)
    total_weight = sum(row["sampling_weight"] for row in exports["analysis"])
    if abs(total_weight - source_manifest["scored_pairs"]) > 1e-5:
        raise AssertionError(
            "Sampling weights do not reconstruct the candidate population"
        )
    allocation_rows = [
        dict(
            candidate_origin=k[0],
            decision=k[1],
            sample_focus=k[2],
            sample_n=n,
            population_n=next(
                int(r["population_n"])
                for r in strata
                if (r["candidate_origin"], r["decision"], r["sample_focus"]) == k
            ),
        )
        for k, n in sorted(quotas.items())
    ]
    manifest = {
        "version": VERSION,
        "sample_id": f"labels_{args.run_id}_{SEED}_{args.size}",
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_table": f"{args.prefix}.string_pair_scores",
        "source_scoring_run_id": args.run_id,
        "source_matching_run_id": source_manifest["source_run_id"],
        "source_input_fingerprint": source_manifest["source_input_fingerprint"],
        "population_pairs": source_manifest["scored_pairs"],
        "sample_pairs": args.size,
        "field_judgments": args.size * len(FIELDS),
        "top_decile_cutoffs": cutoffs,
        "allocation": allocation_rows,
        "sampling_weight_sum": total_weight,
        "group_sizes": dict(
            Counter(r["evaluation_group"] for r in exports["analysis"])
        ),
        "design": "Stratified deterministic-hash sample by origin, prior decision, and comparator ranking discordance; all nonempty strata covered",
        "limitations": [
            "Pilot for selecting comparators within the existing candidate pool; cannot measure matches missed by candidate generation",
            "Raw sample label percentages are not population estimates; use the recorded N/n weights",
            "Top-decile cutoffs are sampling controls, not match-acceptance thresholds",
            "Evaluation groups keep shared records and identical normalized company keys together; review additional related-company groupings before validation",
            "Do not inspect score metadata while labeling; do not select thresholds using held-out labels",
        ],
    }
    args.output.mkdir(parents=True)
    for name in ("pairs", "fields", "analysis"):
        filename = {
            "pairs": "pair_labels.csv",
            "fields": "field_labels.csv",
            "analysis": "analysis_only_scores.csv",
        }[name]
        with (args.output / filename).open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(exports[name][0]))
            writer.writeheader()
            writer.writerows(exports[name])
    (args.output / "sample.json").write_text(
        json.dumps(exports, ensure_ascii=False, indent=2)
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    (args.output / "sampling.sql").write_text(draw_sql)
    print(
        json.dumps(
            {
                "sample_pairs": args.size,
                "strata": len(strata),
                "groups": dict(
                    Counter(f"{r['candidate_origin']}:{r['decision']}" for r in rows)
                ),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
