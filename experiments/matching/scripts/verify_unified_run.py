"""Read-only independent checks of completed unified scoring outputs."""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix",
        default="addressresolution.matching_experiments_aes10130",
        help="Schema holding the run (the deduplication job uses addressresolution.matching)",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.run_id):
        raise ValueError("Invalid run ID")
    if args.output.exists():
        raise ValueError("Verification output already exists")
    prefix = args.prefix

    def query(sql):
        response = subprocess.run(
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
        return json.loads(response.stdout)

    rows = query(
        f"SELECT manifest_json FROM {prefix}.unified_runs WHERE unified_run_id='{args.run_id}' AND status='COMPLETED'"
    )
    if len(rows) != 1:
        raise ValueError("Exactly one completed unified run is required")
    manifest = json.loads(rows[0]["manifest_json"])
    source_run = manifest["source_string_scoring_run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", source_run):
        raise ValueError("Invalid source run")
    local_hash = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    sources = ["lib/unified_matching.py"]
    if manifest.get("country_corrections_by_origin") is not None:
        sources.append("altana_matching_components/__init__.py")
    sources.append("notebooks/score_unified_candidates.py")
    for relative in sources:
        local_hash.update((root / relative).read_bytes())
    assert manifest["source_code_sha256"] == local_hash.hexdigest(), (
        "Deployed code differs from local source"
    )
    model_rows = query(
        f"SELECT model_json FROM {prefix}.unified_models WHERE unified_run_id='{args.run_id}'"
    )
    assert len(model_rows) == 1
    model = json.loads(model_rows[0]["model_json"])
    assert math.isclose(
        model["probability_two_random_records_match"], manifest["training"]["prior"]
    )
    for comparison in model["comparisons"]:
        for parameter in ("m_probability", "u_probability"):
            values = [
                v[parameter]
                for v in comparison["comparison_levels"]
                if not v.get("is_null_level")
            ]
            assert all(math.isfinite(v) and 0 < v < 1 for v in values)
            assert math.isclose(sum(values), 1, abs_tol=1e-10)
    immutable = [
        "content_hash_l",
        "content_hash_r",
        "candidate_origin",
        "decision",
        "match_probability",
        "house_conflict",
        "unit_conflict",
        "unit_missing_one_side",
        "numeric_conflict",
        "region_conflict",
        "country_conflict",
        "parsed_address_conflict",
        "parsed_address_review_required",
    ]
    immutable += [
        f"comparison_{f}_{s}"
        for f in ("company", "street_name", "city")
        for s in ("l", "r")
    ]
    immutable += [
        f"{f}_{m}"
        for f in ("company", "street_name", "city")
        for m in (
            "jaro_winkler_similarity",
            "levenshtein_distance",
            "levenshtein_similarity",
        )
    ]
    unchanged = " AND ".join(f"(a.{name} <=> b.{name})" for name in immutable)
    sql = f"""WITH a AS (SELECT * FROM {prefix}.string_pair_scores WHERE string_scoring_run_id='{source_run}'),
     b AS (SELECT * FROM {prefix}.unified_pair_scores WHERE unified_run_id='{args.run_id}')
     SELECT count(a.id_l) source_pairs,count(b.id_l) scored_pairs,
      count_if(a.id_l IS NULL) extra_pairs,count_if(b.id_l IS NULL) missing_pairs,
      count_if(NOT ({unchanged})) changed_historical_evidence,
      count_if(b.fs_match_probability IS NULL OR isnan(b.fs_match_probability) OR b.fs_match_probability<0 OR b.fs_match_probability>1) invalid_probabilities,
      count_if(b.fs_match_weight IS NULL OR isnan(b.fs_match_weight) OR abs(b.fs_match_weight)>10000) invalid_weights,
      count_if(b.fs_decision='exact_baseline') exact_baseline_pairs,
      count_if(b.fs_decision='proposed_match' AND (b.fs_guard_conflict OR b.fs_guard_review OR b.fs_uses_untrained_level)) guarded_proposals,
      count_if(b.fs_decision='proposed_match' AND NOT coalesce(b.fs_original_text_agreement,false) AND (b.fs_match_probability<0.99 OR NOT coalesce(b.company_jaro_winkler_similarity>=0.85 AND b.street_name_jaro_winkler_similarity>=0.85 AND b.city_jaro_winkler_similarity>=0.85,false))) unsafe_model_proposals,
      count_if(b.fs_decision='proposed_match' AND coalesce(b.fs_original_text_agreement,false) AND NOT (coalesce(b.original_company_agreement OR b.original_street_name_agreement OR b.original_city_agreement,false) AND coalesce(b.original_company_agreement OR b.company_jaro_winkler_similarity>=0.95,false) AND coalesce(b.original_company_agreement OR b.company_jaro_winkler_similarity>=0.85,false) AND coalesce(b.original_street_name_agreement OR b.street_name_jaro_winkler_similarity>=0.85,false) AND coalesce(b.original_city_agreement OR b.city_jaro_winkler_similarity>=0.85,false))) unsafe_original_text_proposals,
      count_if(b.fs_decision='proposed_match' AND coalesce(b.fs_original_text_agreement,false)) original_text_proposals,
      count_if(b.fs_human_decision IS NOT NULL) human_decisions,
      count_if(b.fs_merge_authority IS DISTINCT FROM CASE b.fs_decision WHEN 'exact_baseline' THEN 'exact_rule' WHEN 'proposed_match' THEN 'model_proposal' ELSE 'none' END) inconsistent_merge_authority,
      count_if(b.fs_probability_prior_2x < b.fs_match_probability OR b.fs_probability_prior_4x < b.fs_probability_prior_2x) nonmonotone_prior_sensitivity
     FROM a FULL OUTER JOIN b ON a.id_l=b.id_l AND a.id_r=b.id_r"""
    counts = {key: int(value) for key, value in query(sql)[0].items()}
    for key in (
        "extra_pairs",
        "missing_pairs",
        "changed_historical_evidence",
        "invalid_probabilities",
        "invalid_weights",
        "guarded_proposals",
        "unsafe_model_proposals",
        "unsafe_original_text_proposals",
        "human_decisions",
        "inconsistent_merge_authority",
        "nonmonotone_prior_sensitivity",
    ):
        assert counts[key] == 0, (key, counts[key])
    assert counts["source_pairs"] == counts["scored_pairs"] == manifest["scored_pairs"]
    assert counts["exact_baseline_pairs"] == manifest["exact_baseline_pairs"]
    decisions = query(
        f"SELECT candidate_origin,fs_decision,count(*) pairs FROM {prefix}.unified_pair_scores WHERE unified_run_id='{args.run_id}' GROUP BY candidate_origin,fs_decision ORDER BY candidate_origin,fs_decision"
    )
    transitions = query(
        f"SELECT candidate_origin,decision,fs_decision,count(*) pairs FROM {prefix}.unified_pair_scores WHERE unified_run_id='{args.run_id}' GROUP BY candidate_origin,decision,fs_decision ORDER BY candidate_origin,decision,fs_decision"
    )
    args.output.mkdir(parents=True)
    for name, payload in (
        ("manifest", manifest),
        ("model", model),
        ("verification", counts),
        ("decisions", decisions),
        ("transitions", transitions),
    ):
        (args.output / (name + ".json")).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        )
    (args.output / "verification.sql").write_text(sql)
    print(
        json.dumps(
            {
                "verification": counts,
                "decisions": decisions,
                "training_sessions": manifest["training"]["training_sessions"],
                "unobserved_parameters": manifest["training"]["unobserved_parameters"],
            }
        )
    )


if __name__ == "__main__":
    main()
