"""Verify an appended cleanup run against v1 and reevaluate fixed reference labels."""

import argparse
import copy
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altana_matching_components import VERSION as COMPONENT_VERSION  # noqa: E402
from lib.label_evaluation import evaluate, read_labels, write_report  # noqa: E402
from lib.label_sampling import FIELDS, sql_string  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-run-id", default="95648575728701")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-code-sha256", required=True)
    args = parser.parse_args()
    if not all(
        re.fullmatch(r"[A-Za-z0-9_-]{1,128}", s)
        for s in (args.run_id, args.baseline_run_id)
    ):
        raise ValueError("Invalid run ID")
    if args.output.exists():
        raise ValueError("Output directory exists; never overwrite previous results")
    prefix = "addressresolution.matching_experiments_aes10130"

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

    manifests = {}
    for run_id in (args.baseline_run_id, args.run_id):
        rows = query(
            f"SELECT manifest_json FROM {prefix}.string_scoring_runs WHERE string_scoring_run_id={sql_string(run_id)} AND status='COMPLETED'"
        )
        if len(rows) != 1:
            raise ValueError(f"Exactly one completed manifest is required: {run_id}")
        manifests[run_id] = json.loads(rows[0]["manifest_json"])
    baseline, new = manifests[args.baseline_run_id], manifests[args.run_id]
    assert new["source_code_sha256"] == args.expected_code_sha256, (
        "Deployed code hash differs"
    )
    for key in (
        "source_run_id",
        "source_input_fingerprint",
        "input_embedding_run_id",
        "scored_pairs",
    ):
        assert baseline[key] == new[key], f"Source changed: {key}"
    assert new["street_component_version"] == COMPONENT_VERSION
    assert not new["embeddings_rebuilt"] and not new["hnsw_indexes_rebuilt"]
    immutable = [
        "content_hash_l",
        "content_hash_r",
        "source_run_id",
        "candidate_origin",
        "decision",
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
    immutable += [
        f"comparison_{f}_{s}" for f in ("company", "city") for s in ("l", "r")
    ]
    immutable += [
        f"{f}_{metric}"
        for f in ("company", "city")
        for metric in (
            "levenshtein_distance",
            "levenshtein_similarity",
            "jaro_winkler_similarity",
        )
    ]
    equal = " AND ".join(f"(a.{key} <=> b.{key})" for key in immutable)
    verification_sql = f"""WITH a AS (SELECT * FROM {prefix}.string_pair_scores WHERE string_scoring_run_id={sql_string(args.baseline_run_id)}),
      b AS (SELECT * FROM {prefix}.string_pair_scores WHERE string_scoring_run_id={sql_string(args.run_id)})
      SELECT count(a.id_l) baseline_pairs,count(b.id_l) new_pairs,
       count_if(a.id_l IS NULL) missing_baseline_pairs,count_if(b.id_l IS NULL) missing_new_pairs,
       count_if(NOT ({equal})) changed_immutable_fields,
       count_if(NOT (a.comparison_street_name_l <=> b.previous_street_name_l) OR NOT (a.comparison_street_name_r <=> b.previous_street_name_r)) changed_previous_street_keys,
       count_if(b.parsed_address_conflict AND NOT coalesce(a.house_conflict OR a.unit_conflict OR a.numeric_conflict,false)) newly_exposed_component_conflicts,
       count_if(b.parsed_address_conflict AND b.decision='proposed_match') proposed_pairs_with_parsed_conflicts,
       count_if(b.parsed_address_review_required AND b.decision='proposed_match') proposed_pairs_with_component_review,
       count_if(b.parsed_address_conflict AND b.decision='exact_baseline') exact_pairs_with_parsed_conflicts
      FROM a FULL OUTER JOIN b ON a.id_l=b.id_l AND a.id_r=b.id_r"""
    verification = query(verification_sql)[0]
    for key in (
        "missing_baseline_pairs",
        "missing_new_pairs",
        "changed_immutable_fields",
        "changed_previous_street_keys",
        "exact_pairs_with_parsed_conflicts",
    ):
        assert int(verification[key]) == 0, (key, verification[key])
    assert (
        int(verification["baseline_pairs"])
        == int(verification["new_pairs"])
        == new["scored_pairs"]
    )

    original_sample = json.loads(args.sample.read_text())
    sample = copy.deepcopy(original_sample)
    requested = ",".join(
        f"({sql_string(r['id_l'])},{sql_string(r['id_r'])})" for r in sample["analysis"]
    )
    columns = [
        "id_l",
        "id_r",
        "content_hash_l",
        "content_hash_r",
        "previous_street_name_l",
        "previous_street_name_r",
        "parsed_address_conflict",
        "parsed_address_review_required",
    ]
    columns += [f"comparison_{f}_{side}" for f in FIELDS for side in ("l", "r")]
    columns += [
        f"{f}_{metric}"
        for f in FIELDS
        for metric in (
            "levenshtein_distance",
            "levenshtein_similarity",
            "jaro_winkler_similarity",
        )
    ]
    columns += [
        f"parsed_{c}_{side}"
        for c in ("house_numbers", "units", "buildings", "floors")
        for side in ("l", "r")
    ]
    columns += [
        f"parsed_{c}_{kind}"
        for c in ("house_numbers", "units", "buildings", "floors")
        for kind in ("conflict", "partial_agreement", "missing_one_side")
    ]
    sampled = query(
        f"WITH requested(id_l,id_r) AS (VALUES {requested}) SELECT {','.join('n.' + c for c in columns)} FROM {prefix}.string_pair_scores n JOIN requested r ON n.id_l=r.id_l AND n.id_r=r.id_r WHERE n.string_scoring_run_id={sql_string(args.run_id)}"
    )
    lookup = {(r["id_l"], r["id_r"]): r for r in sampled}
    assert len(lookup) == len(sample["analysis"]) == len(sampled), (
        "Sample pair conservation failed"
    )
    for r in sample["analysis"]:
        refreshed = lookup[(r["id_l"], r["id_r"])]
        for side in ("l", "r"):
            assert r[f"content_hash_{side}"] == refreshed[f"content_hash_{side}"]
            assert (
                r[f"comparison_street_name_{side}"]
                == refreshed[f"previous_street_name_{side}"]
            )
        r.update(refreshed)
        r["source_scoring_run_id"] = args.run_id
    metadata = {r["pair_ref"]: r for r in sample["analysis"]}
    original_fields = {
        (f["pair_ref"], f["field"]): f for f in original_sample["fields"]
    }
    for row in sample["fields"]:
        r = metadata[row["pair_ref"]]
        a = "l" if r["display_a_unique_id"] == r["id_l"] else "r"
        b = "r" if a == "l" else "l"
        row.update(
            value_a=r[f"comparison_{row['field']}_{a}"],
            value_b=r[f"comparison_{row['field']}_{b}"],
        )
    fields = {(f["pair_ref"], f["field"]): f for f in sample["fields"]}
    labels = read_labels(args.labels)
    for row in labels:
        key = (row["pair_ref"], row["field"])
        before, after = original_fields[key], fields[key]
        assert (
            row["value_a"] == before["value_a"] and row["value_b"] == before["value_b"]
        ), "Reference labels changed"
        row.update(
            original_value_a=row["value_a"],
            original_value_b=row["value_b"],
            value_a=after["value_a"],
            value_b=after["value_b"],
            comparison_view_version=new["street_component_version"],
        )
    result = evaluate(sample, labels, unweighted=True)
    result.update(
        source_scoring_run_id=args.run_id,
        baseline_scoring_run_id=args.baseline_run_id,
        original_reference_labels_sha256=hashlib.sha256(
            args.labels.read_bytes()
        ).hexdigest(),
        score_source="Completed Databricks run; original reference judgments fixed before cleanup",
    )
    args.output.mkdir(parents=True)
    (args.output / "manifest.json").write_text(json.dumps(new, indent=2))
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2))
    (args.output / "verification.sql").write_text(verification_sql)
    (args.output / "updated_sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2)
    )
    with (args.output / "reference_labels_cleaned_view.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]))
        writer.writeheader()
        writer.writerows(labels)
    write_report(args.output / "evaluation", result)
    print(
        json.dumps(
            {
                "manifest": new["street_component_summary"],
                "changed_pair_streets": new["changed_street_name_pairs"],
                "verification": verification,
                "evaluation": str(args.output / "evaluation/evaluation.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
