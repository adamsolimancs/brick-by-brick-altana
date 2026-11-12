"""Experiment: can resolved records fill a blank city or street in dlq_incomplete?

Inputs are two CSV exports (see --help). Keys come from the project's own
prepare_record so a lookup means what the matcher means: same country, same
company name key, and the address part the incomplete row still has. A fill is
"unique" when every supporting row agrees on one value; it is a proposal for
review, never written back to Silver.

Superseded for production by attach_incomplete in lib/resolution.py, which the
resolution task runs with a stricter rule (one entity, matching unit, no
region conflict). This script stays as the exploratory report.

Export the inputs with the Databricks CLI, for example:
  databricks experimental aitools tools query "SELECT provider, provider_id, company,
    street_address, city, state, country, routed_from FROM
    addressresolution.silver.dlq_incomplete" --profile <p> --output csv > dlq.csv
  databricks experimental aitools tools query "SELECT provider, provider_id, company,
    street_address, city, state, country, country_key, region_key, city_key, name_key,
    road_key, house_key, entity_id, deduplication_status FROM
    addressresolution.matching.resolved_records WHERE deduplication_run_id = '<run>'"
    --profile <p> --output csv > resolved.csv
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.matching import prepare_record  # noqa: E402


def distinct_values(rows, field):
    values = defaultdict(set)
    for row in rows:
        values[row[field]].add(row["entity_id"])
    return values


def propose(dlq_rows, resolved_rows):
    by_street = defaultdict(list)
    by_city = defaultdict(list)
    by_region = defaultdict(list)
    for row in resolved_rows:
        key = (row["country_key"], row["name_key"])
        by_street[key + (row["house_key"], row["road_key"])].append(row)
        by_city[key + (row["city_key"],)].append(row)
        by_region[key + (row["region_key"],)].append(row)
    summary = Counter()
    proposals = []
    for row in dlq_rows:
        keys = prepare_record(dict(row))
        base = (keys["country_key"], keys["name_key"])
        if not row["city"]:
            missing, field = "city", "city"
            hits = by_street[base + (keys["house_key"], keys["road_key"])]
        elif not row["street_address"]:
            missing, field = "street", "street_address"
            hits = by_city[base + (keys["city_key"],)]
        else:
            summary["other_reason"] += 1
            continue
        same_region = [hit for hit in hits if hit["region_key"] == keys["region_key"]]
        hits = same_region or hits
        candidates = distinct_values(hits, field)
        if not candidates:
            known = (
                "company_in_region_other_address"
                if by_region[base + (keys["region_key"],)]
                else "company_unknown"
            )
            summary[f"{missing}_missing:{known}"] += 1
            continue
        outcome = "unique" if len(candidates) == 1 else "ambiguous"
        summary[f"{missing}_missing:{outcome}"] += 1
        proposals.append(
            {
                "provider": row["provider"],
                "provider_id": row["provider_id"],
                "company": row["company"],
                "country": row["country"],
                "state": row["state"],
                "present": row["street_address"] if missing == "city" else row["city"],
                "missing_field": field,
                "outcome": outcome,
                "proposed": " | ".join(sorted(candidates)),
                "supporting_rows": len(hits),
                "entities": len({e for ids in candidates.values() for e in ids}),
                "donor_provider": ",".join(sorted({h["provider"] for h in hits})),
                "donor_ids": ",".join(sorted({h["provider_id"] for h in hits})[:4]),
                "donor_status": ",".join(
                    sorted({h["deduplication_status"] for h in hits})
                ),
            }
        )
    return summary, proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dlq", required=True, help="dlq_incomplete CSV export")
    parser.add_argument("--resolved", required=True, help="resolved_records CSV export")
    parser.add_argument("--output", required=True, help="proposals CSV to write")
    args = parser.parse_args()
    with open(args.dlq, newline="") as handle:
        dlq_rows = list(csv.DictReader(handle))
    with open(args.resolved, newline="") as handle:
        resolved_rows = list(csv.DictReader(handle))
    summary, proposals = propose(dlq_rows, resolved_rows)
    with open(args.output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(proposals[0].keys()))
        writer.writeheader()
        writer.writerows(proposals)
    print(json.dumps(dict(sorted(summary.items())), indent=1))


if __name__ == "__main__":
    main()
