"""Recompute the benchmark report from a saved labels.csv without calling any model."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.llm_labeling import evaluate, report_markdown  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels", type=Path, required=True, help="Labeling output directory"
    )
    args = parser.parse_args()
    manifest = json.loads((args.labels / "manifest.json").read_text())
    with (args.labels / "labels.csv").open(newline="") as handle:
        rows = [
            dict(row, id_l=row["unique_id_l"], id_r=row["unique_id_r"])
            for row in csv.DictReader(handle)
        ]
    result = evaluate(rows)
    (args.labels / "report.md").write_text(report_markdown(result, manifest))
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("labeled_pairs", "scored_pairs", "unweighted", "weighted")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
