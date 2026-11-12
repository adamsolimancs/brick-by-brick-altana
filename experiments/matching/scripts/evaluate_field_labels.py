"""Evaluate labeled candidate fields without rebuilding embeddings or scoring."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.label_evaluation import evaluate, read_labels, write_report  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Descriptive comparison within the labeled subset; do not estimate population accuracy",
    )
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text())
    result = evaluate(
        sample,
        read_labels(args.labels),
        args.target_precision,
        args.folds,
        args.unweighted,
    )
    result["input_sha256"] = {
        "labels": hashlib.sha256(args.labels.read_bytes()).hexdigest(),
        "sample": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
    }
    write_report(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "labels": result["label_counts"],
                "report": str(args.output / "evaluation.md"),
            }
        )
    )


if __name__ == "__main__":
    main()
