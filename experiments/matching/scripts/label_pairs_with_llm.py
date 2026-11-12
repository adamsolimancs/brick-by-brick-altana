"""Label a stratified sample of scored pairs with a frontier model, then score the pipeline.

Each pair is one independent request with a fixed prompt and a strict JSON
schema. The model never sees the pipeline's decision or scores. Outputs land in
a new local directory (sample, every raw response, labels, report,
disagreements) and are appended to an append-only Delta table next to
``match_labels``. Model labels are a second opinion, not ground truth: read the
disagreement list.

The API key is read from the OPENAI_API_KEY environment variable or from
``--key-file`` (a file holding only the key; keep it under the ignored data/).
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.llm_labeling import (  # noqa: E402
    MODEL,
    PROMPT_VERSION,
    SEED,
    evaluate,
    insert_statements,
    label_pairs,
    pair_from_row,
    report_markdown,
    sample_sql,
    strata_sql,
    stratum_quotas,
)

LABEL_COLUMNS = (
    "labeling_run_id",
    "unified_run_id",
    "candidate_origin",
    "fs_decision",
    "unique_id_l",
    "unique_id_r",
    "content_hash_l",
    "content_hash_r",
    "label",
    "confidence",
    "reason",
    "reviewer",
    "model",
    "prompt_version",
    "reasoning_effort",
    "orientation_swapped",
    "population_n",
    "sample_n",
    "labeled_at",
)
LABEL_TABLE_DDL = """CREATE TABLE IF NOT EXISTS {table} (
  labeling_run_id STRING, unified_run_id STRING, candidate_origin STRING,
  fs_decision STRING, unique_id_l STRING, unique_id_r STRING,
  content_hash_l STRING, content_hash_r STRING, label STRING, confidence DOUBLE,
  reason STRING, reviewer STRING, model STRING, prompt_version STRING,
  reasoning_effort STRING, orientation_swapped BOOLEAN, population_n BIGINT,
  sample_n BIGINT, labeled_at TIMESTAMP
) USING DELTA
COMMENT 'Model-judged company/location pair labels: an independent second opinion, not ground truth; append-only'"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--unified-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prefix", default="addressresolution.matching")
    parser.add_argument(
        "--label-table",
        default="addressresolution.matching_experiments_aes10130.llm_pair_labels",
    )
    parser.add_argument("--key-file", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Draw the sample and write the prompts; call no model and write no table",
    )
    parser.add_argument(
        "--no-table", action="store_true", help="Keep results local only"
    )
    args = parser.parse_args()
    if not 20 <= args.size <= 2000:
        raise ValueError("Sample size must be between 20 and 2000")
    if args.effort not in ("low", "medium", "high", "xhigh", "max"):
        raise ValueError("Unsupported reasoning effort")
    if not re.fullmatch(r"[A-Za-z0-9_.]+", args.label_table):
        raise ValueError("Invalid label table name")
    if args.output.exists():
        raise ValueError(
            "Output directory already exists; labels are never overwritten"
        )

    def query(sql):
        completed = subprocess.run(
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
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
        text = completed.stdout
        start = text.find("[")
        return json.loads(text[start:]) if start >= 0 else []

    runs = query(
        f"SELECT manifest_json FROM {args.prefix}.unified_runs "
        f"WHERE unified_run_id = '{args.unified_run_id}' AND status = 'COMPLETED'"
    )
    if len(runs) != 1:
        raise ValueError("Exactly one completed unified run is required")
    source_run = json.loads(runs[0]["manifest_json"])["source_matching_run_id"]

    populations = {
        (row["candidate_origin"], row["fs_decision"]): int(row["population_n"])
        for row in query(strata_sql(args.prefix, args.unified_run_id, source_run))
    }
    quotas = stratum_quotas(populations, args.size)
    rows = query(sample_sql(args.prefix, args.unified_run_id, source_run, quotas))
    pairs = [pair_from_row(row) for row in rows]
    if len(pairs) != args.size:
        raise ValueError(f"Sampled {len(pairs)} pairs, expected {args.size}")

    labeling_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output.mkdir(parents=True)
    manifest = {
        "labeling_run_id": labeling_run_id,
        "unified_run_id": args.unified_run_id,
        "source_matching_run_id": source_run,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "reasoning_effort": args.effort,
        "seed": SEED,
        "size": args.size,
        "populations": [
            {
                "candidate_origin": o,
                "fs_decision": d,
                "population_n": n,
                "sample_n": quotas[(o, d)],
            }
            for (o, d), n in sorted(populations.items())
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.output / "sample.jsonl").open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "pairs": len(pairs),
                    "quotas": manifest["populations"],
                },
                indent=2,
            )
        )
        return

    from openai import OpenAI

    api_key = (
        args.key_file.read_text().strip()
        if args.key_file
        else os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY or pass --key-file")
    client = OpenAI(api_key=api_key, max_retries=4)
    log_lock = threading.Lock()
    log = (args.output / "responses.jsonl").open("a")

    def judge(system, user, schema):
        response = client.responses.create(
            model=MODEL,
            reasoning={"effort": args.effort},
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "pair_label",
                    "schema": schema,
                    "strict": True,
                }
            },
        )
        raw = response.output_text
        with log_lock:
            log.write(
                json.dumps(
                    {
                        "response_id": response.id,
                        "prompt": user,
                        "raw": raw,
                        "usage": response.usage.model_dump()
                        if response.usage
                        else None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            log.flush()
        return raw

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        labeled = label_pairs(pairs, judge, mapper=pool.map)
    log.close()

    labeled_at = datetime.now(timezone.utc)
    table_rows = [
        {
            "labeling_run_id": labeling_run_id,
            "unified_run_id": args.unified_run_id,
            "candidate_origin": row["candidate_origin"],
            "fs_decision": row["fs_decision"],
            "unique_id_l": row["id_l"],
            "unique_id_r": row["id_r"],
            "content_hash_l": row["content_hash_l"],
            "content_hash_r": row["content_hash_r"],
            "label": row["label"],
            "confidence": row["confidence"],
            "reason": row["reason"],
            "reviewer": MODEL,
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "reasoning_effort": args.effort,
            "orientation_swapped": row["orientation_swapped"],
            "population_n": int(row["population_n"]),
            "sample_n": int(row["sample_n"]),
            "labeled_at": labeled_at.isoformat(),
        }
        for row in labeled
    ]
    with (args.output / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(table_rows)
    write_evaluation(args.output, labeled, manifest)

    if not args.no_table:
        query(LABEL_TABLE_DDL.format(table=args.label_table))
        for statement in insert_statements(args.label_table, table_rows):
            query(statement)
        saved = query(
            f"SELECT count(*) AS n FROM {args.label_table} WHERE labeling_run_id = '{labeling_run_id}'"
        )
        if int(saved[0]["n"]) != len(table_rows):
            raise RuntimeError(
                "Label table row count does not match the labeled sample"
            )
    print(
        json.dumps(
            {
                "labeling_run_id": labeling_run_id,
                "pairs": len(labeled),
                "output": str(args.output),
                "table": None if args.no_table else args.label_table,
            }
        )
    )


def write_evaluation(output, labeled, manifest):
    result = evaluate(labeled)
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False)
    )
    (output / "report.md").write_text(report_markdown(result, manifest))
    with (output / "disagreements.csv").open("w", newline="") as handle:
        fields = [
            "kind",
            "candidate_origin",
            "fs_decision",
            "label",
            "confidence",
            "reason",
            "id_l",
            "id_r",
            "orientation_swapped",
            "prompt",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["disagreements"])
    return result


if __name__ == "__main__":
    main()
