"""Paired field-metric evaluation from independently supplied human labels."""

import csv
import hashlib
import json
import math
import posixpath
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .label_sampling import FIELDS

METRICS = ("jaro_winkler", "levenshtein")
DISPLAY_FIELDS = {"Company": "company", "Street name": "street_name", "City": "city"}
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_labels(path):
    """Read CSV or our workbook by sheet/header names, including reordered rows."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() != ".xlsx":
        raise ValueError("Labels must be a field_labels CSV or .xlsx workbook")
    with zipfile.ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.itertext()) for node in root.findall("s:si", NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        sheet = next((s for s in sheets if s.get("name") == "Field labels"), None)
        if sheet is None:
            raise ValueError("Workbook has no 'Field labels' worksheet")
        rel_id = sheet.get(
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        )
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(r.get("Target") for r in relationships if r.get("Id") == rel_id)
        sheet_path = (
            target.lstrip("/")
            if target.startswith("/")
            else posixpath.normpath("xl/" + target)
        )
        root = ET.fromstring(archive.read(sheet_path))
        rows = []
        headers = None
        for row in root.findall("s:sheetData/s:row", NS):
            values = {}
            for cell in row.findall("s:c", NS):
                col = "".join(c for c in cell.get("r") if c.isalpha())
                value = cell.findtext("s:v", default="", namespaces=NS)
                kind = cell.get("t")
                if kind == "s":
                    value = strings[int(value)]
                elif kind == "inlineStr":
                    value = "".join(cell.find("s:is", NS).itertext())
                # The label/source columns are literal values, never formulas.
                values[col] = (value, cell.find("s:f", NS) is not None)
            text = {col: v for col, (v, _) in values.items()}
            if headers is None:
                if {"Pair", "Field", "Judgment"}.issubset(set(text.values())):
                    headers = {v: col for col, v in text.items()}
                continue
            mapping = {
                "pair_ref": "Pair",
                "field": "Field",
                "value_a": "Compared value A",
                "value_b": "Compared value B",
                "label": "Judgment",
                "reviewer": "Reviewer",
            }
            item = {}
            for key, name in mapping.items():
                if name not in headers:
                    raise ValueError(f"Workbook missing column: {name}")
                value, formula = values.get(headers[name], ("", False))
                if formula:
                    raise ValueError(f"Formula in input column {name}")
                item[key] = value
            for key, name in (
                ("reviewer_type", "Reviewer type"),
                ("human_verified", "Human verified"),
            ):
                if name in headers:
                    value, formula = values.get(headers[name], ("", False))
                    if formula:
                        raise ValueError(f"Formula in provenance column {name}")
                    item[key] = value
            if any(item.values()):
                rows.append(item)
        if headers is None:
            raise ValueError("Field-label headers not found")
        return rows


def join_labels(labels, sample):
    """Reject stale/edited source values and duplicate, unadjudicated judgments."""
    expected = {(r["pair_ref"], r["field"]): r for r in sample["fields"]}
    analysis = {r["pair_ref"]: r for r in sample["analysis"]}
    if len(expected) != len(sample["fields"]) or len(analysis) != len(
        sample["analysis"]
    ):
        raise ValueError("Duplicate keys in sample metadata")
    joined, seen = [], set()
    counts = {f: Counter() for f in FIELDS}
    for row in labels:
        field = DISPLAY_FIELDS.get(row.get("field"), row.get("field"))
        key = (row.get("pair_ref"), field)
        if key not in expected:
            raise ValueError(f"Unknown pair/field: {key}")
        if key in seen:
            raise ValueError(f"Duplicate judgment for {key}; adjudicate separately")
        seen.add(key)
        original = expected[key]
        if row.get("pair_id") and row["pair_id"] != original["pair_id"]:
            raise ValueError(f"Pair identity changed: {key}")
        for name in ("value_a", "value_b"):
            if name not in row or row[name] != original[name]:
                raise ValueError(f"Compared source value changed: {key}, {name}")
        label = (row.get("label") or "").strip().lower()
        if label not in ("", "same", "different", "uncertain"):
            raise ValueError(f"Invalid label for {key}: {label}")
        counts[field][label or "blank"] += 1
        if not label:
            continue
        if not (row.get("reviewer") or "").strip():
            raise ValueError(f"Reviewer required for {key}")
        source = analysis[key[0]]
        weight = float(source["sampling_weight"])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(f"Invalid sampling weight: {key}")
        item = {
            "pair_ref": key[0],
            "field": field,
            "label": label,
            "group": source["evaluation_group"],
            "weight": weight,
            "focus": source["sample_focus"],
            "reviewer_type": row.get("reviewer_type") or "unspecified",
            "human_verified": str(row.get("human_verified", "")).lower() == "true",
        }
        for metric in METRICS:
            score = float(source[f"{field}_{metric}_similarity"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid {metric} score: {key}")
            item[metric] = score
        joined.append(item)
    for ref, field in expected.keys() - seen:
        counts[field]["blank"] += 1
    return joined, {f: dict(c) for f, c in counts.items()}


def grouped_folds(analysis, n_folds):
    """Balance pair counts without using labels or scores; groups stay intact."""
    sizes = Counter(r["evaluation_group"] for r in analysis)
    n_folds = min(n_folds, len(sizes))
    if n_folds < 2:
        return {}
    load = [0] * n_folds
    result = {}
    groups = sorted(
        sizes,
        key=lambda g: (-sizes[g], hashlib.sha256(g.encode()).hexdigest()),
    )
    for group in groups:
        fold = min(range(n_folds), key=lambda f: (load[f], f))
        result[group] = fold
        load[fold] += sizes[group]
    return result


def confusion(rows, metric, threshold):
    totals = dict.fromkeys(("tp", "fp", "fn", "tn"), 0.0)
    for row in rows:
        prediction = row[metric] >= threshold
        positive = row["label"] == "same"
        key = (
            ("tp" if positive else "fp") if prediction else ("fn" if positive else "tn")
        )
        totals[key] += row["weight"]
    tp, fp, fn = (totals[k] for k in ("tp", "fp", "fn"))
    return {
        **totals,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def select_threshold(rows, metric, target_precision):
    """Optimize training recall at the precision target; never use test labels."""
    if {r["label"] for r in rows} != {"same", "different"}:
        return None
    # O(n log n) sweep; process tied scores together, as >= threshold would.
    by_score = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        by_score[row[metric]][row["label"] == "different"] += row["weight"]
    positives = sum(r["weight"] for r in rows if r["label"] == "same")
    tp = fp = 0.0
    candidates = []
    for threshold in sorted(by_score, reverse=True):
        a, b = by_score[threshold]
        tp += a
        fp += b
        precision = tp / (tp + fp)
        if tp and precision >= target_precision:
            candidates.append((tp / positives, precision, threshold))
    return max(candidates)[2] if candidates else None


def weighted_auc(rows, metric):
    """Weighted ROC AUC with half credit for score ties."""
    by_score = defaultdict(lambda: [0.0, 0.0])
    for row in rows:
        by_score[row[metric]][row["label"] == "different"] += row["weight"]
    positive = sum(v[0] for v in by_score.values())
    negative = sum(v[1] for v in by_score.values())
    if not positive or not negative:
        return None
    lower_negative = wins = 0.0
    for score in sorted(by_score):
        p, n = by_score[score]
        wins += p * (lower_negative + 0.5 * n)
        lower_negative += n
    return wins / (positive * negative)


def evaluate(sample, labels, target_precision=0.95, n_folds=5, unweighted=False):
    if not 0 < target_precision <= 1 or not 2 <= n_folds <= 10:
        raise ValueError("Precision must be (0,1]; folds must be 2–10")
    joined, counts = join_labels(labels, sample)
    if unweighted:
        for row in joined:
            row["weight"] = 1.0
    reviewer_types = dict(Counter(r["reviewer_type"] for r in joined))
    fold_map = grouped_folds(sample["analysis"], n_folds)
    result = {
        "status": "labels_required" if not joined else "pilot_evaluation",
        "label_counts": counts,
        "reviewer_types": reviewer_types,
        "human_verified_judgments": sum(r["human_verified"] for r in joined),
        "weighting": "unweighted_labeled_subset" if unweighted else "sampling_weights",
        "target_training_precision": target_precision,
        "folds": len(set(fold_map.values())),
        "fields": {},
        "limitations": [
            "Reviewer field agreement is the target, not overall entity identity",
            "Uncertain judgments are excluded from binary metrics and reported separately",
            "Precision target is exploratory and is not a production acceptance rule",
            "Test-fold precision may fall below the training precision target",
            "Related-company grouping beyond identical names may need manual review",
            "No automatic winner: inspect paired results and obtain more labels if needed",
            "No confidence intervals; pilot estimates are descriptive, not significance claims",
            "Partial/adjudicated-only labels may be selective; original weights do not correct label nonresponse",
            "Candidate-only labels cannot measure retrieval recall or calibrate entity-match probabilities",
        ],
    }
    if "model" in reviewer_types:
        result["status"] = "model_review_pilot"
        result["limitations"].insert(
            0,
            "Model-generated labels are provisional reference judgments, not independently validated human ground truth",
        )
    for field in FIELDS:
        rows = [r for r in joined if r["field"] == field and r["label"] != "uncertain"]
        field_result = {
            "binary_labels": len(rows),
            "distinct_groups": len({r["group"] for r in rows}),
            "same": sum(r["label"] == "same" for r in rows),
            "different": sum(r["label"] == "different" for r in rows),
            "complete_binary_coverage": len(rows) == len(sample["analysis"]),
            "metrics": {},
        }
        # Both algorithms are evaluated on the same folds/cases. If training
        # has both classes but no threshold meets the target, predict none;
        # do not hide that algorithm's failure by dropping its test cases.
        fold_rows = []
        predictions = {m: [] for m in METRICS}
        for fold in sorted(set(fold_map.values())):
            train = [r for r in rows if fold_map.get(r["group"]) != fold]
            test = [r for r in rows if fold_map.get(r["group"]) == fold]
            if not test:
                continue
            thresholds = {
                m: select_threshold(train, m, target_precision) for m in METRICS
            }
            eligible = {r["label"] for r in train} == {"same", "different"}
            fold_rows.append(
                {
                    "fold": fold,
                    "train_n": len(train),
                    "test_n": len(test),
                    "thresholds": thresholds,
                    "eligible": eligible,
                    "abstain_all": [m for m, t in thresholds.items() if t is None]
                    if eligible
                    else [],
                    "reason": None if eligible else "Training labels lack both classes",
                }
            )
            if eligible:
                for metric in METRICS:
                    predictions[metric].extend(
                        {
                            **r,
                            "prediction": float(
                                thresholds[metric] is not None
                                and r[metric] >= thresholds[metric]
                            ),
                        }
                        for r in test
                    )
        field_result["fold_details"] = fold_rows
        for metric in METRICS:
            oof = predictions[metric]
            field_result["metrics"][metric] = {
                "weighted_auc_all_labeled": weighted_auc(rows, metric),
                "out_of_fold_n": len(oof),
                "out_of_fold_complete": bool(rows) and len(oof) == len(rows),
                "weighted_out_of_fold": confusion(oof, "prediction", 0.5)
                if oof
                else None,
                "unweighted_out_of_fold": confusion(
                    [{**r, "weight": 1.0} for r in oof], "prediction", 0.5
                )
                if oof
                else None,
                "disagreement_focus_out_of_fold": confusion(
                    [
                        r
                        for r in oof
                        if r["focus"] in (field + "_jw_only", field + "_lev_only")
                    ],
                    "prediction",
                    0.5,
                )
                if oof
                else None,
                "development_threshold_all_labels": select_threshold(
                    rows, metric, target_precision
                ),
            }
        result["fields"][field] = field_result
    return result


def report_markdown(result):
    lines = [
        "# Field comparator evaluation",
        "",
        f"Status: **{result['status']}**",
        f"Reviewer types: {result['reviewer_types']}; human-verified judgments: {result['human_verified_judgments']}.",
        f"Weighting: {result['weighting']}.",
        "",
        "| Field | Same | Different | Uncertain | Blank |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for field in FIELDS:
        c = result["label_counts"][field]
        lines.append(
            f"| {field} | {c.get('same', 0)} | {c.get('different', 0)} | {c.get('uncertain', 0)} | {c.get('blank', 0)} |"
        )
    if result["status"] == "labels_required":
        lines += [
            "",
            "No human judgments are available. Accuracy, thresholds, and a winning metric cannot be established.",
        ]
    else:
        lines += [
            "",
            f"Training precision target: {result['target_training_precision']:.0%}; grouped folds: {result['folds']}.",
            "Thresholds are selected on training folds only. Results use the same eligible test cases for both metrics. A metric with no feasible training threshold predicts no agreements for that fold.",
            "",
            "| Field | Metric | Test judgments | Precision | Recall | ROC AUC |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]

        def percent(v):
            return "—" if v is None else f"{v:.1%}"

        for field, detail in result["fields"].items():
            for metric, m in detail["metrics"].items():
                stats = m["weighted_out_of_fold"] or {}
                lines.append(
                    f"| {field} | {metric} | {m['out_of_fold_n']} / {detail['binary_labels']} | {percent(stats.get('precision'))} | {percent(stats.get('recall'))} | {percent(m['weighted_auc_all_labeled'])} |"
                )
        lines += [
            "",
            "AUC uses all binary-labeled cases without selecting a threshold. Ineligible folds and per-fold thresholds are recorded in evaluation.json.",
            "All-label development thresholds in JSON are exploratory fits, not held-out estimates or production rules.",
        ]
    lines += ["", "## Limits", "", *["- " + v for v in result["limitations"]], ""]
    return "\n".join(lines)


def write_report(output, result):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    (output / "evaluation.md").write_text(report_markdown(result))
