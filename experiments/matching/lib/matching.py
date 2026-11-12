"""Bounded DuckDB candidate generation and Fellegi-Sunter linkage.

Fitting happens only in the Databricks job. Probabilities are uncalibrated model
estimates; decisions are experimental suggestions, never production merges.
"""

import hashlib
import json
import math
import re
import unicodedata

from altana_matching_components import (
    ALIASES,
    company_name_key,
    normalize_region,
    normalized,
    text_key,
)

FIELDS = ("company", "street_address", "city", "state", "country")
VERSION = "multipass_fs_v2"
SEED = 20261029
MERGE_THRESHOLD = 0.99
REJECT_THRESHOLD = 0.10

UNIT = re.compile(
    r"(?:\b(?:apartment|apt|suite|ste|unit)\.?\s*|#\s*)([\w-]+)\s*$", re.I
)
HOUSE = re.compile(r"^(\d+[a-z]?(?:[-/]\d+[a-z]?)*)\b", re.I)


def digest(value):
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def prepare_record(row):
    """Derive comparison views without changing original business fields."""
    result = dict(row)
    result["unique_id"] = digest([str(row["provider"]), str(row["provider_id"])])
    result["content_hash"] = digest({f: row.get(f) for f in FIELDS})
    # Group membership is identical to lib/gold.py; this ID is experiment-local.
    result["baseline_id"] = digest([normalized(row.get(f)) for f in FIELDS])
    country = normalized(row.get("country")).upper()
    result["country_key"] = country
    result["region_key"] = normalize_region(row.get("state"), country)
    result["city_key"] = text_key(row.get("city"))
    result["company_key"] = text_key(row.get("company"))
    # Legal-form stripping supplies a comparison view, never a merge rule.
    result["name_key"] = company_name_key(row.get("company"))
    street = unicodedata.normalize("NFKC", normalized(row.get("street_address")))
    unit = UNIT.search(street)
    result["unit_key"] = text_key(unit.group(1)) if unit else ""
    road = street[: unit.start()].strip(" ,") if unit else street
    house = HOUSE.match(road)
    result["house_key"] = house.group(1) if house else ""
    road = road[house.end() :] if house else road
    tokens = text_key(road).split()
    if country in {"US", "CA", "GB", "AU"}:
        tokens = [ALIASES.get(token, token) for token in tokens]
    result["road_key"] = " ".join(tokens)
    result["street_key"] = " ".join(
        filter(
            None,
            [
                result["house_key"],
                result["road_key"],
                "unit " + result["unit_key"] if unit else "",
            ],
        )
    )
    building_text = street[: unit.start()] if unit else street
    result["number_signature"] = "|".join(
        re.findall(r"\d+[a-z]?(?:[-/]\d+[a-z]?)*", text_key(building_text))
    )
    result["parser_status"] = (
        "leading_house_number" if house else "unparsed_house_number"
    )
    return result


def blocking_passes():
    """Return ordered, audited passes; equality on empty keys never blocks."""
    return [
        ("exact_baseline", ["baseline_id"]),
        ("region_city_house", ["country_key", "region_key", "city_key", "house_key"]),
        ("region_name_prefix", ["country_key", "region_key", "name_prefix"]),
        ("country_name_rescue", ["country_key", "name_key"]),
        ("city_road_prefix", ["country_key", "city_key", "road_prefix"]),
    ]


def register_records(connection, records):
    import pyarrow as pa

    prepared = [prepare_record(r) for r in records]
    for row in prepared:
        compact = row["name_key"].replace(" ", "")
        row["name_prefix"] = compact[:6] if len(compact) >= 4 else ""
        road = row["road_key"].replace(" ", "")
        row["road_prefix"] = road[:8] if len(road) >= 4 else ""
    ids = [r["unique_id"] for r in prepared]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate source identities must be quarantined before matching"
        )
    if len(prepared) < 2:
        raise ValueError("The experiment needs at least two accepted source records")
    connection.register("input_arrow", pa.Table.from_pylist(prepared))
    connection.execute(
        "CREATE TABLE records AS SELECT * FROM input_arrow ORDER BY unique_id"
    )
    return prepared


def build_candidates(connection, max_pairs=2_000_000):
    """Guard block cardinalities before any pair expansion; never silently cap."""
    audits = []
    queries = []
    rules = []
    for name, columns in blocking_passes():
        present = " AND ".join(f"{c} IS NOT NULL AND {c} <> ''" for c in columns)
        grouping = ", ".join(columns)
        count, largest = connection.execute(f"""
            SELECT coalesce(sum(n * (n - 1) // 2), 0), coalesce(max(n), 0)
            FROM (SELECT count(*) n FROM records WHERE {present} GROUP BY {grouping})
        """).fetchone()
        audits.append(
            {"blocking_pass": name, "raw_pair_count": count, "largest_block": largest}
        )
        if sum(a["raw_pair_count"] for a in audits) > max_pairs:
            raise ValueError(
                f"Candidate budget {max_pairs} exceeded at {name}: {audits}"
            )
        rule = " AND ".join(f"l.{c} = r.{c} AND l.{c} <> ''" for c in columns)
        rules.append(rule)
        queries.append(
            f"SELECT l.unique_id id_l, r.unique_id id_r, '{name}' blocking_pass FROM records l JOIN records r ON {rule} AND l.unique_id < r.unique_id"
        )
    connection.execute(
        "CREATE TABLE candidate_passes AS " + " UNION ALL ".join(queries)
    )
    connection.execute("""CREATE TABLE candidates AS
        SELECT id_l, id_r, string_agg(blocking_pass, ',' ORDER BY blocking_pass) blocking_passes
        FROM candidate_passes GROUP BY id_l, id_r""")
    return audits, rules


def model_settings(rules, prior):
    from splink import SettingsCreator
    import splink.comparison_library as cl

    return SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="unique_id",
        comparisons=[
            cl.JaroWinklerAtThresholds("name_key", [0.95, 0.85, 0.70]).configure(
                term_frequency_adjustments=True
            ),
            cl.JaroWinklerAtThresholds("street_key", [0.95, 0.85, 0.70]),
            cl.JaroWinklerAtThresholds("city_key", [0.95, 0.85]),
        ],
        blocking_rules_to_generate_predictions=rules,
        probability_two_random_records_match=prior,
        retain_intermediate_calculation_columns=True,
        retain_matching_columns=True,
        max_iterations=50,
        em_convergence=0.0001,
    )


def fit_model(connection, rules, baseline_pairs, record_count):
    """Fit unsupervised m/u; expose the unvalidated prior assumption separately."""
    from splink import DuckDBAPI, Linker

    possible_pairs = record_count * (record_count - 1) // 2
    # Exact-rule precision/recall are unknown. Use its observed rate as an
    # explicitly assumed reference prior and retain 2x/4x sensitivity.
    prior = min(0.1, max(baseline_pairs / possible_pairs, 1 / possible_pairs))
    api = DuckDBAPI(connection)
    linker = Linker(api.register("records"), model_settings(rules, prior))
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000, seed=SEED)
    sessions = []
    for column in ("name_key", "street_key"):
        rule = f"l.country_key = r.country_key AND l.{column} = r.{column} AND l.{column} <> ''"
        pairs = connection.execute(
            f"SELECT count(*) FROM records l JOIN records r ON {rule} AND l.unique_id < r.unique_id"
        ).fetchone()[0]
        if pairs < 10:
            raise ValueError(f"Insufficient pairs for EM training on {column}: {pairs}")
        session = linker.training.estimate_parameters_using_expectation_maximisation(
            rule, max_pairs=200_000
        )
        sessions.append(
            {
                "blocking_column": column,
                "available_training_pairs": pairs,
                "parameter_history_records": len(session._iteration_history_records)
                if hasattr(session, "_iteration_history_records")
                else None,
            }
        )
    model = linker.misc.save_model_to_json()
    untrained_levels = [
        {
            "comparison": comparison["output_column_name"],
            "level": level["label_for_charts"],
        }
        for comparison in model["comparisons"]
        for level in comparison["comparison_levels"]
        if not level.get("is_null_level")
        and not all(
            isinstance(level.get(p), (int, float))
            for p in ("m_probability", "u_probability")
        )
    ]
    predictions = linker.inference.predict(threshold_match_probability=0)
    connection.execute(
        f"CREATE TABLE predictions AS SELECT * FROM {predictions.physical_name}"
    )
    return (
        linker,
        model,
        {
            "prior": prior,
            "prior_source": "assumed_observed_exact_pair_rate_floor_one_pair",
            "training_sessions": sessions,
            "untrained_levels_using_library_defaults": untrained_levels,
            "calibration_status": "unvalidated_unsupervised",
        },
    )


def suggestions(connection, prior):
    """Retain raw estimates and apply conservative location/company guards."""
    log_odds = math.log2(prior / (1 - prior))
    connection.execute(f"""CREATE TABLE pair_scores AS
        SELECT p.*, c.blocking_passes,
            l.provider provider_l, l.provider_id provider_id_l,
            r.provider provider_r, r.provider_id provider_id_r,
            l.company company_l, r.company company_r,
            l.street_address street_address_l, r.street_address street_address_r,
            l.city city_l, r.city city_r, l.state state_l, r.state state_r,
            l.country country_l, r.country country_r,
            l.content_hash content_hash_l, r.content_hash content_hash_r,
            l.input_route input_route_l, r.input_route input_route_r,
            l.baseline_id = r.baseline_id exact_baseline,
            l.house_key <> '' AND r.house_key <> '' AND l.house_key <> r.house_key house_conflict,
            l.unit_key <> '' AND r.unit_key <> '' AND l.unit_key <> r.unit_key unit_conflict,
            (l.unit_key = '') <> (r.unit_key = '') unit_missing_one_side,
            l.number_signature <> r.number_signature numeric_conflict,
            l.region_key <> r.region_key region_conflict,
            jaro_winkler_similarity(l.name_key, r.name_key) name_similarity,
            jaro_winkler_similarity(l.street_key, r.street_key) street_similarity,
            jaro_winkler_similarity(l.city_key, r.city_key) city_similarity,
            1.0 / (1.0 + power(2.0, -(p.match_weight - {log_odds} + log2(least({prior} * 2, 0.5) / (1-least({prior} * 2, 0.5)))))) probability_prior_2x,
            1.0 / (1.0 + power(2.0, -(p.match_weight - {log_odds} + log2(least({prior} * 4, 0.5) / (1-least({prior} * 4, 0.5)))))) probability_prior_4x
        FROM predictions p
        JOIN candidates c ON c.id_l=p.unique_id_l AND c.id_r=p.unique_id_r
        JOIN records l ON l.unique_id=p.unique_id_l
        JOIN records r ON r.unique_id=p.unique_id_r
    """)
    connection.execute(f"""CREATE TABLE decisions AS SELECT *,
        CASE
          WHEN exact_baseline THEN 'exact_baseline'
          WHEN house_conflict OR unit_conflict OR numeric_conflict OR region_conflict THEN 'review_conflict'
          WHEN match_probability >= {MERGE_THRESHOLD}
            AND NOT unit_missing_one_side
            AND name_similarity >= 0.85 AND street_similarity >= 0.85 AND city_similarity >= 0.85
            THEN 'proposed_match'
          WHEN unit_missing_one_side THEN 'review_uncertain'
          WHEN match_probability <= {REJECT_THRESHOLD} THEN 'likely_nonmatch'
          ELSE 'review_uncertain'
        END decision,
        'unvalidated_unsupervised' calibration_status,
        false approved_for_merge
        FROM pair_scores""")


def comparison_summary(connection, input_count):
    possible = input_count * (input_count - 1) // 2
    baseline = connection.execute(
        "SELECT count(*) FROM candidate_passes WHERE blocking_pass='exact_baseline'"
    ).fetchone()[0]
    candidate_count = connection.execute("SELECT count(*) FROM candidates").fetchone()[
        0
    ]
    predicted = connection.execute("SELECT count(*) FROM decisions").fetchone()[0]
    retained = connection.execute(
        "SELECT count(*) FROM decisions WHERE exact_baseline"
    ).fetchone()[0]
    if predicted != candidate_count or retained != baseline:
        raise AssertionError(
            f"Candidate/baseline conservation failed: {candidate_count}, {predicted}, {baseline}, {retained}"
        )
    groups = connection.execute(
        "SELECT count(DISTINCT baseline_id) FROM records"
    ).fetchone()[0]
    return {
        "input_records": input_count,
        "possible_pairs": possible,
        "candidate_pairs": candidate_count,
        "pair_reduction_fraction": 1 - candidate_count / possible,
        "exact_baseline_pairs": baseline,
        "exact_baseline_clusters": groups,
        "baseline_pairs_retained": retained,
        "baseline_pair_coverage": retained / baseline if baseline else None,
        "additional_proposed_pairs": connection.execute(
            "SELECT count(*) FROM decisions WHERE decision='proposed_match'"
        ).fetchone()[0],
        "review_conflict_pairs": connection.execute(
            "SELECT count(*) FROM decisions WHERE decision='review_conflict'"
        ).fetchone()[0],
        "review_uncertain_pairs": connection.execute(
            "SELECT count(*) FROM decisions WHERE decision='review_uncertain'"
        ).fetchone()[0],
        "likely_nonmatch_pairs": connection.execute(
            "SELECT count(*) FROM decisions WHERE decision='likely_nonmatch'"
        ).fetchone()[0],
        "precision": None,
        "recall": None,
        "evaluation_status": "no_reviewed_ground_truth_accuracy_not_measured",
    }
