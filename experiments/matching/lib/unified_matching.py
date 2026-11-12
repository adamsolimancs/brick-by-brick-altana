"""Splink fitting and exact saved-pair inference on cleaned comparison fields."""

import copy
import math

VERSION = "unified_cleaned_jw_fs_v5"
ADDRESS_NON_MATCH = "non_match_address_conflict"
SEED = 20261101
THRESHOLDS = {
    "company": [0.95, 0.85, 0.70],
    "street_name": [0.95, 0.85, 0.70],
    "city": [0.95, 0.85],
}
PROBABILITY_FLOOR = 1e-6
MAX_EM_ITERATIONS = 100
# What a pair decision authorizes. The resolver merges only `exact_rule` and
# `model_proposal` pairs, and re-checks the model-proposal safeguards itself.
# A human verdict (`fs_human_decision`) is a separate column that scoring
# never fills; nothing consumes it yet.
MERGE_AUTHORITY = {
    "exact_baseline": "exact_rule",
    "proposed_match": "model_proposal",
}
NO_MERGE_AUTHORITY = "none"


def merge_authority(decision):
    """Spark column: the merge authority implied by an fs_decision column."""
    from pyspark.sql import functions as F

    column = F.lit(NO_MERGE_AUTHORITY)
    for name, authority in MERGE_AUTHORITY.items():
        column = F.when(decision == name, authority).otherwise(column)
    return column


def register_jaro_winkler(connection):
    """Use the same Unicode-aware RapidFuzz implementation as saved features."""
    import pyarrow as pa
    from rapidfuzz.distance import JaroWinkler

    def similarity(left, right):
        return pa.array(
            [
                None
                if a is None or b is None or not a.strip() or not b.strip()
                else JaroWinkler.normalized_similarity(a, b, prefix_weight=0.1)
                for a, b in zip(left.to_pylist(), right.to_pylist())
            ],
            type=pa.float64(),
        )

    connection.create_function(
        "clean_jaro_winkler",
        similarity,
        ["VARCHAR", "VARCHAR"],
        "DOUBLE",
        type="arrow",
        null_handling="special",
    )


def model_settings(prior):
    from splink import SettingsCreator
    import splink.comparison_library as cl

    settings = SettingsCreator(
        link_type="dedupe_only",
        unique_id_column_name="unique_id",
        comparisons=[
            cl.JaroWinklerAtThresholds(name, thresholds).configure(
                term_frequency_adjustments=name == "company"
            )
            for name, thresholds in THRESHOLDS.items()
        ],
        blocking_rules_to_generate_predictions=[],
        probability_two_random_records_match=prior,
        retain_matching_columns=True,
        retain_intermediate_calculation_columns=True,
        max_iterations=MAX_EM_ITERATIONS,
        em_convergence=0.0001,
    ).create_settings_dict("duckdb")
    for comparison in settings["comparisons"]:
        for level in comparison["comparison_levels"]:
            level["sql_condition"] = level["sql_condition"].replace(
                "jaro_winkler_similarity(", "clean_jaro_winkler("
            )
    return settings


def regularize_model(model):
    """Make zero/unobserved tiers explicit and finite; never hide library defaults."""
    result = copy.deepcopy(model)
    result.pop("linker_uid", None)
    unsupported, adjusted = [], []
    for comparison in result["comparisons"]:
        levels = [
            level
            for level in comparison["comparison_levels"]
            if not level.get("is_null_level")
        ]
        for parameter in ("m_probability", "u_probability"):
            values = []
            observed = 0
            for index, level in enumerate(levels):
                value = level.get(parameter)
                known = isinstance(value, (int, float)) and not isinstance(value, bool)
                if known:
                    if not math.isfinite(value) or not 0 <= value <= 1:
                        raise ValueError(
                            f"Invalid fitted parameter: {comparison['output_column_name']}, {parameter}"
                        )
                    observed += 1
                else:
                    unsupported.append(
                        {
                            "comparison": comparison["output_column_name"],
                            "gamma": len(levels) - index - 1,
                            "parameter": parameter,
                            "label": level["label_for_charts"],
                        }
                    )
                    value = 0.0
                if value < PROBABILITY_FLOOR:
                    adjusted.append(
                        {
                            "comparison": comparison["output_column_name"],
                            "gamma": len(levels) - index - 1,
                            "parameter": parameter,
                            "before": value,
                            "floor": PROBABILITY_FLOOR,
                        }
                    )
                values.append(max(value, PROBABILITY_FLOOR))
            if not observed:
                raise ValueError(
                    f"Entire comparison lacks trained {parameter}: {comparison['output_column_name']}"
                )
            total = sum(values)
            for level, value in zip(levels, values):
                level[parameter] = value / total
    return result, {
        "probability_floor_before_normalization": PROBABILITY_FLOOR,
        "unobserved_parameters": unsupported,
        "floored_parameters": adjusted,
    }


def iteration_summary(history, limit=MAX_EM_ITERATIONS):
    """Splink emits one history row per parameter per iteration, not per step."""
    iterations = {int(record.iteration) for record in history}
    if not iterations or iterations != set(range(max(iterations) + 1)):
        raise ValueError("Missing or discontinuous EM iteration history")
    last_iteration = max(iterations)
    return {
        "history_records": len(history),
        "em_iterations": last_iteration,
        "iteration_cap_reached": last_iteration >= limit,
    }


def fit_cleaned_model(connection, baseline_pairs, record_count):
    """Estimate u globally and m by two bounded EM sessions, independent of HNSW."""
    from splink import DuckDBAPI, Linker

    possible = record_count * (record_count - 1) // 2
    if possible < 1:
        raise ValueError("At least two records are required")
    prior = min(0.1, max(baseline_pairs / possible, 1 / possible))
    api = DuckDBAPI(connection)
    linker = Linker(api.register("clean_records"), model_settings(prior))
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000, seed=SEED)
    sessions = []
    for column in ("company", "street_name"):
        rule = f"l.country = r.country AND l.{column} = r.{column} AND l.{column} IS NOT NULL"
        count = connection.execute(
            f"SELECT count(*) FROM clean_records l JOIN clean_records r ON {rule} AND l.unique_id < r.unique_id"
        ).fetchone()[0]
        if count < 10 or count > 200_000:
            raise ValueError(
                f"EM training block {column} has {count} pairs; expected 10–200,000"
            )
        session = linker.training.estimate_parameters_using_expectation_maximisation(
            rule,
            max_pairs=200_000,
            populate_probability_two_random_records_match_from_trained_values=False,
        )
        history = session._iteration_history_records
        sessions.append(
            {
                "blocking_column": column,
                "available_pairs": count,
                **iteration_summary(history),
            }
        )
    fitted = linker.misc.save_model_to_json()
    model, regularization = regularize_model(fitted)
    scorer = Linker(api.register("clean_records"), model)
    return (
        scorer,
        model,
        {
            "prior": prior,
            "possible_pairs": possible,
            "prior_source": "assumed_observed_exact_pair_rate_floor_one_pair",
            "u_sample_max_pairs": 1_000_000,
            "seed": SEED,
            "training_sessions": sessions,
            **regularization,
        },
    )


def predict_saved_pairs(connection, linker, max_pairs=2_000_000):
    """Score only saved_id_pairs, including pairs outside every training block."""
    from splink import DuckDBAPI

    count = connection.execute("SELECT count(*) FROM saved_id_pairs").fetchone()[0]
    if not 1 <= count <= max_pairs:
        raise ValueError(f"Invalid saved candidate count: {count}")
    invalid = connection.execute(
        "SELECT count(*) FROM saved_id_pairs WHERE join_key_l IS NULL OR join_key_r IS NULL OR join_key_l >= join_key_r"
    ).fetchone()[0]
    duplicates = connection.execute(
        "SELECT 1 FROM saved_id_pairs GROUP BY join_key_l,join_key_r HAVING count(*)>1 LIMIT 1"
    ).fetchone()
    missing = connection.execute(
        "SELECT count(*) FROM saved_id_pairs p LEFT JOIN clean_records l ON l.unique_id=p.join_key_l LEFT JOIN clean_records r ON r.unique_id=p.join_key_r WHERE l.unique_id IS NULL OR r.unique_id IS NULL"
    ).fetchone()[0]
    if invalid or duplicates or missing:
        raise ValueError("Invalid, duplicate, or missing-source saved candidate IDs")
    linker.table_management.register_blocked_pairs_for_predict(
        DuckDBAPI(connection).register("saved_id_pairs")
    )
    predictions = linker.inference.predict(threshold_match_probability=0)
    connection.execute(
        f"CREATE VIEW unified_predictions AS SELECT * FROM {predictions.physical_name}"
    )
    observed = connection.execute(
        "SELECT count(*) FROM unified_predictions"
    ).fetchone()[0]
    difference = connection.execute(
        "SELECT join_key_l,join_key_r FROM saved_id_pairs EXCEPT SELECT unique_id_l,unique_id_r FROM unified_predictions"
    ).fetchone()
    if observed != count or difference:
        raise AssertionError("Splink did not preserve the exact saved candidate set")
    return predictions


def apply_pair_geography(frame, eligible):
    """Use a supplied country as pair evidence when independent guards agree."""
    from pyspark.sql import Window, functions as F

    from altana_matching_components import normalize_region

    left, right = F.upper(F.trim("country_l")), F.upper(F.trim("country_r"))
    inferred_l, inferred_r = (
        F.col("fs_country_inferred_l"),
        F.col("fs_country_inferred_r"),
    )
    mismatch = left.rlike("^[A-Z]{2}$") & right.rlike("^[A-Z]{2}$") & (left != right)
    donor = F.when(eligible & mismatch & inferred_l & ~inferred_r, right).when(
        eligible & mismatch & inferred_r & ~inferred_l, left
    )
    region_country = F.coalesce(donor, F.when(left == right, left))
    region_key = F.udf(normalize_region, "string")
    result = (
        frame.withColumn("_geo_donor", donor)
        .withColumn("fs_region_key_l", region_key(F.col("state_l"), region_country))
        .withColumn("fs_region_key_r", region_key(F.col("state_r"), region_country))
    )
    known_regions = (F.col("fs_region_key_l") != "") & (F.col("fs_region_key_r") != "")
    region_agreement = known_regions & (
        F.col("fs_region_key_l") == F.col("fs_region_key_r")
    )
    supported = F.col("_geo_donor").isNotNull() & region_agreement
    result = result.withColumn(
        "_geo_recipient",
        F.when(supported, F.when(inferred_l, F.col("id_l")).otherwise(F.col("id_r"))),
    )
    # Partition eligible votes by recipient and distribute other pairs by ID.
    votes = Window.partitionBy(
        "_geo_recipient",
        F.when(F.col("_geo_recipient").isNull(), F.col("id_l")),
    )
    vote = F.when(F.col("_geo_recipient").isNotNull(), F.col("_geo_donor"))
    agreed = F.min(vote).over(votes) == F.max(vote).over(votes)
    result = result.withColumn(
        "fs_corrected_country", F.when(supported & agreed, F.col("_geo_donor"))
    )
    corrected = F.col("fs_corrected_country").isNotNull()
    return (
        result.withColumn(
            "fs_country_corrected_record_id", F.when(corrected, F.col("_geo_recipient"))
        )
        .withColumn(
            "fs_country_donor_record_id",
            F.when(
                corrected, F.when(inferred_l, F.col("id_r")).otherwise(F.col("id_l"))
            ),
        )
        .withColumn(
            "fs_country_conflict",
            F.coalesce(F.col("country_conflict"), F.lit(False)) & ~corrected,
        )
        .withColumn(
            "fs_region_conflict",
            F.when(corrected, F.lit(False))
            .when((left == right) & known_regions, ~region_agreement)
            .otherwise(F.coalesce(F.col("region_conflict"), F.lit(False))),
        )
        .drop("_geo_recipient", "_geo_donor")
    )


def route_address_conflicts(frame):
    """Close reliable number disagreements and keep narrow typo exceptions reviewable."""
    from pyspark.sql import functions as F

    def normalize_number(value):
        return F.regexp_replace(value, r"(?<![0-9])0+(?=[0-9])", "")

    def parsed(component, side):
        return F.col(f"parsed_{component}_{side}")

    def single(component, side):
        values = parsed(component, side)
        return F.when(F.size(values) == 1, F.get(values, 0))

    def same(component):
        return F.array_sort(parsed(component, "l")).eqNullSafe(
            F.array_sort(parsed(component, "r"))
        )

    def possible_digit_omission(left, right):
        return (
            left.rlike(r"^[0-9]+(?:/[0-9]+)?$")
            & right.rlike(r"^[0-9]+(?:/[0-9]+)?$")
            & (F.abs(F.length(left) - F.length(right)) == 1)
            & (F.levenshtein(left, right) == 1)
        )

    reliable = F.lit(True)
    houses, units, explicit_suites = [], [], []
    for side in ("l", "r"):
        warnings = F.col(f"street_parse_warnings_{side}")
        reliable = reliable & (
            F.size(
                F.array_except(
                    warnings,
                    F.array(
                        F.lit("road_number_tokens_retained"),
                        F.lit("multiple_units"),
                        F.lit("multiple_buildings"),
                        F.lit("multiple_floors"),
                    ),
                )
            )
            == 0
        )
        houses.append(normalize_number(single("house_numbers", side)))
        units.append(normalize_number(single("units", side)))
        suite = F.regexp_extract(
            F.lower(F.trim(F.col(f"street_address_{side}"))),
            r"\b(?:suite|ste)\.?\s*([0-9]+[a-z]?)\s*$",
            1,
        )
        explicit_suites.append(
            F.when(
                (suite != "") & F.array_contains(parsed("units", side), suite),
                normalize_number(suite),
            )
        )

    house_l, house_r = houses
    explicit_l, explicit_r = explicit_suites
    both_explicit = explicit_l.isNotNull() & explicit_r.isNotNull()
    unit_l, unit_r = (
        F.when(both_explicit, explicit).otherwise(unit)
        for explicit, unit in zip(explicit_suites, units)
    )
    same_number_shape = F.regexp_replace(house_l, "[0-9a-z]", "") == F.regexp_replace(
        house_r, "[0-9a-z]", ""
    )
    house_pattern = r"^[0-9]+[a-z]?(?:/[0-9]+[a-z]?)?$"
    house_diff = F.coalesce(
        reliable
        & house_l.rlike(house_pattern)
        & house_r.rlike(house_pattern)
        & same_number_shape
        & (house_l != house_r),
        F.lit(False),
    )
    unit_diff = F.coalesce(
        reliable
        & unit_l.rlike("^[0-9a-z]+$")
        & unit_r.rlike("^[0-9a-z]+$")
        & (unit_l != unit_r),
        F.lit(False),
    )
    very_close = (
        (F.col("fs_match_probability") >= 0.99)
        & (F.col("company_jaro_winkler_similarity") >= 0.95)
        & (F.col("street_name_jaro_winkler_similarity") >= 0.95)
        & (F.col("city_jaro_winkler_similarity") == 1)
        & (F.upper(F.trim("country_l")) == F.upper(F.trim("country_r")))
        & same("buildings")
        & same("floors")
    )
    house_typo = F.coalesce(
        house_diff
        & ~unit_diff
        & same("units")
        & very_close
        & possible_digit_omission(house_l, house_r),
        F.lit(False),
    )
    unit_typo = F.coalesce(
        unit_diff
        & ~house_diff
        & house_l.isNotNull()
        & (house_l == house_r)
        & (F.size(parsed("units", "l")) == 1)
        & (F.size(parsed("units", "r")) == 1)
        & very_close
        & possible_digit_omission(unit_l, unit_r),
        F.lit(False),
    )
    reviewable = F.col("fs_decision").isin("review_conflict", "review_uncertain")
    terminal = reviewable & (house_diff | unit_diff) & ~(house_typo | unit_typo)
    numeric_flag = F.lit(False)
    for name in (
        "house_conflict",
        "unit_conflict",
        "numeric_conflict",
        "parsed_house_numbers_conflict",
        "parsed_units_conflict",
    ):
        numeric_flag = numeric_flag | F.coalesce(F.col(name), F.lit(False))
    reasons = F.filter(
        F.array(
            F.when(terminal & house_diff, "different_house_numbers"),
            F.when(terminal & unit_diff, "different_suite_numbers"),
            F.when(reviewable & house_typo, "possible_house_number_typo"),
            F.when(reviewable & unit_typo, "possible_suite_number_typo"),
            F.when(
                reviewable & numeric_flag & ~house_diff & ~unit_diff,
                "ambiguous_address_number_evidence",
            ),
        ),
        lambda reason: reason.isNotNull(),
    )
    return frame.withColumn("fs_address_conflict_reasons", reasons).withColumn(
        "fs_decision",
        F.when(terminal, ADDRESS_NON_MATCH)
        .when(reviewable & (house_typo | unit_typo), "review_uncertain")
        .otherwise(F.col("fs_decision")),
    )


def add_decisions(frame, training, *, pair_geography=False):
    """Preserve historical evidence; new probabilities cannot override guards."""
    from pyspark.sql import functions as F

    def flag(name):
        return F.coalesce(F.col(name), F.lit(False))

    conflict = F.lit(False)
    for name in (
        "house_conflict",
        "unit_conflict",
        "numeric_conflict",
        "parsed_address_conflict",
    ):
        conflict = conflict | flag(name)
    untrained = F.lit(False)
    for tier in training["unobserved_parameters"]:
        untrained = untrained | (
            F.col("fs_gamma_" + tier["comparison"]) == tier["gamma"]
        )
    review = (
        flag("parsed_address_review_required")
        | flag("unit_missing_one_side")
        | untrained
        | F.lit(
            any(
                s["iteration_cap_reached"]
                for s in training.get("training_sessions", [])
            )
        )
    )

    def similar(name, threshold):
        return F.coalesce(
            F.col(f"{name}_jaro_winkler_similarity") >= threshold, F.lit(False)
        )

    strong = F.lit(True)
    for name in THRESHOLDS:
        strong = strong & similar(name, 0.85)
    # Translation renders the same source text differently across records
    # ("森 Street" as Mori Street and Forest St). When the text before
    # translation agrees, that field is settled; the remaining fields must
    # still meet the proposal thresholds and the company must be near-exact.
    original = {
        name: flag(f"original_{name}_agreement")
        if f"original_{name}_agreement" in frame.columns
        else F.lit(False)
        for name in THRESHOLDS
    }
    original_text = F.lit(False)
    for value in original.values():
        original_text = original_text | value
    original_text = original_text & (original["company"] | similar("company", 0.95))
    for name in THRESHOLDS:
        original_text = original_text & (original[name] | similar(name, 0.85))
    probability = F.col("fs_match_probability")
    if pair_geography:
        frame = apply_pair_geography(
            frame,
            (((probability >= 0.99) & strong) | original_text) & ~conflict & ~review,
        )
        conflict = conflict | flag("fs_country_conflict") | flag("fs_region_conflict")
    else:
        conflict = conflict | flag("country_conflict") | flag("region_conflict")
    result = (
        frame.withColumn("fs_guard_conflict", conflict)
        .withColumn("fs_guard_review", review)
        .withColumn("fs_uses_untrained_level", untrained)
        .withColumn("fs_original_text_agreement", original_text & ~conflict & ~review)
    )
    decision = (
        F.when(flag("source_exact_baseline"), "exact_baseline")
        .when(conflict, "review_conflict")
        .when(
            (((probability >= 0.99) & strong) | original_text) & ~review,
            "proposed_match",
        )
        .when(review, "review_uncertain")
        .when(probability <= 0.10, "likely_nonmatch")
        .otherwise("review_uncertain")
    )
    # Merge authority is explicit so the pair table does not look like a
    # review-only output. Scoring never records a human decision: that column
    # is reserved for a future reviewer verdict and stays NULL here. The
    # resolver's merge policy accepts `exact_rule` and `model_proposal` pairs
    # (after re-checking the safeguards); everything else is review-only.
    result = (
        result.withColumn("fs_decision", decision)
        .withColumn("fs_merge_authority", merge_authority(F.col("fs_decision")))
        .withColumn("fs_human_decision", F.lit(None).cast("string"))
        .withColumn("fs_calibration_status", F.lit("unvalidated_unsupervised"))
    )
    prior = training["prior"]
    for factor in (2, 4):
        alternative = min(prior * factor, 0.5)
        delta = math.log2(alternative / (1 - alternative)) - math.log2(
            prior / (1 - prior)
        )
        weight = F.col("fs_match_weight") + F.lit(delta)
        p = F.when(weight >= 0, 1 / (1 + F.pow(F.lit(2.0), -weight))).otherwise(
            F.pow(F.lit(2.0), weight) / (1 + F.pow(F.lit(2.0), weight))
        )
        result = result.withColumn(f"fs_probability_prior_{factor}x", p)
    return result


def expected_gamma(name):
    """Independent check that Splink used exactly the saved comparison levels."""
    from pyspark.sql import functions as F

    left, right = F.col(f"comparison_{name}_l"), F.col(f"comparison_{name}_r")
    thresholds = THRESHOLDS[name]
    expression = F.when(
        (F.length(left) == 0) | (F.length(right) == 0) | left.isNull() | right.isNull(),
        -1,
    ).when(left == right, len(thresholds) + 1)
    for index, threshold in enumerate(thresholds):
        expression = expression.when(
            F.col(f"{name}_jaro_winkler_similarity") >= threshold,
            len(thresholds) - index,
        )
    return expression.otherwise(0)
