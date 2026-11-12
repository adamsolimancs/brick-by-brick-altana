"""Deterministic company/location resolution with direct, queryable match evidence."""

from collections import Counter, defaultdict
from itertools import combinations

from lib.matching import FIELDS, digest, prepare_record

VERSION = "company_location_resolution_v3"
SELECTION_RULE = "exact_variant_support_then_supplied_country_then_record_id"
SIMILARITIES = (
    "company_jaro_winkler_similarity",
    "street_name_jaro_winkler_similarity",
    "city_jaro_winkler_similarity",
)
GUARDS = (
    "fs_guard_conflict",
    "fs_guard_review",
    "fs_uses_untrained_level",
)
REVIEW_DECISIONS = ("review_conflict", "review_uncertain")
EVIDENCE_FIELDS = (
    "candidate_origin",
    "fs_match_probability",
    "fs_match_weight",
    *SIMILARITIES,
    "fs_mw_company",
    "fs_mw_tf_adj_company",
    "fs_mw_street_name",
    "fs_mw_city",
    "record_cosine",
    "address_cosine",
    "fs_corrected_country",
    "fs_country_corrected_record_id",
    "fs_country_donor_record_id",
    "blocking_passes",
)
EVIDENCE_SCHEMA = """array<struct<
    other_record_id:string, step:string, reason:string,
    candidate_origin:string, fs_match_probability:double, fs_match_weight:double,
    company_jaro_winkler_similarity:double,
    street_name_jaro_winkler_similarity:double, city_jaro_winkler_similarity:double,
    fs_mw_company:double, fs_mw_tf_adj_company:double,
    fs_mw_street_name:double, fs_mw_city:double,
    record_cosine:double, address_cosine:double,
    fs_corrected_country:string, fs_country_corrected_record_id:string,
    fs_country_donor_record_id:string, blocking_passes:string
>>"""
MAPPING_SCHEMA = f"""
    record_id string, entity_id string, canonical_record_id string,
    is_duplicate boolean, is_canonical boolean, deduplication_status string,
    cluster_record_count long, deduplication_steps array<string>,
    merge_authority string,
    deduplication_reasons array<string>, deduplication_evidence {EVIDENCE_SCHEMA},
    needs_match_review boolean,
    match_review_evidence array<struct<other_record_id:string, decision:string,
        fs_match_probability:double, reason_codes:array<string>>>,
    filled_field string, filled_from_record_id string
"""


EXACT_STEP = "exact_normalized_company_address"
ORIGINAL_TEXT_STEP = "original_text_agreement"
FILL_STEP = "incomplete_filled_from_resolved"
# The only dlq_incomplete reasons a resolved record can repair, and the field
# each one leaves blank. A row with any other or additional reason stays parked.
FILLABLE_REASONS = {
    "city:null_or_blank": "city",
    "street_address:null_or_blank": "street_address",
}
# Per similarity, the pair flag saying both records' text before translation
# agrees on that field. Scoring sets fs_original_text_agreement from these.
ORIGINAL_AGREEMENTS = {
    "company_jaro_winkler_similarity": "original_company_agreement",
    "street_name_jaro_winkler_similarity": "original_street_name_agreement",
    "city_jaro_winkler_similarity": "original_city_agreement",
}


def step(pair):
    if pair["fs_decision"] == "exact_baseline":
        return EXACT_STEP
    if pair.get("fs_original_text_agreement") is True:
        return ORIGINAL_TEXT_STEP
    if pair["candidate_origin"] == "cluster_completion":
        return "fellegi_sunter_cluster_completion"
    return (
        "fellegi_sunter_hnsw"
        if pair["candidate_origin"] == "embedding"
        else "fellegi_sunter_blocked"
    )


# What authorized a record's group. `exact_rule`: every direct proof is the
# deterministic exact rule. `model_proposal_policy`: at least one proof is a
# Fellegi-Sunter proposal that this resolver's policy accepted (probability,
# similarity and safeguard checks). `human_review` is reserved for reviewed
# decisions and is never emitted yet. `incomplete_fill_policy`: the only merges
# in the group are incomplete source rows folded in by attach_incomplete.
# `none`: singleton, nothing merged.
MERGE_AUTHORITIES = (
    "exact_rule",
    "model_proposal_policy",
    "human_review",
    "incomplete_fill_policy",
    "none",
)


def merge_authority(proofs):
    if not proofs:
        return "none"
    if all(proof["step"] == EXACT_STEP for proof in proofs):
        return "exact_rule"
    return "model_proposal_policy"


def evidence(pair, other_id):
    stage = step(pair)
    if stage == EXACT_STEP:
        reason = "Company, street, city, region and country agree after case and whitespace normalization."
    elif stage == ORIGINAL_TEXT_STEP:
        agreed = [
            name.split("_jaro_winkler")[0]
            for name, agreement in ORIGINAL_AGREEMENTS.items()
            if pair.get(agreement) is True
        ]
        reason = (
            f"Text before translation agrees on {', '.join(agreed)}; the other "
            "fields met 0.85 Jaro-Winkler, the company was near-exact or agreed "
            "before translation, and address, geography and model safeguards passed."
        )
    else:
        reason = (
            f"Fellegi-Sunter score {pair['fs_match_probability']:.6f} met 0.99; "
            "company, street and city Jaro-Winkler similarities each met 0.85; "
            "address, geography and model safeguards passed."
        )
        if pair.get("fs_corrected_country"):
            reason += " An inferred country was reconciled with a supplied country using the scoring geography rule."
    result = {name: pair.get(name) for name in EVIDENCE_FIELDS}
    # Historical candidates can store blocking passes as an array or JSON string.
    if isinstance(result["blocking_passes"], list):
        result["blocking_passes"] = ",".join(result["blocking_passes"])
    return dict(result, other_record_id=other_id, step=stage, reason=reason)


def review_codes(pair):
    codes = sorted(
        name
        for name, value in pair.items()
        if value is True
        and ("conflict" in name or "review" in name or "missing_one_side" in name)
    )
    codes.extend(pair.get("fs_address_conflict_reasons") or [])
    codes.extend(
        name + "_below_0.85" for name in SIMILARITIES if (pair.get(name) or 0) < 0.85
    )
    if pair["fs_match_probability"] < 0.99:
        codes.append("fellegi_sunter_below_0.99")
    return codes or ["scoring_review_required"]


def original_text_policy(pair):
    """Re-check scoring's original-text proposal from the pair's own evidence."""
    agreements = {
        name: pair.get(agreement) is True
        for name, agreement in ORIGINAL_AGREEMENTS.items()
    }
    if not any(agreements.values()):
        return False
    company = "company_jaro_winkler_similarity"
    if not agreements[company] and (pair.get(company) or 0) < 0.95:
        return False
    return all(
        agreements[name] or (pair.get(name) or 0) >= 0.85 for name in SIMILARITIES
    )


def resolve_records(records, pairs):
    """Require every cross-group pair to pass before joining fuzzy groups.

    This prevents an A-B and B-C match from silently absorbing an incompatible
    A-C pair. Every removed row consequently has a direct proof to its canonical.
    """
    by_id = {record["unique_id"]: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Resolution requires unique accepted source identities")
    groups = defaultdict(set)
    membership = {}
    for record in records:
        group = record["baseline_id"]
        groups[group].add(record["unique_id"])
        membership[record["unique_id"]] = group
    accepted = {}
    review = defaultdict(list)
    for pair in pairs:
        left, right = pair["id_l"], pair["id_r"]
        if left not in by_id or right not in by_id or left >= right:
            raise ValueError("Pair references an invalid or missing source identity")
        if pair["fs_decision"] in {"exact_baseline", "proposed_match"}:
            if pair["fs_decision"] == "exact_baseline":
                if by_id[left]["baseline_id"] != by_id[right]["baseline_id"]:
                    raise ValueError("Exact pair disagrees with the record baseline")
            elif any(pair.get(name) is not False for name in GUARDS):
                raise ValueError("Fuzzy pair bypassed scoring safeguards")
            elif pair.get("fs_original_text_agreement") is True:
                if not original_text_policy(pair):
                    raise ValueError("Fuzzy pair bypassed scoring safeguards")
            elif pair["fs_match_probability"] < 0.99 or any(
                (pair.get(name) or 0) < 0.85 for name in SIMILARITIES
            ):
                raise ValueError("Fuzzy pair bypassed scoring safeguards")
            if (left, right) in accepted:
                raise ValueError("Duplicate accepted pair")
            accepted[left, right] = pair
        elif (
            pair["fs_decision"] in REVIEW_DECISIONS
            and pair["fs_match_probability"] > 0.80
        ):
            for record_id, other in ((left, right), (right, left)):
                review[record_id].append(
                    dict(
                        other_record_id=other,
                        decision=pair["fs_decision"],
                        fs_match_probability=pair["fs_match_probability"],
                        reason_codes=review_codes(pair),
                    )
                )
    for members in groups.values():
        if any(tuple(sorted(ids)) not in accepted for ids in combinations(members, 2)):
            raise ValueError("Exact group is missing scored pair evidence")
    fuzzy = sorted(
        (pair for pair in accepted.values() if pair["fs_decision"] == "proposed_match"),
        key=lambda pair: (-pair["fs_match_probability"], pair["id_l"], pair["id_r"]),
    )
    for pair in fuzzy:
        left, right = pair["id_l"], pair["id_r"]
        left_group, right_group = membership[left], membership[right]
        if left_group == right_group:
            continue
        compatible = all(
            tuple(sorted((a, b))) in accepted
            for a in groups[left_group]
            for b in groups[right_group]
        )
        if not compatible:
            for record_id, other in ((left, right), (right, left)):
                review[record_id].append(
                    dict(
                        other_record_id=other,
                        decision="review_cluster_consistency",
                        fs_match_probability=pair["fs_match_probability"],
                        reason_codes=[
                            "every_cross_group_pair_must_have_passing_evidence"
                        ],
                    )
                )
            continue
        keep, remove = sorted((left_group, right_group))
        groups[keep].update(groups.pop(remove))
        for record_id in groups[keep]:
            membership[record_id] = keep
    result = []
    for members in groups.values():
        support = Counter(tuple(by_id[r].get(f) for f in FIELDS) for r in members)
        canonical = min(
            members,
            key=lambda r: (
                -support[tuple(by_id[r].get(f) for f in FIELDS)],
                str(by_id[r].get("country_inferred")).lower() != "false",
                r,
            ),
        )
        entity_id = digest([VERSION, sorted(members)])
        for record_id in sorted(members):
            others = (
                sorted(members - {record_id}) if record_id == canonical else [canonical]
            )
            proofs = [
                evidence(accepted[tuple(sorted((record_id, other)))], other)
                for other in others
            ]
            duplicate = record_id != canonical
            reasons = [proof["reason"] for proof in proofs]
            if record_id == canonical:
                reasons.insert(
                    0,
                    "Selected the most-supported complete source record, then preferred a supplied country and a stable record ID.",
                )
            if len(members) == 1:
                reasons = [
                    "Retained as a singleton because no compatible duplicate group was established."
                ]
            result.append(
                dict(
                    record_id=record_id,
                    entity_id=entity_id,
                    canonical_record_id=canonical,
                    is_duplicate=duplicate,
                    is_canonical=not duplicate,
                    deduplication_status="duplicate"
                    if duplicate
                    else "canonical"
                    if len(members) > 1
                    else "singleton",
                    cluster_record_count=len(members),
                    deduplication_steps=sorted({proof["step"] for proof in proofs}),
                    merge_authority=merge_authority(proofs),
                    deduplication_reasons=reasons,
                    deduplication_evidence=proofs,
                    needs_match_review=bool(review[record_id]),
                    match_review_evidence=sorted(
                        review[record_id],
                        key=lambda item: (item["other_record_id"], item["decision"]),
                    ),
                    filled_field=None,
                    filled_from_record_id=None,
                )
            )
    return result


def _blank(value):
    return not str(value or "").strip()


def _fill_reason(field):
    if field == "city":
        return (
            "City was blank at the source. One resolved company/location has the "
            "same company name key, country and full street address (house, road "
            "and unit) with no region conflict; the city was copied from its "
            "retained record and this row was folded into it."
        )
    return (
        "Street address was blank at the source. The company has exactly one "
        "resolved location in this city and country with no region conflict; "
        "the street address was copied from its retained record and this row "
        "was folded into it."
    )


def attach_incomplete(mapping, records, incomplete):
    """Fold fillable dlq_incomplete rows into the one resolved entity they fit.

    A row qualifies when its only defect is a blank city or a blank street
    address, its source identity is new and unique, and exactly one resolved
    entity of the same company (name key) and country carries the address part
    the row still has: the full street key for a blank city, the city key for a
    blank street. Regions must not conflict. The blank field is copied from
    that entity's canonical record, and the row joins the entity as a
    duplicate, so it can never become or change the Gold record. Everything
    else stays parked in dlq_incomplete.

    Returns the updated mapping and the completed records (prepare_record
    output plus filled_field and filled_from_record_id).
    """
    by_id = {record["unique_id"]: record for record in records}
    rows = {row["record_id"]: dict(row) for row in mapping}
    by_street = defaultdict(list)
    by_city = defaultdict(list)
    for record_id in rows:
        record = by_id[record_id]
        base = (record["country_key"], record["name_key"])
        by_street[base + (record["street_key"],)].append(record)
        by_city[base + (record["city_key"],)].append(record)
    identities = Counter(
        (str(row.get("provider")), str(row.get("provider_id"))) for row in incomplete
    )
    attached = defaultdict(list)
    for row in incomplete:
        reasons = list(row.get("dlq_reasons") or [])
        if len(reasons) != 1 or reasons[0] not in FILLABLE_REASONS:
            continue
        field = FILLABLE_REASONS[reasons[0]]
        present = "street_address" if field == "city" else "city"
        if any(_blank(row.get(name)) for name in ("provider", "provider_id")):
            continue
        if identities[str(row["provider"]), str(row["provider_id"])] > 1:
            continue
        if not _blank(row.get(field)) or any(
            _blank(row.get(name)) for name in ("company", "country", present)
        ):
            continue
        keys = prepare_record(row)
        if keys["unique_id"] in by_id or not keys["name_key"]:
            continue
        base = (keys["country_key"], keys["name_key"])
        if field == "city":
            if not keys["road_key"]:
                continue
            hits = by_street[base + (keys["street_key"],)]
        else:
            if not keys["city_key"]:
                continue
            hits = by_city[base + (keys["city_key"],)]
        hits = [
            hit
            for hit in hits
            if not keys["region_key"]
            or not hit["region_key"]
            or hit["region_key"] == keys["region_key"]
        ]
        entities = {rows[hit["unique_id"]]["entity_id"] for hit in hits}
        if len(entities) != 1:
            continue
        canonical = rows[hits[0]["unique_id"]]["canonical_record_id"]
        value = by_id[canonical].get(field)
        if _blank(value):
            continue
        completed = prepare_record(dict(row, **{field: value}))
        attached[canonical].append(
            dict(completed, filled_field=field, filled_from_record_id=canonical)
        )
    filled = []
    for canonical, completed_rows in attached.items():
        target = rows[canonical]
        entity_id = target["entity_id"]
        count = target["cluster_record_count"] + len(completed_rows)
        proofs = []
        for completed in sorted(completed_rows, key=lambda item: item["unique_id"]):
            reason = _fill_reason(completed["filled_field"])
            blank_evidence = {name: None for name in EVIDENCE_FIELDS}
            proofs.append(
                dict(
                    blank_evidence,
                    other_record_id=completed["unique_id"],
                    step=FILL_STEP,
                    reason=reason,
                )
            )
            rows[completed["unique_id"]] = dict(
                record_id=completed["unique_id"],
                entity_id=entity_id,
                canonical_record_id=canonical,
                is_duplicate=True,
                is_canonical=False,
                deduplication_status="duplicate",
                cluster_record_count=count,
                deduplication_steps=[FILL_STEP],
                merge_authority="incomplete_fill_policy",
                deduplication_reasons=[reason],
                deduplication_evidence=[
                    dict(
                        blank_evidence,
                        other_record_id=canonical,
                        step=FILL_STEP,
                        reason=reason,
                    )
                ],
                needs_match_review=False,
                match_review_evidence=[],
                filled_field=completed["filled_field"],
                filled_from_record_id=canonical,
            )
            filled.append(completed)
        for row in rows.values():
            if row["entity_id"] == entity_id:
                row["cluster_record_count"] = count
        reasons = target["deduplication_reasons"]
        if target["deduplication_status"] == "singleton":
            reasons = [
                "Retained as the only complete source record of its company and "
                "location; incomplete source records were folded into it."
            ]
            target["deduplication_status"] = "canonical"
        target["deduplication_reasons"] = reasons + [p["reason"] for p in proofs]
        target["deduplication_evidence"] = target["deduplication_evidence"] + proofs
        target["deduplication_steps"] = sorted(
            {*target["deduplication_steps"], FILL_STEP}
        )
        if target["merge_authority"] == "none":
            target["merge_authority"] = "incomplete_fill_policy"
    return list(rows.values()), filled
