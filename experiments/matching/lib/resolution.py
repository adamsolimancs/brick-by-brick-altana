"""Deterministic company/location resolution with direct, queryable match evidence."""

import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from itertools import combinations

from altana_matching_components import text_key

from lib.matching import FIELDS, digest, prepare_record

VERSION = "company_location_resolution_v6"
SELECTION_RULE = (
    "exact_variant_support_then_supplied_country_then_record_id;"
    "settled_groups_prefer_complete_address_then_group_size_then_city_usage"
)
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
SETTLE_STEP = "same_company_address_settled"
# Review decision written by settle_same_company on a same-company pair whose
# two addresses are clearly different places. Both stay separate Gold records
# and the pair no longer holds either of them.
DIFFERENT_ADDRESS = "same_company_different_address"
# How two records of one company relate, and the proof text for each merge.
SETTLED_RELATIONS = {
    "same_text": (
        "Same company name key and country, and the street and city text are "
        "identical once case and punctuation are removed. The matcher had held "
        "the pair on a parse it could not verify."
    ),
    "same_street_other_city": (
        "Same company name key and country, and the street text is identical "
        "once case and punctuation are removed; the city is the same name "
        "written differently (another language, word order or romanization)."
    ),
    "truncated": (
        "Same company name key and country, and one address is the other cut "
        "short (the cut does not fall inside a number). The complete address "
        "is the retained record."
    ),
    "street_spelling": (
        "Same company name key, country and city name, and the two streets "
        "carry the same numbers and building letters in the same order. The "
        "words differ only by dropped characters (at most two, none replaced), "
        "spacing, or an equivalent label such as Ste./Suite, St./Street or "
        "Piso/Floor."
    ),
    "same_original_text": (
        "Same company name key, country and city name, and the street text "
        "before translation is the same on both records (identical, or with "
        "the spelling differences above). The two English renderings differ "
        "only because they were translated separately."
    ),
    "judged_same_address": (
        "Same company name key and country, no region conflict, and the two "
        "streets carry the same numbers and building letters. The street or "
        "city words differ by romanization or translation, and the address "
        "judge model called them one address with confidence at or above 0.90."
    ),
    "missing_number": (
        "Same company name key, country and city name, and the same street "
        "words. One record lacks a house, unit or building number the other "
        "has; every number and building letter it does carry appears on the "
        "other in the same order, so nothing contradicts. The record with "
        "more numbers is the retained record."
    ),
}
# Reviewer hints on a held pair, never a decision: the two streets differ by
# one dropped or swapped digit, and one number is on records from two
# providers that agree while the other is on a single record.
NUMBER_SLIP = "possible_number_slip"
NUMBER_SLIP_OTHER_BACKED = "other_number_reported_by_two_providers"
NUMBER_SLIP_THIS_BACKED = "this_number_reported_by_two_providers"
# A judged pair merges only at or above this model confidence.
JUDGE_THRESHOLD = 0.90
# Words that label a unit, building or floor, or name a road type, mapped to
# one spelling as text_key renders them. Only used to compare two streets; no
# stored value is rewritten.
_LABEL_FORMS = {
    "unit": """apartment apt apartamento appartamento appartement appt apto
        suite ste unit unite unidad interior depto departamento""",
    "building": "building bldg edificio edif batiment bloco gebaude",
    "floor": "floor flr piso piano etage andar planta",
    "street": "street st",
    "road": "road rd",
    "avenue": "avenue ave",
    "boulevard": "boulevard blvd",
    "drive": "drive dr",
    "lane": "lane ln",
    "court": "court ct",
}
LABEL_WORDS = {
    form: label for label, forms in _LABEL_FORMS.items() for form in forms.split()
}
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
# `same_company_address_policy`: the record was merged by settle_same_company,
# or its group has no other kind of merge. `none`: singleton, nothing merged.
MERGE_AUTHORITIES = (
    "exact_rule",
    "model_proposal_policy",
    "human_review",
    "incomplete_fill_policy",
    "same_company_address_policy",
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


# City words that carry no name: directions, "new", settlement types, lake and
# port words, and connectors, as text_key renders them. A city is compared on
# the words that are left.
CITY_FILLER = frozenset(
    """
    north south east west norte sur sul sud nord est ost oeste ouest este leste
    new nueva nuevo nouveau nouvelle nova novo nuovo nuova neu
    city town village district county prefecture province municipality
    shi ku gun machi mura cho son si gu eup myeon dong
    lake see lago lac port puerto porto
    de do da di del la le el of the
    """.split()
)


def _same_city_name(left, right):
    """True when two differently written city keys still name the same city.

    Every name word of one side needs a counterpart on the other: the same
    word, one a prefix of the other, or a close spelling. "Sur de Marciaberg"
    and "South Marciaberg" pass; "Yokohama-shi Seya-ku" and "Sakae-ku,
    Yokohama" do not, because Seya and Sakae are different wards.
    """
    left = [word for word in left.split() if word not in CITY_FILLER]
    right = [word for word in right.split() if word not in CITY_FILLER]
    if not left or not right:
        return False

    def alike(word, other):
        short, long = sorted((word, other), key=len)
        return (len(short) >= 3 and long.startswith(short)) or (
            SequenceMatcher(None, word, other).ratio() >= 0.7
        )

    def covered(words, others):
        return all(any(alike(word, other) for other in others) for word in words)

    return covered(left, right) or covered(right, left)


def _cut_short(short, long):
    """True when `short` is `long` cut off, but not in the middle of a number.

    "44524 jackson drives su" is "44524 jackson drives suite 713" cut short.
    "stretto tomaselli 2" against "stretto tomaselli 229" is not accepted: the
    cut would change a house number, and a different number is a different
    place until someone says otherwise.
    """
    if not short or len(short) >= len(long) or not long.startswith(short):
        return False
    return not short[-1].isdigit() or long[len(short)] == " "


def _text_relation(left, right):
    if left == right:
        return "same"
    short, long = sorted((left, right), key=len)
    return "cut_short" if _cut_short(short, long) else "differs"


def _street_parts(street):
    """Split a street into numbers, single building letters and words.

    Letters and digits written together are separated first, so "Platz1-3"
    and "Platz 1-3" give the same parts. "#" is read as a unit label.
    """
    key = text_key(str(street or "").replace("#", " unit ")).replace("ß", "ss")
    tokens = re.findall(r"\d+|[^\W\d_]+", key)
    numbers = [token for token in tokens if token.isdigit()]
    letters = sorted(
        token
        for token in tokens
        if len(token) == 1 and token.isascii() and not token.isdigit()
    )
    words = [
        token
        for token in tokens
        if not token.isdigit() and not (len(token) == 1 and token.isascii())
    ]
    return numbers, letters, words


def _dropped(word, other):
    """Characters one spelling of a word lacks, or None if it is another word.

    The shorter spelling must be the longer one with characters left out:
    "sute" against "suite" is 1. "roma" against "rosa" and "main" against
    "mann" are None, because a replaced character can be another name.
    """
    short, long = sorted((word, other), key=len)
    remaining = iter(long)
    if all(char in remaining for char in short):
        return len(long) - len(short)
    return None


def _same_identifiers(left, right):
    """Both streets carry the same numbers in order and the same letters."""
    left, right = _street_parts(left), _street_parts(right)
    return bool(left[0] or left[1]) and left[:2] == right[:2]


def _spelling_relation(left, right):
    """ "format", "typo" or None for two streets with different text keys.

    Numbers must be identical and in the same order, and single building
    letters must be the same set: nothing here ever bridges a number. The
    words may then differ by spacing or an equivalent label ("format"), or by
    at most two dropped characters in total with none replaced ("typo"): each
    word is its counterpart with characters left out.
    """
    if not _same_identifiers(left, right):
        return None
    left, right = _street_parts(left)[2], _street_parts(right)[2]
    if not left or not right:
        return None
    if len(left) == len(right):
        dropped = 0
        for word, other in zip(left, right):
            if word == other or LABEL_WORDS.get(word, word) == LABEL_WORDS.get(
                other, other
            ):
                continue
            missing = _dropped(word, other)
            if missing is None or min(len(word), len(other)) < 2:
                dropped = None
                break
            dropped += missing
        if dropped is not None and dropped <= 2:
            return "typo" if dropped else "format"
    # Words run together or split differently: the same letters without spaces.
    return "format" if "".join(left) == "".join(right) else None


def _same_word(word, other):
    return word == other or LABEL_WORDS.get(word, word) == LABEL_WORDS.get(other, other)


def _missing_number(short, long):
    """True when street `short` is `long` without some of its numbers.

    `short` has fewer numbers, and the ones it has appear on `long` in the
    same order; it has no building letter `long` lacks; and its words are
    `long`'s words in order, where `long` may carry extra label words (the
    "Suite" of a unit number `short` does not have); and no missing number
    begins like a number `short` has. "Brooks Suite 115"
    against "999 Brooks Suite 115" passes. "41 # 56" against "41 A" does not:
    each side has something the other lacks.
    """
    short, long = _street_parts(short), _street_parts(long)
    if len(short[0]) >= len(long[0]) or not short[2]:
        return False
    remaining = iter(long[0])
    if not all(number in remaining for number in short[0]):
        return False
    if Counter(short[1]) - Counter(long[1]):
        return False
    # "Tomaselli, 2" against "Tomaselli, 229 Piano 2": the 2 may be 229 cut
    # short, so a missing number that begins like a present one is not read
    # as missing.
    missing = Counter(long[0]) - Counter(short[0])
    if any(
        extra.startswith(number) or number.startswith(extra)
        for extra in missing
        for number in short[0]
    ):
        return False
    words = list(short[2])
    for word in long[2]:
        if words and _same_word(words[0], word):
            words.pop(0)
        elif word not in LABEL_WORDS:
            return False
    return not words


def _number_slip(number, other):
    """One number is the other with one digit dropped or two neighbours swapped."""
    short, long = sorted((number, other), key=len)
    if len(long) - len(short) == 1:
        return any(long[:i] + long[i + 1 :] == short for i in range(len(long)))
    return number != other and any(
        number[:i] + number[i + 1] + number[i] + number[i + 2 :] == other
        for i in range(len(number) - 1)
    )


def _original_street(record):
    """The street as the provider sent it, before any translation."""
    evidence = record.get("translation_evidence_json")
    if evidence:
        try:
            original = json.loads(evidence).get("street_address_original")
        except (TypeError, ValueError):
            original = None
        if original:
            return original
    return record.get("street_address")


def _street_relation(left, right):
    """How two records' streets relate: the first of these that holds.

    "same", "cut_short", "format", "typo" on the Silver text; "original" when
    the text before translation agrees in one of those ways; "fewer_numbers"
    when one street is the other without some of its numbers; else "differs".
    """
    streets = [record.get("street_address") for record in (left, right)]
    relation = _text_relation(*map(text_key, streets))
    if relation != "differs":
        return relation
    relation = _spelling_relation(*streets)
    if relation:
        return relation
    originals = [_original_street(record) for record in (left, right)]
    if originals != streets and (
        text_key(originals[0]) == text_key(originals[1])
        or _spelling_relation(*originals)
    ):
        return "original"
    if _missing_number(*streets) or _missing_number(*streets[::-1]):
        return "fewer_numbers"
    return "differs"


def _city_relation(left, right):
    relation = _text_relation(left["city_key"], right["city_key"])
    if relation == "differs" and _same_city_name(left["city_key"], right["city_key"]):
        return "same_name"
    return relation


def _region_conflict(left, right):
    return bool(
        left["region_key"]
        and right["region_key"]
        and left["region_key"] != right["region_key"]
    )


def pair_key(left_id, right_id):
    return tuple(sorted((left_id, right_id)))


def _identifier(token):
    """House, unit, floor and building identifiers: numbers and single letters."""
    return any(c.isdigit() for c in token) or (len(token) == 1 and token.isascii())


def _clearly_different_street(left, right):
    """No identifier in common and, on each side, a word the other side lacks.

    Deliberately narrow. The same words with another number may be a typo, a
    move along the street or a second office; other words around a shared
    number are usually one address written in two languages or formats.
    Neither is decided here.
    """
    left, right = left.split(), right.split()
    left_ids = set(filter(_identifier, left))
    right_ids = set(filter(_identifier, right))
    if not left_ids or not right_ids or left_ids & right_ids:
        return False
    left = [token for token in left if not _identifier(token)]
    right = [token for token in right if not _identifier(token)]

    def has_unmatched_word(words, others):
        return any(
            all(SequenceMatcher(None, word, other).ratio() < 0.75 for other in others)
            for word in words
        )

    return has_unmatched_word(left, right) and has_unmatched_word(right, left)


def _rule_relation(left, right):
    """What the address text alone says about two same-company records.

    A key of SETTLED_RELATIONS, DIFFERENT_ADDRESS, or None when the text does
    not decide. The street and the city are compared separately and both must
    agree, so a slightly different street in a differently written city is
    still one place.
    """
    if not left["name_key"] or left["name_key"] != right["name_key"]:
        return None
    if left["country_key"] != right["country_key"]:
        return None
    region_conflict = _region_conflict(left, right)
    street = _street_relation(left, right)
    city = _city_relation(left, right)
    if street == "differs":
        streets = [text_key(record.get("street_address")) for record in (left, right)]
        if (region_conflict or city == "differs") and _clearly_different_street(
            *streets
        ):
            return DIFFERENT_ADDRESS
        return None
    if region_conflict or city == "differs":
        return None
    if street == "same":
        return {
            "same": "same_text",
            "cut_short": "truncated",
            "same_name": "same_street_other_city",
        }[city]
    return {
        "cut_short": "truncated",
        "original": "same_original_text",
        "fewer_numbers": "missing_number",
    }.get(street, "street_spelling")


def _number_slip_pair(left, right):
    """Two same-company records whose streets differ by one slipped number.

    Same name key, country and city name, no region conflict, the same
    building letters and street words (as the spelling rule reads them), and
    the same numbers except one, which differs by a dropped digit or two
    swapped digits. This decides nothing: 704 and 7045 can be two buildings,
    and no data says which number is right. settle_same_company only flags
    such a pair for the reviewer (NUMBER_SLIP_CODES) when one number is backed
    by two providers and the other by a single record; the pair stays on hold.
    """
    if not left["name_key"] or left["name_key"] != right["name_key"]:
        return False
    if left["country_key"] != right["country_key"] or _region_conflict(left, right):
        return False
    if _city_relation(left, right) == "differs":
        return False
    left, right = (
        _street_parts(record.get("street_address")) for record in (left, right)
    )
    if left[1] != right[1] or len(left[0]) != len(right[0]) or not left[2]:
        return False
    differing = [(a, b) for a, b in zip(left[0], right[0]) if a != b]
    if len(differing) != 1 or not _number_slip(*differing[0]):
        return False
    return left[2] == right[2] or (
        len(left[2]) == len(right[2])
        and all(_same_word(a, b) for a, b in zip(left[2], right[2]))
    )


def judgeable(left, right):
    """True when only an address judge can say whether two records are one place.

    Same company name key and country, no region conflict, the same street
    numbers and building letters, and street or city words that no rule
    reconciles (romanization, translation, a missing label). A pair with a
    different number or region is never judgeable.
    """
    if not left["name_key"] or left["name_key"] != right["name_key"]:
        return False
    if left["country_key"] != right["country_key"] or _region_conflict(left, right):
        return False
    if _rule_relation(left, right) is not None:
        return False
    return _same_identifiers(left.get("street_address"), right.get("street_address"))


def same_company_relation(left, right, judgments=None):
    """How two records that carry the same company name relate.

    Returns a key of SETTLED_RELATIONS when they are the same place,
    DIFFERENT_ADDRESS when they are clearly two places (a clearly different
    street in another city or region), and None when the data does not say.
    None also covers a different company name key or country.

    `judgments` maps pair_key(unique ids) to an address judge result
    (same_address, confidence). It is consulted only for a judgeable pair,
    and only a "same" answer at JUDGE_THRESHOLD or above settles the pair.
    """
    relation = _rule_relation(left, right)
    if relation is not None or not judgments:
        return relation
    judgment = judgments.get(pair_key(left["unique_id"], right["unique_id"]))
    if _accepted(judgment) and judgeable(left, right):
        return "judged_same_address"
    return None


def _less_complete(record, other):
    """True when `record` is a shortened form of `other`.

    Shortened means the street or city is cut short, the street lacks numbers
    the other has, or the street has the same parts with characters dropped. The fuller spelling is the better
    record to keep.
    """
    streets = [text_key(item.get("street_address")) for item in (record, other)]
    if _cut_short(*streets) or _cut_short(record["city_key"], other["city_key"]):
        return True
    if _missing_number(record.get("street_address"), other.get("street_address")):
        return True
    if (
        _spelling_relation(record.get("street_address"), other.get("street_address"))
        != "typo"
    ):
        return False
    words = [
        "".join(_street_parts(item.get("street_address"))[2])
        for item in (record, other)
    ]
    return len(words[0]) < len(words[1])


def judgment_candidates(mapping, records):
    """Record pairs an address judge must decide before entities can merge.

    Entities linked by same-company review pairs form groups, and entities
    merge only as a whole, so every judgeable pair across two entities of a
    group is returned, not only the pairs the matcher reviewed. Returns
    sorted (left record, right record) tuples in pair_key order.
    """
    by_id = {record["unique_id"]: record for record in records}
    entity = {row["record_id"]: row["entity_id"] for row in mapping}
    members = defaultdict(set)
    for record_id, entity_id in entity.items():
        members[entity_id].add(record_id)
    links = defaultdict(set)
    for row in mapping:
        for item in row["match_review_evidence"]:
            other = entity[item["other_record_id"]]
            if other != row["entity_id"]:
                links[row["entity_id"]].add(other)
                links[other].add(row["entity_id"])
    pairs, seen = set(), set()
    for start in sorted(links):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            current = stack.pop()
            if current not in component:
                component.add(current)
                stack.extend(links[current])
        seen |= component
        for first, second in combinations(sorted(component), 2):
            for left in members[first]:
                for right in members[second]:
                    if judgeable(by_id[left], by_id[right]):
                        pairs.add(pair_key(left, right))
    return [(by_id[left], by_id[right]) for left, right in sorted(pairs)]


def settle_same_company(mapping, records, judgments=None):
    """Decide the same-company review pairs the address text itself settles.

    A review pair whose two records carry the same company name key is one of
    three things (see same_company_relation):

    - the same place written twice (identical text, the same city name written
      differently, one address cut short or missing a number the other has,
      a street that differs only by dropped characters, spacing or an
      equivalent label, the same text before
      translation, or a romanization the address judge accepted at 0.90 or
      more): the two entities are merged;
    - clearly two places: the pair is marked DIFFERENT_ADDRESS, so both stay
      separate Gold records and neither is held for it;
    - undecided (a different number, different words the judge did not
      accept): left as it is.

    `judgments` maps pair_key(record ids) to an address judge result with
    same_address, confidence, reason and model (see judgment_candidates).

    Entities merge only as a whole. Every record of one must be the same place
    as every record of the other, and when a chain of such pairs links entities
    that are not all the same place (one short address that fits two different
    suites), none of them is merged. The retained record is one of the existing
    canonical records: the one that is not a shortened form of another (cut
    short, or the same street with characters dropped), then the one
    with the larger group, then the one whose city spelling more records
    outside the group use in the same country, then a supplied country and a
    stable record ID.
    """
    by_id = {record["unique_id"]: record for record in records}
    rows = {row["record_id"]: dict(row) for row in mapping}
    members = defaultdict(set)
    for row in rows.values():
        members[row["entity_id"]].add(row["record_id"])

    def city(record):
        # Country only: the feeds pair one city with many regions.
        return record["country_key"], record["city_key"]

    city_usage = Counter(city(record) for record in records)

    judgments = judgments or {}

    def relation(left, right):
        return same_company_relation(by_id[left], by_id[right], judgments)

    def backed_by_two_providers(lone, backed):
        """`lone` is one record; `backed` is two providers agreeing on a number."""
        if len(members[lone]) != 1:
            return False
        (single,) = members[lone]
        others = [by_id[record_id] for record_id in members[backed]]
        identifiers = {
            tuple(map(tuple, _street_parts(other.get("street_address"))[:2]))
            for other in others
        }
        return (
            len({str(other.get("provider")) for other in others}) >= 2
            and len(identifiers) == 1
            and all(_number_slip_pair(by_id[single], other) for other in others)
        )

    original_entity = {row["record_id"]: row["entity_id"] for row in rows.values()}

    def number_slip_codes(record_id, other_id):
        """Reviewer hints for a held pair that looks like a slipped number."""
        mine, other = original_entity[record_id], original_entity[other_id]
        if backed_by_two_providers(mine, other):
            return [NUMBER_SLIP, NUMBER_SLIP_OTHER_BACKED]
        if backed_by_two_providers(other, mine):
            return [NUMBER_SLIP, NUMBER_SLIP_THIS_BACKED]
        return []

    def settle_reason(left, right):
        kind = relation(left, right)
        reason = SETTLED_RELATIONS[kind]
        if kind == "judged_same_address":
            judgment = judgments[pair_key(left, right)]
            reason += (
                f" Judge {judgment.get('model')}: confidence "
                f"{judgment['confidence']:.2f}. {judgment.get('reason') or ''}"
            ).rstrip()
        return reason

    def same_place(first, second):
        return all(
            relation(left, right) in SETTLED_RELATIONS
            for left in members[first]
            for right in members[second]
        )

    links = defaultdict(set)
    for row in rows.values():
        for item in row["match_review_evidence"]:
            other = rows[item["other_record_id"]]["entity_id"]
            if other == row["entity_id"] or other in links[row["entity_id"]]:
                continue
            if same_place(row["entity_id"], other):
                links[row["entity_id"]].add(other)
                links[other].add(row["entity_id"])
    seen = set()
    for start in sorted(links):
        if start in seen:
            continue
        component, stack = set(), [start]
        while stack:
            entity = stack.pop()
            if entity not in component:
                component.add(entity)
                stack.extend(links[entity])
        seen |= component
        if not all(same_place(a, b) for a, b in combinations(sorted(component), 2)):
            continue
        canonicals = sorted(
            rows[next(iter(members[entity]))]["canonical_record_id"]
            for entity in component
        )

        def is_cut_short(record_id):
            return any(
                _less_complete(by_id[record_id], by_id[other_id])
                for other_id in canonicals
                if other_id != record_id
            )

        merged = set().union(*(members[entity] for entity in component))
        # How many records outside this group spell the city the same way.
        inside = Counter(city(by_id[record_id]) for record_id in merged)
        winner = min(
            canonicals,
            key=lambda r: (
                is_cut_short(r),
                -rows[r]["cluster_record_count"],
                inside[city(by_id[r])] - city_usage[city(by_id[r])],
                str(by_id[r].get("country_inferred")).lower() != "false",
                r,
            ),
        )
        absorbed = sorted(merged - members[rows[winner]["entity_id"]])
        entity_id = digest([VERSION, sorted(merged)])
        blank_evidence = {name: None for name in EVIDENCE_FIELDS}
        proofs = []
        for record_id in absorbed:
            reason = settle_reason(record_id, winner)
            proofs.append(
                dict(
                    blank_evidence,
                    other_record_id=record_id,
                    step=SETTLE_STEP,
                    reason=reason,
                )
            )
            rows[record_id].update(
                canonical_record_id=winner,
                is_duplicate=True,
                is_canonical=False,
                deduplication_status="duplicate",
                deduplication_steps=[SETTLE_STEP],
                merge_authority="same_company_address_policy",
                deduplication_reasons=[reason],
                deduplication_evidence=[
                    dict(
                        blank_evidence,
                        other_record_id=winner,
                        step=SETTLE_STEP,
                        reason=reason,
                    )
                ],
            )
        target = rows[winner]
        reasons = target["deduplication_reasons"]
        if target["deduplication_status"] == "singleton":
            reasons = [
                "Retained as the most complete record of its company and "
                "location among same-company records the matcher had held apart."
            ]
            target["deduplication_status"] = "canonical"
        target["deduplication_reasons"] = reasons + [p["reason"] for p in proofs]
        target["deduplication_evidence"] = target["deduplication_evidence"] + proofs
        target["deduplication_steps"] = sorted(
            {*target["deduplication_steps"], SETTLE_STEP}
        )
        if target["merge_authority"] == "none":
            target["merge_authority"] = "same_company_address_policy"
        for record_id in merged:
            rows[record_id].update(
                entity_id=entity_id, cluster_record_count=len(merged)
            )
    for row in rows.values():
        if not row["match_review_evidence"]:
            continue
        review = []
        for item in row["match_review_evidence"]:
            other = rows[item["other_record_id"]]
            if other["entity_id"] == row["entity_id"]:
                continue
            if relation(row["record_id"], other["record_id"]) == DIFFERENT_ADDRESS:
                item = dict(
                    item,
                    decision=DIFFERENT_ADDRESS,
                    reason_codes=[
                        *item["reason_codes"],
                        "settled_as_different_address",
                    ],
                )
            else:
                codes = number_slip_codes(row["record_id"], other["record_id"])
                if codes:
                    item = dict(item, reason_codes=[*item["reason_codes"], *codes])
            review.append(item)
        row["match_review_evidence"] = review
        row["needs_match_review"] = any(
            item["decision"] != DIFFERENT_ADDRESS for item in review
        )
    return list(rows.values())


def _blank(value):
    return not str(value or "").strip()


def _fill_reason(field, judgment=None):
    if field == "city" and judgment:
        return (
            "City was blank at the source. One resolved company/location has the "
            "same company name key and country and a street address with the "
            "same numbers and building letters whose words differ, with no "
            f"region conflict. The address judge ({judgment.get('model')}) read "
            "the two streets as one street written two ways with confidence "
            f"{judgment['confidence']:.2f} ({judgment.get('reason')}); the city "
            "was copied from its retained record and this row was folded into it."
        )
    if field == "city":
        return (
            "City was blank at the source. One resolved company/location has the "
            "same company name key, country and street address (the same house, "
            "road and unit, or the same numbers with only dropped characters, "
            "spacing or an equivalent label in the words) with no region "
            "conflict; the city was copied from its retained record and this "
            "row was folded into it."
        )
    return (
        "Street address was blank at the source. The company has exactly one "
        "resolved location in this city (the same city name, however it is "
        "written) and country with no region conflict; "
        "the street address was copied from its retained record and this row "
        "was folded into it."
    )


def _accepted(judgment):
    return bool(
        judgment
        and judgment.get("same_address") is True
        and (judgment.get("confidence") or 0) >= JUDGE_THRESHOLD
    )


def _fillable_rows(by_id, incomplete):
    """The dlq_incomplete rows a fill may consider: (row, field, prepared keys).

    The only defect is a blank city or a blank street address, the source
    identity is new and unique, and the company, country and other address
    part are present.
    """
    identities = Counter(
        (str(row.get("provider")), str(row.get("provider_id"))) for row in incomplete
    )
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
        if not keys["road_key" if field == "city" else "city_key"]:
            continue
        yield row, field, keys


def _fill_hits(field, keys, company):
    """Company records a text rule accepts as the row's address part."""
    if field == "city":
        hits = [
            hit
            for hit in company
            if hit["street_key"] == keys["street_key"]
            or _spelling_relation(hit.get("street_address"), keys["street_address"])
        ]
    else:
        hits = [hit for hit in company if _city_relation(hit, keys) != "differs"]
    return [hit for hit in hits if not _region_conflict(hit, keys)]


def _fill_judgeable(keys, company):
    """Company records only an address judge can tie to a blank-city row.

    Used when no text rule finds the row's street: the records with the same
    street numbers and building letters and no region conflict. A record with
    a different number is never a candidate.
    """
    return [
        hit
        for hit in company
        if _same_identifiers(hit.get("street_address"), keys["street_address"])
        and not _region_conflict(hit, keys)
    ]


def _company_index(records):
    by_company = defaultdict(list)
    for record in records:
        by_company[record["country_key"], record["name_key"]].append(record)
    return by_company


def fill_judgment_candidates(records, incomplete):
    """Blank-city rows and the resolved records a judge must compare them to.

    Returns sorted (prepared row, resolved record) tuples for fillable rows
    whose street no text rule finds among the company's records, paired with
    each record of that company that has the same numbers and letters.
    """
    by_id = {record["unique_id"]: record for record in records}
    by_company = _company_index(records)
    pairs = {}
    for _, field, keys in _fillable_rows(by_id, incomplete):
        company = by_company[keys["country_key"], keys["name_key"]]
        if field != "city" or _fill_hits(field, keys, company):
            continue
        for hit in _fill_judgeable(keys, company):
            pairs[keys["unique_id"], hit["unique_id"]] = (keys, hit)
    return [pairs[key] for key in sorted(pairs)]


def attach_incomplete(mapping, records, incomplete, judgments=None):
    """Fold fillable dlq_incomplete rows into the one resolved entity they fit.

    A row qualifies when its only defect is a blank city or a blank street
    address, its source identity is new and unique, and exactly one resolved
    entity of the same company (name key) and country carries the address part
    the row still has. For a blank city that is the street: the same street
    key, or the same numbers and building letters with words that differ only
    by dropped characters, spacing or an equivalent label. When no rule finds
    the street, a record with the same numbers and letters also counts if the
    address judge read the two streets as one at JUDGE_THRESHOLD or above
    (`judgments`, keyed by pair_key of the two unique ids). For a blank street
    it is the city: the same key or the same city name written differently.
    Regions must not conflict. The blank field is copied from
    that entity's canonical record, and the row joins the entity as a
    duplicate, so it can never become or change the Gold record. Everything
    else stays parked in dlq_incomplete.

    Returns the updated mapping and the completed records (prepare_record
    output plus filled_field and filled_from_record_id).
    """
    by_id = {record["unique_id"]: record for record in records}
    rows = {row["record_id"]: dict(row) for row in mapping}
    by_company = _company_index(by_id[record_id] for record_id in rows)
    attached = defaultdict(list)
    fill_reasons = {}
    for row, field, keys in _fillable_rows(by_id, incomplete):
        company = by_company[keys["country_key"], keys["name_key"]]
        hits = _fill_hits(field, keys, company)
        judgment = None
        if not hits and field == "city" and judgments:
            judged = [
                (hit, judgments.get(pair_key(hit["unique_id"], keys["unique_id"])))
                for hit in _fill_judgeable(keys, company)
            ]
            judged = [item for item in judged if _accepted(item[1])]
            hits = [hit for hit, _ in judged]
            if judged:
                judgment = max(
                    (item[1] for item in judged), key=lambda item: item["confidence"]
                )
        entities = {rows[hit["unique_id"]]["entity_id"] for hit in hits}
        if len(entities) != 1:
            continue
        canonical = rows[hits[0]["unique_id"]]["canonical_record_id"]
        value = by_id[canonical].get(field)
        if _blank(value):
            continue
        completed = prepare_record(dict(row, **{field: value}))
        fill_reasons[completed["unique_id"]] = _fill_reason(field, judgment)
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
            reason = fill_reasons[completed["unique_id"]]
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
