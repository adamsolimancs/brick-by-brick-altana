"""Same-company review pairs: merge the same place, release two clear places."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.matching import prepare_record  # noqa: E402
from lib.resolution import (  # noqa: E402
    DIFFERENT_ADDRESS,
    SETTLE_STEP,
    judgeable,
    judgment_candidates,
    pair_key,
    resolve_records,
    NUMBER_SLIP,
    NUMBER_SLIP_OTHER_BACKED,
    NUMBER_SLIP_THIS_BACKED,
    same_company_relation,
    settle_same_company,
)


def source(provider_id, street="44524 Jackson Drives Suite 713", **overrides):
    row = dict(
        provider="provider_a",
        provider_id=provider_id,
        company="Acme Ltd",
        street_address=street,
        city="Erinmouth",
        state="MA",
        country="US",
        country_inferred="false",
    )
    return prepare_record(dict(row, **overrides))


def review(left, right):
    left, right = sorted((left["unique_id"], right["unique_id"]))
    return dict(
        id_l=left,
        id_r=right,
        fs_decision="review_uncertain",
        fs_match_probability=0.999,
        candidate_origin="blocked",
        fs_guard_review=True,
    )


def settle(records, pairs, exact=(), judgments=None):
    """Review every listed pair, settle, and index the result by provider_id."""
    scored = [review(*pair) for pair in pairs]
    scored += [dict(review(*pair), fs_decision="exact_baseline") for pair in exact]
    mapping = resolve_records(records, scored)
    by_id = {record["unique_id"]: record["provider_id"] for record in records}
    return {
        by_id[row["record_id"]]: row
        for row in settle_same_company(mapping, records, judgments)
    }


def judged(left, right, same=True, confidence=0.95):
    key = pair_key(left["unique_id"], right["unique_id"])
    return {
        key: dict(
            same_address=same,
            confidence=confidence,
            reason="Two romanizations of one street name.",
            model="test-judge",
        )
    }


def translated(provider_id, street, original, **overrides):
    evidence = '{"street_address_original": "%s"}' % original
    return source(provider_id, street, translation_evidence_json=evidence, **overrides)


class RelationTests(unittest.TestCase):
    def relation(self, left, right, **overrides):
        return same_company_relation(source("1", left), source("2", right, **overrides))

    def test_punctuation_and_case_do_not_make_two_addresses(self):
        self.assertEqual(
            self.relation(
                "Viaducto Mónaco 968 859 Suite 284",
                "VIADUCTO MONACO 968 859, suite 284",
            ),
            "same_text",
        )

    def test_same_street_in_a_differently_written_city(self):
        self.assertEqual(
            self.relation(
                "9839 Hill Haven", "9839 Hill Haven", city="Sur de Erinmouth"
            ),
            "same_street_other_city",
        )

    def test_same_street_in_another_city_is_not_decided(self):
        same = "545 Yoshida Street"
        for city, other in (
            ("Yokohama-shi Seya-ku", "Sakae-ku, Yokohama"),
            ("Kozushima-mura", "Niijima Village"),
            ("Zhangjiagang County", "Changshu County"),
        ):
            left = source("1", same, city=city)
            right = source("2", same, city=other)
            self.assertIsNone(same_company_relation(left, right))

    def test_city_translations_word_order_and_romanization_are_one_city(self):
        same = "545 Yoshida Street"
        for city, other in (
            ("New Sonya", "Nueva Sonya"),
            ("Ost-Elizabeth", "East Elizabeth"),
            ("Josué do Sul", "South Joshua"),
            ("See Jennifer", "Lake Jennifer"),
            ("Charlottemouth", "bocca di Charlotte"),
            ("Aoba-ku, Yokohama", "Yokohama-shi Aoba-ku"),
            ("Katori District, Kanzaki Town", "Katori-gun Kozaki-machi"),
        ):
            left = source("1", same, city=city)
            right = source("2", same, city=other)
            self.assertEqual(
                same_company_relation(left, right), "same_street_other_city"
            )

    def test_address_cut_short_is_the_same_place(self):
        self.assertEqual(
            self.relation("44524 Jackson Drives Su", "44524 Jackson Drives Suite 713"),
            "truncated",
        )
        self.assertEqual(
            self.relation("1209 N Orange St", "1209 N Orange St, Suite 352"),
            "truncated",
        )

    def test_cut_inside_a_number_is_not_decided(self):
        self.assertIsNone(
            self.relation("Stretto Tomaselli, 2", "Stretto Tomaselli, 229 Piano 2")
        )

    def test_different_number_is_never_decided(self):
        for left, right in (
            ("1697 David Street", "169 David Street"),
            (
                "561 Jason Field Suite 301 Unit 152",
                "561 Jason Field Suite 124 Unit 152",
            ),
            (
                "Calle Norte Palomo 213 Interior 79",
                "Calle Norte Palomo 213 Interior 079",
            ),
            ("Kadeplatz 65/30", "Kadeplatz 6530"),
            # A typo in the words does not excuse a different number.
            ("27 Hançer Mission, Suite 215", "277 Haçer Mission, Suite 215"),
            # A different building letter is a different building.
            ("54195 Matthews Burg, Bilding C", "54195 Matthews Burg, Building D"),
        ):
            self.assertIsNone(self.relation(left, right), (left, right))
            self.assertFalse(judgeable(source("1", left), source("2", right)))

    def test_dropped_characters_spacing_and_labels_are_one_street(self):
        for left, right in (
            ("82 Ullmannweg Uit 110", "82 Ullmannweg Unit 110"),
            (
                "49345 Melissa Corners Apt 457 Bilding D",
                "49345 Melissa Corners Apt. 457 Building D",
            ),
            # One character dropped on each side.
            (
                "2899 Joseph Cove pt. 085 - Bilding C",
                "2899 Joseph Cove Apt. 085 - Building C",
            ),
            ("AvenidaPaulo Costa, 89", "Avenida Paulo Costa, 89"),
            ("Dimitrios-Buchholz-Platz1-3", "Dimitrios-Buchholz-Platz 1-3"),
            ("0016 bAILEY rOADaPT. 240", "0016 bAILEY rOAD aPT. 240"),
            ("837 Mark Fork Piso 32", "837 Mark Fork Floor 32"),
            ("6892 House Union # 678", "6892 House Union Apt. 678"),
            (
                "Via Detti, 42 Appartamento 4 Ste. 513",
                "Via Detti, 42 Appartamento 4, Suite 513",
            ),
            ("Blvd. Durango 894 384", "Boulevard Durango 894 384"),
            ("352 Watanabe St. Sute 487", "352 Watanabe Street - Suite 487"),
            ("Süßebierstraße 8133 # 431", "SÜSSEBIERSTRASSE 8133 APT 431"),
            (
                "Chongqing Street X Building, Floor 43",
                "Chongqing Street Building X, Floor 43",
            ),
        ):
            self.assertEqual(
                self.relation(left, right), "street_spelling", (left, right)
            )

    def test_replaced_characters_or_other_words_are_not_a_typo(self):
        for left, right in (
            ("12 Via Roma", "12 Via Rosa"),
            ("5 Hill Road", "5 Hall Road"),
            ("63/44 Thiwar Road Floor 25", "63/44 Theewor Road, floor 25"),
            ("Ugol'naya Embankment 7, Building A", "Coal Embankment, 7, Building A"),
            # Three dropped characters is past the limit.
            ("49345 Melissa Corners Bilding D", "49345 Melisa Corers Building D"),
        ):
            self.assertIsNone(self.relation(left, right), (left, right))

    def test_street_spelling_in_a_differently_written_city_is_one_place(self):
        self.assertEqual(
            self.relation(
                "2444 Martinez Roads - Floor 44",
                "2444 Martinez Roads, Piano 44",
                city="Est Erinmouth",
            ),
            "street_spelling",
        )
        self.assertEqual(
            self.relation(
                "44524 Jackson Drives Su",
                "44524 Jackson Drives Suite 713",
                city="Sur de Erinmouth",
            ),
            "truncated",
        )

    def test_same_text_before_translation_is_one_street(self):
        korean = "대구광역시 동대문구 서초대8로 지하979 (영식홍황마을)"
        left = translated(
            "1", "Daegu-si Seocho-dae 8-ro Underground 979 (Honghwang-ma-eul)", korean
        )
        right = translated(
            "2", "Daegu-si Seocho-dae 8-ro Underground 979 (Honghwang-maeur)", korean
        )
        self.assertEqual(same_company_relation(left, right), "same_original_text")
        # One character dropped in the original is still the same text.
        left = translated("1", "071/4 Kanchanaruk Road", "071/4 ถนนคณานุรักษ์")
        right = translated("2", "071/4 Thanon Kanunarak", "071/4 ถนคณานุรักษ์")
        self.assertEqual(same_company_relation(left, right), "same_original_text")
        # A whole name word that differs in the original is not.
        left = translated("1", "386 Hashimoto Street", "386 橋本 Street")
        right = translated("2", "386 Hon Street", "386 本 Street")
        self.assertIsNone(same_company_relation(left, right))

    def test_judge_settles_only_a_confident_same_answer(self):
        left = source("1", "63/44 Thiwar Road Floor 25")
        right = source("2", "63/44 Theewor Road, floor 25")
        self.assertTrue(judgeable(left, right))
        self.assertIsNone(same_company_relation(left, right))
        self.assertEqual(
            same_company_relation(left, right, judged(left, right)),
            "judged_same_address",
        )
        self.assertEqual(
            same_company_relation(left, right, judged(left, right, confidence=0.9)),
            "judged_same_address",
        )
        for answer in (
            judged(left, right, confidence=0.89),
            judged(left, right, same=False, confidence=0.99),
        ):
            self.assertIsNone(same_company_relation(left, right, answer))

    def test_judge_cannot_bridge_a_number_region_or_company(self):
        base = "63/44 Thiwar Road Floor 25"
        for right in (
            source("2", "63/45 Theewor Road, floor 25"),
            source("2", "63/44 Theewor Road, floor 25", state="NY"),
            source("2", "63/44 Theewor Road, floor 25", company="Other Ltd"),
            source("2", "63/44 Theewor Road, floor 25", country="CA"),
        ):
            left = source("1", base)
            self.assertFalse(judgeable(left, right))
            self.assertNotEqual(
                same_company_relation(left, right, judged(left, right)),
                "judged_same_address",
            )

    def test_same_street_in_an_unrecognized_city_goes_to_the_judge(self):
        left = source("1", "684 Meghan Knolls Ste. 123", city="Bohnenmund")
        right = source("2", "684 Meghan Knolls, Suite 123", city="Beanmouth")
        self.assertIsNone(same_company_relation(left, right))
        self.assertTrue(judgeable(left, right))

    def test_cut_short_street_in_another_city_is_not_decided(self):
        self.assertIsNone(
            self.relation(
                "44524 Jackson Drives Su",
                "44524 Jackson Drives Suite 713",
                city="Boston",
            )
        )

    def test_another_street_and_building_elsewhere_is_a_different_address(self):
        self.assertEqual(
            self.relation("Cui Street Building o", "Gu Street Building t", state="NY"),
            DIFFERENT_ADDRESS,
        )
        self.assertEqual(
            self.relation(
                "Cui Street Building o", "Gu Street Building t", city="Boston"
            ),
            DIFFERENT_ADDRESS,
        )
        # Same city and region: could be a move or a second office.
        self.assertIsNone(
            self.relation("Cui Street Building o", "Gu Street Building t")
        )

    def test_other_company_country_or_region_is_never_the_same_place(self):
        same = "44524 Jackson Drives Suite 713"
        self.assertIsNone(self.relation(same, same, company="Other Ltd"))
        self.assertIsNone(self.relation(same, same, country="CA"))
        self.assertIsNone(self.relation(same, same, state="NY"))


class SettleTests(unittest.TestCase):
    def test_cut_short_record_folds_into_the_complete_one(self):
        short = source("1", "44524 Jackson Drives Su")
        full = source("2")
        rows = settle([short, full], [(short, full)])
        self.assertEqual(rows["1"]["entity_id"], rows["2"]["entity_id"])
        self.assertTrue(rows["2"]["is_canonical"])
        self.assertEqual(rows["2"]["deduplication_status"], "canonical")
        self.assertEqual(rows["2"]["merge_authority"], "same_company_address_policy")
        self.assertEqual(rows["2"]["cluster_record_count"], 2)
        self.assertTrue(rows["1"]["is_duplicate"])
        self.assertEqual(rows["1"]["canonical_record_id"], rows["2"]["record_id"])
        self.assertEqual(rows["1"]["deduplication_steps"], [SETTLE_STEP])
        self.assertEqual(
            rows["1"]["deduplication_evidence"][0]["other_record_id"],
            rows["2"]["record_id"],
        )
        for row in rows.values():
            self.assertFalse(row["needs_match_review"])
            self.assertEqual(row["match_review_evidence"], [])

    def test_larger_group_keeps_its_record_when_nothing_is_cut_short(self):
        lone = source("1", "9839 Hill Haven", city="Sur de Erinmouth")
        pair = [source("2", "9839 Hill Haven"), source("3", "9839 Hill Haven")]
        rows = settle(
            [lone, *pair], [(lone, pair[0]), (lone, pair[1])], exact=[tuple(pair)]
        )
        self.assertEqual(len({row["entity_id"] for row in rows.values()}), 1)
        self.assertFalse(rows["1"]["is_canonical"])
        self.assertEqual(sum(row["is_canonical"] for row in rows.values()), 1)
        self.assertEqual({row["cluster_record_count"] for row in rows.values()}, {3})

    def test_kept_city_is_the_spelling_other_records_use(self):
        used, unused = "South Erinmouth", "Sur de Erinmouth"
        # Swap the provider ids so the result cannot come from the ID order.
        for first, second in (("1", "2"), ("2", "1")):
            local = source(first, "9839 Hill Haven", city=used)
            foreign = source(second, "9839 Hill Haven", city=unused)
            neighbour = source("3", "1 Other Rd", company="Zeta", city=used, state="NY")
            rows = settle([local, foreign, neighbour], [(local, foreign)])
            self.assertTrue(rows[first]["is_canonical"])
            self.assertFalse(rows[second]["is_canonical"])

    def test_short_address_that_fits_two_suites_merges_with_neither(self):
        short = source("1", "1209 N Orange St")
        first = source("2", "1209 N Orange St Suite 352")
        second = source("3", "1209 N Orange St Suite 941")
        rows = settle([short, first, second], [(short, first), (short, second)])
        self.assertEqual(len({row["entity_id"] for row in rows.values()}), 3)
        self.assertTrue(all(row["needs_match_review"] for row in rows.values()))

    def test_different_address_is_released_and_stays_separate(self):
        here = source("1", "Cui Street Building o")
        there = source("2", "Gu Street Building t", state="NY")
        rows = settle([here, there], [(here, there)])
        self.assertNotEqual(rows["1"]["entity_id"], rows["2"]["entity_id"])
        for row in rows.values():
            self.assertTrue(row["is_canonical"])
            self.assertFalse(row["needs_match_review"])
            (item,) = row["match_review_evidence"]
            self.assertEqual(item["decision"], DIFFERENT_ADDRESS)

    def test_typo_record_folds_into_the_fuller_spelling(self):
        # Swap the provider ids so the result cannot come from the ID order.
        for first, second in (("1", "2"), ("2", "1")):
            typo = source(first, "49345 Melissa Corners Apt 457 Bilding D")
            full = source(second, "49345 Melissa Corners Apt. 457 Building D")
            rows = settle([typo, full], [(typo, full)])
            self.assertEqual(rows["1"]["entity_id"], rows["2"]["entity_id"])
            self.assertTrue(rows[second]["is_canonical"])
            self.assertEqual(rows[first]["deduplication_steps"], [SETTLE_STEP])
            self.assertIn("dropped characters", rows[first]["deduplication_reasons"][0])

    def test_judged_pair_merges_and_records_the_judgment(self):
        left = source("1", "63/44 Thiwar Road Floor 25")
        right = source("2", "63/44 Theewor Road, floor 25")
        records = [left, right]
        mapping = resolve_records(records, [review(left, right)])
        self.assertEqual(
            [
                pair_key(a["unique_id"], b["unique_id"])
                for a, b in judgment_candidates(mapping, records)
            ],
            [pair_key(left["unique_id"], right["unique_id"])],
        )
        rows = settle(records, [(left, right)], judgments=judged(left, right))
        self.assertEqual(rows["1"]["entity_id"], rows["2"]["entity_id"])
        (duplicate,) = [row for row in rows.values() if row["is_duplicate"]]
        reason = duplicate["deduplication_reasons"][0]
        self.assertIn("test-judge", reason)
        self.assertIn("confidence 0.95", reason)
        self.assertIn("Two romanizations of one street name.", reason)
        # Without a judgment, or with a weak one, the pair stays on hold.
        for answer in (None, judged(left, right, confidence=0.8)):
            rows = settle(records, [(left, right)], judgments=answer)
            self.assertNotEqual(rows["1"]["entity_id"], rows["2"]["entity_id"])
            self.assertTrue(all(row["needs_match_review"] for row in rows.values()))

    def test_every_cross_entity_pair_is_a_judge_candidate(self):
        # Entities merge only as a whole, so the unreviewed pair is judged too.
        first = source("1", "63/44 Thiwar Road Floor 25")
        twin = source("3", "63/44 Thiwar Road Floor 25")
        other = source("2", "63/44 Theewor Road, floor 25")
        records = [first, twin, other]
        scored = [
            review(first, other),
            dict(review(first, twin), fs_decision="exact_baseline"),
        ]
        mapping = resolve_records(records, scored)
        self.assertEqual(len(judgment_candidates(mapping, records)), 2)
        one_answer = judged(first, other)
        rows = settle(
            records, [(first, other)], exact=[(first, twin)], judgments=one_answer
        )
        self.assertEqual(len({row["entity_id"] for row in rows.values()}), 2)
        both = {**one_answer, **judged(twin, other)}
        rows = settle(records, [(first, other)], exact=[(first, twin)], judgments=both)
        self.assertEqual(len({row["entity_id"] for row in rows.values()}), 1)

    def test_number_conflict_is_not_a_judge_candidate(self):
        one = source("1", "1697 David Street")
        two = source("2", "169 David Street")
        mapping = resolve_records([one, two], [review(one, two)])
        self.assertEqual(judgment_candidates(mapping, [one, two]), [])

    def test_undecided_pair_and_other_company_are_left_alone(self):
        one = source("1", "1697 David Street")
        two = source("2", "169 David Street")
        neighbour = source("3", "1697 David Street", company="Other Ltd")
        rows = settle([one, two, neighbour], [(one, two), (one, neighbour)])
        self.assertEqual(len({row["entity_id"] for row in rows.values()}), 3)
        self.assertTrue(all(row["needs_match_review"] for row in rows.values()))
        self.assertEqual(
            {item["decision"] for item in rows["1"]["match_review_evidence"]},
            {"review_uncertain"},
        )


class MissingNumberTests(unittest.TestCase):
    def relation(self, left, right):
        return same_company_relation(source("1", left), source("2", right))

    def test_a_missing_number_is_less_complete_not_a_conflict(self):
        for left, right in (
            ("Maeed Brooks Suite 115", "999 Maeed Brooks Suite 115"),
            ("12 Main Street", "Suite 5, 12 Main Street"),
            ("Main Street", "12 Main St"),
        ):
            with self.subTest(left=left):
                self.assertEqual(self.relation(left, right), "missing_number")
                self.assertEqual(self.relation(right, left), "missing_number")

    def test_each_side_having_something_the_other_lacks_is_not_decided(self):
        for left, right in (
            ("Ronda Granados 41 # 56", "Ronda Granados 41 A"),
            ("Privada Jordania 52", "Privada Jordania 524 Edif. 050 , Depto. 045"),
            ("Leonova Lane", "Leonov St, building 171, apt 5/6"),
            ("Main Street Suite 1", "12 Main Street Suite 1"),
        ):
            with self.subTest(left=left):
                self.assertIsNone(self.relation(left, right))

    def test_fuller_record_is_kept(self):
        short, full = (
            source("1", "Maeed Brooks Suite 115"),
            source("2", "999 Maeed Brooks Suite 115", provider="provider_b"),
        )
        rows = settle([short, full], [(short, full)])
        self.assertTrue(rows["2"]["is_canonical"])
        self.assertTrue(rows["1"]["is_duplicate"])
        self.assertFalse(rows["2"]["needs_match_review"])


class NumberSlipFlagTests(unittest.TestCase):
    """A lone number against one two providers agree on is flagged, not merged."""

    def group(self, lone_street):
        lone = source("1", lone_street, provider="provider_b")
        first = source("2", "93/17 Saha Street")
        second = source("3", "93/17 Saha Street", provider="provider_b")
        return lone, first, second

    def codes(self, rows, provider_id):
        return [
            code
            for item in rows[provider_id]["match_review_evidence"]
            for code in item["reason_codes"]
        ]

    def settle(self, records, exact=True):
        lone, first, second = records
        return settle(
            list(records), [(lone, first)], exact=[(first, second)] if exact else []
        )

    def test_slipped_number_stays_held_and_is_flagged_on_both_sides(self):
        for street in ("93/7 Saha Street", "39/17 Saha Street"):
            with self.subTest(street=street):
                rows = self.settle(self.group(street))
                self.assertTrue(rows["1"]["is_canonical"])
                self.assertTrue(rows["1"]["needs_match_review"])
                self.assertNotEqual(rows["1"]["entity_id"], rows["2"]["entity_id"])
                self.assertIn(NUMBER_SLIP, self.codes(rows, "1"))
                self.assertIn(NUMBER_SLIP_OTHER_BACKED, self.codes(rows, "1"))
                self.assertIn(NUMBER_SLIP_THIS_BACKED, self.codes(rows, "2"))

    def test_no_flag_without_two_providers_or_for_another_number(self):
        lone = source("1", "93/7 Saha Street", provider="provider_b")
        same_provider = (
            lone,
            source("2", "93/17 Saha Street"),
            source("3", "93/17 Saha Street"),
        )
        cases = [same_provider] + [
            self.group(street)
            for street in (
                "93/27 Saha Street",
                "93/170 Saha Road",
                "612/17 Saha Street",
            )
        ]
        for records in cases:
            with self.subTest(street=records[0]["street_address"]):
                rows = self.settle(records)
                self.assertNotIn(NUMBER_SLIP, self.codes(rows, "1"))
                self.assertTrue(rows["1"]["needs_match_review"])


if __name__ == "__main__":
    unittest.main()
