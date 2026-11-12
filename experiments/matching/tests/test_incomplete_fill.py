"""Incomplete source rows fold into the one resolved entity that completes them."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.matching import prepare_record  # noqa: E402
from lib.resolution import (  # noqa: E402
    FILL_STEP,
    attach_incomplete,
    resolve_records,
)


def source(provider_id, provider="provider_a", **overrides):
    row = dict(
        provider=provider,
        provider_id=provider_id,
        company="Acme Ltd",
        street_address="12 Main St Suite 5",
        city="Boston",
        state="MA",
        country="US",
        country_inferred="false",
    )
    return dict(row, **overrides)


def incomplete(provider_id, field, **overrides):
    return dict(
        source(provider_id, provider="provider_b", **{field: ""}),
        dlq_reasons=[f"{field}:null_or_blank"],
        **overrides,
    )


def resolved(*rows):
    records = [prepare_record(row) for row in rows]
    return resolve_records(records, []), records


class IncompleteFillTests(unittest.TestCase):
    def attach(self, rows, incomplete_rows):
        mapping, records = resolved(*rows)
        updated, filled = attach_incomplete(mapping, records, incomplete_rows)
        return {row["record_id"]: row for row in updated}, filled, records

    def test_blank_city_is_filled_from_the_gold_record_and_folded_in(self):
        rows, filled, records = self.attach(
            [source("1")], [incomplete("9", "city", company="ACME LTD")]
        )
        gold = rows[records[0]["unique_id"]]
        self.assertEqual(len(filled), 1)
        self.assertEqual(filled[0]["city"], "Boston")
        self.assertEqual(filled[0]["filled_field"], "city")
        added = rows[filled[0]["unique_id"]]
        self.assertTrue(added["is_duplicate"])
        self.assertFalse(added["is_canonical"])
        self.assertEqual(added["canonical_record_id"], gold["record_id"])
        self.assertEqual(added["entity_id"], gold["entity_id"])
        self.assertEqual(added["deduplication_steps"], [FILL_STEP])
        self.assertEqual(added["merge_authority"], "incomplete_fill_policy")
        self.assertEqual(
            added["deduplication_evidence"][0]["other_record_id"], gold["record_id"]
        )
        # The Gold record stays canonical and now documents the fold.
        self.assertTrue(gold["is_canonical"])
        self.assertEqual(gold["deduplication_status"], "canonical")
        self.assertEqual(gold["cluster_record_count"], 2)
        self.assertEqual(added["cluster_record_count"], 2)
        self.assertEqual(gold["merge_authority"], "incomplete_fill_policy")
        self.assertIn(FILL_STEP, gold["deduplication_steps"])
        self.assertIsNone(gold["filled_field"])

    def test_blank_street_is_filled_when_the_company_has_one_location_in_the_city(
        self,
    ):
        rows, filled, _ = self.attach(
            [source("1")], [incomplete("9", "street_address")]
        )
        self.assertEqual(filled[0]["street_address"], "12 Main St Suite 5")
        self.assertEqual(filled[0]["filled_field"], "street_address")
        self.assertEqual(len(rows), 2)

    def test_two_locations_in_the_city_leave_a_blank_street_parked(self):
        rows, filled, _ = self.attach(
            [source("1"), source("2", street_address="99 Other Rd")],
            [incomplete("9", "street_address")],
        )
        self.assertEqual(filled, [])
        self.assertEqual(len(rows), 2)

    def test_a_different_unit_does_not_supply_the_city(self):
        _, filled, _ = self.attach(
            [source("1")],
            [incomplete("9", "city", street_address="12 Main St Suite 9")],
        )
        self.assertEqual(filled, [])

    def test_unknown_company_and_region_conflict_stay_parked(self):
        _, filled, _ = self.attach(
            [source("1")],
            [
                incomplete("8", "city", company="Other Co"),
                incomplete("9", "city", state="NY"),
            ],
        )
        self.assertEqual(filled, [])

    def test_only_single_blank_field_rows_with_a_new_unique_identity_qualify(self):
        extra_reason = incomplete("5", "city")
        extra_reason["dlq_reasons"].append("created_at:invalid_date")
        _, filled, _ = self.attach(
            [source("1")],
            [
                extra_reason,
                dict(incomplete("6", "city"), dlq_reasons=["city:too_long"]),
                incomplete("7", "city"),
                incomplete("7", "city"),
                # Same source identity as the resolved record.
                dict(incomplete("1", "city"), provider="provider_a"),
            ],
        )
        self.assertEqual(filled, [])

    def test_fill_joins_an_existing_group_without_changing_its_canonical(self):
        first, second = source("1"), source("2", provider="provider_b")
        records = [prepare_record(first), prepare_record(second)]
        pair = dict(
            id_l=min(r["unique_id"] for r in records),
            id_r=max(r["unique_id"] for r in records),
            fs_decision="exact_baseline",
            fs_match_probability=1.0,
            candidate_origin="blocked",
        )
        mapping = resolve_records(records, [pair])
        canonical = next(row for row in mapping if row["is_canonical"])
        updated, filled = attach_incomplete(mapping, records, [incomplete("9", "city")])
        rows = {row["record_id"]: row for row in updated}
        self.assertEqual(len(filled), 1)
        self.assertEqual(
            rows[filled[0]["unique_id"]]["canonical_record_id"],
            canonical["record_id"],
        )
        after = rows[canonical["record_id"]]
        self.assertEqual(after["merge_authority"], "exact_rule")
        self.assertEqual(after["cluster_record_count"], 3)
        self.assertEqual(sum(row["is_canonical"] for row in updated), 1)
        # The input mapping is not mutated.
        self.assertEqual(canonical["cluster_record_count"], 2)


if __name__ == "__main__":
    unittest.main()
