"""Comparison keys fold what the benchmark showed are not real differences."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altana_matching_components import (  # noqa: E402
    company_name_key,
    normalize_region,
    text_key,
)


class TextKeyTests(unittest.TestCase):
    def test_latin_diacritics_fold_for_comparison_only(self):
        self.assertEqual(text_key("281 Satō Street"), "281 sato street")
        self.assertEqual(text_key("Kondō"), text_key("Kondo"))
        self.assertEqual(text_key("Ürümqi City"), "urumqi city")
        self.assertEqual(text_key("İnönü Ertaş Şti."), "inonu ertas sti")
        self.assertEqual(text_key("Zürich"), "zurich")

    def test_non_latin_scripts_keep_their_marks(self):
        thai = "ถนนเวียงจันทึก"
        self.assertEqual(text_key(thai), thai)
        hindi = "नई दिल्ली"
        self.assertEqual(text_key(hindi), hindi)
        self.assertEqual(text_key("横浜市中区"), "横浜市中区")
        self.assertEqual(text_key("수원시 권선구"), "수원시 권선구")
        self.assertEqual(text_key("Ромашка"), "ромашка")

    def test_letters_without_decomposition_are_unchanged(self):
        self.assertEqual(text_key("Straße Łódź Ørsted"), "straße łodz ørsted")


class CompanyNameKeyTests(unittest.TestCase):
    def test_trailing_legal_forms_are_removed_from_the_comparison_view(self):
        cases = {
            "Ray Inc": "ray",
            "Vision PLC": "vision",
            "Vision Corporation": "vision",
            "Segura Memorial Hospital Ltd.": "segura memorial hospital",
            "Atkinson, Parkin and Andrews GmbH": "atkinson parkin and andrews",
            "Kobayashi Farm AG": "kobayashi farm",
            "Bibo Cheng Information Technology Co., Ltd.": (
                "bibo cheng information technology"
            ),
            "FPUH Bek Sp. z o.o.": "fpuh bek",
            "Hethur Stiftung & Co. KG": "hethur stiftung",
            "Viewwin Network S.A.": "viewwin network",
            "Acme S.A. de C.V.": "acme",
            "Nordic Pty Ltd": "nordic",
            "İnönü Ertaş Ltd. Şti.": "inonu ertas",
            "Maison Dupont SARL": "maison dupont",
        }
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(company_name_key(value), expected)

    def test_cjk_and_cyrillic_entity_markers_fold_at_either_end(self):
        self.assertEqual(company_name_key("维旺明网络有限公司 S.A."), "维旺明网络")
        self.assertEqual(company_name_key("维旺明网络有限公司"), "维旺明网络")
        self.assertEqual(company_name_key("有限会社高橋農林 AG"), "高橋農林")
        self.assertEqual(company_name_key("佐藤情報合同会社"), "佐藤情報")
        self.assertEqual(company_name_key("주식회사 비전 PLC"), "비전")
        self.assertEqual(company_name_key("ООО Ромашка"), "ромашка")

    def test_a_name_that_is_only_a_legal_form_keeps_its_full_key(self):
        self.assertEqual(company_name_key("Ltd"), "ltd")
        self.assertEqual(company_name_key("株式会社"), "株式会社")
        self.assertEqual(company_name_key(""), "")
        self.assertEqual(company_name_key(None), "")

    def test_words_inside_a_name_are_never_removed(self):
        self.assertEqual(company_name_key("Inc Magazine Group"), "inc magazine group")
        self.assertEqual(company_name_key("Company Store"), "company store")


class RegionKeyTests(unittest.TestCase):
    def test_country_scoped_aliases_still_apply(self):
        self.assertEqual(normalize_region("Nova Scotia", "CA"), "ns")
        self.assertEqual(normalize_region("NS", "CA"), "ns")
        self.assertEqual(normalize_region("Washington", "US"), "wa")
        self.assertEqual(normalize_region("US-WA", "US"), "wa")

    def test_known_name_and_code_pairs_agree_under_any_country(self):
        self.assertEqual(
            normalize_region("Maryland", "TR"), normalize_region("MD", "TR")
        )
        self.assertEqual(
            normalize_region("South Carolina", "TR"), normalize_region("SC", "TR")
        )
        self.assertEqual(
            normalize_region("Alberta", "JP"), normalize_region("AB", "JP")
        )
        self.assertEqual(normalize_region("Maryland", ""), "md")

    def test_unknown_regions_stay_as_text_keys(self):
        self.assertEqual(normalize_region("Rheinland-Pfalz", "DE"), "rheinland pfalz")
        self.assertEqual(normalize_region("Podlaskie", "PL"), "podlaskie")
        self.assertEqual(normalize_region("", "US"), "")
        self.assertEqual(normalize_region(None, None), "")


if __name__ == "__main__":
    unittest.main()
