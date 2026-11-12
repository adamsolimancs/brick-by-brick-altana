"""Address components remain available while road-name comparisons exclude them."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altana_matching_components import parse_street_components as parse  # noqa: E402


class AddressComponentsTests(unittest.TestCase):
    def test_edificio_building_label_works_on_us_addresses(self):
        expected = parse("194 Example Road Suite 938, Building C", "US")
        for marker in ("Edifício", "Edificio"):
            with self.subTest(marker=marker):
                self.assertEqual(
                    parse(f"194 Example Road Suite 938, {marker} C", "US"),
                    expected,
                )
        self.assertEqual(expected["buildings"], ["c"])
        self.assertEqual(expected["units"], ["938"])
        self.assertEqual(expected["road_key"], "example rd")

    def test_plural_road_words_do_not_invent_letter_components(self):
        # Reproduces an observed saved-candidate parsing error. The road name
        # ends in Flats; it does not contain a second flat with identifier S.
        r = parse("Flat 77 Example Flats", "GB")
        self.assertEqual(r["road_key"], "example flats")
        self.assertEqual(r["units"], ["77"])
        self.assertNotIn("multiple_units", r["warnings"])
        for word in ("Flats", "Units", "Buildings", "Floors", "Houses"):
            with self.subTest(word=word):
                r = parse(f"12 Example {word}", "GB")
                self.assertEqual(r["road_key"], "example " + word.lower())
                self.assertEqual(r["units"], [])
                self.assertEqual(r["buildings"], [])
                self.assertEqual(r["floors"], [])
                self.assertEqual(r["house_numbers"], ["12"])

    def test_component_boundaries_keep_explicit_letters_and_compact_numbers(self):
        for text in ("12 Main Street Suite S", "12 Main Street Suite:S"):
            self.assertEqual(parse(text, "GB")["units"], ["s"])
        r = parse("12 Main Street Apt12 Building S Floor3", "GB")
        self.assertEqual(r["units"], ["12"])
        self.assertEqual(r["buildings"], ["s"])
        self.assertEqual(r["floors"], ["3"])
        self.assertEqual(r["road_key"], "main st")

    def test_numbered_russian_names_and_incomplete_component_suffixes(self):
        for text in [
            "Alley 70 Let Octyabrya 1/7, Building 11 - Floor 29",
            "70 Let Octyabrya Alley 1/7, Building 11 Floor 29",
        ]:
            r = parse(text, "RU")
            self.assertEqual(r["house_numbers"], ["1/7"])
            self.assertIn("70 let octyabrya", r["road_key"])
        for text, country, expected in [
            ("Etzlerstraße 45 - Foor 12", "DE", "etzlerstraße 45 foor 12"),
            (
                "Paolo-Mülichen-Ring 4/4 - loor 14",
                "DE",
                "paolo mulichen ring 4 4 loor 14",
            ),
            (
                "Daegu-si Jongno-gu Seonun 583-geori 660 Floo 35",
                "KR",
                "daegu si jongno gu seonun 583 geori 660 floo 35",
            ),
            (
                "Avenida de Lucas Vallejo 77 pt. 02 Apt 44",
                "ES",
                "avenida de lucas vallejo 77 pt 02",
            ),
        ]:
            with self.subTest(text=text):
                r = parse(text, country)
                self.assertEqual(r["house_numbers"], [])
                self.assertEqual(r["road_key"], expected)
                self.assertIn("ambiguous_trailing_numbers_retained", r["warnings"])

    def test_observed_component_abbreviations_and_missing_separator(self):
        for text in ["Flat 98n Hall Common, Unit 183", "Flt 98n Hall Common, Unit 183"]:
            r = parse(text, "GB")
            self.assertEqual(r["units"], ["183", "98n"])
            self.assertEqual(r["road_key"], "hall common")
        a = parse("144 Young Ramp Suite 469 Apt 446", "US")
        b = parse("144 Young RampSuite 469 Apt 446", "US")
        self.assertEqual(a["road_key"], b["road_key"])
        self.assertEqual(a["units"], b["units"])
        self.assertEqual(
            parse("Estúdio 7 George Junctions, Suite 109", "GB")["units"], ["109", "7"]
        )

    def test_chained_components_and_leading_apartments(self):
        cases = [
            (
                "157 Bullock Place Suite 368, Floor 12",
                "US",
                "bullock place",
                ["157"],
                ["368"],
                [],
                ["12"],
            ),
            (
                "53360 Clark Square Suite 298 Building B",
                "CA",
                "clark square",
                ["53360"],
                ["298"],
                ["b"],
                [],
            ),
            (
                "858 Bennett Knoll Suite 541, Suite 970",
                "US",
                "bennett knoll",
                ["858"],
                ["541", "970"],
                [],
                [],
            ),
            (
                "Flat 21 78 Watson Siding Floor 7",
                "AU",
                "watson siding",
                ["78"],
                ["21"],
                [],
                ["7"],
            ),
            (
                "Studio 52 Natalie Underpass",
                "GB",
                "natalie underpass",
                [],
                ["52"],
                [],
                [],
            ),
            (
                "Apt. 850 0 Catherine Park",
                "AU",
                "catherine park",
                ["0"],
                ["850"],
                [],
                [],
            ),
            (
                "123 Main Street #4 - Building C 3rd Floor",
                "US",
                "main st",
                ["123"],
                ["4"],
                ["c"],
                ["3"],
            ),
        ]
        for street, country, road, house, units, buildings, floors in cases:
            with self.subTest(street=street):
                r = parse(street, country)
                self.assertEqual(
                    [
                        r[k]
                        for k in (
                            "road_key",
                            "house_numbers",
                            "units",
                            "buildings",
                            "floors",
                        )
                    ],
                    [road, house, units, buildings, floors],
                )

    def test_country_scoped_trailing_house_numbers(self):
        cases = [
            ("Gehringerweg 47-26", "DE", "gehringerweg", "47-26"),
            ("Jäkelstr. 41/84", "DE", "jakelstr", "41/84"),
            ("Faassingel 67 Ste. 344", "NL", "faassingel", "67"),
            ("Vicolo Viola, 6 Piano 9", "IT", "vicolo viola", "6"),
            (
                "Ampliación Rodríquez 923 Edif. 595, Depto. 510",
                "MX",
                "ampliacion rodriquez",
                "923",
            ),
            (
                "Alameda de Melisa Pedro 939 Apt. 15",
                "ES",
                "alameda de melisa pedro",
                "939",
            ),
            ("Praia Paulo da Paz, 32, Apt 278", "BR", "praia paulo da paz", "32"),
            (
                "Kurchatova Highway 648 Building 4 - Unit 158",
                "RU",
                "kurchatova highway",
                "648",
            ),
            (
                "Gangnamdae 193-gil 68 (Yeongil-yun-eup)",
                "KR",
                "gangnamdae 193 gil yeongil yun eup",
                "68",
            ),
        ]
        for street, country, road, house in cases:
            with self.subTest(street=street):
                r = parse(street, country)
                self.assertEqual(r["road_key"], road)
                self.assertEqual(r["house_numbers"], [house])
        for street, country, road in [
            ("ul. Jadwigi 67/33", "PL", "ul jadwigi"),
            ("Área Mariah Jesus, 9, Apt 372", "BR", "area mariah jesus"),
            ("Diagonal Nayarit 292 Interior 931", "MX", "diagonal nayarit"),
            (
                "Ulsan-si Yongsan-gu Teheran-ga 917",
                "KR",
                "ulsan si yongsan gu teheran ga",
            ),
            ("Level 5 674 Emma Close Apt 397", "AU", "emma close"),
        ]:
            with self.subTest(street=street):
                self.assertEqual(parse(street, country)["road_key"], road)

    def test_component_positions_and_multiple_values_are_retained(self):
        a = parse("Guangzhou Road H Building, Suite 242", "CN")
        b = parse("Guangzhou Road Building h Suite 242", "CN")
        self.assertEqual(a["road_key"], b["road_key"])
        self.assertEqual(a["buildings"], b["buildings"])
        self.assertEqual(a["units"], b["units"])
        self.assertEqual(parse("Cao Road Z Seat", "CN")["buildings"], ["z"])
        r = parse("Ge Road Building f, Building E", "CN")
        self.assertEqual(r["buildings"], ["e", "f"])
        self.assertIn("multiple_buildings", r["warnings"])
        self.assertEqual(
            parse("Sandy village, house 7, building 7/7", "RU")["house_numbers"], ["7"]
        )

    def test_numbered_roads_and_ambiguous_numbers_are_not_deleted(self):
        for street, country, expected in [
            ("1209 5th Avenue Suite 4", "US", "5th ave"),
            ("Route 66", "US", "route 66"),
            ("Route #66", "US", "route 66"),
            ("7 Mile Road", "US", "7 mile rd"),
            ("Highway 101", "RU", "highway 101"),
            ("Plein 1945", "NL", "plein 1945"),
            ("Calle 42", "ES", "calle 42"),
            ("Circuito Sur Montoya 315 134", "MX", "circuito sur montoya 315 134"),
            ("Ackermannplatz 78 33", "DE", "ackermannplatz 78 33"),
            ("Unit Circle", "US", "unit circle"),
            ("12 Main Road Suie 243", "US", "main rd suie 243"),
            ("Some Road 10", "ZZ", "some road 10"),
        ]:
            with self.subTest(street=street):
                self.assertEqual(parse(street, country)["road_key"], expected)
        self.assertIn(
            "ambiguous_trailing_numbers_retained",
            parse("Ackermannplatz 78 33", "DE")["warnings"],
        )
        self.assertIn(
            "unresolved_component_marker_retained",
            parse("12 Main Road Suie 243", "US")["warnings"],
        )

    def test_slash_semantics_and_country_scoping(self):
        a = parse("85/18 Baker Highway", "AU")
        self.assertEqual((a["house_numbers"], a["units"]), (["18"], ["85"]))
        b = parse("93/17 Saha Street", "IN")
        self.assertEqual((b["house_numbers"], b["units"]), (["93/17"], []))
        self.assertEqual(parse("Guangzhou Road H Building", "US")["buildings"], [])

    def test_missing_road_and_unicode(self):
        for street in (None, "", "  ", "Suite 4"):
            with self.subTest(street=street):
                r = parse(street, "US")
                self.assertEqual(r["road_key"], "")
                self.assertIn("missing_road_name", r["warnings"])
        r = parse("１２３ Erdoğan Plaza Suite ４", "TR")
        # Accents on Latin letters fold in comparison keys; source text is untouched.
        self.assertEqual(r["road_key"], "erdogan plaza")
        self.assertEqual(r["house_numbers"], ["123"])
        self.assertEqual(r["units"], ["4"])


if __name__ == "__main__":
    unittest.main()
