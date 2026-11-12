"""Conservative components for street-name comparisons on accepted address text.

This is a comparison view, not a postal-address validator. Explicit component
labels are required except for leading house numbers and country-scoped trailing
house-number formats. Unresolved tokens are retained with diagnostics.
"""

import re
import unicodedata

VERSION = "street_components_v6"
ALIASES = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "crescent": "cres",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}


def normalized(value):
    """Preserve the existing comparison/baseline Unicode-whitespace semantics."""
    return re.sub(r"\s+", " ", value or "").strip().lower()


def _fold_latin_marks(text):
    """Drop accents from Latin letters only; other scripts keep their marks.

    Satō/Sato, Kondō/Kondo and Ürümqi/Urumqi are the same name romanized with
    and without diacritics. Thai, Devanagari and Hebrew marks are letters of
    the word, not decoration, so a mark is folded only after a Latin base.
    """
    kept = []
    latin_base = False
    for char in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(char):
            if latin_base:
                continue
        else:
            latin_base = unicodedata.name(char, "").startswith("LATIN")
        kept.append(char)
    return unicodedata.normalize("NFKC", "".join(kept))


def text_key(value):
    text = _fold_latin_marks(unicodedata.normalize("NFKC", normalized(value)))
    return " ".join(
        "".join(
            c if c.isalnum() or unicodedata.category(c) in ("Mn", "Mc") else " "
            for c in text
        ).split()
    )


# Legal-entity forms, as text_key renders them. They are removed from the ends
# of a company comparison name only: "Ray Inc" and "Ray" compare as the same
# name, but a name is never merged on this view alone (see lib/gold.py).
LEGAL_SUFFIXES = (
    "s a de c v",
    "s de r l de c v",
    "s de r l",
    "san tic ltd sti",
    "ltd sti",
    "gmbh co kg",
    "sp z o o sp k",
    "sp z o o",
    "co ltd",
    "co limited",
    "co inc",
    "co kg",
    "co",
    "pty ltd",
    "pte ltd",
    "pvt ltd",
    "s a s",
    "s r l",
    "s p a",
    "s a",
    "s l",
    "b v",
    "n v",
    "e v",
    "k k",
    "a s",
    "a ş",
    "aş",
    "llc",
    "llp",
    "lp",
    "ltd",
    "ltda",
    "limited",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "plc",
    "gmbh",
    "mbh",
    "ag",
    "kg",
    "kgaa",
    "ohg",
    "ug",
    "sa",
    "sas",
    "sarl",
    "srl",
    "spa",
    "sl",
    "bv",
    "nv",
    "oy",
    "ab",
    "pty",
    "pte",
    "pvt",
    "sti",
    "şti",
    "ооо",
    "зао",
    "оао",
    "пао",
    "ао",
)
# Space-free scripts carry the entity form inside the token, at either end.
LEGAL_AFFIXES = (
    "股份有限公司",
    "有限责任公司",
    "有限公司",
    "株式会社",
    "有限会社",
    "合同会社",
    "合資会社",
    "合名会社",
    "주식회사",
    "유한회사",
)
_SUFFIX_PATTERN = re.compile(
    r"(?:^| )(?:"
    + "|".join(sorted(map(re.escape, LEGAL_SUFFIXES), key=len, reverse=True))
    + r")$"
)
_PREFIX_PATTERN = re.compile(
    r"^(?:" + "|".join(map(re.escape, ("ооо", "зао", "оао", "пао", "ао"))) + r")(?: |$)"
)


def company_name_key(value):
    """Company comparison name: text_key with legal-entity forms folded away.

    Removal repeats until nothing matches ("Co., Ltd." then "Co"), and a name
    that is nothing but a legal form keeps its full text key.
    """
    key = text_key(value)
    name = key
    while True:
        previous = name
        name = _SUFFIX_PATTERN.sub("", name).strip()
        name = _PREFIX_PATTERN.sub("", name).strip()
        for affix in LEGAL_AFFIXES:
            if name.endswith(affix):
                name = name[: -len(affix)].strip()
            if name.startswith(affix):
                name = name[len(affix) :].strip()
        if name == previous:
            break
    return name or key


# Comparison aliases only. Preserve source text and keep mappings country-scoped.
REGIONS = {
    "US": dict(
        zip(
            "alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada ohio oklahoma oregon pennsylvania tennessee texas utah vermont virginia washington wisconsin wyoming".split(),
            "al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi mn ms mo mt ne nv oh ok or pa tn tx ut vt va wa wi wy".split(),
        )
    )
    | {
        "district of columbia": "dc",
        "new hampshire": "nh",
        "new jersey": "nj",
        "new mexico": "nm",
        "new york": "ny",
        "north carolina": "nc",
        "north dakota": "nd",
        "rhode island": "ri",
        "south carolina": "sc",
        "south dakota": "sd",
        "west virginia": "wv",
    },
    "CA": {
        "alberta": "ab",
        "british columbia": "bc",
        "manitoba": "mb",
        "new brunswick": "nb",
        "newfoundland and labrador": "nl",
        "northwest territories": "nt",
        "nova scotia": "ns",
        "nunavut": "nu",
        "ontario": "on",
        "prince edward island": "pe",
        "quebec": "qc",
        "saskatchewan": "sk",
        "yukon": "yt",
    },
    "AU": {
        "new south wales": "nsw",
        "victoria": "vic",
        "queensland": "qld",
        "south australia": "sa",
        "western australia": "wa",
        "tasmania": "tas",
        "northern territory": "nt",
        "australian capital territory": "act",
    },
}


# Fallback for countries without their own table: a name and its code still
# compare equal ("Maryland" vs "MD" filed under TR). Only full names map, so an
# unknown code stays itself; the benchmark found 560 pairs held back by this.
ANY_COUNTRY_REGIONS = {
    name: code for aliases in REGIONS.values() for name, code in aliases.items()
}
if len(ANY_COUNTRY_REGIONS) != sum(len(aliases) for aliases in REGIONS.values()):
    raise ValueError("Region names must not map to different codes across countries")


def normalize_region(value, country):
    """Build a region comparison key scoped to the record's country."""
    region = text_key(value)
    country = normalized(country).upper()
    aliases = REGIONS.get(country, {})
    prefix = country.lower() + " "
    if aliases and region.startswith(prefix):
        subdivision = region[len(prefix) :]
        if subdivision in aliases or subdivision in aliases.values():
            region = subdivision
    if region in aliases:
        return aliases[region]
    return ANY_COUNTRY_REGIONS.get(region, region)


IDENTIFIER = r"(?:[a-z]?\d+[a-z]?(?:[-/]\d+[a-z]?)*|[a-z])"
HOUSE_NUMBER = r"\d+[a-z]?(?:[-/]\d+[a-z]?)*"

GLOBAL_LABELS = {
    "units": r"apartment|apt|suite|ste|unit",
    "buildings": r"building|bldg|block|edifício|edificio",
    "floors": r"floor|flr|level|lvl",
    "house_numbers": r"house",
}
LOCAL_LABELS = {
    "GB": {"units": r"flat|flt|studio|estúdio|estudio"},
    "AU": {"units": r"flat"},
    "MX": {
        "units": r"depto|departamento|interior",
        "buildings": r"edif",
        "floors": r"piso",
    },
    "ES": {"units": r"puerta|depto|departamento", "floors": r"piso|planta"},
    "IT": {"floors": r"piano"},
    "BR": {"units": r"apto|apartamento", "buildings": r"bloco", "floors": r"andar"},
    "FR": {
        "units": r"appartement|appt",
        "buildings": r"bâtiment|batiment",
        "floors": r"étage|etage",
    },
}
# These patterns require both a recognizable road form and a distinctive name
# before treating a trailing number as a house number. Route 66/Plein 1945 and
# numbered Korean road names consequently keep their own numbers.
TRAILING_ROAD_FORMS = {
    "DE": r"(?:straße|strasse|str|weg|allee|platz|ring|gasse|ufer|damm|steig|pfad|chaussee)\b",
    "NL": r"(?:straat|weg|laan|singel|plein|dreef|pad|hof|ring|kade|steeg)\b",
    "RU": r"\b(?:street|road|avenue|lane|alley|highway|boulevard|embankment|square)\b",
    "KR": r"(?:geori|gil|daero|distance|ga|ro)\b",
    "ES": r"\b(?:callejón|callejon|calle|paseo|alameda|pasadizo|avenida|urbanización|urbanizacion|ronda|carretera|camino|plaza)\b",
    "IT": r"\b(?:vicolo|incrocio|borgo|viale|via|piazza|corso|strada)\b",
    "BR": r"\b(?:rua|avenida|travessa|alameda|residencial|praia|feira|esplanada|vereda|vila|rodovia|praça|praca|área|area|núcleo|nucleo)\b",
    "MX": r"\b(?:calle|circuito|periférico|periferico|eje vial|ampliación|ampliacion|avenida|calzada|cerrada|diagonal|prolongación|prolongacion)\b",
    "PL": r"\b(?:ulica|ul|aleja|osiedle|plac)\b",
}
CONNECTORS = re.compile(
    r"\b(?:de|del|des|der|la|las|los|el|do|da|dos|das|van|het|den)\b"
)
LEADING_HOUSE = re.compile(rf"^\s*({HOUSE_NUMBER})(?=\s|,|$)")
TRAILING_HOUSE = re.compile(rf"(?<![\w/-])({HOUSE_NUMBER})(?=\s*(?:\([^)]*\)\s*)?$)")
NUMBERED_ROAD_PREFIX = re.compile(r"^\d+\s+(?:mile|kilometer|kilometre|km)\b")
REVERSED_CN_BUILDING = re.compile(rf"(?<!\w)({IDENTIFIER})\s+(?:building|block|seat)\b")
REVERSED_FLOOR = re.compile(r"(?<!\w)(\d+)(?:st|nd|rd|th)?\s+(?:floor|flr)\b")
HASH_UNIT = re.compile(rf"#\s*({IDENTIFIER})(?![\w/-])")


def _labels(country):
    result = dict(GLOBAL_LABELS)
    for key, local in LOCAL_LABELS.get(country, {}).items():
        result[key] += "|" + local
    return result


def _supports_trailing(country, prefix):
    prefix = prefix.rstrip(". ")
    pattern = TRAILING_ROAD_FORMS.get(country)
    if not pattern or not re.search(pattern, prefix):
        return False
    numbered_russian_name = country == "RU" and re.search(r"\b\d+\s+let\b", prefix)
    if country in {"DE", "NL", "RU", "KR"}:
        ends_with_road_type = re.search(pattern + r"\s*$", prefix)
        # Russian translated names may put the road type first. Permit that
        # order only without unexplained numeric tails (or a known '70 Let'
        # road name), not 'Street Name 660 Floo 35'.
        starts_with_road_type = country == "RU" and re.match(pattern, prefix)
        if not ends_with_road_type and not (
            starts_with_road_type
            and (not re.search(r"\d", prefix) or numbered_russian_name)
        ):
            return False
    elif re.search(r"\d", prefix):
        numbered_road = re.fullmatch(
            r"(?:calle|avenida|via|ulica|ul)\.?\s+\d+[a-z]?", prefix
        )
        if not numbered_road:
            return False
    # Remove the road type and articles; a bare 'Highway 101' must not be
    # reduced to 'Highway'. Digits left in prefix can be the road's own name.
    distinctive = text_key(CONNECTORS.sub(" ", re.sub(pattern, " ", prefix)))
    return bool(distinctive)


def parse_street_components(street, country):
    """Return a clean road key plus retained component values and warnings."""
    country = normalized(country).upper()
    raw = unicodedata.normalize("NFKC", street or "")
    separated = re.sub(
        r"(?<=[a-z])(?=(?:Apartment|Apt|Suite|Ste|Unit|Building|Floor|Level)\.?\s*\d)",
        " ",
        raw,
    )
    street = normalized(separated)
    result = {
        key: []
        for key in (
            "house_numbers",
            "units",
            "buildings",
            "floors",
            "actions",
            "warnings",
        )
    }
    labels = _labels(country)
    if separated != raw:
        result["actions"].append("repaired_camel_case_component_separator")
    spans = []

    def add(start, end, kind, value, action):
        if any(start < b and end > a for a, b in spans):
            return
        spans.append((start, end))
        result[kind].append(value)
        result["actions"].append(action)

    for kind, names in labels.items():
        # A label must end as a word, except for compact numeric forms such as
        # Apt12. Without this boundary, 'Flats' becomes 'Flat' + unit 's', and
        # 'Buildings'/'Floors' similarly invent components from road names.
        pattern = re.compile(
            rf"(?<!\w)(?:{names})(?=\b|\d)\.?\s*[:#-]?\s*({IDENTIFIER})(?![\w/-])"
        )
        for match in pattern.finditer(street):
            add(*match.span(), kind, match.group(1), "explicit_" + kind)
    for match in HASH_UNIT.finditer(street):
        # A numbered route is not a suite, e.g. Route #66.
        if re.search(r"\b(?:route|highway|rte|hwy)\s*$", street[: match.start()]):
            continue
        add(*match.span(), "units", match.group(1), "explicit_units")
    for match in REVERSED_FLOOR.finditer(street):
        add(*match.span(), "floors", match.group(1), "explicit_floors")
    if country == "CN":
        for match in REVERSED_CN_BUILDING.finditer(street):
            add(*match.span(), "buildings", match.group(1), "reversed_buildings")
    road = street
    for start, end in sorted(spans, reverse=True):
        road = road[:start] + " " + road[end:]
    road = re.sub(r"\s+", " ", road).strip(" ,;:-")

    # Prefer a recognized trailing house before a leading number: in
    # '70 Let Octyabrya Alley 1/7', 70 belongs to the road and 1/7 is the house.
    if not result["house_numbers"]:
        trailing = TRAILING_HOUSE.search(road)
        if trailing:
            prefix = road[: trailing.start()].strip(" ,;:-")
            if re.search(r"\d\s*$", prefix):
                result["warnings"].append("ambiguous_trailing_numbers_retained")
            elif _supports_trailing(country, prefix):
                result["house_numbers"].append(trailing.group(1))
                result["actions"].append("country_scoped_trailing_house_number")
                road = road[: trailing.start()] + " " + road[trailing.end() :]
            elif country in TRAILING_ROAD_FORMS and re.search(r"\d", prefix):
                result["warnings"].append("ambiguous_trailing_numbers_retained")
    leading = LEADING_HOUSE.match(road)
    numbered_name = NUMBERED_ROAD_PREFIX.search(road) or (
        country == "RU" and re.match(r"\d+\s+let\b", road)
    )
    if not result["house_numbers"] and leading and not numbered_name:
        remainder = road[leading.end() :].strip(" ,;:-")
        # A bare number, '42 Street', etc. is ambiguous without a named road.
        if text_key(remainder) and text_key(remainder) not in {
            "street",
            "st",
            "road",
            "rd",
            "avenue",
            "ave",
            "highway",
            "route",
        }:
            value = leading.group(1)
            if country == "AU" and re.fullmatch(r"\d+[a-z]?/\d+[a-z]?", value):
                unit, value = value.split("/")
                result["units"].append(unit)
                result["actions"].append("australian_unit_house_slash")
            result["house_numbers"].append(value)
            result["actions"].append("leading_house_number")
            road = remainder
    tokens = text_key(road).split()
    if country in {"US", "CA", "GB", "AU"}:
        tokens = [ALIASES.get(token, token) for token in tokens]
    result["road_key"] = " ".join(tokens)
    all_labels = "|".join(labels.values())
    if re.search(rf"\b(?:{all_labels}|suie)\b", road):
        result["warnings"].append("unresolved_component_marker_retained")
    if re.search(r"\d", road):
        result["warnings"].append("road_number_tokens_retained")
    if not result["road_key"]:
        result["warnings"].append("missing_road_name")
    for key in ("house_numbers", "units", "buildings", "floors"):
        result[key] = sorted(set(result[key]))
        if len(result[key]) > 1:
            result["warnings"].append("multiple_" + key)
    result["actions"] = sorted(set(result["actions"]))
    result["warnings"] = sorted(set(result["warnings"]))
    return result
