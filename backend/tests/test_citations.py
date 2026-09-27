"""Citation normaliser (PLAN 3.4): real citation strings from the corpus and its metadata.

Every input below was copied from a judgment text or a metadata field in the snapshot,
OCR damage included ('sec' for SCC, '[1966) 2', 'S. C. R.').
"""

import pytest

from juris.ingest.citations import find_citations, normalise_citation, parse_citation

REAL = [
    # SCC
    ("(2016) 11 SCC 313", "(2016) 11 SCC 313"),
    ("(1973) 2 SCC 535", "(1973) 2 SCC 535"),
    ("(2011) 8 SCC 1", "(2011) 8 SCC 1"),
    ("(1984) 2 sec 112", "(1984) 2 SCC 112"),
    ("[2000] 6 sec 545", "(2000) 6 SCC 545"),
    ("(2003) a sec 559", None),  # OCR'd volume: not guessed
    ("1989 Supp (1) SCC 499", "1989 Supp (1) SCC 499"),
    ("1995 Supp.(1) SCC 596", "1995 Supp (1) SCC 596"),
    ("1992 Supp.(1) sec 471", "1992 Supp (1) SCC 471"),
    ("1994 Supp (3) SCC 606", "1994 Supp (3) SCC 606"),
    ("(2012) 4 SCC (Civ) 777", "(2012) 4 SCC (Civ) 777"),
    ("(2013) 4 SCC (Cri) 1", "(2013) 4 SCC (Cri) 1"),
    # SCR
    ("[2015] 3 S.C.R. 243", "[2015] 3 SCR 243"),
    ("[1963] SUPP. 2 S.C.R. 915", "[1963] Supp 2 SCR 915"),
    ("[1997] SUPP. 1 S.C.R. 298", "[1997] Supp 1 SCR 298"),
    ("[1995] Supp 2 SCR 733", "[1995] Supp 2 SCR 733"),
    ("[1966) 2 S.C.R. 229", "[1966] 2 SCR 229"),
    ("[1960] 2 S. C.R. 671", "[1960] 2 SCR 671"),
    ("[1975]3 SCR 619", "[1975] 3 SCR 619"),
    ("[1978] 3 S. C. R. 134", "[1978] 3 SCR 134"),
    ("[1978) 1 SCR 641", "[1978] 1 SCR 641"),
    ("(1985) Supp. 3 SCR 123", "[1985] Supp 3 SCR 123"),
    ("(1963) 1 SCR 47", "[1963] 1 SCR 47"),
    ("[2024] 5 SCR 612", "[2024] 5 SCR 612"),
    # AIR
    ("AIR 1966 SC 1068", "AIR 1966 SC 1068"),
    ("AIR (1995) SC 1", "AIR 1995 SC 1"),
    ("AIR (1959) SC 781", "AIR 1959 SC 781"),
    ("AIR 1978 Mad. 134", "AIR 1978 Mad 134"),
    ("AIR 1978 Madras 54", "AIR 1978 Mad 54"),
    ("AIR 1932 Madras 605", "AIR 1932 Mad 605"),
    ("AIR 1962 Cal. 485", "AIR 1962 Cal 485"),
    ("AIR 1984 All 147", "AIR 1984 All 147"),
    # neutral citations
    ("2015 INSC 163", "2015 INSC 163"),
    ("2015INSC163", "2015 INSC 163"),
    ("2025 INSC 1286", "2025 INSC 1286"),
    ("2008 : INSC:925", "2008 INSC 925"),
    ("2022 : INSC : 207", "2022 INSC 207"),
    ("2023:INSC:838", "2023 INSC 838"),
    ("2023:DHC:9316", "2023:DHC:9316"),
    # other reporters
    ("2009 SCC OnLine Del 2472", "2009 SCC OnLine Del 2472"),
    ("2021 SCC OnLine SC 456", "2021 SCC OnLine SC 456"),
    ("2019 SCC OnLine TS 1765", "2019 SCC OnLine TS 1765"),
    ("2014 SCC Online Ilom 1825", None),  # OCR'd court: not guessed
    ("(2007) 8 SCALE 110", "(2007) 8 SCALE 110"),
    ("[1995] 5 SCALE 620", "(1995) 5 SCALE 620"),
    ("(2006] 3 SCALE 82", "(2006) 3 SCALE 82"),
    ("JT 2013 (5) SC 142", "JT 2013 (5) SC 142"),
    ("JT2013 (5) SC 142", "JT 2013 (5) SC 142"),
    ("JT 2013(2) SC 362", "JT 2013 (2) SC 362"),
]


@pytest.mark.parametrize(("raw", "canonical"), REAL)
def test_real_citation_strings(raw: str, canonical: str | None) -> None:
    assert normalise_citation(raw) == canonical


def test_there_are_at_least_30_real_strings() -> None:
    assert len(REAL) >= 30


@pytest.mark.parametrize(
    "text",
    ["Rs. 25/- per Sq. Yd.", "(1990) 3 SCC", "AIR 1959 XYZ 781", "Section 9 of the Act", "1994"],
)
def test_non_citations_are_none(text: str) -> None:
    assert normalise_citation(text) is None


def test_parsed_fields() -> None:
    c = parse_citation("[1963] SUPP. 2 S.C.R. 915")
    assert c is not None
    assert (c.reporter, c.year, c.volume, c.supp, c.page) == ("SCR", 1963, 2, True, 915)
    c = parse_citation("AIR 1978 Madras 54")
    assert c is not None and (c.reporter, c.series, c.page) == ("AIR", "Mad", 54)


def test_variants_of_one_citation_share_a_canonical_form() -> None:
    variants = ["[1966] 2 S.C.R. 229", "[1966) 2 S. C. R. 229", "(1966) 2 SCR 229."]
    assert {normalise_citation(v) for v in variants} == {"[1966] 2 SCR 229"}


def test_find_citations_in_running_text() -> None:
    text = (
        "relied on (2002) 3 SCC 66 and AIR 1965 SC 1234; see also [1963] 3 S.C.R. 22 : "
        "2023 INSC 736, and 1995 Supp.(1) SCC 596."
    )
    assert [c.canonical for c in find_citations(text)] == [
        "(2002) 3 SCC 66",
        "AIR 1965 SC 1234",
        "[1963] 3 SCR 22",
        "2023 INSC 736",
        "1995 Supp (1) SCC 596",
    ]
