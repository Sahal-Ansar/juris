"""Reporter citations: parse and normalise to one canonical form (PLAN 3.4, reused by 3.9).

Handles the reporters found in the corpus, including the OCR damage of old SCR volumes:

========== ======================================= =====================================
Reporter   Canonical form                          Variants accepted
========== ======================================= =====================================
INSC       ``2023 INSC 1043``                      ``2023INSC1043``, ``2023:INSC:838``
SCR        ``[1963] 3 SCR 22``                     ``S.C.R.``, ``S. C. R.``, ``(1963)``,
           ``[1963] Supp 2 SCR 915``               ``[1966) 2``, ``Supp.``/``Suppl.``
SCC        ``(2002) 3 SCC 66``                     ``S.C.C.``, OCR ``sec``, ``[2000] 6``
           ``1994 Supp (2) SCC 116``               ``Supp.(1)``, ``(1994) Supp 2 SCC``
           ``(2005) 1 SCC (Cri) 123``              ``(L&S)``, ``(Civ)``, ``(Tax)``
AIR        ``AIR 1965 SC 1234``                    ``A.I.R.``, ``AIR (1995) SC 1``,
                                                   court names (``Madras`` -> ``Mad``)
SCC OnLine ``2021 SCC OnLine SC 456``              ``Online``
SCALE      ``(2007) 8 SCALE 110``                  ``[1995] 5 SCALE``
JT         ``JT 2013 (5) SC 142``                  ``JT2013 (5) SC 142``
HC neutral ``2023:DHC:9316``, ``2024:BHC-OS:1234`` spaces around colons
========== ======================================= =====================================

Anything else returns ``None``: an unrecognised string is never guessed into a citation.
"""

import re
from dataclasses import dataclass

Reporter = str  # "INSC" | "SCR" | "SCC" | "AIR" | "SCC OnLine" | "SCALE" | "JT" | "HCNC"


@dataclass(frozen=True)
class ReporterCitation:
    reporter: Reporter
    year: int
    page: int  # page, or the running number of a neutral / online citation
    volume: int | None = None
    supp: bool = False
    series: str | None = None  # SCC sub-series ("Cri", "L&S") or AIR/online/neutral court
    raw: str = ""

    @property
    def canonical(self) -> str:
        v, s = self.volume, self.series
        match self.reporter:
            case "INSC":
                return f"{self.year} INSC {self.page}"
            case "SCR":
                vol = f"{v} " if v is not None else ""
                return f"[{self.year}] {'Supp ' if self.supp else ''}{vol}SCR {self.page}"
            case "SCC":
                sub = f" ({s})" if s else ""
                if self.supp:
                    vol = f" ({v})" if v is not None else ""
                    return f"{self.year} Supp{vol} SCC{sub} {self.page}"
                return f"({self.year}) {v} SCC{sub} {self.page}"
            case "AIR":
                return f"AIR {self.year} {s} {self.page}"
            case "SCC OnLine":
                return f"{self.year} SCC OnLine {s} {self.page}"
            case "SCALE":
                return f"({self.year}) {v} SCALE {self.page}"
            case "JT":
                return f"JT {self.year} ({v}) SC {self.page}"
            case "HCNC":
                return f"{self.year}:{s}:{self.page}"
        raise ValueError(f"unknown reporter {self.reporter}")


# Brackets around the year: OCR mixes them up ("[1966) 2 S.C.R.", "(2008] 7").
_O, _C = r"[\(\[{]", r"[\)\]}]"
_YEAR = r"(?P<year>1[89]\d\d|20\d\d)"
_SUPP = r"(?P<supp>(?i:supp)(?:[lL])?\.?)"  # "Supp.", "Suppl.", AWS's "SUPP."
_SCR = r"S\.?\s*C\.?\s*R\.?"
_SCC = r"(?:S\.?\s*C\.?\s*C\.?|sec|SCC)"
_SUB = r"(?:\(\s*(?P<sub>Cri|L\s*&\s*S|Civ|Tax|Jour)\s*\)\s*)?"
_PAGE = r"(?P<page>\d{1,5})"

_PATTERNS: list[tuple[Reporter, re.Pattern[str]]] = [
    ("INSC", re.compile(rf"\b{_YEAR}\s*:?\s*INSC\s*:?\s*{_PAGE}\b")),
    (
        "SCR",
        re.compile(
            rf"{_O}\s*{_YEAR}\s*{_C}\s*(?:{_SUPP}\s*)?\(?(?P<vol>\d{{1,2}})?\)?\s*{_SCR}\s*{_PAGE}\b"
        ),
    ),
    (
        "SCC",  # 1994 Supp (2) SCC 116 / (1994) Supp 2 SCC 116
        re.compile(
            rf"(?:{_O}\s*|\b){_YEAR}(?:\s*{_C})?\s*{_SUPP}\s*\(?(?P<vol>\d{{1,2}})?\)?\s*"
            rf"{_SCC}\s*{_SUB}{_PAGE}\b"
        ),
    ),
    (
        "SCC",
        re.compile(rf"{_O}\s*{_YEAR}\s*{_C}\s*(?P<vol>\d{{1,2}})\s*{_SCC}\s*{_SUB}{_PAGE}\b"),
    ),
    (
        "AIR",
        re.compile(
            rf"\bA\.?\s*I\.?\s*R\.?\s*(?:{_O}\s*)?{_YEAR}(?:\s*{_C})?\s*"
            r"(?P<court>[A-Z][A-Za-z&\.]{0,12}(?:\s?&\s?[A-Z][A-Za-z\.]{0,3})?)\s*"
            r"(?:\(\s*[A-Za-z\s\.]{0,20}\)\s*)?"  # "(FB)", "(Bench)"
            rf"{_PAGE}\b"
        ),
    ),
    (
        "SCC OnLine",
        re.compile(
            rf"\b{_YEAR}\s+SCC\s+On\s?[Ll]ine\s+(?P<court>[A-Z][A-Za-z&]{{1,8}})\s+{_PAGE}\b"
        ),
    ),
    ("SCALE", re.compile(rf"{_O}\s*{_YEAR}\s*{_C}\s*(?P<vol>\d{{1,2}})\s*SCALE\s*{_PAGE}\b")),
    ("JT", re.compile(rf"\bJT\s*{_YEAR}\s*\(\s*(?P<vol>\d{{1,2}})\s*\)\s*SC\s*{_PAGE}\b")),
    (
        "HCNC",  # High Court neutral citations: 2023:DHC:9316, 2024:BHC-OS:1234, 2023:KER:5
        re.compile(rf"\b{_YEAR}\s*:\s*(?P<court>[A-Z]{{2,5}}(?:-[A-Z]{{2,4}})?)\s*:\s*{_PAGE}\b"),
    ),
]

# AIR court abbreviations (AIR's own style); full names and OCR/dot variants map onto them.
AIR_COURTS = {
    "SC": "SC", "PC": "PC", "FC": "FC",
    "ALL": "All", "ALLAHABAD": "All", "AP": "AP", "ANDH": "AP", "BOM": "Bom", "BOMBAY": "Bom",
    "CAL": "Cal", "CALCUTTA": "Cal", "DEL": "Del", "DELHI": "Del", "GAU": "Gau", "GAUH": "Gau",
    "GUJ": "Guj", "GUJARAT": "Guj", "HP": "HP", "J&K": "J&K", "JK": "J&K", "KANT": "Kant",
    "KAR": "Kant", "KER": "Ker", "KERALA": "Ker", "MAD": "Mad", "MADRAS": "Mad", "MP": "MP",
    "MYS": "Mys", "MYSORE": "Mys", "NAG": "Nag", "NAGPUR": "Nag", "ORI": "Ori", "ORISSA": "Ori",
    "PAT": "Pat", "PATNA": "Pat", "P&H": "P&H", "PUNJ": "Punj", "PUNJAB": "Punj", "RAJ": "Raj",
    "RAJASTHAN": "Raj", "LAH": "Lah", "LAHORE": "Lah", "OUDH": "Oudh", "SIND": "Sind",
    "RANG": "Rang", "RANGOON": "Rang", "TRAV": "Trav", "TRAV-CO": "Trav", "HYD": "Hyd",
    "MANI": "Mani", "TRIP": "Tri", "SIK": "Sikkim", "JHAR": "Jhar", "CHH": "Chh", "UTR": "Utr",
}  # fmt: skip
SCC_SERIES = {"CRI": "Cri", "L&S": "L&S", "CIV": "Civ", "TAX": "Tax", "JOUR": "Jour"}
SCC_ONLINE_COURTS = {
    "SC", "Del", "Bom", "Cal", "Mad", "All", "Ker", "Kar", "Guj", "Pat", "Ori", "Raj", "MP",
    "P&H", "AP", "TS", "Gau", "HP", "J&K", "Jhar", "Chh", "Utt", "Tri", "Mani", "Megh", "Sikk",
    "NCLAT", "NGT", "CCI", "ITAT",
}  # fmt: skip


def _air_court(raw: str) -> str | None:
    key = raw.upper().replace(".", "").replace(" ", "")
    return AIR_COURTS.get(key)


def _build(reporter: Reporter, m: re.Match[str]) -> ReporterCitation | None:
    groups = m.groupdict()
    year, page = int(groups["year"]), int(groups["page"])
    vol = int(groups["vol"]) if groups.get("vol") else None
    supp = bool(groups.get("supp"))
    series: str | None = None
    if reporter == "SCC":
        if not supp and vol is None:
            return None
        if groups.get("sub"):
            series = SCC_SERIES[re.sub(r"\s", "", groups["sub"]).upper()]
    elif reporter == "AIR":
        series = _air_court(groups["court"])
        if series is None:
            return None
    elif reporter == "SCC OnLine":
        court = groups["court"]
        series = court if court in SCC_ONLINE_COURTS else None
        if series is None:
            return None
    elif reporter == "HCNC":
        series = groups["court"]
        if series == "INSC":
            reporter, series = "INSC", None
    return ReporterCitation(reporter, year, page, vol, supp, series, m.group(0))


def parse_citation(text: str) -> ReporterCitation | None:
    """The citation that ``text`` consists of (surrounding punctuation allowed), or None."""
    s = text.strip().strip(".,;:")
    for reporter, pattern in _PATTERNS:
        m = pattern.fullmatch(s) or pattern.fullmatch(s.rstrip(")]"))
        if m and (c := _build(reporter, m)):
            return c
    return None


def normalise_citation(text: str) -> str | None:
    """Canonical form of a single citation string, e.g. '(1984) 2 sec 112' -> '(1984) 2 SCC 112'."""
    c = parse_citation(text)
    return c.canonical if c else None


def find_citations(text: str) -> list[ReporterCitation]:
    """All recognised citations in running text, in order (overlaps resolved left to right)."""
    hits: list[tuple[int, int, ReporterCitation]] = []
    for reporter, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            if c := _build(reporter, m):
                hits.append((m.start(), m.end(), c))
    hits.sort(key=lambda h: (h[0], -h[1]))
    out, end = [], -1
    for start, stop, c in hits:
        if start >= end:
            out.append(c)
            end = stop
    return out
