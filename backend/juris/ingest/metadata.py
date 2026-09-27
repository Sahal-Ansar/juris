"""Legal metadata normalisation and cross-source merge (PLAN 3.4).

Sources, joined on the eCourts CNR:

- ``aws_sc``: AWS Indian Supreme Court metadata (title, parties, the author judge, SCR and INSC
  citations, decision date, disposal) plus its ``raw_html`` card, which carries the full coram
  (the author starred), the case number and a headnote excerpt;
- ``aws_hc``: AWS Indian High Court metadata (``"RFA/436/2016 of A Vs B"`` titles, the judge
  string, dates, a mostly empty disposal);
- ``kanoongpt``: KanoonGPT/indian-case-laws (Apache-2.0), a normalised republication of the same
  records (split parties, coram list, bench size, docket, SC headnote);
- ``text``: the judgment itself (the SCR bench line ``[A AND B, JJ.]`` and the citations printed
  in its header).

Every field keeps the source it came from, and disagreements between sources are recorded as
conflicts rather than resolved silently. A value no source gives reliably stays ``None``: it is
never guessed.
"""

import ast
import datetime as dt
import html
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from juris.ingest.citations import find_citations, parse_citation
from juris.models import CourtLevel, Document, DocumentKind

Source = Literal["aws_sc", "aws_hc", "kanoongpt", "text"]
Disposition = Literal[
    "allowed", "partly_allowed", "dismissed", "disposed", "remanded", "referred", "withdrawn",
    "directions", "modified",
]  # fmt: skip

# ---- Courts ------------------------------------------------------------------------------

# canonical name, level, match keys (lower-case words that identify the court)
_COURTS: list[tuple[str, CourtLevel, tuple[str, ...]]] = [
    ("Supreme Court of India", CourtLevel.SUPREME_COURT, ("supreme court",)),
    ("High Court of Allahabad", CourtLevel.HIGH_COURT, ("allahabad",)),
    ("High Court of Andhra Pradesh", CourtLevel.HIGH_COURT, ("andhra",)),
    ("High Court of Bombay", CourtLevel.HIGH_COURT, ("bombay",)),
    ("High Court of Calcutta", CourtLevel.HIGH_COURT, ("calcutta",)),
    ("High Court of Chhattisgarh", CourtLevel.HIGH_COURT, ("chhattisgarh",)),
    ("High Court of Delhi", CourtLevel.HIGH_COURT, ("delhi",)),
    ("High Court of Gauhati", CourtLevel.HIGH_COURT, ("gauhati", "guwahati")),
    ("High Court of Gujarat", CourtLevel.HIGH_COURT, ("gujarat",)),
    ("High Court of Himachal Pradesh", CourtLevel.HIGH_COURT, ("himachal",)),
    ("High Court of Jammu and Kashmir", CourtLevel.HIGH_COURT, ("jammu",)),
    ("High Court of Jharkhand", CourtLevel.HIGH_COURT, ("jharkhand",)),
    ("High Court of Karnataka", CourtLevel.HIGH_COURT, ("karnataka", "mysore")),
    ("High Court of Kerala", CourtLevel.HIGH_COURT, ("kerala",)),
    ("High Court of Madhya Pradesh", CourtLevel.HIGH_COURT, ("madhya pradesh",)),
    ("High Court of Madras", CourtLevel.HIGH_COURT, ("madras",)),
    ("High Court of Manipur", CourtLevel.HIGH_COURT, ("manipur",)),
    ("High Court of Meghalaya", CourtLevel.HIGH_COURT, ("meghalaya",)),
    ("High Court of Orissa", CourtLevel.HIGH_COURT, ("orissa", "odisha")),
    ("High Court of Patna", CourtLevel.HIGH_COURT, ("patna",)),
    ("High Court of Punjab and Haryana", CourtLevel.HIGH_COURT, ("punjab",)),
    ("High Court of Rajasthan", CourtLevel.HIGH_COURT, ("rajasthan",)),
    ("High Court of Sikkim", CourtLevel.HIGH_COURT, ("sikkim",)),
    ("High Court of Telangana", CourtLevel.HIGH_COURT, ("telangana",)),
    ("High Court of Tripura", CourtLevel.HIGH_COURT, ("tripura",)),
    ("High Court of Uttarakhand", CourtLevel.HIGH_COURT, ("uttarakhand",)),
    ("National Company Law Appellate Tribunal", CourtLevel.TRIBUNAL, ("nclat",)),
    ("National Company Law Tribunal", CourtLevel.TRIBUNAL, ("nclt",)),
    ("National Green Tribunal", CourtLevel.TRIBUNAL, ("green tribunal", "ngt")),
    ("Income Tax Appellate Tribunal", CourtLevel.TRIBUNAL, ("income tax appellate", "itat")),
    ("Customs, Excise and Service Tax Appellate Tribunal", CourtLevel.TRIBUNAL, ("cestat",)),
    ("Central Administrative Tribunal", CourtLevel.TRIBUNAL, ("administrative tribunal",)),
    ("National Consumer Disputes Redressal Commission", CourtLevel.TRIBUNAL, ("ncdrc",)),
    ("Securities Appellate Tribunal", CourtLevel.TRIBUNAL, ("securities appellate",)),
    ("Appellate Tribunal for Electricity", CourtLevel.TRIBUNAL, ("aptel", "for electricity")),
    ("Judicial Committee of the Privy Council", CourtLevel.OTHER, ("privy council",)),
    ("Federal Court of India", CourtLevel.OTHER, ("federal court",)),
]
_COURT_CODES = {"SCI": "Supreme Court of India", "7~26": "High Court of Delhi"}
_COURT_CODES["27~1"] = "High Court of Bombay"


def canonical_court(name: str | None) -> tuple[str, CourtLevel] | None:
    """('High Court of Bombay', HIGH_COURT) for 'Bombay High Court', or None if unknown."""
    if not name:
        return None
    key = re.sub(r"[^a-z ]", " ", name.lower())
    key = re.sub(r"\s+", " ", key)
    if name.strip() in _COURT_CODES:
        key = _COURT_CODES[name.strip()].lower()
    for canonical, level, keys in _COURTS:
        if any(k in key for k in keys):
            # "... High Court of Delhi" and "Delhi" alone are both the High Court; a tribunal or
            # the Supreme Court is only matched by its own words.
            if level == CourtLevel.HIGH_COURT and "high court" not in key and len(key.split()) > 2:
                continue
            return canonical, level
    return None


# ---- Dates -------------------------------------------------------------------------------

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1
)}  # fmt: skip


def parse_date(value: Any) -> dt.date | None:
    """A date from the formats the sources use; None when absent or not a valid date."""
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    try:
        if m := re.fullmatch(r"(\d{4})-(\d\d)-(\d\d)(?:[ T][\d:.]+)?", s):
            return dt.date(int(m[1]), int(m[2]), int(m[3]))
        if m := re.fullmatch(r"(\d\d?)[-/.](\d\d?)[-/.](\d{4})", s):
            return dt.date(int(m[3]), int(m[2]), int(m[1]))
        if m := re.fullmatch(r"([A-Za-z]{3,9})\.?\s+(\d\d?),?\s+(\d{4})", s):
            month = _MONTHS.get(m[1][:3].lower())
            return dt.date(int(m[3]), month, int(m[2])) if month else None
    except ValueError:
        return None
    return None


# ---- Judges ------------------------------------------------------------------------------

_HONORIFIC = re.compile(
    r"\b(?:HON'?BLE|THE|MR|MRS|MS|SHRI|SMT|KUM|JUSTICE|CHIEF|ACTING|DR)\b\.?", re.IGNORECASE
)
_TITLE_SUFFIX = re.compile(r"(?:,?\s*\b(?:C\.?\s*J\.?\s*I?|J\.?\s*J?|CJI)\.?)+\s*$", re.I)


def clean_judge(name: str) -> str | None:
    """'HON'BLE MR. JUSTICE VALMIKI J. MEHTA' -> 'VALMIKI J. MEHTA'; None if nothing is left."""
    s = name.replace("*", " ").replace(chr(0x2019), "'").strip(" ,;-")  # curly apostrophe
    s = _TITLE_SUFFIX.sub("", s)
    s = _HONORIFIC.sub(" ", s)
    s = re.sub(r"\s*\.\s*", ". ", s)  # "J.L.KAPUR" / "J. L. KAPUR" -> "J. L. KAPUR"
    s = re.sub(r"(?<=\b[A-Z])\. (?=[A-Z]\.)", ".", s.upper())  # -> "J.L. KAPUR"
    s = re.sub(r"\s+", " ", s).strip(" .,-")
    return s if len(re.sub(r"[^A-Z]", "", s)) >= 2 else None


def split_judges(value: str | None) -> list[str]:
    """Judge names from a comma/'and'-separated string, cleaned and de-duplicated."""
    if not value:
        return []
    parts = re.split(r",|;|\bAND\b|&", value, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        if (name := clean_judge(part)) and name not in out:
            out.append(name)
    return out


# "[A AND B, JJ.]", "(A, B and C JJ.)", "[A, C.J.]"; OCR turns "JJ.]" into "n.·1" or "JJ.)".
_BENCH_LINE = re.compile(
    r"[\[(]\s*(?P<names>[A-Z][^\[\]()]{2,400}?)\s*,?\s*(?:J\s*J|C\.\s*J\.?\s*I?|n)\.?\s*"
    r"(?:[\])]|·\s*1)",
    re.S,
)


def bench_from_text(text: str) -> list[str] | None:
    """The bench named in an SCR report's bench line, e.g. '[A.P. SEN AND L.M. SHARMA, JJ.]'.

    Only the first page is searched, and "[Per X, J.]" opinion notes are skipped. Returns None
    when there is no bench line (HC judgments, OCR damage).
    """
    head = text[:8000]
    for m in _BENCH_LINE.finditer(head):
        names = m.group("names")
        letters = re.sub(r"[^A-Za-z]", "", re.sub(r"\b(?:and|AND)\b", "", names))
        upper = sum(ch.isupper() for ch in letters) / max(1, len(letters))
        if re.match(r"(?i)per\b", names) or re.search(r"\d", names) or upper < 0.7:
            continue
        judges = split_judges(re.sub(r"\s+", " ", names))
        if judges and all(len(j) <= 40 for j in judges):
            return judges
    return None


# ---- Parties, case numbers, dispositions -------------------------------------------------

_VERSUS = re.compile(r"\s+(?:versus|vs\.?|v\.)\s+", re.IGNORECASE)


def split_title(title: str | None) -> tuple[str | None, str | None, str | None]:
    """(case number, petitioner, respondent) from 'RFA/436/2016 of A Vs B' or 'A versus B'."""
    if not title:
        return None, None, None
    number = None
    if m := re.match(r"^\s*([A-Z][A-Z().\- ]*\/\d+\/\d{4})\s+of\s+", title):
        number, title = m.group(1), title[m.end() :]
    parts = _VERSUS.split(title, maxsplit=1)
    if len(parts) != 2:
        return number, None, None
    petitioner, respondent = (re.sub(r"\s+", " ", p).strip(" .,") or None for p in parts)
    return number, petitioner, respondent


_DISPOSITIONS: list[tuple[str, Disposition]] = [
    (r"partly allowed|partly|allowed in part", "partly_allowed"),
    (r"remand|remitted", "remanded"),
    (r"referred|reference answered|larger bench", "referred"),
    (r"withdrawn", "withdrawn"),
    (r"direction", "directions"),
    (r"modif", "modified"),
    (r"dismiss", "dismissed"),
    (r"allow|set aside|leave granted & allowed", "allowed"),
    (r"dispos", "disposed"),
]


def normalise_disposition(value: str | None) -> Disposition | None:
    """'Appeal(s) allowed' -> 'allowed'; None for empty or non-dispositions ('JUDGEMENT')."""
    if not value:
        return None
    s = value.lower()
    for pattern, label in _DISPOSITIONS:
        if re.search(pattern, s):
            return label
    return None


# ---- Source records ----------------------------------------------------------------------


def _card_field(raw_html: str, label: str) -> str | None:
    m = re.search(rf"{label}\s*:\s*</span>\s*<font[^>]*>\s*([^<]*?)\s*</font>", raw_html)
    return html.unescape(m.group(1)).strip() or None if m else None


def parse_sc_card(raw_html: str | None) -> dict[str, Any]:
    """Coram (author starred), case number and bench size from the AWS SC ``raw_html`` card."""
    if not raw_html:
        return {}
    out: dict[str, Any] = {}
    if m := re.search(r"Coram\s*:\s*(.*?)</strong>", raw_html, re.S):
        names = html.unescape(re.sub(r"<[^>]+>", "", m.group(1)))  # the author's <sup>*</sup>
        coram = [part for part in names.split(",") if part.strip(" *")]
        out["coram"] = [n for p in coram if (n := clean_judge(p))]
        starred = [clean_judge(p) for p in coram if "*" in p]
        out["author"] = starred[0] if starred else None
    if number := _card_field(raw_html, "Case No"):
        out["case_number"] = number
    if (bench := _card_field(raw_html, "Bench")) and (m := re.match(r"(\d+)\s+Judges?", bench)):
        out["bench_size"] = int(m.group(1))
    return out


def _kg_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        parsed = ast.literal_eval(value) if value.startswith("[") else [value]
    except (ValueError, SyntaxError):
        parsed = json.loads(value) if value.startswith("[") else [value]
    return [str(v) for v in parsed]


# ---- Merged record -----------------------------------------------------------------------


@dataclass
class JudgmentMetadata:
    doc_id: str
    cnr: str | None = None
    title: str | None = None
    petitioner: str | None = None
    respondent: str | None = None
    case_number: str | None = None
    court: str | None = None
    court_level: CourtLevel | None = None
    judges: list[str] = field(default_factory=list)
    author: str | None = None
    bench_strength: int | None = None
    decision_date: dt.date | None = None
    citations: list[str] = field(default_factory=list)  # canonical, own citations
    neutral_citation: str | None = None
    disposition: Disposition | None = None
    disposition_raw: str | None = None
    headnote: str | None = None
    sources: dict[str, str] = field(default_factory=dict)  # field -> source
    conflicts: list[str] = field(default_factory=list)
    kanoongpt_match: Literal["cnr", "none", "rejected"] = "none"
    bench_counts: dict[str, int] = field(default_factory=dict)  # bench size per source

    def set(self, name: str, value: Any, source: Source) -> None:
        """Set a field from a source unless it is empty; the first source to set it wins."""
        if value in (None, "", []) or getattr(self, name) not in (None, "", []):
            return
        setattr(self, name, value)
        self.sources[name] = source

    def check(self, name: str, other: Any, source: Source) -> None:
        """Record a conflict if ``source`` disagrees with the value already set."""
        mine = getattr(self, name)
        if mine in (None, "", []) or other in (None, "", []):
            return
        if _comparable(mine) != _comparable(other):
            self.conflicts.append(f"{name}: {self.sources.get(name)}={mine!s} {source}={other!s}")

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["decision_date"] = self.decision_date.isoformat() if self.decision_date else None
        out["court_level"] = self.court_level.value if self.court_level else None
        return out

    def to_document(self, snapshot_id: str, licence: str, source_url: str | None) -> Document:
        """The corpus ``Document`` for this judgment (title falls back to the doc ID)."""
        return Document(
            id=self.doc_id,
            kind=DocumentKind.JUDGMENT,
            title=self.title or self.doc_id,
            court=self.court,
            court_level=self.court_level,
            bench_strength=self.bench_strength,
            judges=self.judges,
            decision_date=self.decision_date,
            cnr=self.cnr,
            citations=self.citations,
            source_url=source_url,
            licence=licence,
            corpus_snapshot=snapshot_id,
        )


def _comparable(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"[^a-z0-9]", "", value.lower())
    if isinstance(value, list):
        return sorted(_comparable(v) for v in value)
    return value


def _title_overlap(a: str | None, b: str | None) -> float:
    ta = set(re.findall(r"[a-z]{3,}", (a or "").lower()))
    tb = set(re.findall(r"[a-z]{3,}", (b or "").lower()))
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _add_citation(meta: JudgmentMetadata, raw: str | None, source: Source) -> None:
    if not raw:
        return
    c = parse_citation(raw)
    if c is None:
        meta.conflicts.append(f"citation: {source} gave unparseable {raw!r}")
        return
    if c.canonical not in meta.citations:
        meta.citations.append(c.canonical)
        meta.sources.setdefault("citations", source)
    if c.reporter in ("INSC", "HCNC"):
        if meta.neutral_citation and meta.neutral_citation != c.canonical:
            meta.conflicts.append(
                f"neutral_citation: {meta.neutral_citation} vs {source}={c.canonical}"
            )
        meta.set("neutral_citation", c.canonical, source)


def merge(
    doc_id: str,
    aws: dict[str, Any] | None,
    source: Literal["aws_sc", "aws_hc"],
    kg: dict[str, Any] | None = None,
    text: str | None = None,
) -> JudgmentMetadata:
    """Merge one judgment's AWS row, KanoonGPT row (joined on CNR) and text into a record."""
    meta = JudgmentMetadata(doc_id)
    aws = aws or {}
    if source == "aws_sc":
        card = parse_sc_card(aws.get("raw_html"))
        meta.set("cnr", aws.get("cnr"), source)
        meta.set("title", aws.get("title"), source)
        meta.set("petitioner", aws.get("petitioner"), source)
        meta.set("respondent", aws.get("respondent"), source)
        meta.set("case_number", card.get("case_number"), source)
        meta.set("judges", card.get("coram") or [], source)
        meta.set("author", card.get("author"), source)
        if meta.judges:
            meta.bench_counts["aws_sc coram"] = len(meta.judges)
        if card.get("bench_size"):
            meta.bench_counts["aws_sc bench"] = card["bench_size"]
        _add_citation(meta, aws.get("citation"), source)
        _add_citation(meta, aws.get("case_id"), source)
    else:
        number, petitioner, respondent = split_title(aws.get("title"))
        meta.set("cnr", aws.get("cnr"), source)
        meta.set("title", aws.get("title"), source)
        meta.set("case_number", number, source)
        meta.set("petitioner", petitioner, source)
        meta.set("respondent", respondent, source)
        meta.set("judges", split_judges(aws.get("judge")), source)
        if meta.judges:
            meta.bench_counts["aws_hc judges"] = len(meta.judges)
    if court := canonical_court(aws.get("court")):
        meta.set("court", court[0], source)
        meta.set("court_level", court[1], source)
    meta.set("decision_date", parse_date(aws.get("decision_date")), source)
    meta.set("disposition", normalise_disposition(aws.get("disposal_nature")), source)
    meta.set("disposition_raw", aws.get("disposal_nature") or None, source)

    if kg is not None:
        kg_date = parse_date(kg.get("decision_date"))
        same_date = meta.decision_date is None or kg_date is None or kg_date == meta.decision_date
        overlap = _title_overlap(meta.title, kg.get("case_title"))
        if kg.get("cnr_number") != meta.cnr or not (same_date or overlap >= 0.5):
            meta.kanoongpt_match = "rejected"
        else:
            meta.kanoongpt_match = "cnr"
            _merge_kanoongpt(meta, kg)

    if text:
        bench = bench_from_text(text)
        if bench:
            meta.bench_counts["text"] = len(bench)
            meta.set("judges", bench, "text")
        for c in find_citations(text[:400]):  # the report's own header, e.g. "[2023] 12 S.C.R."
            if c.reporter in ("SCR", "INSC", "HCNC"):
                _add_citation(meta, c.canonical, "text")

    _settle_bench(meta)
    return meta


def _settle_bench(meta: JudgmentMetadata) -> None:
    """Bench strength by strict majority of the sources that count the bench, else None.

    Counts come from coram lists (AWS SC card, KanoonGPT, the HC judge field), explicit sizes
    (the AWS card's "Bench : 2 Judges", KanoonGPT's "2 Judges") and the SCR bench line. AWS
    SC's single ``judge`` column names only the author, so it is not a count.
    """
    counts = meta.bench_counts
    if not counts:
        return
    if len(set(counts.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in counts.items())
        meta.conflicts.append(f"bench_strength: {detail}")
    value, votes = Counter(counts.values()).most_common(1)[0]
    if votes * 2 > len(counts):
        meta.bench_strength = value
        meta.sources["bench_strength"] = next(k for k, v in counts.items() if v == value)
    # A single-judge bench's judgment is that judge's.
    if meta.bench_strength == 1 and len(meta.judges) == 1:
        meta.set("author", meta.judges[0], meta.sources["judges"])  # type: ignore[arg-type]


def _merge_kanoongpt(meta: JudgmentMetadata, kg: dict[str, Any]) -> None:
    src: Source = "kanoongpt"
    coram = [n for v in _kg_list(kg.get("coram_members")) if (n := clean_judge(v))]
    meta.check("judges", coram, src)
    meta.set("judges", coram, src)
    if coram:
        meta.bench_counts["kanoongpt coram"] = len(coram)
    if m := re.match(r"(\d+)\s+Judges?", kg.get("bench_name") or ""):
        meta.bench_counts["kanoongpt bench"] = int(m.group(1))
    for name in ("petitioner", "respondent"):
        meta.check(name, kg.get(f"party_{name}"), src)
        meta.set(name, kg.get(f"party_{name}"), src)
    meta.set("case_number", kg.get("docket_number"), src)
    meta.check("decision_date", parse_date(kg.get("decision_date")), src)
    meta.set("decision_date", parse_date(kg.get("decision_date")), src)
    if court := canonical_court(kg.get("court_name")):
        meta.check("court", court[0], src)
        meta.set("court", court[0], src)
        meta.set("court_level", court[1], src)
    kg_disp = normalise_disposition(kg.get("disposition_text"))
    meta.check("disposition", kg_disp, src)
    meta.set("disposition", kg_disp, src)
    meta.set("disposition_raw", kg.get("disposition_text"), src)
    _add_citation(meta, kg.get("law_report_citation"), src)
    _add_citation(meta, kg.get("neutral_citation"), src)
    meta.set("headnote", kg.get("headnote_text"), src)
