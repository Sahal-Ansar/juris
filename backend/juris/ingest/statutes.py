"""Section-level statute ingestion (PLAN 3.5).

Input: the cleaned text of an India Code consolidated Act (3.2): an "arrangement of sections",
then the body, page by page, each page ending in its footnotes ("1.\\nSubs. by Act 18 of
2018, s. 3, ...") and a bare page number. Output: one ``StatuteSection`` per section, sub-sections
kept with their section, amendment notes resolved from the footnote markers (``[1]``, ``1 [``)
on the same page.

Section starts are recognised only in the order the arrangement lists them, so numbered
clauses, illustrations or footnotes can't pose as sections. Temporal validity is coarse unless
curated (D-014): ``in_force_from`` is the Act's commencement from ``configs/statutes/acts.yaml``;
``configs/statutes/amendments.yaml`` overrides it per section with dates checked against the
official text, and only those sections get ``amendments_curated=True``.
"""

import datetime as dt
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from juris.config import REPO_ROOT
from juris.models import StatuteMeta

STATUTE_CONFIGS = REPO_ROOT / "configs" / "statutes"

# ---- Acts and aliases --------------------------------------------------------------------


@dataclass(frozen=True)
class Act:
    act_id: str
    title: str
    year: int
    act_no: str
    commencement: dt.date | None
    commencement_source: str | None
    aliases: tuple[str, ...] = ()


def _alias_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9 ]", " ", name.lower().replace("&", " and "))
    key = re.sub(r"\bthe\b", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def load_acts(path: Path | None = None) -> dict[str, Act]:
    data = yaml.safe_load((path or STATUTE_CONFIGS / "acts.yaml").read_text(encoding="utf-8"))
    acts = {}
    for act_id, a in data["acts"].items():
        commencement = a.get("commencement") or {}
        acts[act_id] = Act(
            act_id=act_id,
            title=a["title"],
            year=int(a["year"]),
            act_no=a["act_no"],
            commencement=commencement.get("date"),
            commencement_source=commencement.get("source"),
            aliases=tuple(a.get("aliases", [])),
        )
    return acts


def resolve_act(name: str, acts: dict[str, Act]) -> str | None:
    """Canonical Act ID for 'ICA', 'Contract Act', 'Indian Contract Act, 1872' ...; else None.

    Any alias may carry the Act's year ('Contract Act, 1872'); a different year is a different
    Act ('Specific Relief Act, 1877' is the 1963 Act's predecessor) and doesn't resolve.
    """
    key = _alias_key(name)
    m = re.fullmatch(r"(.*?)\s*(\d{4})", key)
    base, year = (m.group(1), int(m.group(2))) if m else (key, None)
    for act in acts.values():
        names = {_alias_key(n) for n in (act.act_id, act.title, *act.aliases)}
        names = {re.sub(r"\s*\d{4}$", "", n) for n in names}  # aliases stored with a year
        if key in names or (base in names and year in (None, act.year)):
            return act.act_id
    return None


# ---- Sections ----------------------------------------------------------------------------


@dataclass
class StatuteSection:
    act_id: str
    act: str
    act_year: int
    section: str
    title: str
    text: str
    char_start: int  # offsets into the cleaned Act text (3.2)
    char_end: int
    page_start: int
    page_end: int
    chapter: str | None = None
    repealed: bool = False
    sub_sections: list[str] = field(default_factory=list)  # "(1)", "(2)", ... in order
    amendment_notes: list[str] = field(default_factory=list)  # footnotes on this section
    amended_by: list[str] = field(default_factory=list)  # "Act 18 of 2018", from the notes
    in_force_from: dt.date | None = None
    in_force_to: dt.date | None = None
    amendments_curated: bool = False
    curation_source: str | None = None

    def meta(self) -> StatuteMeta:
        return StatuteMeta(
            act=self.act,
            section=self.section,
            title=self.title,
            in_force_from=self.in_force_from,
            in_force_to=self.in_force_to,
            amended_by=self.amended_by,
            amendments_curated=self.amendments_curated,
        )

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("in_force_from", "in_force_to"):
            out[key] = out[key].isoformat() if out[key] else None
        return out


def section_key(no: str) -> tuple[int, str]:
    m = re.fullmatch(r"(\d+)([A-Z]*)", no)
    return (int(m.group(1)), m.group(2)) if m else (0, no)


_DASHES = chr(0x2014) + chr(0x2013)  # India Code heads a section "N. Title." + em dash + text
_BODY_START = re.compile(rf"(?m)^\W{{0,3}}1\.[ \t]*[A-Z][^\n]{{0,150}}?[{_DASHES}]")
_ARRANGEMENT_NO = re.compile(r"(?:^|(?<=\s)|(?<=\]))(\d{1,3}[A-Z]{0,3})\.(?=\s|\[)")


def _arrangement_title(segment: str) -> str:
    kept: list[str] = []
    for line in (ln.strip() for ln in segment.split("\n")):
        if not line:
            continue
        if _PAGE_NO.fullmatch(line) or line == "SECTIONS" or _HEADING.match(line):
            break
        if kept and _is_cross_heading(line):
            break
        kept.append(line)
    title = re.sub(r"\s+", " ", " ".join(kept)).strip(" .")
    return re.sub(r"\.\s*$", "", title)


def arrangement(text: str) -> tuple[list[tuple[str, str]], int]:
    """(section, title) pairs listed before the body, in order, and the body's start offset."""
    m = _BODY_START.search(text)
    if not m:
        raise ValueError("no section 1 heading found")
    head = text[: m.start()]
    found = list(_ARRANGEMENT_NO.finditer(head))
    entries: list[tuple[str, str]] = []
    for i, n in enumerate(found):
        no = n.group(1)
        if any(no == e[0] for e in entries):
            continue
        if entries and section_key(no) <= section_key(entries[-1][0]):
            continue
        stop = found[i + 1].start() if i + 1 < len(found) else len(head)
        entries.append((no, _arrangement_title(head[n.end() : stop])))
    return entries, m.start()


_PAGE_NO = re.compile(r"^\d{1,4}$")
_FOOTNOTE_START = re.compile(r"^\d{1,2}\.$")
_HEADING = re.compile(r"^\[?\s*(CHAPTER|PART)\b[^\n]*$")
_EDITORIAL = re.compile(r"^\[Vide [^\]]*\]$")
_MARKERS = re.compile(r"\[(\d{1,2})\]|(?:(?<=^)|(?<=[\s\w.,;:]))(\d{1,2})\s*(?=\[|\*)")
_SUB_SECTION = re.compile(r"^\W{0,3}\((\d{1,3}[A-Z]?)\s*\)")
# "Subs. by Act 18 of 2018, s. 3", "rep. by the Indian Sale of Goods Act, 1930 (3 of 1930)"
_AMENDED_BY = re.compile(
    r"\b(?:Subs|Ins|Omitted|omitted|Rep|rep|Added|Inserted|inserted|Renumbered|substituted)\b"
    r"\.?[^;]{0,160}?\bby\s+(?:the\s+)?(?:[A-Z][\w.,&' -]{0,90}?\((\d{1,3})\s+of\s+(\d{4})\)"
    r"|Act\s+(\d{1,3})\s+of\s+(\d{4}))"
)


@dataclass
class _Line:
    text: str
    start: int
    page: int


def _pages(text: str, body_start: int) -> tuple[list[_Line], dict[int, dict[int, str]]]:
    """Body lines with offsets and pages, and each page's footnotes {page: {n: text}}."""
    lines: list[_Line] = []
    notes: dict[int, dict[int, str]] = {}
    page, in_notes = 1, False
    note_text: list[str] = []
    pos = body_start
    for raw in text[body_start:].split("\n"):
        stripped = raw.strip()
        if _PAGE_NO.fullmatch(stripped):
            if note_text:
                notes[page] = _split_notes("\n".join(note_text))
            page, in_notes, note_text = page + 1, False, []
        elif in_notes or _FOOTNOTE_START.fullmatch(stripped):
            in_notes = True
            note_text.append(stripped)
        elif stripped and not _EDITORIAL.fullmatch(stripped):
            lines.append(_Line(raw, pos, page))
        pos += len(raw) + 1
    if note_text:
        notes[page] = _split_notes("\n".join(note_text))
    return lines, notes


def _split_notes(block: str) -> dict[int, str]:
    """Footnotes are numbered 1, 2, 3 ... on each page; split the block on that sequence."""
    out: dict[int, str] = {}
    n, pos, starts = 1, 0, []
    while m := re.compile(rf"(?:^|(?<=[\s.]))({n})\.(?=\s|$)").search(block, pos):
        starts.append((n, m.start(), m.end()))
        n, pos = n + 1, m.end()
    for i, (k, _, end) in enumerate(starts):
        stop = starts[i + 1][1] if i + 1 < len(starts) else len(block)
        out[k] = re.sub(r"\s+", " ", block[end:stop]).strip()
    return out


def _section_start(line: str, expected: str) -> int | None:
    """Offset in ``line`` where section ``expected`` starts ("1[16. ...", "- 3.Com...")."""
    lead = re.match(rf"^[\s{_DASHES}\-]*", line)
    i = lead.end() if lead else 0
    for k in (0, 1, 2):  # an optional footnote marker before "[", e.g. "1 [10." / "2[19A."
        head = line[i : i + k]
        if k and not head.isdigit():
            continue
        m = re.match(
            rf"\s*\[?\s*({re.escape(expected)})\s*\.(?=[\s\[\"'A-Z({_DASHES}]|$)", line[i + k :]
        )
        if m:
            return i
    return None


def _normalise(text: str) -> str:
    text = re.sub(r"\[(\d{1,2})\]", "", text)  # footnote markers
    text = re.sub(r"(?:(?<=^)|(?<=\n))\d{1,2}\s*(?=\[)", "", text)
    text = re.sub(r"(?<=[\s\w.,;:])\d{1,2}\s*(?=\*)", "", text)
    text = re.sub(r"\((\w{1,6})\s+\)", r"(\1)", text)  # "(1 )" -> "(1)"
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r" *\n *", "\n", text).strip()


def _amended_by(notes: list[str]) -> list[str]:
    """Amending Acts named after an amendment verb ("Subs. by Act 18 of 2018")."""
    out: list[str] = []
    for note in notes:
        for m in _AMENDED_BY.finditer(note):
            number, year = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            if (ref := f"Act {number} of {year}") not in out:
                out.append(ref)
    return out


_SCHEDULE = re.compile(
    r"^\s*(?:\d{1,2}\s*)?\[?\s*(?:THE\s+)?(?:FIRST\s+|SECOND\s+|THIRD\s+)?SCHEDULE\b"
)
_REPEALED = re.compile(r"\[\s*Repealed|\]\s*Rep\.|^\s*Rep\.|\brep\. by\b", re.IGNORECASE)


def split_act(text: str, act: Act) -> list[StatuteSection]:
    """Split one Act's cleaned text into sections (see module docstring)."""
    entries, body_start = arrangement(text)
    # The arrangement can skip a section the body has (Contract Act ss. 113-114): fill gaps.
    expected: list[str] = []
    for no, _ in entries:
        if expected and no.isdigit() and (last_no := section_key(expected[-1])[0]) < int(no) - 1:
            expected += [str(n) for n in range(last_no + 1, int(no))]
        expected.append(no)
    titles = dict(entries)
    lines, notes = _pages(text, body_start)
    starts: list[tuple[int, str, int]] = []  # (line index, section no, offset in line)
    nxt = 0
    for idx, line in enumerate(lines):
        # look a few entries ahead: a section missing from the body must not stall the rest
        for j in range(nxt, min(nxt + 4, len(expected))):
            off = _section_start(line.text, expected[j])
            if off is not None:
                starts.append((idx, expected[j], off))
                nxt = j + 1
                break
    if not starts:
        return []
    # the last section ends where the Schedule begins
    end_of_sections = next(
        (i for i in range(starts[-1][0] + 1, len(lines)) if _SCHEDULE.match(lines[i].text)),
        len(lines),
    )
    headings = {
        i: re.sub(r"\s+", " ", ln.text.strip())
        for i, ln in enumerate(lines)
        if _HEADING.match(ln.text.strip())
    }

    found: dict[str, StatuteSection] = {}
    chapter: str | None = None
    for k, (idx, no, off) in enumerate(starts):
        stop = starts[k + 1][0] if k + 1 < len(starts) else end_of_sections
        for h in sorted(i for i in headings if i <= idx):
            chapter = headings[h]
        body = [ln for i, ln in enumerate(lines[idx:stop], idx) if i not in headings]
        # drop trailing cross-headings in capitals ("CONTRACTS WHICH CAN BE ...")
        while len(body) > 1 and (_is_cross_heading(body[-1].text) or _is_run_in(body[-1].text)):
            body.pop()
        first, last = body[0], body[-1]
        raw = first.text[off:] + "".join("\n" + ln.text for ln in body[1:])
        page_notes: list[str] = []
        for ln in body:
            for m in _MARKERS.finditer(ln.text):
                n = int(m.group(1) or m.group(2))
                if (note := notes.get(ln.page, {}).get(n)) and note not in page_notes:
                    page_notes.append(note)
        text_norm = _normalise(raw)
        repealed = bool(_REPEALED.search(text_norm[:300]))
        found[no] = StatuteSection(
            act_id=act.act_id,
            act=act.title,
            act_year=act.year,
            section=no,
            title=_title(no, titles.get(no, ""), text_norm),
            text=text_norm,
            char_start=first.start + off,
            char_end=last.start + len(last.text),
            page_start=first.page,
            page_end=last.page,
            chapter=chapter,
            repealed=repealed,
            sub_sections=_sub_sections(text_norm),
            amendment_notes=page_notes,
            amended_by=_amended_by(page_notes),
            in_force_from=act.commencement,
        )

    # Sections the arrangement lists as repealed but the body only marks "* * *" get a stub.
    sections: list[StatuteSection] = []
    for no in expected:
        title = titles.get(no, "")
        if no in found:
            sections.append(found[no])
        elif _REPEALED.search(title) and sections:
            prev = sections[-1]
            sections.append(
                StatuteSection(
                    act_id=act.act_id,
                    act=act.title,
                    act_year=act.year,
                    section=no,
                    title="",
                    text=f"{no}. [Repealed.]",
                    char_start=prev.char_end,
                    char_end=prev.char_end,
                    page_start=prev.page_end,
                    page_end=prev.page_end,
                    chapter=prev.chapter,
                    repealed=True,
                    amendment_notes=[x for x in prev.amendment_notes if _REPEALED.search(x)],
                    in_force_from=act.commencement,
                )
            )
    return sections


def _title(no: str, listed: str, text: str) -> str:
    """The arrangement's title, or the body heading's when the arrangement only says Repealed."""
    if listed and not _REPEALED.search(listed):
        return listed
    head = re.match(
        rf"^\[?\s*{re.escape(no)}\s*\.\s*\[?(?P<title>[^\n\]]*?)\s*\]?\s*(?:[.]?\s*[{_DASHES}]|\]|$)",
        text,
    )
    return (head.group("title") if head else "").strip(f" .[]'\"-{_DASHES}")


def _is_run_in(line: str) -> bool:
    """A cross-heading in sentence case ("Contract of sale"): short, no closing punctuation."""
    s = line.strip()
    return 0 < len(s) <= 60 and s[0].isupper() and not re.search(r"[.;:,\]\)\-]$", s)


def _sub_sections(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        m = _SUB_SECTION.match(line) or re.search(
            rf"[{_DASHES}]\s*\((\d{{1,3}}[A-Z]?)\)", line[:200]
        )
        if m and (label := f"({m.group(1)})") not in out:
            out.append(label)
    return out


def _is_cross_heading(line: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", line)
    return len(letters) >= 6 and letters.isupper() and not re.search(r"\d", line)


# ---- Curated temporal validity (D-014) ---------------------------------------------------


def load_amendments(path: Path | None = None) -> list[dict[str, Any]]:
    data = yaml.safe_load((path or STATUTE_CONFIGS / "amendments.yaml").read_text("utf-8"))
    entries: list[dict[str, Any]] = data.get("amendments") or []
    return entries


def _expand(spec: str | int) -> list[str]:
    """'76-123' -> ['76', ..., '123']; '14A' -> ['14A']."""
    s = str(spec)
    if m := re.fullmatch(r"(\d+)-(\d+)", s):
        return [str(n) for n in range(int(m.group(1)), int(m.group(2)) + 1)]
    return [s]


def apply_amendments(sections: list[StatuteSection], entries: list[dict[str, Any]]) -> int:
    """Apply curated entries to matching sections; returns how many sections were updated."""
    by_key = {(s.act_id, s.section): s for s in sections}
    updated = set()
    for entry in entries:
        for spec in entry["sections"]:
            for no in _expand(spec):
                s = by_key.get((entry["act"], no))
                if s is None:
                    continue
                if "in_force_from" in entry:
                    s.in_force_from = entry["in_force_from"]
                if "in_force_to" in entry:
                    s.in_force_to = entry["in_force_to"]
                if entry.get("by") and entry["by"] not in s.amended_by:
                    s.amended_by.append(entry["by"])
                s.amendments_curated = True
                s.curation_source = entry["source"]
                updated.add((s.act_id, s.section))
    return len(updated)
