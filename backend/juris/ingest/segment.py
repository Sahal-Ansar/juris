"""Paragraph segmentation of cleaned judgment text (PLAN 3.3).

Explicit numbering: every line that starts with a paragraph number is a candidate (``12.``,
``12)``, ``(12)``, ``38-39.``, ``12.`` alone on a line, ``10.In ...``, and the same after a
one-character SCR margin token, e.g. ``A 36.`` or ``8 30.``; a first paragraph printed on the
judge's line is split off). A *chain* is a run of candidates in document order whose numbers
rise by one; small gaps are allowed and recorded (numbers lost to OCR). A judgment can hold
several chains, one per opinion, since separate opinions restart at 1. Chains are validated so
that numbered items inside a paragraph ("(1) ... (2) ...", a quoted statute's "1. 2. 3.") and
the reporter's numbered headnote holdings can't pose as paragraph numbering: a chain must be of
one style and spread over its region of the text, not bunched in one place. A paragraph whose
number also opens a nearby line (a quoted paragraph, issue or clause) is flagged ``contested``.
Method and hand-check: docs/data/segmentation_check.md.

Without a valid chain (most judgments before ~2000), paragraphs are inferred from layout: a
short line ending a sentence closes a paragraph. Inferred paragraphs are numbered ``p-1``,
``p-2`` ... and flagged ``numbering="inferred"``, never passed off as the judgment's own.
Front matter (title, bench, headnote, counsel) is split off, and the SCR headnote is labelled
when detectable.
"""

import bisect
import re
import statistics
from dataclasses import dataclass, field, replace
from typing import Literal

Section = Literal["front", "headnote", "body"]
Numbering = Literal["explicit", "inferred"]
Style = Literal["dot", "bracket"]

_CANDIDATE = re.compile(
    r"^(?:[A-Ha-h0-9~.·]\s+)?"  # optional SCR margin token (letter, or OCR'd as 8/~/.)
    r"(?:\((?P<bracket>\d{1,3})\)|(?P<num>\d{1,3})(?:\s*[-\u2013]\s*(?P<to>\d{1,3}))?[.)])"
    r"(?:\s+(?=\S)|\s*$|(?=[A-Z\"'(]))"
)
_SUB_MARKER = re.compile(r"^\(((?:[ivxlc]{1,6})|[a-z]|\d{1,2})\)\s", re.I)
_BODY_START = re.compile(
    r"(?i)^\s*((?:j\s*u\s*d\s*g\s*m\s*e\s*n\s*t|o\s*r\s*d\s*e\s*r)\s*[:.-]?\s*$|"
    r"judgment of the court was "
    r"delivered by|the judgment of the court was delivered|"
    r".{0,60}\b(?:j\.|cji\b\.?)\s*[:-]?\s*(?:\([^)]{0,60}\)\s*)?$)"  # "X, J. (concurring)"
)
# "The Judgment of Wanchoo and Shah, JJ. was delivered by", "The following Judgment/Order of
# the Court was delivered by", "1953. Nov. 16. The Judgment of the Court were delivered by":
# matched on a line joined with the next two, since the names of a large bench wrap.
_DELIVERED = re.compile(
    r"(?i)^.{0,40}?\bthe\s+(?:following\s+)?(?:judgments?|orders?)(?:\s*/\s*orders?)?\b"
    r".{0,200}?\b(?:was|were)\s+delivered\b"
)
# The SCR opening of a judgment: the judge's name, "J." and a dash, then the text on the same
# line ("MUKHERJEA J.-The facts ...", "KHANNA, J.-This judgment ...", "BHAN, J. : This ...").
_JUDGE_OPENING = re.compile(
    r"^\s*[A-Z][\w.'\u2019 ]{1,40}?,?\s+(?:C\.\s*)?J\.\s*[-\u2013\u2014:]{1,2}\s*[A-Z(\"']"
)
# Attribution lines after "delivered by", skipped: "Shah, J. delivered a dissenting opinion."
# and a wrapped "Opinion." (a judge's name alone stays, as the body's first line elsewhere).
_ATTRIBUTION = re.compile(
    r"(?i)^\s*(?:.{0,80}\b(?:delivered|ga\.?ve)\s+(?:a\s+)?(?:separate|dissenting|concurring)"
    r"\b.{0,20}|opinions?\.?)\s*$"
)
MAX_FRONT_SHARE = 0.6  # front matter (with a long headnote) never runs this far into a judgment
_HEADNOTE_START = re.compile(r"(?i)^\s*(headnotes?\b|held\s*[:,-])")
# The jurisdiction line is upper case ("CIVIL APPELLATE JURISDICTION :"); in lower case the same
# words occur in the body ("... appellate jurisdiction over the decision").
_HEADNOTE_END = re.compile(
    r"^\s*(?:(?:CIVIL|CRIMINAL|ORIGINAL|APPELLATE)\b.{0,40}\bJURIS|"
    r"(?i:list of citations|case law cited|from the judgment and order))"
)
_SENTENCE_END = re.compile(r"[.:;?!\"')]\s*$")
MIN_CHAIN = 3
MAX_GAP = 3
MAX_START = 6  # a chain may start late when the first numbers are lost (at -0.5 each)
LONG_CHAIN = 10  # chains this long are accepted without the spread check
MIN_BRACKET_CHAIN = 20  # "(1) (2)" paragraphs are rare; shorter runs are footnotes or tables
MIN_SPREAD = 0.5


@dataclass(frozen=True)
class Paragraph:
    no: str
    text: str
    char_start: int
    char_end: int
    section: Section
    numbering: Numbering
    page_start: int
    page_end: int
    part: int = 1  # opinion/chain index within the judgment (numbering restarts per part)
    markers: tuple[tuple[str, int], ...] = ()  # sub-markers "(iii)", "(a)" with clean offsets
    # Another line within two paragraphs carries the same number (a quoted paragraph, issue
    # or clause): the choice may be wrong, so pinpoint citations should check the text.
    contested: bool = False


@dataclass
class Segmentation:
    paragraphs: list[Paragraph]
    numbering: Literal["explicit", "inferred"]
    parts: int = 0
    missing_numbers: list[int] = field(default_factory=list)
    rejected_candidates: int = 0

    @property
    def explicit(self) -> list[Paragraph]:
        return [p for p in self.paragraphs if p.numbering == "explicit"]


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Candidate:
    line: int
    first: int  # paragraph number (first of a range like "38-39")
    last: int
    style: Style
    punct: str = "."  # "." , ")" or "(": judgments keep one; quoted material often differs
    quoted_like: bool = False  # looks like a quoted item: an issue ("Whether ...") or clause
    after_colon: bool = False  # the previous line is a lead-in ("... as under:-")


# SCR prints the first paragraph on the judge's line: "S.B. SINHA, J. 1. Leave granted." or
# "Dr. ARIJIT PASAYAT. 1. Leave granted." (not a headnote's "HELD: 1. ...").
_INLINE_FIRST = re.compile(
    r"^(?P<head>.{0,60}?(?:\bJ\b\s*[.,:~]?|[A-Z]{3,}\.))\s+(?=1\s*[.)]\s+\S)"
)


def _lines(text: str) -> list[_Line]:
    out, pos = [], 0
    for raw in text.split("\n"):
        m = _INLINE_FIRST.match(raw)
        head = m.group("head") if m else ""
        if (
            m
            and "held" not in head.lower()
            and (re.search(r"\bJ\b\W*$", head) or not re.search(r"[a-z]{3}", head))
        ):
            out.append(_Line(head, pos, pos + len(head)))
            out.append(_Line(raw[m.end() :], pos + m.end(), pos + len(raw)))
        else:
            out.append(_Line(raw, pos, pos + len(raw)))
        pos += len(raw) + 1
    return out


def candidate_of(line: str, index: int = 0) -> Candidate | None:
    m = _CANDIDATE.match(line)
    if not m:
        return None
    if m.group("bracket"):
        n = int(m.group("bracket"))
        return Candidate(index, n, n, "bracket", "(")
    first = int(m.group("num"))
    last = int(m.group("to")) if m.group("to") else first
    if last < first or last - first > 5:
        last = first
    punct = ")" if m.group(0).rstrip().endswith(")") or ")" in m.group(0)[-3:] else "."
    quoted = bool(_QUOTED_ITEM.match(line[m.end() :]))
    return Candidate(index, first, last, "dot", punct, quoted)


# Issues framed by a court ("3. Whether the plaintiff ...", "4. Did the father ...") and
# agreement clauses ("10. That Party No.2 ...") are numbered items quoted inside paragraphs, as
# are elided paragraphs of a quoted judgment ("23. XXX XXX", "24. ...", "25. * * *").
_QUOTED_ITEM = re.compile(
    r"(?:whether|did|does|do|is|are|was|were|what|which|to what|how|that)\b|"
    r"(?:x{3}|\.{3}|…|\*)",
    re.I,
)
SAME_PUNCT_BONUS = 0.25  # "1) 2) 3)" or "1. 2. 3.": quoted paragraphs often use the other
FOREIGN_PUNCT_PENALTY = 1.0  # per member not in the chain's own (first member's) punctuation
CLUSTER_PENALTY = 0.3  # list items sit on consecutive lines; real paragraphs are spaced out
QUOTED_ITEM_PENALTY = 0.6
AFTER_COLON_PENALTY = 0.4  # "framed the following issues:-" / "reads as under:" then a list
_LEAD_IN = re.compile(r"[:\u2014]\s*-?\s*$")


def best_chain(candidates: list[Candidate]) -> list[int]:
    """Indices of the best rising chain (one style).

    Score per member: +1, -0.5 per skipped number, +0.25 for keeping the predecessor's
    punctuation, -1 if its punctuation differs from the chain's first member (one OCR misread
    costs little; a run of quoted "N." paragraphs inside a "N)" judgment can't win), -0.3 if
    it is within 2 lines of the predecessor, -0.6 if it reads like a quoted issue/clause,
    -0.4 if it follows a lead-in colon. A chain starts at 1-3. Ties go to the later
    predecessor (the candidate nearest the paragraph, not a headnote holding or list item
    earlier on), and to the earlier end, so trailing stray numbers don't extend it.
    """
    n = len(candidates)
    neg = float("-inf")
    score = [neg] * n
    back = [-1] * n
    home = [c.punct for c in candidates]  # punctuation of the chain's first member
    for i, c in enumerate(candidates):
        own = -(QUOTED_ITEM_PENALTY if c.quoted_like else 0)
        own -= AFTER_COLON_PENALTY if c.after_colon else 0
        if 1 <= c.first <= MAX_START:
            score[i] = 1 - 0.5 * (c.first - 1) + own
        for j in range(i):
            p = candidates[j]
            step = c.first - p.last
            if p.style == c.style and 1 <= step <= MAX_GAP and score[j] > neg:
                s = score[j] + 1 - 0.5 * (step - 1) + own
                s += SAME_PUNCT_BONUS if p.punct == c.punct else 0
                s -= FOREIGN_PUNCT_PENALTY if c.punct != home[j] else 0
                s -= CLUSTER_PENALTY if c.line - p.line <= 2 else 0
                s = round(s, 6)  # exact ties, so the tie-break rule applies
                if s >= score[i]:
                    score[i], back[i], home[i] = s, j, home[j]
    if not n or max(score) == neg:
        return []
    end = max(range(n), key=lambda i: (score[i], -i))
    chain = []
    while end != -1:
        chain.append(end)
        end = back[end]
    return chain[::-1]


def _spread_ok(members: list[Candidate], lines: list[_Line], region_end: int) -> bool:
    # Long "1. 2. 3." runs are paragraph numbering; bracket runs must always spread (a
    # "(1) ... (14)" list of charges is long but bunched in one part of the judgment).
    if len(members) >= LONG_CHAIN and members[0].style != "bracket":
        return True
    first = lines[members[0].line].start
    last = lines[members[-1].line].start
    return (last - first) >= MIN_SPREAD * max(1, region_end - first)


def _opinion_starts_before(lines: list[_Line], line: int, window: int = 10) -> bool:
    """A separate opinion opens with a header ('D.Y. CHANDRACHUD, J.', 'JUDGMENT', 'ORDER')."""
    return any(_BODY_START.match(lines[i].text) for i in range(max(0, line - window), line))


def _valid(members: list[Candidate], lines: list[_Line], region_end: int, extra: bool) -> bool:
    if len(members) < MIN_CHAIN:
        return False
    # Old reports never number paragraphs "(1) (2)": short bracket runs are enumerations or
    # footnote markers ("(3) A.I.R. 1939 Sind 56").
    if members[0].style == "bracket" and len(members) < MIN_BRACKET_CHAIN:
        return False
    # A later chain is a separate opinion only if an opinion header precedes it; otherwise it
    # is a numbered list inside a paragraph (e.g. directions at the end of a judgment).
    if extra and not _opinion_starts_before(lines, members[0].line):
        return False
    return _spread_ok(members, lines, region_end)


def find_chains(
    candidates: list[Candidate], lines: list[_Line], text_end: int
) -> list[list[Candidate]]:
    """Non-overlapping valid chains, left to right (one per opinion)."""
    chains: list[list[Candidate]] = []
    regions = [(0, len(lines))]  # line ranges still free
    while regions:
        lo, hi = regions.pop()
        pool = [c for c in candidates if lo <= c.line < hi]
        region_end = lines[hi - 1].end if hi > lo else text_end
        members = [pool[i] for i in best_chain(pool)]
        if not _valid(members, lines, region_end, extra=bool(chains)):
            continue
        chains.append(members)
        regions.append((lo, members[0].line))
        regions.append((members[-1].line + 1, hi))
    return sorted(chains, key=lambda ch: ch[0].line)


def _page(page_starts: list[int], offset: int) -> int:
    return max(1, bisect.bisect_right(page_starts, offset))


def _markers(lines: list[_Line]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (f"({m.group(1)})", line.start) for line in lines[1:] if (m := _SUB_MARKER.match(line.text))
    )


def _make(
    lines: list[_Line],
    no: str,
    section: Section,
    numbering: Numbering,
    text: str,
    page_starts: list[int],
    part: int = 1,
    contested: bool = False,
) -> Paragraph:
    start, end = lines[0].start, lines[-1].end
    return Paragraph(
        no=no,
        text=text[start:end],
        char_start=start,
        char_end=end,
        section=section,
        numbering=numbering,
        page_start=_page(page_starts, start),
        page_end=_page(page_starts, max(start, end - 1)),
        part=part,
        markers=_markers(lines),
        contested=contested,
    )


def _infer_blocks(lines: list[_Line]) -> list[list[_Line]]:
    """Split lines into paragraphs by layout: a short line that ends a sentence closes one."""
    widths = [len(line.text) for line in lines if line.text.strip()]
    if not widths:
        return []
    typical = statistics.median(widths)
    blocks: list[list[_Line]] = []
    current: list[_Line] = []
    for line in lines:
        if not line.text.strip():
            continue
        current.append(line)
        if len(line.text) < 0.7 * typical and _SENTENCE_END.search(line.text):
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


def _headnote_span(lines: list[_Line]) -> tuple[int, int]:
    """Line range of a reporter's headnote, or (0, 0) when its start or end isn't found."""
    start = next((i for i, line in enumerate(lines) if _HEADNOTE_START.match(line.text)), None)
    if start is None:
        return 0, 0
    end = next(
        (i for i in range(start + 1, len(lines)) if _HEADNOTE_END.search(lines[i].text)), None
    )
    return (start, end) if end is not None else (0, 0)


def _front_sections(lines: list[_Line]) -> list[tuple[Section, list[_Line]]]:
    start = next((i for i, line in enumerate(lines) if _HEADNOTE_START.match(line.text)), None)
    if start is None:
        return [("front", lines)] if lines else []
    end = next(
        (i for i in range(start + 1, len(lines)) if _HEADNOTE_END.search(lines[i].text)),
        len(lines),
    )
    parts: list[tuple[Section, list[_Line]]] = []
    if lines[:start]:
        parts.append(("front", lines[:start]))
    parts.append(("headnote", lines[start:end]))
    if lines[end:]:
        parts.append(("front", lines[end:]))
    return parts


def _body_start(lines: list[_Line]) -> int:
    """First body line of a judgment without numbered paragraphs (0 if no marker is found).

    The marker is the "judgment ... was delivered by" attribution (the attribution lines after
    it are skipped), a judge's opening line ("DAS J.-This appeal ...", where the body starts on
    that line), or a judgment/judge heading line. Only the first 60% of the text is searched:
    a late "ORDER" or "..., J." line is not where the body begins.
    """
    limit = int(len(lines) * MAX_FRONT_SHARE)
    for i in range(limit):
        window = " ".join(line.text for line in lines[i : i + 3])
        if _DELIVERED.match(window):
            j = next(k for k in range(i, i + 3) if "delivered" in lines[k].text.lower()) + 1
            while j < len(lines) and _ATTRIBUTION.match(lines[j].text):
                j += 1
            return j
        if _JUDGE_OPENING.match(lines[i].text):
            return i
        if _BODY_START.match(lines[i].text):
            return i + 1
    return 0


def segment(text: str, page_starts: list[int] | None = None) -> Segmentation:
    page_starts = page_starts or [0]
    lines = _lines(text)
    candidates = []
    previous = ""
    for i, line in enumerate(lines):
        c = candidate_of(line.text, i)
        if c:
            candidates.append(replace(c, after_colon=bool(_LEAD_IN.search(previous))))
        if line.text.strip():
            previous = line.text
    # Numbered headnote holdings ("HELD: 1. ... 2. ...") are the reporter's, not the court's.
    lo, hi = _headnote_span(lines)
    chains = find_chains([c for c in candidates if not lo <= c.line < hi], lines, len(text))

    front_end = chains[0][0].line if chains else _body_start(lines)

    paragraphs: list[Paragraph] = []
    counters = {"front": 0, "headnote": 0}
    for section, part_lines in _front_sections(lines[:front_end]):
        for block in _infer_blocks(part_lines):
            counters[section] += 1
            prefix = "h" if section == "headnote" else "f"
            no = f"{prefix}-{counters[section]}"
            paragraphs.append(_make(block, no, section, "inferred", text, page_starts))

    missing: list[int] = []
    if not chains:
        for n, block in enumerate(_infer_blocks(lines[front_end:]), start=1):
            paragraphs.append(_make(block, f"p-{n}", "body", "inferred", text, page_starts))
        return Segmentation(paragraphs, "inferred", 0, [], len(candidates))

    chosen = {c.line for members in chains for c in members}
    rivals: dict[tuple[Style, int], list[int]] = {}
    for c in candidates:
        if c.line not in chosen:
            rivals.setdefault((c.style, c.first), []).append(c.line)
    used = 0
    for part, members in enumerate(chains, start=1):
        next_chain_line = chains[part][0].line if part < len(chains) else len(lines)
        prev_chain_end = chains[part - 2][-1].line + 1 if part > 1 else 0
        for k, c in enumerate(members):
            stop = members[k + 1].line if k + 1 < len(members) else next_chain_line
            block = [line for line in lines[c.line : stop] if line.text.strip()]
            no = str(c.first) if c.last == c.first else f"{c.first}-{c.last}"
            lo = members[k - 2].line if k >= 2 else prev_chain_end
            hi = members[k + 2].line if k + 2 < len(members) else next_chain_line
            contested = any(lo <= o < hi for o in rivals.get((c.style, c.first), ()))
            paragraphs.append(
                _make(block, no, "body", "explicit", text, page_starts, part, contested)
            )
            if k:
                missing.extend(range(members[k - 1].last + 1, c.first))
        used += len(members)
    return Segmentation(paragraphs, "explicit", len(chains), missing, len(candidates) - used)
