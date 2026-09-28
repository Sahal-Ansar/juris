"""Case and statute citation edges from judgment text (PLAN 3.9, D-013).

Case citations (``juris.ingest.citations``) resolve against the AWS index of every Supreme
Court judgment (not only the corpus):

- INSC neutral citations match ``case_id`` exactly;
- SCR citations match the report's first page, or else fall inside a report's page range
  (pinpoint citations such as "[1963] 3 SCR 45"); Supplementary volumes are keyed apart;
- SCC, AIR and other reporters resolve through a parallel-citation alias table, built from
  citations printed together ("(2015) 4 SCC 136 : [2015] 3 S.C.R. 1") where one member
  resolves. An alias that points at two different judgments is dropped as ambiguous.

A resolved case gets ``target_ref`` (the document ID it has or would have, "SC-2015_3_243_286")
and ``target_doc_id`` when that judgment is in the corpus. Citations of the judgment itself
(the SCR reporter line, pinpoints into its own pages) are dropped.

Statute mentions ("Section 16(c) of the Specific Relief Act", "ss. 73 and 74 of the Contract
Act") resolve to ``statute_sections`` through the Act alias table (3.5).

A treatment cue ("overruled", "per incuriam", "distinguished", "followed", "relied on", ...)
in the citing sentence is stored with its text as a weak signal, never as verified treatment.
"""

import bisect
import datetime as dt
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from juris.ingest.citations import ReporterCitation, find_citation_spans
from juris.ingest.statutes import Act, resolve_act

# ---- Supreme Court index -----------------------------------------------------------------

_PATH = re.compile(r"^(S_)?(\d{4})_(\d+)_(\d+)_(\d+)$")


@dataclass
class ScIndex:
    """Every SC judgment's report position and neutral citation, from AWS metadata rows."""

    by_insc: dict[str, str] = field(default_factory=dict)
    by_scr: dict[tuple[int, bool, int], list[tuple[int, int, str]]] = field(default_factory=dict)

    @classmethod
    def from_rows(cls, rows: Iterable[dict[str, Any]]) -> "ScIndex":
        index = cls()
        seen: set[str] = set()
        for row in rows:
            path = row.get("path") or ""
            m = _PATH.match(path)
            if not m or path in seen:
                continue
            seen.add(path)
            doc = f"SC-{path}"
            key = (int(m[2]), bool(m[1]), int(m[3]))
            index.by_scr.setdefault(key, []).append((int(m[4]), int(m[5]), doc))
            if insc := re.sub(r"\s+", " ", row.get("case_id") or "").strip():
                index.by_insc[insc] = doc
        for spans in index.by_scr.values():
            spans.sort()
        return index

    def resolve(self, c: ReporterCitation) -> tuple[str, str] | None:
        """(document ID, how) for an INSC or SCR citation; None otherwise."""
        if c.reporter == "INSC":
            doc = self.by_insc.get(c.canonical)
            return (doc, "direct") if doc else None
        if c.reporter != "SCR" or c.volume is None:
            return None
        # First page only. A page inside another report's range was, in a hand-checked sample,
        # mostly a misprint ("(1972) 2 SCR 331" for 33), not a pinpoint, so it stays unresolved.
        for start, _end, doc in self.by_scr.get((c.year, c.supp, c.volume), []):
            if start == c.page:
                return doc, "direct"
        return None


# ---- mentions ----------------------------------------------------------------------------

_PARALLEL_GAP = re.compile(r"^\s*[:=]\s*$")


@dataclass
class Mention:
    start: int
    end: int
    citation: ReporterCitation


def parallel_groups(text: str) -> tuple[list[Mention], list[list[int]]]:
    """Citations in ``text`` and groups of indices printed together ("A : B = C")."""
    mentions = [Mention(s, e, c) for s, e, c in find_citation_spans(text)]
    groups: list[list[int]] = []
    for i in range(len(mentions)):
        gap = text[mentions[i - 1].end : mentions[i].start] if i else ""
        if i and _PARALLEL_GAP.match(gap) and len(gap) <= 6:
            groups[-1].append(i)
        else:
            groups.append([i])
    return mentions, groups


@dataclass
class AliasTable:
    """Canonical citation -> document ID, learnt from parallel citations."""

    targets: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    evidence: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def learn(self, citations: Sequence[ReporterCitation], index: ScIndex) -> None:
        resolved = {r[0] for c in citations if (r := index.resolve(c))}
        if len(resolved) != 1:  # none, or the group spans several judgments
            return
        (doc,) = resolved
        for c in citations:
            if c.reporter not in ("SCR", "INSC"):
                self.targets[c.canonical].add(doc)
                self.evidence[c.canonical] += 1

    def resolve(self, c: ReporterCitation) -> str | None:
        docs = self.targets.get(c.canonical)
        return next(iter(docs)) if docs and len(docs) == 1 else None

    def rows(self) -> list[dict[str, Any]]:
        return [
            {"alias": alias, "target_ref": next(iter(docs)), "evidence": self.evidence[alias]}
            for alias, docs in sorted(self.targets.items())
            if len(docs) == 1
        ]

    @property
    def ambiguous(self) -> int:
        return sum(1 for docs in self.targets.values() if len(docs) > 1)


# ---- treatment cues ----------------------------------------------------------------------

_CUES: list[tuple[str, str]] = [
    (r"per\s+incuriam", "per_incuriam"),
    (r"not\s+(?:a\s+)?good\s+law|no\s+longer\s+good\s+law", "overruled"),
    (r"overrul\w*", "overruled"),
    (r"doubted", "doubted"),
    (r"reversed", "reversed"),
    (r"distinguish\w*", "distinguished"),
    (r"affirmed", "affirmed"),
    (r"approved", "approved"),
    (r"followed|following\s+the", "followed"),
    (r"relied\s+(?:up)?on|placed\s+reliance|reliance\s+(?:was\s+|is\s+)?placed", "relied_on"),
    (r"referred\s+to", "referred"),
]
_CUE = re.compile("|".join(f"(?P<c{i}>{p})" for i, (p, _) in enumerate(_CUES)), re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.;]\s+(?=[A-Z(\[\"])|\n(?=\s*(?:\d+\.|\(\w+\)|[A-Z]))")


def treatment_cue(text: str, start: int, end: int, window: int = 250) -> tuple[str, str] | None:
    """(label, cue text) of the cue closest to the citation within its sentence."""
    lo = max(0, start - window)
    before = [m.end() for m in _SENTENCE_END.finditer(text, lo, start)]
    lo = before[-1] if before else lo
    after = _SENTENCE_END.search(text, end, min(len(text), end + window))
    hi = after.start() + 1 if after else min(len(text), end + window)
    best: tuple[int, str, str] | None = None
    for m in _CUE.finditer(text, lo, hi):
        label = next(_CUES[int(k[1:])][1] for k, v in m.groupdict().items() if v)
        dist = start - m.end() if m.end() <= start else m.start() - end
        if best is None or dist < best[0]:
            best = (dist, label, m.group(0))
    return (best[1], re.sub(r"\s+", " ", best[2])) if best else None


# ---- statute mentions --------------------------------------------------------------------

_SECTION_REF = re.compile(
    r"\b(?:[Ss]ections?|[Ss]ecs?\.|[Ss]s?\.|u/s\.?)\s*"
    r"(?P<nums>\d{1,3}[A-Z]{0,2}(?:\s*\([0-9A-Za-z]{1,5}\))*"
    r"(?:\s*(?:,|and|&|to|or|/)\s*\d{1,3}[A-Z]{0,2}(?:\s*\([0-9A-Za-z]{1,5}\))*){0,8})"
    r"\s+of\s+(?:the\s+)?"
    # "Sale of Goods Act", "Transfer of Property Act": capitalised words and small joiners
    r"(?P<act>[A-Z][\w.'&-]*\s+(?:(?:[A-Z][\w.'&-]*|of|and|for|the|&)\s+){0,7}?"
    r"(?:Act|Code)(?:,?\s*(?:1[89]|20)\d\d)?)"
)


@dataclass
class StatuteMention:
    start: int  # offset of this section number
    end: int
    section: str
    act_name: str
    act_id: str | None
    raw: str


def statute_mentions(text: str, acts: dict[str, Act]) -> list[StatuteMention]:
    out = []
    for m in _SECTION_REF.finditer(text):
        act_name = re.sub(r"\s+", " ", m.group("act")).strip()
        act_id = resolve_act(act_name, acts)
        nums_start = m.start("nums")
        nums = re.sub(r"\([^)]*\)", lambda p: " " * len(p.group(0)), m.group("nums"))
        for n in re.finditer(r"\d{1,3}[A-Z]{0,2}", nums):
            out.append(
                StatuteMention(
                    start=nums_start + n.start(),
                    end=nums_start + n.end(),
                    section=n.group(0),
                    act_name=act_name,
                    act_id=act_id,
                    raw=re.sub(r"\s+", " ", m.group(0)),
                )
            )
    return out


# ---- edges -------------------------------------------------------------------------------


@dataclass
class Edge:
    source_doc_id: str
    kind: str  # case | statute
    raw: str
    canonical: str | None
    reporter: str | None
    char_start: int
    char_end: int
    target_ref: str | None = None
    target_doc_id: str | None = None
    target_section_id: str | None = None
    resolution: str | None = None  # direct | range | alias | act
    source_para: str | None = None
    source_para_seq: int | None = None
    treatment: str | None = None
    treatment_cue: str | None = None
    context: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _context(text: str, start: int, end: int, pad: int = 160) -> str:
    return re.sub(r"\s+", " ", text[max(0, start - pad) : min(len(text), end + pad)]).strip()


def document_edges(
    doc_id: str,
    text: str,
    paragraphs: Sequence[dict[str, Any]],
    index: ScIndex,
    aliases: AliasTable,
    corpus: set[str],
    sections: set[str],
    acts: dict[str, Act],
    decided: dt.date | None = None,
) -> list[Edge]:
    """Every case and statute citation in one judgment, resolved where possible."""
    starts = [p["char_start"] for p in paragraphs]

    def para(offset: int) -> tuple[str | None, int | None]:
        i = bisect.bisect_right(starts, offset) - 1
        return (paragraphs[i]["no"], i) if i >= 0 else (None, None)

    edges: list[Edge] = []
    mentions, groups = parallel_groups(text)
    resolved: list[tuple[str | None, str | None]] = []
    for mention in mentions:
        hit = index.resolve(mention.citation)
        resolved.append(hit if hit else (aliases.resolve(mention.citation), "alias"))
    # Parallel citations name one judgment; if they resolve to different ones, one of them is
    # misprinted and none is trusted ("[1952] 1 SCR 683 : 1952 INSC 63" naming two cases).
    for group in groups:
        if len({resolved[i][0] for i in group if resolved[i][0]}) > 1:
            for i in group:
                resolved[i] = (None, "conflict")
    self_cites = {
        i for group in groups if any(resolved[i][0] == doc_id for i in group) for i in group
    }
    for i, mention in enumerate(mentions):
        c = mention.citation
        target, how = resolved[i]
        if i in self_cites:
            continue  # the judgment's own citation (reporter line) and its parallels
        cue = treatment_cue(text, mention.start, mention.end)
        no, seq = para(mention.start)
        edges.append(
            Edge(
                source_doc_id=doc_id,
                kind="case",
                raw=re.sub(r"\s+", " ", text[mention.start : mention.end]),
                canonical=c.canonical,
                reporter=c.reporter,
                char_start=mention.start,
                char_end=mention.end,
                target_ref=target,
                target_doc_id=target if target in corpus else None,
                resolution=how if target or how == "conflict" else None,
                source_para=no,
                source_para_seq=seq,
                treatment=cue[0] if cue else None,
                treatment_cue=cue[1] if cue else None,
                context=_context(text, mention.start, mention.end),
            )
        )
    for s in statute_mentions(text, acts):
        act = acts.get(s.act_id) if s.act_id else None
        # "the Specific Relief Act" (no year) in a 1955 judgment is the 1877 Act, not ours
        undated = not re.search(r"\d{4}", s.act_name)
        if act and act.commencement and decided and decided < act.commencement and undated:
            s.act_id = None
        section_id = f"{s.act_id}:{s.section}" if s.act_id else None
        no, seq = para(s.start)
        edges.append(
            Edge(
                source_doc_id=doc_id,
                kind="statute",
                raw=s.raw,
                canonical=section_id or f"{s.act_name}:{s.section}",
                reporter=None,
                char_start=s.start,
                char_end=s.end,
                target_ref=f"ACT-{s.act_id}" if s.act_id else None,
                target_doc_id=f"ACT-{s.act_id}"
                if s.act_id and f"ACT-{s.act_id}" in corpus
                else None,
                target_section_id=section_id if section_id in sections else None,
                resolution="act" if s.act_id else None,
                source_para=no,
                source_para_seq=seq,
                context=_context(text, s.start, s.end),
            )
        )
    return edges
