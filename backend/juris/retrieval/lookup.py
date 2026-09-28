"""Structured lookups: exact fetches when a provision or a case is named (PLAN 4.5, D-032).

- Provisions: ``parse_provision`` reads "s. 15 ICA", "Section 15 of the Indian Contract Act",
  "Contract Act s.15", "u/s 16(2) of the Contract Act, 1872" ... as (Act ID, section,
  sub-section), with the Act resolved through the alias table (``configs/statutes/acts.yaml``).
  ``get_section`` returns the section's text, validity and chunks.
- Cases by citation: the citation is normalised (``juris.ingest.citations``), then matched
  against the corpus documents' own citations, the parallel-citation aliases (an SCC or AIR
  citation to its SCR twin, D-013) and, for judgments outside the corpus, the citation graph.
- Cases by title: trigram word similarity (``pg_trgm``) over document titles, which tolerates
  OCR damage ("SATYABRATA GROSE") and partial names ("Kailash Nath Associates").
- Citing and cited cases, and the judgments citing a statute section, from the citation graph
  (3.9), one entry per document with the paragraphs and treatment cues (a weak signal, D-013).
"""

import datetime as dt
import re
from dataclasses import dataclass, field
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from juris.ingest.citations import parse_citation
from juris.ingest.statutes import Act, load_acts, resolve_act

# "s. 15", "S.15", "ss. 73", "Sec 10", "Section 16(2)(a)", "u/s 55", "section 19A"
_SECTION = re.compile(
    r"(?<![\w/])(?:u/ss?\.?|sections?|secs?\.?|ss?\.?)\s*"
    r"(?P<num>\d{1,3}[A-Za-z]{0,2})(?P<sub>(?:\s*\(\s*[0-9A-Za-z]{1,5}\s*\))*)",
    re.IGNORECASE,
)
_VERSUS = re.compile(r"\s+(?:v|vs)\.?\s+", re.IGNORECASE)
_FILLER = re.compile(r"^(?:\s|,|;|:|-|of\b|the\b|under\b|in\b)+|(?:\s|,|;|:|-|of|the)+$", re.I)


@dataclass(frozen=True)
class Provision:
    act_id: str
    section: str  # "15", "19A"
    sub_section: str | None = None  # "(2)", "(2)(a)"

    @property
    def section_id(self) -> str:
        return f"{self.act_id}:{self.section}"


@dataclass(frozen=True)
class Section:
    section_id: str
    act_id: str
    act: str
    act_year: int
    section: str
    title: str
    text: str
    repealed: bool
    in_force_from: dt.date | None
    in_force_to: dt.date | None
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseMatch:
    citation: str  # the normalised citation that was looked up
    target_ref: str  # the judgment's document ID ("SC-2003_3_691_741"), in the corpus or not
    doc_id: str | None  # set when the judgment is in the corpus
    title: str | None
    how: str  # document | alias | citation_graph


@dataclass(frozen=True)
class TitleMatch:
    doc_id: str
    title: str
    decision_date: dt.date | None
    score: float  # trigram word similarity, 0-1


@dataclass(frozen=True)
class LinkedCase:
    """A judgment on the other end of citation edges, with where and how it cites."""

    ref: str  # the judgment's document ID ("SC-1978_2_272_337"), in the corpus or not
    doc_id: str | None  # set when the judgment is in the corpus
    title: str | None
    decision_date: dt.date | None
    mentions: int
    paragraphs: list[str]  # the citing judgment's paragraph numbers
    citations: list[str]  # the citation strings as normalised
    treatments: list[str]  # weak cue labels ("followed", "distinguished" ...), D-013


@lru_cache
def _acts() -> dict[str, Act]:
    return load_acts()


def parse_provision(reference: str) -> Provision | None:
    """(Act, section, sub-section) from a free-text reference; None if either part is missing.

    The Act may come before or after the section ("Contract Act s.15", "s. 15 ICA").
    """
    m = _SECTION.search(reference)
    if not m:
        return None
    rest = f"{reference[: m.start()]} {reference[m.end() :]}"
    act_name = _FILLER.sub("", re.sub(r"\s+", " ", rest).strip())
    act_id = resolve_act(act_name, _acts()) if act_name else None
    if act_id is None:
        return None
    sub = re.sub(r"\s+", "", m.group("sub")) or None
    return Provision(act_id, m.group("num").upper(), sub)


class Lookup:
    """Exact lookups over the corpus database."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # ---- provisions ----------------------------------------------------------------------

    def get_section(self, act: str, section: str) -> Section | None:
        """A section by Act (ID or alias) and number ("15", "19a" -> "19A")."""
        act_id = resolve_act(act, _acts())
        if act_id is None:
            return None
        with self.engine.connect() as conn:
            return self._section(conn, f"{act_id}:{section.strip().upper()}")

    def lookup_provision(self, reference: str) -> Section | None:
        provision = parse_provision(reference)
        if provision is None:
            return None
        with self.engine.connect() as conn:
            return self._section(conn, provision.section_id)

    def _section(self, conn: Connection, section_id: str) -> Section | None:
        row = conn.execute(
            text(
                "SELECT section_id, act_id, act, act_year, section, title, text, repealed, "
                "in_force_from, in_force_to FROM statute_sections WHERE section_id = :s"
            ),
            {"s": section_id},
        ).one_or_none()
        if row is None:
            return None
        chunks = conn.execute(
            text(
                "SELECT chunk_id FROM chunks WHERE statute_section_id = :s "
                "ORDER BY char_start, chunk_id"
            ),
            {"s": section_id},
        )
        return Section(
            section_id=row[0], act_id=row[1], act=row[2], act_year=row[3], section=row[4],
            title=row[5], text=row[6], repealed=row[7], in_force_from=row[8],
            in_force_to=row[9], chunk_ids=[r[0] for r in chunks],
        )  # fmt: skip

    # ---- cases ---------------------------------------------------------------------------

    def get_case_by_citation(self, citation: str) -> CaseMatch | None:
        """The judgment a reporter or neutral citation names; None if unknown or unparseable."""
        parsed = parse_citation(citation)
        if parsed is None:
            return None
        canonical = parsed.canonical
        with self.engine.connect() as conn:
            doc = conn.execute(
                text(
                    "SELECT doc_id FROM documents WHERE citations @> ARRAY[CAST(:c AS text)] "
                    "OR neutral_citation = :c ORDER BY doc_id LIMIT 1"
                ),
                {"c": canonical},
            ).scalar()
            target, how = (doc, "document") if doc else (None, "")
            if target is None:
                target = conn.execute(
                    text("SELECT target_ref FROM citation_aliases WHERE alias = :c"),
                    {"c": canonical},
                ).scalar()
                how = "alias"
            if target is None:  # a judgment outside the corpus, known from citing judgments
                target = conn.execute(
                    text(
                        "SELECT target_ref FROM citation_edges WHERE canonical = :c "
                        "AND target_ref IS NOT NULL AND resolution IS DISTINCT FROM 'conflict' "
                        "GROUP BY target_ref ORDER BY count(*) DESC, target_ref LIMIT 1"
                    ),
                    {"c": canonical},
                ).scalar()
                how = "citation_graph"
            if target is None:
                return None
            title = conn.execute(
                text("SELECT title FROM documents WHERE doc_id = :d"), {"d": target}
            ).scalar()
        return CaseMatch(canonical, target, target if title else None, title, how)

    def get_case_by_title(
        self, title: str, limit: int = 5, min_score: float = 0.4
    ) -> list[TitleMatch]:
        """Judgments whose title best contains the words of ``title`` (fuzzy, ranked)."""
        # the stored titles say "versus"; queries usually say "v." or "vs"
        query = re.sub(r"\s+", " ", _VERSUS.sub(" versus ", title)).strip()
        if not query:
            return []
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT doc_id, title, decision_date, "
                    "word_similarity(lower(:q), lower(title)) AS s FROM documents "
                    "WHERE kind = 'judgment' AND word_similarity(lower(:q), lower(title)) >= :m "
                    "ORDER BY s DESC, decision_date DESC NULLS LAST, doc_id LIMIT :n"
                ),
                {"q": query, "m": min_score, "n": limit},
            ).all()
        return [TitleMatch(r[0], r[1], r[2], float(r[3])) for r in rows]

    # ---- the citation graph --------------------------------------------------------------

    def get_citing_cases(self, doc_id: str) -> list[LinkedCase]:
        """Corpus judgments that cite ``doc_id`` (by resolved edge), most mentions first."""
        return self._linked(
            "e.source_doc_id",
            "e.kind = 'case' AND (e.target_doc_id = :d OR e.target_ref = :d)",
            doc_id,
        )

    def get_cited_cases(self, doc_id: str) -> list[LinkedCase]:
        """Judgments that ``doc_id`` cites and that resolved (in the corpus or not)."""
        return self._linked(
            "coalesce(e.target_doc_id, e.target_ref)",
            "e.kind = 'case' AND e.source_doc_id = :d AND e.target_ref IS NOT NULL",
            doc_id,
        )

    def get_cases_citing_section(self, act: str, section: str) -> list[LinkedCase]:
        """Corpus judgments that cite a statute section (resolved statute mentions)."""
        act_id = resolve_act(act, _acts())
        if act_id is None:
            return []
        return self._linked(
            "e.source_doc_id",
            "e.kind = 'statute' AND e.target_section_id = :d",
            f"{act_id}:{section.strip().upper()}",
        )

    def _linked(self, key: str, where: str, value: str) -> list[LinkedCase]:
        sql = f"""
            SELECT {key} AS k, d.doc_id, d.title, d.decision_date,
                   count(*) AS n,
                   array_agg(DISTINCT e.source_para) FILTER (WHERE e.source_para IS NOT NULL),
                   array_agg(DISTINCT e.canonical) FILTER (WHERE e.canonical IS NOT NULL),
                   array_agg(DISTINCT e.treatment) FILTER (WHERE e.treatment IS NOT NULL)
              FROM citation_edges e
              LEFT JOIN documents d ON d.doc_id = {key}
             WHERE {where}
             GROUP BY k, d.doc_id, d.title, d.decision_date
             ORDER BY n DESC, d.decision_date DESC NULLS LAST, k
        """
        with self.engine.connect() as conn:
            rows = conn.execute(text(sql), {"d": value}).all()
        return [
            LinkedCase(
                ref=r[0],
                doc_id=r[1],
                title=r[2],
                decision_date=r[3],
                mentions=r[4],
                paragraphs=sorted(r[5] or [], key=_para_key),
                citations=sorted(r[6] or []),
                treatments=sorted(r[7] or []),
            )
            for r in rows
        ]


def _para_key(no: str) -> tuple[int, str]:
    m = re.match(r"\d+", no)
    return (int(m.group(0)) if m else 10**9, no)
