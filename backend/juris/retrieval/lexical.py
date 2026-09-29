"""Lexical retrieval over chunks with Postgres full-text search (PLAN 4.1, D-028).

The index is ``chunks.tsv`` (migration 0005): the chunk text at weight A plus the case title,
or the Act, section and heading, at weight B, after ``juris_legal_text`` has rewritten
citation shorthand ("s. 74", "u/s 74", "Art. 14"). Queries go through the same function.

Section and article references in a query ("Section 16(2)", "s. 74", "Art. 14") become phrase
terms, so "16" and "2" must be adjacent to "section". Users may also quote phrases and
exclude terms with ``-word``, as in ``websearch_to_tsquery``.

Modes:

- ``any`` (default): chunks matching at least one term, ranked by IDF-weighted term coverage
  (the share of the query's information a chunk contains; IDF from ``lexeme_stats``) plus a
  small ``ts_rank`` term for frequency. Postgres ranking has no IDF, so in an OR query a
  common word repeated would beat a rare one.
- ``all``: every term must match (``websearch_to_tsquery`` semantics, including ``or``),
  ranked by ``ts_rank_cd``.
- ``phrase``: the whole query as one phrase (``phraseto_tsquery``), ranked by ``ts_rank_cd``.

Scores are in [0, 2) for ``any`` and [0, 1) for the others (rank / (rank + 1)).

Latency: scoring costs a few microseconds per candidate chunk, so candidates are bounded by
``max_candidates``. In ``any`` mode the candidates are the chunks containing one of the
query's rarest terms, taking terms in order of rarity while their chunk counts sum to at most
``max_candidates``. The chunks this leaves out contain only the query's commonest terms.
When even the rarest term is in more chunks than that (a query of words like "court"), or in
the other modes, an arbitrary ``max_candidates`` of the matching chunks are ranked: such a
query carries almost no lexical signal.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from juris.retrieval.filters import SearchFilters, filter_sql

Mode = Literal["any", "all", "phrase"]

# "Section 16(2)", "s. 74", "ss. 73", "Sec 10", "u/s 55", "Art. 14", "Article 21(1)(a)"
_REFERENCE = re.compile(
    r"(?<![\w/])(?:u/ss?\.?|sections?|secs?\.?|ss?\.|arts?\.|articles?)\s*"
    r"\d+[A-Za-z]{0,2}(?:\s*\(\s*[0-9A-Za-z]{1,5}\s*\))*",
    re.IGNORECASE,
)
_TOKEN = re.compile(r'(-?)"([^"]*)"?|(\S+)')
_WORDLIKE = re.compile(r"[^\W_]")  # a letter or digit
_EXCLUDE = re.compile(r"-[^\W_]")  # "-word" excludes; "--" or a lone "-" doesn't

_ANALYSE = text(
    """
    SELECT t.i, websearch_to_tsquery('english', juris_legal_text(t.term))::text,
           (SELECT min(coalesce(s.ndoc, 0))
              FROM unnest(tsvector_to_array(to_tsvector('english', juris_legal_text(t.term)))) x
              LEFT JOIN lexeme_stats s ON s.lexeme = x)
      FROM unnest(CAST(:terms AS text[])) WITH ORDINALITY AS t(term, i)
    """
)
_QUERIES = {
    "all": "websearch_to_tsquery('english', juris_legal_text(:q))",
    "phrase": "phraseto_tsquery('english', juris_legal_text(:q))",
}


@dataclass(frozen=True)
class ParsedQuery:
    text: str  # the query with references quoted, in websearch_to_tsquery syntax
    terms: list[str] = field(default_factory=list)  # words and "quoted phrases"
    excluded: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Term:
    tsquery: str
    df: int  # chunks containing it (a phrase: its rarest word's count, an upper bound)
    idf: float


@dataclass(frozen=True)
class LexicalHit:
    chunk_id: str
    doc_id: str
    score: float
    rank: int  # 1-based


def quote_references(query: str) -> str:
    """Wrap section/article references outside quotes in quotes, so they match as phrases."""
    parts = query.split('"')
    for i in range(0, len(parts), 2):  # even parts are outside quotes
        parts[i] = _REFERENCE.sub(lambda m: f'"{m.group(0)}"', parts[i])
    return '"'.join(parts)


def parse_query(query: str) -> ParsedQuery:
    quoted = quote_references(query)
    terms: list[str] = []
    excluded: list[str] = []
    for m in _TOKEN.finditer(quoted):
        if m.group(3) is not None:
            word = m.group(3)
            # "or" is an operator; a token without letters or digits ("----", "...") can't
            # match anything, and a run of dashes would read as nested negations
            if word.lower() == "or" or not _WORDLIKE.search(word):
                continue
            if _EXCLUDE.match(word):
                excluded.append(word[1:])
            else:
                terms.append(word)
        elif m.group(2).strip():
            (excluded if m.group(1) else terms).append(f'"{m.group(2).strip()}"')
    return ParsedQuery(quoted, terms, excluded)


def idf(df: int, n: int) -> float:
    """BM25's IDF: rare terms weigh more, and a term in every chunk still weighs a little."""
    return math.log(1 + (n - df + 0.5) / (df + 0.5))


def candidate_terms(terms: list[Term], budget: int) -> list[Term]:
    """The rarest terms whose chunk counts sum to at most ``budget``; [] if none fits."""
    picked: list[Term] = []
    total = 0
    for term in sorted(terms, key=lambda t: t.df):
        if total + term.df > budget:
            break
        picked.append(term)
        total += term.df
    return picked


def _or(queries: list[str]) -> str:
    return " | ".join(f"({q})" for q in queries)


def _and(queries: list[str]) -> str:
    return " & ".join(f"({q})" for q in queries)


class LexicalRetriever:
    """Ranked chunk IDs for a keyword query, with filters (``SearchFilters``)."""

    def __init__(
        self,
        engine: Engine,
        max_candidates: int = 10_000,
        density_weight: float = 0.1,
        max_terms: int | None = None,
    ) -> None:
        self.engine = engine
        self.max_candidates = max_candidates
        self.density_weight = density_weight  # weight of ts_rank against term coverage
        # a long query (a fact pattern, a whole judgment) keeps only its rarest terms in
        # ``any`` mode: hundreds of terms would each be checked against every candidate
        self.max_terms = max_terms
        self._n: int | None = None

    def chunk_count(self, conn: Connection) -> int:
        if self._n is None:
            self._n = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
        return self._n

    def analyse(self, conn: Connection, terms: list[str]) -> list[Term]:
        """Each term's tsquery and chunk frequency; terms that are only stop words drop out."""
        if not terms:
            return []
        n = self.chunk_count(conn)
        rows = conn.execute(_ANALYSE, {"terms": terms}).all()
        out: dict[str, Term] = {}
        for _, tsquery, df in sorted(rows):
            if tsquery and tsquery not in out:
                out[tsquery] = Term(tsquery, df or 0, idf(df or 0, n))
        return list(out.values())

    def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        k: int = 50,
        mode: Mode = "any",
    ) -> list[LexicalHit]:
        where, params = filter_sql(filters)
        params |= {"k": k, "cap": self.max_candidates}
        with self.engine.connect() as conn:
            if mode == "any":
                sql = self._any_sql(conn, parse_query(query), where, params)
                if sql is None:
                    return []
            elif mode in _QUERIES:
                params["q"] = parse_query(query).text if mode == "all" else query
                sql = f"""
                    SELECT chunk_id, doc_id, ts_rank_cd(tsv, q, 32) AS score FROM (
                        SELECT c.chunk_id, c.doc_id, c.tsv, q
                          FROM {_QUERIES[mode]} AS q, chunks c
                          JOIN documents d ON d.doc_id = c.doc_id
                         WHERE numnode(q) > 0 AND c.tsv @@ q AND {where}
                         LIMIT :cap
                    ) s
                    ORDER BY score DESC, chunk_id LIMIT :k
                """
            else:
                raise ValueError(f"unknown mode: {mode!r}")
            rows = conn.execute(text(sql), params).all()
        return [LexicalHit(r[0], r[1], float(r[2]), i) for i, r in enumerate(rows, 1)]

    def _any_sql(
        self, conn: Connection, parsed: ParsedQuery, where: str, params: dict[str, Any]
    ) -> str | None:
        """SQL for ``any`` mode (filling ``params``); None when no term can match."""
        terms = [t for t in self.analyse(conn, parsed.terms) if t.df > 0]
        if not terms:
            return None
        if self.max_terms is not None and len(terms) > self.max_terms:
            terms = sorted(terms, key=lambda t: (-t.idf, t.tsquery))[: self.max_terms]
        total = sum(t.idf for t in terms)
        rare = candidate_terms(terms, self.max_candidates)
        # candidates: chunks with a rarer term; if every term is common, chunks with all of them
        params["cand"] = (
            _or([t.tsquery for t in rare]) if rare else _and([t.tsquery for t in terms])
        )
        params["all"] = _or([t.tsquery for t in terms])
        params["lam"] = self.density_weight
        coverage = []
        for i, t in enumerate(terms):
            params[f"t{i}"] = t.tsquery
            coverage.append(
                f"CASE WHEN tsv @@ CAST(:t{i} AS tsquery) THEN {t.idf / total!r}::float8 ELSE 0 END"
            )
        exclude = ""
        excluded = [t.tsquery for t in self.analyse(conn, parsed.excluded)]
        if excluded:
            exclude = "AND NOT c.tsv @@ CAST(:excl AS tsquery)"
            params["excl"] = _or(excluded)
        return f"""
            SELECT chunk_id, doc_id,
                   {" + ".join(coverage)} + :lam * ts_rank(tsv, CAST(:all AS tsquery), 32) AS score
              FROM (
                SELECT c.chunk_id, c.doc_id, c.tsv
                  FROM chunks c JOIN documents d ON d.doc_id = c.doc_id
                 WHERE c.tsv @@ CAST(:cand AS tsquery) {exclude} AND {where}
                 LIMIT :cap
              ) s
             ORDER BY score DESC, chunk_id LIMIT :k
        """
