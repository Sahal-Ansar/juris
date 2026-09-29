"""Lexical retrieval (PLAN 4.1): query parsing, filters, and ranking on a tiny corpus.

The ``db`` tests build a handful of chunks in a fresh database (``engine`` fixture), so each
ranking rule is checked on text written to exercise it.
"""

import datetime as dt
from collections.abc import Iterator
from itertools import pairwise
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import insert, text
from sqlalchemy.engine import Engine

from juris.db import models as m
from juris.db.load import refresh_lexeme_stats
from juris.models import CourtLevel, DocumentKind
from juris.retrieval.filters import SearchFilters, act_ids, filter_sql
from juris.retrieval.lexical import (
    LexicalRetriever,
    Term,
    candidate_terms,
    idf,
    parse_query,
    quote_references,
)
from tests.conftest import migrate

# ---- no database needed ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Section 16(2) undue influence", '"Section 16(2)" undue influence'),
        ("s. 74 and u/s 73", '"s. 74" and "u/s 73"'),
        ("ss. 73 of the Act", '"ss. 73" of the Act'),
        ("Sec 10 and Sec.19A", '"Sec 10" and "Sec.19A"'),
        ("Art. 14 and Article 21(1)(a)", '"Art. 14" and "Article 21(1)(a)"'),
        ('"Section 74 of the Contract Act" penalty', '"Section 74 of the Contract Act" penalty'),
        ("section of the Act", "section of the Act"),  # no number: not a reference
        ("the parties' 5 claims", "the parties' 5 claims"),  # "s" needs its full stop
    ],
)
def test_references_become_phrases(query: str, expected: str) -> None:
    assert quote_references(query) == expected


def test_parse_query_splits_terms_phrases_and_exclusions() -> None:
    parsed = parse_query('Section 74 penalty or forfeiture -"earnest money" -interest')
    assert parsed.terms == ['"Section 74"', "penalty", "forfeiture"]
    assert parsed.excluded == ['"earnest money"', "interest"]
    assert parsed.text.startswith('"Section 74" penalty or forfeiture')


def test_punctuation_tokens_are_not_terms_or_exclusions() -> None:
    # judgments quoted as queries carry rulers of dashes; "--" read as negations broke Postgres
    parsed = parse_query("penalty ------------- clause -- ... - x-ray -forfeiture")
    assert parsed.terms == ["penalty", "clause", "x-ray"]
    assert parsed.excluded == ["forfeiture"]


def test_idf_prefers_rare_terms() -> None:
    assert idf(1, 1000) > idf(100, 1000) > idf(1000, 1000) > 0


def test_candidate_terms_take_the_rarest_within_budget() -> None:
    terms = [Term("'a'", 500, 1.0), Term("'b'", 10, 3.0), Term("'c'", 80, 2.0)]
    assert [t.tsquery for t in candidate_terms(terms, 100)] == ["'b'", "'c'"]
    assert [t.tsquery for t in candidate_terms(terms, 1000)] == ["'b'", "'c'", "'a'"]
    assert candidate_terms(terms, 5) == []


def test_filters_validate_and_resolve_act_aliases() -> None:
    with pytest.raises(ValidationError):
        SearchFilters(date_from=dt.date(2020, 1, 1), date_to=dt.date(2019, 1, 1))
    with pytest.raises(ValidationError):
        SearchFilters(court="supreme_court")  # type: ignore[call-arg]
    assert act_ids(("ICA", "Specific Relief Act, 1963", "sale_of_goods_act")) == [
        "contract_act",
        "specific_relief_act",
        "sale_of_goods_act",
    ]
    with pytest.raises(ValueError, match="unknown Act"):
        act_ids(("Specific Relief Act, 1877",))
    assert filter_sql(None) == ("TRUE", {})
    sql, params = filter_sql(SearchFilters(court_levels=(CourtLevel.SUPREME_COURT,)))
    assert "court_level" in sql and params == {"f_courts": ["supreme_court"]}


# ---- database ----------------------------------------------------------------------------

SC_OLD = "SC-1990_1_1_10"  # SC, 1990, cites ICA s. 74
SC_NEW = "SC-2015_2_1_20"  # SC, 2015
HC = "HC-DLHC0001_1_2019-01-01"  # Delhi HC, 2019
ACT = "ACT-contract_act"

CHUNKS: dict[str, tuple[str, str, str | None]] = {
    # chunk_id: (doc_id, text, context_header)
    f"{SC_OLD}#c0001": (
        SC_OLD,
        "The penalty clause is hit by s. 74 of the Act and only reasonable compensation is due.",
        "RAMAN v. KRISHNA — Supreme Court of India — 1990 — ¶ 4",
    ),
    f"{SC_OLD}#c0002": (
        SC_OLD,
        "Undue influence under Section 16(2) arises where one party dominates the will.",
        "RAMAN v. KRISHNA — Supreme Court of India — 1990 — ¶ 5",
    ),
    f"{SC_NEW}#c0001": (
        SC_NEW,
        "Section 2 of the Act is a definition clause; clause 16 sets the price. Contract, "
        "contract, contract: the contract was a contract of sale.",
        "MEHTA TRADERS v. UNION OF INDIA — Supreme Court of India — 2015 — ¶ 16",
    ),
    f"{SC_NEW}#c0002": (
        SC_NEW,
        "The seller pleaded frustration; a contract becomes void when performance is "
        "impossible. Force majeure was not pleaded.",
        "MEHTA TRADERS v. UNION OF INDIA — Supreme Court of India — 2015 — ¶ 17",
    ),
    f"{HC}#c0001": (
        HC,
        "The earnest money was forfeited under clause 9 of the contract as liquidated damages.",
        "RFA/1/2019 of ZORAWAR Vs PEARL ESTATES — High Court of Delhi — 2019 — ¶ 3",
    ),
    f"{ACT}#s74": (
        ACT,
        "74. When a contract has been broken, if a sum is named in the contract as the amount "
        "to be paid in case of such breach, the party complaining of the breach is entitled "
        "to receive reasonable compensation not exceeding the amount so named or the penalty.",
        "Indian Contract Act, 1872 — s. 74 — Compensation for breach of contract where "
        "penalty stipulated for",
    ),
}


@pytest.fixture
def corpus(engine: Engine) -> Iterator[Engine]:
    docs: list[dict[str, Any]] = [
        {"doc_id": SC_OLD, "kind": "judgment", "court_level": "supreme_court",
         "decision_date": dt.date(1990, 3, 1), "title": "Raman v. Krishna"},
        {"doc_id": SC_NEW, "kind": "judgment", "court_level": "supreme_court",
         "decision_date": dt.date(2015, 6, 1), "title": "Mehta Traders v. Union of India"},
        {"doc_id": HC, "kind": "judgment", "court_level": "high_court",
         "decision_date": dt.date(2019, 1, 1), "title": "Zorawar v. Pearl Estates"},
        {"doc_id": ACT, "kind": "statute", "court_level": None, "decision_date": None,
         "title": "Indian Contract Act, 1872"},
    ]  # fmt: skip
    with engine.begin() as conn:
        migrate(conn)
        conn.execute(insert(m.Snapshot).values(snapshot_id="s", slice_name="t"))
        conn.execute(
            insert(m.Document),
            [{**d, "snapshot_id": "s", "source": "test", "licence": "test"} for d in docs],
        )
        conn.execute(
            insert(m.StatuteSection).values(
                section_id="contract_act:74", doc_id=ACT, act_id="contract_act",
                act="Indian Contract Act", act_year=1872, section="74",
                title="Compensation for breach", text=CHUNKS[f"{ACT}#s74"][1],
                char_start=0, char_end=10, page_start=1, page_end=1,
            )
        )  # fmt: skip
        conn.execute(
            insert(m.Chunk),
            [
                {"chunk_id": cid, "doc_id": doc, "text": body, "context_header": header,
                 "char_start": 0, "char_end": len(body),
                 "statute_section_id": "contract_act:74" if doc == ACT else None}
                for cid, (doc, body, header) in CHUNKS.items()
            ],
        )  # fmt: skip
        conn.execute(
            insert(m.CitationEdge).values(
                source_doc_id=SC_OLD, raw="s. 74 of the Act", char_start=0, char_end=10,
                kind="statute", target_section_id="contract_act:74",
            )
        )  # fmt: skip
        refresh_lexeme_stats(conn)
    yield engine


def ids(hits: list[Any]) -> list[str]:
    return [h.chunk_id for h in hits]


@pytest.mark.db
def test_index_normalises_shorthand_and_stores_inline(corpus: Engine) -> None:
    with corpus.connect() as conn:
        assert (
            conn.execute(text("SELECT juris_legal_text('u/s 55, s. 74, ss. 73-74 and Art. 14')"))
            .scalar_one()
            == "under section 55, section 74, section 73 74 and article 14"
        )  # fmt: skip
        storage: str = conn.execute(
            text("SELECT attstorage FROM pg_attribute "
                 "WHERE attrelid = 'chunks'::regclass AND attname = 'tsv'")
        ).scalar_one()  # fmt: skip
        assert storage == "m"
        assert conn.execute(text("SELECT count(*) FROM lexeme_stats")).scalar_one() > 50


@pytest.mark.db
def test_section_references_match_as_phrases(corpus: Engine) -> None:
    r = LexicalRetriever(corpus)
    # "s. 74" in the text and "s. 74" in the Act's header both match "Section 74"
    assert set(ids(r.search("Section 74", mode="all"))) == {f"{SC_OLD}#c0001", f"{ACT}#s74"}
    # "Section 2 ... clause 16" has the numbers, but not as "section 16(2)"
    assert ids(r.search("Section 16(2)", mode="all")) == [f"{SC_OLD}#c0002"]
    assert ids(r.search("s. 16(2)")) == [f"{SC_OLD}#c0002"]


@pytest.mark.db
def test_case_titles_in_the_header_are_searchable(corpus: Engine) -> None:
    hits = LexicalRetriever(corpus).search("Mehta Traders")
    assert set(ids(hits)) == {f"{SC_NEW}#c0001", f"{SC_NEW}#c0002"}
    # the paragraph label is not indexed: "16" only matches the text of c0001
    hits = LexicalRetriever(corpus).search("16", mode="all")
    assert set(ids(hits)) == {f"{SC_OLD}#c0002", f"{SC_NEW}#c0001"}


@pytest.mark.db
def test_any_mode_weights_rare_terms_over_repeated_common_ones(corpus: Engine) -> None:
    hits = LexicalRetriever(corpus).search("contract frustration")
    # "contract" x6 loses to one "frustration", which is in one chunk only
    assert hits[0].chunk_id == f"{SC_NEW}#c0002"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(a.score >= b.score for a, b in pairwise(hits))


@pytest.mark.db
def test_all_mode_needs_every_term_and_any_mode_ranks_full_matches_first(corpus: Engine) -> None:
    r = LexicalRetriever(corpus)
    assert ids(r.search("earnest money forfeited penalty", mode="all")) == []
    any_hits = r.search("earnest money forfeited penalty")
    assert any_hits[0].chunk_id == f"{HC}#c0001"
    assert {f"{SC_OLD}#c0001", f"{ACT}#s74"} <= set(ids(any_hits))  # "penalty" only
    assert ids(r.search("penalty compensation", mode="all"))[0] in {
        f"{SC_OLD}#c0001",
        f"{ACT}#s74",
    }


@pytest.mark.db
def test_phrase_mode_and_exclusions(corpus: Engine) -> None:
    r = LexicalRetriever(corpus)
    hits = r.search("reasonable compensation", mode="phrase")
    assert set(ids(hits)) == {f"{SC_OLD}#c0001", f"{ACT}#s74"}
    assert ids(r.search("compensation reasonable", mode="phrase")) == []
    assert ids(r.search("penalty -breach")) == [f"{SC_OLD}#c0001"]
    assert ids(r.search('penalty -"reasonable compensation"')) == []
    assert ids(r.search("penalty -breach", mode="all")) == [f"{SC_OLD}#c0001"]


@pytest.mark.db
def test_stop_words_and_unknown_words_return_nothing(corpus: Engine) -> None:
    r = LexicalRetriever(corpus)
    for mode in ("any", "all", "phrase"):
        assert r.search("the of and", mode=mode) == []  # type: ignore[arg-type]
        assert r.search("xylophone", mode=mode) == []  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown mode"):
        r.search("contract", mode="fuzzy")  # type: ignore[arg-type]


@pytest.mark.db
def test_filters(corpus: Engine) -> None:
    r = LexicalRetriever(corpus)

    def docs(f: SearchFilters, q: str = "contract penalty compensation") -> set[str]:
        return {h.doc_id for h in r.search(q, f)}

    assert docs(SearchFilters()) == {SC_OLD, SC_NEW, HC, ACT}
    # court level and dates constrain judgments; the statute passes them
    assert docs(SearchFilters(court_levels=(CourtLevel.HIGH_COURT,))) == {HC, ACT}
    assert docs(SearchFilters(date_from=dt.date(2000, 1, 1))) == {SC_NEW, HC, ACT}
    assert docs(SearchFilters(date_to=dt.date(2015, 6, 1))) == {SC_OLD, SC_NEW, ACT}
    assert docs(SearchFilters(doc_kinds=(DocumentKind.JUDGMENT,))) == {SC_OLD, SC_NEW, HC}
    assert docs(SearchFilters(doc_kinds=(DocumentKind.STATUTE,))) == {ACT}
    # the Act's own text, and judgments citing it
    assert docs(SearchFilters(acts=("Contract Act",))) == {SC_OLD, ACT}
    assert docs(SearchFilters(acts=("SRA",))) == set()
    assert docs(SearchFilters(doc_ids=(HC,))) == {HC}
    both = SearchFilters(
        court_levels=(CourtLevel.SUPREME_COURT,), doc_kinds=(DocumentKind.JUDGMENT,)
    )
    assert docs(both) == {SC_OLD, SC_NEW}
    for mode in ("all", "phrase"):
        hits = r.search("contract", SearchFilters(doc_ids=(HC,)), mode=mode)  # type: ignore[arg-type]
        assert {h.doc_id for h in hits} == {HC}


@pytest.mark.db
def test_k_and_the_candidate_cap(corpus: Engine) -> None:
    r = LexicalRetriever(corpus)
    assert len(r.search("contract", k=2)) == 2
    assert len(r.search("contract", k=2, mode="all")) == 2
    # a cap below every term's chunk count still returns a (partial) ranking
    capped = LexicalRetriever(corpus, max_candidates=1)
    assert len(capped.search("contract")) == 1
    assert len(capped.search("contract", mode="all")) == 1


@pytest.mark.db
def test_max_terms_keeps_the_rarest_terms_of_a_long_query(corpus: Engine) -> None:
    query = "contract contract penalty frustration seller impossible performance pleaded"
    capped = LexicalRetriever(corpus, max_terms=1).search(query)
    # the rarest term here is in one chunk; the other terms no longer count
    assert {h.chunk_id for h in capped} == {f"{SC_NEW}#c0002"}
    assert len(LexicalRetriever(corpus).search(query)) > len(capped)
