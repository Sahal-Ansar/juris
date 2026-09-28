"""Structured lookups (PLAN 4.5): provisions, citations, titles and the citation graph.

The ``db`` tests build a few documents, sections, aliases and edges in a fresh database.
"""

import datetime as dt
from typing import Any

import pytest
from sqlalchemy import insert
from sqlalchemy.engine import Engine

from juris.db import models as m
from juris.retrieval.lookup import Lookup, Provision, parse_provision
from tests.conftest import migrate

# ---- no database needed ------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference",
    [
        "s. 15 ICA",
        "Section 15 of the Indian Contract Act",
        "Contract Act s.15",
        "S.15 of ICA",
        "sec 15, Indian Contract Act, 1872",
        "section 15 under the Contract Act",
        "ICA, s 15",
    ],
)
def test_the_same_section_however_it_is_written(reference: str) -> None:
    assert parse_provision(reference) == Provision("contract_act", "15")


def test_sub_sections_letters_and_other_acts() -> None:
    assert parse_provision("u/s 16(2) of the Contract Act, 1872") == Provision(
        "contract_act", "16", "(2)"
    )
    assert parse_provision("Section 16 (2) (a) ICA") == Provision("contract_act", "16", "(2)(a)")
    assert parse_provision("s. 19a ICA") == Provision("contract_act", "19A")
    assert parse_provision("section 10 SRA").section_id == "specific_relief_act:10"  # type: ignore[union-attr]
    assert parse_provision("sec. 64A Sale of Goods Act") == Provision("sale_of_goods_act", "64A")


@pytest.mark.parametrize(
    "reference",
    [
        "section 15",  # no Act
        "Indian Contract Act",  # no section
        "s. 12 Specific Relief Act, 1877",  # the repealed 1877 Act is a different Act
        "Section 15 of the Transfer of Property Act",  # not in the corpus
        "",
    ],
)
def test_unresolvable_references(reference: str) -> None:
    assert parse_provision(reference) is None


# ---- database ----------------------------------------------------------------------------

SAW = "SC-2003_3_691_741"  # ONGC v. Saw Pipes
SSANG = "SC-2019_7_522_606"  # cites Saw Pipes twice, and s. 74
GROSE = "SC-1954_1_310_326"  # Satyabrata Ghose (title OCR-damaged)
OUTSIDE = "SC-1963_3_1_30"  # a judgment outside the corpus, known only from citations
ACT = "ACT-contract_act"


@pytest.fixture
def corpus(engine: Engine) -> Engine:
    docs: list[dict[str, Any]] = [
        {"doc_id": SAW, "kind": "judgment", "title": "OIL & NATURAL GAS CORPORATION LTD. "
         "versus SAW PIPES LTD.", "decision_date": dt.date(2003, 4, 17),
         "citations": ["[2003] 3 SCR 691", "2003 INSC 241"], "neutral_citation": "2003 INSC 241"},
        {"doc_id": SSANG, "kind": "judgment", "title": "SSANGYONG CONSTRUCTION CO. LTD. versus "
         "NATIONAL HIGHWAYS AUTHORITY OF INDIA", "decision_date": dt.date(2019, 5, 8),
         "citations": ["[2019] 7 SCR 522"]},
        {"doc_id": GROSE, "kind": "judgment", "title": "SATYABRATA GROSE versus MUGNEERAM "
         "BANGUR & CO., AND ANOTHER.", "decision_date": dt.date(1953, 11, 16),
         "citations": ["[1954] SCR 310"]},
        {"doc_id": ACT, "kind": "statute", "title": "Indian Contract Act, 1872"},
    ]  # fmt: skip
    sections = [
        ("15", '"Coercion" defined', False),
        ("19A", "Power to set aside contract induced by undue influence", False),
        ("74", "Compensation for breach of contract where penalty stipulated for", False),
        ("76", "Goods defined", True),
    ]
    edges = [
        (SSANG, "(2003) 5 SCC 705", SAW, SAW, "case", None, "10", "followed", "alias"),
        (SSANG, "[2003] 3 SCR 691", SAW, SAW, "case", None, "12", None, "direct"),
        (SAW, "[1963] 3 SCR 1", None, OUTSIDE, "case", None, "p-31", None, "direct"),
        (GROSE, "[1963] 3 SCR 1", None, OUTSIDE, "case", None, "3", None, "direct"),
        (SSANG, "s. 74", None, None, "statute", "contract_act:74", "20", None, "act"),
    ]
    with engine.begin() as conn:
        migrate(conn)
        conn.execute(insert(m.Snapshot).values(snapshot_id="s", slice_name="t"))
        conn.execute(
            insert(m.Document),
            [
                {
                    "neutral_citation": None,
                    "decision_date": None,
                    "citations": [],
                    **d,
                    "snapshot_id": "s",
                    "source": "test",
                    "licence": "test",
                }
                for d in docs
            ],  # every row with the same keys, for executemany
        )
        conn.execute(
            insert(m.StatuteSection),
            [
                {"section_id": f"contract_act:{no}", "doc_id": ACT, "act_id": "contract_act",
                 "act": "Indian Contract Act", "act_year": 1872, "section": no, "title": title,
                 "text": f"{no}. {title}.", "repealed": repealed, "char_start": 0,
                 "char_end": 5, "page_start": 1, "page_end": 1,
                 "in_force_from": dt.date(1872, 9, 1),
                 "in_force_to": dt.date(1930, 7, 1) if repealed else None}
                for no, title, repealed in sections
            ],
        )  # fmt: skip
        conn.execute(
            insert(m.Chunk),
            [
                {"chunk_id": f"{ACT}#s74-{i}", "doc_id": ACT, "text": "t", "char_start": i,
                 "char_end": i + 1, "statute_section_id": "contract_act:74"}
                for i in (2, 1)
            ],
        )  # fmt: skip
        conn.execute(
            insert(m.CitationAlias).values(alias="(2003) 5 SCC 705", target_ref=SAW, evidence=3)
        )
        conn.execute(
            insert(m.CitationEdge),
            [
                {"source_doc_id": src, "raw": canon, "canonical": canon, "target_doc_id": tdoc,
                 "target_ref": tref, "kind": kind, "target_section_id": tsec, "source_para": para,
                 "treatment": treat, "resolution": how, "char_start": i, "char_end": i + 5}
                for i, (src, canon, tdoc, tref, kind, tsec, para, treat, how) in enumerate(edges)
            ],
        )  # fmt: skip
    return engine


@pytest.mark.db
def test_the_three_forms_fetch_the_same_section(corpus: Engine) -> None:
    lk = Lookup(corpus)
    found = [
        lk.lookup_provision(r)
        for r in ("s. 15 ICA", "Section 15 of the Indian Contract Act", "Contract Act s.15")
    ]
    assert all(s is not None and s.section_id == "contract_act:15" for s in found)
    assert found[0] == lk.get_section("Indian Contract Act", "15")
    s74 = lk.get_section("ICA", "74")
    assert s74 is not None and s74.title.startswith("Compensation for breach")
    assert s74.chunk_ids == [f"{ACT}#s74-1", f"{ACT}#s74-2"]  # in text order
    assert lk.get_section("contract_act", "19a").section == "19A"  # type: ignore[union-attr]
    repealed = lk.get_section("ICA", "76")
    assert repealed is not None and repealed.repealed and repealed.in_force_to
    assert lk.get_section("ICA", "999") is None
    assert lk.get_section("Transfer of Property Act", "15") is None
    assert lk.lookup_provision("section 15") is None


@pytest.mark.db
def test_citations_resolve_through_documents_aliases_and_the_graph(corpus: Engine) -> None:
    lk = Lookup(corpus)
    by_scr = lk.get_case_by_citation("[2003] 3 S.C.R. 691")  # normalised before lookup
    assert by_scr is not None and (by_scr.doc_id, by_scr.how) == (SAW, "document")
    assert by_scr.citation == "[2003] 3 SCR 691"
    by_insc = lk.get_case_by_citation("2003 INSC 241")
    assert by_insc is not None and by_insc.doc_id == SAW
    by_scc = lk.get_case_by_citation("(2003) 5 S.C.C. 705")  # the SCC twin, via the alias
    assert by_scc is not None and (by_scc.doc_id, by_scc.how) == (SAW, "alias")
    outside = lk.get_case_by_citation("(1963) 3 SCR 1")
    assert outside is not None
    assert (outside.target_ref, outside.doc_id, outside.how) == (OUTSIDE, None, "citation_graph")
    assert lk.get_case_by_citation("(1999) 1 SCC 1") is None  # parses, but unknown
    assert lk.get_case_by_citation("not a citation") is None


@pytest.mark.db
def test_titles_match_fuzzily(corpus: Engine) -> None:
    lk = Lookup(corpus)
    ghose = lk.get_case_by_title("Satyabrata Ghose v. Mugneeram Bangur")
    assert ghose and ghose[0].doc_id == GROSE and ghose[0].score > 0.7
    assert lk.get_case_by_title("Saw Pipes")[0].doc_id == SAW
    assert lk.get_case_by_title("ONGC vs Saw Pipes")[0].doc_id == SAW
    assert lk.get_case_by_title("Indian Contract Act") == []  # statutes are not cases
    assert lk.get_case_by_title("Zebra Crossing Holdings") == []
    assert lk.get_case_by_title("  ") == []


@pytest.mark.db
def test_citing_and_cited_cases(corpus: Engine) -> None:
    lk = Lookup(corpus)
    citing = lk.get_citing_cases(SAW)
    assert [c.doc_id for c in citing] == [SSANG]
    assert citing[0].mentions == 2 and citing[0].paragraphs == ["10", "12"]
    assert citing[0].citations == ["(2003) 5 SCC 705", "[2003] 3 SCR 691"]
    assert citing[0].treatments == ["followed"]
    cited = lk.get_cited_cases(SAW)
    assert [(c.ref, c.doc_id, c.title) for c in cited] == [(OUTSIDE, None, None)]
    outside_citers = lk.get_citing_cases(OUTSIDE)  # a judgment outside the corpus
    assert {c.doc_id for c in outside_citers} == {SAW, GROSE}
    assert outside_citers[0].decision_date == dt.date(2003, 4, 17)  # newest first on a tie
    s74 = lk.get_cases_citing_section("Contract Act", "74")
    assert [(c.doc_id, c.paragraphs) for c in s74] == [(SSANG, ["20"])]
    assert lk.get_cases_citing_section("ICA", "15") == []
    assert lk.get_cases_citing_section("Transfer of Property Act", "74") == []
    assert lk.get_citing_cases("SC-nothing") == []
