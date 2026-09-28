"""Citation graph (PLAN 3.9): resolution, aliases, conflicts, statute mentions, cues.

Citation strings and case names below are real (from the corpus and the AWS index); the
surrounding sentences are shortened.
"""

import datetime as dt

from juris.ingest.citation_graph import (
    AliasTable,
    ScIndex,
    document_edges,
    parallel_groups,
    statute_mentions,
    treatment_cue,
)
from juris.ingest.citations import parse_citation
from juris.ingest.statutes import load_acts

INDEX = ScIndex.from_rows(
    [
        {"path": "1978_2_621_776", "case_id": "1978 INSC 16"},  # Maneka Gandhi
        {"path": "1986_2_278_387", "case_id": "1986 INSC 81"},  # Central Inland Water
        {"path": "1986_2_278_387", "case_id": "1986 INSC 81"},  # duplicate row (AWS)
        {"path": "S_1985_3_909_1024", "case_id": "1985 INSC 256"},  # LIC v. Escorts (Supp)
        {"path": "1972_2_318_343", "case_id": "1971 INSC 290"},  # Sadiq Ali
        {"path": "1952_1_683_690", "case_id": "1952 INSC 30"},
        {"path": "1953_1_210_215", "case_id": "1952 INSC 63"},
        {"path": "2015_3_243_286", "case_id": "2015 INSC 163"},
    ]
)


def cite(text: str):  # type: ignore[no-untyped-def]
    c = parse_citation(text)
    assert c is not None, text
    return c


def test_index_resolves_insc_and_first_page_scr_only() -> None:
    assert INDEX.resolve(cite("[1978] 2 SCR 621")) == ("SC-1978_2_621_776", "direct")
    assert INDEX.resolve(cite("1986 INSC 81")) == ("SC-1986_2_278_387", "direct")
    assert INDEX.resolve(cite("1985 (3) Suppl. SCR 909")) == ("SC-S_1985_3_909_1024", "direct")
    assert INDEX.resolve(cite("[1985] 3 SCR 909")) is None  # not the Supplementary volume
    assert INDEX.resolve(cite("(1972) 2 SCR 331")) is None  # inside Sadiq Ali: a misprint
    assert INDEX.resolve(cite("(1986) 3 SCC 156")) is None  # SCC needs an alias


def test_parallel_groups_join_colon_and_equals_only() -> None:
    text = (
        "Central Inland Water Transport v. Brojo Nath Ganguly (1986) 3 SCC 156 : [1986] 2 SCR "
        "278; Maneka Gandhi [1978] 2 SCR 621 = (1978) 1 SCC 248. Also (1990) 3 SCC 517 and "
        "1996 Supp (2) SCC 1."
    )
    mentions, groups = parallel_groups(text)
    names = [[mentions[i].citation.canonical for i in g] for g in groups]
    assert names == [
        ["(1986) 3 SCC 156", "[1986] 2 SCR 278"],
        ["[1978] 2 SCR 621", "(1978) 1 SCC 248"],
        ["(1990) 3 SCC 517"],
        ["1996 Supp (2) SCC 1"],
    ]


def test_aliases_learnt_from_groups_and_ambiguity_dropped() -> None:
    table = AliasTable()
    table.learn([cite("(1986) 3 SCC 156"), cite("[1986] 2 SCR 278")], INDEX)
    table.learn([cite("AIR 1986 SC 1571"), cite("1986 INSC 81")], INDEX)
    table.learn([cite("(1978) 1 SCC 248"), cite("(1978) 2 SCC 1")], INDEX)  # nothing resolves
    assert table.resolve(cite("(1986) 3 SCC 156")) == "SC-1986_2_278_387"
    assert table.resolve(cite("AIR 1986 SC 1571")) == "SC-1986_2_278_387"
    assert table.resolve(cite("(1978) 1 SCC 248")) is None
    table.learn([cite("(1986) 3 SCC 156"), cite("[1978] 2 SCR 621")], INDEX)  # contradicts
    assert table.resolve(cite("(1986) 3 SCC 156")) is None and table.ambiguous == 1


def test_document_edges_resolve_skip_self_and_distrust_conflicts() -> None:
    aliases = AliasTable()
    aliases.learn([cite("(1986) 3 SCC 156"), cite("[1986] 2 SCR 278")], INDEX)
    text = (
        "[2015] 3 S.C.R. 243 : 2015 INSC 163\n"
        "1. Central Inland Water (1986) 3 SCC 156 was followed. Maneka Gandhi [1978] 2 SCR "
        "621 was relied on.\n"
        "2. Shamarao Parulekar [1952] 1 SCR 683 : 1952 INSC 63 is not in point."
    )
    p1, p2 = text.index("1. Central"), text.index("2. Shamarao")
    paragraphs = [
        {"no": "f-1", "char_start": 0, "char_end": p1 - 1},
        {"no": "1", "char_start": p1, "char_end": p2 - 1},
        {"no": "2", "char_start": p2, "char_end": len(text)},
    ]
    corpus = {"SC-2015_3_243_286", "SC-1986_2_278_387"}
    edges = document_edges(
        "SC-2015_3_243_286", text, paragraphs, INDEX, aliases, corpus, set(), load_acts()
    )
    got = {e.canonical: (e.target_ref, e.target_doc_id, e.resolution, e.source_para) for e in edges}
    assert "[2015] 3 SCR 243" not in got and "2015 INSC 163" not in got  # its own citations
    assert got["(1986) 3 SCC 156"] == ("SC-1986_2_278_387", "SC-1986_2_278_387", "alias", "1")
    assert got["[1978] 2 SCR 621"] == ("SC-1978_2_621_776", None, "direct", "1")  # not in corpus
    # two different judgments printed as one parallel citation: neither is trusted
    assert got["[1952] 1 SCR 683"] == (None, None, "conflict", "2")
    assert got["1952 INSC 63"] == (None, None, "conflict", "2")
    by = {e.canonical: e for e in edges}
    assert by["[1978] 2 SCR 621"].treatment == "relied_on"
    assert by["(1986) 3 SCC 156"].treatment == "followed"


def test_statute_mentions_resolve_through_act_aliases() -> None:
    text = (
        "under Section 16(c) of the Specific Relief Act, 1963, ss. 73 and 74 of the Contract "
        "Act, Section 55 of the Sale of Goods Act, 1930, Section 54 of the Transfer of Property "
        "Act and Section 12 of the Specific Relief Act, 1877."
    )
    got = [(m.section, m.act_id) for m in statute_mentions(text, load_acts())]
    assert got == [
        ("16", "specific_relief_act"),
        ("73", "contract_act"),
        ("74", "contract_act"),
        ("55", "sale_of_goods_act"),
        ("54", None),
        ("12", None),  # the 1877 Act, a different statute
    ]


def test_undated_act_before_its_commencement_is_the_older_act() -> None:
    text = "1. The suit is under Section 42 of the Specific Relief Act."
    paragraphs = [{"no": "1", "char_start": 0, "char_end": len(text)}]
    corpus: set[str] = set()
    common = (INDEX, AliasTable(), corpus, {"specific_relief_act:42"}, load_acts())
    old = document_edges("SC-1955_1_1_2", text, paragraphs, *common, decided=dt.date(1955, 1, 1))
    new = document_edges("SC-1990_1_1_2", text, paragraphs, *common, decided=dt.date(1990, 1, 1))
    assert old[0].target_section_id is None and old[0].resolution is None
    assert new[0].target_section_id == "specific_relief_act:42"


def test_treatment_cue_is_the_nearest_in_the_sentence() -> None:
    text = (
        "The earlier view was followed for years. In Kharak Singh v. State of U.P. AIR 1963 SC "
        "1295, partly overruled in Maneka Gandhi, the Court held otherwise."
    )
    start = text.index("AIR 1963")
    end = start + len("AIR 1963 SC 1295")
    assert treatment_cue(text, start, end) == ("overruled", "overruled")
    assert treatment_cue("No cue here (1990) 3 SCC 517.", 12, 28) is None
