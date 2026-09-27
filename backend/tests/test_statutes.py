"""Statute ingestion (PLAN 3.5).

India Code text from the corpus isn't redistributable (docs/DATA_NOTICE.md), so the committed
tests use a small synthetic Act laid out the way India Code prints one (arrangement of sections,
footnotes and a page number at each page foot). The real Acts are checked when the local
snapshot is present (skipped otherwise, e.g. in CI).
"""

import datetime as dt
import gzip
import json
from pathlib import Path

import pytest

from juris.config import get_settings
from juris.ingest.statutes import (
    Act,
    apply_amendments,
    arrangement,
    load_acts,
    load_amendments,
    resolve_act,
    split_act,
)

EM = chr(0x2014)
ACT = Act("demo_act", "Demo Act", 1999, "1 of 1999", dt.date(1999, 4, 1), "s. 1", ("DA",))

DEMO = "\n".join(
    [
        "THE DEMO ACT, 1999",
        "ARRANGEMENT OF SECTIONS",
        "SECTIONS",
        "1.",
        "Short title and commencement.",
        "2.",
        "Definitions.",
        "CHAPTER II",
        "REMEDIES",
        "3.",
        "Specific performance.",
        "3A.",
        "Power to engage experts.",
        "4.",
        "Compensation.",
        "6.",
        "Costs.",
        "7.",
        "[Repealed.].",
        "1",
        "THE DEMO ACT, 1999",
        "ACT NO. 1 OF 1999",
        f"1. Short title and commencement.{EM} (1 ) This Act may be called the Demo Act, 1999.",
        "(2 ) It shall come into force on the 1st day of April, 1999.",
        f"2. Definitions.{EM}In this Act,{EM}",
        '(a ) "court" means a civil court;',
        "CHAPTER II",
        "REMEDIES",
        f"1 [3. Specific performance.{EM} The court shall enforce a contract.]",
        "1.",
        "Subs. by Act 5 of 2018, s. 2, for section 3 (w.e.f. 1-10-2018).",
        "2",
        f"3A. Power to engage experts.{EM} (1 ) The court may engage experts.",
        "(2 ) The expert shall be paid.",
        "Compensation",  # a run-in cross-heading
        f"4.Compensation.{EM}A party may claim compensation[1] [in addition to] performance.",
        "1.",
        'Subs. by Act 5 of 2018, s. 3, for "or" (w.e.f. 1-10-2018).',
        "3",
        f"5. Interest.{EM}Interest may be awarded.",  # missing from the arrangement
        f"6. Costs.{EM}Costs follow the event.",
        "1* * * * *",
        "1.",
        "Section 7 rep. by the Repealing Act, 2001 (9 of 2001), s. 2 (w.e.f. 1-1-2002).",
        "4",
        "THE SCHEDULE",
        "1. Roads.",
        "2. Railways.",
    ]
)


def sections() -> dict[str, object]:
    return {s.section: s for s in split_act(DEMO, ACT)}


def test_arrangement_lists_sections_and_finds_the_body() -> None:
    entries, body_start = arrangement(DEMO)
    assert [no for no, _ in entries] == ["1", "2", "3", "3A", "4", "6", "7"]
    assert dict(entries)["3A"] == "Power to engage experts"
    assert DEMO[body_start:].startswith("1. Short title")


def test_split_keeps_every_section_in_order() -> None:
    secs = split_act(DEMO, ACT)
    assert [s.section for s in secs] == ["1", "2", "3", "3A", "4", "5", "6", "7"]
    by_no = {s.section: s for s in secs}
    assert by_no["5"].title == "Interest"  # recovered though the arrangement skipped it
    assert by_no["3"].chapter == "CHAPTER II"
    assert by_no["6"].text == f"6. Costs.{EM}Costs follow the event.\n* * * * *"  # schedule cut


def test_sub_sections_stay_with_their_section() -> None:
    by_no = {s.section: s for s in split_act(DEMO, ACT)}
    assert by_no["1"].sub_sections == ["(1)", "(2)"]
    assert "(2) The expert shall be paid." in by_no["3A"].text
    assert "Compensation" not in by_no["3A"].text.split("\n")[-1]  # cross-heading dropped


def test_footnotes_resolve_to_amendments_and_markers_are_removed() -> None:
    by_no = {s.section: s for s in split_act(DEMO, ACT)}
    assert by_no["3"].amended_by == ["Act 5 of 2018"]
    assert by_no["3"].amendment_notes == [
        "Subs. by Act 5 of 2018, s. 2, for section 3 (w.e.f. 1-10-2018)."
    ]
    assert by_no["4"].text.startswith("4.Compensation.")
    assert "[in addition to]" in by_no["4"].text and "[1]" not in by_no["4"].text
    assert by_no["1"].amended_by == [] and not by_no["1"].amendments_curated
    assert by_no["1"].in_force_from == dt.date(1999, 4, 1)  # coarse: the Act's commencement


def test_repealed_section_listed_only_in_the_arrangement_gets_a_stub() -> None:
    stub = {s.section: s for s in split_act(DEMO, ACT)}["7"]
    assert stub.repealed and stub.text == "7. [Repealed.]"
    assert stub.amendment_notes == [
        "Section 7 rep. by the Repealing Act, 2001 (9 of 2001), s. 2 (w.e.f. 1-1-2002)."
    ]


def test_curated_amendments_override_temporal_fields() -> None:
    secs = split_act(DEMO, ACT)
    entries = [
        {
            "act": "demo_act",
            "sections": ["3", "3A"],
            "by": "Act 5 of 2018",
            "in_force_from": dt.date(2018, 10, 1),
            "source": "test",
        },
        {
            "act": "demo_act",
            "sections": ["7"],
            "by": "Act 9 of 2001",
            "in_force_to": dt.date(2001, 12, 31),
            "source": "test",
        },
    ]
    assert apply_amendments(secs, entries) == 3
    by_no = {s.section: s for s in secs}
    assert by_no["3A"].in_force_from == dt.date(2018, 10, 1) and by_no["3A"].amendments_curated
    assert by_no["3A"].amended_by == ["Act 5 of 2018"]
    assert by_no["7"].in_force_to == dt.date(2001, 12, 31)
    assert not by_no["4"].amendments_curated  # noted in a footnote but not curated
    meta = by_no["3"].meta()
    assert meta.section == "3" and meta.amendments_curated and meta.in_force_to is None


@pytest.mark.parametrize(
    ("name", "act_id"),
    [
        ("ICA", "contract_act"),
        ("Contract Act", "contract_act"),
        ("Indian Contract Act, 1872", "contract_act"),
        ("the Indian Contract Act", "contract_act"),
        ("SRA", "specific_relief_act"),
        ("Specific Relief Act, 1963", "specific_relief_act"),
        ("Sale of Goods Act, 1930", "sale_of_goods_act"),
        ("Indian Sale of Goods Act, 1930", "sale_of_goods_act"),
        ("Transfer of Property Act", None),
    ],
)
def test_alias_table(name: str, act_id: str | None) -> None:
    assert resolve_act(name, load_acts()) == act_id


def test_curation_file_entries_name_known_acts_and_sources() -> None:
    acts = load_acts()
    for entry in load_amendments():
        assert entry["act"] in acts
        assert entry["source"] and "checked" in entry["source"]
        assert "in_force_from" in entry or "in_force_to" in entry


# ---- the real Acts, when the local snapshot is available ---------------------------------

SNAPSHOT = "mvp_contract-1f53c208a8"


def real_text(act_id: str) -> str:
    path = get_settings().data_dir / "interim" / "parsed" / SNAPSHOT / f"ACT-{act_id}.json.gz"
    if not Path(path).exists():
        pytest.skip("local corpus snapshot not available")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        text: str = json.load(fh)["clean_text"]
    return text


def test_contract_act_has_every_section_1_to_238() -> None:
    secs = split_act(real_text("contract_act"), load_acts()["contract_act"])
    numbers = [s.section for s in secs]
    assert all(str(n) in numbers for n in range(1, 267))
    assert len(numbers) == 268  # India Code lists 268: ss. 1-266 plus 19A and 178A
    repealed = {s.section for s in secs if s.repealed}
    assert repealed == {str(n) for n in (*range(76, 124), *range(239, 267))}


def test_real_acts_match_india_code_counts_and_curation() -> None:
    acts, amendments = load_acts(), load_amendments()
    counts = {}
    for act_id in ("specific_relief_act", "sale_of_goods_act"):
        secs = split_act(real_text(act_id), acts[act_id])
        apply_amendments(secs, amendments)
        counts[act_id] = len(secs)
        if act_id == "specific_relief_act":
            by_no = {s.section: s for s in secs}
            assert by_no["10"].in_force_from == dt.date(2018, 10, 1)
            assert by_no["20C"].amended_by == ["Act 18 of 2018"]
            assert by_no["43"].repealed and by_no["43"].in_force_to == dt.date(1974, 12, 19)
    assert counts == {"specific_relief_act": 48, "sale_of_goods_act": 67}
