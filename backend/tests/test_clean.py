"""Cleaning (PLAN 3.2): golden excerpts from real judgments plus rule-level unit tests.

Golden inputs in ``golden/parse/`` are the raw text layers of a few pages of real
judgments (AWS Indian SC/HC Judgments, CC-BY-4.0; see golden/parse/README.md).
"""

import json
from pathlib import Path
from typing import Any

import pytest

from juris.ingest.clean import CleanResult, clean, latin_ratio, ocr_noise

GOLDEN = Path(__file__).parent / "golden" / "parse"
BREAK = chr(0xFFFE)  # pdfium's line-break hyphen
RSQUO = chr(0x2019)  # right single quotation mark

# doc_id -> (must appear in the clean text, must not appear)
EXPECT: dict[str, tuple[list[str], list[str]]] = {
    "SC-2023_12_979_1033": (
        [
            "[2023] 12 S.C.R. 979 : 2023 INSC 736",  # the reporter line on page 1 is kept
            "fixed deposits",  # split ligature "fi xed" repaired
            "Bangalore Club's case",  # curly apostrophe normalised
        ],
        ["SUPREME COURT REPORTS", "fi xed", RSQUO],
    ),
    "SC-1963_3_22_183": (
        ["R. VISWANATHAN", "One Ramalingam died at Bangalore"],
        ["COUR1' REPOR", "SUPREME COURT REPORTS", BREAK],
    ),
    "SC-S_1996_7_641_643": (
        [
            "NATIONAL TEXTILE CORPORATION (U.P.) LTD. ETC.\nv.",  # trailing margin "A" gone
            "[K. RAMASWAMY AND G.B. PATTANAIK, JJ.]\n",  # trailing margin "B" gone
            "plaintiff-appellant",  # real hyphen at a line break kept
        ],
        ["SUPREME COURT REPORTS", "\nA\n", "\nB\n", BREAK],
    ),
    "SC-2015_13_1_1056": (["organic development of civil society"], [BREAK]),
    "SC-S_2006_1_587_602": (
        ["[S.B. SINHA AND P.P. NAOLEKAR, JJ.]"],
        ["SUPREME COURT REPORTS", BREAK],
    ),
    "HC-DLHC010792242017_1_2018-03-22": (
        ["IN THE HIGH COURT OF DELHI AT NEW DELHI", "1. These are a batch of appeals"],
        ["Page 1 of 60", "Page 2 of 60", BREAK],
    ),
    "HC-HCBM020032982020_1_2024-04-10": (
        ["COMM ARBITRATION PETITION NO.342 OF 2020", "2. The limited issue"],
        ["8-carbp-342-2020.doc", "Digitally signed", "+0530", RSQUO],
    ),
}


def golden(doc_id: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((GOLDEN / f"{doc_id}.json").read_text(encoding="utf-8"))
    return data


@pytest.mark.parametrize("doc_id", sorted(EXPECT))
def test_golden_excerpts(doc_id: str) -> None:
    case = golden(doc_id)
    result = clean(case["pages"], case["profile"])
    present, absent = EXPECT[doc_id]
    for snippet in present:
        assert snippet in result.text, snippet
    for snippet in absent:
        assert snippet not in result.text, snippet
    assert latin_ratio(result.text) > 0.95


@pytest.mark.parametrize("doc_id", sorted(EXPECT))
def test_every_clean_char_maps_back_to_its_raw_char(doc_id: str) -> None:
    case = golden(doc_id)
    raw = "\f".join(case["pages"])
    result = clean(case["pages"], case["profile"])
    for i, ch in enumerate(result.text):
        if ch in " \n\"'-":  # collapsed whitespace, normalised quotes, kept break hyphens
            continue
        assert raw[result.to_raw(i)] == ch, i
    assert result.page_starts == sorted(result.page_starts)
    for page, start in enumerate(result.page_starts, start=1):
        if start < len(result.text) and result.text[start] != "\n":
            assert raw[: result.to_raw(start)].count("\f") == page - 1


def lines(text: str) -> list[str]:
    return text.split("\n")


def test_repeated_headers_and_page_numbers_are_dropped() -> None:
    bodies = ["The suit was filed.", "Issues were framed.", "Evidence was led.", "Appeal allowed."]
    pages = [f"Case No. 7/2020 Page {n} of 4\n{body}\n{n}" for n, body in enumerate(bodies, 1)]
    result = clean(pages, "hc")
    assert lines(result.text) == bodies
    # 4 running heads + 4 page numbers (bare numbers recur as "#" and may count as either)
    dropped = result.stats["dropped_repeated_header"] + result.stats["dropped_page_number"]
    assert dropped == 8
    assert result.page_of(result.text.index("Evidence")) == 3


def test_signature_block_is_dropped() -> None:
    page = "\n".join(
        [
            "5. The petition is dismissed.",
            "Digitally",
            "signed by",
            "A B NAME",
            "Date: 2024.04.16",
            "17:03:43 +0530",
            "6. No costs.",
        ]
    )
    assert lines(clean([page], "hc").text) == ["5. The petition is dismissed.", "6. No costs."]


def test_text_starting_with_digitally_is_not_a_signature() -> None:
    page = "\n".join(
        [
            "12. The agreement was signed on paper.",
            "Digitally signed agreements are equally binding under the Act, and the",
            "parties here exchanged signed copies by e-mail on 3rd March.",
            "13. The appeal fails.",
        ]
    )
    assert len(lines(clean([page], "hc").text)) == 4


def test_break_hyphens_join_or_keep_by_vocabulary() -> None:
    page = "\n".join(
        [
            "the understanding of the parties",
            f"a new under{BREAK}standing was reached",
            "the plaintiff and the appellant",
            f"the plaintiff{BREAK}appellant filed a suit",
            f"Justice Bala{BREAK}krishanaiya",
        ]
    )
    text = clean([page], "hc").text
    assert "a new understanding was reached" in text
    assert "the plaintiff-appellant filed" in text
    assert "Justice Balakrishanaiya" in text


def test_ligature_splits_are_repaired_but_real_words_are_not() -> None:
    page = "the fi xed deposit and its eff ect\nthe staff member said off the record\nan effect"
    text = clean([page], "hc").text
    assert "fixed deposit" in text and "its effect" in text
    assert "staff member" in text and "off the record" in text


def test_scr_margin_letters() -> None:
    page = "\n".join(
        [
            "THE PARTIES A",
            "v.",
            "THE STATE B",
            "A",
            "B",
            "The contract was signed by the C",
            "appellants and D",
            "E the respondent in Bombay E",
            "F on the same day. F",
            "A and the parties",
            "G were bound.",
            "H the deed was stamped.",
            "B later registered.",
            "A notice was issued.",
        ]
    )
    assert lines(clean([page], "sc").text) == [
        "THE PARTIES",
        "v.",
        "THE STATE",
        "The contract was signed by the",
        "appellants and",
        "the respondent in Bombay",
        "on the same day.",
        "and the parties",  # "A and": a margin letter, not the article
        "were bound.",
        "the deed was stamped.",
        "later registered.",
        "A notice was issued.",  # sentence-initial article kept
    ]


def test_statute_profile_drops_picture_markers_only() -> None:
    text = "\n".join(
        [
            "THE SAMPLE ACT, 2000",
            "==> picture [66 x 80] intentionally omitted <==",
            "1. Short title.",
            "A Section starting with A.",
        ]
    )
    assert lines(clean([text], "statute").text) == [
        "THE SAMPLE ACT, 2000",
        "1. Short title.",
        "A Section starting with A.",
    ]


def test_to_raw_rejects_offsets_outside_the_text() -> None:
    result: CleanResult = clean(["abc"], "hc")
    assert result.to_raw(0) == 0 and result.to_raw(2) == 2
    with pytest.raises(IndexError):
        result.to_raw(3)


def test_quality_signals() -> None:
    devanagari = "".join(chr(c) for c in (0x0905, 0x0928, 0x0941, 0x092C, 0x0902, 0x0927))
    assert latin_ratio("The contract") == 1.0
    assert latin_ratio(f"{devanagari} {devanagari} contract") < 0.85
    assert ocr_noise("SUPl{EMl!: COURT REPOR'r8 fine words here") > 0
    assert ocr_noise("clean words only") == 0
