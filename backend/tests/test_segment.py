"""Paragraph segmentation (PLAN 3.3): golden excerpts plus synthetic cases for each rule.

The synthetic judgments mimic patterns found in the corpus (see docs/data/segmentation_check.md):
issue lists and agreement clauses inside a paragraph, quoted paragraphs of other judgments,
SCR margin tokens, separate opinions, and unnumbered old reports.
"""

import json
from pathlib import Path

import pytest

from juris.ingest.clean import clean
from juris.ingest.segment import Paragraph, Segmentation, candidate_of, segment

GOLDEN = Path(__file__).parent / "golden" / "parse"
FILLER = [
    "The facts are set out in the order under appeal and need not be repeated here at",
    "length, save to note that the parties were heard at length on every point raised.",
]


def body(*paras: str) -> str:
    """Join paragraphs, padding each with filler lines so real paragraphs are spaced out."""
    return "\n".join(line for p in paras for line in (p, *FILLER))


def numbers(seg: Segmentation, part: int | None = None) -> list[str]:
    return [p.no for p in seg.explicit if part is None or p.part == part]


def first_line(p: Paragraph) -> str:
    return p.text.split("\n")[0]


# --- golden excerpts ------------------------------------------------------------------------


def golden(doc_id: str) -> tuple[str, list[int]]:
    case = json.loads((GOLDEN / f"{doc_id}.json").read_text(encoding="utf-8"))
    result = clean(case["pages"], case["profile"])
    return result.text, result.page_starts


def test_golden_hc_judgment_is_numbered_1_to_9() -> None:
    text, page_starts = golden("HC-DLHC010792242017_1_2018-03-22")
    seg = segment(text, page_starts)
    assert seg.numbering == "explicit" and seg.parts == 1 and seg.missing_numbers == []
    assert numbers(seg) == [str(n) for n in range(1, 10)]
    assert first_line(seg.explicit[0]).startswith("1. These are a batch of appeals")
    assert all(p.section == "front" for p in seg.paragraphs if p.numbering == "inferred")
    for p in seg.paragraphs:  # offsets index the clean text; pages rise
        assert text[p.char_start : p.char_end] == p.text
        assert 1 <= p.page_start <= p.page_end
    assert [p.page_start for p in seg.explicit] == sorted(p.page_start for p in seg.explicit)


def test_golden_split_bench_attribution_starts_the_body() -> None:
    # Bhagwandas Kedia (1965): "The Judgment of Wanchoo and Shah. JJ. was delivered by / Shah.
    # J. Hidayatullah, J. delivered a dissenting Opinion." Before the fix only the closing
    # "ORDER" matched, and the whole majority judgment was labelled front matter.
    text, page_starts = golden("SC-1966_1_656_682")
    seg = segment(text, page_starts)
    front = [p for p in seg.paragraphs if p.section == "front"]
    body = [p for p in seg.paragraphs if p.section == "body"]
    assert any(p.section == "headnote" for p in seg.paragraphs)
    assert front[-1].text.startswith("The Judgment of Wanchoo and Shah")
    assert "dissenting" in front[-1].text  # the attribution stays in the front matter
    assert body[0].text.startswith("Shah, J.") and "Girdharilal" in body[0].text
    assert any("ORDER" in p.text for p in body)  # the closing order is body text
    assert len(body) > len(front)


@pytest.mark.parametrize(
    "doc_id", ["SC-1963_3_22_183", "SC-S_1996_7_641_643", "SC-2023_12_979_1033"]
)
def test_golden_unnumbered_excerpts_are_inferred_not_invented(doc_id: str) -> None:
    text, page_starts = golden(doc_id)
    seg = segment(text, page_starts)
    assert seg.numbering == "inferred" and not seg.explicit
    assert all(p.no.startswith(("f-", "h-", "p-")) for p in seg.paragraphs)


# --- candidates -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("line", "first", "last", "punct"),
    [
        ("12. The appeal fails.", 12, 12, "."),
        ("12) The appeal fails.", 12, 12, ")"),
        ("(12) The appeal fails.", 12, 12, "("),
        ("38-39. Both points fail.", 38, 39, "."),
        ("10.In the result", 10, 10, "."),
        ("A 36. The margin letter precedes the number.", 36, 36, "."),
        (". 24. Secondly, the Trial Court erred.", 24, 24, "."),
        ("7.", 7, 7, "."),
    ],
)
def test_candidate_forms(line: str, first: int, last: int, punct: str) -> None:
    c = candidate_of(line)
    assert c is not None and (c.first, c.last, c.punct) == (first, last, punct)


@pytest.mark.parametrize("line", ["1990 was the year.", "Rs. 12.50 was paid.", "12.5% interest"])
def test_non_candidates(line: str) -> None:
    assert candidate_of(line) is None


def test_quoted_items_are_marked() -> None:
    assert candidate_of("3. Whether the plaintiff is entitled to a decree?").quoted_like  # type: ignore[union-attr]
    assert candidate_of("10. That Party No.2 hereby confirms the deal.").quoted_like  # type: ignore[union-attr]
    assert candidate_of("23. XXX XXX").quoted_like  # type: ignore[union-attr]
    assert not candidate_of("3. The suit was decreed.").quoted_like  # type: ignore[union-attr]


# --- chains ---------------------------------------------------------------------------------


def test_simple_chain_with_margin_tokens_ranges_and_gaps() -> None:
    text = body(
        "IN THE SUPREME COURT OF INDIA",
        "1. Leave granted.",
        "A 2. The appellant filed a suit for specific performance.",
        "3. The trial court decreed the suit.",
        "4-5. The High Court reversed the decree on both points.",
        "7. We have heard learned counsel for the parties.",
        "8. The appeal is allowed.",
    )
    seg = segment(text)
    assert numbers(seg) == ["1", "2", "3", "4-5", "7", "8"]
    assert seg.missing_numbers == [6]
    assert not any(p.contested for p in seg.explicit)
    assert seg.paragraphs[0].no == "f-1" and seg.paragraphs[0].section == "front"


def test_issue_list_does_not_take_paragraph_slots() -> None:
    text = body(
        "1. The plaintiff sued for possession.",
        "2. The trial court framed the following issues:-",
        "1. Whether the plaintiff is the owner of the suit land?",
        "2. Whether the suit is barred by limitation?",
        "3. Whether the defendant is in adverse possession?",
        "3. The trial court answered issue no. 1 for the plaintiff.",
        "4. The appeal is dismissed.",
    )
    seg = segment(text)
    assert numbers(seg) == ["1", "2", "3", "4"]
    assert first_line(seg.explicit[2]).startswith("3. The trial court answered")
    assert seg.explicit[2].contested  # "3. Whether ..." nearby: flagged for checking
    assert not seg.explicit[3].contested


def test_agreement_clauses_inside_a_paragraph_are_skipped() -> None:
    text = body(
        "1. The parties entered into an agreement to sell.",
        "2. The relevant clauses read as under:",
        "3. That the seller shall execute the sale deed.",
        "4. That the buyer shall pay the balance.",
        "3. The buyer defaulted on the balance payment.",
        "4. The appeal is dismissed.",
    )
    assert [first_line(p)[:14] for p in segment(text).explicit] == [
        "1. The parties",
        "2. The relevan",
        "3. The buyer d",
        "4. The appeal ",
    ]


def test_quoted_paragraphs_in_other_punctuation_do_not_hijack_the_chain() -> None:
    # The judgment numbers "N)"; a long quotation of another judgment numbers "N.".
    quoted = [f"{n}. In the earlier case the court held point {n}." for n in range(4, 12)]
    text = body(
        "1) Leave granted.",
        "2) The facts are these.",
        "3) Learned counsel relied on the following passage:",
        *quoted,
        "4) We are unable to accept the submission.",
        "5) The appeal is dismissed.",
    )
    seg = segment(text)
    assert numbers(seg) == ["1", "2", "3", "4", "5"]
    assert all(first_line(p)[1] == ")" for p in seg.explicit)


def test_one_ocr_misread_punctuation_is_kept() -> None:
    text = body(*(f"{n}. Paragraph {n} of the judgment." for n in range(1, 8)))
    text = text.replace("5. Paragraph 5", "5) Paragraph 5")
    assert numbers(segment(text)) == [str(n) for n in range(1, 8)]


def test_elided_quoted_paragraphs_lose_to_the_real_one() -> None:
    text = body(
        "1. The appellants are the defendants.",
        "2. The High Court relied on the following passage:",
        "3. The appellants challenged the finding.",
        "4. This Court has held as under:",
        '"2. XXX XXX',
        "3. XXX XXX",
        '4. The seller must prove the deed."',
        "5. The appeal is allowed.",
    )
    seg = segment(text)
    assert numbers(seg) == ["1", "2", "3", "4", "5"]
    assert first_line(seg.explicit[2]).startswith("3. The appellants challenged")


def test_short_bracket_runs_are_not_paragraph_numbering() -> None:
    text = body(
        "The appellant was convicted on the following charges:",
        "(1) theft of the goods;",
        "(2) criminal breach of trust;",
        "(3) cheating.",
        "The conviction was upheld in appeal.",
    )
    seg = segment(text)
    assert seg.numbering == "inferred" and not seg.explicit
    assert all(p.no.startswith(("f-", "p-")) for p in seg.paragraphs)


def test_first_paragraph_on_the_judges_line_is_split_off() -> None:
    text = body(
        "The Judgment of the Court was delivered by",
        "S.B. SINHA, J. 1. Leave granted.",
        "2. The appellant is a company.",
        "3. The appeal is allowed.",
    )
    seg = segment(text)
    assert numbers(seg) == ["1", "2", "3"]
    assert seg.explicit[0].text.startswith("1. Leave granted.")
    assert text[seg.explicit[0].char_start :].startswith("1. Leave granted.")


def test_headnote_held_is_not_split_off() -> None:
    text = body("HELD: 1. The notice was valid.", "The appeal was dismissed.")
    assert segment(text).numbering == "inferred"


def test_separate_opinions_are_parts() -> None:
    text = body(
        "A.B. KUMAR, J.",
        *(f"{n}. Paragraph {n} of the leading opinion." for n in range(1, 13)),
        "C.D. RAO, J.",
        "1. I agree with my learned brother.",
        "2. I add a few words on the second question.",
        "3. The reference is answered as above.",
    )
    seg = segment(text)
    assert seg.parts == 2
    assert numbers(seg, 1) == [str(n) for n in range(1, 13)]
    assert numbers(seg, 2) == ["1", "2", "3"]


def test_numbered_directions_without_a_header_are_not_an_opinion() -> None:
    text = body(
        *(f"{n}. Paragraph {n} of the judgment." for n in range(1, 12)),
        "12. We therefore direct that:",
        "1. the respondents shall pay the arrears;",
        "2. the payment shall be made within eight weeks;",
        "3. the petitioners shall be reinstated.",
    )
    seg = segment(text)
    assert seg.parts == 1 and numbers(seg) == [str(n) for n in range(1, 13)]
    assert "1. the respondents shall pay" in seg.explicit[-1].text


def test_unnumbered_judgment_falls_back_to_inferred_paragraphs() -> None:
    long = "This line is a full-width line of judgment text, as in any reported case here."
    text = "\n".join(
        ["JUDGMENT", long, long, "and the appeal must fail.", long, long, "So ordered."]
    )
    seg = segment(text)
    assert seg.numbering == "inferred" and seg.parts == 0
    body_paras = [p for p in seg.paragraphs if p.section == "body"]
    assert [p.no for p in body_paras] == ["p-1", "p-2"]
    assert body_paras[0].text.endswith("and the appeal must fail.")


LONG = "This line is a full-width line of judgment text, as in any reported case here."
FRONT = [
    "ABC LTD. v. XYZ LTD.",
    "HELD: the appeal fails.",
    LONG,
    "and so it is held.",
    "CIVIL APPELLATE JURISDICTION: Civil Appeal No. 1 of 1970.",
    LONG,
]


def body_start(seg: Segmentation) -> str:
    return next(p for p in seg.paragraphs if p.section == "body").text.split("\n")[0]


def test_wrapped_attribution_and_dissent_line_are_front_matter() -> None:
    text = "\n".join(
        [
            *FRONT,
            "The Judgment of D. G. Palekar and V. R. Krishna Iyer, JJ. was",
            "delivered by Krishna Iyer, J. R. S. Sarkaria, J. gave a dissenting",
            "Opinion.",
            "KRISHNA IYER, J.-The appellant sued for the price of goods sold.",
            *[LONG] * 12,
            "ORDER",
            "The appeal is dismissed.",
        ]
    )
    seg = segment(text)
    assert body_start(seg).startswith("KRISHNA IYER, J.-The appellant")
    assert [p.section for p in seg.paragraphs if "dissenting" in p.text] == ["front"]


@pytest.mark.parametrize(
    "marker",
    [
        "The following Judgment/Order of the Court was delivered by",
        "1953. Nov. 16. The Judgment of the Court were delivered by",
    ],
)
def test_attribution_variants(marker: str) -> None:
    text = "\n".join([*FRONT, marker, "DAS J.-This appeal arises out of a suit.", *[LONG] * 12])
    assert body_start(segment(text)) == "DAS J.-This appeal arises out of a suit."


def test_judges_opening_line_starts_the_body() -> None:
    text = "\n".join(
        [*FRONT, "MUKHERJEA J.-The facts giving rise to this appeal are", *[LONG] * 12]
    )
    assert body_start(segment(text)).startswith("MUKHERJEA J.-The facts")


def test_a_late_marker_is_not_the_start_of_the_body() -> None:
    # Only the closing "ORDER" matches: the judgment must not become front matter.
    text = "\n".join(["ABC LTD. v. XYZ LTD.", *[LONG] * 30, "ORDER", "Appeal dismissed."])
    seg = segment(text)
    assert all(p.section == "body" for p in seg.paragraphs)


def test_sub_markers_are_recorded() -> None:
    text = body(
        "1. The contentions are these:",
        "(i) the notice was defective;",
        "(ii) the suit is barred.",
        "2. Neither contention has merit.",
        "3. The appeal is dismissed.",
    )
    first = segment(text).explicit[0]
    assert [m for m, _ in first.markers] == ["(i)", "(ii)"]
    assert all(text[offset:].startswith(m) for m, offset in first.markers)


def test_numbered_headnote_holdings_are_not_the_courts_numbering() -> None:
    text = body(
        "HELD: 1. The notice under Section 80 was valid.",
        "2. The suit was within limitation.",
        "3. The High Court erred in reversing the decree.",
        "4. The appeal must be allowed.",
        "CIVIL APPELLATE JURISDICTION : Civil Appeal No. 1 of 1990.",
        "The Judgment of the Court was delivered by",
        "A.B. KUMAR, J.",
        *(f"{n}. Paragraph {n} of the judgment." for n in range(1, 4)),
    )
    seg = segment(text)
    assert numbers(seg) == ["1", "2", "3"]
    assert seg.explicit[0].text.startswith("1. Paragraph 1")
    assert any(p.section == "headnote" for p in seg.paragraphs)


def test_concurring_opinion_header_with_a_note() -> None:
    text = body(
        *(f"{n}. Paragraph {n} of the leading opinion." for n in range(1, 13)),
        "R.F. NARIMAN, J. (concurring in the result)",
        "1. I agree with the conclusion.",
        "2. I add a few words.",
        "3. The appeals are disposed of.",
    )
    seg = segment(text)
    assert seg.parts == 2 and numbers(seg, 2) == ["1", "2", "3"]


def test_chain_may_start_late_when_the_first_numbers_are_lost() -> None:
    text = body(*(f"{n}. Paragraph {n} of the judgment." for n in range(6, 20)))
    seg = segment(text)
    assert numbers(seg) == [str(n) for n in range(6, 20)]
