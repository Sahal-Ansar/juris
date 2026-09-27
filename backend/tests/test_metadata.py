"""Metadata normalisation and merge (PLAN 3.4). Inputs are values seen in the AWS and
KanoonGPT metadata or on judgments' first pages; unknowns must come back as None."""

import datetime as dt

import pytest

from juris.ingest.metadata import (
    bench_from_text,
    canonical_court,
    clean_judge,
    merge,
    normalise_disposition,
    parse_date,
    parse_sc_card,
    split_judges,
    split_title,
)
from juris.models import CourtLevel


@pytest.mark.parametrize(
    ("name", "canonical", "level"),
    [
        ("Supreme Court of India", "Supreme Court of India", CourtLevel.SUPREME_COURT),
        ("High Court of Delhi", "High Court of Delhi", CourtLevel.HIGH_COURT),
        ("Bombay High Court", "High Court of Bombay", CourtLevel.HIGH_COURT),
        ("27~1", "High Court of Bombay", CourtLevel.HIGH_COURT),
        (
            "High Court of Punjab and Haryana",
            "High Court of Punjab and Haryana",
            CourtLevel.HIGH_COURT,
        ),
        ("Madras High Court", "High Court of Madras", CourtLevel.HIGH_COURT),
        ("NCLAT", "National Company Law Appellate Tribunal", CourtLevel.TRIBUNAL),
        ("Privy Council", "Judicial Committee of the Privy Council", CourtLevel.OTHER),
    ],
)
def test_canonical_court(name: str, canonical: str, level: CourtLevel) -> None:
    assert canonical_court(name) == (canonical, level)


@pytest.mark.parametrize("name", [None, "", "Delhi Development Authority", "Some Registry"])
def test_unknown_court_is_none(name: str | None) -> None:
    assert canonical_court(name) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("26-02-2015", dt.date(2015, 2, 26)),
        ("2019-11-19 00:00:00", dt.date(2019, 11, 19)),
        ("2020-07-09", dt.date(2020, 7, 9)),
        ("MAY 6, 2008", dt.date(2008, 5, 6)),
        ("August 22, 1973", dt.date(1973, 8, 22)),
        (dt.datetime(2019, 11, 19), dt.date(2019, 11, 19)),
        ("31-02-2015", None),
        ("", None),
        (None, None),
        ("sometime in 2015", None),
    ],
)
def test_parse_date(value: object, expected: dt.date | None) -> None:
    assert parse_date(value) == expected


@pytest.mark.parametrize(
    ("raw", "clean"),
    [
        ("HON'BLE MR. JUSTICE VALMIKI J. MEHTA", "VALMIKI J. MEHTA"),
        ("HON'BLE SHRI JUSTICE A. K. MENON", "A.K. MENON"),
        ("HON'BLE MS. JUSTICE MUKTA GUPTA", "MUKTA GUPTA"),
        ("SHRI JUSTICE J P DEVADHAR", "J P DEVADHAR"),
        ("DIPAK MISRA *", "DIPAK MISRA"),
        ("S. R. DAS C.J.", "S.R. DAS"),
        ("THE CHIEF JUSTICE -", None),
    ],
)
def test_clean_judge(raw: str, clean: str | None) -> None:
    assert clean_judge(raw) == clean


def test_split_judges() -> None:
    assert split_judges(
        "HON'BLE SHRI JUSTICE M. S. SONAK,HON'BLE SHRI JUSTICE PRITHVIRAJ K. CHAVAN"
    ) == [
        "M.S. SONAK",
        "PRITHVIRAJ K. CHAVAN",
    ]
    assert split_judges("K. RAMASWAMY, B.L. HANSARIA") == ["K. RAMASWAMY", "B.L. HANSARIA"]
    assert split_judges(None) == []


@pytest.mark.parametrize(
    ("text", "bench"),
    [
        ("v.\nX\nOCTOBER 23, 1996\n[K. RAMASWAMY AND G.B. PATTANAIK, JJ.]\nHeadnote", 2),
        ("DECEMBER 15,2016\n(R. K. AGRAWAL AND R. BANUMATHI, JJ.)\nCode of Civil Procedure", 2),
        (
            "[MEHR CHAND MAHAJAN C.J., MUKHERJEA, VIVIAN\nBOSE, BHAGWATI and VENKATARAMA AYYAR "
            "JJ.]",
            5,
        ),
        ("[P. B. GAJENDRAGADKAR, C. J., K. N. WANCHOO AND J. C. SHAH, JJ.]", 3),
        ("[S. R. DAS, BHAGWATI and JAGANNADHADAS n.·1\nMaster and Servant", 3),
    ],
)
def test_bench_from_text(text: str, bench: int) -> None:
    judges = bench_from_text(text)
    assert judges is not None and len(judges) == bench


@pytest.mark.parametrize(
    "text",
    [
        "[Per Gupta and Tulzapurkar, JJ.]\nThe appeal",  # a per-opinion note, not the bench
        "(Respondent-In-person)\nThe appeal",
        "IN THE HIGH COURT OF DELHI AT NEW DELHI\n+ RFA 765/2015",
    ],
)
def test_no_bench_line(text: str) -> None:
    assert bench_from_text(text) is None


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        (
            "RFA/436/2016 of DR KIMTI LAL Vs HARPAL SINGH & ANR",
            ("RFA/436/2016", "DR KIMTI LAL", "HARPAL SINGH & ANR"),
        ),
        (
            "CS(COMM)/1066/2018 of SANJAY GUPTA Vs BANKOF MAHARASHTRA",
            ("CS(COMM)/1066/2018", "SANJAY GUPTA", "BANKOF MAHARASHTRA"),
        ),
        (
            "K.P. MANU versus CHAIRMAN, SCRUTINY COMMITTEE",
            (None, "K.P. MANU", "CHAIRMAN, SCRUTINY COMMITTEE"),
        ),
        ("IN RE: THE SPECIAL COURTS BILL, 1978", (None, None, None)),
    ],
)
def test_split_title(title: str, expected: tuple[str | None, ...]) -> None:
    assert split_title(title) == expected


@pytest.mark.parametrize(
    ("raw", "label"),
    [
        ("Appeal(s) allowed", "allowed"),
        ("Case Partly allowed", "partly_allowed"),
        ("Dismissed", "dismissed"),
        ("Disposed off", "disposed"),
        ("DISPOSED OFF", "disposed"),
        ("Reference answered", "referred"),
        ("Matter referred to larger bench", "referred"),
        ("Remitted to Lower Court", "remanded"),
        ("Directions issued", "directions"),
        ("Leave Granted & Allowed", "allowed"),
        ("Disposed off/Dismissed for default", "dismissed"),
        ("JUDGEMENT", None),
        ("Hearing Adjourned", None),
        ("", None),
    ],
)
def test_normalise_disposition(raw: str, label: str | None) -> None:
    assert normalise_disposition(raw) == label


# The AWS card for [2015] 3 S.C.R. 243, verbatim (headnote excerpt shortened).
SC_CARD = (
    "<font size='4'> <strong>K.P. MANU<span class='fst-italic'> versus </span>CHAIRMAN, SCRUTINY "
    "COMMITTEE </strong>- <span class='escrText'>[2015] 3 S.C.R. 243</span><span "
    "class='ncDisplay'>2015 INSC 163</span><input type='hidden' id='cnr' value=ESCR010003942015>"
    '</button></font><br><strong>Coram : DIPAK MISRA<sup style="color: #268e97;font-size: 22px;'
    'top: 0;" class="tooltip-sup" data-tooltip="Author">*</sup>, V. GOPALA GOWDA</strong>'
    "<br> C &middot;Kera/a (Scheduled Castes and Scheduled Tribes) ... - Held: Not sustainable"
    "<br><strong class='caseDetailsTD' ><span style='color:#212F3D' > Decision Date :</span>"
    "<font color='green'> 26-02-2015</font><span style='color:#212F3D' > | Case No :</span>"
    "<font color='green'> CIVIL APPEAL No. 7065/2008</font><span style='color:#212F3D' > | "
    "Disposal Nature :</span><font color='green'> Appeal(s) allowed</font>   <span "
    "style='color:#212F3D' > |  Bench :</span><font color='green'> 2 Judges</font></strong>"
)
AWS_SC = {
    "title": "K.P. MANU versus CHAIRMAN, SCRUTINY COMMITTEE",
    "petitioner": "K.P. MANU",
    "respondent": "CHAIRMAN, SCRUTINY COMMITTEE",
    "judge": "DIPAK MISRA",
    "citation": "[2015] 3 S.C.R. 243",
    "case_id": "2015 INSC 163",
    "cnr": "ESCR010003942015",
    "decision_date": "26-02-2015",
    "disposal_nature": "Appeal(s) allowed",
    "court": "Supreme Court of India",
    "raw_html": SC_CARD,
}


def test_parse_sc_card() -> None:
    card = parse_sc_card(SC_CARD)
    assert card == {
        "coram": ["DIPAK MISRA", "V. GOPALA GOWDA"],
        "author": "DIPAK MISRA",
        "case_number": "CIVIL APPEAL No. 7065/2008",
        "bench_size": 2,
    }


def test_merge_aws_sc_only() -> None:
    meta = merge("SC-2015_3_243_286", AWS_SC, "aws_sc")
    assert meta.court == "Supreme Court of India" and meta.court_level == CourtLevel.SUPREME_COURT
    assert meta.judges == ["DIPAK MISRA", "V. GOPALA GOWDA"] and meta.bench_strength == 2
    assert meta.author == "DIPAK MISRA"
    assert meta.citations == ["[2015] 3 SCR 243", "2015 INSC 163"]
    assert meta.neutral_citation == "2015 INSC 163"
    assert meta.decision_date == dt.date(2015, 2, 26)
    assert meta.disposition == "allowed"
    assert meta.headnote is None and meta.kanoongpt_match == "none"
    assert meta.sources["judges"] == "aws_sc" and meta.conflicts == []


def test_merge_with_kanoongpt_fills_gaps_and_records_conflicts() -> None:
    kg = {
        "cnr_number": "ESCR010003942015",
        "case_title": "K.P. MANU versus CHAIRMAN, SCRUTINY COMMITTEE",
        "party_petitioner": "K.P. MANU",
        "party_respondent": "CHAIRMAN, SCRUTINY COMMITTEE",
        "coram_members": "['DIPAK MISRA', 'V. GOPALA GOWDA', 'N.V. RAMANA']",
        "bench_name": "3 Judges",
        "decision_date": "2015-02-26",
        "court_name": "Supreme Court of India",
        "disposition_text": "Appeal(s) allowed",
        "law_report_citation": "[2015] 3 S.C.R. 243",
        "neutral_citation": "2015INSC163",
        "headnote_text": "Kerala Act, 1996 - Grant of caste certificate ...",
    }
    meta = merge("SC-2015_3_243_286", AWS_SC, "aws_sc", kg)
    assert meta.kanoongpt_match == "cnr"
    assert meta.headnote and meta.sources["headnote"] == "kanoongpt"
    assert meta.judges == ["DIPAK MISRA", "V. GOPALA GOWDA"]  # AWS keeps its value ...
    assert any(c.startswith("judges:") for c in meta.conflicts)  # ... and the clash is recorded
    assert meta.citations == ["[2015] 3 SCR 243", "2015 INSC 163"]  # no duplicates


def test_kanoongpt_row_for_another_case_is_rejected() -> None:
    kg = {
        "cnr_number": "ESCR010003942015",
        "case_title": "STATE OF BIHAR versus RAM LAL",
        "decision_date": "1999-01-01",
        "headnote_text": "unrelated",
    }
    meta = merge("SC-2015_3_243_286", AWS_SC, "aws_sc", kg)
    assert meta.kanoongpt_match == "rejected" and meta.headnote is None


def test_merge_aws_hc_leaves_unknowns_empty() -> None:
    aws = {
        "title": "RFA/765/2015 of UNION OF INDIA Vs G SINGH & ANR",
        "judge": "HON'BLE MS. JUSTICE PRATHIBA M. SINGH",
        "cnr": "DLHC015044742015",
        "decision_date": "2018-07-04 00:00:00",
        "disposal_nature": "",
        "court": "High Court of Delhi",
    }
    meta = merge("HC-DLHC015044742015_1_2018-07-04", aws, "aws_hc")
    assert (meta.case_number, meta.petitioner, meta.respondent) == (
        "RFA/765/2015",
        "UNION OF INDIA",
        "G SINGH & ANR",
    )
    assert meta.judges == ["PRATHIBA M. SINGH"] and meta.bench_strength == 1
    assert meta.court_level == CourtLevel.HIGH_COURT
    assert meta.disposition is None and meta.citations == []
    assert meta.author == "PRATHIBA M. SINGH"  # a single-judge bench's judgment is hers


def test_bench_strength_is_the_majority_count_or_none() -> None:
    kg = {
        "cnr_number": "ESCR010003942015",
        "case_title": AWS_SC["title"],
        "decision_date": "2015-02-26",
        "coram_members": "['DIPAK MISRA', 'V. GOPALA GOWDA', 'N.V. RAMANA']",
        "bench_name": "3 Judges",
    }
    # coram 2 and card bench 2 (AWS) vs KanoonGPT coram 3 and bench 3: no majority
    meta = merge("SC-2015_3_243_286", AWS_SC, "aws_sc", kg)
    assert meta.bench_strength is None
    assert any(c.startswith("bench_strength:") for c in meta.conflicts)
    # add the text's bench line (2 judges): 3 of 5 sources say 2
    text = "[DIPAK MISRA AND V. GOPALA GOWDA, JJ.]"
    meta = merge("SC-2015_3_243_286", AWS_SC, "aws_sc", kg, text)
    assert meta.bench_strength == 2 and meta.bench_counts["text"] == 2


def test_text_supplies_bench_and_header_citations() -> None:
    aws = dict(AWS_SC, raw_html=None, citation=None, case_id=None)
    text = "[2023] 12 S.C.R. 979 : 2023 INSC 736\nX\nv.\nY\n[S.B. SINHA AND P.P. NAOLEKAR, JJ.]\n"
    meta = merge("SC-2023_12_979_1033", aws, "aws_sc", text=text)
    assert meta.citations == ["[2023] 12 SCR 979", "2023 INSC 736"]
    assert meta.judges == ["S.B. SINHA", "P.P. NAOLEKAR"] and meta.sources["judges"] == "text"
    assert meta.bench_strength == 2


def test_to_document() -> None:
    meta = merge("SC-2015_3_243_286", AWS_SC, "aws_sc")
    doc = meta.to_document("snap-1", "CC-BY-4.0", "https://example.org/x.pdf")
    assert doc.id == "SC-2015_3_243_286" and doc.bench_strength == 2
    assert doc.citations == ["[2015] 3 SCR 243", "2015 INSC 163"]
    assert doc.court_level == CourtLevel.SUPREME_COURT and doc.cnr == "ESCR010003942015"
