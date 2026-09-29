"""Answer metrics (PLAN 5.4) on the 2.4 mock run with mock gold; graders via FakeProvider."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from juris.config import REPO_ROOT, ModelSpec
from juris.db.models import LlmCall
from juris.eval.juris_eval import JurisEvalItem
from juris.eval.metrics import (
    Answer,
    ClaimEvidencePair,
    Grader,
    authority_recall,
    citation_faithfulness,
    citation_validity,
    cost_latency,
    key_point_coverage,
    quality_rubric,
    score_answers,
    unsupported_claim_rate,
)
from juris.eval.metrics.claims import final_statements, uncited_sentences
from juris.eval.metrics.common import section_id, split_sentences
from juris.eval.metrics.cost import CallLike
from juris.eval.metrics.graders import (
    ASSERTIONS_PROMPT,
    KEY_POINTS_PROMPT,
    RUBRIC_PROMPT,
    GraderOutputError,
    KeyPointGrade,
    RubricGrade,
    render_analysis,
)
from juris.events.fold import CaseView
from juris.llm.cost import CallRecord
from juris.llm.gateway import LLMGateway
from juris.llm.providers.fake import FakeProvider
from juris.models import Argument, Citation, Claim, CreatedBy, StatuteMeta
from juris.models.common import EntailmentLabel, Stage
from juris.prompts import load_prompt
from juris.verify.quote import normalise_quote, quote_in_text

RUNS = REPO_ROOT / "fixtures" / "runs"
GOLD = Path(__file__).parent / "golden" / "metrics" / "coercion_gold.yaml"
GRADER_MODEL = ModelSpec(provider="fake", name="grader-model", effort=None)


def load_view(name: str) -> CaseView:
    return CaseView.model_validate_json((RUNS / f"{name}.caseview.json").read_text("utf-8"))


@pytest.fixture
def view() -> CaseView:
    return load_view("coercion_full")


@pytest.fixture
def item() -> JurisEvalItem:
    return JurisEvalItem.model_validate(yaml.safe_load(GOLD.read_text("utf-8")))


def key_points_json(verdicts: Sequence[str]) -> str:
    return json.dumps(
        {
            "judgements": [
                {"index": i, "verdict": v, "reason": "Stated in the summary."}
                for i, v in enumerate(verdicts, 1)
            ]
        }
    )


RUBRIC = json.dumps(
    {
        c: {"score": s, "reason": "As the rubric describes."}
        for c, s in [
            ("correctness", 4),
            ("completeness", 4),
            ("reasoning_consistency", 5),
            ("uncertainty_calibration", 4),
        ]
    }
)
# The coercion summary's uncited sentences: framing, an open fact, a report on the record.
ASSERTIONS = json.dumps(
    {
        "sentences": [
            {"index": 1, "legal_assertion": False},
            {"index": 2, "legal_assertion": True},
            {"index": 3, "legal_assertion": False},
        ]
    }
)


def grader(responses: dict[str, list[str]]) -> tuple[Grader, FakeProvider]:
    fake = FakeProvider(responses)
    gateway = LLMGateway({"fake": fake}, prices={}, cache=None, cache_mode="off")
    return Grader(gateway, GRADER_MODEL), fake


# ---- the quote rule and sentence splitting -------------------------------------------------


def test_quote_normalisation() -> None:
    text = "The Court said: “consent  was\ncaused by coercion” and set it aside."
    assert quote_in_text('"consent was caused by coercion"', text)
    assert quote_in_text("  consent was\tcaused ", text)
    assert not quote_in_text("consent was obtained by coercion", text)
    assert not quote_in_text("Consent was caused", text)
    assert quote_in_text("Consent was caused", text, casefold=True)
    assert not quote_in_text("   ", text)
    left, right = chr(0x2018), chr(0x2019)
    assert normalise_quote(f"It{right}s  {left}so{right}") == "It's 'so'"


def test_split_sentences_keeps_legal_abbreviations() -> None:
    text = (
        "In Ramesh v. Suresh the Court read s. 15 widely. Section 19 then applies [E-003]. "
        'A. K. Sen J. agreed. It said "no bar." Delay alone is not enough.'
    )
    assert split_sentences(text) == [
        "In Ramesh v. Suresh the Court read s. 15 widely.",
        "Section 19 then applies [E-003].",
        "A. K. Sen J. agreed.",
        'It said "no bar."',
        "Delay alone is not enough.",
    ]


# ---- deterministic metrics on the mock run -------------------------------------------------


def test_citation_validity(view: CaseView) -> None:
    result = citation_validity(view)
    # 15 citations; C-008's quote from E-018 is not in the chunk (the fixture's invalid one).
    assert result.values == {
        "citation_validity": pytest.approx(14 / 15),
        "citation_validity.existence": 1.0,
        "citation_validity.analysis_refs": 1.0,
    }
    assert [(i["claim_id"], i["evidence_id"], i["reason"]) for i in result.details["invalid"]] == [
        ("C-008", "E-018", "quote not in the chunk")
    ]


def test_citation_validity_catches_unknown_evidence(view: CaseView) -> None:
    broken = view.model_copy(deep=True)
    broken.evidence.pop("E-006")
    result = citation_validity(broken)
    assert result.values["citation_validity"] == pytest.approx(13 / 15)
    assert result.values["citation_validity.existence"] == pytest.approx(14 / 15)


async def test_citation_faithfulness_from_the_record(view: CaseView) -> None:
    result = await citation_faithfulness(view)
    assert result.details["outcomes"] == {
        "does_not_support": 1,
        "invalid": 1,
        "partially_supports": 3,
        "supports": 10,
    }
    assert result.value == pytest.approx(10 / 15)
    assert result.values["citation_faithfulness.lenient"] == pytest.approx(13 / 15)


def with_unverified_citation(view: CaseView) -> CaseView:
    """Adds C-011, citing E-005 with a real quote, that no gate has verified."""
    out = view.model_copy(deep=True)
    by = CreatedBy(stage=Stage.S5, agent="counsel_a")
    out.claims["C-011"] = Claim(
        id="C-011",
        text="A new claim.",
        position_id="I-1.A",
        author_agent="counsel_a",
        created_by=by,
    )
    quote = out.chunks[out.evidence["E-005"].chunk_id].text[:40]
    out.arguments["A-009"] = Argument(
        id="A-009",
        claim_ids=["C-011"],
        reasoning="Because.",
        citations=[
            Citation(claim_id="C-011", evidence_id="E-005", pinpoint="para 12", quote=quote)
        ],
        created_by=by,
    )
    return out


async def test_citation_faithfulness_judges_unverified_pairs(view: CaseView) -> None:
    extended = with_unverified_citation(view)
    unjudged = await citation_faithfulness(extended)
    assert unjudged.details["outcomes"]["unjudged"] == 1
    assert unjudged.value == pytest.approx(10 / 15)  # the unjudged pair is left out

    seen: list[ClaimEvidencePair] = []

    async def judge(pairs: Sequence[ClaimEvidencePair]) -> list[EntailmentLabel]:
        seen.extend(pairs)
        return [EntailmentLabel.SUPPORTS] * len(pairs)

    judged = await citation_faithfulness(extended, judge)
    assert [(p.claim_id, p.evidence_id, p.claim) for p in seen] == [
        ("C-011", "E-005", "A new claim.")
    ]
    assert seen[0].passage.startswith(seen[0].quote)
    assert judged.value == pytest.approx(11 / 16)


def test_final_statements(view: CaseView) -> None:
    assert view.analysis is not None
    statements = final_statements(view.analysis)
    assert [s.where for s in statements] == [
        "I-1.A",
        "I-1.B",
        "I-2.A",
        "I-2.B",
        *(f"summary {n}" for n in range(1, 6)),
    ]
    assert statements[5].refs == ("E-001", "E-016", "E-007", "E-014")
    assert [s.where for s in uncited_sentences(view.analysis)] == [
        "summary 1",
        "summary 3",
        "summary 5",
    ]


def test_unsupported_claim_rate(view: CaseView) -> None:
    # Unclassified: the 3 uncited summary sentences all count, and are unsupported.
    plain = unsupported_claim_rate(view)
    assert plain.value == pytest.approx(3 / 9)
    # C-002 (unsupported) and C-008 (invalid quote) have no verified support in the record.
    assert plain.values["unsupported_claim_rate.record"] == pytest.approx(2 / 10)
    # Classified: only the open-fact sentence is a legal assertion.
    classified = unsupported_claim_rate(view, [False, True, False])
    assert classified.value == pytest.approx(1 / 7)
    assert classified.values["unsupported_claim_rate.uncited"] == pytest.approx(1 / 7)
    assert classified.details["not_legal_assertions"] == ["summary 1", "summary 5"]
    with pytest.raises(ValueError, match="uncited sentences"):
        unsupported_claim_rate(view, [True])


def test_unsupported_claim_rate_counts_unverified_citations(view: CaseView) -> None:
    # E-006 was cited only for C-002, which the verifier found unsupported. Summary 4 is
    # made to cite nothing else.
    assert view.analysis is not None
    summary = view.analysis.overall_summary
    for old, new in (("[E-003]", ""), ("[E-010, E-017]", ""), ("[E-018]", "[E-006]")):
        summary = summary.replace(old, new)
    changed = view.model_copy(deep=True)
    changed.analysis = view.analysis.model_copy(update={"overall_summary": summary})
    result = unsupported_claim_rate(changed, [False, True, False])
    assert [u["where"] for u in result.details["unsupported"]] == ["summary 3", "summary 4"]
    assert result.value == pytest.approx(2 / 7)


def test_authority_recall(view: CaseView, item: JurisEvalItem) -> None:
    result = authority_recall(view, item)
    assert result.values == {
        "authority_recall": pytest.approx(4 / 6),
        "authority_recall.supporting": pytest.approx(3 / 4),
        "authority_recall.contrary": pytest.approx(1 / 2),
        "authority_recall.sections": pytest.approx(2 / 3),
    }
    assert result.details == {
        "missing_supporting": ["MOCK-SC-1978-02"],
        "missing_contrary": ["MOCK-SC-2016-06"],
        "missing_sections": ["contract_act:14"],
    }
    no_contrary = item.model_copy(update={"gold_contrary_authorities": []})
    assert authority_recall(view, no_contrary).values["authority_recall.contrary"] is None


def test_section_ids_from_act_names() -> None:
    # The retrieval service fills StatuteMeta.act with the Act's name; the mock uses its ID.
    assert section_id(StatuteMeta(act="Indian Contract Act", section="15")) == "contract_act:15"
    assert section_id(StatuteMeta(act="contract_act", section="74")) == "contract_act:74"
    assert section_id(StatuteMeta(act="Sale of Goods Act", section="16")) == (
        "sale_of_goods_act:16"
    )
    assert section_id(StatuteMeta(act="Unknown Act", section="1")) == "Unknown Act:1"


def test_cost_latency_from_totals(view: CaseView) -> None:
    result = cost_latency(view)
    assert result.values == {
        "cost_latency.llm_calls": 19,
        "cost_latency.input_tokens": 405_800,
        "cost_latency.output_tokens": 37_000,
        "cost_latency.total_tokens": 442_800,
        "cost_latency.usd": pytest.approx(2.3632),
        "cost_latency.wall_seconds": 495.0,  # 10:00:00 -> 10:08:15
    }
    assert result.value is None  # no single headline value


def test_cost_latency_from_calls(view: CaseView) -> None:
    at = datetime(2026, 9, 29, tzinfo=UTC)
    calls: list[CallLike] = [
        CallRecord(
            provider="fake",
            model="m",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.5,
            cached=False,
        ),
        CallRecord(
            provider="fake",
            model="m",
            input_tokens=100,
            output_tokens=10,
            cost_usd=0.0,
            cached=True,
        ),
        LlmCall(
            provider="fake",
            model="m",
            input_tokens=50,
            output_tokens=5,
            cost_usd=None,
            cached=False,
            tags={},
            at=at,
        ),
    ]
    result = cost_latency(view, calls, wall_seconds=12.5)
    assert result.values["cost_latency.llm_calls"] == 3
    assert result.values["cost_latency.input_tokens"] == 150
    assert result.values["cost_latency.usd"] == pytest.approx(0.5)
    assert result.values["cost_latency.wall_seconds"] == 12.5
    assert result.details == {"unpriced_calls": 1, "source": "calls"}


# ---- graders -------------------------------------------------------------------------------


def test_grader_prompts_are_versioned_and_fill() -> None:
    for name in (KEY_POINTS_PROMPT, RUBRIC_PROMPT, ASSERTIONS_PROMPT):
        template = load_prompt(name)
        assert template.id == name and template.version == 1


async def test_graders_with_fake_provider(view: CaseView, item: JurisEvalItem) -> None:
    g, fake = grader(
        {
            KEY_POINTS_PROMPT: [key_points_json(["covered"] * 3 + ["partially", "covered"])],
            RUBRIC_PROMPT: [RUBRIC],
            ASSERTIONS_PROMPT: [ASSERTIONS],
        }
    )
    sentences = [s.text for s in uncited_sentences(view.analysis)]  # type: ignore[arg-type]
    key_points, rubric, legal = await g.grade(item, view, sentences)

    coverage = key_point_coverage(item, key_points)
    assert coverage.value == pytest.approx(4.5 / 5)
    assert coverage.values["key_point_coverage.full"] == pytest.approx(4 / 5)
    assert coverage.details["judgements"][3]["verdict"] == "partially"
    quality = quality_rubric(rubric)
    assert quality.value == pytest.approx(4.25)
    assert quality.values["quality_rubric.reasoning_consistency"] == 5.0
    assert legal == [False, True, False]

    prompts = {c.tags["agent"]: c for c in fake.calls}
    assert set(prompts) == {KEY_POINTS_PROMPT, RUBRIC_PROMPT, ASSERTIONS_PROMPT}
    for call in fake.calls:
        assert call.model == GRADER_MODEL and call.tags["item"] == "JE-900"
        assert "$" not in call.messages[0].content  # every field was filled
    key_prompt = prompts[KEY_POINTS_PROMPT].messages[0].content
    assert "5. Whether the Rs 48 lakh was admitted" in key_prompt
    # Authorities are named by title for the grader, not by document ID.
    assert "[MOCK] Northstar Logistics v. Gupta Warehousing" in key_prompt
    assert "Weightier: [MOCK] Metro Rail Corp." in render_analysis(view)
    assert "[MOCK] Horizon Power Ltd." in prompts[RUBRIC_PROMPT].messages[0].content  # gold
    assert (
        "3. The claim that a blacklisting threat" in prompts[ASSERTIONS_PROMPT].messages[0].content
    )


async def test_grader_rejects_wrong_key_point_numbers(view: CaseView, item: JurisEvalItem) -> None:
    g, _ = grader({KEY_POINTS_PROMPT: [key_points_json(["covered"] * 4)]})
    with pytest.raises(GraderOutputError, match="key points 1-5"):
        await g.key_points(item, view)


def test_grades_are_validated() -> None:
    bad = json.loads(RUBRIC)
    bad["correctness"]["score"] = 6
    with pytest.raises(ValueError):
        RubricGrade.model_validate(bad)
    with pytest.raises(ValueError):
        KeyPointGrade.model_validate({"judgements": [{"index": 1, "verdict": "mostly"}]})


# ---- the suite -----------------------------------------------------------------------------


async def test_suite_on_the_mock_runs(view: CaseView, item: JurisEvalItem) -> None:
    failed = load_view("failure_case")
    assert failed.analysis is None
    g, fake = grader(
        {
            KEY_POINTS_PROMPT: [key_points_json(["covered"] * 3 + ["partially", "covered"])],
            RUBRIC_PROMPT: [RUBRIC],
            ASSERTIONS_PROMPT: [ASSERTIONS],
        }
    )
    result = await score_answers(
        [Answer(item, view), Answer(item.model_copy(update={"id": "JE-901"}), failed)], grader=g
    )
    assert len(fake.calls) == 3  # the failed run has no analysis to grade

    ok, bad = (i.values for i in result.items)
    assert ok["answered"] == 1.0 and bad["answered"] == 0.0
    assert ok["unsupported_claim_rate"] == pytest.approx(1 / 7)
    assert ok["key_point_coverage"] == pytest.approx(0.9) and bad["key_point_coverage"] == 0.0
    assert ok["quality_rubric"] == pytest.approx(4.25) and bad["quality_rubric"] is None
    assert bad["authority_recall"] == 0.0

    agg = result.aggregate()
    coverage = agg["key_point_coverage"]
    assert coverage is not None and coverage.n == 2 and coverage.mean == pytest.approx(0.45)
    assert coverage.lo == pytest.approx(0.0) and coverage.hi == pytest.approx(0.9)
    rubric = agg["quality_rubric"]
    assert rubric is not None and rubric.n == 1 and rubric.mean == pytest.approx(4.25)
    table = result.table()
    assert "| key_point_coverage | 0.450 | 0.000-0.900 | 2 |" in table
    json.dumps([i.to_json() for i in result.items])  # serialisable


async def test_suite_without_grader(view: CaseView, item: JurisEvalItem) -> None:
    result = await score_answers([Answer(item, view)])
    values = result.items[0].values
    assert values["key_point_coverage"] is None and values["quality_rubric"] is None
    assert values["unsupported_claim_rate"] == pytest.approx(3 / 9)  # sentences unclassified
    assert values["citation_validity"] == pytest.approx(14 / 15)
    assert "| quality_rubric | n/a | n/a | 0 |" in result.table()
