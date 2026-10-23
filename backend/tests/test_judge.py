"""The Judge (S9) and final verification (S10; PLAN 7.11) on scripted outputs: the analysis
covers every issue and cites only citable evidence, unverified treatment is listed, and S10
removes the Judge's citations that do not hold. No model, no database."""

import copy
import json
from typing import Any

from juris.agents.judge import NOT_ANALYSED, JudgeOutput, answers, normalise_refs
from juris.llm.types import ProviderRequest, ProviderResponse
from juris.models.common import Stage
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.final_verify import FinalVerificationStage, drop_ref, sentences
from juris.pipeline.stages.judge import JudgeStage
from tests.test_b1 import ACT, FATEH, SATYABRATA
from tests.test_b2 import EXTRA
from tests.test_jury import setup as jury_setup
from tests.test_rebuttal import Script, at

JUDGE = "judge"
WRONG = "The seller cannot simply forfeit"


class SentenceJudge:
    """The verifier at S10: a sentence containing ``WRONG`` is not supported."""

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        body = request.messages[0].content
        claim = body.split("<claim>\n", 1)[1].split("\n</claim>", 1)[0]
        label = "does_not_support" if WRONG in claim else "supports"
        text = json.dumps({"label": label, "justification": "Scripted."})
        return ProviderResponse(
            text=text, input_tokens=10, output_tokens=10, stop_reason="end_turn"
        )


OUT: dict[str, Any] = {
    "issues": [
        {
            "issue_id": "I-1",
            "positions": [
                {
                    "position_id": "I-1.B",
                    "summary": "A reasonable sum may be kept [E-003].",
                    "key_evidence": ["E-003"],
                },
                {
                    "position_id": "I-1.A",
                    "summary": "Compensation is capped [E-001]. Loss must be proved [E-2].",
                    "key_evidence": ["E-1"],
                },
            ],
            "key_conflicts": [
                {
                    "evidence_a": "E-002",
                    "evidence_b": "E-003",
                    "distinction": "Fateh Chand is about penalties.",
                    "weightier": "a",
                    "reason": "Larger bench.",
                }
            ],
            "leaning": "leaning_A",
            "confidence": "moderate",
            "confidence_reason": "B's main claim is only weakly supported.",
            "unresolved": [{"question": "u1", "answer": "It does not hold."}],
        }
    ],
    "overall_summary": f"{WRONG} [E-001]. Some forfeiture is allowed [E-003].",
    "limitations": ["The facts are thin.", " "],
}


async def setup(*judge: Any) -> tuple[StageContext, Script]:
    """S0-S8 done (no jury ballots), one unresolved question (U1) on I-1; E-004 is registered
    but never cited, so not citable."""
    ctx, provider = await jury_setup({("judge", JUDGE): list(judge)})
    ctx.record.register_evidence(EXTRA, ["I-1"], stage=Stage.S5, agent="researcher")
    return at(ctx, Stage.S9), provider


async def test_the_judge_writes_the_analysis_and_s10_checks_its_citations() -> None:
    ctx, provider = await setup(OUT)
    await JudgeStage().run(ctx)
    analysis = ctx.record.view.analysis
    assert analysis is not None and analysis.created_by.stage is Stage.S9
    [issue] = analysis.issues
    a, b = issue.positions  # in position order
    assert [(r.evidence_id, r.pinpoint) for r in a.key_evidence] == [("E-001", "s. 74")]
    assert a.supporting_authorities == [FATEH]  # E-001 is a statute; E-2 is Fateh Chand
    assert b.supporting_authorities == [SATYABRATA]
    conflict = issue.key_conflicts[0]
    assert (conflict.authority_a, conflict.authority_b, conflict.weightier) == (
        FATEH, SATYABRATA, FATEH
    )  # fmt: skip
    assert (issue.leaning.value, issue.confidence.value) == ("leaning_A", "moderate")
    assert issue.unresolved_questions == ["Does O-001 hold? Answer: It does not hold."]
    assert analysis.limitations[0] == "The facts are thin."
    treatment = [x for x in analysis.limitations if x.startswith("Treatment of")]
    assert len(treatment) == 2 and "Fateh Chand v. Balkishan Dass" in treatment[0]
    assert {s.document_id: s.pinpoints_used for s in analysis.sources} == {
        ACT: ["s. 74"],
        SATYABRATA: ["paras 3-5"],
        FATEH: ["para 12"],
    }
    assert ctx.run.values["s9.inline_citations"] == 5 and ctx.run.values["s9.key_evidence"] == 2
    # the evidence graph's checks (PLAN 8.2) are listed as limitations and shown to the Judge
    checks = [
        "I-1 is contested: I-1.A, I-1.B rest on verified evidence.",
        "Claim C-002 still stands, but its support is only weak (partial).",
    ]
    assert analysis.limitations[1:3] == [f"Record check: {c}" for c in checks]
    assert ctx.run.values["s9.record_checks"] == 2
    # what the judge saw: only citable evidence, the record checks, the questions to answer
    body = provider.prompts(JUDGE)[0].messages[0].content
    assert "Record checks:\n" + "\n".join(f"- {c}" for c in checks) + "\n\n" in body
    citable_block = body.split("Citable evidence (cite only these):\n")[1].split("\n\n")[0]
    assert "[E-001] Indian Contract Act, 1872, s. 74:" in citable_block
    assert "[E-004]" not in citable_block
    assert "U1: Does O-001 hold?" in body
    # S10: the first sentence of the overall summary is not supported by E-001
    provider.judge = SentenceJudge()  # type: ignore[assignment]
    await FinalVerificationStage().run(at(ctx, Stage.S10))
    final = ctx.record.view.analysis
    assert final is not None and final.created_by.stage is Stage.S10
    assert final.overall_summary == f"{WRONG}. Some forfeiture is allowed [E-003]."
    assert final.limitations[-1] == (
        "Stated without support: its citations did not check out at the final verification: "
        f'"{WRONG}."'
    )
    assert final.issues[0].positions[0].summary == (
        "Compensation is capped [E-001]. Loss must be proved [E-002]."
    )
    values = ctx.run.values
    assert values["s10.judge_citations"] == 5 and values["s10.judge_unsupported"] == 1
    assert values["s10.judge_not_citable"] == 0 and values["s10.sentences_flagged"] == 1
    assert {s.document_id for s in final.sources} == {ACT, FATEH, SATYABRATA}  # E-001 is key


async def test_the_judge_output_is_checked_against_the_record() -> None:
    bad = copy.deepcopy(OUT)
    issue = bad["issues"][0]
    issue["positions"][1]["key_evidence"] = ["E-001", "E-004", "E-404"]  # E-004 never verified
    issue["key_conflicts"][0]["evidence_b"] = "E-002"  # the same judgment twice
    issue["unresolved"] = []
    bad["overall_summary"] = "word " * 251
    ctx, provider = await setup(bad, OUT)
    await JudgeStage().run(ctx)
    repair = next(r for r in provider.calls if r.tags.get("repair") == "record")
    feedback = repair.messages[2].content
    assert "issues[0].positions[1].key_evidence[1]: 'E-004' is not in the citable evidence" in (
        feedback
    )
    assert "issues[0].key_conflicts[0]: a conflict needs two different judgments" in feedback
    assert "unresolved question U1 has no answer" in feedback
    assert "the overall summary has 251 words (1 to 250)" in feedback
    assert ctx.record.view.analysis is not None  # the repair was clean


async def test_gaps_the_judge_leaves_are_filled_and_listed() -> None:
    empty = {**OUT, "issues": [], "overall_summary": "word " * 300}
    ctx, _ = await setup(empty, empty)
    await JudgeStage().run(ctx)
    analysis = ctx.record.view.analysis
    assert analysis is not None and ctx.run.values["s9.analysis"] == 1
    [issue] = analysis.issues
    assert (issue.leaning.value, issue.confidence.value) == ("insufficient_evidence", "low")
    assert [p.summary for p in issue.positions] == [NOT_ANALYSED, NOT_ANALYSED]
    assert issue.unresolved_questions == ["Does O-001 hold? Answer: not addressed by the Judge."]
    assert len(analysis.overall_summary.split()) == 250
    assert analysis.limitations[1:4] == [
        "The Judge gave no analysis of I-1; it is marked insufficient.",
        "The Judge did not answer unresolved question U1.",
        "The overall summary was cut to 250 words.",
    ]
    failure = ctx.record.view.validation_failures[-1]  # soft problems: nothing dropped
    assert failure.agent == "judge" and failure.dropped is None
    assert "the analysis does not cover I-1" in failure.errors


async def test_claim_ids_and_loosely_filed_answers_are_understood() -> None:
    out = copy.deepcopy(OUT)
    issue = out["issues"][0]
    issue["positions"][0]["key_evidence"] = ["C-2", "A-002"]  # B's claim and its argument
    issue["positions"][1]["key_evidence"] = ["A-404"]  # nothing: dropped after the repair
    issue["unresolved"] = []
    out["issues"].append(
        {**issue, "issue_id": "I-2", "unresolved": [{"question": "U1", "answer": "Yes."}]}
    )  # an issue the case does not have, with the answer in it
    ctx, provider = await setup(out, out)
    await JudgeStage().run(ctx)
    analysis = ctx.record.view.analysis
    assert analysis is not None and len(analysis.issues) == 1
    b = analysis.issues[0].positions[1]
    assert [r.evidence_id for r in b.key_evidence] == ["E-003"]
    assert analysis.issues[0].positions[0].key_evidence == []
    # the answer was filed under an issue the case does not have, which was dropped with it
    assert analysis.issues[0].unresolved_questions == [
        "Does O-001 hold? Answer: not addressed by the Judge."
    ]
    # but an answer filed under another (real) issue counts
    assert answers(JudgeOutput.model_validate(out)) == {"U1": "Yes."}
    feedback = next(r for r in provider.calls if r.tags.get("repair")).messages[2].content
    assert "issues[1]: 'I-2' is not an issue of this case" in feedback
    assert "issues[0].positions[1].key_evidence[0]: 'A-404' is not in the citable evidence" in (
        feedback
    )


async def test_uncitable_inline_citations_are_removed_without_a_call() -> None:
    out = copy.deepcopy(OUT)
    out["overall_summary"] = "Kept as is [E-004]. Fine [E-003]."
    ctx, provider = await setup(out)
    await JudgeStage().run(ctx)
    before = len(provider.calls)
    await FinalVerificationStage().passthrough(at(ctx, Stage.S10))
    assert len(provider.calls) == before  # passthrough: code checks only
    final = ctx.record.view.analysis
    assert final is not None and final.overall_summary == "Kept as is. Fine [E-003]."
    assert ctx.run.values["s10.judge_not_citable"] == 1


def test_sentences_and_citation_removal() -> None:
    assert sentences("One [E-1]. Two? Three!  ") == ["One [E-1].", "Two?", "Three!"]
    assert sentences("Held. [E-1] [E-2]. Next.") == ["Held. [E-1] [E-2].", "Next."]
    assert drop_ref("Held [E-12] and [E-120].", "E-012") == "Held and [E-120]."
    assert drop_ref("Held [ e 7 ].", "E-007") == "Held."


async def test_inline_references_become_evidence_ids() -> None:
    ctx, _ = await setup()
    view = ctx.record.view
    # C-001's argument cites E-001; A-003 cites E-002; [para 4] is not a reference
    text = "Capped [C-1, E-3]. Proved [A-003] . See [para 4]. Nothing [C-404]."
    assert normalise_refs(text, view) == (
        "Capped [E-001] [E-003]. Proved [E-002]. See [para 4]. Nothing."
    )
