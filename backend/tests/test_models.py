import copy
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from juris.config import Settings, load_profile, resolve_run_config
from juris.models import (
    DISCLAIMER,
    Argument,
    CaseAnalysis,
    Chunk,
    Citation,
    Claim,
    Counterargument,
    CreatedBy,
    Document,
    Evidence,
    EvidenceRef,
    GitState,
    IdAllocator,
    Issue,
    IssueAnalysis,
    JuryBallot,
    Justification,
    KeyConflict,
    Objection,
    Position,
    PositionAnalysis,
    PrecedentCard,
    RubricScores,
    Run,
    RunTotals,
    SourceRef,
    StatuteMeta,
    TreatmentSignal,
    VerificationResult,
)

BY = {"stage": "S2", "agent": "counsel_a"}
VERIFIED = {
    "status": "verified",
    "check": "entailment",
    "label": "supports",
    "justification": "The quoted passage states the rule directly.",
    "verifier_model": "claude-opus-5-5",
    "created_by": {"stage": "S4", "agent": "verifier"},
}
CITATION = {
    "claim_id": "C-001",
    "evidence_id": "E-001",
    "pinpoint": "para 12",
    "quote": "compensation for any loss or damage caused",
    "verification": VERIFIED,
}
RUBRIC = {
    "evidence_strength": 4,
    "precedent_fit": 3,
    "counterargument_handling": 4,
    "unresolved_risk": 2,
}
ISSUE_ANALYSIS: dict[str, Any] = {
    "issue_id": "I-1",
    "issue_text": "Is the liquidated damages clause enforceable as a penalty?",
    "positions": [
        {
            "position_id": "I-1.A",
            "summary": "Only reasonable compensation up to the stipulated sum is recoverable.",
            "supporting_authorities": ["SC-2015-KAILASH-NATH"],
            "key_evidence": [{"evidence_id": "E-001", "pinpoint": "para 43"}],
        },
        {"position_id": "I-1.B", "summary": "The stipulated sum is a genuine pre-estimate."},
    ],
    "key_conflicts": [
        {
            "authority_a": "SC-2015-KAILASH-NATH",
            "authority_b": "SC-1963-FATEH-CHAND",
            "distinction": "Proof of actual loss where loss is ascertainable.",
            "weightier": "SC-2015-KAILASH-NATH",
            "reason": "Later judgment applying the earlier ratio.",
        }
    ],
    "leaning": "leaning_A",
    "confidence": "moderate",
    "confidence_reason": "Two consistent SC authorities; facts partly assumed.",
    "unresolved_questions": ["Was actual loss pleaded?"],
}


def run_example() -> dict[str, Any]:
    config = resolve_run_config(Settings(_env_file=None), load_profile("b0"))
    return {
        "run_id": "20260926T000000Z-b0-abc123",
        "status": "completed",
        "profile": "b0",
        "config": config.model_dump(mode="json"),
        "git": {"commit": "a" * 40, "dirty": False},
        "models": {"judge": {"provider": "anthropic", "name": "claude-opus-5-5"}},
        "prompt_hashes": {"judge@v1": "f" * 64},
        "corpus_snapshot_id": "snap-2026-09",
        "embedding_model": "bge-m3",
        "seeds": {"profile": 0},
        "started_at": "2026-09-26T00:00:00Z",
        "ended_at": "2026-09-26T00:05:00Z",
        "totals": {"llm_calls": 3, "input_tokens": 100, "output_tokens": 50, "usd": 0.0014},
    }


EXAMPLES: dict[type[BaseModel], dict[str, Any]] = {
    CreatedBy: BY,
    Document: {
        "id": "SC-2023-INSC-1043",
        "kind": "judgment",
        "title": "Eldeco Housing v. Ashok Vidyarthi",
        "court": "Supreme Court of India",
        "court_level": "supreme_court",
        "bench_strength": 2,
        "judges": ["Vikram Nath"],
        "decision_date": "2023-11-30",
        "cnr": "ESCR010008242023",
        "citations": ["2023 INSC 1043", "[2023] 16 S.C.R. 872"],
        "source_url": "https://indian-supreme-court-judgments.s3.amazonaws.com/x.pdf",
        "licence": "CC-BY-4.0",
        "corpus_snapshot": "snap-2026-09",
    },
    StatuteMeta: {
        "act": "specific_relief_act",
        "section": "10",
        "title": "Specific performance in respect of contracts",
        "in_force_from": "2018-10-01",
        "amended_by": ["Act 18 of 2018"],
        "amendments_curated": True,
    },
    Chunk: {
        "id": "SC-2023-INSC-1043#p12-14",
        "document_id": "SC-2023-INSC-1043",
        "text": "12. The respondent sought compensation for any loss or damage caused ...",
        "para_start": 12,
        "para_end": 14,
        "char_start": 1040,
        "char_end": 2310,
    },
    Evidence: {
        "id": "E-001",
        "chunk_id": "SC-2023-INSC-1043#p12-14",
        "document_id": "SC-2023-INSC-1043",
        "issue_ids": ["I-1"],
        "retrieval_score": 0.61,
        "rerank_score": 0.83,
        "retrieved_by_query": "liquidated damages penalty section 74",
        "registered_at_stage": "S1",
        "created_by": {"stage": "S1", "agent": "researcher"},
    },
    Position: {"id": "I-1.A", "statement": "The clause is a penalty."},
    Issue: {
        "id": "I-1",
        "question": "Is the liquidated damages clause enforceable?",
        "positions": [
            {"id": "I-1.A", "statement": "The clause is a penalty."},
            {"id": "I-1.B", "statement": "The clause is a genuine pre-estimate."},
        ],
        "created_by": {"stage": "S0", "agent": "issue_framer"},
    },
    VerificationResult: VERIFIED,
    Citation: CITATION,
    Claim: {
        "id": "C-001",
        "text": "Section 74 caps recovery at reasonable compensation.",
        "position_id": "I-1.A",
        "author_agent": "counsel_a",
        "status": "verified",
        "created_by": BY,
    },
    Argument: {
        "id": "A-001",
        "claim_ids": ["C-001"],
        "reasoning": "S.74 read with the cited ratio limits recovery.",
        "citations": [CITATION],
        "created_by": BY,
    },
    Counterargument: {
        "id": "R-001",
        "targets": "C-001",
        "argument_id": "A-002",
        "created_by": {"stage": "S5", "agent": "counsel_b"},
    },
    TreatmentSignal: {
        "kind": "followed",
        "by_document_id": "SC-2019-X",
        "cue_text": "followed in",
    },
    PrecedentCard: {
        "document_id": "SC-2015-KAILASH-NATH",
        "principle": "Compensation under s.74 requires loss unless loss is hard to prove.",
        "material_facts": "Forfeiture of earnest money after a failed auction.",
        "distinguishing_factors": ["public auction"],
        "treatment": [{"kind": "followed", "by_document_id": "SC-2019-X", "cue_text": "followed"}],
        "treatment_verified": True,
        "fit_by_argument": {"A-001": "directly_supports"},
        "created_by": {"stage": "S3", "agent": "precedent_analyst"},
    },
    Objection: {
        "id": "O-001",
        "type": "distinguishable_precedent",
        "target_id": "C-001",
        "raised_by": "cross_examiner",
        "text": "The cited case concerned a public auction, not a private sale.",
        "response": "The ratio on s.74 is general.",
        "resolved": True,
        "created_by": {"stage": "S6", "agent": "cross_examiner"},
    },
    RubricScores: RUBRIC,
    Justification: {"text": "Strong verified support.", "cited_ids": ["E-001", "C-001"]},
    JuryBallot: {
        "juror_id": "J-1",
        "per_position_scores": {"I-1.A": RUBRIC, "I-1.B": {**RUBRIC, "evidence_strength": 2}},
        "justifications": [{"text": "A is better supported.", "cited_ids": ["E-001"]}],
        "created_by": {"stage": "S8", "agent": "juror_1"},
    },
    EvidenceRef: {"evidence_id": "E-001", "pinpoint": "para 43"},
    PositionAnalysis: ISSUE_ANALYSIS["positions"][0],
    KeyConflict: ISSUE_ANALYSIS["key_conflicts"][0],
    IssueAnalysis: ISSUE_ANALYSIS,
    SourceRef: {
        "document_id": "SC-2015-KAILASH-NATH",
        "court": "Supreme Court of India",
        "date": "2015-01-20",
        "citation": "(2015) 4 SCC 136",
        "pinpoints_used": ["para 43"],
    },
    CaseAnalysis: {
        "question": "Can the seller forfeit the full earnest money?",
        "facts": ["Buyer defaulted on the balance payment."],
        "assumptions": ["No actual loss was pleaded."],
        "issues": [ISSUE_ANALYSIS],
        "overall_summary": "Recovery is likely limited to reasonable compensation [E-001].",
        "limitations": ["Treatment of SC-1963-FATEH-CHAND not verified."],
        "sources": [{"document_id": "SC-2015-KAILASH-NATH"}],
        "created_by": {"stage": "S9", "agent": "judge"},
    },
    GitState: {"commit": "a" * 40, "dirty": False},
    RunTotals: {"llm_calls": 1},
}

# One enum-typed field per model (dotted path into the example), for the bad-enum test.
ENUM_FIELDS: dict[type[BaseModel], str] = {
    CreatedBy: "stage",
    Document: "kind",
    Evidence: "registered_at_stage",
    Issue: "created_by.stage",
    VerificationResult: "status",
    Citation: "verification.check",
    Claim: "status",
    TreatmentSignal: "kind",
    PrecedentCard: "fit_by_argument.A-001",
    Objection: "type",
    IssueAnalysis: "leaning",
    CaseAnalysis: "issues.0.confidence",
    Run: "status",
}


def _set(data: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    data = copy.deepcopy(data)
    target: Any = data
    *parents, last = path.split(".")
    for key in parents:
        target = target[int(key)] if isinstance(target, list) else target[key]
    target[last] = value
    return data


def example(model: type[BaseModel]) -> dict[str, Any]:
    return run_example() if model is Run else copy.deepcopy(EXAMPLES[model])


ALL: list[type[BaseModel]] = [*EXAMPLES, Run]


@pytest.mark.parametrize("model", ALL, ids=lambda m: m.__name__)
def test_valid_example(model: type[BaseModel]) -> None:
    instance = model.model_validate(example(model))
    # Round-trips through JSON unchanged.
    assert model.model_validate_json(instance.model_dump_json()) == instance


@pytest.mark.parametrize("model", ALL, ids=lambda m: m.__name__)
def test_extra_field_rejected(model: type[BaseModel]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate({**example(model), "surprise": 1})


@pytest.mark.parametrize(
    "model",
    [m for m in ALL if any(f.is_required() for f in m.model_fields.values())],
    ids=lambda m: m.__name__,
)
def test_missing_required_field_rejected(model: type[BaseModel]) -> None:
    required = next(name for name, f in model.model_fields.items() if f.is_required())
    data = example(model)
    del data[required]
    with pytest.raises(ValidationError, match="Field required"):
        model.model_validate(data)


@pytest.mark.parametrize("model", list(ENUM_FIELDS), ids=lambda m: m.__name__)
def test_bad_enum_rejected(model: type[BaseModel]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(_set(example(model), ENUM_FIELDS[model], "not_a_valid_value"))


def test_models_are_immutable() -> None:
    claim = Claim.model_validate(EXAMPLES[Claim])
    with pytest.raises(ValidationError):
        claim.status = "falls"  # type: ignore[misc, assignment]
    assert claim.model_copy(update={"status": "falls"}).status == "falls"


@pytest.mark.parametrize(
    ("model", "path", "value"),
    [
        (Evidence, "id", "E-1"),  # IDs need 3+ digits
        (Claim, "position_id", "I-1.D"),  # at most 3 positions
        (Issue, "positions.1.id", "I-2.B"),  # positions must belong to the issue
        (Chunk, "char_end", 10),  # char_end before char_start
        (Chunk, "para_end", None),  # para range must be complete
        (StatuteMeta, "in_force_to", "2000-01-01"),  # before in_force_from
        (VerificationResult, "label", None),  # entailment check needs a label
        (VerificationResult, "check", "quote"),  # a passed quote check can't be 'verified'
        (Objection, "target_id", "nonsense"),  # must target a record ID
        (RubricScores, "precedent_fit", 6),  # scores are 1-5
        (JuryBallot, "justifications", []),  # jurors must justify
        (KeyConflict, "weightier", "SC-OTHER"),  # must be one of the two
        (IssueAnalysis, "positions.1.position_id", "I-2.B"),
        (CaseAnalysis, "disclaimer", "Not legal advice."),
        (CaseAnalysis, "overall_summary", "word " * 251),
    ],
)
def test_model_rules(model: type[BaseModel], path: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(_set(example(model), path, value))


def test_chunk_accepts_page_pinpoints_for_old_judgments() -> None:
    data = {k: v for k, v in EXAMPLES[Chunk].items() if k not in ("para_start", "para_end")}
    chunk = Chunk.model_validate({**data, "page_start": 358, "page_end": 359})
    assert chunk.para_start is None and chunk.page_start == 358
    with pytest.raises(ValidationError, match="paragraph range, a page range"):
        Chunk.model_validate(data)


def test_unknown_treatment_defaults_to_unverified() -> None:
    card = {
        k: v
        for k, v in EXAMPLES[PrecedentCard].items()
        if k not in ("treatment", "treatment_verified")
    }
    parsed = PrecedentCard.model_validate(card)
    assert parsed.treatment == [] and parsed.treatment_verified is False


def test_case_analysis_disclaimer_default() -> None:
    assert CaseAnalysis.model_validate(EXAMPLES[CaseAnalysis]).disclaimer == DISCLAIMER


def test_id_allocator() -> None:
    ids = IdAllocator()
    assert [ids.next("E"), ids.next("E"), ids.next("C")] == ["E-001", "E-002", "C-001"]
    issue = ids.issue()
    assert issue == "I-1"
    assert [IdAllocator.position(issue, i) for i in range(3)] == ["I-1.A", "I-1.B", "I-1.C"]
    assert ids.juror() == "J-1"
    with pytest.raises(ValueError):
        ids.next("X")
    with pytest.raises(ValueError):
        IdAllocator.position(issue, 3)
    for _ in range(998):
        last = ids.next("E")
    assert last == "E-1000"
    Evidence.model_validate({**EXAMPLES[Evidence], "id": last})


def test_every_model_exports_a_json_schema() -> None:
    for model in ALL:
        schema = model.model_json_schema()
        assert schema["type"] == "object"
