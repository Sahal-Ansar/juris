"""The Citation Verifier (PLAN 6.1): deterministic checks, label mapping, events, the judge.

No live model and no database: the judge runs on a FakeProvider, events on the in-memory store.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from juris.config import ModelSpec
from juris.eval.metrics import citation_faithfulness
from juris.events import CaseView, EventEmitter, InMemoryEventStore, fold
from juris.events import catalog as c
from juris.llm.errors import StructuredOutputError
from juris.llm.gateway import LLMGateway
from juris.llm.providers.fake import FakeProvider
from juris.models import Argument, Chunk, Citation, Claim, CreatedBy, Document, Evidence
from juris.models.common import (
    ClaimStatus,
    EntailmentLabel,
    Stage,
    VerificationCheck,
    VerificationStatus,
)
from juris.pipeline.evidence_registry import EvidenceRegistry
from juris.prompts import load_prompt
from juris.retrieval.service import Hit
from juris.verify import (
    CitationVerifier,
    ClaimEvidencePair,
    Judgement,
    LLMEntailmentJudge,
    VerifierOutputError,
    ViewEvidence,
    check_citation,
    claim_statuses,
)

MODEL = ModelSpec(provider="fake", name="verifier-model", effort=None)
BY = CreatedBy(stage=Stage.S3, agent="counsel_a")
TEXT = "The Court held: “consent  was caused by\ncoercion” and the contract is voidable."


@dataclass
class Texts:
    """An in-memory ``EvidenceTexts``."""

    chunks: dict[str, str | None] = field(default_factory=lambda: {"E-001": TEXT, "E-002": None})

    def __contains__(self, evidence_id: str, /) -> bool:
        return evidence_id in self.chunks

    def chunk_text(self, evidence_id: str) -> str | None:
        return self.chunks.get(evidence_id)


def cite(claim_id: str = "C-001", evidence_id: str = "E-001", quote: str = "consent") -> Citation:
    return Citation(claim_id=claim_id, evidence_id=evidence_id, pinpoint="para 1", quote=quote)


def claim(claim_id: str, status: ClaimStatus = ClaimStatus.PROPOSED) -> Claim:
    return Claim(
        id=claim_id,
        text=f"Claim {claim_id}.",
        position_id="I-1.A",
        author_agent="counsel_a",
        status=status,
        created_by=BY,
    )


class FakeJudge:
    """Returns the labels it is given in order, recording the pairs it was asked about."""

    def __init__(self, *labels: EntailmentLabel) -> None:
        self.labels = list(labels)
        self.seen: list[ClaimEvidencePair] = []

    async def judge(self, pairs: Sequence[ClaimEvidencePair]) -> list[Judgement]:
        self.seen.extend(pairs)
        return [Judgement(self.labels.pop(0), f"because {p.claim_id}") for p in pairs]


# ---- checks 1-2 ----------------------------------------------------------------------------


def test_check_citation_existence_and_quote() -> None:
    texts = Texts()
    assert check_citation(texts, cite(evidence_id="E-404")) == (
        VerificationCheck.EXISTENCE,
        "unknown evidence",
    )
    assert check_citation(texts, cite(evidence_id="E-002")) == (
        VerificationCheck.QUOTE,
        "chunk not in the record",
    )
    assert check_citation(texts, cite(quote="consent was obtained by coercion")) == (
        VerificationCheck.QUOTE,
        "quote not in the chunk",
    )
    assert check_citation(texts, cite(quote="consent was caused by coercion")) is None


def test_check_citation_forgives_typography_and_whitespace_only() -> None:
    texts = Texts()
    assert check_citation(texts, cite(quote='"consent was   caused by coercion"')) is None
    assert check_citation(texts, cite(quote="The  Court\theld:")) is None


def test_check_citation_casefold_is_a_parameter() -> None:
    texts = Texts()
    shouting = cite(quote="THE COURT HELD")
    assert check_citation(texts, shouting) is not None
    assert check_citation(texts, shouting, casefold=True) is None


async def test_invalid_citations_never_reach_the_judge() -> None:
    judge = FakeJudge(EntailmentLabel.SUPPORTS)
    verifier = CitationVerifier(Texts(), judge, verifier_model="m")
    citations = [
        cite(evidence_id="E-404"),
        cite(evidence_id="E-002"),
        cite(quote="a different word"),
        cite(),
    ]
    results = await verifier.verify({"C-001": claim("C-001")}, citations, stage=Stage.S4)
    assert [(r.status, r.check) for r in results] == [
        (VerificationStatus.INVALID, VerificationCheck.EXISTENCE),
        (VerificationStatus.INVALID, VerificationCheck.QUOTE),
        (VerificationStatus.INVALID, VerificationCheck.QUOTE),
        (VerificationStatus.VERIFIED, VerificationCheck.ENTAILMENT),
    ]
    assert [r.justification for r in results[:3]] == [
        "unknown evidence",
        "chunk not in the record",
        "quote not in the chunk",
    ]
    assert [r.verifier_model for r in results] == [None, None, None, "m"]
    assert [p.quote for p in judge.seen] == ["consent"]


async def test_casefold_flows_through_the_verifier() -> None:
    citations = [cite(quote="THE COURT HELD")]
    claims = {"C-001": claim("C-001")}
    strict = await CitationVerifier(Texts(), None).verify(claims, citations, stage=Stage.S4)
    assert strict[0].status is VerificationStatus.INVALID
    folded = CitationVerifier(Texts(), FakeJudge(EntailmentLabel.SUPPORTS), casefold=True)
    assert (await folded.verify(claims, citations, stage=Stage.S4))[0].status is (
        VerificationStatus.VERIFIED
    )


# ---- entailment mapping --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "status"),
    [
        (EntailmentLabel.SUPPORTS, VerificationStatus.VERIFIED),
        (EntailmentLabel.PARTIALLY_SUPPORTS, VerificationStatus.WEAK),
        (EntailmentLabel.DOES_NOT_SUPPORT, VerificationStatus.UNSUPPORTED),
        (EntailmentLabel.CONTRADICTS, VerificationStatus.UNSUPPORTED),
    ],
)
async def test_label_maps_to_status(label: EntailmentLabel, status: VerificationStatus) -> None:
    judge = FakeJudge(label)
    verifier = CitationVerifier(Texts(), judge, verifier_model="verifier-model")
    [result] = await verifier.verify({"C-001": claim("C-001")}, [cite()], stage=Stage.S7)
    assert result.status is status
    assert result.check is VerificationCheck.ENTAILMENT
    assert result.label is label
    assert result.justification == "because C-001"
    assert result.verifier_model == "verifier-model"
    assert result.created_by == CreatedBy(stage=Stage.S7, agent="citation_verifier")
    assert judge.seen == [ClaimEvidencePair("C-001", "Claim C-001.", "E-001", "consent", TEXT)]


async def test_a_judge_is_required_for_entailment() -> None:
    with pytest.raises(ValueError, match="no judge"):
        await CitationVerifier(Texts(), None).verify(
            {"C-001": claim("C-001")}, [cite()], stage=Stage.S4
        )


async def test_unknown_claim_is_an_error() -> None:
    with pytest.raises(ValueError, match="C-009"):
        await CitationVerifier(Texts(), FakeJudge()).verify(
            {"C-001": claim("C-001")}, [cite("C-009")], stage=Stage.S4
        )


def test_claim_statuses_take_the_best_citation() -> None:
    from juris.models import VerificationResult

    def result(status: VerificationStatus) -> VerificationResult:
        if status is VerificationStatus.INVALID:
            return VerificationResult(
                status=status,
                check=VerificationCheck.QUOTE,
                justification="x",
                created_by=BY,
            )
        label = {
            VerificationStatus.VERIFIED: EntailmentLabel.SUPPORTS,
            VerificationStatus.WEAK: EntailmentLabel.PARTIALLY_SUPPORTS,
            VerificationStatus.UNSUPPORTED: EntailmentLabel.DOES_NOT_SUPPORT,
        }[status]
        return VerificationResult(
            status=status,
            check=VerificationCheck.ENTAILMENT,
            label=label,
            justification="x",
            created_by=BY,
        )

    statuses = [
        VerificationStatus.INVALID,
        VerificationStatus.VERIFIED,
        VerificationStatus.WEAK,
        VerificationStatus.INVALID,
        VerificationStatus.UNSUPPORTED,
        VerificationStatus.WEAK,
        VerificationStatus.INVALID,
    ]
    claims = ["C-001", "C-001", "C-002", "C-002", "C-003", "C-003", "C-004"]
    assert claim_statuses([cite(cid) for cid in claims], [result(s) for s in statuses]) == {
        "C-001": ClaimStatus.VERIFIED,
        "C-002": ClaimStatus.WEAK,
        "C-003": ClaimStatus.WEAK,
        "C-004": ClaimStatus.UNSUPPORTED,
    }


# ---- events --------------------------------------------------------------------------------


class Corpus:
    def records(self, chunk_id: str) -> tuple[Document, Chunk]:
        document = Document(
            id="D-1", kind="judgment", title="A v. B", licence="CC-BY-4.0", corpus_snapshot="s"
        )
        chunk = Chunk(
            id=chunk_id, document_id="D-1", text=TEXT, para_start=1, para_end=1,
            char_start=0, char_end=len(TEXT),
        )  # fmt: skip
        return document, chunk


def hit(chunk_id: str) -> Hit:
    return Hit(
        chunk_id=chunk_id, doc_id="D-1", rank=1, text=TEXT, context_header=None,
        title="A v. B", query="q", retrieval_score=0.5, rerank_score=0.9,
    )  # fmt: skip


def run_with_claims(*claims: Claim) -> tuple[InMemoryEventStore, EventEmitter, EvidenceRegistry]:
    store = InMemoryEventStore()
    emitter = EventEmitter(store, run_id="run-1", case_id="case-1")
    emitter.emit(c.CaseCreatedPayload(question="q", profile="juris_full", corpus_snapshot_id="s"))
    registry = EvidenceRegistry(Corpus(), emitter)
    registry.register(hit("D-1#p1"), ["I-1"], Stage.S1)
    for each in claims:
        emitter.emit(c.ClaimCreatedPayload(claim=each), stage=Stage.S3, agent="counsel_a")
    return store, emitter, registry


async def test_events_and_fold() -> None:
    claims = {
        "C-001": claim("C-001"),
        "C-002": claim("C-002"),
        "C-003": claim("C-003", ClaimStatus.VERIFIED),  # already right: no event
        "C-004": claim("C-004", ClaimStatus.CONTESTED),  # a later stage's: left alone
    }
    store, emitter, registry = run_with_claims(*claims.values())
    citations = [
        cite("C-002", quote="not in the chunk"),
        cite("C-001"),
        cite("C-003"),
        cite("C-004"),
        cite("C-001", quote="the contract is voidable"),
    ]
    judge = FakeJudge(
        EntailmentLabel.SUPPORTS,
        EntailmentLabel.SUPPORTS,
        EntailmentLabel.CONTRADICTS,
        EntailmentLabel.PARTIALLY_SUPPORTS,
    )
    verifier = CitationVerifier(registry, judge, emitter, verifier_model="verifier-model")
    results = await verifier.verify(claims, citations, stage=Stage.S4)

    after = store.read("run-1", after_seq=store.last_seq("run-1") - 7)
    events = [e for e in store.read("run-1") if e.agent == "citation_verifier"]
    assert [e.type for e in events] == ["verification_result"] * 5 + ["claim_status_changed"] * 2
    assert after  # sanity: the run has more than the verifier's events
    assert [(e.payload.claim_id, e.payload.evidence_id) for e in events[:5]] == [  # type: ignore[union-attr]
        (cit.claim_id, cit.evidence_id) for cit in citations
    ]
    assert [e.payload.result for e in events[:5]] == results  # type: ignore[union-attr]
    changes = [(e.payload.claim_id, e.payload.status) for e in events[5:]]  # type: ignore[union-attr]
    assert changes == [("C-001", ClaimStatus.VERIFIED), ("C-002", ClaimStatus.UNSUPPORTED)]
    assert all(e.stage is Stage.S4 for e in events)

    view = fold(store.read("run-1"))
    assert [(v.claim_id, v.result) for v in view.verifications] == list(
        zip([cit.claim_id for cit in citations], results, strict=True)
    )
    assert {k: v.status for k, v in view.claims.items()} == {
        "C-001": ClaimStatus.VERIFIED,
        "C-002": ClaimStatus.UNSUPPORTED,
        "C-003": ClaimStatus.VERIFIED,
        "C-004": ClaimStatus.CONTESTED,
    }


async def test_no_emitter_no_events() -> None:
    results = await CitationVerifier(Texts(), FakeJudge(EntailmentLabel.SUPPORTS)).verify(
        {"C-001": claim("C-001")}, [cite()], stage=Stage.S4
    )
    assert len(results) == 1


async def test_a_folded_run_is_evidence_too() -> None:
    store, emitter, registry = run_with_claims(claim("C-001"))
    view = fold(store.read("run-1"))
    evidence = ViewEvidence(view)
    assert "E-001" in evidence
    assert "E-009" not in evidence
    assert evidence.chunk_text("E-001") == TEXT
    assert check_citation(evidence, cite(quote="consent was caused by coercion")) is None
    assert check_citation(evidence, cite(evidence_id="E-009")) is not None
    assert isinstance(view, CaseView)
    assert emitter.run_id == "run-1"
    assert registry.chunk_text("E-001") == TEXT


def test_registry_chunk_and_chunk_text() -> None:
    _, _, registry = run_with_claims()
    assert registry.chunk("E-001").id == "D-1#p1"
    assert registry.chunk("E-001").text == TEXT
    assert registry.chunk_text("E-001") == TEXT
    assert registry.chunk_text("E-404") is None
    with pytest.raises(KeyError):
        registry.chunk("E-404")
    assert "E-001" in registry


# ---- the LLM judge -------------------------------------------------------------------------


def verdicts(*labels: str, start: int = 1) -> str:
    return json.dumps(
        {
            "judgements": [
                {"index": n, "label": label, "justification": f"reason {label}"}
                for n, label in enumerate(labels, start)
            ]
        }
    )


def single(label: str) -> str:
    return json.dumps({"label": label, "justification": f"reason {label}"})


def pair(n: int, passage: str = "The passage.") -> ClaimEvidencePair:
    return ClaimEvidencePair(f"C-{n:03d}", f"Claim {n}.", f"E-{n:03d}", f"quote {n}", passage)


def llm_judge(
    responses: list[str | Exception], **options: int
) -> tuple[LLMEntailmentJudge, FakeProvider]:
    fake = FakeProvider({"citation_verifier": list(responses)})
    gateway = LLMGateway({"fake": fake}, prices={}, cache=None, cache_mode="off")
    return LLMEntailmentJudge(gateway, MODEL, **options), fake


async def test_batches_run_in_order_and_map_back() -> None:
    judge, fake = llm_judge(
        [
            verdicts("supports", "contradicts"),
            verdicts("partially_supports", "does_not_support"),
            single("supports"),  # the odd last pair goes alone
        ],
        batch_size=2,
    )
    judgements = await judge.judge([pair(n) for n in range(1, 6)])
    assert fake.calls[2].messages[-1].content.count("<pair number=") == 1
    assert len(fake.calls) == 3
    assert [j.label for j in judgements] == [
        EntailmentLabel.SUPPORTS,
        EntailmentLabel.CONTRADICTS,
        EntailmentLabel.PARTIALLY_SUPPORTS,
        EntailmentLabel.DOES_NOT_SUPPORT,
        EntailmentLabel.SUPPORTS,
    ]
    assert judgements[1].justification == "reason contradicts"


async def test_batch_output_is_matched_by_index_not_position() -> None:
    reordered = json.dumps(
        {
            "judgements": [
                {"index": 2, "label": "contradicts", "justification": "b"},
                {"index": 1, "label": "supports", "justification": "a"},
            ]
        }
    )
    judge, _ = llm_judge([reordered], batch_size=2)
    assert await judge(  # __call__: labels only
        [pair(1), pair(2)]
    ) == [EntailmentLabel.SUPPORTS, EntailmentLabel.CONTRADICTS]


async def test_identical_pairs_are_judged_once() -> None:
    judge, fake = llm_judge([verdicts("supports", "does_not_support")], batch_size=2)
    twin = ClaimEvidencePair("C-099", "Claim 1.", "E-001", "quote 1", "The passage.")
    judgements = await judge.judge([pair(1), pair(2), twin, pair(1)])
    assert len(fake.calls) == 1
    assert fake.calls[0].messages[-1].content.count("<pair number=") == 2
    assert [j.label for j in judgements] == [
        EntailmentLabel.SUPPORTS,
        EntailmentLabel.DOES_NOT_SUPPORT,
        EntailmentLabel.SUPPORTS,
        EntailmentLabel.SUPPORTS,
    ]


@pytest.mark.parametrize(
    "bad",
    [
        verdicts("supports"),  # a pair missing
        verdicts("supports", "supports", start=2),  # numbers 2, 3
        json.dumps(
            {
                "judgements": [
                    {"index": 1, "label": "supports", "justification": "a"},
                    {"index": 1, "label": "supports", "justification": "b"},
                ]
            }
        ),  # a repeated number
    ],
)
async def test_a_bad_batch_falls_back_to_single_pair_calls(bad: str) -> None:
    judge, fake = llm_judge([bad, single("contradicts"), single("supports")], batch_size=2)
    judgements = await judge.judge([pair(1), pair(2)])
    assert len(fake.calls) == 3
    assert [j.label for j in judgements] == [EntailmentLabel.CONTRADICTS, EntailmentLabel.SUPPORTS]
    assert fake.calls[1].messages[-1].content.count("<pair number=") == 1


async def test_unparseable_output_also_falls_back() -> None:
    # the gateway's repair retry makes a second call before it gives up
    judge, fake = llm_judge(
        ["nope", "nope", single("contradicts"), single("supports")], batch_size=2
    )
    judgements = await judge.judge([pair(1), pair(2)])
    assert len(fake.calls) == 4
    assert [j.label for j in judgements] == [EntailmentLabel.CONTRADICTS, EntailmentLabel.SUPPORTS]


async def test_a_failing_single_call_raises() -> None:
    judge, _ = llm_judge([verdicts("supports"), "nope", "nope"], batch_size=2)
    with pytest.raises(VerifierOutputError, match="C-001"):
        await judge.judge([pair(1), pair(2)])


async def test_a_single_pair_call_sends_the_single_schema() -> None:
    judge, fake = llm_judge([single("supports")])
    assert await judge([pair(1)]) == [EntailmentLabel.SUPPORTS]
    schema = fake.calls[0].json_schema
    assert schema is not None
    assert set(schema["properties"]) == {"label", "justification"}


async def test_a_batch_call_sends_the_batch_schema() -> None:
    judge, fake = llm_judge([verdicts("supports", "supports")], batch_size=2)
    await judge([pair(1), pair(2)])
    schema = fake.calls[0].json_schema
    assert schema is not None
    assert set(schema["properties"]) == {"judgements"}


async def test_an_empty_batch_list_is_a_bad_batch() -> None:
    judge, fake = llm_judge(
        ['{"judgements": []}', '{"judgements": []}', single("supports"), single("supports")],
        batch_size=2,
    )
    assert await judge([pair(1), pair(2)]) == [EntailmentLabel.SUPPORTS] * 2
    assert len(fake.calls) == 4


def test_gateway_failure_type_is_what_the_judge_catches() -> None:
    assert issubclass(StructuredOutputError, Exception)


async def test_the_prompt_carries_claim_quote_and_passage() -> None:
    judge, fake = llm_judge([single("supports")])
    await judge.judge(
        [ClaimEvidencePair("C-001", "A $5 claim.", "E-001", "the quote", "Context $here.")]
    )
    request = fake.calls[0]
    body = request.messages[-1].content
    assert "A $5 claim." in body
    assert "<quote>\nthe quote\n</quote>" in body
    assert "Context $here." in body
    assert request.temperature == 0.0
    template = load_prompt("verifier_entailment")
    assert request.tags["prompt"] == f"verifier_entailment@v{template.version}"
    assert request.tags["agent"] == "citation_verifier"


async def test_long_passages_are_cut_around_the_quote() -> None:
    passage = "a" * 500 + " THE QUOTE " + "b" * 500
    judge, _ = llm_judge([], max_passage_chars=100)
    shown = judge.render([ClaimEvidencePair("C-001", "x", "E-001", "THE QUOTE", passage)])
    assert "THE QUOTE" in shown
    assert shown.count("a") + shown.count("b") < 120
    assert "[...]" in shown


async def test_the_llm_judge_plugs_into_the_verifier_and_the_metric() -> None:
    store, emitter, registry = run_with_claims(claim("C-001"))
    judge, _ = llm_judge([single("partially_supports")])
    verifier = CitationVerifier(registry, judge, emitter, verifier_model=MODEL.name)
    citations = [cite("C-001", quote="the contract is voidable")]
    [result] = await verifier.verify({"C-001": claim("C-001")}, citations, stage=Stage.S4)
    assert result.status is VerificationStatus.WEAK
    assert result.verifier_model == "verifier-model"

    # the same judge scores pairs the run never verified
    view = fold(store.read("run-1"))
    view.verifications.clear()
    view.arguments["A-001"] = Argument(
        id="A-001", claim_ids=["C-001"], reasoning="Because.", citations=citations, created_by=BY
    )
    assert isinstance(view.evidence["E-001"], Evidence)
    judge2, _ = llm_judge([single("supports")])
    metric = await citation_faithfulness(view, judge2)
    assert metric.value == 1.0
