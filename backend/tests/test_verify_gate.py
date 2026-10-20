"""Verification gates S4/S7/S10 (PLAN 7.9) on a scripted entailment judge: after S4 every S2
citation has a result; claim statuses follow the best result over all gates. No model, no
database."""

import json

import pytest

from juris.baselines.base import RunContext
from juris.config import ModelsConfig, ModelSpec, PipelineProfile, Settings, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.llm.gateway import LLMGateway
from juris.llm.types import ProviderRequest, ProviderResponse
from juris.models import Citation
from juris.models.common import ClaimStatus, Stage, VerificationStatus
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.verify import VerificationGate, best_status, pending
from tests.test_b1 import HITS, SATYABRATA_QUOTE, STATUTE_QUOTE, FakeService

NOT_A_PENALTY = "It is not a penalty."
FATEH_QUOTE = "The party must prove actual loss in the case of forfeiture of earnest money."
# the fake judge labels a pair by its quote (default: supports)
VERDICTS = {SATYABRATA_QUOTE: "partially_supports", NOT_A_PENALTY: "contradicts"}


class QuoteJudge:
    """Answers the entailment prompt with a label chosen by the pair's quote."""

    name = "fake"

    def __init__(self) -> None:
        self.calls: list[ProviderRequest] = []

    def prompts(self, prompt: str) -> list[ProviderRequest]:
        return [c for c in self.calls if c.tags["prompt"].startswith(prompt + "@")]

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        body = request.messages[0].content
        quote = body.split("<quote>\n", 1)[1].split("\n</quote>", 1)[0]
        label = VERDICTS.get(quote, "supports")
        text = json.dumps({"label": label, "justification": "Scripted."})
        return ProviderResponse(
            text=text, input_tokens=10, output_tokens=10, stop_reason="end_turn"
        )


def cite(claim_id: str, evidence_id: str, quote: str) -> Citation:
    return Citation(claim_id=claim_id, evidence_id=evidence_id, pinpoint="p", quote=quote)


def argue(record: CaseRecord, claim_id: str, *cites: tuple[str, str], stage: Stage) -> None:
    record.add_argument([claim_id], "r", [cite(claim_id, e, q) for e, q in cites], stage=stage,
                        agent="counsel")  # fmt: skip


def context() -> tuple[StageContext, QuoteJudge]:
    provider = QuoteJudge()
    model = ModelSpec(provider="fake", name="judge-model", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    config = resolve_run_config(settings, PipelineProfile(name="t", kind="juris"))
    item = load_split("dev")[0]
    emitter = EventEmitter(InMemoryEventStore(), run_id="r", case_id=item.id)
    emitter.emit(CaseCreatedPayload(question=item.question, profile="t", corpus_snapshot_id="s"))
    record = CaseRecord(emitter, FakeService())
    record.frame_issue("Can the seller forfeit?", ["No.", "Yes."], stage=Stage.S0, agent="f")
    for hit in HITS:  # E-001 statute, E-002 Fateh Chand, E-003 Satyabrata Ghose
        record.register_evidence(hit, ["I-1"], stage=Stage.S1, agent="researcher")
    S2 = Stage.S2
    claims = [("Capped.", "I-1.A"), ("May be kept.", "I-1.B"), ("Never a penalty.", "I-1.B")]
    for text, position in claims:
        record.add_claim(text, position, stage=S2, agent="counsel")
    argue(record, "C-001", ("E-001", STATUTE_QUOTE), ("E-002", FATEH_QUOTE), stage=S2)
    argue(record, "C-002", ("E-003", SATYABRATA_QUOTE), stage=S2)
    argue(record, "C-003", ("E-003", NOT_A_PENALTY), stage=S2)
    run = RunContext(item, config, LLMGateway({"fake": provider}, prices={}), emitter, 0)
    return StageContext(run, record, BudgetManager(config.budget, Spend), Stage.S4, "on"), provider


def at(ctx: StageContext, stage: Stage) -> StageContext:
    return StageContext(ctx.run, ctx.record, ctx.budget, stage, "on")


def statuses(ctx: StageContext) -> dict[str, str]:
    return {c.id: c.status.value for c in ctx.record.view.claims.values()}


async def test_after_s4_every_s2_citation_has_a_result() -> None:
    ctx, provider = context()
    await VerificationGate(Stage.S4).run(ctx)
    view = ctx.record.view
    cited = {(c.claim_id, c.evidence_id) for a in view.arguments.values() for c in a.citations}
    checked = {(v.claim_id, v.evidence_id) for v in view.verifications}
    assert cited == checked and len(view.verifications) == 4
    assert pending(view) == []
    results = {(v.claim_id, v.evidence_id): v.result for v in view.verifications}
    assert results[("C-001", "E-001")].status is VerificationStatus.VERIFIED
    assert results[("C-001", "E-001")].verifier_model == "judge-model"
    assert results[("C-002", "E-003")].status is VerificationStatus.WEAK
    assert results[("C-003", "E-003")].status is VerificationStatus.UNSUPPORTED
    assert statuses(ctx) == {"C-001": "verified", "C-002": "weak", "C-003": "unsupported"}
    assert len(provider.prompts("verifier_entailment")) == 4
    events = [e for e in ctx.run.emitter.events() if e.type == "verification_result"]
    assert {(e.stage, e.agent) for e in events} == {(Stage.S4, "citation_verifier")}
    assert ctx.run.values == {
        "s4.checked": 4.0,
        "s4.verified": 2.0,  # E-001 for C-001 and the default "supports" for E-002
        "s4.weak": 1.0,
        "s4.unsupported": 1.0,
        "s4.invalid": 0.0,
        "s4.claims_changed": 3.0,
    }


async def test_a_later_gate_checks_only_new_citations_and_never_downgrades() -> None:
    ctx, provider = context()
    await VerificationGate(Stage.S4).run(ctx)
    S5 = Stage.S5
    argue(ctx.record, "C-001", ("E-003", SATYABRATA_QUOTE), stage=S5)  # weak, for a verified
    argue(ctx.record, "C-002", ("E-001", STATUTE_QUOTE), stage=S5)  # verified, for a weak one
    before = len(provider.calls)
    s7 = at(ctx, Stage.S7)
    await VerificationGate(Stage.S7).run(s7)
    assert len(provider.calls) - before == 2  # only the two new citations
    assert statuses(ctx) == {"C-001": "verified", "C-002": "verified", "C-003": "unsupported"}
    assert ctx.run.values["s7.checked"] == 2.0 and ctx.run.values["s7.claims_changed"] == 1.0
    changed = [e for e in ctx.run.emitter.events() if e.type == "claim_status_changed"][-1]
    assert changed.stage is Stage.S7
    assert changed.payload.reason == "best of 2 verified citation(s) is verified"


async def test_statuses_owned_by_later_stages_are_left_alone() -> None:
    ctx, _ = context()
    await VerificationGate(Stage.S4).run(ctx)
    ctx.record.set_claim_status("C-003", ClaimStatus.CONTESTED, "objection O-001",
                                stage=Stage.S6, agent="examiner")  # fmt: skip
    argue(ctx.record, "C-003", ("E-001", STATUTE_QUOTE), stage=Stage.S5)
    await VerificationGate(Stage.S7).run(at(ctx, Stage.S7))
    assert statuses(ctx)["C-003"] == "contested"
    assert ctx.record.best_verification("E-001") is VerificationStatus.VERIFIED


async def test_a_gate_with_nothing_new_makes_no_call() -> None:
    ctx, provider = context()
    await VerificationGate(Stage.S4).run(ctx)
    before = len(provider.calls)
    await VerificationGate(Stage.S10).run(at(ctx, Stage.S10))
    assert len(provider.calls) == before
    assert ctx.run.values["s10.checked"] == 0.0


async def test_passthrough_runs_the_code_checks_only() -> None:
    ctx, provider = context()
    # a quote that is not in its chunk (written straight to the store, as counsel never could)
    from juris.events.catalog import ArgumentCreatedPayload
    from juris.models import Argument, CreatedBy

    bad = Citation(claim_id="C-003", evidence_id="E-002", pinpoint="p", quote="no such words")
    ctx.run.emitter.emit(
        ArgumentCreatedPayload(
            argument=Argument(
                id="A-099",
                claim_ids=["C-003"],
                reasoning="r",
                citations=[bad],
                created_by=CreatedBy(stage=Stage.S2, agent="counsel"),
            )
        ),
        stage=Stage.S2,
    )
    await VerificationGate(Stage.S4).passthrough(ctx)
    assert provider.calls == []
    view = ctx.record.view
    assert [(v.claim_id, v.evidence_id, v.result.status.value) for v in view.verifications] == [
        ("C-003", "E-002", "invalid")
    ]
    # C-003's other citation is unchecked, so its status stays as it was
    assert statuses(ctx) == {"C-001": "proposed", "C-002": "proposed", "C-003": "proposed"}
    assert ctx.run.values["s4.invalid"] == 1.0


def test_gates_run_only_at_s4_s7_s10() -> None:
    with pytest.raises(ValueError, match="S4, S7 or S10"):
        VerificationGate(Stage.S2)


def test_best_status() -> None:
    V = VerificationStatus
    assert best_status([V.INVALID, V.WEAK]) is ClaimStatus.WEAK
    assert best_status([V.UNSUPPORTED, V.VERIFIED]) is ClaimStatus.VERIFIED
    assert best_status([V.INVALID]) is ClaimStatus.UNSUPPORTED
