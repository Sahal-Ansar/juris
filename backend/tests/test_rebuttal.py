"""Counsel's rebuttal mode and S5 (PLAN 7.7) on scripted outputs: each side answers the other's
claims, targeted searches register evidence before it is cited, the S7 gate verifies it, and a
second round answers the first. No model, no database."""

import json
from typing import Any

from juris.agents.rebuttal import RebuttalCounsel
from juris.agents.researcher import Researcher
from juris.baselines.base import RunContext
from juris.config import (
    BudgetConfig,
    DeliberationParams,
    ModelsConfig,
    ModelSpec,
    PipelineProfile,
    Settings,
    resolve_run_config,
)
from juris.eval.juris_eval import load_split
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.llm.gateway import LLMGateway
from juris.llm.types import ProviderRequest, ProviderResponse
from juris.models import Citation, CreatedBy, PrecedentCard
from juris.models.common import PrecedentFit, Stage
from juris.pipeline.budget import BudgetManager, ledger_spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.rebuttal import RebuttalStage
from juris.pipeline.stages.verify import VerificationGate
from juris.retrieval.lookup import LinkedCase
from juris.retrieval.service import Hit
from tests.test_b1 import FATEH, HITS, SATYABRATA_QUOTE, STATUTE_QUOTE
from tests.test_b2 import EXTRA, EXTRA_TEXT, MoreService
from tests.test_verify_gate import FATEH_QUOTE, QuoteJudge

SEARCH, REBUT = "counsel_search", "counsel_rebuttal"


class Service(MoreService):
    """B2's fake retrieval (one chunk beyond S1's) with the Researcher's lookups, unused here."""

    def section_hits(self, reference: str) -> list[Hit]:
        return []

    def cases_citing_section(self, act: str, section: str) -> list[LinkedCase]:
        return []


class Script:
    """Counsel replies keyed by (agent, prompt), in order; the verifier labels by quote."""

    name = "fake"

    def __init__(self, replies: dict[tuple[str, str], list[Any]]) -> None:
        self.replies = {k: [json.dumps(r) for r in v] for k, v in replies.items()}
        self.judge = QuoteJudge()
        self.calls: list[ProviderRequest] = []

    def prompts(self, prompt: str) -> list[ProviderRequest]:
        return [c for c in self.calls if c.tags["prompt"].startswith(prompt + "@")]

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.calls.append(request)
        prompt = request.tags["prompt"].split("@")[0]
        if prompt == "verifier_entailment":
            return await self.judge.complete(request)
        queue = self.replies.get((request.tags["agent"], prompt))
        assert queue, f"no reply left for {request.tags['agent']} / {prompt}"
        text = queue.pop(0)
        return ProviderResponse(
            text=text, input_tokens=10, output_tokens=len(text), stop_reason="end_turn"
        )


def counter(target: str, text: str, *cites: tuple[str, str]) -> dict[str, Any]:
    return {
        "target_claim_id": target,
        "text": text,
        "reasoning": "Because.",
        "citations": [{"evidence_id": e, "quote": q} for e, q in cites],
    }


def context(
    replies: dict[tuple[str, str], list[Any]],
    *,
    rounds: int = 1,
    searches: int = 3,
    max_tokens: int | None = None,
) -> tuple[StageContext, Script, Service]:
    """S0-S4 already done: issue I-1 with C-001 (A, cites the statute), C-002 and C-003 (B)."""
    provider = Script(replies)
    model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    params = DeliberationParams(rebuttal_rounds=rounds, extra_searches_per_counsel=searches)
    profile = PipelineProfile(
        name="t", kind="juris", params=params, budget=BudgetConfig(max_tokens_per_case=max_tokens)
    )
    config = resolve_run_config(settings, profile)
    item = load_split("dev")[0]
    emitter = EventEmitter(InMemoryEventStore(), run_id="r", case_id=item.id)
    emitter.emit(CaseCreatedPayload(question=item.question, profile="t", corpus_snapshot_id="s"))
    service = Service(hits=[*HITS, EXTRA])
    record = CaseRecord(emitter, service, params=params)
    record.frame_issue("Can the seller forfeit?", ["No.", "Yes."], stage=Stage.S0, agent="f")
    for hit in HITS:  # E-001 statute, E-002 Fateh Chand, E-003 Satyabrata Ghose
        record.register_evidence(hit, ["I-1"], stage=Stage.S1, agent="researcher")
    S2 = Stage.S2
    for text, position, evidence, quote in (
        ("Compensation is capped.", "I-1.A", "E-001", STATUTE_QUOTE),
        ("A reasonable sum may be kept.", "I-1.B", "E-003", SATYABRATA_QUOTE),
        ("Loss need not be proved.", "I-1.B", "E-002", FATEH_QUOTE),
    ):
        claim = record.add_claim(text, position, stage=S2, agent="counsel")
        cite = Citation(claim_id=claim.id, evidence_id=evidence, pinpoint="p", quote=quote)
        record.add_argument([claim.id], "r", [cite], stage=S2, agent="counsel")
    card = PrecedentCard(
        document_id=FATEH,
        principle="p",
        material_facts="f",
        fit_by_argument={"A-003": PrecedentFit.DISTINGUISHABLE},
        created_by=CreatedBy(stage=Stage.S3, agent="precedent_analyst"),
    )
    record.add_precedent_card(card, stage=Stage.S3, agent="precedent_analyst")
    gateway = LLMGateway({"fake": provider}, prices={})
    run = RunContext(item, config, gateway, emitter, 0)
    budget = BudgetManager(config.budget, ledger_spend(gateway.ledger, {}))
    return StageContext(run, record, budget, Stage.S5, "on"), provider, service


def at(ctx: StageContext, stage: Stage) -> StageContext:
    return StageContext(ctx.run, ctx.record, ctx.budget, stage, "on")


async def verified(ctx: StageContext) -> None:
    """S4 on the opening claims: C-001 verified, C-002 weak, C-003 verified."""
    await VerificationGate(Stage.S4).run(at(ctx, Stage.S4))


def stage(service: Service) -> RebuttalStage:
    researcher = Researcher(service)
    return RebuttalStage(lambda: researcher)


A_SEARCH = {"queries": ["earnest money part of the price"]}
A_REBUT = {"counterarguments": [counter("C-2", "Earnest money is part of the price.",
                                        ("E-004", EXTRA_TEXT))]}  # fmt: skip
B_REBUT = {"counterarguments": [counter("C-001", "Earnest money is no penalty.",
                                        ("E-003", "It is not a penalty."))]}  # fmt: skip


async def test_each_side_answers_the_other_and_s7_verifies_the_new_evidence() -> None:
    replies: dict[tuple[str, str], list[Any]] = {
        ("counsel_a", SEARCH): [A_SEARCH],
        ("counsel_a", REBUT): [A_REBUT],
        ("counsel_b", SEARCH): [{"queries": []}],
        ("counsel_b", REBUT): [B_REBUT],
    }
    ctx, provider, service = context(replies)
    await verified(ctx)
    await stage(service).run(ctx)
    view = ctx.record.view
    # the search registered the new chunk for the issue before counsel cited it
    assert view.evidence["E-004"].chunk_id == EXTRA.chunk_id
    assert view.evidence["E-004"].issue_ids == ["I-1"]
    query = view.queries[-1]
    assert (query.query, query.stage, query.agent) == (
        A_SEARCH["queries"][0],
        Stage.S5,
        "counsel_a",
    )
    counters = [(r.id, r.targets, r.argument_id) for r in view.counterarguments.values()]
    assert counters == [("R-001", "C-002", "A-004"), ("R-002", "C-001", "A-005")]  # A first
    assert [(c.id, c.position_id) for c in list(view.claims.values())[3:]] == [
        ("C-004", "I-1.A"),
        ("C-005", "I-1.B"),
    ]
    assert [(c.evidence_id, c.pinpoint) for c in view.arguments["A-004"].citations] == [
        ("E-004", "para 13")
    ]
    values = ctx.run.values
    assert values["s5.rounds"] == 1 and values["s5.searches"] == 1
    assert values["s5.evidence_added"] == 1 and values["s5.counterarguments"] == 2
    assert values["s5.citation_validity_after_repair"] == 1.0 and values["s5.repairs"] == 0
    # S7 checks the rebuttal's citations, the new evidence included
    await VerificationGate(Stage.S7).run(at(ctx, Stage.S7))
    checked = {(v.claim_id, v.evidence_id) for v in ctx.record.view.verifications}
    assert {("C-004", "E-004"), ("C-005", "E-003")} <= checked
    assert ctx.run.values["s7.checked"] == 2
    assert len(provider.prompts(REBUT)) == 2 and len(provider.prompts(SEARCH)) == 2


async def test_counsel_sees_the_strongest_targets_with_their_assessments() -> None:
    ctx, _, _ = context({})
    await verified(ctx)
    view = ctx.record.view
    a = RebuttalCounsel("I-1", "I-1.A")
    assert [c.id for c in a.targets(view)] == ["C-003", "C-002"]  # verified before weak
    a.found = ["E-003"]  # as if its search had found Satyabrata Ghose
    # its own finds first, then what the targets cite, then the rest
    assert a.shown(view) == ["E-003", "E-002", "E-001"]
    fields = a.input_view(view)
    assert fields["targets"] == (
        "[C-003] I-1.B, verified: Loss need not be proved.\n  reasoning: r\n"
        "  cites E-002: verification verified; precedent fit distinguishable\n\n"
        "[C-002] I-1.B, weak: A reasonable sum may be kept.\n  reasoning: r\n"
        "  cites E-003: verification weak"
    )
    assert fields["max_counters"] == "3" and fields["position_id"] == "I-1.A"
    b = RebuttalCounsel("I-1", "I-1.B")
    assert [c.id for c in b.targets(view)] == ["C-001"]
    assert b.shown(view) == ["E-001", "E-002", "E-003"]


async def test_bad_targets_go_back_once_then_are_dropped() -> None:
    bad = {
        "counterarguments": [
            counter("C-001", "Own claim.", ("E-001", STATUTE_QUOTE)),  # A's own claim
            counter("C-2", "Answers C-002.", ("E-001", STATUTE_QUOTE)),
            counter("C-002", "Again C-002.", ("E-001", STATUTE_QUOTE)),  # one per claim
            counter("C-003", "No evidence."),
            counter("C-003", "One too many.", ("E-001", STATUTE_QUOTE)),
        ]
    }
    replies: dict[tuple[str, str], list[Any]] = {
        ("counsel_a", REBUT): [bad, bad],
        ("counsel_b", REBUT): [{"counterarguments": []}],
    }
    ctx, provider, service = context(replies, searches=0)
    await RebuttalStage(lambda: Researcher(service), max_counters=4).run(ctx)
    repair = next(r for r in provider.calls if r.tags.get("repair") == "record")
    feedback = repair.messages[2].content
    assert "counterarguments[0]: 'C-001' is not one of the claims to answer" in feedback
    assert "counterarguments[2]: C-002 is already answered; one per claim" in feedback
    assert "counterarguments[3]: every counterargument needs at least one citation" in feedback
    assert "counterarguments[4]: at most 4 counterarguments" in feedback
    view = ctx.record.view
    assert [(r.targets, r.argument_id) for r in view.counterarguments.values()] == [
        ("C-002", "A-004")
    ]
    assert view.validation_failures[0].agent == "counsel_a"
    assert provider.prompts(SEARCH) == []  # no searches allowed
    assert ctx.run.values["s5.repairs"] == 1


async def test_searches_are_capped_and_checked() -> None:
    too_many = {"queries": ["a b", "A  B", "", "c d", "e f"]}
    replies: dict[tuple[str, str], list[Any]] = {
        ("counsel_a", SEARCH): [too_many, too_many],
        ("counsel_a", REBUT): [{"counterarguments": []}],
        ("counsel_b", SEARCH): [{"queries": []}],
        ("counsel_b", REBUT): [{"counterarguments": []}],
    }
    ctx, provider, service = context(replies, searches=2)
    await stage(service).run(ctx)
    repair = next(r for r in provider.calls if r.tags.get("repair") == "record")
    feedback = repair.messages[2].content
    assert "queries[1]: the same search twice" in feedback
    assert "queries[2]: at most 2 searches" in feedback
    assert [q.query for q in ctx.record.view.queries if q.stage is Stage.S5] == ["a b"]
    assert len(service.searches) == 1


async def test_a_second_round_answers_the_first() -> None:
    replies: dict[tuple[str, str], list[Any]] = {
        ("counsel_a", REBUT): [A_REBUT, {"counterarguments": []}],
        ("counsel_b", REBUT): [
            {"counterarguments": []},
            {
                "counterarguments": [
                    counter("C-004", "Not part of the price here.", ("E-003", SATYABRATA_QUOTE))
                ]
            },
        ],
    }
    ctx, provider, service = context(replies, rounds=2, searches=0)
    ctx.record.register_evidence(EXTRA, ["I-1"], stage=Stage.S1, agent="researcher")  # E-004
    await stage(service).run(ctx)
    view = ctx.record.view
    assert [(r.targets, r.created_by.agent) for r in view.counterarguments.values()] == [
        ("C-002", "counsel_a"),
        ("C-004", "counsel_b"),  # round 2: B answers A's round-1 counterargument
    ]
    round_two = [c for c in provider.prompts(REBUT) if c.tags["round"] == "2"]
    a_two = next(c for c in round_two if c.tags["agent"] == "counsel_a")
    assert "[C-002]" not in a_two.messages[0].content  # already answered by A
    assert "[C-003]" in a_two.messages[0].content
    assert ctx.run.values["s5.rounds"] == 2


async def test_the_second_round_is_skipped_when_the_budget_cannot_cover_it() -> None:
    replies: dict[tuple[str, str], list[Any]] = {
        ("counsel_a", REBUT): [{"counterarguments": []}],
        ("counsel_b", REBUT): [{"counterarguments": []}],
    }
    ctx, provider, service = context(replies, rounds=2, searches=0, max_tokens=100)
    await stage(service).run(ctx)  # round 1 spends 2 x (10 + 24) = 68; 32 are left
    assert len(provider.calls) == 2
    warnings = [e for e in ctx.run.emitter.events() if e.type == "budget_warning"]
    assert [w.payload.action for w in warnings] == ["skipped rebuttal round 2"]
    assert ctx.run.values["s5.rounds"] == 1


async def test_no_rounds_and_passthrough_write_nothing() -> None:
    ctx, provider, service = context({}, rounds=0)
    await stage(service).run(ctx)
    await stage(service).passthrough(ctx)
    assert provider.calls == [] and ctx.record.view.counterarguments == {}
    assert ctx.run.values["s5.rounds"] == 0
