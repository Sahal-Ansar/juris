"""Counsel and S2 (PLAN 7.5) on canned outputs: both sides at once with ordered writes, what
counsel sees, the citation checks, and the validity counts. No model, no database."""

import asyncio
import json
from typing import Any

from juris.agents.counsel import Counsel
from juris.baselines.base import RunContext
from juris.config import ModelsConfig, ModelSpec, PipelineProfile, Settings, resolve_run_config
from juris.eval.juris_eval import load_split
from juris.events import EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.llm.gateway import LLMGateway
from juris.llm.providers.fake import FakeProvider
from juris.llm.types import ProviderRequest, ProviderResponse
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.opening import OpeningStage
from tests.test_b1 import (
    FATEH_TEXT,
    HITS,
    SATYABRATA_QUOTE,
    SATYABRATA_TEXT,
    STATUTE_QUOTE,
    STATUTE_TEXT,
    FakeService,
)

FATEH_QUOTE = "The party must prove actual loss in the case of forfeiture of earnest money."


class SlowA:
    """Answers like ``FakeProvider``, but Counsel A's calls finish last."""

    name = "fake"

    def __init__(self, inner: FakeProvider) -> None:
        self.inner = inner

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        if request.tags.get("agent") == "counsel_a":
            await asyncio.sleep(0.05)
        return await self.inner.complete(request)


def claim(text: str, *cites: tuple[str, str], reasoning: str = "Because.") -> dict[str, Any]:
    return {
        "text": text,
        "reasoning": reasoning,
        "citations": [{"evidence_id": e, "quote": q} for e, q in cites],
    }


A_OUT = {"claims": [claim("Compensation is capped by s. 74.", ("E-001", STATUTE_QUOTE))]}
B_OUT = {
    "claims": [
        claim("A reasonable earnest sum may be kept.", ("E-3", SATYABRATA_QUOTE)),  # loose ID
        claim("Loss must be proved.", ("E-002", FATEH_QUOTE), ("E-002", FATEH_QUOTE)),  # twice
    ]
}


def context(responses: dict[str, list[Any]], *, slow_a: bool = False) -> tuple[
    StageContext, FakeProvider
]:  # fmt: skip
    fake = FakeProvider(
        {k: [r if isinstance(r, str) else json.dumps(r) for r in v] for k, v in responses.items()}
    )
    provider: Any = SlowA(fake) if slow_a else fake
    model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    config = resolve_run_config(settings, PipelineProfile(name="t", kind="juris"))
    item = load_split("dev")[0]
    emitter = EventEmitter(InMemoryEventStore(), run_id="r", case_id=item.id)
    emitter.emit(CaseCreatedPayload(question=item.question, profile="t", corpus_snapshot_id="s"))
    record = CaseRecord(emitter, FakeService())
    record.frame_issue("Can the seller forfeit the earnest money?", ["No.", "Yes."],
                       stage=Stage.S0, agent="issue_framer", facts=["Earnest money was paid."],
                       assumptions=["No arbitration clause."])  # fmt: skip
    for hit in HITS:
        record.register_evidence(hit, ["I-1"], stage=Stage.S1, agent="researcher")
    run = RunContext(item, config, LLMGateway({"fake": provider}, prices={}), emitter, 0)
    budget = BudgetManager(config.budget, Spend)
    return StageContext(run, record, budget, Stage.S2, "on"), fake


async def test_both_sides_argue_at_once_and_are_written_in_order() -> None:
    ctx, _ = context({"counsel_a": [A_OUT], "counsel_b": [B_OUT]}, slow_a=True)
    await OpeningStage().run(ctx)
    view = ctx.record.view
    done = [e.agent for e in ctx.run.emitter.events() if e.type == "agent_completed"]
    assert done == ["counsel_b", "counsel_a"]  # B finished first ...
    claims = [(c.id, c.position_id, c.author_agent) for c in view.claims.values()]
    assert claims == [  # ... but A's claims come first
        ("C-001", "I-1.A", "counsel_a"),
        ("C-002", "I-1.B", "counsel_b"),
        ("C-003", "I-1.B", "counsel_b"),
    ]
    citations = {a.id: [(c.evidence_id, c.pinpoint) for c in a.citations]
                 for a in view.arguments.values()}  # fmt: skip
    assert citations == {
        "A-001": [("E-001", "s. 74")],
        "A-002": [("E-003", "paras 3-5")],  # "E-3" read as E-003
        "A-003": [("E-002", "para 12")],  # the repeated citation once
    }
    assert view.arguments["A-001"].reasoning == "Because."
    values = ctx.run.values
    assert values["s2.claims"] == 3 and values["s2.citations"] == 3
    assert values["s2.citations_proposed"] == 4 and values["s2.citation_failures_first"] == 0
    assert values["s2.citation_validity_after_repair"] == 1.0
    assert values["s2.repairs"] == 0


async def test_each_counsel_sees_its_own_position_and_the_same_evidence() -> None:
    ctx, provider = context({"counsel_a": [A_OUT], "counsel_b": [B_OUT]})
    await OpeningStage().run(ctx)
    prompts = {r.tags["agent"]: r.messages[0].content for r in provider.calls}
    a, b = prompts["counsel_a"], prompts["counsel_b"]
    assert '<your_position id="I-1.A">\nNo.\n</your_position>' in a
    assert "<other_positions>\nI-1.B: Yes.\n</other_positions>" in a
    assert '<your_position id="I-1.B">\nYes.\n</your_position>' in b
    assert "- Earnest money was paid.\n- (assumed) No arbitration clause." in a
    assert f"[E-001] Indian Contract Act, 1872, s. 74\n{STATUTE_TEXT}" in a
    assert f"[E-002] Fateh Chand v. Balkishan Dass, para 12\n{FATEH_TEXT}" in a
    evidence = a[a.index("<evidence>") :]
    assert evidence == b[b.index("<evidence>") :]  # one pool for both sides
    swapped = a.replace("I-1.A", "X").replace("I-1.B", "I-1.A").replace("X", "I-1.B")
    swapped = swapped.replace("\nNo.\n", "\nTMP\n").replace("I-1.A: Yes.", "I-1.A: No.")
    assert swapped.replace("\nTMP\n", "\nYes.\n") == b  # the same prompt but for the position


async def test_bad_citations_go_back_once_then_are_dropped() -> None:
    bad = {
        "claims": [
            claim("Capped.", ("E-001", STATUTE_QUOTE), ("E-009", "made up")),
            claim("Loss must be proved.", ("E-002", "the party need not prove loss")),
            claim("No reason given.", ("E-003", SATYABRATA_QUOTE), reasoning=" "),
        ]
    }
    ctx, provider = context({"counsel_a": [bad, bad], "counsel_b": [{"claims": []}]})
    await OpeningStage().run(ctx)
    repair = next(r for r in provider.calls if r.tags.get("repair") == "record")
    feedback = repair.messages[2].content
    assert "claims[0].citations[1]: 'E-009' is not one of the evidence items you were given" in (
        feedback
    )
    assert "claims[1].citations[0]: E-002: " in feedback and "quote check" in feedback
    assert "claims[2]: the claim has no reasoning" in feedback
    view = ctx.record.view
    assert [c.text for c in view.claims.values()] == ["Capped."]
    assert [c.evidence_id for c in view.arguments["A-001"].citations] == ["E-001"]
    values = ctx.run.values
    assert values["s2.citations_proposed"] == 4 and values["s2.citation_failures_first"] == 2
    assert values["s2.citations_after_repair"] == 4
    assert values["s2.citation_failures_after_repair"] == 2
    assert values["s2.citation_validity_after_repair"] == 0.5
    assert values["s2.repairs"] == 1
    assert view.validation_failures[0].agent == "counsel_a"


async def test_a_repair_that_fixes_the_quotes_counts_as_valid() -> None:
    bad = {"claims": [claim("Loss must be proved.", ("E-002", "the party need not prove loss"))]}
    good = {"claims": [claim("Loss must be proved.", ("E-002", FATEH_QUOTE))]}
    ctx, _ = context({"counsel_a": [bad, good], "counsel_b": [B_OUT]})
    await OpeningStage().run(ctx)
    values = ctx.run.values
    assert values["s2.citation_failures_first"] == 1
    assert values["s2.citation_failures_after_repair"] == 0
    assert values["s2.citation_validity_after_repair"] == 1.0
    assert ctx.record.view.validation_failures == []


async def test_the_evidence_budget_limits_what_can_be_cited() -> None:
    budget = len(STATUTE_TEXT) + len(FATEH_TEXT)  # E-001 and E-002 fit; E-003 does not
    out = {"claims": [claim("Kept.", ("E-003", SATYABRATA_QUOTE))]}
    ctx, provider = context({"counsel_a": [out, out], "counsel_b": [{"claims": []}]})
    await OpeningStage(max_evidence_chars=budget).run(ctx)
    prompt = provider.calls[0].messages[0].content
    assert "[E-002]" in prompt and "[E-003]" not in prompt and SATYABRATA_TEXT not in prompt
    assert ctx.record.view.claims == {}  # E-003 was not given, so the citation is dropped


async def test_evidence_of_another_issue_is_not_shown() -> None:
    ctx, _ = context({})
    ctx.record.frame_issue("Second issue?", ["Yes.", "No."], stage=Stage.S0, agent="f")
    shown = Counsel("I-2", "I-2.A").shown(ctx.record.view)
    assert shown == []
    ctx.record.register_evidence(HITS[0], ["I-2"], stage=Stage.S1, agent="researcher")
    assert Counsel("I-2", "I-2.A").shown(ctx.record.view) == ["E-001"]


async def test_surplus_claims_are_dropped() -> None:
    many = {"claims": [claim(f"Claim {n}.", ("E-001", STATUTE_QUOTE)) for n in range(4)]}
    ctx, _ = context({"counsel_a": [many, many], "counsel_b": [{"claims": []}]})
    await OpeningStage(max_claims=2).run(ctx)
    assert [c.text for c in ctx.record.view.claims.values()] == ["Claim 0.", "Claim 1."]


async def test_passthrough_writes_nothing() -> None:
    ctx, provider = context({})
    await OpeningStage().passthrough(ctx)
    assert provider.calls == [] and ctx.record.view.claims == {}
