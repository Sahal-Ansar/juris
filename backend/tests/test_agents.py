"""The agent framework (PLAN 7.2): a dummy counsel agent end to end on canned responses,
through one repair cycle and the drop step. No model, no database."""

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from juris.agents.base import Agent, Problem, _item, where
from juris.baselines.base import RunContext
from juris.config import (
    ModelsConfig,
    ModelSpec,
    PipelineProfile,
    Role,
    Settings,
    resolve_run_config,
)
from juris.eval.juris_eval import load_split
from juris.events import CaseView, EventEmitter, InMemoryEventStore
from juris.events.catalog import CaseCreatedPayload
from juris.llm.gateway import LLMGateway
from juris.llm.providers.fake import FakeProvider
from juris.models import Citation
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.prompts import prompt_hashes
from tests.test_b1 import FATEH_TEXT, HITS, SATYABRATA_QUOTE, STATUTE_QUOTE, FakeService

TEMPLATE = """---
id: dummy_counsel
version: 1
---
Argue for $position.

<question>
$question
</question>

<evidence>
$evidence
</evidence>
"""


class DCitation(BaseModel):
    evidence_id: str
    quote: str


class DClaim(BaseModel):
    position_id: str
    text: str
    citations: list[DCitation]


class DOut(BaseModel):
    claims: list[DClaim]
    note: str = ""


class DummyCounsel(Agent[DOut]):
    role = Role.COUNSEL
    prompt = "dummy_counsel"
    duty = "Counsel argues the strongest evidence-backed case for one position."
    output_model = DOut

    def input_view(self, view: CaseView) -> dict[str, str]:
        evidence = "\n".join(
            f"[{e.id}] {view.chunks[e.chunk_id].text}" for e in view.evidence.values()
        )
        question = next(iter(view.issues.values())).question
        return {"position": "I-1.A", "question": question, "evidence": evidence}

    def validate(self, output: DOut, record: CaseRecord) -> list[Problem]:
        positions = {p.id for i in record.view.issues.values() for p in i.positions}
        problems: list[Problem] = []
        if output.note == "off-topic":
            problems.append(Problem("the output does not answer the question"))
        for n, claim in enumerate(output.claims):
            if claim.position_id not in positions:
                problems.append(Problem(f"unknown position {claim.position_id!r}", ("claims", n)))
            if not claim.citations:
                problems.append(Problem("a claim needs at least one citation", ("claims", n)))
            for k, c in enumerate(claim.citations):
                where_ = ("claims", n, "citations", k)
                try:
                    citation = Citation(
                        claim_id="C-999", evidence_id=c.evidence_id, pinpoint="p", quote=c.quote
                    )
                except ValidationError:
                    problems.append(Problem(f"{c.evidence_id!r} is not an evidence ID", where_))
                    continue
                error = record.check_citation(citation)
                if error is not None:
                    problems.append(Problem(error, where_))
        return problems

    def apply(self, output: DOut, record: CaseRecord, *, stage: Stage) -> None:
        for c in output.claims:
            claim = record.add_claim(c.text, c.position_id, stage=stage, agent=self.name)
            citations = [
                Citation(claim_id=claim.id, evidence_id=x.evidence_id, pinpoint="p", quote=x.quote)
                for x in c.citations
            ]
            record.add_argument([claim.id], "r", citations, stage=stage, agent=self.name)


BAD_QUOTE = "the party need not prove anything at all"


def claim(position: str, text: str, *cites: tuple[str, str]) -> dict[str, Any]:
    return {
        "position_id": position,
        "text": text,
        "citations": [{"evidence_id": e, "quote": q} for e, q in cites],
    }


GOOD = {"claims": [claim("I-1.A", "Capped.", ("E-001", STATUTE_QUOTE))]}
BAD = {"claims": [claim("I-1.A", "Capped.", ("E-002", BAD_QUOTE))]}
MIXED = {
    "claims": [
        claim("I-1.A", "Only a bad quote.", ("E-002", BAD_QUOTE)),
        claim("I-1.A", "One good, one bad.", ("E-003", SATYABRATA_QUOTE), ("E-9", "x")),
        claim("I-1.C", "No such position.", ("E-001", STATUTE_QUOTE)),
    ]
}


@pytest.fixture(autouse=True)
def template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "dummy_counsel.md").write_text(TEMPLATE, encoding="utf-8")
    monkeypatch.setattr(DummyCounsel, "prompts_dir", tmp_path)


def context(*responses: Any, seed: int = 7) -> tuple[StageContext, FakeProvider]:
    provider = FakeProvider(
        {"counsel_a": [r if isinstance(r, str) else json.dumps(r) for r in responses]}
    )
    model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    config = resolve_run_config(settings, PipelineProfile(name="t", kind="juris"))
    emitter = EventEmitter(InMemoryEventStore(), run_id="JE-001-s0", case_id="JE-001")
    emitter.emit(CaseCreatedPayload(question="Q?", profile="t", corpus_snapshot_id="snap"))
    record = CaseRecord(emitter, FakeService())
    record.frame_issue("Can the seller forfeit?", ["No.", "Yes."], stage=Stage.S0, agent="f")
    for hit in HITS:
        record.register_evidence(hit, ["I-1"], stage=Stage.S1, agent="researcher")
    run = RunContext(
        load_split("dev")[0], config, LLMGateway({"fake": provider}, prices={}), emitter, seed
    )
    budget = BudgetManager(config.budget, lambda: Spend())
    return StageContext(run, record, budget, Stage.S2, "on"), provider


def types(ctx: StageContext) -> list[str]:
    return [e.type for e in ctx.run.emitter.events()]


async def test_a_valid_output_is_applied_without_repair() -> None:
    ctx, provider = context(GOOD)
    output = await DummyCounsel("counsel_a").run(ctx)
    assert output is not None and output.claims[0].text == "Capped."
    assert len(provider.calls) == 1
    request = provider.calls[0]
    assert request.system is not None
    assert request.system.startswith("You are the counsel in Juris")
    assert "argues the strongest evidence-backed case" in request.system
    assert '"claims"' in request.system  # the output schema
    body = request.messages[0].content
    assert "Argue for I-1.A." in body and FATEH_TEXT in body and "[E-002]" in body
    assert request.tags == {
        "agent": "counsel_a",
        "role": "counsel",
        "stage": "S2",
        "item": "JE-001",
        "prompt": "dummy_counsel@v1",
    }
    view = ctx.record.view
    assert view.claims["C-001"].author_agent == "counsel_a"
    assert view.arguments["A-001"].citations[0].quote == STATUTE_QUOTE
    assert view.agents["counsel_a"].status == "done"
    assert "validation_failed" not in types(ctx)


async def test_one_repair_cycle_fixes_the_output() -> None:
    ctx, provider = context(BAD, GOOD)
    output = await DummyCounsel("counsel_a").run(ctx)
    assert output is not None and len(provider.calls) == 2
    repair = provider.calls[1]
    assert [m.role for m in repair.messages] == ["user", "assistant", "user"]
    assert json.loads(repair.messages[1].content) == BAD  # its own previous output
    feedback = repair.messages[2].content
    assert "The Case Record rejected your output" in feedback
    assert "- claims[0].citations[0]: E-002: " in feedback and "quote check" in feedback
    assert repair.tags["repair"] == "record"
    assert ctx.record.view.arguments["A-001"].citations[0].evidence_id == "E-001"
    assert "validation_failed" not in types(ctx)


async def test_what_still_fails_after_the_repair_is_dropped() -> None:
    ctx, provider = context(MIXED, MIXED)
    output = await DummyCounsel("counsel_a").run(ctx)
    assert output is not None and len(provider.calls) == 2
    # claim 0 lost its only citation, so it went too; claim 2 named no real position
    assert [c.text for c in output.claims] == ["One good, one bad."]
    assert [c.evidence_id for c in output.claims[0].citations] == ["E-003"]
    view = ctx.record.view
    assert [c.text for c in view.claims.values()] == ["One good, one bad."]
    failure = view.validation_failures[0]
    assert failure.agent == "counsel_a" and failure.stage is Stage.S2
    assert failure.dropped == "claims[2], claims[1].citations[1], claims[0].citations[0], claims[0]"
    assert any(e == "claims[2]: unknown position 'I-1.C'" for e in failure.errors)
    assert any(
        e.startswith("claims[0]: a claim needs at least one citation") for e in failure.errors
    )
    assert len(view.validation_failures) == 1


async def test_a_repair_that_breaks_a_valid_citation_does_not_cost_it() -> None:
    # the repair fixes nothing and breaks claim 1's good citation (as a small model did in S5)
    worse = copy.deepcopy(MIXED)
    worse["claims"][1]["citations"][0]["quote"] = "when " + SATYABRATA_QUOTE
    ctx, _ = context(MIXED, worse)
    output = await DummyCounsel("counsel_a").run(ctx)
    assert output is not None and [c.text for c in output.claims] == ["One good, one bad."]
    failure = ctx.record.view.validation_failures[0]  # the errors of the attempt kept
    assert failure.dropped == "claims[2], claims[1].citations[1], claims[0].citations[0], claims[0]"


async def test_a_clean_but_shorter_repair_is_not_preferred() -> None:
    two = {"claims": [GOOD["claims"][0], claim("I-1.A", "Bad.", ("E-002", BAD_QUOTE))]}
    ctx, _ = context(two, {"claims": []})  # the repair gives up instead of fixing the quote
    output = await DummyCounsel("counsel_a").run(ctx)
    assert output is not None and [c.text for c in output.claims] == ["Capped."]
    assert ctx.record.view.validation_failures[0].dropped == "claims[1].citations[0], claims[1]"


async def test_a_problem_with_the_whole_output_drops_it() -> None:
    off = {**GOOD, "note": "off-topic"}
    ctx, _ = context(off, off)
    assert await DummyCounsel("counsel_a").run(ctx) is None
    view = ctx.record.view
    assert view.claims == {}
    assert view.validation_failures[0].dropped == "the whole output"
    assert view.validation_failures[0].errors == ["the output does not answer the question"]
    assert view.agents["counsel_a"].status == "done"


async def test_output_that_never_fits_the_schema_is_dropped() -> None:
    ctx, provider = context("not json", "still not json")  # the call and the gateway's retry
    assert await DummyCounsel("counsel_a").run(ctx) is None
    assert len(provider.calls) == 2
    failure = ctx.record.view.validation_failures[0]
    assert failure.dropped == "the whole output"
    assert failure.errors[0].startswith("the output did not match the schema")
    assert types(ctx)[-1] == "agent_completed"


async def test_a_repair_that_breaks_the_schema_keeps_the_first_output() -> None:
    ctx, provider = context(MIXED, "nope", "nope")
    output = await DummyCounsel("counsel_a").run(ctx)
    assert len(provider.calls) == 3
    assert output is not None and [c.text for c in output.claims] == ["One good, one bad."]


async def test_the_seed_offset_separates_agents_of_one_role() -> None:
    ctx, provider = context(GOOD, seed=7)
    await DummyCounsel("counsel_a", seed_offset=2).run(ctx)
    assert provider.calls[0].seed == 9


def test_the_shared_prompts_are_versioned_in_the_manifest() -> None:
    hashes = prompt_hashes()
    assert "agent_shared@v1" in hashes and "agent_repair@v1" in hashes


def test_paths() -> None:
    assert where(("claims", 2, "citations", 0)) == "claims[2].citations[0]"
    assert where(("issues", 1, "positions")) == "issues[1].positions"
    assert _item(("issues", 1, "positions")) == ("issues", 1)
    assert _item(("note",)) == ()
    assert str(Problem("bad", ("claims", 0))) == "claims[0]: bad"
    assert str(Problem("bad")) == "bad"
