"""The Researcher and S1 (PLAN 7.4) on a fake retrieval service: ranking, the per-judgment
cap, provisions as hints, shared evidence, S5's targeted search. No model, no database."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from juris.agents.researcher import Researcher, select
from juris.baselines.base import RunContext
from juris.config import (
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
from juris.llm.providers.fake import FakeProvider
from juris.models import Chunk, Document, StatuteMeta
from juris.models.common import DocumentKind, Stage
from juris.pipeline.budget import BudgetManager, Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.research import ResearchFailed, ResearchStage
from juris.retrieval.lookup import LinkedCase
from juris.retrieval.service import Hit

ACT = "ACT-contract_act"


def hit(chunk_id: str, doc_id: str, rank: int = 1, query: str = "q") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        rank=rank,
        text=f"text of {chunk_id}",
        context_header=chunk_id,
        title=doc_id,
        query=query,
        retrieval_score=1 / rank,
    )


RANKED = [
    hit("J1-1", "SC-J1"),
    hit("J1-2", "SC-J1"),
    hit("s16", ACT),
    hit("J1-3", "SC-J1"),
    hit("J1-4", "SC-J1"),
    hit("J2-1", "SC-J2"),
    hit("J1-5", "SC-J1"),
    hit("J2-2", "SC-J2"),
    hit("s19A", ACT),
]
SECTIONS = {
    "s. 16 of the Indian Contract Act": [hit("s16", ACT)],
    "s. 146 of the Indian Contract Act": [hit("s146", ACT)],
    "s. 74 of the Indian Contract Act": [hit("s74", ACT)],
}
# the citation graph: which judgments cite a section
CITING = {"74": ["SC-J1", "SC-J2", "SC-J3"], "146": ["SC-J1"]}


def linked(doc_id: str) -> LinkedCase:
    return LinkedCase(ref=doc_id, doc_id=doc_id, title=None, decision_date=None, mentions=1,
                      paragraphs=[], citations=[], treatments=[])  # fmt: skip


@dataclass
class FakeRetrieval:
    ranked: list[Hit] = field(default_factory=lambda: list(RANKED))
    searches: list[dict[str, Any]] = field(default_factory=list)

    def search(self, queries: Sequence[str], *, k: int = 10, rerank: bool = True) -> list[Hit]:
        self.searches.append({"queries": list(queries), "k": k, "rerank": rerank})
        return self.ranked[:k]

    def section_hits(self, reference: str) -> list[Hit]:
        return SECTIONS.get(reference, [])

    def cases_citing_section(self, act: str, section: str) -> list[LinkedCase]:
        return [linked(d) for d in CITING.get(section, [])]

    def records(self, chunk_id: str) -> tuple[Document, Chunk]:
        statute = chunk_id.startswith("s")
        doc_id = ACT if statute else f"SC-{chunk_id.split('-')[0]}"
        document = Document(
            id=doc_id,
            kind=DocumentKind.STATUTE if statute else DocumentKind.JUDGMENT,
            title=doc_id,
            licence="test",
            corpus_snapshot="snap",
        )
        body = f"text of {chunk_id}"
        where: dict[str, Any] = (
            {"statute": StatuteMeta(act="Indian Contract Act", section=chunk_id[1:])}
            if statute
            else {"para_start": 1, "para_end": 1}
        )
        chunk = Chunk(
            id=chunk_id, document_id=doc_id, text=body, char_start=0, char_end=len(body), **where
        )
        return document, chunk


def context(service: FakeRetrieval, *, params: DeliberationParams | None = None) -> StageContext:
    model = ModelSpec(provider="fake", name="m", effort=None)
    settings = Settings(_env_file=None, models=ModelsConfig(default=model))  # type: ignore[call-arg]
    profile = PipelineProfile(name="t", kind="juris", params=params or DeliberationParams())
    config = resolve_run_config(settings, profile)
    item = load_split("dev")[0]
    emitter = EventEmitter(InMemoryEventStore(), run_id="r", case_id=item.id)
    emitter.emit(CaseCreatedPayload(question=item.question, profile="t", corpus_snapshot_id="s"))
    record = CaseRecord(emitter, service, params=profile.params)
    gateway = LLMGateway({"fake": FakeProvider({})}, prices={})
    run = RunContext(item, config, gateway, emitter, 0)
    return StageContext(run, record, BudgetManager(config.budget, Spend), Stage.S1, "on")


def frame(ctx: StageContext, *queries: tuple[str, str], issues: int = 1) -> None:
    for n in range(issues):
        issue = ctx.record.frame_issue(f"Issue {n + 1}?", ["Yes.", "No."], stage=Stage.S0,
                                       agent="issue_framer")  # fmt: skip
        for query, kind in queries:
            ctx.record.add_query(query, [issue.id], stage=Stage.S0, agent="issue_framer",
                                 kind=kind)  # type: ignore[arg-type]  # fmt: skip


def chunks(ctx: StageContext) -> list[str]:
    return [e.chunk_id for e in ctx.record.view.evidence.values()]


def test_the_pool_is_ranked_capped_and_led_by_confirmed_provisions() -> None:
    service = FakeRetrieval()
    ctx = context(service)
    frame(
        ctx,
        ("presumption of undue influence", "search"),
        ("s. 16 of the Indian Contract Act", "provision"),
        ("s. 146 of the Indian Contract Act", "provision"),
        ("burden of proof", "search"),
    )
    researcher = Researcher(service, candidate_k=60, per_document=3)
    [pool] = researcher.research(ctx.record, stage=Stage.S1, min_items=2, max_items=6)
    assert service.searches == [
        {"queries": ["presumption of undue influence", "burden of proof"], "k": 60, "rerank": False}
    ]
    # s. 16 first (confirmed by the search), J1 capped at 3, then J2
    assert chunks(ctx) == ["s16", "J1-1", "J1-2", "J1-3", "J2-1", "J2-2"]
    assert pool.evidence == ["E-001", "E-002", "E-003", "E-004", "E-005", "E-006"]
    assert pool.provisions_kept == ["s. 16 of the Indian Contract Act"]
    assert pool.provisions_unconfirmed == ["s. 146 of the Indian Contract Act"]
    assert pool.candidates == 9
    evidence = ctx.record.view.evidence["E-001"]
    assert evidence.issue_ids == ["I-1"] and evidence.registered_at_stage is Stage.S1
    assert evidence.created_by.agent == "researcher"


def test_the_cap_is_relaxed_to_reach_the_minimum() -> None:
    service = FakeRetrieval([h for h in RANKED if h.doc_id == "SC-J1"])
    ctx = context(service)
    frame(ctx, ("q", "search"))
    Researcher(service, per_document=3).research(ctx.record, stage=Stage.S1, min_items=5,
                                                  max_items=25)  # fmt: skip
    assert chunks(ctx) == ["J1-1", "J1-2", "J1-3", "J1-4", "J1-5"]


def test_issues_share_evidence_and_fall_back_to_their_question() -> None:
    service = FakeRetrieval()
    ctx = context(service)
    frame(ctx, issues=2)  # no queries at all
    Researcher(service).research(ctx.record, stage=Stage.S1, min_items=1, max_items=2)
    assert [s["queries"] for s in service.searches] == [["Issue 1?"], ["Issue 2?"]]
    view = ctx.record.view
    assert list(view.evidence) == ["E-001", "E-002"]  # the same two chunks for both issues
    assert view.evidence["E-001"].issue_ids == ["I-1", "I-2"]


def test_targeted_search_registers_only_new_chunks() -> None:
    service = FakeRetrieval()
    ctx = context(service)
    frame(ctx, ("q", "search"))
    researcher = Researcher(service, per_document=3)
    researcher.research(ctx.record, stage=Stage.S1, min_items=1, max_items=3)  # J1-1, J1-2, s16
    found = researcher.targeted_search(
        ctx.record, "rebut the presumption", ["I-1"], stage=Stage.S5, agent="counsel_b", k=2
    )
    # J1-3 is the next J1 chunk (J1 holds 2), then J1-4 would break the cap: J2-1 instead
    assert [e.chunk_id for e in found] == ["J1-3", "J2-1"]
    assert [e.id for e in found] == ["E-004", "E-005"]
    assert all(e.registered_at_stage is Stage.S5 for e in found)
    query = ctx.record.view.queries[-1]
    assert (query.query, query.agent, query.stage) == ("rebut the presumption", "counsel_b",
                                                       Stage.S5)  # fmt: skip


def test_select_caps_judgments_but_not_statutes() -> None:
    statute = [hit(f"s{n}", ACT) for n in range(5)]
    assert [h.chunk_id for h in select(statute, 5, 0, 1)] == ["s0", "s1", "s2", "s3", "s4"]
    judgment = [hit(f"j{n}", "SC-X") for n in range(5)]
    assert [h.chunk_id for h in select(judgment, 5, 0, 2)] == ["j0", "j1"]
    assert [h.chunk_id for h in select(judgment, 3, 4, 2)] == ["j0", "j1", "j2"]  # most wins


async def test_the_stage_records_its_pools() -> None:
    service = FakeRetrieval()
    ctx = context(service, params=DeliberationParams(evidence_per_issue_min=2,
                                                     evidence_per_issue_max=4))  # fmt: skip
    frame(ctx, ("q", "search"), ("s. 146 of the Indian Contract Act", "provision"))
    await ResearchStage(lambda: service).run(ctx)
    assert chunks(ctx) == ["J1-1", "J1-2", "s16", "J1-3"]
    values = ctx.run.values
    assert (values["s1.evidence"], values["s1.documents"]) == (4.0, 2.0)
    assert (values["s1.provisions_kept"], values["s1.provisions_unconfirmed"]) == (0.0, 1.0)
    assert ctx.run.details["s1.pools"][0]["evidence"] == ["E-001", "E-002", "E-003", "E-004"]


async def test_a_search_that_finds_nothing_fails_the_stage() -> None:
    service = FakeRetrieval([])
    ctx = context(service)
    frame(ctx, ("q", "search"))
    with pytest.raises(ResearchFailed, match="found no evidence"):
        await ResearchStage(lambda: service).run(ctx)


async def test_passthrough_searches_each_question_for_the_minimum() -> None:
    service = FakeRetrieval()
    ctx = context(service, params=DeliberationParams(evidence_per_issue_min=3))
    frame(ctx, ("ignored", "search"))
    await ResearchStage(lambda: service).passthrough(ctx)
    assert service.searches == [{"queries": ["Issue 1?"], "k": 3, "rerank": False}]
    assert chunks(ctx) == ["J1-1", "J1-2", "s16"]


def test_the_service_is_made_once() -> None:
    made: list[int] = []

    def factory() -> FakeRetrieval:
        made.append(1)
        return FakeRetrieval()

    stage = ResearchStage(factory)
    assert made == []
    assert stage.researcher is stage.researcher and made == [1]


def test_the_citation_graph_confirms_a_section_the_search_missed() -> None:
    service = FakeRetrieval([*RANKED[:2], hit("J3-1", "SC-J3"), *RANKED[5:8]])  # no statute
    ctx = context(service)
    frame(
        ctx,
        ("liquidated damages", "search"),
        ("s. 74 of the Indian Contract Act", "provision"),  # cited by J1, J2, J3: kept
        ("s. 146 of the Indian Contract Act", "provision"),  # cited by J1 only: left out
        ("Indian Penal Code s. 300", "provision"),  # not a corpus Act: left out
    )
    researcher = Researcher(service, cite_min=3)
    [pool] = researcher.research(ctx.record, stage=Stage.S1, min_items=1, max_items=4)
    assert chunks(ctx) == ["s74", "J1-1", "J1-2", "J3-1"]
    assert pool.provisions_kept == ["s. 74 of the Indian Contract Act"]
    assert pool.provisions_unconfirmed == [
        "s. 146 of the Indian Contract Act",
        "Indian Penal Code s. 300",
    ]
    stricter = context(service)
    frame(stricter, ("q", "search"), ("s. 74 of the Indian Contract Act", "provision"))
    Researcher(service, cite_min=4).research(stricter.record, stage=Stage.S1, min_items=1,
                                             max_items=4)  # fmt: skip
    assert "s74" not in chunks(stricter)
