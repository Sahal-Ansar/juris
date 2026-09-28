"""Retrieval service and evidence registry (PLAN 4.6).

The registry tests use a fake corpus and an in-memory event store. The service tests run the
real lexical and dense retrievers (fake encoders) over the small corpus from ``test_lexical``.
"""

import datetime as dt
from dataclasses import dataclass

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine

from juris.db.load import refresh_lexeme_stats
from juris.events import Event, InMemoryEventStore, fold
from juris.events.catalog import CaseCreatedPayload
from juris.events.store import EventEmitter
from juris.models import Chunk, Document, Stage
from juris.pipeline.evidence_registry import EvidenceRegistry
from juris.retrieval.dense import DenseRetriever
from juris.retrieval.embed import embed_missing
from juris.retrieval.hybrid import HybridRetriever
from juris.retrieval.lexical import LexicalRetriever
from juris.retrieval.rerank import Reranker
from juris.retrieval.service import Hit, RetrievalService
from tests.test_embed import FakeEncoder
from tests.test_lexical import ACT, HC, SC_NEW, SC_OLD, corpus  # noqa: F401 (fixture)
from tests.test_rerank import OverlapEncoder

# ---- the registry (no database) ----------------------------------------------------------


@dataclass
class FakeCorpus:
    calls: int = 0

    def records(self, chunk_id: str) -> tuple[Document, Chunk]:
        self.calls += 1
        if chunk_id.startswith("missing"):
            raise KeyError(chunk_id)
        doc_id = chunk_id.split("#")[0]
        document = Document(
            id=doc_id, kind="judgment", title="A v. B", licence="CC-BY-4.0", corpus_snapshot="s"
        )
        chunk = Chunk(
            id=chunk_id, document_id=doc_id, text="text", para_start=0, para_end=0,
            char_start=0, char_end=4,
        )  # fmt: skip
        return document, chunk


def hit(chunk_id: str, query: str = "q", score: float = 0.5) -> Hit:
    return Hit(
        chunk_id=chunk_id, doc_id=chunk_id.split("#")[0], rank=1, text="text",
        context_header=None, title="A v. B", query=query, retrieval_score=score,
        rerank_score=0.9,
    )  # fmt: skip


def started_run(run_id: str = "run-1") -> tuple[InMemoryEventStore, EventEmitter]:
    """A store whose run has begun (every run starts with ``case_created``)."""
    store = InMemoryEventStore()
    emitter = EventEmitter(store, run_id=run_id, case_id="case-1")
    emitter.emit(CaseCreatedPayload(question="q", profile="juris_full", corpus_snapshot_id="s"))
    return store, emitter


def evidence_events(store: InMemoryEventStore, run_id: str = "run-1") -> list[Event]:
    return [e for e in store.read(run_id) if e.type == "evidence_registered"]


def new_registry() -> tuple[EvidenceRegistry, InMemoryEventStore, FakeCorpus]:
    store, emitter = started_run()
    corpus_ = FakeCorpus()
    return EvidenceRegistry(corpus_, emitter), store, corpus_


def test_the_same_chunk_gets_the_same_id() -> None:
    registry, store, corpus_ = new_registry()
    first = registry.register(hit("D#c1", "first query"), ["I-1"], Stage.S1)
    again = registry.register(hit("D#c1", "another query", 0.1), ["I-1"], Stage.S2)
    assert first.id == again.id == "E-001" and again is first
    assert len(evidence_events(store)) == 1  # nothing new: no second event
    assert corpus_.calls == 1
    other = registry.register(hit("D#c2"), ["I-1"], Stage.S1)
    assert other.id == "E-002"
    assert "E-001" in registry and "E-999" not in registry and len(registry) == 2
    assert registry.for_chunk("D#c2") == other and registry.for_chunk("D#c9") is None
    assert [e.id for e in registry] == ["E-001", "E-002"]
    with pytest.raises(KeyError):
        registry.get("E-003")


def test_a_new_issue_extends_the_evidence_under_the_same_id() -> None:
    registry, store, _ = new_registry()
    registry.register(hit("D#c1", "first query"), ["I-1"], Stage.S1)
    grown = registry.register(hit("D#c1", "later"), ["I-2", "I-1"], Stage.S3, agent="counsel_a")
    assert grown.id == "E-001" and grown.issue_ids == ["I-1", "I-2"]
    # the first registration's provenance is kept
    assert (grown.retrieved_by_query, grown.registered_at_stage) == ("first query", Stage.S1)
    events = evidence_events(store)
    assert len(events) == 2
    assert events[1].agent == "counsel_a" and events[1].stage == Stage.S3
    view = fold(store.read("run-1"))
    assert view.evidence["E-001"].issue_ids == ["I-1", "I-2"]
    assert set(view.documents) == {"D"} and set(view.chunks) == {"D#c1"}


def test_evidence_carries_scores_query_and_provenance() -> None:
    registry, store, _ = new_registry()
    evidence = registry.register(hit("D#c1", "earnest money", 0.03), ["I-1"], Stage.S1)
    assert (evidence.retrieval_score, evidence.rerank_score) == (0.03, 0.9)
    assert evidence.retrieved_by_query == "earnest money" and evidence.document_id == "D"
    assert evidence.created_by.agent == "researcher"
    payload = evidence_events(store)[0].payload
    assert payload.chunk.id == "D#c1" and payload.document.id == "D"  # type: ignore[union-attr]


def test_bad_registrations_use_up_no_id() -> None:
    registry, store, _ = new_registry()
    with pytest.raises(ValueError, match="at least one issue"):
        registry.register(hit("D#c1"), [], Stage.S1)
    with pytest.raises(KeyError):
        registry.register(hit("missing#c1"), ["I-1"], Stage.S1)
    assert registry.register(hit("D#c1"), ["I-1"], Stage.S1).id == "E-001"
    assert len(evidence_events(store)) == 1


def test_ids_are_deterministic_across_replays() -> None:
    runs = []
    for _ in range(2):
        registry, _, _ = new_registry()
        runs.append(
            [registry.register(hit(c), ["I-1"], Stage.S1).id for c in ("a#1", "b#1", "a#1", "c#1")]
        )
    assert runs[0] == runs[1] == ["E-001", "E-002", "E-001", "E-003"]


# ---- the service (database) --------------------------------------------------------------


@pytest.fixture
def service(corpus: Engine) -> RetrievalService:  # noqa: F811
    with corpus.begin() as conn:
        # judgment chunks need a paragraph or page range to become domain models
        conn.execute(
            text(
                "UPDATE chunks SET para_start = 0, para_end = 0, page_start = 1, page_end = 1 "
                "WHERE statute_section_id IS NULL"
            )
        )
        refresh_lexeme_stats(conn)
    embed_missing(corpus, FakeEncoder())
    hybrid = HybridRetriever(
        LexicalRetriever(corpus),
        DenseRetriever(corpus, FakeEncoder()),
        reranker=Reranker(corpus, OverlapEncoder()),
    )
    return RetrievalService(corpus, hybrid)


@pytest.mark.db
def test_search_returns_hits_with_text_and_provenance(service: RetrievalService) -> None:
    hits = service.search("earnest money forfeited", k=3)
    assert hits[0].chunk_id == f"{HC}#c0001"
    assert hits[0].text.startswith("The earnest money") and hits[0].title
    assert hits[0].context_header and "Delhi" in hits[0].context_header
    assert hits[0].query == "earnest money forfeited"
    assert hits[0].rerank_score is not None and hits[0].retrieval_score
    assert [h.rank for h in hits] == [1, 2, 3]
    plain = service.search("earnest money forfeited", k=3, rerank=False)
    assert all(h.rerank_score is None for h in plain)
    assert service.search(["", " "]) == []


@pytest.mark.db
def test_each_hit_names_the_query_that_found_it(service: RetrievalService) -> None:
    hits = service.search(["earnest money forfeited", "Mehta Traders frustration"], k=10)
    by_chunk = {h.chunk_id: h for h in hits}
    assert by_chunk[f"{HC}#c0001"].query == "earnest money forfeited"
    assert by_chunk[f"{SC_NEW}#c0002"].query == "Mehta Traders frustration"


@pytest.mark.db
def test_filters_pass_through(service: RetrievalService) -> None:
    from juris.retrieval.filters import SearchFilters

    hits = service.search("contract penalty", SearchFilters(date_from=dt.date(2016, 1, 1)))
    assert {h.doc_id for h in hits} <= {HC, ACT}


@pytest.mark.db
def test_lookups_and_section_hits(service: RetrievalService) -> None:
    section = service.section("Contract Act s.74")
    assert section is not None and section.section_id == "contract_act:74"
    hits = service.section_hits("s. 74 ICA")
    assert [h.chunk_id for h in hits] == [f"{ACT}#s74"]
    assert hits[0].retrieval_score is None and hits[0].query == "s. 74 ICA"
    assert service.section_hits("s. 999 ICA") == []
    cited = service.cases_citing_section("ICA", "74")
    assert [c.doc_id for c in cited] == [SC_OLD]


@pytest.mark.db
def test_records_are_valid_domain_models(service: RetrievalService) -> None:
    document, chunk = service.records(f"{SC_OLD}#c0001")
    assert document.id == SC_OLD and document.corpus_snapshot == "s"
    assert chunk.document_id == SC_OLD and chunk.statute is None
    act_doc, act_chunk = service.records(f"{ACT}#s74")
    assert act_doc.kind == "statute" and act_chunk.statute is not None
    assert act_chunk.statute.section == "74"
    with pytest.raises(KeyError):
        service.records("nope#c1")


@pytest.mark.db
def test_a_baseline_in_a_few_lines(service: RetrievalService) -> None:
    # what a single-agent RAG baseline (6.3) writes, apart from building the service
    hits = service.search("Is a penalty clause enforceable beyond reasonable compensation?", k=3)
    context = "\n\n".join(f"[{h.chunk_id}] {h.text}" for h in hits)
    assert f"[{hits[0].chunk_id}]" in context and len(hits) == 3


@pytest.mark.db
def test_registering_search_hits_emits_events_the_ui_can_fold(
    service: RetrievalService,
) -> None:
    store, emitter = started_run("r")
    registry = EvidenceRegistry(service, emitter)
    hits = service.search("penalty compensation", k=3) + service.section_hits("s. 74 ICA")
    ids = [registry.register(h, ["I-1"], Stage.S1).id for h in hits]
    assert len(set(ids)) == len({h.chunk_id for h in hits})  # the Act chunk registered once
    view = fold(store.read("r"))
    assert set(view.evidence) == set(ids)
    assert all(view.chunks[e.chunk_id].document_id == e.document_id for e in registry)
