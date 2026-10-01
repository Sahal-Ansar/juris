"""The per-run evidence registry: the only way retrieved text enters a run (PLAN 4.6, D-033).

IDEA_final §8, invariant 1: a citation must reference a registered ``Evidence`` ID. Agents get
hits from the retrieval service and register the ones they use; the Case Record then accepts
citations only to IDs this registry issued.

- IDs are ``E-001``, ``E-002`` ... in registration order (``IdAllocator``), so a replayed run
  gets the same IDs.
- One chunk is one piece of evidence: registering it again returns the same ``Evidence``.
  If the new registration names issues the chunk wasn't registered for, the issue list grows
  and the updated evidence is emitted again under the same ID. Otherwise nothing is emitted.
  The first registration's scores, query and stage are kept.
- Every new or changed registration emits ``evidence_registered`` with the evidence, its
  document and its chunk, so the UI never needs a second lookup.
"""

from collections.abc import Iterator, Sequence
from typing import Protocol

from juris.events import EventEmitter
from juris.events.catalog import EvidenceRegisteredPayload
from juris.models import Chunk, CreatedBy, Document, Evidence, IdAllocator, Stage


class Registrable(Protocol):
    """What registration needs from a hit (``juris.retrieval.service.Hit`` has it)."""

    @property
    def chunk_id(self) -> str: ...
    @property
    def query(self) -> str: ...
    @property
    def retrieval_score(self) -> float | None: ...
    @property
    def rerank_score(self) -> float | None: ...


class CorpusRecords(Protocol):
    def records(self, chunk_id: str) -> tuple[Document, Chunk]: ...


class EvidenceRegistry:
    def __init__(
        self, corpus: CorpusRecords, emitter: EventEmitter, ids: IdAllocator | None = None
    ) -> None:
        self.corpus = corpus
        self.emitter = emitter
        self.ids = ids or IdAllocator()
        self._by_id: dict[str, Evidence] = {}
        self._by_chunk: dict[str, str] = {}
        self._records: dict[str, tuple[Document, Chunk]] = {}

    def register(
        self,
        hit: Registrable,
        issue_ids: Sequence[str],
        stage: Stage,
        agent: str = "researcher",
    ) -> Evidence:
        """The evidence for ``hit``'s chunk, registering it (and emitting the event) if new."""
        issues = list(dict.fromkeys(issue_ids))
        if not issues:
            raise ValueError("evidence must be registered for at least one issue")
        existing_id = self._by_chunk.get(hit.chunk_id)
        if existing_id is not None:
            existing = self._by_id[existing_id]
            merged = list(dict.fromkeys([*existing.issue_ids, *issues]))
            if merged == existing.issue_ids:
                return existing
            evidence = Evidence.model_validate({**existing.model_dump(), "issue_ids": merged})
        else:
            # load the records first: an unknown chunk raises before an ID is used up
            self._records[hit.chunk_id] = self.corpus.records(hit.chunk_id)
            evidence = Evidence(
                id=self.ids.next("E"),
                chunk_id=hit.chunk_id,
                document_id=self._records[hit.chunk_id][0].id,
                issue_ids=issues,
                retrieval_score=hit.retrieval_score,
                rerank_score=hit.rerank_score,
                retrieved_by_query=hit.query,
                registered_at_stage=stage,
                created_by=CreatedBy(stage=stage, agent=agent),
            )
        document, chunk = self._records[hit.chunk_id]
        self.emitter.emit(
            EvidenceRegisteredPayload(evidence=evidence, document=document, chunk=chunk),
            stage=stage,
            agent=agent,
        )
        self._by_id[evidence.id] = evidence
        self._by_chunk[hit.chunk_id] = evidence.id
        return evidence

    def get(self, evidence_id: str) -> Evidence:
        """A registered evidence item; KeyError for an ID this run never issued."""
        return self._by_id[evidence_id]

    def chunk(self, evidence_id: str) -> Chunk:
        """The chunk a registered evidence item points at; KeyError for an unknown ID."""
        return self._records[self._by_id[evidence_id].chunk_id][1]

    def chunk_text(self, evidence_id: str) -> str | None:
        """The chunk's text, or None for an ID this run never issued."""
        return self.chunk(evidence_id).text if evidence_id in self._by_id else None

    def for_chunk(self, chunk_id: str) -> Evidence | None:
        evidence_id = self._by_chunk.get(chunk_id)
        return self._by_id[evidence_id] if evidence_id else None

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[Evidence]:
        """Evidence in registration order."""
        return iter(self._by_id.values())
