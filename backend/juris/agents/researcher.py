"""The Researcher (S1, and S5's targeted searches; IDEA_final §6.1, PLAN 7.4): builds each
issue's evidence pool from the Retrieval Service and registers it in the Case Record.

It is code, not a model. It never says which side an item supports. Per issue:

1. **Search.** S0's search queries for the issue (the issue's question if there are none) go
   to hybrid search together (RRF over all queries, D-037), for ``candidate_k`` candidates.
2. **Provisions, as hints.** S0's statute references are often wrong on a small model (D-050:
   s. 146 for s. 16). A referenced section is confirmed when the issue's search found one of its
   chunks, or when at least ``cite_min`` of the candidate judgments cite it (the citation
   graph, PLAN 3.9). On dev, the graph test kept all 6 correct references of 33 and 4 wrong
   ones; the search test alone kept 1 correct and 2 wrong (D-051). A confirmed section's chunks
   (by exact lookup) go first; the others are counted as unconfirmed and left out.
3. **Select.** In rank order, up to ``max_items``, at most ``per_document`` chunks from one
   judgment, so one long judgment cannot flood the pool. Statute chunks are not capped: each is
   a different provision. If the cap leaves fewer than ``min_items``, the cap is relaxed to fill
   up to ``min_items``.
4. **Register** each selected chunk for the issue. A chunk already registered for another issue
   gains this issue too (same ``E-`` ID).

``targeted_search`` serves S5: one query from counsel, up to ``k`` chunks the run does not
already hold, registered before counsel may cite them.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from juris.models import Chunk, Document, Evidence
from juris.models.common import Stage
from juris.pipeline.case_record import CaseRecord
from juris.retrieval.lookup import LinkedCase, parse_provision
from juris.retrieval.service import Hit

AGENT = "researcher"


class ResearchService(Protocol):
    """What the Researcher needs from retrieval (``RetrievalService``; tests pass a fake)."""

    def search(self, queries: Sequence[str], *, k: int = 10, rerank: bool = True) -> list[Hit]: ...

    def section_hits(self, reference: str) -> list[Hit]: ...

    def cases_citing_section(self, act: str, section: str) -> list[LinkedCase]: ...

    def records(self, chunk_id: str) -> tuple[Document, Chunk]: ...


@dataclass
class IssuePool:
    """What S1 did for one issue (recorded with the run's details)."""

    issue_id: str
    queries: list[str]
    candidates: int = 0
    evidence: list[str] = field(default_factory=list)  # E- IDs, in selection order
    provisions_kept: list[str] = field(default_factory=list)
    provisions_unconfirmed: list[str] = field(default_factory=list)


class Researcher:
    def __init__(
        self,
        service: ResearchService,
        *,
        candidate_k: int = 60,
        per_document: int = 3,
        cite_min: int = 3,
        rerank: bool = False,  # D-037: no consistent gain, seconds per query
        name: str = AGENT,
    ) -> None:
        self.service = service
        self.candidate_k = candidate_k
        self.per_document = per_document
        self.cite_min = cite_min
        self.rerank = rerank
        self.name = name

    def research(
        self, record: CaseRecord, *, stage: Stage, min_items: int, max_items: int
    ) -> list[IssuePool]:
        """Build and register every issue's pool (S1)."""
        view = record.view
        pools: list[IssuePool] = []
        for issue in view.issues.values():
            mine = [q for q in view.queries if issue.id in q.issue_ids]
            queries = [q.query for q in mine if q.kind == "search"] or [issue.question]
            pool = IssuePool(issue.id, queries)
            candidates = self.service.search(queries, k=self.candidate_k, rerank=self.rerank)
            pool.candidates = len(candidates)
            first: dict[str, Hit] = {}
            for reference in (q.query for q in mine if q.kind == "provision"):
                section = self._confirmed(reference, candidates)
                if section:
                    first |= {h.chunk_id: h for h in section if h.chunk_id not in first}
                    pool.provisions_kept.append(reference)
                else:
                    pool.provisions_unconfirmed.append(reference)
            ordered = list(first.values())
            ordered += [h for h in candidates if h.chunk_id not in first]
            for hit in select(ordered, max_items, min_items, self.per_document):
                evidence = record.register_evidence(hit, [issue.id], stage=stage, agent=self.name)
                pool.evidence.append(evidence.id)
            pools.append(pool)
        return pools

    def _confirmed(self, reference: str, candidates: Sequence[Hit]) -> list[Hit]:
        """The section's chunks if the issue's candidates confirm the reference, else []."""
        provision = parse_provision(reference)
        section = self.service.section_hits(reference) if provision else []
        if not section:
            return []
        found = {h.chunk_id: h for h in candidates}
        if any(h.chunk_id in found for h in section):
            return [found.get(h.chunk_id, h) for h in section]  # keep the search's scores
        assert provision is not None
        citing = {
            c.doc_id
            for c in self.service.cases_citing_section(provision.act_id, provision.section)
            if c.doc_id
        }
        judgments = {h.doc_id for h in candidates if _capped(h)}
        return section if len(citing & judgments) >= self.cite_min else []

    def targeted_search(
        self,
        record: CaseRecord,
        query: str,
        issue_ids: Sequence[str],
        *,
        stage: Stage,
        agent: str,
        k: int = 5,
    ) -> list[Evidence]:
        """S5: ``agent``'s query, up to ``k`` new chunks registered for ``issue_ids``. The
        query is logged as ``query_issued`` under the asking agent."""
        record.add_query(query, issue_ids, stage=stage, agent=agent)
        hits = self.service.search([query], k=self.candidate_k, rerank=self.rerank)
        held = Counter(e.document_id for e in record.view.evidence.values())
        new = [h for h in hits if record.registry.for_chunk(h.chunk_id) is None]
        out: list[Evidence] = []
        for hit in select(new, k, 0, self.per_document, held):
            out.append(record.register_evidence(hit, issue_ids, stage=stage, agent=self.name))
        return out


def select(
    hits: Sequence[Hit],
    most: int,
    least: int,
    per_document: int,
    held: Counter[str] | None = None,
) -> list[Hit]:
    """Up to ``most`` hits in order, at most ``per_document`` per judgment (counting ``held``,
    the chunks already taken from each document); then, if fewer than ``least``, the next ones
    regardless of the cap."""
    taken: Counter[str] = Counter(held or {})
    chosen: list[Hit] = []
    for hit in hits:
        if len(chosen) >= most:
            break
        if _capped(hit) and taken[hit.doc_id] >= per_document:
            continue
        taken[hit.doc_id] += 1
        chosen.append(hit)
    if len(chosen) < least:
        ids = {h.chunk_id for h in chosen}
        rest = [h for h in hits if h.chunk_id not in ids]
        chosen += rest[: min(least, most) - len(chosen)]
        order = {h.chunk_id: n for n, h in enumerate(hits)}
        chosen.sort(key=lambda h: order[h.chunk_id])
    return chosen


def _capped(hit: Hit) -> bool:
    """Judgments are capped per document; a statute's chunks are separate provisions."""
    return not hit.doc_id.startswith("ACT-")
