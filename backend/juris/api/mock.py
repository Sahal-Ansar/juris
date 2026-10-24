"""Mock mode (PLAN 9.3, ``JURIS_API_MODE=mock``): the 2.4 fixtures through the real endpoints,
with no model, retrieval or database.

- The fixtures' two runs are in the store from the start, as finished runs:
  ``run_mock_coercion_full`` (completed) and ``run_mock_failure`` (failed on its budget).
- ``POST /cases`` runs ``MockSystem``: the coercion run's proceedings again, as a new run of
  the question asked, paced by the fixture's timestamps (``JURIS_API_MOCK_SPEED`` times faster,
  no gap longer than ``MAX_GAP`` seconds), so the stream behaves as a live one.
- ``GET /documents/{id}`` serves the fixtures' mock documents, their chunks as paragraphs.

Every mock document is marked ``[MOCK]`` with a ``mock://`` source (2.4), so nothing served in
this mode passes for a real authority.
"""

import asyncio
from pathlib import Path
from typing import Literal

from juris.api.schemas import DocumentDetail, DocumentParagraph
from juris.baselines.base import RunContext
from juris.config import REPO_ROOT
from juris.events.catalog import Event, parse_event
from juris.events.fold import CaseView, fold
from juris.events.store import EventStore

FIXTURES = REPO_ROOT / "fixtures" / "runs"
RUNS = ("coercion_full", "failure_case")
MAX_GAP = 2.0  # seconds: the fixture's model calls take minutes
SNAPSHOT = "mock-snapshot-2026-09"


def fixture_events(name: str, fixtures: Path = FIXTURES) -> list[Event]:
    lines = (fixtures / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [parse_event(line) for line in lines if line.strip()]


def gap(previous: Event, event: Event, speed: float, max_gap: float) -> float:
    """Seconds to wait before ``event``: its distance from ``previous``, sped up and capped."""
    return min(max((event.ts - previous.ts).total_seconds(), 0.0) / speed, max_gap)


def load_fixtures[S: EventStore](store: S, fixtures: Path = FIXTURES) -> S:
    """Store the fixtures' runs (as recorded: their own IDs and timestamps)."""
    for name in RUNS:
        for event in fixture_events(name, fixtures):
            store.append(event)
    return store


class MockSystem:
    """The coercion fixture's proceedings, re-emitted as this run's. The run manager emits
    ``case_created`` and the end of the run, so the fixture's own are left out."""

    corpus_snapshot_id = SNAPSHOT
    embedding_model = "mock"

    def __init__(self, speed: float = 10.0, fixture: str = "coercion_full") -> None:
        self.speed = speed
        self.events = fixture_events(fixture)

    async def run(self, ctx: RunContext) -> None:
        previous = self.events[0]
        for event in self.events[1:]:
            if event.type in {"run_completed", "run_failed"}:
                break
            await asyncio.sleep(gap(previous, event, self.speed, MAX_GAP))
            ctx.emitter.emit(event.payload, stage=event.stage, agent=event.agent)
            previous = event


def detail(view: CaseView, doc_id: str) -> DocumentDetail:
    """A mock document with its chunks standing in for its paragraphs (the fixtures carry
    no full text), in document order."""
    chunks = sorted(
        (c for c in view.chunks.values() if c.document_id == doc_id), key=lambda c: c.char_start
    )
    paragraphs = []
    part: Literal["body", "section"]
    for n, chunk in enumerate(chunks):
        if chunk.statute is not None:
            no, part = chunk.statute.section, "section"
        else:
            start, end = chunk.para_start, chunk.para_end
            no = f"p-{n + 1}" if start is None else str(start + 1)
            if start is not None and end is not None and end > start:
                no = f"{start + 1}-{end + 1}"
            part = "body"
        paragraphs.append(
            DocumentParagraph(
                seq=chunk.para_start if chunk.para_start is not None else n,
                no=no,
                part=part,
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
        )
    return DocumentDetail(document=view.documents[doc_id], paragraphs=paragraphs)


class FixtureDocuments:
    """The documents of the fixtures' runs (``GET /documents/{id}`` in mock mode)."""

    def __init__(self, fixtures: Path = FIXTURES) -> None:
        self._docs: dict[str, DocumentDetail] = {}
        for name in RUNS:
            view = fold(fixture_events(name, fixtures))
            for doc_id in view.documents:
                self._docs.setdefault(doc_id, detail(view, doc_id))

    def document(self, doc_id: str) -> DocumentDetail | None:
        return self._docs.get(doc_id)
