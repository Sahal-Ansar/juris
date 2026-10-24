"""The REST API (PLAN 9.1-9.2, IDEA_final §12): the UI's way in.

    uv run uvicorn --factory juris.api.app:create_app --port 8000

- ``POST /cases``: a question and a profile → ``case_id``, ``run_id``; the run is queued.
- ``GET /cases/{case_id}``: the question and its runs with their status.
- ``GET /runs/{run_id}``: the ``CaseView`` so far (a fold of the run's events).
- ``GET /runs/{run_id}/analysis``: the ``CaseAnalysis``, once written (404 before).
- ``GET /runs/{run_id}/graph``: the evidence graph (PLAN 8.1).
- ``GET /runs/{run_id}/events``: the run's events as Server-Sent Events, live, resumable
  with ``Last-Event-ID`` (``juris.api.stream``, PLAN 9.2).
- ``GET /evidence/{run_id}/{evidence_id}``: evidence, its chunk and document, who cites it.
- ``GET /documents/{doc_id}``: a corpus document with its paragraphs.
- ``GET /configs``: the profiles ``POST /cases`` accepts.

Every response is a schema-exported type (``scripts/export_schemas.py``); the OpenAPI
document is exported with them (``schemas/openapi.json``). CORS allows the local frontend
(``JURIS_API_CORS_ORIGINS``).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from juris.api import stream
from juris.api.documents import DbDocuments, DocumentStore
from juris.api.runs import RunManager, UnknownProfile
from juris.api.schemas import (
    CaseCreated,
    CaseInfo,
    CaseRequest,
    ConfigInfo,
    DocumentDetail,
    EvidenceCitation,
    EvidenceDetail,
)
from juris.config import Settings, get_settings
from juris.eval.metrics.common import pair_results
from juris.events.fold import CaseView
from juris.models import CaseAnalysis
from juris.models.graph import EvidenceGraph

NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"description": "No such case, run or object"}}


def _not_found(what: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"{what} not found")


def create_app(
    runs: RunManager | None = None,
    documents: DocumentStore | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """The app over a run manager and a document store (by default from the settings)."""
    settings = settings or (runs.settings if runs is not None else get_settings())
    manager = runs or RunManager(settings=settings, concurrency=settings.api_concurrency)
    store = documents or DbDocuments()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await manager.close()

    app = FastAPI(
        title="Juris",
        version="0.1.0",
        summary="Evidence-grounded multi-agent legal case deliberation (Indian contract law)",
        lifespan=lifespan,
    )
    app.state.runs = manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def view_of(run_id: str) -> CaseView:
        view = manager.view(run_id)
        if view is None:
            raise _not_found(f"run {run_id}")
        return view

    @app.post("/cases", status_code=status.HTTP_202_ACCEPTED, response_model=CaseCreated)
    async def create_case(body: CaseRequest) -> CaseCreated:
        """Open a case and queue a run of it; follow it at ``GET /runs/{run_id}``."""
        try:
            return manager.submit(body.question, body.profile)
        except UnknownProfile as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    @app.get("/cases/{case_id}", response_model=CaseInfo, responses=NOT_FOUND)
    async def get_case(case_id: str) -> CaseInfo:
        case = manager.case(case_id)
        if case is None:
            raise _not_found(f"case {case_id}")
        return case

    @app.get("/runs/{run_id}", response_model=CaseView, responses=NOT_FOUND)
    async def get_run(run_id: str) -> CaseView:
        return view_of(run_id)

    @app.get("/runs/{run_id}/analysis", response_model=CaseAnalysis, responses=NOT_FOUND)
    async def get_analysis(run_id: str) -> CaseAnalysis:
        analysis = view_of(run_id).analysis
        if analysis is None:
            raise _not_found(f"the analysis of run {run_id}")
        return analysis

    @app.get("/runs/{run_id}/graph", response_model=EvidenceGraph, responses=NOT_FOUND)
    async def get_graph(run_id: str) -> EvidenceGraph:
        return view_of(run_id).graph

    @app.get(
        "/runs/{run_id}/events",
        response_class=StreamingResponse,
        responses={
            200: {
                "description": "The run's events in seq order, as Server-Sent Events: "
                "`id` is the seq, `data` an Event envelope; `: ping` comments while quiet; "
                "`event: end` after run_completed or run_failed",
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
            },
            204: {"description": "The run has ended and the client has every event"},
            **NOT_FOUND,
        },
    )
    async def get_events(
        run_id: str,
        after: int | None = Query(None, ge=0, description="Resume after this seq"),
        last_event_id: str | None = Header(None, description="Sent by EventSource on reconnect"),
    ) -> Response:
        """Stream the run's events, from the start or after ``Last-Event-ID`` (or ``after``)."""
        view_of(run_id)  # 404 for an unknown run
        after_seq = after or 0
        if last_event_id is not None:
            if not last_event_id.strip().isdigit():
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Last-Event-ID must be a seq")
            after_seq = int(last_event_id)
        if stream.ended(manager, run_id, after_seq):
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        return StreamingResponse(
            stream.events(manager, run_id, after_seq, settings.api_sse_ping_seconds),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/evidence/{run_id}/{evidence_id}", response_model=EvidenceDetail, responses=NOT_FOUND)
    async def get_evidence(run_id: str, evidence_id: str) -> EvidenceDetail:
        view = view_of(run_id)
        evidence = view.evidence.get(evidence_id)
        if evidence is None:
            raise _not_found(f"evidence {evidence_id} in run {run_id}")
        results = pair_results(view)
        citations = [
            EvidenceCitation(
                claim_id=c.claim_id,
                argument_id=argument.id,
                pinpoint=c.pinpoint,
                quote=c.quote,
                verification=results.get((c.claim_id, c.evidence_id)),
            )
            for argument in view.arguments.values()
            for c in argument.citations
            if c.evidence_id == evidence_id
        ]
        return EvidenceDetail(
            run_id=run_id,
            evidence=evidence,
            document=view.documents[evidence.document_id],
            chunk=view.chunks[evidence.chunk_id],
            citations=citations,
        )

    @app.get("/documents/{doc_id}", response_model=DocumentDetail, responses=NOT_FOUND)
    async def get_document(doc_id: str) -> DocumentDetail:
        document = store.document(doc_id)
        if document is None:
            raise _not_found(f"document {doc_id}")
        return document

    @app.get("/configs", response_model=list[ConfigInfo])
    async def get_configs() -> list[ConfigInfo]:
        return [
            ConfigInfo(
                name=p.name,
                kind=p.kind,
                description=p.description,
                stages=p.stages.model_dump(mode="json"),
            )
            for p in manager.profiles()
        ]

    return app
