"""The REST API (PLAN 9.1-9.3) through ``httpx.AsyncClient``: a case runs end to end on a
scripted model and a fake retrieval service, and every response validates against the exported
schemas; the event stream delivers every event once, in order, also across a reconnect; replay
keeps the recorded pace; mock mode serves the fixtures. The document route reads a fake store
here and Postgres in a ``db`` test; another ``db`` test restarts the server on its database."""

import asyncio
import datetime as dt
import json
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from juris.api.app import create_app
from juris.api.documents import DbDocuments
from juris.api.mock import FIXTURES, fixture_events, gap
from juris.api.runs import INTERRUPTED, RunManager
from juris.api.schemas import DocumentDetail, DocumentParagraph
from juris.baselines.base import RunContext
from juris.config import REPO_ROOT, ModelsConfig, ModelSpec, Settings
from juris.db import models as m
from juris.db.events import DbEventStore
from juris.eval.runner import GatewayFactory
from juris.events.catalog import CaseCreatedPayload, StageStartedPayload
from juris.events.store import EventEmitter
from juris.models import Document
from juris.models.common import Stage
from juris.pipeline.system import JurisSystem
from tests.conftest import migrate
from tests.test_b1 import HITS
from tests.test_system import FullService, Model

QUESTION = "Can the seller forfeit the earnest money when the buyer backs out?"
SCHEMAS = REPO_ROOT / "schemas"


def check(schema: str, body: Any) -> None:
    """Raise if ``body`` does not validate against ``schemas/<schema>.schema.json``."""
    loaded = json.loads((SCHEMAS / f"{schema}.schema.json").read_text(encoding="utf-8"))
    jsonschema.validators.validator_for(loaded)(loaded).validate(body)


class FakeDocuments:
    def document(self, doc_id: str) -> DocumentDetail | None:
        if doc_id != "SC-1":
            return None
        return DocumentDetail(
            document=Document(
                id="SC-1", kind="judgment", title="A v. B", licence="public",
                corpus_snapshot="snap",
            ),
            paragraphs=[DocumentParagraph(seq=0, no="1", part="body", text="Held.")],
        )  # fmt: skip


class Broken:
    corpus_snapshot_id = "snap"
    embedding_model = "none"

    async def run(self, ctx: RunContext) -> None:
        raise RuntimeError("retrieval is down")


class Gated:
    """A run that waits (in its own thread) until the test opens the gate."""

    corpus_snapshot_id = "snap"
    embedding_model = "none"

    def __init__(self) -> None:
        self.gate = threading.Event()

    async def run(self, ctx: RunContext) -> None:
        await asyncio.to_thread(self.gate.wait, 10)


@dataclass
class Api:
    client: httpx.AsyncClient
    runs: RunManager
    model: Model


def settings(tmp_path: Path, **extra: Any) -> Settings:
    spec = ModelSpec(provider="fake", name="m", effort=None)
    return Settings(  # type: ignore[call-arg]
        _env_file=None, models=ModelsConfig(default=spec), data_dir=tmp_path,
        api_sse_ping_seconds=0.05, **extra,
    )  # fmt: skip


@pytest.fixture
async def api(tmp_path: Path) -> AsyncIterator[Api]:
    model = Model()
    runs = RunManager(
        settings=settings(tmp_path),
        systems={
            "juris": lambda: JurisSystem(service=lambda: FullService(hits=list(HITS))),
            "b0": Broken,
            "b2": Gated,
        },
        gateways=GatewayFactory({"fake": model}, prices={}),
    )
    app = create_app(runs, FakeDocuments())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield Api(client, runs, model)
    await runs.close()


async def test_a_case_runs_end_to_end(api: Api, tmp_path: Path) -> None:
    client = api.client
    configs = (await client.get("/configs")).json()
    names = {c["name"] for c in configs}
    assert {"juris_full", "no_jury", "b0"} <= names and "b1" not in names and "dummy" not in names
    for config in configs:
        check("api_config_info", config)

    created = await client.post("/cases", json={"question": QUESTION, "profile": "juris_full"})
    assert created.status_code == 202
    check("api_case_created", created.json())
    case_id, run_id = created.json()["case_id"], created.json()["run_id"]
    await api.runs.wait(run_id)

    case = (await client.get(f"/cases/{case_id}")).json()
    check("api_case_info", case)
    assert case["question"] == QUESTION
    assert [(r["run_id"], r["status"], r["profile"]) for r in case["runs"]] == [
        (run_id, "completed", "juris_full")
    ]

    view = (await client.get(f"/runs/{run_id}")).json()
    check("case_view", view)
    assert view["status"] == "completed" and view["case_id"] == case_id
    assert view["question"] == QUESTION and view["profile"] == "juris_full"

    analysis = (await client.get(f"/runs/{run_id}/analysis")).json()
    check("case_analysis", analysis)
    assert analysis == view["analysis"] and analysis["created_by"]["stage"] == "S10"

    graph = (await client.get(f"/runs/{run_id}/graph")).json()
    check("evidence_graph", graph)
    assert graph == view["graph"] and graph["nodes"]

    evidence = (await client.get(f"/evidence/{run_id}/E-001")).json()
    check("api_evidence_detail", evidence)
    assert evidence["evidence"]["id"] == "E-001"
    assert evidence["chunk"]["id"] == evidence["evidence"]["chunk_id"]
    assert evidence["document"]["id"] == evidence["evidence"]["document_id"]
    [citation] = evidence["citations"]  # counsel A cites E-001 once
    assert citation["verification"]["status"] == "verified"

    # the finished run is also on disk, like a run_case.py run
    out = tmp_path / "runs" / "api" / run_id
    assert {p.name for p in out.iterdir()} == {
        "manifest.json", "events.jsonl", "caseview.json", "calls.json",
    }  # fmt: skip
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["status"] == (
        "completed"
    )
    assert len(api.model.calls) > 10


async def test_a_failed_run_is_reported(api: Api) -> None:
    created = await api.client.post("/cases", json={"question": QUESTION, "profile": "b0"})
    run_id = created.json()["run_id"]
    await api.runs.wait(run_id)
    view = (await api.client.get(f"/runs/{run_id}")).json()
    check("case_view", view)
    assert view["status"] == "failed" and "retrieval is down" in view["error"]
    case = (await api.client.get(f"/cases/{created.json()['case_id']}")).json()
    assert case["runs"][0]["status"] == "failed"
    assert (await api.client.get(f"/runs/{run_id}/analysis")).status_code == 404


async def test_bad_requests_and_unknown_ids(api: Api) -> None:
    client = api.client
    unknown = await client.post("/cases", json={"question": QUESTION, "profile": "b1"})
    assert unknown.status_code == 422 and "b1" in unknown.json()["detail"]
    assert (await client.post("/cases", json={"question": "Short?"})).status_code == 422
    assert (await client.post("/cases", json={"question": QUESTION, "x": 1})).status_code == 422
    for path in ("/cases/CASE-0", "/runs/r-0", "/runs/r-0/analysis", "/runs/r-0/graph",
                 "/evidence/r-0/E-001", "/documents/SC-404"):  # fmt: skip
        response = await client.get(path)
        assert response.status_code == 404, path
        assert "not found" in response.json()["detail"]
    created = (await client.post("/cases", json={"question": QUESTION})).json()  # juris_full
    await api.runs.wait(created["run_id"])
    missing = await client.get(f"/evidence/{created['run_id']}/E-999")
    assert missing.status_code == 404


async def test_documents_and_cors(api: Api) -> None:
    document = await api.client.get("/documents/SC-1")
    check("api_document_detail", document.json())
    assert document.json()["paragraphs"][0]["text"] == "Held."
    preflight = await api.client.options(
        "/cases",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"},
    )
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    other = await api.client.get("/configs", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_the_default_profiles_are_every_runnable_one(tmp_path: Path) -> None:
    names = {p.name for p in RunManager(settings=settings(tmp_path)).profiles()}
    assert {"juris_full", "b0", "b1", "b2", "no_verifier_gates"} <= names
    assert "dummy" not in names


@pytest.mark.db
def test_documents_from_the_database(engine: Engine) -> None:
    with engine.begin() as conn:
        migrate(conn)
    with Session(engine) as session:
        session.add(m.Snapshot(snapshot_id="s", slice_name="x", counts={}, notes=[]))
        session.flush()
        session.add_all([
            m.Document(doc_id="SC-1", snapshot_id="s", kind="judgment", source="t",
                       licence="public", title="A v. B", headnote="On forfeiture.",
                       decision_date=dt.date(1963, 1, 1), citations=["AIR 1963 SC 1405"]),
            m.Document(doc_id="ACT-x", snapshot_id="s", kind="statute", source="t",
                       licence="public", title="An Act"),
        ])  # fmt: skip
        session.flush()
        session.add_all([
            m.Paragraph(doc_id="SC-1", seq=s, no=str(s + 1), part=1, section="body",
                        numbering="explicit", char_start=s * 10, char_end=s * 10 + 9,
                        page_start=1, page_end=1, text=t)
            for s, t in ((1, "Second."), (0, "First."), (2, " "))
        ])  # fmt: skip
        session.add_all([
            m.StatuteSection(section_id=f"x:{n}", doc_id="ACT-x", act_id="x", act="An Act",
                             act_year=1872, section=str(n), title=f"Title {n}", text=f"{n}. Text",
                             repealed=False, sub_sections=[], amendment_notes=[], amended_by=[],
                             amendments_curated=False, char_start=100 - n, char_end=110 - n,
                             page_start=1, page_end=1)
            for n in (74, 73)
        ])  # fmt: skip
        session.commit()
    store = DbDocuments(engine)
    judgment = store.document("SC-1")
    assert judgment is not None and judgment.headnote == "On forfeiture."
    assert judgment.document.citations == ["AIR 1963 SC 1405"]
    assert [(p.seq, p.text) for p in judgment.paragraphs] == [(0, "First."), (1, "Second.")]
    statute = store.document("ACT-x")
    assert statute is not None
    sections = [(p.no, p.part, p.title) for p in statute.paragraphs]
    assert sections == [("74", "section", "Title 74"), ("73", "section", "Title 73")]  # Act order
    assert store.document("SC-404") is None
    check("api_document_detail", judgment.model_dump(mode="json"))


# ---- the event stream (PLAN 9.2) ---------------------------------------------------------


def blocks(body: str) -> list[dict[str, str]]:
    """The stream's messages: each a dict of its fields; a comment is ``{":": text}``."""
    out = []
    for block in body.split("\n\n"):
        if not block:
            continue
        fields: dict[str, str] = {}
        for line in block.split("\n"):
            name, _, value = line.partition(":")
            fields[name or ":"] = value.removeprefix(" ")
        out.append(fields)
    return out


def sent(body: str) -> list[dict[str, Any]]:
    """The events in a stream's body (pings and control messages left out)."""
    return [json.loads(b["data"]) for b in blocks(body) if "id" in b]


async def test_the_stream_sends_every_event_in_order(api: Api) -> None:
    created = (await api.client.post("/cases", json={"question": QUESTION})).json()
    run_id = created["run_id"]
    response = await api.client.get(f"/runs/{run_id}/events")  # from the start, to the end
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    messages = blocks(response.text)
    assert messages[0] == {"retry": "3000"}
    assert messages[-1] == {"event": "end", "data": "{}"}
    events = sent(response.text)
    seqs = [e["seq"] for e in events]
    assert seqs == list(range(1, len(events) + 1)) and len(events) > 50
    assert [int(b["id"]) for b in messages if "id" in b] == seqs
    stored = [e.model_dump(mode="json") for e in api.runs.store.read(run_id)]
    assert events == stored  # everything the run stored, as stored
    assert events[0]["type"] == "case_created" and events[-1]["type"] == "run_completed"
    for event in events:
        check("event", event)


async def test_reconnecting_resumes_without_gaps_or_duplicates(api: Api) -> None:
    client = api.client
    run_id = (await client.post("/cases", json={"question": QUESTION})).json()["run_id"]
    full = sent((await client.get(f"/runs/{run_id}/events")).text)
    half = len(full) // 2
    resumed = await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(half)})
    assert full[:half] + sent(resumed.text) == full
    by_query = await client.get(f"/runs/{run_id}/events", params={"after": half})
    assert sent(by_query.text) == full[half:]
    # once the client has the last event there is nothing to resume: EventSource stops
    done = await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(full))})
    assert done.status_code == 204 and done.text == ""
    bad = await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "evt_3"})
    assert bad.status_code == 400
    assert (await client.get("/runs/r-0/events")).status_code == 404


async def test_a_live_stream_pings_while_quiet_and_ends_with_the_run(api: Api) -> None:
    run_id = (await api.client.post("/cases", json={"question": QUESTION, "profile": "b2"})).json()[
        "run_id"
    ]
    listening = asyncio.create_task(api.client.get(f"/runs/{run_id}/events"))
    await asyncio.sleep(0.4)  # the run waits at its gate: only case_created so far
    assert not listening.done()
    gated = api.runs.system("b2")
    assert isinstance(gated, Gated)
    gated.gate.set()
    response = await asyncio.wait_for(listening, 10)
    messages = blocks(response.text)
    kinds = [
        json.loads(m["data"])["type"] if "id" in m else m.get("event") or m.get(":") or "retry"
        for m in messages
    ]
    assert kinds[:2] == ["retry", "case_created"]
    assert kinds[-2:] == ["run_completed", "end"]
    pings = kinds[2:-2]
    assert len(pings) >= 2 and set(pings) == {"ping"}  # nothing else happened meanwhile


async def test_a_wake_up_with_nothing_new_does_not_end_the_wait(api: Api) -> None:
    """A wake scheduled for an event the stream already has must not count as quiet (it
    would send a ping early)."""
    run_id = (await api.client.post("/cases", json={"question": QUESTION, "profile": "b2"})).json()[
        "run_id"
    ]
    last = api.runs.store.last_seq(run_id)
    waiting = asyncio.create_task(api.runs.wait_events(run_id, last, 0.5))
    await asyncio.sleep(0.05)
    api.runs._wake(run_id)  # stale: no new event
    await asyncio.sleep(0.2)
    assert not waiting.done()
    assert await waiting == []  # at the timeout
    gated = api.runs.system("b2")
    assert isinstance(gated, Gated)
    gated.gate.set()
    [completed] = await api.runs.wait_events(run_id, last, 5)
    assert completed.type == "run_completed"


# ---- persistence, replay and mock mode (PLAN 9.3) -----------------------------------------


async def test_replay_restreams_a_finished_run_at_its_pace(tmp_path: Path) -> None:
    app = create_app(settings=settings(tmp_path, api_mode="mock"))
    runs: RunManager = app.state.runs
    stored = runs.store.read("run_mock_failure")
    expected = sum(gap(a, b, 50, 0.1) for a, b in pairwise(stored))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        started = time.monotonic()
        response = await client.get(
            "/runs/run_mock_failure/events", params={"replay": True, "speed": 50, "max_gap": 0.1}
        )
        took = time.monotonic() - started
        assert sent(response.text) == [e.model_dump(mode="json") for e in stored]
        messages = blocks(response.text)
        assert messages[0] == {"retry": "3000"} and messages[-1] == {"event": "end", "data": "{}"}
        assert {":": "ping"} in messages  # gaps of 0.1 s are longer than the 0.05 s ping
        assert expected > 0.5 and expected * 0.9 <= took <= expected + 2
        resumed = await client.get(
            "/runs/run_mock_failure/events",
            params={"replay": True, "speed": 1000},
            headers={"Last-Event-ID": "20"},
        )
        assert [e["seq"] for e in sent(resumed.text)] == list(range(21, len(stored) + 1))
        done = await client.get(
            "/runs/run_mock_failure/events",
            params={"replay": True},
            headers={"Last-Event-ID": str(len(stored))},
        )
        assert done.status_code == 204
    await runs.close()


async def test_a_run_still_going_cannot_be_replayed(api: Api) -> None:
    created = await api.client.post("/cases", json={"question": QUESTION, "profile": "b2"})
    run_id = created.json()["run_id"]
    response = await api.client.get(f"/runs/{run_id}/events", params={"replay": True})
    assert response.status_code == 409 and "has not finished" in response.json()["detail"]
    gated = api.runs.system("b2")
    assert isinstance(gated, Gated)
    gated.gate.set()


async def test_mock_mode_serves_the_fixtures_through_the_same_routes(tmp_path: Path) -> None:
    app = create_app(settings=settings(tmp_path, api_mode="mock", api_mock_speed=2000))
    runs: RunManager = app.state.runs
    fixture = fixture_events("coercion_full")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # the fixtures' own runs, finished
        case = (await client.get("/cases/CASE-MOCK-001")).json()
        check("api_case_info", case)
        assert [(r["run_id"], r["status"]) for r in case["runs"]] == [
            ("run_mock_coercion_full", "completed")
        ]
        failed = (await client.get("/runs/run_mock_failure")).json()
        check("case_view", failed)
        assert failed["status"] == "failed"
        view = (await client.get("/runs/run_mock_coercion_full")).json()
        expected = json.loads((FIXTURES / "coercion_full.caseview.json").read_text("utf-8"))
        assert view == expected
        check("case_analysis", (await client.get("/runs/run_mock_coercion_full/analysis")).json())
        evidence_id = next(iter(view["evidence"]))
        detail = await client.get(f"/evidence/run_mock_coercion_full/{evidence_id}")
        check("api_evidence_detail", detail.json())
        doc_id = detail.json()["document"]["id"]
        document = (await client.get(f"/documents/{doc_id}")).json()
        check("api_document_detail", document)
        assert document["document"]["title"].startswith("[MOCK]") and document["paragraphs"]
        assert (await client.get("/documents/SC-1")).status_code == 404

        # a new case replays the coercion run's proceedings, live
        created = await client.post("/cases", json={"question": QUESTION, "profile": "b1"})
        run_id = created.json()["run_id"]
        events = sent((await client.get(f"/runs/{run_id}/events")).text)
        for event in events:
            check("event", event)
        assert [e["type"] for e in events] == [e.type for e in fixture]
        assert [e["payload"] for e in events[1:-1]] == [
            e.payload.model_dump(mode="json") for e in fixture[1:-1]
        ]
        assert events[0]["payload"]["question"] == QUESTION
        live = (await client.get(f"/runs/{run_id}")).json()
        assert live["status"] == "completed" and live["analysis"] == view["analysis"]
    await runs.close()


@pytest.mark.db
async def test_runs_outlive_the_server(engine: Engine, tmp_path: Path) -> None:
    """A finished run, and one cut off by a shutdown, read back by a new server on the same
    database: the same events, view and case; the cut-off run is closed as failed."""
    with engine.begin() as conn:
        migrate(conn)

    def server() -> RunManager:
        return RunManager(
            settings=settings(tmp_path),
            systems={"juris": lambda: JurisSystem(service=lambda: FullService(hits=list(HITS)))},
            gateways=GatewayFactory({"fake": Model()}, prices={}),
            store=DbEventStore(engine),
            db_url=engine.url.set(drivername="postgresql").render_as_string(hide_password=False),
        )

    first = server()
    created = first.submit(QUESTION, "juris_full")
    await first.wait(created.run_id)
    before = [e.model_dump(mode="json") for e in first.store.read(created.run_id)]
    view_before = first.view(created.run_id)
    cut = EventEmitter(first.store, run_id="run-cut", case_id="CASE-cut")  # died mid-run
    cut.emit(CaseCreatedPayload(question=QUESTION, profile="b2", corpus_snapshot_id="snap"))
    cut.emit(StageStartedPayload(), stage=Stage.S0)
    await first.close()

    second = server()
    app = create_app(second, FakeDocuments())  # restores what the store holds
    assert second.restore() == 0  # once only
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        case = (await client.get(f"/cases/{created.case_id}")).json()
        assert [(r["run_id"], r["status"]) for r in case["runs"]] == [(created.run_id, "completed")]
        assert case["question"] == QUESTION
        view = (await client.get(f"/runs/{created.run_id}")).json()
        assert view_before is not None and view == view_before.model_dump(mode="json")
        stream = await client.get(f"/runs/{created.run_id}/events")
        assert sent(stream.text) == before
        replayed = await client.get(
            f"/runs/{created.run_id}/events", params={"replay": True, "speed": 1000}
        )
        assert sent(replayed.text) == before

        cut_case = (await client.get("/cases/CASE-cut")).json()
        assert [(r["run_id"], r["status"]) for r in cut_case["runs"]] == [("run-cut", "failed")]
        cut_view = (await client.get("/runs/run-cut")).json()
        assert cut_view["status"] == "failed" and cut_view["error"] == INTERRUPTED
        cut_events = sent((await client.get("/runs/run-cut/events")).text)
        assert [e["type"] for e in cut_events] == ["case_created", "stage_started", "run_failed"]
    await second.close()
    with engine.connect() as conn:
        status = conn.execute(
            text("SELECT status FROM runs WHERE run_id = :r"), {"r": created.run_id}
        ).scalar()
    assert status == "completed"  # the manifest too
