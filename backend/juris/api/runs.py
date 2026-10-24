"""Cases and runs behind the API (PLAN 9.1): submit a question, run it in the background, read
it back from its events.

``submit`` opens the run at once (``case_created`` is its first event, so ``GET /runs/{id}``
answers straight away) and queues it. A few workers (``concurrency``, default 1: the local
model has one GPU) take runs off an asyncio queue and execute each in a thread with its own
event loop: the system emits its events into the store, then ``run_completed`` or
``run_failed``. Every read is a fold of the stored events.

The thread keeps the server answering during a run: retrieval (loading the models, embedding
and reranking a search) is synchronous and would otherwise hold the server's loop for seconds.
Model clients and the response cache belong to one loop and thread, so each run makes its own
gateway (``GatewayFactory.from_settings``) in its thread. Only the run's thread writes its
events; the server reads them. Each append wakes the run's listeners on the server's loop
(``wait_events``, for the event stream, PLAN 9.2).

The server keeps its events in Postgres (``juris.db.events``, PLAN 9.3); tests and mock mode
keep them in memory. ``restore`` rebuilds the cases and runs from the store when the server
starts, so finished runs outlive it; a run the store holds no end for was cut off by the last
shutdown, and is closed with ``run_failed``. Each finished run is also written to
``<runs_dir>/api/<run_id>/`` like a ``run_case.py`` run: manifest, events, folded view and
model calls (the manifest also to the ``runs`` table when ``db_url`` is given).
"""

import asyncio
import json
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from juris.api.schemas import CaseCreated, CaseInfo, RunInfo, RunStatus
from juris.baselines.b0 import B0System
from juris.baselines.b1 import B1System
from juris.baselines.b2 import B2System
from juris.baselines.base import RunContext, System
from juris.config import PipelineProfile, Settings, get_settings, load_profile, resolve_run_config
from juris.eval.juris_eval import free_item
from juris.eval.runner import GatewayFactory
from juris.events.catalog import (
    CaseCreatedPayload,
    Event,
    RunCompletedPayload,
    RunFailedPayload,
)
from juris.events.fold import CaseView, fold
from juris.events.store import EventEmitter, EventStore, InMemoryEventStore
from juris.llm.gateway import LLMGateway
from juris.pipeline.system import JurisSystem
from juris.runs import RunRecorder, new_run_id

# The configurations the API runs, by profile kind (the runner's, without its test system).
SYSTEMS: dict[str, Callable[[], System]] = {
    "juris": JurisSystem,
    "b0": B0System,
    "b1": B1System,
    "b2": B2System,
}


INTERRUPTED = "interrupted: the server stopped before the run finished"


class UnknownProfile(ValueError):
    """No profile of that name that the API can run."""


class RunLog(EventStore, Protocol):
    """An event store that can list its runs (to restore them after a restart)."""

    def bounds(self) -> list[tuple[Event, Event]]: ...


class _NotifyingStore:
    """An event store that reports each append (from any thread) to ``on_append``."""

    def __init__(self, inner: RunLog, on_append: Callable[[str], None]) -> None:
        self._inner = inner
        self._on_append = on_append

    def append(self, event: Event) -> None:
        self._inner.append(event)
        self._on_append(event.run_id)

    def read(self, run_id: str, after_seq: int = 0) -> list[Event]:
        return self._inner.read(run_id, after_seq)

    def last_seq(self, run_id: str) -> int:
        return self._inner.last_seq(run_id)

    def bounds(self) -> list[tuple[Event, Event]]:
        return self._inner.bounds()


@dataclass
class _Run:
    run_id: str
    case_id: str
    question: str
    profile: str
    created_at: datetime
    status: RunStatus = "queued"
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Case:
    case_id: str
    question: str
    created_at: datetime
    run_ids: list[str] = field(default_factory=list)


class RunManager:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        systems: Mapping[str, Callable[[], System]] = SYSTEMS,
        gateways: Callable[[], LLMGateway] | None = None,  # default: one factory per run
        store: RunLog | None = None,
        concurrency: int = 1,
        out_dir: Path | None = None,
        db_url: str | None = None,  # also record each run's manifest in the ``runs`` table
    ) -> None:
        self.settings = settings or get_settings()
        self.store: RunLog = _NotifyingStore(store or InMemoryEventStore(), self._appended)
        self.out_dir = out_dir or self.settings.runs_dir / "api"
        self._db_url = db_url
        self._factories = dict(systems)
        self._systems: dict[str, System] = {}
        self._gateways = gateways
        self._concurrency = concurrency
        self._cases: dict[str, _Case] = {}
        self._runs: dict[str, _Run] = {}
        self._queue: asyncio.Queue[tuple[str, PipelineProfile]] | None = None
        self._workers: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None  # the server's, set on first submit
        self._listeners: dict[str, set[asyncio.Future[None]]] = {}

    # ---- configurations ----------------------------------------------------------------

    def profiles(self) -> list[PipelineProfile]:
        """Every profile in the configs directory that the API can run, by name."""
        found = [load_profile(p) for p in sorted(self.settings.pipeline_configs_dir.glob("*.yaml"))]
        return [p for p in found if p.kind in self._factories]

    def profile(self, name: str) -> PipelineProfile:
        for profile in self.profiles():
            if profile.name == name:
                return profile
        raise UnknownProfile(f"no runnable profile named {name!r} (see GET /configs)")

    def system(self, kind: str) -> System:
        """One instance per kind, kept for the process (Juris's retrieval loads once)."""
        if kind not in self._systems:
            self._systems[kind] = self._factories[kind]()
        return self._systems[kind]

    # ---- cases and runs ----------------------------------------------------------------

    def submit(self, question: str, profile_name: str) -> CaseCreated:
        """Open a case and its first run, and queue the run. Raises ``UnknownProfile``."""
        profile = self.profile(profile_name)
        self._loop = asyncio.get_running_loop()
        now = datetime.now(UTC)
        case = _Case(f"CASE-{secrets.token_hex(4)}", question, now)
        run = _Run(new_run_id(profile.name), case.case_id, question, profile.name, now)
        case.run_ids.append(run.run_id)
        self._cases[case.case_id] = case
        self._runs[run.run_id] = run
        EventEmitter(self.store, run_id=run.run_id, case_id=case.case_id).emit(
            CaseCreatedPayload(
                question=question,
                profile=profile.name,
                corpus_snapshot_id=self.system(profile.kind).corpus_snapshot_id,
            )
        )
        self._start_workers().put_nowait((run.run_id, profile))
        return CaseCreated(case_id=case.case_id, run_id=run.run_id)

    def restore(self) -> int:
        """Rebuild the cases and runs the store holds and this manager does not (at startup);
        returns how many. A run with no end stored was cut off by the last shutdown: it is
        closed with ``run_failed``, so its record and its stream end honestly."""
        restored = 0
        for first, last in self.store.bounds():
            payload = first.payload
            if first.run_id in self._runs or not isinstance(payload, CaseCreatedPayload):
                continue
            case = self._cases.setdefault(
                first.case_id, _Case(first.case_id, payload.question, first.ts)
            )
            run = _Run(first.run_id, first.case_id, payload.question, payload.profile, first.ts)
            if last.type not in {"run_completed", "run_failed"}:
                EventEmitter(self.store, run_id=run.run_id, case_id=run.case_id).emit(
                    RunFailedPayload(reason=INTERRUPTED)
                )
            run.status = "completed" if last.type == "run_completed" else "failed"
            run.done.set()
            case.run_ids.append(run.run_id)
            self._runs[run.run_id] = run
            restored += 1
        return restored

    def case(self, case_id: str) -> CaseInfo | None:
        case = self._cases.get(case_id)
        if case is None:
            return None
        runs = [self._runs[r] for r in case.run_ids]
        return CaseInfo(
            case_id=case.case_id,
            question=case.question,
            created_at=case.created_at,
            runs=[
                RunInfo(
                    run_id=r.run_id, profile=r.profile, status=r.status,
                    created_at=r.created_at,
                )
                for r in runs
            ],
        )  # fmt: skip

    def view(self, run_id: str) -> CaseView | None:
        """The run's record as it stands: a fold of its events so far."""
        events = self.store.read(run_id)
        return fold(events) if events else None

    async def wait(self, run_id: str) -> None:
        """Until the run has finished (completed or failed)."""
        await self._runs[run_id].done.wait()

    async def wait_events(self, run_id: str, after_seq: int, timeout: float) -> list[Event]:
        """The run's events after ``after_seq``: at once if there are any, else the next ones
        to arrive, or [] after ``timeout`` seconds. A wake-up that brings nothing new (one
        scheduled for an event the caller already has) goes back to waiting."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        listeners = self._listeners.setdefault(run_id, set())
        while True:
            listener: asyncio.Future[None] = loop.create_future()
            listeners.add(listener)
            try:
                events = self.store.read(run_id, after_seq)  # after registering: none missed
                remaining = deadline - loop.time()
                if events or remaining <= 0:
                    return events
                try:
                    await asyncio.wait_for(listener, remaining)
                except TimeoutError:
                    return []
            finally:
                listeners.discard(listener)

    def _appended(self, run_id: str) -> None:
        """Called by the store in whichever thread appended: wake the listeners on the loop."""
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._wake, run_id)

    def _wake(self, run_id: str) -> None:
        for listener in self._listeners.get(run_id, ()):
            if not listener.done():
                listener.set_result(None)

    # ---- execution ---------------------------------------------------------------------

    def _start_workers(self) -> asyncio.Queue[tuple[str, PipelineProfile]]:
        """The queue, with its workers started on first use (inside the server's loop)."""
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._workers = [
                asyncio.create_task(self._work(), name=f"juris-run-worker-{n}")
                for n in range(self._concurrency)
            ]
        return self._queue

    async def _work(self) -> None:
        assert self._queue is not None
        while True:
            run_id, profile = await self._queue.get()
            run = self._runs[run_id]
            run.status = "running"
            try:
                await asyncio.to_thread(asyncio.run, self._execute(run, profile))
            except Exception as exc:  # a broken run must not stop the worker
                view = self.view(run.run_id)
                if view is None or view.status == "running":
                    EventEmitter(self.store, run_id=run.run_id, case_id=run.case_id).emit(
                        RunFailedPayload(reason=f"{type(exc).__name__}: {exc}")
                    )
            finally:
                view = self.view(run.run_id)
                run.status = "completed" if view and view.status == "completed" else "failed"
                run.done.set()
                self._queue.task_done()

    async def _execute(self, run: _Run, profile: PipelineProfile) -> None:
        config = resolve_run_config(self.settings, profile)
        system = self.system(profile.kind)
        # in this run's thread: its own model clients and cache connection (module docstring)
        own: GatewayFactory | None = None
        if self._gateways is not None:
            gateway = self._gateways()
        else:
            own = GatewayFactory.from_settings(self.settings)
            gateway = own()
        emitter = EventEmitter(self.store, run_id=run.run_id, case_id=run.case_id)
        recorder = RunRecorder(
            config,
            corpus_snapshot_id=system.corpus_snapshot_id,
            embedding_model=system.embedding_model,
            ledger=gateway.ledger,
            runs_dir=self.out_dir,
            seeds={"item": profile.seed},
            db_url=self._db_url,
            run_id=run.run_id,
        )
        item = free_item(run.question, run.case_id)
        ctx = RunContext(item, config, gateway, emitter, profile.seed)
        try:
            with recorder:
                await system.run(ctx)
        except Exception as exc:  # recorded as the run's end, like the runner
            emitter.emit(RunFailedPayload(reason=f"{type(exc).__name__}: {exc}"))
        else:
            emitter.emit(RunCompletedPayload(totals=recorder.manifest.totals))
        events = self.store.read(run.run_id)
        out = recorder.run_dir
        (out / "events.jsonl").write_text(
            "".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8"
        )
        (out / "caseview.json").write_text(fold(events).model_dump_json(), encoding="utf-8")
        (out / "calls.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in gateway.ledger.records]),
            encoding="utf-8",
        )
        if own is not None and own.cache is not None:
            own.cache.close()

    async def close(self) -> None:
        """Cancel the workers (server shutdown); queued and running runs are abandoned."""
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers, self._queue = [], None
