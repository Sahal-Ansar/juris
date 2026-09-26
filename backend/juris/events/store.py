"""Append-only event storage. In-memory now; Postgres-backed in PLAN 9.3."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast

from juris.events.catalog import EVENT_TYPES, Event
from juris.events.envelope import EnvelopeBase, event_id
from juris.models import JurisModel
from juris.models.common import Stage


class EventOrderError(ValueError):
    """An event that would break the per-run ordering (seq not increasing, wrong case)."""


class EventStore(Protocol):
    def append(self, event: Event) -> None: ...

    def read(self, run_id: str, after_seq: int = 0) -> list[Event]: ...

    def last_seq(self, run_id: str) -> int: ...


class InMemoryEventStore:
    def __init__(self) -> None:
        self._runs: dict[str, list[Event]] = {}

    def append(self, event: Event) -> None:
        events = self._runs.setdefault(event.run_id, [])
        if events:
            last = events[-1]
            if event.seq <= last.seq:
                raise EventOrderError(
                    f"run {event.run_id}: seq {event.seq} is not after {last.seq}"
                )
            if event.case_id != last.case_id:
                raise EventOrderError(f"run {event.run_id} belongs to case {last.case_id}")
        elif event.type != "case_created":
            raise EventOrderError(f"run {event.run_id} must start with case_created")
        events.append(event)

    def read(self, run_id: str, after_seq: int = 0) -> list[Event]:
        return [e for e in self._runs.get(run_id, []) if e.seq > after_seq]

    def last_seq(self, run_id: str) -> int:
        events = self._runs.get(run_id)
        return events[-1].seq if events else 0


_PAYLOAD_TO_EVENT: dict[type[JurisModel], type[EnvelopeBase]] = {
    cls.model_fields["payload"].annotation: cls  # type: ignore[misc]
    for cls in EVENT_TYPES.values()
}


class EventEmitter:
    """Wraps payloads in envelopes (next seq, derived event ID, UTC timestamp) and stores them."""

    def __init__(
        self,
        store: EventStore,
        *,
        run_id: str,
        case_id: str,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._store = store
        self.run_id = run_id
        self.case_id = case_id
        self._clock = clock

    def emit(
        self, payload: JurisModel, *, stage: Stage | None = None, agent: str | None = None
    ) -> Event:
        cls = _PAYLOAD_TO_EVENT.get(type(payload))
        if cls is None:
            raise TypeError(f"{type(payload).__name__} is not an event payload")
        seq = self._store.last_seq(self.run_id) + 1
        event = cast(
            Event,
            cls.model_validate(
                {
                    "event_id": event_id(seq),
                    "run_id": self.run_id,
                    "case_id": self.case_id,
                    "seq": seq,
                    "ts": self._clock(),
                    "stage": stage,
                    "agent": agent,
                    "payload": payload,
                }
            ),
        )
        self._store.append(event)
        return event
