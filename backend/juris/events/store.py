"""Append-only event storage: in memory here; in Postgres for the API (``juris.db.events``,
PLAN 9.3). Both keep the same ordering rules (``check_next``)."""

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


def check_next(event: Event, last_seq: int, last_case_id: str | None) -> None:
    """Raise ``EventOrderError`` unless ``event`` may follow the run's last stored event (none
    yet: ``last_seq`` 0): a run starts with ``case_created``, its seq increases, its case stays."""
    if last_case_id is None:
        if event.type != "case_created":
            raise EventOrderError(f"run {event.run_id} must start with case_created")
        return
    if event.seq <= last_seq:
        raise EventOrderError(f"run {event.run_id}: seq {event.seq} is not after {last_seq}")
    if event.case_id != last_case_id:
        raise EventOrderError(f"run {event.run_id} belongs to case {last_case_id}")


class InMemoryEventStore:
    def __init__(self) -> None:
        self._runs: dict[str, list[Event]] = {}

    def append(self, event: Event) -> None:
        events = self._runs.setdefault(event.run_id, [])
        last = events[-1] if events else None
        check_next(event, last.seq if last else 0, last.case_id if last else None)
        events.append(event)

    def read(self, run_id: str, after_seq: int = 0) -> list[Event]:
        return [e for e in self._runs.get(run_id, []) if e.seq > after_seq]

    def last_seq(self, run_id: str) -> int:
        events = self._runs.get(run_id)
        return events[-1].seq if events else 0

    def bounds(self) -> list[tuple[Event, Event]]:
        """Each stored run's first and last event, in the order the runs were stored."""
        return [(events[0], events[-1]) for events in self._runs.values() if events]


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

    def events(self, after_seq: int = 0) -> list[Event]:
        """This run's stored events after ``after_seq``, whoever emitted them."""
        return self._store.read(self.run_id, after_seq)
