"""The event log in Postgres (PLAN 9.3): the API's store, so runs outlive the server.

One row per event in ``events`` (migration 0001): the envelope's fields as columns, the
payload as JSONB. Reading parses each row back into its event (``parse_event``), so what comes
out validates exactly as what went in. The ordering rules are the in-memory store's
(``check_next``). Only a run's own thread appends to it, so reading the run's last row and then
inserting is safe; the primary key (run_id, seq) would refuse a duplicate anyway.
"""

from datetime import UTC
from typing import Any

from sqlalchemy import Row, func, insert, select
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.engine import Engine

from juris.db.models import EventRow
from juris.events.catalog import Event, parse_event
from juris.events.store import check_next

EVENTS = EventRow.__table__


def to_row(event: Event) -> dict[str, Any]:
    return {
        "run_id": event.run_id,
        "seq": event.seq,
        "event_id": event.event_id,
        "case_id": event.case_id,
        "type": event.type,
        "stage": event.stage,
        "agent": event.agent,
        "schema_version": event.schema_version,
        "ts": event.ts,
        "payload": event.payload.model_dump(mode="json"),
    }


def from_row(row: Row[Any]) -> Event:
    return parse_event(
        {
            "schema_version": row.schema_version,
            "event_id": row.event_id,
            "run_id": row.run_id,
            "case_id": row.case_id,
            "seq": row.seq,
            "ts": row.ts.astimezone(UTC),  # as the session's time zone gives it, back to UTC
            "stage": row.stage,
            "agent": row.agent,
            "type": row.type,
            "payload": row.payload,
        }
    )


class DbEventStore:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def append(self, event: Event) -> None:
        with self.engine.begin() as conn:
            last = conn.execute(
                select(EventRow.seq, EventRow.case_id)
                .where(EventRow.run_id == event.run_id)
                .order_by(EventRow.seq.desc())
                .limit(1)
            ).one_or_none()
            check_next(event, last.seq if last else 0, last.case_id if last else None)
            conn.execute(insert(EventRow).values(to_row(event)))

    def read(self, run_id: str, after_seq: int = 0) -> list[Event]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(EVENTS)
                .where(EventRow.run_id == run_id, EventRow.seq > after_seq)
                .order_by(EventRow.seq)
            ).all()
        return [from_row(r) for r in rows]

    def last_seq(self, run_id: str) -> int:
        with self.engine.connect() as conn:
            last = conn.execute(
                select(func.max(EventRow.seq)).where(EventRow.run_id == run_id)
            ).scalar()
        return last or 0

    def bounds(self) -> list[tuple[Event, Event]]:
        """Each stored run's first and last event, in the order the runs began."""
        per_run = select(EVENTS).ext(distinct_on(EventRow.run_id))
        with self.engine.connect() as conn:
            first, last = (
                conn.execute(per_run.order_by(EventRow.run_id, order)).all()
                for order in (EventRow.seq, EventRow.seq.desc())
            )
        ends = {r.run_id: from_row(r) for r in last}
        pairs = [(from_row(r), ends[r.run_id]) for r in first]
        return sorted(pairs, key=lambda pair: (pair[0].ts, pair[0].run_id))
