"""The run's events as Server-Sent Events (PLAN 9.2, IDEA_final §12): the live proceedings.

``GET /runs/{run_id}/events`` sends every event of the run in ``seq`` order, then each new one
as it is stored, in the SSE wire format::

    retry: 3000                      once, first: reconnect after 3 s

    id: 12                           the event's seq
    data: {"event_id": ...}          the envelope, as in schemas/event.schema.json

    : ping                           a comment when the run has been quiet for a while

    event: end                       after run_completed or run_failed; the server closes
    data: {}

Reconnecting with ``Last-Event-ID: 12`` (what ``EventSource`` sends) or ``?after=12`` resumes
at seq 13, so a client sees every event once. A reconnect after the run's last event gets
``204 No Content``, which tells ``EventSource`` to stop reconnecting.
"""

from collections.abc import AsyncIterator

from juris.api.runs import RunManager
from juris.events.catalog import Event

RETRY_MS = 3000
TERMINAL = frozenset({"run_completed", "run_failed"})
PING = b": ping\n\n"
END = b"event: end\ndata: {}\n\n"


def frame(event: Event) -> bytes:
    """One event in the wire format (the JSON envelope is a single line)."""
    return f"id: {event.seq}\ndata: {event.model_dump_json()}\n\n".encode()


def ended(runs: RunManager, run_id: str, after_seq: int) -> bool:
    """Whether the run's terminal event is stored and the client has had it. Read from the
    store, not the run's status, which the worker sets only after writing the run's files."""
    last = runs.store.last_seq(run_id)
    tail = runs.store.read(run_id, last - 1)
    return bool(tail) and tail[-1].type in TERMINAL and last <= after_seq


async def events(
    runs: RunManager, run_id: str, after_seq: int, ping_seconds: float
) -> AsyncIterator[bytes]:
    """The stream's bytes, until the run's terminal event (or the client goes away)."""
    yield f"retry: {RETRY_MS}\n\n".encode()
    last = after_seq
    while True:
        batch = await runs.wait_events(run_id, last, ping_seconds)
        if not batch:
            yield PING
            continue
        for event in batch:
            yield frame(event)
            last = event.seq
            if event.type in TERMINAL:
                yield END
                return
