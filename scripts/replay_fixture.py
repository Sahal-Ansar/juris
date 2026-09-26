"""Replay a fixture's events with realistic delays, as the SSE stream will send them.

Each event is validated, then printed as an SSE frame (``id:``, ``event:``, ``data:``).
The delay between events follows their timestamps, divided by ``--speed``.

Usage:
    uv run scripts/replay_fixture.py fixtures/runs/coercion_full.jsonl
    uv run scripts/replay_fixture.py fixtures/runs/failure_case.jsonl --speed 10 --max-delay 0.5
    uv run scripts/replay_fixture.py fixtures/runs/coercion_full.jsonl --no-delay | head
"""

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from juris.events import Event, parse_event


def read_events(path: Path) -> list[Event]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [parse_event(line) for line in lines if line.strip()]


def sse_frame(event: Event) -> str:
    return f"id: {event.seq}\nevent: {event.type}\ndata: {event.model_dump_json()}\n\n"


def replay(
    events: list[Event],
    *,
    speed: float = 1.0,
    max_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Yield SSE frames, sleeping between them according to the event timestamps."""
    previous = None
    for event in events:
        if previous is not None and speed > 0:
            gap = (event.ts - previous.ts).total_seconds() / speed
            sleep(min(max(gap, 0.0), max_delay))
        previous = event
        yield sse_frame(event)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    parser.add_argument("--max-delay", type=float, default=2.0, help="Cap per gap (seconds)")
    parser.add_argument("--no-delay", action="store_true")
    args = parser.parse_args(argv)
    events = read_events(args.fixture)
    for frame in replay(events, speed=0 if args.no_delay else args.speed, max_delay=args.max_delay):
        sys.stdout.write(frame)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
