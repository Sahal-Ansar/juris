"""Versioned events: they drive the Case Record and stream to the UI (IDEA_final §12)."""

from juris.events.catalog import EVENT_ADAPTER, EVENT_TYPES, Event, parse_event
from juris.events.envelope import SCHEMA_VERSION, EnvelopeBase, event_id
from juris.events.fold import CaseView, FoldError, fold
from juris.events.store import EventEmitter, EventOrderError, EventStore, InMemoryEventStore

__all__ = [
    "EVENT_ADAPTER",
    "EVENT_TYPES",
    "SCHEMA_VERSION",
    "CaseView",
    "EnvelopeBase",
    "Event",
    "EventEmitter",
    "EventOrderError",
    "EventStore",
    "FoldError",
    "InMemoryEventStore",
    "event_id",
    "fold",
    "parse_event",
]
