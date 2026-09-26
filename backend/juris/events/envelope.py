"""The versioned event envelope (IDEA_final §12)."""

from typing import Annotated

from pydantic import AwareDatetime, Field, StringConstraints

from juris.models.common import JurisModel, Stage

SCHEMA_VERSION = "1.0"

EventIdStr = Annotated[str, StringConstraints(pattern=r"^evt_\d{6,}$")]


def event_id(seq: int) -> str:
    """``evt_000123``: the event ID is derived from ``seq``, so it is unique per run."""
    return f"evt_{seq:06d}"


class EnvelopeBase(JurisModel):
    """Fields shared by every event. Subclasses add ``type`` (a Literal) and ``payload``."""

    schema_version: Annotated[str, StringConstraints(pattern=r"^1\.\d+$")] = SCHEMA_VERSION
    event_id: EventIdStr
    run_id: Annotated[str, StringConstraints(min_length=1)]
    case_id: Annotated[str, StringConstraints(min_length=1)]
    seq: int = Field(ge=1, description="Strictly increasing per run, starting at 1")
    ts: AwareDatetime
    stage: Stage | None = Field(default=None, description="None for run-level events")
    agent: str | None = Field(default=None, description="None for events emitted by code")
