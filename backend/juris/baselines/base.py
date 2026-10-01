"""What the experiment runner (PLAN 5.5) needs from a system: B0-B2, Juris, or the dummy.

The runner opens the run (manifest, ``case_created``) and closes it (``run_completed`` with
the call totals, or ``run_failed`` if ``run`` raises). In between, the system emits its own
events, ending with ``analysis_ready``, and makes its model calls through ``ctx.gateway`` so
they are cached and costed per run.

A system may also record values only it can compute (e.g. how many of B0's cited
authorities exist, PLAN 6.2) in ``ctx.values`` and ``ctx.details``; the runner stores them
with the run's scores as the ``system`` metric, so they are aggregated like the others.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from juris.config import RunConfig
from juris.eval.juris_eval import JurisEvalItem
from juris.events.store import EventEmitter
from juris.llm.gateway import LLMGateway


@dataclass(frozen=True)
class RunContext:
    item: JurisEvalItem
    config: RunConfig
    gateway: LLMGateway
    emitter: EventEmitter
    seed: int
    values: dict[str, float | None] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class System(Protocol):
    corpus_snapshot_id: str
    embedding_model: str

    async def run(self, ctx: RunContext) -> None: ...
