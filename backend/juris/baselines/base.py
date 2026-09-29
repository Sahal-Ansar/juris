"""What the experiment runner (PLAN 5.5) needs from a system: B0-B2, Juris, or the dummy.

The runner opens the run (manifest, ``case_created``) and closes it (``run_completed`` with
the call totals, or ``run_failed`` if ``run`` raises). In between, the system emits its own
events, ending with ``analysis_ready``, and makes its model calls through ``ctx.gateway`` so
they are cached and costed per run.
"""

from dataclasses import dataclass
from typing import Protocol

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


class System(Protocol):
    corpus_snapshot_id: str
    embedding_model: str

    async def run(self, ctx: RunContext) -> None: ...
