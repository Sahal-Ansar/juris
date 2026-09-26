"""``Run``: the manifest that makes a run reproducible and auditable (IDEA_final §8, §15).

Written and updated by ``juris.runs.RunRecorder``.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from juris.config import ModelSpec, Role, RunConfig
from juris.models.common import JurisModel

RunStatus = Literal["running", "completed", "failed"]


class GitState(JurisModel):
    commit: str
    dirty: bool


class RunTotals(JurisModel):
    llm_calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    unpriced_calls: int = 0


class Run(JurisModel):
    """Everything needed to reproduce a run: config, code, models, prompts, data, seeds, cost."""

    run_id: str
    status: RunStatus
    profile: str
    config: RunConfig
    git: GitState
    models: dict[Role, ModelSpec]
    prompt_hashes: dict[str, str]
    corpus_snapshot_id: str
    embedding_model: str
    seeds: dict[str, int]
    started_at: datetime
    ended_at: datetime | None = None
    error: str | None = None
    totals: RunTotals = Field(default_factory=RunTotals)
