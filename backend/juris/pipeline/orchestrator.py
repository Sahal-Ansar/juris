"""The orchestrator (IDEA_final §6; PLAN 7.1): the fixed stage machine S0-S10.

It runs the stages in order, each according to the profile's ``stages`` setting:

- ``on``: ``stage_started``, the stage's ``run``, ``stage_completed``;
- ``passthrough``: the same events (mode and outcome ``passthrough``) around the stage's
  ``passthrough``, the minimal version that keeps later stages fed (for ablations, PLAN 7.12);
- ``off``: only ``stage_completed`` with outcome ``skipped``.

Budget (``budget.py``): before an *optional* stage, if what is left will not cover the stage's
estimate, the stage is skipped with a ``budget_warning``. Required stages always run, so a case
always ends with an analysis. Stages skip their own optional steps (a second rebuttal round,
extra searches, some jurors) through ``StageContext.budget`` and ``warn_budget``.

Failure: an exception in a stage becomes ``StageFailed`` naming the stage. The runner records
it as ``run_failed``, and every event written before it stays in the record. The orchestrator
is a ``System`` (``baselines.base``), so the experiment runner opens and closes the run.
"""

import asyncio
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from juris.baselines.base import RunContext
from juris.config import DeliberationParams, StageMode, StagesConfig
from juris.events.catalog import BudgetWarningPayload, StageCompletedPayload, StageStartedPayload
from juris.models.common import Stage
from juris.pipeline.budget import BudgetManager, Spend, ledger_spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.evidence_registry import CorpusRecords

# The stages in order, with their ``StagesConfig`` field and a display name.
STAGES: tuple[tuple[Stage, str, str], ...] = (
    (Stage.S0, "s0_issue_framing", "issue framing"),
    (Stage.S1, "s1_research", "research"),
    (Stage.S2, "s2_opening", "opening arguments"),
    (Stage.S3, "s3_precedent", "precedent analysis"),
    (Stage.S4, "s4_verify", "verification gate 1"),
    (Stage.S5, "s5_rebuttal", "rebuttal"),
    (Stage.S6, "s6_cross_exam", "cross-examination"),
    (Stage.S7, "s7_verify", "verification gate 2"),
    (Stage.S8, "s8_jury", "jury"),
    (Stage.S9, "s9_judge", "judge"),
    (Stage.S10, "s10_final_verify", "final verification"),
)
STAGE_NAMES = {stage: name for stage, _, name in STAGES}


class PipelineConfigError(ValueError):
    """The pipeline cannot run as configured (e.g. a stage is on but has no implementation)."""


class StageFailed(RuntimeError):
    """A stage raised; the message names the stage and the cause."""

    def __init__(self, stage: Stage, cause: BaseException) -> None:
        self.stage = stage
        super().__init__(f"{stage.value} {STAGE_NAMES[stage]}: {type(cause).__name__}: {cause}")


@dataclass(frozen=True)
class StageContext:
    """What a stage gets: the run, the Case Record, the budget, and its own mode."""

    run: RunContext
    record: CaseRecord
    budget: BudgetManager
    stage: Stage
    mode: StageMode

    @property
    def params(self) -> DeliberationParams:
        return self.run.config.profile.params

    def warn_budget(self, action: str) -> None:
        """Log that ``action`` (what was skipped or reduced) was forced by the budget."""
        used = self.budget.used
        self.run.emitter.emit(
            BudgetWarningPayload(
                tokens_used=used.tokens,
                tokens_budget=self.budget.budget.max_tokens_per_case,
                usd_used=used.usd,
                usd_budget=self.budget.budget.max_usd_per_case,
                action=action,
            ),
            stage=self.stage,
        )

    async def gather[T](self, *calls: Coroutine[Any, Any, T]) -> list[T]:
        """Run independent calls at once (Counsel A and B, jurors, per-precedent analyses);
        the gateway's semaphore caps how many reach the provider. If one fails, the others
        are cancelled and its exception is raised."""
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(c) for c in calls]
        except* Exception as failed:
            raise failed.exceptions[0] from None
        return [t.result() for t in tasks]


class StageRunner(Protocol):
    """One stage's implementation (PLAN 7.3-7.11)."""

    stage: Stage
    optional: bool  # may be skipped when the budget cannot cover it

    def estimate(self, ctx: StageContext) -> Spend | None:
        """Expected spend, for the budget check of an optional stage (None: unknown)."""
        ...

    async def run(self, ctx: StageContext) -> None: ...

    async def passthrough(self, ctx: StageContext) -> None: ...


class Orchestrator:
    """Runs ``stages`` for one case. A ``System`` for the experiment runner."""

    def __init__(
        self,
        stages: Sequence[StageRunner],
        corpus: Callable[[], CorpusRecords],
        *,
        corpus_snapshot_id: str,
        embedding_model: str,
    ) -> None:
        by_stage: dict[Stage, StageRunner] = {}
        for runner in stages:
            if runner.stage in by_stage:
                raise PipelineConfigError(f"two implementations of {runner.stage.value}")
            by_stage[runner.stage] = runner
        self.stages: Mapping[Stage, StageRunner] = by_stage
        self._corpus = corpus
        self.corpus_snapshot_id = corpus_snapshot_id
        self.embedding_model = embedding_model

    def check(self, modes: StagesConfig) -> None:
        """Every stage that is not off needs an implementation."""
        missing = [
            f"{stage.value} ({name})"
            for stage, field, name in STAGES
            if getattr(modes, field) != "off" and stage not in self.stages
        ]
        if missing:
            raise PipelineConfigError(f"no implementation for {', '.join(missing)}")

    async def run(self, ctx: RunContext) -> None:
        modes = ctx.config.profile.stages
        self.check(modes)  # before any event, so a misconfigured run writes nothing
        record = CaseRecord(ctx.emitter, self._corpus(), params=ctx.config.profile.params)
        budget = BudgetManager(
            ctx.config.budget, ledger_spend(ctx.gateway.ledger, ctx.gateway.prices)
        )
        emit = ctx.emitter.emit
        for stage, field, name in STAGES:
            mode: StageMode = getattr(modes, field)
            if mode == "off":
                emit(
                    StageCompletedPayload(outcome="skipped", reason="off in the profile"),
                    stage=stage,
                )
                continue
            runner = self.stages[stage]
            sctx = StageContext(ctx, record, budget, stage, mode)
            if mode == "on" and runner.optional and not budget.allows(runner.estimate(sctx)):
                sctx.warn_budget(f"skipped {stage.value} ({name})")
                emit(StageCompletedPayload(outcome="skipped", reason="budget"), stage=stage)
                continue
            emit(StageStartedPayload(mode=mode), stage=stage)
            try:
                if mode == "passthrough":
                    await runner.passthrough(sctx)
                else:
                    await runner.run(sctx)
            except Exception as exc:
                raise StageFailed(stage, exc) from exc
            emit(
                StageCompletedPayload(
                    outcome="passthrough" if mode == "passthrough" else "completed"
                ),
                stage=stage,
            )
        if record.view.analysis is None:
            raise PipelineConfigError("the pipeline finished without an analysis (S9)")
