"""S9, the Judge (IDEA_final §6.1, PLAN 7.11): one call writes the ``CaseAnalysis`` through the
Case Record (``set_analysis`` checks invariants 3 and 5). S10 then checks its citations.

The Judge may cite only verified or weak evidence, so every argument citation needs a result
first. When some have none (the no_verifier_gates ablation turns S4 and S7 off, PLAN 7.12),
they are verified here, by S10's gate, before the Judge runs: the one final check then comes
before the Judge instead of after it, and the agents of S5-S8 worked without verdicts.

Recorded (``s9.*``): citations verified first, the record checks the Judge was given (PLAN 8.2),
whether an analysis was written, its issues, and the inline citations and key evidence it gives.

Passthrough runs the Judge too: every pipeline needs an analysis.
"""

from juris.agents.judge import Judge, inline_refs, record_checks
from juris.models.common import Stage
from juris.pipeline.budget import Spend
from juris.pipeline.orchestrator import StageContext
from juris.pipeline.stages.verify import VerificationGate, pending


class JudgeStage:
    stage = Stage.S9
    optional = False

    def __init__(self, *, batch_size: int = 1, max_passage_chars: int | None = None) -> None:
        self.gate = VerificationGate(
            Stage.S10, batch_size=batch_size, max_passage_chars=max_passage_chars
        )

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped

    async def run(self, ctx: StageContext) -> None:
        values = ctx.run.values
        unchecked = len(pending(ctx.record.view))
        values["s9.verified_first"] = float(unchecked)
        if unchecked:
            final = StageContext(ctx.run, ctx.record, ctx.budget, Stage.S10, ctx.mode)
            await self.gate.run(final)
        checks = record_checks(ctx.record.view)
        values["s9.record_checks"] = float(len({c for found in checks.values() for c in found}))
        await Judge().run(ctx)
        analysis = ctx.record.view.analysis
        values["s9.analysis"] = float(analysis is not None)
        if analysis is None:
            return
        positions = [p for i in analysis.issues for p in i.positions]
        values["s9.issues"] = float(len(analysis.issues))
        values["s9.inline_citations"] = float(
            sum(len(inline_refs(p.summary)) for p in positions)
            + len(inline_refs(analysis.overall_summary))
        )
        values["s9.key_evidence"] = float(sum(len(p.key_evidence) for p in positions))

    async def passthrough(self, ctx: StageContext) -> None:
        await self.run(ctx)
