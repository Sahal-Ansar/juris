"""Verification gates S4, S7 and S10 (IDEA_final §6, PLAN 7.9): the Citation Verifier (6.1) on
every citation that has no result yet.

One class serves all three gates. Each gate:

1. **Finds what is new:** every argument citation whose (claim, evidence) pair has no
   verification result yet. At S4 that is all of S2's citations; at S7, the ones S5-S6 added.
   S10 also covers anything added after S7; the Judge's own references are 7.11's.
2. **Verifies** them: existence and exact quote in code, then the entailment judge (the
   VERIFIER role's model), one result per citation.
3. **Writes** each result through the Case Record (``add_verification``, which checks that the
   citation exists).
4. **Updates the claims** it touched. A claim's status follows its best result over *every*
   gate so far, not just this one: a claim verified at S4 stays verified if an S7 citation is
   weak. Only the verifier's own statuses are changed (proposed, verified, weak, unsupported);
   contested, survives and falls belong to later stages. Changes go through
   ``set_claim_status``, so the lifecycle is checked.

The gates are required. A profile can turn S4 and S7 off (the ``no_verifier_gates`` ablation,
7.12), leaving S10 to check everything. Passthrough runs only the code checks (existence and
exact quote): failing citations are recorded as invalid, the rest stay unverified.

Recorded (``s4.*``, ``s7.*``, ``s10.*``): citations checked, the count per status, and claims
whose status changed.
"""

from collections.abc import Iterable

from juris.config import Role
from juris.events import CaseView
from juris.models import Citation, CreatedBy, VerificationResult
from juris.models.common import ClaimStatus, Stage, VerificationStatus
from juris.pipeline.budget import Spend
from juris.pipeline.case_record import CaseRecord
from juris.pipeline.orchestrator import StageContext
from juris.verify import CitationVerifier, LLMEntailmentJudge, check_citation
from juris.verify.entailment import AGENT

GATES = (Stage.S4, Stage.S7, Stage.S10)
OWNED = frozenset(
    {ClaimStatus.PROPOSED, ClaimStatus.VERIFIED, ClaimStatus.WEAK, ClaimStatus.UNSUPPORTED}
)
_CLAIM_STATUS = {
    VerificationStatus.VERIFIED: ClaimStatus.VERIFIED,
    VerificationStatus.WEAK: ClaimStatus.WEAK,
}


def pending(view: CaseView) -> list[Citation]:
    """Argument citations with no verification result yet, once per (claim, evidence)."""
    done = {(v.claim_id, v.evidence_id) for v in view.verifications}
    out: dict[tuple[str, str], Citation] = {}
    for argument in view.arguments.values():
        for citation in argument.citations:
            key = (citation.claim_id, citation.evidence_id)
            if key not in done:
                out.setdefault(key, citation)
    return list(out.values())


def best_status(statuses: Iterable[VerificationStatus]) -> ClaimStatus:
    """A claim's status from its citations' results: any verified, else any weak, else
    unsupported (invalid citations support nothing)."""
    found = set(statuses)
    for status in (VerificationStatus.VERIFIED, VerificationStatus.WEAK):
        if status in found:
            return _CLAIM_STATUS[status]
    return ClaimStatus.UNSUPPORTED


class VerificationGate:
    optional = False

    def __init__(
        self, stage: Stage, *, batch_size: int = 1, max_passage_chars: int | None = None
    ) -> None:
        if stage not in GATES:
            raise ValueError(f"a verification gate runs at S4, S7 or S10, not {stage.value}")
        self.stage = stage
        self.batch_size = batch_size  # 1: batching mixed up pairs on the local model (D-043)
        self.max_passage_chars = max_passage_chars

    def estimate(self, ctx: StageContext) -> Spend | None:
        return None  # required: never skipped

    async def run(self, ctx: StageContext) -> None:
        record = ctx.record
        view = record.view
        todo = pending(view)
        results: list[VerificationResult] = []
        if todo:
            model = ctx.run.config.model_for(Role.VERIFIER)
            verifier = CitationVerifier(
                evidence=record.registry,
                judge=LLMEntailmentJudge(
                    ctx.run.gateway,
                    model,
                    batch_size=self.batch_size,
                    seed=ctx.run.seed,
                    max_passage_chars=self.max_passage_chars,
                ),
                verifier_model=model.name,
            )
            results = await verifier.verify(view.claims, todo, stage=self.stage)
            self._write(ctx, todo, results)
        else:
            self._values(ctx, [], [], 0)

    async def passthrough(self, ctx: StageContext) -> None:
        """Code checks only: a citation failing existence or quote is recorded as invalid."""
        record = ctx.record
        failed: list[Citation] = []
        results: list[VerificationResult] = []
        by = CreatedBy(stage=self.stage, agent=AGENT)
        for citation in pending(record.view):
            failure = check_citation(record.registry, citation)
            if failure is not None:
                check, reason = failure
                failed.append(citation)
                results.append(
                    VerificationResult(
                        status=VerificationStatus.INVALID,
                        check=check,
                        justification=reason,
                        created_by=by,
                    )
                )
        self._write(ctx, failed, results)

    def _write(
        self, ctx: StageContext, citations: list[Citation], results: list[VerificationResult]
    ) -> None:
        record = ctx.record
        for citation, result in zip(citations, results, strict=True):
            record.add_verification(
                citation.claim_id, citation.evidence_id, result, stage=self.stage, agent=AGENT
            )
        changed = self._update_claims(record, dict.fromkeys(c.claim_id for c in citations))
        self._values(ctx, citations, results, changed)

    def _update_claims(self, record: CaseRecord, claim_ids: Iterable[str]) -> int:
        """Set each claim's status from all its results so far; returns how many changed. A
        claim with a citation still unchecked (passthrough) keeps its status."""
        view = record.view
        unchecked = {(c.claim_id, c.evidence_id) for c in pending(view)}
        changed = 0
        for claim_id in claim_ids:
            claim = view.claims[claim_id]
            if claim.status not in OWNED or any(c == claim_id for c, _ in unchecked):
                continue
            statuses = [v.result.status for v in view.verifications if v.claim_id == claim_id]
            status = best_status(statuses)
            reason = (
                f"best of {len(statuses)} verified citation(s) is {max(statuses, key=_rank).value}"
            )
            if record.set_claim_status(claim_id, status, reason, stage=self.stage, agent=AGENT):
                changed += 1
        return changed

    def _values(
        self,
        ctx: StageContext,
        citations: list[Citation],
        results: list[VerificationResult],
        changed: int,
    ) -> None:
        prefix = self.stage.value.lower()
        values = ctx.run.values
        values[f"{prefix}.checked"] = float(len(citations))
        for status in VerificationStatus:
            values[f"{prefix}.{status.value}"] = float(sum(r.status is status for r in results))
        values[f"{prefix}.claims_changed"] = float(changed)


def _rank(status: VerificationStatus) -> int:
    return [
        VerificationStatus.INVALID,
        VerificationStatus.UNSUPPORTED,
        VerificationStatus.WEAK,
        VerificationStatus.VERIFIED,
    ].index(status)
