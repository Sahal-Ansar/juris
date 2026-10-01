"""The Citation Verifier (IDEA_final §6.1, S4/S7/S10): three checks, cheapest first.

1. Existence: the citation's evidence ID is registered for the run, else ``invalid``.
2. Quote: the quote is a normalised exact substring of the evidence's chunk, else ``invalid``.
3. Entailment: a model judges whether the passage supports the claim; ``supports`` is
   ``verified``, ``partially_supports`` is ``weak``, the other two are ``unsupported``.

Checks 1-2 are code (``check_citation``), shared with the answer metrics so both apply one
rule; only citations that pass them reach the model. A claim's status follows its best
citation: any verified makes it ``verified``, else any weak makes it ``weak``, else it is
``unsupported``. Statuses owned by later stages (contested, survives, falls) are left alone.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from juris.events import EventEmitter
from juris.events.catalog import ClaimStatusChangedPayload, VerificationResultPayload
from juris.events.fold import CaseView
from juris.models import Chunk, Citation, Claim, CreatedBy, VerificationResult
from juris.models.common import (
    ClaimStatus,
    EntailmentLabel,
    Stage,
    VerificationCheck,
    VerificationStatus,
)
from juris.verify.entailment import AGENT, ClaimEvidencePair, PairJudge
from juris.verify.quote import quote_in_text

UNKNOWN_EVIDENCE = "unknown evidence"
NO_CHUNK = "chunk not in the record"
QUOTE_MISSING = "quote not in the chunk"

LABEL_STATUS = {
    EntailmentLabel.SUPPORTS: VerificationStatus.VERIFIED,
    EntailmentLabel.PARTIALLY_SUPPORTS: VerificationStatus.WEAK,
    EntailmentLabel.DOES_NOT_SUPPORT: VerificationStatus.UNSUPPORTED,
    EntailmentLabel.CONTRADICTS: VerificationStatus.UNSUPPORTED,
}
# The verifier's own statuses; a claim in any other one belongs to a later stage.
_OWNED = frozenset(
    {ClaimStatus.PROPOSED, ClaimStatus.VERIFIED, ClaimStatus.WEAK, ClaimStatus.UNSUPPORTED}
)


class EvidenceTexts(Protocol):
    """The run's evidence as the verifier reads it (``EvidenceRegistry`` and ``ViewEvidence``)."""

    def __contains__(self, evidence_id: str, /) -> bool: ...

    def chunk_text(self, evidence_id: str) -> str | None:
        """The evidence's chunk text; None if the evidence or its chunk is not known."""
        ...


def chunk_text(view: CaseView, evidence_id: str) -> str | None:
    evidence = view.evidence.get(evidence_id)
    if evidence is None:
        return None
    chunk: Chunk | None = view.chunks.get(evidence.chunk_id)
    return chunk.text if chunk is not None else None


@dataclass(frozen=True)
class ViewEvidence:
    """A folded run (``CaseView``) as ``EvidenceTexts``."""

    view: CaseView

    def __contains__(self, evidence_id: str, /) -> bool:
        return evidence_id in self.view.evidence

    def chunk_text(self, evidence_id: str) -> str | None:
        return chunk_text(self.view, evidence_id)


def check_citation(
    evidence: EvidenceTexts, citation: Citation, *, casefold: bool = False
) -> tuple[VerificationCheck, str] | None:
    """The deterministic checks. The failing check and why, or None if both pass."""
    if citation.evidence_id not in evidence:
        return VerificationCheck.EXISTENCE, UNKNOWN_EVIDENCE
    text = evidence.chunk_text(citation.evidence_id)
    if text is None:
        return VerificationCheck.QUOTE, NO_CHUNK
    if not quote_in_text(citation.quote, text, casefold=casefold):
        return VerificationCheck.QUOTE, QUOTE_MISSING
    return None


def claim_statuses(
    citations: Sequence[Citation], results: Sequence[VerificationResult]
) -> dict[str, ClaimStatus]:
    """Each cited claim's status from its best citation result (by claim ID)."""
    if len(citations) != len(results):
        raise ValueError(f"{len(results)} results for {len(citations)} citations")
    by_claim: defaultdict[str, set[VerificationStatus]] = defaultdict(set)
    for citation, result in zip(citations, results, strict=True):
        by_claim[citation.claim_id].add(result.status)
    out: dict[str, ClaimStatus] = {}
    for claim_id, statuses in sorted(by_claim.items()):
        if VerificationStatus.VERIFIED in statuses:
            out[claim_id] = ClaimStatus.VERIFIED
        elif VerificationStatus.WEAK in statuses:
            out[claim_id] = ClaimStatus.WEAK
        else:
            out[claim_id] = ClaimStatus.UNSUPPORTED
    return out


@dataclass
class CitationVerifier:
    evidence: EvidenceTexts
    judge: PairJudge | None
    emitter: EventEmitter | None = None
    verifier_model: str | None = None
    casefold: bool = False
    agent: str = AGENT

    async def verify(
        self, claims: Mapping[str, Claim], citations: Sequence[Citation], *, stage: Stage
    ) -> list[VerificationResult]:
        """One result per citation, in order. With an emitter, also emits a
        ``verification_result`` per citation, then ``claim_status_changed`` for each claim
        whose verifier-owned status differs from what its citations show."""
        for citation in citations:
            if citation.claim_id not in claims:
                raise ValueError(f"citation to unknown claim {citation.claim_id}")
        by = CreatedBy(stage=stage, agent=self.agent)
        results: list[VerificationResult | None] = []
        pending: list[tuple[int, ClaimEvidencePair]] = []
        for index, citation in enumerate(citations):
            failure = check_citation(self.evidence, citation, casefold=self.casefold)
            if failure is None:
                results.append(None)
                pending.append(
                    (
                        index,
                        ClaimEvidencePair(
                            claim_id=citation.claim_id,
                            claim=claims[citation.claim_id].text,
                            evidence_id=citation.evidence_id,
                            quote=citation.quote,
                            passage=self.evidence.chunk_text(citation.evidence_id) or "",
                        ),
                    )
                )
                continue
            check, reason = failure
            results.append(
                VerificationResult(
                    status=VerificationStatus.INVALID,
                    check=check,
                    justification=reason,
                    verifier_model=None,
                    created_by=by,
                )
            )
        if pending:
            if self.judge is None:
                raise ValueError("citations need the entailment check but no judge is set")
            judgements = await self.judge.judge([pair for _, pair in pending])
            if len(judgements) != len(pending):
                raise ValueError(f"the judge returned {len(judgements)} for {len(pending)} pairs")
            for (index, _), judgement in zip(pending, judgements, strict=True):
                results[index] = VerificationResult(
                    status=LABEL_STATUS[judgement.label],
                    check=VerificationCheck.ENTAILMENT,
                    label=judgement.label,
                    justification=judgement.justification.strip() or judgement.label.value,
                    verifier_model=self.verifier_model,
                    created_by=by,
                )
        done = [r for r in results if r is not None]
        if self.emitter is not None:
            for citation, result in zip(citations, done, strict=True):
                self.emitter.emit(
                    VerificationResultPayload(
                        claim_id=citation.claim_id, evidence_id=citation.evidence_id, result=result
                    ),
                    stage=stage,
                    agent=self.agent,
                )
            self._update_claims(claims, citations, done, stage)
        return done

    def _update_claims(
        self,
        claims: Mapping[str, Claim],
        citations: Sequence[Citation],
        results: Sequence[VerificationResult],
        stage: Stage,
    ) -> None:
        assert self.emitter is not None
        counts: defaultdict[str, defaultdict[VerificationStatus, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for citation, result in zip(citations, results, strict=True):
            counts[citation.claim_id][result.status] += 1
        for claim_id, status in claim_statuses(citations, results).items():
            current = claims[claim_id].status
            if current not in _OWNED or current is status:
                continue
            c = counts[claim_id]
            self.emitter.emit(
                ClaimStatusChangedPayload(
                    claim_id=claim_id,
                    status=status,
                    reason=(
                        f"{sum(c.values())} citations: {c[VerificationStatus.VERIFIED]} verified, "
                        f"{c[VerificationStatus.WEAK]} weak, "
                        f"{c[VerificationStatus.UNSUPPORTED]} unsupported, "
                        f"{c[VerificationStatus.INVALID]} invalid"
                    ),
                ),
                stage=stage,
                agent=self.agent,
            )
