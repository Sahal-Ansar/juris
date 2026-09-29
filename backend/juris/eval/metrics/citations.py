"""Citation validity and citation faithfulness (IDEA_final §11.2).

Both look at every citation in the record (``Argument.citations``: claim, evidence ID,
pinpoint, verbatim quote), not only the ones the final analysis repeats, because that is
where the quotes are.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from juris.eval.metrics.common import (
    MetricResult,
    analysis_refs,
    argument_citations,
    chunk_text,
    pair_results,
    ratio,
)
from juris.events.fold import CaseView
from juris.models import Citation
from juris.models.common import EntailmentLabel
from juris.verify.quote import quote_in_text


def _invalid_reason(view: CaseView, citation: Citation) -> str | None:
    """The deterministic checks, cheapest first (IDEA_final §6.1 checks 1-2)."""
    if citation.evidence_id not in view.evidence:
        return "unknown evidence"
    text = chunk_text(view, citation.evidence_id)
    if text is None:
        return "chunk not in the record"
    if not quote_in_text(citation.quote, text):
        return "quote not in the chunk"
    return None


def citation_validity(view: CaseView) -> MetricResult:
    """Share of citations that reference registered evidence with an exact quote.

    Sub-values: ``.existence`` (the evidence ID is registered) and ``.analysis_refs`` (the
    evidence IDs the final analysis cites are registered; it carries no quotes).
    """
    citations = argument_citations(view)
    invalid = [
        {"claim_id": c.claim_id, "evidence_id": c.evidence_id, "reason": reason, "quote": c.quote}
        for c in citations
        if (reason := _invalid_reason(view, c)) is not None
    ]
    unknown = sum(1 for i in invalid if i["reason"] == "unknown evidence")
    refs = analysis_refs(view.analysis) if view.analysis is not None else []
    unknown_refs = [r for r in refs if r not in view.evidence]
    return MetricResult(
        "citation_validity",
        {
            "citation_validity": ratio(len(citations) - len(invalid), len(citations)),
            "citation_validity.existence": ratio(len(citations) - unknown, len(citations)),
            "citation_validity.analysis_refs": ratio(len(refs) - len(unknown_refs), len(refs)),
        },
        {"citations": len(citations), "invalid": invalid, "unknown_analysis_refs": unknown_refs},
    )


@dataclass(frozen=True)
class ClaimEvidencePair:
    """What the entailment check reads: does ``passage`` (quoted) support ``claim``?"""

    claim_id: str
    claim: str
    evidence_id: str
    quote: str
    passage: str


class EntailmentJudge(Protocol):
    """The verifier's entailment check (PLAN 6.1), for pairs the run did not verify."""

    async def __call__(self, pairs: Sequence[ClaimEvidencePair]) -> list[EntailmentLabel]: ...


async def citation_faithfulness(
    view: CaseView, judge: EntailmentJudge | None = None
) -> MetricResult:
    """Share of claim-citation pairs whose quoted passage supports the claim.

    A citation that fails existence or the exact quote counts as not supporting. Otherwise
    the label is the run's own latest verification of that (claim, evidence) pair; pairs
    the run never verified go to ``judge``, and without one they are left out as unjudged.
    ``.lenient`` also counts ``partially_supports``.
    """
    recorded = pair_results(view)
    outcomes: list[str] = []
    pending: list[tuple[int, ClaimEvidencePair]] = []
    for citation in argument_citations(view):
        if _invalid_reason(view, citation) is not None:
            outcomes.append("invalid")
            continue
        result = recorded.get((citation.claim_id, citation.evidence_id))
        if result is not None and result.label is not None:
            outcomes.append(result.label.value)
            continue
        claim = view.claims.get(citation.claim_id)
        outcomes.append("unjudged")
        pending.append(
            (
                len(outcomes) - 1,
                ClaimEvidencePair(
                    claim_id=citation.claim_id,
                    claim=claim.text if claim is not None else "",
                    evidence_id=citation.evidence_id,
                    quote=citation.quote,
                    passage=chunk_text(view, citation.evidence_id) or "",
                ),
            )
        )
    if judge is not None and pending:
        labels = await judge([pair for _, pair in pending])
        if len(labels) != len(pending):
            raise ValueError(f"the judge returned {len(labels)} labels for {len(pending)} pairs")
        for (index, _), label in zip(pending, labels, strict=True):
            outcomes[index] = label.value
    counts = Counter(outcomes)
    judged = len(outcomes) - counts["unjudged"]
    supports = counts[EntailmentLabel.SUPPORTS.value]
    partial = counts[EntailmentLabel.PARTIALLY_SUPPORTS.value]
    return MetricResult(
        "citation_faithfulness",
        {
            "citation_faithfulness": ratio(supports, judged),
            "citation_faithfulness.lenient": ratio(supports + partial, judged),
        },
        {"pairs": len(outcomes), "outcomes": dict(sorted(counts.items()))},
    )
