"""Unsupported-claim rate in the final output (IDEA_final §11.2).

The final output's statements are each position's summary (cited through its key evidence
and any inline ``E-###``) and each sentence of the overall summary (cited inline). A
statement is supported when at least one evidence item it cites has a ``verified`` or
``weak`` verification in the record.

An uncited position summary is unsupported. An uncited summary sentence is unsupported only
if it makes a legal assertion (§10: "every sentence carrying a legal assertion has ≥ 1
citation"); the grader decides which do (``graders.Grader.assertions``). Without that
decision, every uncited sentence counts, which overstates the rate for framing sentences
("On I-1 the record is balanced.").
"""

from collections.abc import Sequence
from dataclasses import dataclass

from juris.eval.metrics.common import (
    EVIDENCE_REF,
    SUPPORTED,
    MetricResult,
    evidence_support,
    pair_results,
    ratio,
    split_sentences,
)
from juris.events.fold import CaseView
from juris.models import CaseAnalysis


@dataclass(frozen=True)
class Statement:
    where: str  # "I-1.A" or "summary 2"
    text: str
    refs: tuple[str, ...]


def final_statements(analysis: CaseAnalysis) -> list[Statement]:
    out: list[Statement] = []
    for issue in analysis.issues:
        for position in issue.positions:
            refs = [r.evidence_id for r in position.key_evidence]
            refs += EVIDENCE_REF.findall(position.summary)
            out.append(
                Statement(position.position_id, position.summary, tuple(dict.fromkeys(refs)))
            )
    for n, sentence in enumerate(split_sentences(analysis.overall_summary), 1):
        refs = EVIDENCE_REF.findall(sentence)
        out.append(Statement(f"summary {n}", sentence, tuple(dict.fromkeys(refs))))
    return out


def uncited_sentences(analysis: CaseAnalysis) -> list[Statement]:
    """Summary sentences with no evidence ID: the ones the grader classifies."""
    return [s for s in final_statements(analysis) if s.where.startswith("summary") and not s.refs]


def unsupported_claim_rate(
    view: CaseView, legal_assertion: Sequence[bool] | None = None
) -> MetricResult:
    """Share of the final output's statements that lack verified support.

    ``legal_assertion`` gives, for each of ``uncited_sentences`` in order, whether it makes a
    legal assertion. Sub-values: ``.uncited`` (share of counted statements with no citation
    at all) and ``.record`` (share of the record's claims with no verified or weak support,
    whether or not the analysis repeats them).
    """
    support = evidence_support(view)
    supported_claims = {
        claim_id for (claim_id, _), r in pair_results(view).items() if r.status in SUPPORTED
    }
    record = ratio(len(set(view.claims) - supported_claims), len(view.claims))
    if view.analysis is None:
        return MetricResult(
            "unsupported_claim_rate",
            {
                "unsupported_claim_rate": None,
                "unsupported_claim_rate.uncited": None,
                "unsupported_claim_rate.record": record,
            },
            {"note": "no analysis"},
        )
    uncited = uncited_sentences(view.analysis)
    if legal_assertion is not None and len(legal_assertion) != len(uncited):
        raise ValueError(f"{len(legal_assertion)} decisions for {len(uncited)} uncited sentences")
    not_assertions = {
        s.where for s, legal in zip(uncited, legal_assertion or [], strict=False) if not legal
    }
    counted = [s for s in final_statements(view.analysis) if s.where not in not_assertions]
    unsupported = [s for s in counted if not any(support.get(r) in SUPPORTED for r in s.refs)]
    return MetricResult(
        "unsupported_claim_rate",
        {
            "unsupported_claim_rate": ratio(len(unsupported), len(counted)),
            "unsupported_claim_rate.uncited": ratio(
                sum(1 for s in counted if not s.refs), len(counted)
            ),
            "unsupported_claim_rate.record": record,
        },
        {
            "statements": len(counted),
            "not_legal_assertions": sorted(not_assertions),
            "assertions_classified": legal_assertion is not None,
            "unsupported": [{"where": s.where, "text": s.text} for s in unsupported],
        },
    )
