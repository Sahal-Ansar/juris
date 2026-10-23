"""Answer metrics (PLAN 5.4, IDEA_final §11.2): score any config's ``CaseAnalysis`` and the
record behind it against a Juris-Eval item.

- ``citation_validity``: citations with registered evidence and an exact quote.
- ``citation_faithfulness``: claim-citation pairs whose passage supports the claim.
- ``unsupported_claim_rate``: final-output statements without verified support.
- ``authority_recall``: gold supporting/contrary authorities and sections in the analysis.
- ``key_point_coverage`` and ``quality_rubric``: LLM graders (``Grader``).
- ``cost_latency``: LLM calls, tokens, dollars, wall-clock.
- Graph checks (PLAN 8.2, ``graph_checks``): orphan and weak-only claims, unverified
  positions, contested issues and circular support in the run's evidence graph.

``score_answers`` runs them all and aggregates with bootstrap 95% CIs.
"""

from juris.eval.metrics.authorities import authority_recall
from juris.eval.metrics.citations import (
    ClaimEvidencePair,
    EntailmentJudge,
    citation_faithfulness,
    citation_validity,
)
from juris.eval.metrics.claims import unsupported_claim_rate
from juris.eval.metrics.common import MetricResult
from juris.eval.metrics.cost import cost_latency
from juris.eval.metrics.graders import Grader, key_point_coverage, quality_rubric
from juris.eval.metrics.graph import GRAPH_METRICS, graph_checks
from juris.eval.metrics.suite import METRICS, Answer, ItemScores, SuiteResult, score_answers

__all__ = [
    "GRAPH_METRICS",
    "METRICS",
    "Answer",
    "ClaimEvidencePair",
    "EntailmentJudge",
    "Grader",
    "ItemScores",
    "MetricResult",
    "SuiteResult",
    "authority_recall",
    "citation_faithfulness",
    "citation_validity",
    "cost_latency",
    "graph_checks",
    "key_point_coverage",
    "quality_rubric",
    "score_answers",
    "unsupported_claim_rate",
]
