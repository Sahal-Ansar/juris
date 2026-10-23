"""Graph integrity metrics (PLAN 8.2): the evidence graph's checks (``juris.graph.checks``) as
rates over the run's claims, positions and issues.

- ``orphan_claim_rate``: claims none of whose citations verified or was weak.
- ``weak_only_claim_rate``: claims supported only by weak citations.
- ``unverified_position_rate``: positions with no standing claim citing verified evidence.
- ``contested_issue_rate``: issues where two or more positions have verified support.
- ``circular_support``: loops in the support relation (a count; 0 on a sound graph).

Lower is better for all but the contested rate, which describes the case rather than the
run's quality. A rate is None when the run has nothing to count (B0 records no claims).
"""

from juris.eval.metrics.common import MetricResult, ratio
from juris.events.fold import CaseView
from juris.graph.checks import run_checks

GRAPH_METRICS = (
    "orphan_claim_rate",
    "weak_only_claim_rate",
    "unverified_position_rate",
    "contested_issue_rate",
    "circular_support",
)


def graph_checks(view: CaseView) -> list[MetricResult]:
    """One result per check, each with the IDs it found in ``details``."""
    checks = run_checks(view.graph)
    claims, positions, issues = len(checks.claims), len(checks.positions), len(checks.issues)
    return [
        MetricResult(
            "orphan_claim_rate",
            {"orphan_claim_rate": ratio(len(checks.orphan_claims), claims)},
            {"claims": claims, "ids": checks.orphan_claims},
        ),
        MetricResult(
            "weak_only_claim_rate",
            {"weak_only_claim_rate": ratio(len(checks.weak_only_claims), claims)},
            {"claims": claims, "ids": checks.weak_only_claims},
        ),
        MetricResult(
            "unverified_position_rate",
            {"unverified_position_rate": ratio(len(checks.unverified_positions), positions)},
            {"positions": positions, "ids": checks.unverified_positions},
        ),
        MetricResult(
            "contested_issue_rate",
            {"contested_issue_rate": ratio(len(checks.contested_issues), issues)},
            {"issues": issues, "ids": checks.contested_issues},
        ),
        MetricResult(
            "circular_support",
            {"circular_support": float(len(checks.cycles)) if claims else None},
            {"cycles": checks.cycles},
        ),
    ]
