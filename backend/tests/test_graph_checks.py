"""Integrity checks over the evidence graph (PLAN 8.2): each check on a small hand-made graph,
the Judge's hints, and the metrics on the mock runs."""

from juris.eval.metrics.graph import GRAPH_METRICS, graph_checks
from juris.graph.checks import cycles, hints, run_checks
from juris.models.common import ClaimStatus, VerificationStatus
from juris.models.graph import EdgeType, EvidenceGraph, GraphEdge, GraphNode, NodeType
from tests.test_graph import view

V, W, U, X = (
    VerificationStatus.VERIFIED,
    VerificationStatus.WEAK,
    VerificationStatus.UNSUPPORTED,
    VerificationStatus.INVALID,
)


def issue(id: str) -> GraphNode:
    return GraphNode(id=id, type=NodeType.ISSUE, label=id, issue_ids=[id])


def position(id: str) -> GraphNode:
    return GraphNode(id=id, type=NodeType.POSITION, label=id, issue_ids=[id.split(".")[0]])


def claim(id: str, position_id: str, status: ClaimStatus = ClaimStatus.SURVIVES) -> GraphNode:
    return GraphNode(
        id=id,
        type=NodeType.CLAIM,
        label=id,
        issue_ids=[position_id.split(".")[0]],
        position_id=position_id,
        claim_status=status,
    )


def evidence(id: str) -> GraphNode:
    return GraphNode(id=id, type=NodeType.EVIDENCE, label=id)


def supports(source: str, target: str) -> GraphEdge:
    return GraphEdge(source=source, target=target, type=EdgeType.SUPPORTS)


def cites(claim_id: str, evidence_id: str, status: VerificationStatus | None) -> GraphEdge:
    return GraphEdge(source=claim_id, target=evidence_id, type=EdgeType.CITES, verification=status)


def case() -> EvidenceGraph:
    """I-1: A has a standing verified claim (C-1), B only a fallen one (C-5). I-2: A and B both
    verified (contested); C has no claims. Claims C-2 (weak only), C-3 (nothing held), C-4 (no
    citations) are on I-1.A."""
    nodes = [issue("I-1"), issue("I-2")]
    nodes += [position(p) for p in ("I-1.A", "I-1.B", "I-2.A", "I-2.B", "I-2.C")]
    nodes += [
        claim("C-1", "I-1.A"),
        claim("C-2", "I-1.A"),
        claim("C-3", "I-1.A", ClaimStatus.CONTESTED),
        claim("C-4", "I-1.A", ClaimStatus.FALLS),
        claim("C-5", "I-1.B", ClaimStatus.FALLS),
        claim("C-6", "I-2.A"),
        claim("C-7", "I-2.B", ClaimStatus.PROPOSED),
    ]
    nodes += [evidence(f"E-{n}") for n in range(1, 5)]
    edges = [supports(n.id, n.position_id or "") for n in nodes if n.type is NodeType.CLAIM]
    edges += [
        cites("C-1", "E-1", V),
        cites("C-1", "E-2", U),
        cites("C-2", "E-2", W),
        cites("C-2", "E-3", X),
        cites("C-3", "E-3", U),
        cites("C-3", "E-4", None),  # not checked: no support yet
        cites("C-5", "E-1", V),
        cites("C-6", "E-1", V),
        cites("C-7", "E-4", V),
    ]
    return EvidenceGraph(nodes=nodes, edges=edges)


def test_orphan_and_weak_only_claims() -> None:
    checks = run_checks(case())
    assert checks.orphan_claims == ["C-3", "C-4"]  # nothing held; no citations at all
    assert checks.weak_only_claims == ["C-2"]  # a weak and an invalid citation, none verified
    assert len(checks.claims) == 7


def test_positions_with_no_verified_support() -> None:
    checks = run_checks(case())
    # I-1.B's only verified claim fell; I-2.C has no claims. A proposed claim (C-7) counts.
    assert checks.unverified_positions == ["I-1.B", "I-2.C"]


def test_contested_issues() -> None:
    assert run_checks(case()).contested_issues == ["I-2"]  # I-1 has one side verified


def test_circular_support() -> None:
    graph = case()
    assert run_checks(graph).cycles == []
    looped = graph.model_copy(
        update={
            "nodes": [*graph.nodes, GraphNode(id="A-1", type=NodeType.ARGUMENT, label="A-1")],
            "edges": [
                *graph.edges,
                supports("A-1", "C-6"),
                supports("I-2.A", "A-1"),  # the position backs the argument behind its own claim
                supports("C-2", "C-2"),  # a claim that supports itself
            ],
        }
    )
    # A-1 -> C-6 -> I-2.A -> A-1, and C-2 alone; members in graph order
    assert run_checks(looped).cycles == [["I-2.A", "C-6", "A-1"], ["C-2"]]
    found = hints(looped)
    assert found["I-1"][-1] == "Circular support: C-2 supports itself."
    assert found["I-2"][-1] == "Circular support: I-2.A, C-6, A-1 support one another."
    # cites and quotes count backwards: the evidence supports the claim citing it
    assert cycles(["C-1", "E-1"], {"E-1": ["C-1"], "C-1": ["E-1"]}) == [["C-1", "E-1"]]
    assert cycles(["a", "b", "c"], {"a": ["b"], "b": ["c"]}) == []


def test_hints_for_the_judge() -> None:
    found = hints(case())
    assert found == {
        "I-1": [
            "I-1.B has no verified support.",
            "Claim C-3 still stands, but none of its citations verified or was weak.",
            "Claim C-2 still stands, but its support is only weak (partial).",
        ],
        "I-2": [
            "I-2.C has no verified support.",
            "I-2 is contested: I-2.A, I-2.B rest on verified evidence.",
        ],
    }  # C-4 is an orphan too, but it fell
    bare = EvidenceGraph(nodes=[issue("I-1"), position("I-1.A"), position("I-1.B")])
    assert hints(bare) == {"I-1": ["On I-1 no position rests on verified evidence."]}


def test_checks_on_the_mock_runs() -> None:
    checks = run_checks(view().graph)
    assert checks.orphan_claims == ["C-002", "C-008"]  # both fell
    assert checks.weak_only_claims == ["C-006"]
    assert checks.unverified_positions == [] and checks.cycles == []
    assert checks.contested_issues == ["I-1", "I-2"]
    failed = run_checks(view("failure_case").graph)
    assert failed.orphan_claims == ["C-002"] and failed.unverified_positions == ["I-1.A", "I-1.B"]


def test_metrics() -> None:
    results = {r.name: r for r in graph_checks(view())}
    assert list(results) == list(GRAPH_METRICS)
    values = {name: r.value for name, r in results.items()}
    assert values == {
        "orphan_claim_rate": 0.2,
        "weak_only_claim_rate": 0.1,
        "unverified_position_rate": 0.0,
        "contested_issue_rate": 1.0,
        "circular_support": 0.0,
    }
    assert results["orphan_claim_rate"].details == {"claims": 10, "ids": ["C-002", "C-008"]}
