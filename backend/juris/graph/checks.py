"""Integrity checks over the evidence graph (PLAN 8.2): where the record's support is thin.

``run_checks(graph)`` reads the support the graph records and finds:

- **orphan claims**: no citation of the claim verified or was weak (all unsupported, invalid
  or unchecked, or none at all), so nothing in the record supports it;
- **weak-only claims**: supported, but only by weak (partial) citations, none verified;
- **unverified positions**: no standing claim of the position (status not falls or
  unsupported) cites verified evidence;
- **contested issues**: two or more positions of the issue have verified support, so the
  verified evidence pulls both ways;
- **circular support**: support that leads back to where it started. Support runs from
  evidence to the claim citing it, from a document or statute section to the evidence quoting
  it, and along ``supports`` edges (argument to claim, claim to position, ...). The builder's
  edges cannot close a loop today; the check guards the graph against new edge kinds.

The results are metrics (``juris.eval.metrics.graph``) and, as ``hints``, limitations the
Judge sees before S9 and ``build_analysis`` lists.
"""

from dataclasses import dataclass, field

from juris.models.common import ClaimStatus, VerificationStatus
from juris.models.graph import EdgeType, EvidenceGraph, GraphNode, NodeType

FALLEN = frozenset({ClaimStatus.FALLS, ClaimStatus.UNSUPPORTED})


@dataclass
class GraphChecks:
    """IDs in graph order. ``claims``, ``positions`` and ``issues`` are the totals checked."""

    claims: list[str] = field(default_factory=list)
    positions: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    orphan_claims: list[str] = field(default_factory=list)
    weak_only_claims: list[str] = field(default_factory=list)
    unverified_positions: list[str] = field(default_factory=list)
    contested_issues: list[str] = field(default_factory=list)
    cycles: list[list[str]] = field(default_factory=list)


def support_edges(graph: EvidenceGraph) -> dict[str, list[str]]:
    """Each node's successors in the support relation, from the supporter to the supported."""
    out: dict[str, list[str]] = {}
    for edge in graph.edges:
        if edge.type is EdgeType.SUPPORTS:
            out.setdefault(edge.source, []).append(edge.target)
        elif edge.type in (EdgeType.CITES, EdgeType.QUOTES):  # these point at the supporter
            out.setdefault(edge.target, []).append(edge.source)
    return out


def cycles(nodes: list[str], successors: dict[str, list[str]]) -> list[list[str]]:
    """Strongly connected components with a loop (Tarjan's, without recursion), each in node
    order."""
    order = {n: i for i, n in enumerate(nodes)}
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    found: list[list[str]] = []

    def visit(node: str) -> None:
        index[node] = low[node] = len(index)
        stack.append(node)
        on_stack.add(node)

    for root in nodes:
        if root in index:
            continue
        visit(root)
        work = [(root, iter(successors.get(root, [])))]
        while work:
            node, rest = work[-1]
            for nxt in rest:
                if nxt not in index:
                    visit(nxt)
                    work.append((nxt, iter(successors.get(nxt, []))))
                    break
                if nxt in on_stack:
                    low[node] = min(low[node], index[nxt])
            else:
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1 or node in successors.get(node, []):
                        found.append(sorted(component, key=lambda n: order.get(n, len(order))))
    return found


def run_checks(graph: EvidenceGraph) -> GraphChecks:
    """The five checks on one graph (see the module docstring)."""
    nodes = {n.id: n for n in graph.nodes}
    of_type = {t: [n for n in graph.nodes if n.type is t] for t in NodeType}
    cited: dict[str, set[VerificationStatus | None]] = {}
    backing: dict[str, list[str]] = {}  # position -> its standing claims
    for edge in graph.edges:
        if edge.type is EdgeType.CITES:
            cited.setdefault(edge.source, set()).add(edge.verification)
        elif edge.type is EdgeType.SUPPORTS and _is(nodes, edge.target, NodeType.POSITION):
            claim = nodes.get(edge.source)
            if claim and claim.type is NodeType.CLAIM and claim.claim_status not in FALLEN:
                backing.setdefault(edge.target, []).append(claim.id)
    checks = GraphChecks(
        claims=[n.id for n in of_type[NodeType.CLAIM]],
        positions=[n.id for n in of_type[NodeType.POSITION]],
        issues=[n.id for n in of_type[NodeType.ISSUE]],
    )
    verified: set[str] = set()
    for claim_id in checks.claims:
        results = cited.get(claim_id, set())
        if VerificationStatus.VERIFIED in results:
            verified.add(claim_id)
        elif VerificationStatus.WEAK in results:
            checks.weak_only_claims.append(claim_id)
        else:
            checks.orphan_claims.append(claim_id)
    supported = {p for p in checks.positions if any(c in verified for c in backing.get(p, []))}
    checks.unverified_positions = [p for p in checks.positions if p not in supported]
    by_issue: dict[str, int] = {}
    for position_id in supported:
        for issue_id in nodes[position_id].issue_ids:
            by_issue[issue_id] = by_issue.get(issue_id, 0) + 1
    checks.contested_issues = [i for i in checks.issues if by_issue.get(i, 0) >= 2]
    checks.cycles = cycles([n.id for n in graph.nodes], support_edges(graph))
    return checks


def _is(nodes: dict[str, GraphNode], node_id: str, type: NodeType) -> bool:
    node = nodes.get(node_id)
    return node is not None and node.type is type


def _ids(ids: list[str]) -> str:
    return ", ".join(ids)


def hints(graph: EvidenceGraph, checks: GraphChecks | None = None) -> dict[str, list[str]]:
    """The checks as sentences for the Judge, by issue ("" for those on no issue). Claims that
    fell or were found unsupported are left out: the record already says they count for
    nothing."""
    checks = checks or run_checks(graph)
    nodes = {n.id: n for n in graph.nodes}
    out: dict[str, list[str]] = {}

    def add(issue_ids: list[str], text: str) -> None:
        for issue_id in issue_ids or [""]:
            out.setdefault(issue_id, []).append(text)

    unverified = set(checks.unverified_positions)
    for issue_id in checks.issues:
        positions = [p for p in checks.positions if issue_id in nodes[p].issue_ids]
        missing = [p for p in positions if p in unverified]
        if positions and len(missing) == len(positions):
            add([issue_id], f"On {issue_id} no position rests on verified evidence.")
        elif missing:
            add([issue_id], f"{_ids(missing)} {_has(missing)} no verified support.")
        if issue_id in checks.contested_issues:
            both = [p for p in positions if p not in unverified]
            add([issue_id], f"{issue_id} is contested: {_ids(both)} rest on verified evidence.")
    for found, text in (
        (checks.orphan_claims, "none of {its} citations verified or was weak"),
        (checks.weak_only_claims, "{its} support is only weak (partial)"),
    ):
        standing = [nodes[c] for c in found if nodes[c].claim_status not in FALLEN]
        for issue_id in dict.fromkeys(i for c in standing for i in c.issue_ids or [""]):
            ids = [c.id for c in standing if issue_id in (c.issue_ids or [""])]
            its = "its" if len(ids) == 1 else "their"
            add([issue_id], f"{_claims(ids)} still {_stand(ids)}, but {text.format(its=its)}.")
    for loop in checks.cycles:
        issues = list(dict.fromkeys(i for n in loop for i in nodes[n].issue_ids))
        many = len(loop) > 1
        what = f"{_ids(loop)} support one another" if many else f"{loop[0]} supports itself"
        add(issues, f"Circular support: {what}.")
    return out


def _has(ids: list[str]) -> str:
    return "has" if len(ids) == 1 else "have"


def _claims(ids: list[str]) -> str:
    return f"Claim {ids[0]}" if len(ids) == 1 else f"Claims {_ids(ids)}"


def _stand(ids: list[str]) -> str:
    return "stands" if len(ids) == 1 else "stand"
