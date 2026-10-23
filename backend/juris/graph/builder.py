"""``build_graph(view) -> EvidenceGraph``: the Case Record as nodes and edges (PLAN 8.1).

Pure and deterministic: nodes and edges come out in record order, so the same view always
gives the same graph. Every node is an object in the record; an edge whose end is not a node
(an objection aimed at an unknown ID, a treatment by a judgment outside the record) is left
out. ``check_graph`` states those rules and returns what breaks them.

Weights (0-1) say how strong a relation is: a citation's best verification result (verified
1, weak 0.5, unsupported or invalid 0), a claim's status for its position (verified or
survives 1, weak or contested 0.5, unsupported or falls 0, proposed unknown), and a
precedent's fit with an argument (directly 1, by analogy 0.5, not at all 0; as a challenge,
contrary 1, distinguishable 0.5).
"""

from typing import TYPE_CHECKING

from juris.models import Chunk, ClaimStatus, PrecedentFit, VerificationStatus
from juris.models.graph import EdgeType, EvidenceGraph, GraphEdge, GraphNode, NodeType
from juris.pipeline.citing import pinpoint

if TYPE_CHECKING:
    from juris.events.fold import CaseView, VerificationRecord

LABEL_CHARS = 160

_RANK = (
    VerificationStatus.INVALID,
    VerificationStatus.UNSUPPORTED,
    VerificationStatus.WEAK,
    VerificationStatus.VERIFIED,
)
_CITATION_WEIGHT = {
    VerificationStatus.VERIFIED: 1.0,
    VerificationStatus.WEAK: 0.5,
    VerificationStatus.UNSUPPORTED: 0.0,
    VerificationStatus.INVALID: 0.0,
}
_CLAIM_WEIGHT: dict[ClaimStatus, float | None] = {
    ClaimStatus.PROPOSED: None,
    ClaimStatus.VERIFIED: 1.0,
    ClaimStatus.SURVIVES: 1.0,
    ClaimStatus.WEAK: 0.5,
    ClaimStatus.CONTESTED: 0.5,
    ClaimStatus.UNSUPPORTED: 0.0,
    ClaimStatus.FALLS: 0.0,
}
_FIT: dict[PrecedentFit, tuple[EdgeType, float]] = {
    PrecedentFit.DIRECTLY_SUPPORTS: (EdgeType.SUPPORTS, 1.0),
    PrecedentFit.SUPPORTS_BY_ANALOGY: (EdgeType.SUPPORTS, 0.5),
    PrecedentFit.DOES_NOT_SUPPORT: (EdgeType.SUPPORTS, 0.0),
    PrecedentFit.DISTINGUISHABLE: (EdgeType.CHALLENGES, 0.5),
    PrecedentFit.CONTRARY: (EdgeType.CHALLENGES, 1.0),
}

# (source type, edge type, target type) that may appear; check_graph enforces it.
ALLOWED: frozenset[tuple[NodeType, EdgeType, NodeType]] = frozenset(
    {
        (NodeType.POSITION, EdgeType.ABOUT_ISSUE, NodeType.ISSUE),
        (NodeType.EVIDENCE, EdgeType.ABOUT_ISSUE, NodeType.ISSUE),
        (NodeType.CLAIM, EdgeType.SUPPORTS, NodeType.POSITION),
        (NodeType.ARGUMENT, EdgeType.SUPPORTS, NodeType.CLAIM),
        (NodeType.ARGUMENT, EdgeType.SUPPORTS, NodeType.COUNTERARGUMENT),
        (NodeType.DOCUMENT, EdgeType.SUPPORTS, NodeType.ARGUMENT),
        (NodeType.DOCUMENT, EdgeType.CHALLENGES, NodeType.ARGUMENT),
        (NodeType.COUNTERARGUMENT, EdgeType.CHALLENGES, NodeType.CLAIM),
        (NodeType.CLAIM, EdgeType.CITES, NodeType.EVIDENCE),
        (NodeType.EVIDENCE, EdgeType.QUOTES, NodeType.DOCUMENT),
        (NodeType.EVIDENCE, EdgeType.QUOTES, NodeType.STATUTE_SECTION),
        (NodeType.DOCUMENT, EdgeType.CASE_CITES_CASE, NodeType.DOCUMENT),
    }
    | {(NodeType.OBJECTION, EdgeType.TARGETS, t) for t in NodeType}
)


def short(text: str, limit: int = LABEL_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def section_id(chunk: Chunk) -> str | None:
    """The statute section a chunk belongs to (``ACT-contract_act#s74``), or None."""
    return f"{chunk.document_id}#s{chunk.statute.section}" if chunk.statute else None


def best_results(view: "CaseView") -> dict[tuple[str, str], "VerificationRecord"]:
    """Each (claim, evidence) pair's best verification result over every gate."""
    out: dict[tuple[str, str], VerificationRecord] = {}
    for record in view.verifications:
        key = (record.claim_id, record.evidence_id)
        held = out.get(key)
        if held is None or _RANK.index(record.result.status) > _RANK.index(held.result.status):
            out[key] = record
    return out


class _Builder:
    def __init__(self, view: "CaseView") -> None:
        self.view = view
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[GraphEdge, None] = {}  # an ordered set
        self.position_issue = {p.id: i.id for i in view.issues.values() for p in i.positions}

    def node(self, node: GraphNode) -> None:
        self.nodes.setdefault(node.id, node)

    def edge(self, source: str, target: str, type: EdgeType, **attrs: object) -> None:
        """Add the edge if both ends are nodes and it is not there yet."""
        if source in self.nodes and target in self.nodes:
            edge = GraphEdge(source=source, target=target, type=type, **attrs)  # type: ignore[arg-type]
            self.edges.setdefault(edge)

    def issues_of(self, position_ids: list[str]) -> list[str]:
        found = [self.position_issue.get(p) for p in position_ids]
        return list(dict.fromkeys(i for i in found if i is not None))

    def build(self) -> EvidenceGraph:
        view = self.view
        for issue in view.issues.values():
            self.node(
                GraphNode(
                    id=issue.id,
                    type=NodeType.ISSUE,
                    label=short(issue.question),
                    issue_ids=[issue.id],
                )
            )
            for position in issue.positions:
                self.node(
                    GraphNode(
                        id=position.id,
                        type=NodeType.POSITION,
                        label=short(position.statement),
                        issue_ids=[issue.id],
                        position_id=position.id,
                    )
                )
                self.edge(position.id, issue.id, EdgeType.ABOUT_ISSUE)
        self.sources()
        for claim in view.claims.values():
            self.node(
                GraphNode(
                    id=claim.id,
                    type=NodeType.CLAIM,
                    label=short(claim.text),
                    issue_ids=self.issues_of([claim.position_id]),
                    position_id=claim.position_id,
                    claim_status=claim.status,
                )
            )
            self.edge(
                claim.id, claim.position_id, EdgeType.SUPPORTS, weight=_CLAIM_WEIGHT[claim.status]
            )
        self.arguments()
        self.counterarguments()
        self.precedents()
        self.objections()
        return EvidenceGraph(nodes=list(self.nodes.values()), edges=list(self.edges))

    def sources(self) -> None:
        """Evidence, and the document or statute section each one quotes."""
        view = self.view
        for evidence in view.evidence.values():
            chunk = view.chunks.get(evidence.chunk_id)
            document = view.documents.get(evidence.document_id)
            if chunk is None or document is None:
                continue
            where = pinpoint(chunk, None)
            section = section_id(chunk)
            if section is not None and chunk.statute is not None:
                statute = chunk.statute
                name = f"{statute.act}, s. {statute.section}"
                source = GraphNode(
                    id=section,
                    type=NodeType.STATUTE_SECTION,
                    label=short(f"{name}: {statute.title}" if statute.title else name),
                )
            else:
                card = view.precedent_cards.get(document.id)
                source = GraphNode(
                    id=document.id,
                    type=NodeType.DOCUMENT,
                    label=short(document.title),
                    treatment_verified=card.treatment_verified if card else None,
                )
            held = self.nodes.get(source.id)
            issues = list(dict.fromkeys([*(held.issue_ids if held else []), *evidence.issue_ids]))
            self.nodes[source.id] = source.model_copy(update={"issue_ids": issues})
            self.node(
                GraphNode(
                    id=evidence.id,
                    type=NodeType.EVIDENCE,
                    label=short(f"{document.title}, {where}"),
                    issue_ids=list(evidence.issue_ids),
                )
            )
            for issue_id in evidence.issue_ids:
                self.edge(evidence.id, issue_id, EdgeType.ABOUT_ISSUE)
            self.edge(evidence.id, source.id, EdgeType.QUOTES, pinpoint=where)

    def arguments(self) -> None:
        view = self.view
        best = best_results(view)
        seen: set[tuple[str, str, str]] = set()
        for argument in view.arguments.values():
            positions = [view.claims[c].position_id for c in argument.claim_ids if c in view.claims]
            self.node(
                GraphNode(
                    id=argument.id,
                    type=NodeType.ARGUMENT,
                    label=short(argument.reasoning),
                    issue_ids=self.issues_of(positions),
                    position_id=positions[0] if positions else None,
                )
            )
            for claim_id in argument.claim_ids:
                self.edge(argument.id, claim_id, EdgeType.SUPPORTS)
            for citation in argument.citations:
                key = (citation.claim_id, citation.evidence_id, argument.id)
                if key in seen:
                    continue
                seen.add(key)
                record = best.get(key[:2])
                result = record.result if record else None
                self.edge(
                    citation.claim_id,
                    citation.evidence_id,
                    EdgeType.CITES,
                    weight=_CITATION_WEIGHT[result.status] if result else None,
                    verification=result.status if result else None,
                    label=result.label if result else None,
                    pinpoint=citation.pinpoint,
                    argument_id=argument.id,
                )

    def counterarguments(self) -> None:
        view = self.view
        for counter in view.counterarguments.values():
            target = view.claims.get(counter.targets)
            argument = self.nodes.get(counter.argument_id)
            label = view.arguments[counter.argument_id].reasoning if argument else counter.targets
            self.node(
                GraphNode(
                    id=counter.id,
                    type=NodeType.COUNTERARGUMENT,
                    label=short(label),
                    issue_ids=self.issues_of([target.position_id]) if target else [],
                    position_id=argument.position_id if argument else None,
                )
            )
            self.edge(counter.argument_id, counter.id, EdgeType.SUPPORTS)
            self.edge(counter.id, counter.targets, EdgeType.CHALLENGES)

    def precedents(self) -> None:
        for card in self.view.precedent_cards.values():
            for argument_id, fit in card.fit_by_argument.items():
                kind, weight = _FIT[fit]
                self.edge(card.document_id, argument_id, kind, weight=weight, fit=fit)
            for signal in card.treatment:
                self.edge(
                    signal.by_document_id,
                    card.document_id,
                    EdgeType.CASE_CITES_CASE,
                    treatment=signal.kind,
                )

    def objections(self) -> None:
        for objection in self.view.objections.values():
            target = self.nodes.get(objection.target_id)
            self.node(
                GraphNode(
                    id=objection.id,
                    type=NodeType.OBJECTION,
                    label=short(objection.text),
                    issue_ids=list(target.issue_ids) if target else [],
                    position_id=target.position_id if target else None,
                    resolved=objection.resolved,
                )
            )
            self.edge(objection.id, objection.target_id, EdgeType.TARGETS)


def build_graph(view: "CaseView") -> EvidenceGraph:
    """The evidence graph of a (possibly unfinished) run."""
    return _Builder(view).build()


def in_record(node: GraphNode, view: "CaseView") -> bool:
    """Whether the node's ID names an object of its type in the record."""
    match node.type:
        case NodeType.ISSUE:
            return node.id in view.issues
        case NodeType.POSITION:
            return any(p.id == node.id for i in view.issues.values() for p in i.positions)
        case NodeType.CLAIM:
            return node.id in view.claims
        case NodeType.ARGUMENT:
            return node.id in view.arguments
        case NodeType.COUNTERARGUMENT:
            return node.id in view.counterarguments
        case NodeType.OBJECTION:
            return node.id in view.objections
        case NodeType.EVIDENCE:
            return node.id in view.evidence
        case NodeType.DOCUMENT:
            return node.id in view.documents
        case NodeType.STATUTE_SECTION:
            return any(section_id(c) == node.id for c in view.chunks.values())


def check_graph(graph: EvidenceGraph, view: "CaseView") -> list[str]:
    """What is wrong with a graph of this view: an empty list if it is valid."""
    problems: list[str] = []
    types: dict[str, NodeType] = {}
    for node in graph.nodes:
        if node.id in types:
            problems.append(f"node {node.id} appears twice")
        types[node.id] = node.type
        if not in_record(node, view):
            problems.append(f"{node.type.value} node {node.id} is not in the record")
    seen: set[GraphEdge] = set()
    for edge in graph.edges:
        name = f"{edge.type.value} edge {edge.source} -> {edge.target}"
        if edge.source not in types or edge.target not in types:
            problems.append(f"{name} has an end that is not a node")
            continue
        if (types[edge.source], edge.type, types[edge.target]) not in ALLOWED:
            problems.append(
                f"{name} is not allowed ({types[edge.source].value} to {types[edge.target].value})"
            )
        if edge in seen:
            problems.append(f"{name} appears twice")
        seen.add(edge)
    return problems
