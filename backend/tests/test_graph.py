"""The evidence graph (PLAN 8.1): built from the Case Record, valid at every point of a run."""

import json
from collections import Counter

from juris.config import REPO_ROOT
from juris.events import CaseView, fold, parse_event
from juris.graph.builder import build_graph, check_graph, section_id
from juris.models import CreatedBy, Objection, ObjectionType, Stage
from juris.models.graph import EdgeType, GraphEdge, GraphNode, NodeType

RUNS = REPO_ROOT / "fixtures" / "runs"


def view(name: str = "coercion_full") -> CaseView:
    lines = (RUNS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return fold(parse_event(json.loads(line)) for line in lines if line.strip())


def edges(v: CaseView, type: EdgeType) -> dict[tuple[str, str], GraphEdge]:
    return {(e.source, e.target): e for e in v.graph.edges if e.type is type}


def test_full_run_graph_is_valid_and_has_every_type() -> None:
    v = view()
    graph = v.graph
    assert check_graph(graph, v) == []
    assert {n.type for n in graph.nodes} == set(NodeType)
    assert {e.type for e in graph.edges} == set(EdgeType)
    counts = Counter(n.type for n in graph.nodes)
    assert counts[NodeType.EVIDENCE] == len(v.evidence) == 20
    assert counts[NodeType.CLAIM] == len(v.claims)
    assert counts[NodeType.OBJECTION] == len(v.objections)
    assert build_graph(v) == graph  # deterministic


def test_citations_carry_their_best_verification() -> None:
    cites = edges(view(), EdgeType.CITES)
    invalid = cites[("C-008", "E-018")]
    assert invalid.verification is not None and invalid.verification.value == "invalid"
    assert invalid.weight == 0.0 and invalid.argument_id is not None
    assert cites[("C-007", "E-018")].weight == 0.5  # weak
    assert cites[("C-002", "E-006")].verification.value == "unsupported"  # type: ignore[union-attr]
    assert all(e.pinpoint for e in cites.values())


def test_claims_support_positions_by_status() -> None:
    v = view()
    supports = edges(v, EdgeType.SUPPORTS)
    assert supports[("C-003", v.claims["C-003"].position_id)].weight == 1.0  # survives
    assert supports[("C-006", v.claims["C-006"].position_id)].weight == 0.5  # weak
    assert supports[("C-002", v.claims["C-002"].position_id)].weight == 0.0  # falls


def test_rebuttal_precedent_and_objection_edges() -> None:
    v = view()
    graph = v.graph
    challenges = edges(v, EdgeType.CHALLENGES)
    supports = edges(v, EdgeType.SUPPORTS)
    for counter in v.counterarguments.values():
        assert (counter.id, counter.targets) in challenges
        assert (counter.argument_id, counter.id) in supports
    fit = challenges[("MOCK-SC-1962-01", "A-001")]
    assert fit.fit is not None and fit.fit.value == "distinguishable" and fit.weight == 0.5
    assert supports[("MOCK-SC-1978-02", "A-002")].weight == 0.0  # does not support
    treated = edges(v, EdgeType.CASE_CITES_CASE)
    assert treated[("MOCK-SC-1994-03", "MOCK-SC-1962-01")].treatment.value == "followed"  # type: ignore[union-attr]
    targets = edges(v, EdgeType.TARGETS)
    assert {(o.id, o.target_id) for o in v.objections.values()} == set(targets)
    nodes = {n.id: n for n in graph.nodes}
    assert nodes["O-002"].resolved is True and nodes["O-001"].resolved is False
    assert nodes["MOCK-SC-2003-04"].treatment_verified is False


def test_statute_evidence_quotes_its_section() -> None:
    v = view()
    quotes = edges(v, EdgeType.QUOTES)
    for evidence in v.evidence.values():
        chunk = v.chunks[evidence.chunk_id]
        source = section_id(chunk) or chunk.document_id
        assert (evidence.id, source) in quotes
    sections = [n for n in v.graph.nodes if n.type is NodeType.STATUTE_SECTION]
    assert sections and all(n.id not in v.documents for n in sections)


def test_graph_is_valid_after_every_event() -> None:
    lines = (RUNS / "coercion_full.jsonl").read_text(encoding="utf-8").splitlines()
    events = [parse_event(json.loads(line)) for line in lines if line.strip()]
    v = fold(events[:1])
    for event in events[1:]:
        v = fold([event], start=v)
        assert check_graph(v.graph, v) == [], event.type


def test_graph_is_serialised_and_rebuilt_on_read() -> None:
    v = view()
    data = json.loads(v.model_dump_json())
    assert len(data["graph"]["nodes"]) == len(v.graph.nodes)
    data["graph"] = {"nodes": [], "edges": []}  # a stale graph is ignored, not trusted
    back = CaseView.model_validate(data)
    assert back == v and back.graph == v.graph
    failed = view("failure_case")
    assert check_graph(failed.graph, failed) == []


def test_unknown_ends_are_left_out_and_checks_catch_bad_graphs() -> None:
    v = view()
    stray = Objection(
        id="O-099",
        type=ObjectionType.LOGICAL_GAP,
        target_id="C-999",
        raised_by="cross_examiner",
        text="Aimed at nothing.",
        created_by=CreatedBy(stage=Stage.S6, agent="cross_examiner"),
    )
    v.objections["O-099"] = stray
    graph = v.graph
    assert check_graph(graph, v) == []
    assert not [e for e in graph.edges if e.source == "O-099"]
    bad = graph.model_copy(
        update={
            "nodes": [*graph.nodes, GraphNode(id="C-500", type=NodeType.CLAIM, label="x")],
            "edges": [
                *graph.edges,
                graph.edges[0],
                GraphEdge(source="E-001", target="I-9", type=EdgeType.ABOUT_ISSUE),
                GraphEdge(source="I-1", target="C-001", type=EdgeType.CITES),
            ],
        }
    )
    problems = check_graph(bad, v)
    assert any("C-500 is not in the record" in p for p in problems)
    assert any("appears twice" in p for p in problems)
    assert any("not a node" in p for p in problems)
    assert any("not allowed (issue to claim)" in p for p in problems)
