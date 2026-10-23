"""Build and check the evidence graph of saved runs (PLAN 8.1).

    uv run scripts/graph_stats.py eval/results/juris_full-dev-n5-s1
    uv run scripts/graph_stats.py <run dir or caseview.json> ...

Finds every ``caseview.json`` under the given paths, builds its graph and checks it
(``juris.graph.builder.check_graph``). Prints one line per run (nodes, edges, problems), then
node and edge counts by type over all of them. Exits 1 if any graph has a problem.
"""

import sys
from collections import Counter
from pathlib import Path

from juris.events import CaseView
from juris.graph.builder import check_graph


def caseviews(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        found += [path] if path.is_file() else sorted(path.rglob("caseview.json"))
    return found


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    nodes: Counter[str] = Counter()
    edges: Counter[str] = Counter()
    cites: Counter[str] = Counter()
    bad = 0
    files = caseviews([Path(a) for a in argv])
    for file in files:
        view = CaseView.model_validate_json(file.read_text(encoding="utf-8"))
        graph = view.graph
        problems = check_graph(graph, view)
        bad += bool(problems)
        nodes.update(n.type.value for n in graph.nodes)
        edges.update(e.type.value for e in graph.edges)
        cites.update(
            e.verification.value if e.verification else "unchecked"
            for e in graph.edges
            if e.type.value == "cites"
        )
        print(f"{file.parent.name}: {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
              f"{len(problems)} problems")  # fmt: skip
        for problem in problems:
            print(f"  - {problem}")
    print(f"\n{len(files)} runs, {bad} with problems")
    print("nodes:", ", ".join(f"{k} {v}" for k, v in nodes.most_common()))
    print("edges:", ", ".join(f"{k} {v}" for k, v in edges.most_common()))
    print("citations:", ", ".join(f"{k} {v}" for k, v in cites.most_common()))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
