"""Run the evidence graph's integrity checks on saved runs (PLAN 8.2).

    uv run scripts/graph_checks.py eval/results/juris_full-dev-n5-s1
    uv run scripts/graph_checks.py <run dir or caseview.json> ... [--hints]

Finds every ``caseview.json`` under the given paths and runs ``juris.graph.checks`` on its
graph. Prints one line per run (claims, positions, issues and what each check found), then the
totals and the mean of each metric (``juris.eval.metrics.graph``) per experiment. ``--hints``
also prints the Judge's record checks for each run.
"""

import statistics
import sys
from pathlib import Path

from juris.eval.metrics.graph import GRAPH_METRICS, graph_checks
from juris.events import CaseView
from juris.graph.checks import hints, run_checks


def caseviews(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        found += [path] if path.is_file() else sorted(path.rglob("caseview.json"))
    return found


def experiment(file: Path) -> str:
    """The experiment directory's name (``<exp>/runs/<run>/caseview.json``), else the parent's."""
    run = file.parent
    return run.parent.parent.name if run.parent.name == "runs" else run.parent.name


def main(argv: list[str]) -> int:
    show_hints = "--hints" in argv
    paths = [Path(a) for a in argv if a != "--hints"]
    if not paths:
        print(__doc__)
        return 2
    totals: dict[str, dict[str, int]] = {}
    values: dict[str, dict[str, list[float]]] = {}
    for file in caseviews(paths):
        view = CaseView.model_validate_json(file.read_text(encoding="utf-8"))
        graph = view.graph
        checks = run_checks(graph)
        name = experiment(file)
        total = totals.setdefault(name, {})
        for key, found in vars(checks).items():
            total[key] = total.get(key, 0) + len(found)
        total["runs"] = total.get("runs", 0) + 1
        for result in graph_checks(view):
            if result.value is not None:
                values.setdefault(name, {}).setdefault(result.name, []).append(result.value)
        print(
            f"{name}/{file.parent.name}: {len(checks.claims)} claims, "
            f"{len(checks.positions)} positions, {len(checks.issues)} issues | "
            f"orphan {checks.orphan_claims}, weak-only {checks.weak_only_claims}, "
            f"unverified {checks.unverified_positions}, contested {checks.contested_issues}, "
            f"cycles {len(checks.cycles)}"
        )
        if show_hints:
            for issue_id, lines in hints(graph, checks).items():
                for line in lines:
                    print(f"    [{issue_id or 'case'}] {line}")
    print("\n| experiment | runs | claims | orphan | weak-only | positions | unverified | issues "
          "| contested | cycles |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, t in totals.items():
        print(
            f"| {name} | {t['runs']} | {t['claims']} | {t['orphan_claims']} | "
            f"{t['weak_only_claims']} | {t['positions']} | {t['unverified_positions']} | "
            f"{t['issues']} | {t['contested_issues']} | {t['cycles']} |"
        )
    print("\n| experiment | " + " | ".join(GRAPH_METRICS) + " |")
    print("|---|" + "---|" * len(GRAPH_METRICS))
    for name in totals:
        means = [values.get(name, {}).get(m) for m in GRAPH_METRICS]
        cells = [f"{statistics.mean(v):.3f}" if v else "n/a" for v in means]
        print(f"| {name} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
