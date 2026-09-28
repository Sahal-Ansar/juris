"""Extract the case and statute citation graph (PLAN 3.9, D-013) and report resolution rates.

Pass 1 learns the parallel-citation alias table from every judgment; pass 2 extracts and
resolves each citation. Writes ``<data_dir>/processed/citations/<snapshot>.edges.jsonl.gz`` and
``.aliases.jsonl.gz`` (loaded into Postgres by ``scripts/load_corpus.py``) and
``docs/data/citation_graph.md``.

Usage: uv run scripts/extract_citations.py [--snapshot ID] [--workers N]
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from juris.config import REPO_ROOT, get_settings
from juris.ingest.citation_graph import AliasTable, ScIndex, document_edges, parallel_groups
from juris.ingest.citations import ReporterCitation
from juris.ingest.statutes import load_acts

REPORT = REPO_ROOT / "docs" / "data" / "citation_graph.md"
STYLES = ["SCR", "INSC", "SCC", "AIR", "SCC OnLine", "SCALE", "JT", "HCNC"]

# worker state (set once per process)
_STATE: dict[str, Any] = {}


def _read_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _init(data_dir: str, snapshot: str, aliases: dict[str, list[str]] | None) -> None:
    rows: list[dict[str, Any]] = []
    for f in sorted(glob.glob(f"{data_dir}/raw/aws_sc/metadata/*.parquet")):
        rows += pq.read_table(f, columns=["path", "case_id"]).to_pylist()
    _STATE["index"] = ScIndex.from_rows(rows)
    _STATE["data_dir"], _STATE["snapshot"] = Path(data_dir), snapshot
    table = AliasTable()
    for alias, docs in (aliases or {}).items():
        table.targets[alias] = set(docs)
    _STATE["aliases"] = table
    snap = json.loads((Path(data_dir) / "snapshots" / f"{snapshot}.json").read_text("utf-8"))
    _STATE["corpus"] = {d["doc_id"] for d in snap["documents"]}
    sections = set()
    for path in (Path(data_dir) / "processed" / "statutes").glob("*.jsonl"):
        with path.open(encoding="utf-8") as fh:
            sections |= {f"{path.stem}:{json.loads(line)['section']}" for line in fh}
    _STATE["sections"] = sections
    _STATE["acts"] = load_acts()
    decided: dict[str, dt.date] = {}
    meta = Path(data_dir) / "interim" / "metadata" / snapshot / "metadata.jsonl"
    with meta.open(encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            if record.get("decision_date"):
                decided[record["doc_id"]] = dt.date.fromisoformat(record["decision_date"])
    _STATE["decided"] = decided


def _text(doc_id: str) -> str:
    base = _STATE["data_dir"] / "interim" / "parsed" / _STATE["snapshot"]
    text: str = _read_gz(base / f"{doc_id}.json.gz")["clean_text"]
    return text


def _groups(doc_id: str) -> list[list[dict[str, Any]]]:
    mentions, groups = parallel_groups(_text(doc_id))
    return [[mentions[i].citation.__dict__ for i in group] for group in groups if len(group) > 1]


def _edges(doc_id: str) -> list[dict[str, Any]]:
    seg = _STATE["data_dir"] / "interim" / "segmented" / _STATE["snapshot"] / f"{doc_id}.json.gz"
    paragraphs = _read_gz(seg)["paragraphs"]
    edges = document_edges(
        doc_id,
        _text(doc_id),
        paragraphs,
        _STATE["index"],
        _STATE["aliases"],
        _STATE["corpus"],
        _STATE["sections"],
        _STATE["acts"],
        _STATE["decided"].get(doc_id),
    )
    return [e.to_json() for e in edges]


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def write_report(
    snapshot: str, edges: list[dict[str, Any]], table: AliasTable, secs: float
) -> None:
    cases = [e for e in edges if e["kind"] == "case"]
    statutes = [e for e in edges if e["kind"] == "statute"]
    rows = []
    for style in STYLES:
        group = [e for e in cases if e["reporter"] == style]
        if not group:
            continue
        resolved = [e for e in group if e["target_ref"]]
        in_corpus = [e for e in group if e["target_doc_id"]]
        how = Counter(e["resolution"] for e in resolved)
        rows.append(
            f"| {style} | {len(group):,} | {len(resolved):,} ({pct(len(resolved), len(group))}) | "
            f"{len(in_corpus):,} ({pct(len(in_corpus), len(group))}) | "
            + ", ".join(f"{k} {v:,}" for k, v in how.most_common())
            + " |"
        )
    resolved_all = [e for e in cases if e["target_ref"]]
    acts = Counter(e["canonical"].split(":")[0] for e in statutes if e["target_ref"])
    unresolved_acts = Counter(
        e["canonical"].rsplit(":", 1)[0] for e in statutes if not e["target_ref"]
    )
    treatments = Counter(e["treatment"] for e in cases if e["treatment"])
    cited = Counter(e["target_doc_id"] for e in cases if e["target_doc_id"])
    lines = [
        "# Citation graph (PLAN 3.9)",
        "",
        f"Snapshot `{snapshot}`, generated by `scripts/extract_citations.py` ({secs:.0f} s). "
        "Edges: the `citation_edges` table (and `<data_dir>/processed/citations/`). "
        "Precision of resolved edges on a hand-checked sample: see "
        "[citation_check.md](citation_check.md).",
        "",
        "## Case citations: resolution per reporter style (D-013)",
        "",
        "*Resolved* = identified as a specific Supreme Court judgment (any of the ~38k in the "
        "AWS index); *in corpus* = that judgment is in this snapshot.",
        "",
        "| Style | Citations | Resolved | In corpus | How |",
        "|---|---|---|---|---|",
        *rows,
        f"| **All** | {len(cases):,} | {len(resolved_all):,} "
        f"({pct(len(resolved_all), len(cases))}) "
        f"| {sum(1 for e in cases if e['target_doc_id']):,} | |",
        "",
        "`direct` = exact INSC or first-page SCR match; `alias` = through a parallel citation "
        "printed with an SCR/INSC twin. An SCR page inside another report's pages is left "
        "unresolved (in the hand check such pages were mostly misprints). Parallel citations "
        "resolving to different judgments are all left unresolved: "
        f"{sum(1 for e in cases if e['resolution'] == 'conflict'):,} citations.",
        "",
        "## Parallel-citation aliases",
        "",
        f"- Aliases learnt (SCC, AIR, SCC OnLine, SCALE, JT -> judgment): {len(table.rows()):,}",
        f"- Dropped as ambiguous (one alias, several judgments): {table.ambiguous:,}",
        "",
        "## Statute mentions",
        "",
        f"- Mentions ('Section X of the Y Act'): {len(statutes):,}",
        f"- Resolved to a corpus Act: {sum(acts.values()):,} "
        f"({', '.join(f'{k} {v:,}' for k, v in acts.most_common())})",
        f"- Resolved to a section in `statute_sections`: "
        f"{sum(1 for e in statutes if e['target_section_id']):,}",
        "- Most mentioned other Acts (not in the corpus): "
        + ", ".join(f"{k} {v:,}" for k, v in unresolved_acts.most_common(8)),
        "",
        "## Treatment cues (weak signal)",
        "",
        "Cue words in the citing sentence; stored with the cue text, never as verified "
        "treatment (D-013).",
        "",
        ", ".join(f"{k} {v:,}" for k, v in treatments.most_common()) or "None.",
        "",
        "## Most cited corpus judgments",
        "",
        *[f"- `{doc}`: {n}" for doc, n in cited.most_common(10)],
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract the citation graph (PLAN 3.9)")
    parser.add_argument("--snapshot", default="mvp_contract-1f53c208a8")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args(argv)

    data_dir = get_settings().data_dir
    started = time.monotonic()
    seg_dir = data_dir / "interim" / "segmented" / args.snapshot
    docs = sorted(p.name.removesuffix(".json.gz") for p in seg_dir.glob("*.json.gz"))

    # pass 1: aliases
    _init(str(data_dir), args.snapshot, None)
    table = AliasTable()
    with cf.ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(str(data_dir), args.snapshot, None)
    ) as pool:
        for groups in pool.map(_groups, docs, chunksize=16):
            for group in groups:
                table.learn([ReporterCitation(**c) for c in group], _STATE["index"])
    aliases = {k: sorted(v) for k, v in table.targets.items()}

    # pass 2: edges
    edges: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(
        args.workers, initializer=_init, initargs=(str(data_dir), args.snapshot, aliases)
    ) as pool:
        for part in pool.map(_edges, docs, chunksize=16):
            edges += part

    out = data_dir / "processed" / "citations"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("edges", edges), ("aliases", table.rows())):
        with gzip.open(out / f"{args.snapshot}.{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    write_report(args.snapshot, edges, table, time.monotonic() - started)
    cases = sum(1 for e in edges if e["kind"] == "case")
    print(f"{len(edges):,} edges ({cases:,} case, {len(edges) - cases:,} statute) -> {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
