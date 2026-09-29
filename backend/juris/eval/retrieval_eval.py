"""Retrieval evaluation (PLAN 5.3, D-037): metrics, systems and runs.

Retrieval returns chunks; relevance is judged on *units*, the authorities a benchmark names:
a judgment's ``doc_id`` or, in our corpus, a statute section's ``section_id``. A system's
ranked chunks become ranked units by first occurrence (a unit ranks where its best chunk does).

Metrics (IDEA_final §11.2), binary relevance, averaged over queries:
Recall@{10,25,50}, Precision@10, MRR (of the first relevant unit, over the full list) and
nDCG@10 (ideal DCG over min(|relevant|, 10) relevant units).

A query can be several texts (hand-written search queries, or windows of a long fact pattern);
hybrid systems fuse them with RRF (4.3), single retrievers fuse their per-text lists the same
way.
"""

import math
import random
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Engine

from juris.retrieval.hybrid import HybridRetriever, rrf

KS = (10, 25, 50)


class ChunkHit(Protocol):
    @property
    def chunk_id(self) -> str: ...
    @property
    def doc_id(self) -> str: ...
    @property
    def rank(self) -> int: ...


# ---- metrics -------------------------------------------------------------------------------


def metrics(ranked: Sequence[str], relevant: frozenset[str] | set[str]) -> dict[str, float]:
    """Retrieval metrics for one query: ``ranked`` unit IDs (best first), binary relevance."""
    if not relevant:
        raise ValueError("a query needs at least one relevant unit")
    out = {f"R@{k}": len(set(ranked[:k]) & relevant) / len(relevant) for k in KS}
    out["P@10"] = len(set(ranked[:10]) & relevant) / 10
    first = next((i for i, u in enumerate(ranked, 1) if u in relevant), None)
    out["MRR"] = 1 / first if first else 0.0
    dcg = sum(1 / math.log2(i + 1) for i, u in enumerate(ranked[:10], 1) if u in relevant)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), 10) + 1))
    out["nDCG@10"] = dcg / ideal
    return out


METRICS = [f"R@{k}" for k in KS] + ["P@10", "MRR", "nDCG@10"]


# ---- systems -------------------------------------------------------------------------------


UnitOf = Callable[[ChunkHit], str]


def doc_unit(hit: ChunkHit) -> str:
    return hit.doc_id


def corpus_unit(engine: Engine) -> UnitOf:
    """Our corpus: statute chunks count as their section (``contract_act:74``), others as
    their document."""
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT chunk_id, statute_section_id FROM chunks "
                "WHERE statute_section_id IS NOT NULL"
            )
        ).all()
    sections: dict[str, str] = {r[0]: r[1] for r in rows}
    return lambda hit: sections.get(hit.chunk_id, hit.doc_id)


def to_units(hits: Sequence[ChunkHit], unit_of: UnitOf) -> list[str]:
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(unit_of(hit), None)
    return list(seen)


class ChunkSearch(Protocol):
    """One text in, ranked chunks out (a lexical or dense retriever, adapted)."""

    def __call__(self, query: str, k: int) -> Sequence[ChunkHit]: ...


@dataclass
class System:
    name: str
    run: Callable[[Sequence[str]], list[str]]  # query texts -> ranked unit IDs


@dataclass(frozen=True)
class _Ranked:
    chunk_id: str
    doc_id: str
    rank: int


def fuse_units(lists: Sequence[list[str]]) -> list[str]:
    """RRF over per-text unit rankings, so a unit found by several texts rises even when
    each text found a different chunk of it (evaluation only; agents fuse chunks, 4.3)."""
    if len(lists) == 1:
        return list(lists[0])
    named = {
        f"q{i}": [_Ranked(u, u, r) for r, u in enumerate(units, 1)] for i, units in enumerate(lists)
    }
    return [h.chunk_id for h in rrf(named)]


def single(name: str, search: ChunkSearch, unit_of: UnitOf, depth: int = 200) -> System:
    """One retriever; several texts are searched separately and fused on units."""

    def run(texts: Sequence[str]) -> list[str]:
        return fuse_units([to_units(search(t, depth), unit_of) for t in texts])

    return System(name, run)


def hybrid(
    name: str, retriever: HybridRetriever, unit_of: UnitOf, k: int = 200, rerank: bool = False
) -> System:
    """Lexical + dense fused on chunks per text (4.3, reranked against that text), then the
    texts fused on units."""

    def run(texts: Sequence[str]) -> list[str]:
        per_text = [to_units(retriever.search(t, k=k, rerank=rerank), unit_of) for t in texts]
        return fuse_units(per_text)

    return System(name, run)


# ---- queries -------------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalQuery:
    id: str
    texts: tuple[str, ...]  # one or more texts searched together
    relevant: frozenset[str]


def windows(text_: str, words: int = 300, limit: int | None = None) -> list[str]:
    """Split a long text into consecutive windows of ``words`` words (at most ``limit``)."""
    tokens = text_.split()
    out = [" ".join(tokens[i : i + words]) for i in range(0, len(tokens), words)] or [""]
    return out[:limit] if limit else out


# ---- runs ----------------------------------------------------------------------------------


@dataclass
class Result:
    system: str
    dataset: str
    per_query: dict[str, dict[str, float]] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def mean(self) -> dict[str, float]:
        return {m: statistics.mean(q[m] for q in self.per_query.values()) for m in METRICS}

    @property
    def seconds_per_query(self) -> float:
        return self.seconds / max(1, len(self.per_query))


def evaluate(system: System, dataset: str, queries: Sequence[EvalQuery]) -> Result:
    result = Result(system.name, dataset)
    started = time.perf_counter()
    for q in queries:
        result.per_query[q.id] = metrics(system.run(q.texts), q.relevant)
    result.seconds = time.perf_counter() - started
    return result


def bootstrap_ci(
    values: Sequence[float], resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval of the mean (queries resampled with replacement)."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
    return means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]


def table(results: Sequence[Result], note: Mapping[str, str] | None = None) -> str:
    """Markdown table: one row per system, the metrics, a 95% bootstrap interval for
    nDCG@10 and seconds per query."""
    head = "| System | " + " | ".join(METRICS) + " | nDCG@10 95% CI | s/query |"
    rows = [head, "|" + "---|" * (len(METRICS) + 3)]
    best = {m: max(r.mean[m] for r in results) for m in METRICS}
    for r in results:
        cells = []
        for m in METRICS:
            v = f"{r.mean[m]:.3f}"
            cells.append(f"**{v}**" if math.isclose(r.mean[m], best[m]) and len(results) > 1 else v)
        lo, hi = bootstrap_ci([q["nDCG@10"] for q in r.per_query.values()])
        label = r.system + (f" ({note[r.system]})" if note and r.system in note else "")
        rows.append(
            f"| {label} | " + " | ".join(cells) + f" | {lo:.3f}-{hi:.3f} | "
            f"{r.seconds_per_query:.2f} |"
        )
    return "\n".join(rows)
