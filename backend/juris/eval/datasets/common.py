"""Common formats for external benchmarks (PLAN 5.2, D-036).

Retrieval benchmarks become a ``RetrievalDataset``: queries with the IDs of their relevant
documents, and the benchmark's own candidate pool. Our corpus and theirs share no IDs (D-011),
so retrieval is evaluated over the pool, indexed separately with the same retrieval code (5.3).

Entailment benchmarks (for the citation verifier's sanity check, 6.1) become an
``EntailmentDataset``: premise, hypothesis and a yes/no label.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from juris.config import get_settings


def benchmarks_dir(data_dir: Path | None = None) -> Path:
    """``<data_dir>/raw/benchmarks``: downloaded benchmarks, never committed (licences, D-010)."""
    return (data_dir or get_settings().data_dir) / "raw" / "benchmarks"


class BenchmarkMissing(FileNotFoundError):
    """The benchmark's files are not on disk; the message says how to get them."""


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    title: str | None = None


@dataclass(frozen=True)
class RetrievalQuery:
    id: str
    text: str
    relevant: frozenset[str]


@dataclass
class RetrievalDataset:
    name: str
    split: str
    queries: list[RetrievalQuery]
    documents: Mapping[str, Document]  # the candidate pool
    licence: str
    notes: str = ""

    def validate(self) -> list[str]:
        """Problems: no queries, empty texts, queries without relevant documents, relevant
        IDs missing from the pool."""
        problems = [] if self.queries else ["no queries"]
        problems += [
            f"document {d.id}: empty text" for d in self.documents.values() if not d.text.strip()
        ]
        for q in self.queries:
            if not q.text.strip():
                problems.append(f"query {q.id}: empty text")
            if not q.relevant:
                problems.append(f"query {q.id}: no relevant documents")
            missing = sorted(q.relevant - self.documents.keys())
            if missing:
                problems.append(f"query {q.id}: relevant not in pool: {missing[:5]}")
        return problems


@dataclass(frozen=True)
class EntailmentExample:
    id: str
    premise: str
    hypothesis: str
    label: bool  # True: the premise entails (supports) the hypothesis
    meta: Mapping[str, str] = field(default_factory=dict)


@dataclass
class EntailmentDataset:
    name: str
    split: str
    examples: list[EntailmentExample]
    licence: str
    notes: str = ""

    def validate(self) -> list[str]:
        problems = [] if self.examples else ["no examples"]
        for e in self.examples:
            if not e.premise.strip() or not e.hypothesis.strip():
                problems.append(f"example {e.id}: empty premise or hypothesis")
        ids = [e.id for e in self.examples]
        if len(ids) != len(set(ids)):
            problems.append("duplicate example IDs")
        return problems
