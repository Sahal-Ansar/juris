"""Retrieval evaluation (PLAN 5.3): metrics by hand, units, fusion, and pool indexing."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Engine
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from juris.eval.datasets import Document
from juris.eval.pool import index_pool
from juris.eval.retrieval_eval import (
    EvalQuery,
    Result,
    doc_unit,
    evaluate,
    metrics,
    single,
    table,
    to_units,
    windows,
)
from juris.eval.stats import bootstrap_ci
from juris.retrieval.lexical import LexicalRetriever
from tests.conftest import migrate


def test_metrics_by_hand() -> None:
    ranked = ["x", "a", "y", "b"] + [f"z{i}" for i in range(60)] + ["c"]
    m = metrics(ranked, {"a", "b", "c"})
    assert m["R@10"] == pytest.approx(2 / 3) and m["R@50"] == pytest.approx(2 / 3)
    assert m["P@10"] == pytest.approx(0.2)
    assert m["MRR"] == pytest.approx(1 / 2)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m["nDCG@10"] == pytest.approx(dcg / ideal)
    assert metrics(["q"], {"a"})["MRR"] == 0.0
    perfect = metrics(["a", "b"], {"a", "b"})
    assert perfect["nDCG@10"] == pytest.approx(1.0) and perfect["R@10"] == 1.0
    with pytest.raises(ValueError):
        metrics(["a"], set())


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    rank: int


def hits(*chunk_ids: str) -> list[Hit]:
    return [Hit(c, c.split("#")[0], i) for i, c in enumerate(chunk_ids, 1)]


def test_units_rank_by_their_best_chunk() -> None:
    assert to_units(hits("A#1", "B#1", "A#2", "C#1"), doc_unit) == ["A", "B", "C"]


def test_windows() -> None:
    words = " ".join(str(i) for i in range(7))
    assert windows(words, 3) == ["0 1 2", "3 4 5", "6"]
    assert windows(words, 3, limit=2) == ["0 1 2", "3 4 5"]
    assert windows("", 3) == [""]


@dataclass
class Fixed:
    by_query: dict[str, list[Hit]]

    def __call__(self, query: str, k: int) -> Sequence[Hit]:
        return self.by_query[query][:k]


def test_single_system_fuses_several_texts() -> None:
    search = Fixed({"q1": hits("A#1", "B#1"), "q2": hits("B#2", "C#1")})
    system = single("s", search, doc_unit)
    assert system.run(["q1"]) == ["A", "B"]
    assert system.run(["q1", "q2"])[0] == "B"  # found by both texts
    result = evaluate(system, "d", [EvalQuery("x", ("q1", "q2"), frozenset({"C"}))])
    assert result.per_query["x"]["R@10"] == 1.0
    other = Result("t", "d", {"x": {**result.per_query["x"], "R@10": 0.5}})
    rendered = table([result, other])
    assert "| s | **1.000**" in rendered and "| t | 0.500" in rendered


def test_bootstrap_interval_brackets_the_mean() -> None:
    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 5
    lo, hi = bootstrap_ci(values)
    assert lo < sum(values) / len(values) < hi and lo >= 0 and hi <= 1
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)


@pytest.mark.db
def test_pool_indexing_is_resumable_and_searchable(engine: Engine) -> None:
    with engine.begin() as conn:
        migrate(conn)
    docs = [
        Document("C1", "A v State\nSupreme Court of India\n\nThe appellant was dismissed "
                 "from service without any inquiry into the charges.", "A v State"),
        Document("C2", "B v Union\nSupreme Court of India\n\nThe contract was frustrated "
                 "when the land was requisitioned for military purposes.", "B v Union"),
    ]  # fmt: skip
    tok = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))  # one token per word
    tok.pre_tokenizer = Whitespace()
    first = index_pool(engine, docs, "judgment", "test", tokenizer=tok)
    assert (first["documents"], first["new_documents"]) == (2, 2) and first["chunks"] >= 2
    again = index_pool(engine, docs, "judgment", "test", tokenizer=tok)
    assert again["new_documents"] == 0 and again["chunks"] == first["chunks"]
    with engine.connect() as conn:
        headers = conn.execute(text("SELECT context_header FROM chunks ORDER BY chunk_id"))
        assert all(h[0].startswith(("A v State", "B v Union")) for h in headers)
    found = LexicalRetriever(engine).search("contract frustrated requisitioned")
    assert found[0].doc_id == "C2"
