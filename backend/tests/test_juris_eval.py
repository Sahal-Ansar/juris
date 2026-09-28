"""Juris-Eval seed set (PLAN 5.1): format, fixed split, generated files, corpus checks.

The corpus check itself runs against the real database (``scripts/juris_eval.py check``);
here it runs on a throwaway database to show it catches each kind of error.
"""

import json
from typing import Any

import jsonschema
import pytest
from pydantic import ValidationError
from sqlalchemy import insert
from sqlalchemy.engine import Engine

from juris.db import models as m
from juris.eval.juris_eval import (
    EVAL_DIR,
    JurisEvalItem,
    check_against_corpus,
    load_source,
    load_split,
    schema,
    split_lines,
)
from tests.conftest import migrate

# The split is fixed (PLAN 5.1): moving an item between dev and test needs a decision.
DEV = {f"JE-{n:03d}" for n in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)}
TEST = {f"JE-{n:03d}" for n in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)}


def test_the_seed_set_has_20_items_in_a_fixed_split() -> None:
    items = load_source()
    assert len(items) == 20
    assert {i.id for i in items if i.split == "dev"} == DEV
    assert {i.id for i in items if i.split == "test"} == TEST


def test_generated_files_match_the_source() -> None:
    items = load_source()
    for split in ("dev", "test"):
        assert (EVAL_DIR / f"{split}.jsonl").read_text(encoding="utf-8") == split_lines(
            items, split
        ), f"{split}.jsonl is stale: run uv run scripts/juris_eval.py build"
        assert {i.id for i in load_split(split)} == (DEV if split == "dev" else TEST)
    committed = json.loads((EVAL_DIR / "schema.json").read_text(encoding="utf-8"))
    assert committed == json.loads(json.dumps(schema()))


def test_every_line_validates_against_the_schema() -> None:
    validator = jsonschema.Draft202012Validator(schema())
    for split in ("dev", "test"):
        for line in (EVAL_DIR / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            validator.validate(json.loads(line))


def test_items_await_review_and_every_topic_differs() -> None:
    items = load_source()
    assert all(i.reviewed_by is None for i in items)  # until a human reviews them
    assert len({i.topic for i in items}) == 20


def valid() -> dict[str, Any]:
    authority = {
        "doc_id": "SC-1", "citation": "[2003] 3 SCR 691", "title": "A v. B",
        "pinpoints": ["12"], "proposition": "A rule.",
    }  # fmt: skip
    return {
        "id": "JE-100", "split": "dev", "topic": "t", "question": "q?", "facts": "f",
        "gold_issues": ["i"], "gold_supporting_authorities": [authority],
        "gold_sections": ["contract_act:74"], "key_points": ["a", "b"],
        "difficulty": "easy",
    }  # fmt: skip


@pytest.mark.parametrize(
    "change",
    [
        {"id": "JE-1"},
        {"split": "train"},
        {"gold_supporting_authorities": []},
        {"gold_sections": ["Contract Act s. 74"]},
        {"key_points": ["only one"]},
        {"difficulty": "trivial"},
        {"unexpected": True},
    ],
)
def test_malformed_items_are_rejected(change: dict[str, Any]) -> None:
    JurisEvalItem.model_validate(valid())
    with pytest.raises(ValidationError):
        JurisEvalItem.model_validate({**valid(), **change})


def test_an_authority_cannot_be_both_supporting_and_contrary() -> None:
    raw = valid()
    raw["gold_contrary_authorities"] = raw["gold_supporting_authorities"]
    with pytest.raises(ValidationError, match="both supporting and contrary"):
        JurisEvalItem.model_validate(raw)


@pytest.mark.db
def test_the_corpus_check_catches_each_kind_of_error(engine: Engine) -> None:
    with engine.begin() as conn:
        migrate(conn)
        conn.execute(insert(m.Snapshot).values(snapshot_id="s", slice_name="t"))
        conn.execute(
            insert(m.Document),
            [
                {"doc_id": "SC-1", "snapshot_id": "s", "kind": "judgment", "title": "A v. B",
                 "source": "t", "licence": "t", "citations": ["[2003] 3 SCR 691"],
                 "neutral_citation": "2003 INSC 241"},
                {"doc_id": "ACT-contract_act", "snapshot_id": "s", "kind": "statute",
                 "title": "ICA", "source": "t", "licence": "t", "citations": [],
                 "neutral_citation": None},
            ],
        )  # fmt: skip
        conn.execute(
            insert(m.Paragraph).values(
                doc_id="SC-1", seq=0, no="12", part=1, section="body", numbering="explicit",
                char_start=0, char_end=5, page_start=1, page_end=1, text="A rule.",
            )
        )  # fmt: skip
        conn.execute(
            insert(m.StatuteSection).values(
                section_id="contract_act:74", doc_id="ACT-contract_act", act_id="contract_act",
                act="Indian Contract Act", act_year=1872, section="74", title="t", text="t",
                char_start=0, char_end=1, page_start=1, page_end=1,
            )
        )  # fmt: skip
        good = JurisEvalItem.model_validate(valid())
        assert check_against_corpus(conn, [good]) == []
        # the neutral citation, in another spelling, is the document's too
        alt = valid()
        alt["gold_supporting_authorities"][0]["citation"] = "2003 INSC 241"
        assert check_against_corpus(conn, [JurisEvalItem.model_validate(alt)]) == []

        bad = valid()
        bad["gold_supporting_authorities"] = [
            {**bad["gold_supporting_authorities"][0], "citation": "(2003) 5 SCC 705",
             "pinpoints": ["12", "99"]},
            {**bad["gold_supporting_authorities"][0], "doc_id": "SC-missing"},
            {**bad["gold_supporting_authorities"][0], "doc_id": "ACT-contract_act"},
        ]  # fmt: skip
        bad["gold_sections"] = ["contract_act:74", "contract_act:999"]
        problems = check_against_corpus(conn, [JurisEvalItem.model_validate(bad)])
        joined = "\n".join(problems)
        assert "is not the document's" in joined
        assert "no paragraph '99'" in joined
        assert "SC-missing: not in the corpus" in joined
        assert "not a judgment" in joined
        assert "unknown section contract_act:999" in joined
