"""Juris-Eval: the end-to-end contract-law benchmark (PLAN 5.1, D-034).

Each item is a fact pattern and question with gold issues, the authorities a good answer
should rely on (supporting) and must deal with (contrary), the statute sections, and key
points. Every gold authority is a judgment in the corpus, cited with a paragraph pinpoint
that was read in the corpus text, and a one-sentence proposition it stands for.

The source of truth is ``eval/juris_eval/items.yaml`` (easy to review and edit). The
``dev.jsonl`` / ``test.jsonl`` files are generated from it by ``scripts/juris_eval.py``,
which also checks every item against the corpus database. Items stay ``reviewed_by: null``
until a person reviews them. ``test`` items are for reporting only: never tune on them.
"""

import json
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import Field, StringConstraints, model_validator
from sqlalchemy import text
from sqlalchemy.engine import Connection

from juris.config import REPO_ROOT
from juris.ingest.citations import normalise_citation
from juris.models import JurisModel

EVAL_DIR = REPO_ROOT / "eval" / "juris_eval"
SOURCE = EVAL_DIR / "items.yaml"
SPLITS = ("dev", "test")

NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SectionId = Annotated[str, StringConstraints(pattern=r"^[a-z_]+:\d+[A-Z]{0,2}$")]


class Authority(JurisModel):
    """A judgment in the corpus and what it is authority for in this item."""

    doc_id: NonEmpty
    citation: NonEmpty = Field(
        description="A reporter citation of the judgment, e.g. '[2003] 3 SCR 691'"
    )
    title: NonEmpty
    pinpoints: list[NonEmpty] = Field(
        min_length=1, description="Paragraph numbers in the corpus text ('12', 'h-2', 'p-8')"
    )
    proposition: NonEmpty = Field(description="The rule the authority stands for, in one sentence")


class JurisEvalItem(JurisModel):
    id: Annotated[str, StringConstraints(pattern=r"^JE-\d{3}$")]
    split: Literal["dev", "test"]
    topic: NonEmpty
    question: NonEmpty
    facts: NonEmpty
    gold_issues: list[NonEmpty] = Field(min_length=1)
    gold_supporting_authorities: list[Authority] = Field(min_length=1)
    gold_contrary_authorities: list[Authority] = Field(default_factory=list)
    gold_sections: list[SectionId] = Field(min_length=1)
    key_points: list[NonEmpty] = Field(min_length=2)
    difficulty: Literal["easy", "medium", "hard"]
    notes: str | None = None
    reviewed_by: str | None = None

    @model_validator(mode="after")
    def _authorities_differ(self) -> Self:
        supporting = {a.doc_id for a in self.gold_supporting_authorities}
        both = supporting & {a.doc_id for a in self.gold_contrary_authorities}
        if both:
            raise ValueError(f"{self.id}: {sorted(both)} both supporting and contrary")
        return self


def load_source(path: Path = SOURCE) -> list[JurisEvalItem]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    items = [JurisEvalItem.model_validate(raw) for raw in data["items"]]
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item IDs")
    return items


def split_lines(items: list[JurisEvalItem], split: str) -> str:
    """One JSON object per line, in ID order, without the ``split`` field (the file says it)."""
    rows = sorted((i for i in items if i.split == split), key=lambda i: i.id)
    return "".join(
        json.dumps(i.model_dump(mode="json", exclude={"split"}), ensure_ascii=False) + "\n"
        for i in rows
    )


def load_split(split: str, directory: Path = EVAL_DIR) -> list[JurisEvalItem]:
    """Items of a generated split file (``dev.jsonl`` or ``test.jsonl``)."""
    lines = (directory / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    return [JurisEvalItem.model_validate({**json.loads(line), "split": split}) for line in lines]


def schema() -> dict[str, object]:
    """JSON Schema of one line of ``dev.jsonl`` / ``test.jsonl``."""
    out = JurisEvalItem.model_json_schema()
    out["properties"].pop("split", None)  # type: ignore[union-attr]
    out["required"] = [r for r in out["required"] if r != "split"]  # type: ignore[union-attr]
    out["$id"] = "juris_eval_item"
    out["title"] = "Juris-Eval item"
    return out


def check_against_corpus(conn: Connection, items: list[JurisEvalItem]) -> list[str]:
    """Problems found: authorities not in the corpus, citations naming another document,
    pinpoints that are not paragraphs of the judgment, unknown statute sections."""
    problems: list[str] = []
    for item in items:
        for kind, authorities in (
            ("supporting", item.gold_supporting_authorities),
            ("contrary", item.gold_contrary_authorities),
        ):
            for a in authorities:
                where = f"{item.id} {kind} {a.doc_id}"
                row = conn.execute(
                    text(
                        "SELECT kind, citations, neutral_citation FROM documents WHERE doc_id = :d"
                    ),
                    {"d": a.doc_id},
                ).one_or_none()
                if row is None:
                    problems.append(f"{where}: not in the corpus")
                    continue
                if row[0] != "judgment":
                    problems.append(f"{where}: not a judgment")
                canonical = normalise_citation(a.citation)
                if canonical is None or canonical not in {*(row[1] or []), row[2]}:
                    problems.append(f"{where}: citation {a.citation!r} is not the document's")
                known: set[str] = set(
                    conn.execute(
                        text("SELECT no FROM paragraphs WHERE doc_id = :d AND no = ANY(:n)"),
                        {"d": a.doc_id, "n": a.pinpoints},
                    ).scalars()
                )
                for p in a.pinpoints:
                    if p not in known:
                        problems.append(f"{where}: no paragraph {p!r}")
        found: set[str] = set(
            conn.execute(
                text("SELECT section_id FROM statute_sections WHERE section_id = ANY(:s)"),
                {"s": item.gold_sections},
            ).scalars()
        )
        problems += [
            f"{item.id}: unknown section {s}" for s in item.gold_sections if s not in found
        ]
    return problems
