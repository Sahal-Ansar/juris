"""Corpus objects: documents and chunks. Created at ingestion, shared by every run."""

import datetime as dt
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from juris.models.common import ChunkId, CourtLevel, DocumentId, DocumentKind, JurisModel


class Document(JurisModel):
    """A judgment or statute in a corpus snapshot."""

    id: DocumentId
    kind: DocumentKind
    title: Annotated[str, StringConstraints(min_length=1)]
    court: str | None = None
    court_level: CourtLevel | None = None
    bench_strength: int | None = Field(default=None, ge=1)
    judges: list[str] = Field(default_factory=list)
    decision_date: dt.date | None = None
    cnr: str | None = Field(default=None, description="eCourts case number; join key (D-010)")
    citations: list[str] = Field(
        default_factory=list,
        description="The document's own citations, e.g. '2023 INSC 1043', '[2023] 16 S.C.R. 872'",
    )
    source_url: str | None = None
    licence: Annotated[str, StringConstraints(min_length=1)]
    corpus_snapshot: Annotated[str, StringConstraints(min_length=1)]


class StatuteMeta(JurisModel):
    """Section-level statute fields (IDEA_final §7.3, D-014)."""

    act: Annotated[str, StringConstraints(min_length=1)]
    section: Annotated[str, StringConstraints(min_length=1)]
    title: str | None = None
    in_force_from: dt.date | None = None
    in_force_to: dt.date | None = Field(default=None, description="None = still in force")
    amended_by: list[str] = Field(default_factory=list)
    amendments_curated: bool = Field(
        default=False, description="True only if dates came from configs/statutes/amendments.yaml"
    )

    @model_validator(mode="after")
    def _interval(self) -> Self:
        if self.in_force_from and self.in_force_to and self.in_force_to < self.in_force_from:
            raise ValueError("in_force_to is before in_force_from")
        return self


class Chunk(JurisModel):
    """A retrievable span of a document, with paragraph (or page) provenance.

    Judgments before ~1970 have no numbered paragraphs, so they carry pages instead (D-010).
    """

    id: ChunkId
    document_id: DocumentId
    text: Annotated[str, StringConstraints(min_length=1)]
    para_start: int | None = Field(default=None, ge=0)
    para_end: int | None = Field(default=None, ge=0)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    section: str | None = Field(default=None, description="Section heading, if any")
    statute: StatuteMeta | None = None

    @model_validator(mode="after")
    def _ranges(self) -> Self:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        for lo, hi, name in (
            (self.para_start, self.para_end, "para"),
            (self.page_start, self.page_end, "page"),
        ):
            if (lo is None) != (hi is None):
                raise ValueError(f"{name}_start and {name}_end must be set together")
            if lo is not None and hi is not None and hi < lo:
                raise ValueError(f"{name}_end is before {name}_start")
        if self.para_start is None and self.page_start is None and self.statute is None:
            raise ValueError("a chunk needs a paragraph range, a page range, or statute metadata")
        return self
