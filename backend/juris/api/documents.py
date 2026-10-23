"""Corpus documents for ``GET /documents/{id}``: metadata and full text by paragraph.

A judgment's paragraphs come from ``paragraphs`` (their ``seq`` is what a chunk's
``para_start``/``para_end`` count); a statute has no paragraph rows, so its sections stand in
for them, in the Act's order. Tests pass a fake ``DocumentStore``.
"""

from typing import Protocol

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from juris.api.schemas import DocumentDetail, DocumentParagraph
from juris.config import get_settings
from juris.models import Document


class DocumentStore(Protocol):
    def document(self, doc_id: str) -> DocumentDetail | None: ...


class DbDocuments:
    """Reads the corpus database; the engine is made on first use, so the API starts without
    Postgres and only the document route needs it."""

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(get_settings().database_url())
        return self._engine

    def document(self, doc_id: str) -> DocumentDetail | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT doc_id, kind, title, court, court_level, bench_strength, judges, "
                    "decision_date, cnr, citations, source_url, licence, snapshot_id, headnote "
                    "FROM documents WHERE doc_id = :d"
                ),
                {"d": doc_id},
            ).one_or_none()
            if row is None:
                return None
            if row.kind == "statute":
                rows = conn.execute(
                    text(
                        "SELECT section, title, text, page_start, page_end FROM statute_sections "
                        "WHERE doc_id = :d ORDER BY char_start"
                    ),
                    {"d": doc_id},
                ).all()
                paragraphs = [
                    DocumentParagraph(
                        seq=n,
                        no=r.section,
                        part="section",
                        title=r.title or None,
                        text=r.text,
                        page_start=r.page_start,
                        page_end=r.page_end,
                    )
                    for n, r in enumerate(rows)
                    if r.text.strip()
                ]
            else:
                rows = conn.execute(
                    text(
                        "SELECT seq, no, section, text, page_start, page_end FROM paragraphs "
                        "WHERE doc_id = :d ORDER BY seq"
                    ),
                    {"d": doc_id},
                ).all()
                paragraphs = [
                    DocumentParagraph(
                        seq=r.seq,
                        no=r.no,
                        part=r.section,
                        text=r.text,
                        page_start=r.page_start,
                        page_end=r.page_end,
                    )
                    for r in rows
                    if r.text.strip()
                ]
        document = Document(
            id=row.doc_id,
            kind=row.kind,
            title=row.title,
            court=row.court,
            court_level=row.court_level,
            bench_strength=row.bench_strength,
            judges=row.judges or [],
            decision_date=row.decision_date,
            cnr=row.cnr,
            citations=row.citations or [],
            source_url=row.source_url,
            licence=row.licence,
            corpus_snapshot=row.snapshot_id,
        )
        return DocumentDetail(document=document, headnote=row.headnote, paragraphs=paragraphs)
