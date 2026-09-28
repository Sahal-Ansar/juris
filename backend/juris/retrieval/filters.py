"""Search filters shared by the retrievers (PLAN 4.1-4.3): court level, dates, kind, Act.

Court level and date range describe judgments, so they constrain judgments only: a statute
chunk passes them (use ``doc_kinds`` to keep or drop statutes). A judgment with no decision
date fails a date filter.

``acts`` takes Act IDs or any alias ('ICA', 'Contract Act, 1872'). A chunk matches when it is
from that Act's text, or from a judgment that cites the Act (resolved citation edges, 3.9).
"""

import datetime as dt
from functools import lru_cache
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator

from juris.ingest.statutes import load_acts, resolve_act
from juris.models import CourtLevel, DocumentKind


class SearchFilters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    court_levels: tuple[CourtLevel, ...] = ()
    date_from: dt.date | None = None  # inclusive
    date_to: dt.date | None = None  # inclusive
    doc_kinds: tuple[DocumentKind, ...] = ()
    acts: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _date_order(self) -> Self:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from is after date_to")
        return self


@lru_cache
def _acts() -> dict[str, Any]:
    return load_acts()


def act_ids(names: tuple[str, ...]) -> list[str]:
    """Canonical Act IDs; an unknown name raises rather than silently matching nothing."""
    out = []
    for name in names:
        act_id = resolve_act(name, _acts())
        if act_id is None:
            raise ValueError(f"unknown Act: {name!r}")
        out.append(act_id)
    return out


def filter_sql(filters: SearchFilters | None, chunk: str = "c", doc: str = "d") -> tuple[str, dict]:
    """A WHERE fragment over ``chunks`` (alias ``chunk``) joined to ``documents`` (``doc``)."""
    if filters is None:
        return "TRUE", {}
    parts: list[str] = []
    params: dict[str, Any] = {}
    judgment_only = f"{doc}.kind <> 'judgment' OR "
    if filters.doc_kinds:
        parts.append(f"{doc}.kind = ANY(:f_kinds)")
        params["f_kinds"] = [k.value for k in filters.doc_kinds]
    if filters.court_levels:
        parts.append(f"({judgment_only}{doc}.court_level = ANY(:f_courts))")
        params["f_courts"] = [c.value for c in filters.court_levels]
    if filters.date_from:
        parts.append(f"({judgment_only}{doc}.decision_date >= :f_from)")
        params["f_from"] = filters.date_from
    if filters.date_to:
        parts.append(f"({judgment_only}{doc}.decision_date <= :f_to)")
        params["f_to"] = filters.date_to
    if filters.doc_ids:
        parts.append(f"{chunk}.doc_id = ANY(:f_docs)")
        params["f_docs"] = list(filters.doc_ids)
    if filters.acts:
        parts.append(
            f"({chunk}.doc_id IN (SELECT doc_id FROM statute_sections WHERE act_id = ANY(:f_acts))"
            f" OR {chunk}.doc_id IN (SELECT e.source_doc_id FROM citation_edges e"
            " JOIN statute_sections s ON s.section_id = e.target_section_id"
            " WHERE s.act_id = ANY(:f_acts)))"
        )
        params["f_acts"] = act_ids(filters.acts)
    return (" AND ".join(parts) or "TRUE"), params
