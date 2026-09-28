"""The query set and helpers for the retrieval latency benchmarks (``scripts/bench_*.py``).

Legal terms, section references, case names, questions, single common words and filtered
searches, over the MVP contract slice. Latency only: quality is measured in PLAN 5.3.
"""

import datetime as dt

from sqlalchemy import text
from sqlalchemy.engine import Engine

from juris.models import CourtLevel, DocumentKind
from juris.retrieval.filters import SearchFilters

Query = tuple[str, str, SearchFilters | None]  # (kind, query, filters)

SC = SearchFilters(court_levels=(CourtLevel.SUPREME_COURT,))
QUERIES: list[Query] = [
    ("term", "undue influence", None),
    ("term", "frustration of contract", None),
    ("term", "liquidated damages", None),
    ("term", "readiness and willingness", None),
    ("term", "quantum meruit", None),
    ("term", "privity of contract", None),
    ("term", "restraint of trade", None),
    ("section", "Section 74", None),
    ("section", "s. 16(2) undue influence", None),
    ("section", "Section 20 of the Specific Relief Act", None),
    ("section", "u/s 73 compensation for breach", None),
    ("section", "Section 56 impossibility of performance", None),
    ("section", "Section 19A", None),
    ("case", "Satyabrata Ghose v. Mugneeram Bangur", None),
    ("case", "Fateh Chand v. Balkishan Dass", None),
    ("case", "ONGC v. Saw Pipes", None),
    ("case", "Kailash Nath Associates", None),
    ("case", "Mohori Bibee", None),
    (
        "question",
        "Is a contract entered into under coercion by a threat to a third party "
        "voidable at the option of the coerced party?",
        None,
    ),
    ("question", "Can earnest money be forfeited without proof of actual loss?", None),
    (
        "question",
        "When is time of the essence in a contract for sale of immovable property?",
        None,
    ),
    (
        "question",
        "Does a minor's agreement bind the minor or can it be ratified on majority?",
        None,
    ),
    ("common", "contract", None),
    ("common", "court", None),
    ("common", "agreement party", None),
    ("filtered", "specific performance", SC),
    (
        "filtered",
        "liquidated damages",
        SearchFilters(date_from=dt.date(2000, 1, 1), date_to=dt.date(2020, 12, 31)),
    ),
    ("filtered", "specific performance discretion", SearchFilters(acts=("SRA",))),
    ("filtered", "compensation for loss", SearchFilters(doc_kinds=(DocumentKind.STATUTE,))),
    ("filtered", "contract", SearchFilters(court_levels=(CourtLevel.HIGH_COURT,), acts=("ICA",))),
    (
        "filtered",
        "penalty clause reasonable compensation",
        SearchFilters(date_from=dt.date(2010, 1, 1), date_to=dt.date(2012, 12, 31)),
    ),
    (
        "filtered",
        "agreement void for uncertainty",
        SearchFilters(date_to=dt.date(1959, 12, 31)),
    ),
]


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def label(query: str, filters: SearchFilters | None) -> str:
    if not filters:
        return query
    return f"{query} [{filters.model_dump(exclude_defaults=True, mode='json')}]"


def header(engine: Engine, chunk_id: str) -> str:
    """The chunk's context header (what it is), to show next to a timing."""
    with engine.connect() as conn:
        value: str = conn.execute(
            text("SELECT coalesce(context_header, chunk_id) FROM chunks WHERE chunk_id = :c"),
            {"c": chunk_id},
        ).scalar_one()
    return value


def chunk_count(engine: Engine) -> int:
    with engine.connect() as conn:
        n: int = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    return n
