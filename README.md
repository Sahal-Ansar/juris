# Juris

Evidence-grounded multi-agent legal case deliberation for Indian law. Agents argue a case over retrieved judgments and statutes, and every claim in the final analysis must cite verified evidence.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pre-commit install
```

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest
```

## Layout

- `backend/juris/`: the Python package (LLM gateway, ingestion, retrieval, agents, pipeline, API, eval)
- `backend/tests/`: tests
- `eval/`, `schemas/`, `fixtures/`: eval sets, exported JSON Schema, UI mock runs
- `apps/web/`: frontend
- `docs/`: technical docs and reports

Planning docs live in `Plan/` (kept local, not in git). Start with `Plan/STRUCTURE.md`.
