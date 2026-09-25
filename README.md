# Juris

Evidence-grounded multi-agent legal case deliberation for Indian law. Agents argue a case over retrieved judgments and statutes, and every claim in the final analysis must cite verified evidence.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync
uv run pre-commit install
```

## Database

Postgres 16 with pgvector runs in Docker. On Windows, use Docker Desktop with the WSL2 backend.

```bash
cp .env.example .env
docker compose up -d
uv run scripts/db_check.py
```

`db_check.py` prints `OK` with the Postgres and pgvector versions. The port and credentials come from `.env`.

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
- `scripts/`: CLI entry points (`db_check.py`)
- `eval/`, `schemas/`, `fixtures/`: eval sets, exported JSON Schema, UI mock runs
- `apps/web/`: frontend
- `docs/`: technical docs and reports

Planning docs live in `Plan/` (kept local, not in git). Start with `Plan/STRUCTURE.md`.
