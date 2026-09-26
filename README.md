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

## Configuration

- `.env` holds secrets and machine settings (database, API keys). See `.env.example` and `backend/juris/config.py`.
- `configs/pipeline/*.yaml` are named system configurations (`juris_full`, `b0`, `b1`, `b2`): stage switches, deliberation parameters, and optional model and budget overrides.

## Runs

Every run writes a manifest (git commit, resolved config, models per role, prompt hashes, corpus snapshot, seeds, token and dollar totals) to `data/runs/<run_id>/manifest.json`, and optionally to the `runs` table in Postgres. To try it without an API key:

```bash
uv run scripts/dummy_run.py --db
```

## UI contract (schemas)

The Pydantic models are the source of truth. `schemas/` holds the exported JSON Schemas and `schemas/ts/juris.d.ts` (TypeScript types for the UI). After changing a model:

```bash
npm ci                               # once: json-schema-to-typescript
uv run scripts/export_schemas.py     # rewrite schemas/ and schemas/ts/
```

CI fails if the committed files are stale (`--check`).

## Checks

```bash
uv run ruff check
uv run ruff format --check
uv run mypy
uv run pytest            # add -m "not db" if Postgres isn't running
```

## Layout

- `backend/juris/`: the Python package (LLM gateway, ingestion, retrieval, agents, pipeline, API, eval)
- `backend/tests/`: tests
- `configs/pipeline/`: pipeline profiles
- `scripts/`: CLI entry points (`db_check.py`)
- `eval/`, `schemas/`, `fixtures/`: eval sets, exported JSON Schema, UI mock runs
- `apps/web/`: frontend
- `docs/`: technical docs and reports

Planning docs live in `Plan/` (kept local, not in git). Start with `Plan/STRUCTURE.md`.
