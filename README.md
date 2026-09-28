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

Alembic owns the schema (`backend/juris/db/models.py`, migrations in `backend/juris/db/migrations/`). Create or update it, then load a processed corpus snapshot:

```bash
uv run alembic upgrade head
uv run scripts/load_corpus.py
```

The loader upserts by stable IDs, so running it again changes nothing. Vectors live in `chunk_embeddings`, one row per chunk and embedding model, and each model in `configs/embeddings.yaml` gets its own HNSW index.

Embedding needs PyTorch with CUDA, kept in an optional dependency group so CI and a plain `uv sync` don't download it. The bge-m3 weights go in `<data_dir>/models/BAAI__bge-m3` (config, `pytorch_model.bin`, tokenizer files at the revision pinned in `juris/ingest/tokens.py`). The run is resumable:

```bash
uv sync --group embed
uv run --group embed scripts/embed_corpus.py
uv run --group embed scripts/embed_corpus.py --check "ACT-contract_act#s73-1"
```

## Configuration

- `.env` holds secrets and machine settings (database, API keys). See `.env.example` and `backend/juris/config.py`.
- `configs/pipeline/*.yaml` are named system configurations (`juris_full`, `b0`, `b1`, `b2`): stage switches, deliberation parameters, and optional model and budget overrides.

## Corpus

The corpus slice is defined in `configs/corpus/mvp_contract.yaml` and downloaded by:

```bash
uv run scripts/acquire.py all      # resumable; about 23 GB streamed, about 2.3 GB kept
```

Data goes to `JURIS_DATA_DIR` (default `data/`; keep it outside cloud-synced folders). Sources and licences: `docs/data/sources.md`, `docs/DATA_NOTICE.md`.

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
- `scripts/`: CLI entry points (`db_check.py`, `load_corpus.py`, the ingestion steps)
- `eval/`, `schemas/`, `fixtures/`: eval sets, exported JSON Schema, UI mock runs
- `apps/web/`: frontend
- `docs/`: technical docs and reports

Planning docs live in `Plan/` (kept local, not in git). Start with `Plan/STRUCTURE.md`.
