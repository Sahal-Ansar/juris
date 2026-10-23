# REST API (PLAN 9.1)

Code: `backend/juris/api/` (`app.py` routes, `runs.py` cases and the run queue, `documents.py`
corpus documents, `schemas.py` the API's own bodies). Tests: `backend/tests/test_api.py`.
Decision: D-063. Contract: `schemas/openapi.json`, `schemas/api_*.schema.json`,
`schemas/ts/juris.d.ts`.

```bash
uv run --group embed uvicorn --factory juris.api.app:create_app --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

## Routes

| Route | Returns | Notes |
|---|---|---|
| `POST /cases` | 202 `CaseCreated` (`case_id`, `run_id`) | body `CaseRequest`: `question` (≥ 10 characters), `profile` (default `juris_full`); 422 for an unknown profile |
| `GET /cases/{case_id}` | `CaseInfo` | the question and its runs, each `queued`, `running`, `completed` or `failed` |
| `GET /runs/{run_id}` | `CaseView` | the record so far, folded from the run's events; includes the evidence graph |
| `GET /runs/{run_id}/analysis` | `CaseAnalysis` | 404 until the Judge has written it, and for a run that failed without one |
| `GET /runs/{run_id}/graph` | `EvidenceGraph` | the graph alone (PLAN 8.1) |
| `GET /evidence/{run_id}/{evidence_id}` | `EvidenceDetail` | the evidence, its chunk and document, and every citation of it with its latest verification |
| `GET /documents/{doc_id}` | `DocumentDetail` | a corpus document from Postgres with its text by paragraph; a statute's sections stand in for paragraphs |
| `GET /configs` | `ConfigInfo[]` | the profiles `POST /cases` accepts: Juris, its ablations, B0-B2 |

An unknown case, run, evidence or document is a 404 with `{"detail": "... not found"}`; a bad
body is FastAPI's 422. CORS allows `JURIS_API_CORS_ORIGINS` (default the local frontend on port
3000), `GET` and `POST`.

A paragraph's `seq` is what a chunk's `para_start` and `para_end` count, so the UI can go from a
citation to the paragraph in the full document (IDEA_final §13, traceability).

## How runs execute

`POST /cases` opens the case and its run at once: `case_created` is stored before the response,
so `GET /runs/{run_id}` answers immediately. The run is queued; `JURIS_API_CONCURRENCY` workers
(default 1: the local model has one GPU) take runs in order and execute each in a thread with
its own event loop. The system emits its events, then `run_completed` or `run_failed`; a run
that raises is recorded as failed and the worker goes on to the next.

Why a thread: retrieval is synchronous (loading the embedding and reranking models on first
use, embedding and reranking each search), and on the server's loop it held every request.
Model clients and the SQLite response cache belong to one loop and thread, so each run makes
its own gateway in its thread (its own clients, cache connection and cost ledger, over the
shared cache file). Only the run's thread writes its events; requests read them.

The event store is in memory: runs last as long as the server (PLAN 9.3 adds the database).
Each finished run is also written to `<runs_dir>/api/<run_id>/` (`manifest.json`,
`events.jsonl`, `caseview.json`, `calls.json`), as `run_case.py` does.

## Live check (2026-10-24, local 7B)

`uvicorn` on the real settings, JE-007's question as a free question, `juris_full`:

| | first version (runs on the server's loop) | in a thread |
|---|---|---|
| run | 34 calls, none cached, about 5 min | the same 34 calls, all from the cache, about 1 min |
| polls while running | 65, every 5 s | 52, every 1 s |
| slowest response | 40.5 s (the retrieval models loading), then 22.6 s (a search) | 2.5 s |
| mean response | 1.18 s | 0.11 s |

Each run was the first of a fresh server, so both loaded the retrieval models and ran every
search; only the model calls differ (live against cached), and they never blocked. Both runs
completed with an analysis (62 evidence, 4 claims). `GET /configs` lists the ten runnable
profiles; `GET /documents/ACT-contract_act` returns the Contract Act's 268 sections; the
analysis, graph and evidence routes answered for the finished run. The remaining pauses of a
second or two are most likely the run's thread holding the interpreter (tokenising, Python
work between model calls).
