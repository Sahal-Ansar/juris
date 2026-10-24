# REST API and event stream (PLAN 9.1-9.3)

Code: `backend/juris/api/` (`app.py` routes, `runs.py` cases and the run queue, `documents.py`
corpus documents, `schemas.py` the API's own bodies, `stream.py` the event stream, `mock.py`
mock mode) and `backend/juris/db/events.py` (the event log in Postgres). Tests:
`backend/tests/test_api.py`, `backend/tests/test_events.py`. Decisions: D-063 (API), D-064
(stream), D-065 (persistence, replay, mock mode). Contract: `schemas/openapi.json`,
`schemas/api_*.schema.json`, `schemas/ts/juris.d.ts`.

```bash
uv run --group embed uvicorn --factory juris.api.app:create_app --port 8000   # live
JURIS_API_MODE=mock uv run uvicorn --factory juris.api.app:create_app --port 8000
```

Live mode needs Postgres (`docker compose up -d`, migrated to head) and the model; mock mode
needs neither.

Interactive docs at `http://localhost:8000/docs`.

## Routes

| Route | Returns | Notes |
|---|---|---|
| `POST /cases` | 202 `CaseCreated` (`case_id`, `run_id`) | body `CaseRequest`: `question` (≥ 10 characters), `profile` (default `juris_full`); 422 for an unknown profile |
| `GET /cases/{case_id}` | `CaseInfo` | the question and its runs, each `queued`, `running`, `completed` or `failed` |
| `GET /runs/{run_id}` | `CaseView` | the record so far, folded from the run's events; includes the evidence graph |
| `GET /runs/{run_id}/analysis` | `CaseAnalysis` | 404 until the Judge has written it, and for a run that failed without one |
| `GET /runs/{run_id}/graph` | `EvidenceGraph` | the graph alone (PLAN 8.1) |
| `GET /runs/{run_id}/events` | `text/event-stream` | the run's events, live; `?replay=true` re-streams a finished run; see below |
| `GET /evidence/{run_id}/{evidence_id}` | `EvidenceDetail` | the evidence, its chunk and document, and every citation of it with its latest verification |
| `GET /documents/{doc_id}` | `DocumentDetail` | a corpus document from Postgres with its text by paragraph; a statute's sections stand in for paragraphs |
| `GET /configs` | `ConfigInfo[]` | the profiles `POST /cases` accepts: Juris, its ablations, B0-B2 |

An unknown case, run, evidence or document is a 404 with `{"detail": "... not found"}`; a bad
body is FastAPI's 422. CORS allows `JURIS_API_CORS_ORIGINS` (default the local frontend on port
3000), `GET` and `POST`.

A paragraph's `seq` is what a chunk's `para_start` and `para_end` count, so the UI can go from a
citation to the paragraph in the full document (IDEA_final §13, traceability).

## The event stream

`GET /runs/{run_id}/events` sends every event of the run in `seq` order, then each new one as it
is stored, and closes after the run's last event:

```text
retry: 3000                      once: reconnect after 3 s

id: 12                           the event's seq
data: {"event_id": "evt_000012", "seq": 12, "type": "claim_created", ...}

: ping                           a comment after JURIS_API_SSE_PING_SECONDS (15) of quiet

event: end                       after run_completed or run_failed, then the server closes
data: {}
```

`data` is the envelope exactly as stored (`schemas/event.schema.json`), on one line. In a
browser, `new EventSource("/runs/<id>/events")` with `onmessage` gets every event; listen for
`end` to close it. On a dropped connection `EventSource` reconnects with `Last-Event-ID: <seq>`
and the stream resumes at the next seq, so no event is missed or repeated; `?after=<seq>` does
the same for clients that cannot set the header. Once the client has the run's last event, a
reconnect gets `204 No Content`, which tells `EventSource` to stop. An unknown run is a 404, a
`Last-Event-ID` that is not a seq a 400.

### Replay

`GET /runs/{run_id}/events?replay=true&speed=2` re-streams a finished run in the same format,
paced by its events' timestamps: each gap divided by `speed` (default 1) and capped at
`max_gap` seconds (default 2, at most 60), since a single model call can hold a run for half a
minute. Through a wait longer than the ping interval it sends `: ping`. `Last-Event-ID` and
`after` resume a replay as they resume a live stream, and a reconnect after the end gets
`204`. Replaying a run that has not finished is a `409`: follow it live instead.

New events reach the stream without polling: the run's thread appends to the store, and the
store wakes the run's open streams on the server's loop (`call_soon_threadsafe`). Whether a
run has ended is read from the store (its last event is `run_completed` or `run_failed`), not
from the run's status, which is set only after the run's files are written.

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

## Persistence

The server keeps its events in Postgres, in the `events` table (migration 0001): one row per
event, the envelope's fields as columns and the payload as JSONB. Reading parses each row back
into its event, so the stream sends exactly what the run emitted, and the store applies the
in-memory store's ordering rules (a run starts with `case_created`, its seq increases, its case
stays the same). Each run's manifest also goes to the `runs` table.

When the server starts, `create_app` restores every run the store holds: its case (the
question and its runs) and its status, read from its first and last events. All routes
then work for runs from before the restart. A run whose last stored event is not
`run_completed` or `run_failed` was cut off by the last shutdown. It is closed with
`run_failed` ("interrupted: the server stopped before the run finished"), so its view, its
case and its stream end honestly. Queued runs are not resumed: a restart abandons them like
running ones.

Each finished run is also written to `<runs_dir>/api/<run_id>/` (`manifest.json`,
`events.jsonl`, `caseview.json`, `calls.json`), as `run_case.py` does. Tests keep events in
memory, except two `db` tests that restart the server and round-trip every event type.

## Mock mode

`JURIS_API_MODE=mock` serves the 2.4 fixtures through the same routes, with no model,
retrieval or database:

- The fixtures' runs are in the store from the start, finished: `run_mock_coercion_full`
  (case `CASE-MOCK-001`, completed) and `run_mock_failure` (`CASE-MOCK-002`, failed on its
  budget). Their views equal `fixtures/runs/*.caseview.json`, and they can be replayed.
- `POST /cases` (any profile) runs the coercion run's proceedings again as a new run of the
  question asked: the same payloads, paced by the fixture's timestamps `JURIS_API_MOCK_SPEED`
  (10) times faster, so the full run takes about 50 s and the stream behaves as a live one.
- `GET /documents/{id}` serves the fixtures' mock documents, their chunks standing in for
  paragraphs (the fixtures carry no full text).

Every mock document is titled `[MOCK] ...` with a `mock://` source. Mock runs' manifests go to
`<runs_dir>/api-mock/`.

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

## Live check of persistence, replay and mock mode (2026-10-24, local 7B)

`uvicorn` in live mode on the working database (`JURIS_API_SSE_PING_SECONDS=2`), JE-007's
question, `juris_full`, model calls from the cache:

- The run streamed 184 events, seq 1 to 184 with no gaps. Then the server was killed
  (`Stop-Process -Force`) and started again. The new server listed the case with its run
  `completed`, `GET /runs/{id}` gave the same view (62 evidence, 4 claims, the analysis), and
  the stream sent the same 184 events, identical. `Last-Event-ID: 40` resumed at 41, and the
  last seq got `204`.
- Replay at `speed=2`: the run took 137 s, with 126 s of that in five gaps of more than 4 s
  (model calls). With the default `max_gap` of 2 s the replay took 15 s (16 s expected). With
  `max_gap=60` it took 69 s (69 s expected), with 31 pings through the long gaps. The replayed
  events were identical to the originals and valid against `event.schema.json`.
- A second run was killed partway through, after 181 events. After the restart it was
  `failed` with the interrupted reason, and its stream ended with that `run_failed` (182
  events).
- Mock mode on a fresh server: `CASE-MOCK-001` and the failed fixture run were there, and
  `GET /documents/MOCK-ICA-S15` returned the mock section. A new case streamed 166
  schema-valid events in 51 s (49.5 s expected at speed 10), ending with `run_completed`.
  Its view validated against `case_view`, and a replay of the coercion fixture at speed 20
  sent its 166 events in 26 s.

## Live check of the event stream (2026-10-24, local 7B)

`uvicorn` with `JURIS_API_SSE_PING_SECONDS=2`; `curl -N` on `GET /runs/{id}/events` right after
`POST /cases` (JE-007's question, `juris_full`, model calls from the cache):

- 184 events, seq 1 to 184 with no gap, from `case_created` to `run_completed`, then
  `event: end`; the stream opened while the run was at its start and followed it live.
- Reconnecting with `Last-Event-ID: 40` resumed at seq 41 (the other 144 events); with the last
  seq, `204 No Content`.
- Pings: the first check sent 51 where the run's quiet gaps allow about 29. A wake-up scheduled
  for an event the stream had already sent ended a wait with nothing new, which the stream took
  for quiet. After the fix (such a wake-up goes back to waiting, and a test covers it) the
  second run sent 11 pings, exactly the number its quiet gaps of 2 s allow.
