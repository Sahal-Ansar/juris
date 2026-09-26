# Mock run fixtures (for the UI)

These files let you build every screen before the backend exists. They are exactly what the backend will send, validated against the same schemas.

> **Everything here is fictitious.** Every judgment, party, quote and paragraph number is invented. Documents are titled `[MOCK] …` and link to `mock://…`, and statute text is paraphrased. Never show these as real law outside development.

## Files

| File | What it is |
|---|---|
| `runs/coercion_full.jsonl` | A complete successful run (166 events), one JSON event per line, in order. |
| `runs/coercion_full.caseview.json` | The same run folded into a `CaseView`: the state the UI shows at the end. |
| `runs/failure_case.jsonl` | A short run that fails (27 events): two validation failures, a budget warning, skipped stages, `run_failed`. |
| `runs/failure_case.caseview.json` | Its folded `CaseView` (`status: "failed"`, no analysis). |

## Types

Import TypeScript types from `schemas/ts/juris.d.ts` (generated from the backend models; don't edit):

- `Event`: union of every event type, discriminated on `type`. Each event is an envelope (`event_id`, `run_id`, `case_id`, `seq`, `ts`, `stage`, `agent`, `type`) plus a typed `payload`.
- `CaseView`: the read model (issues, evidence with documents and chunks, claims, arguments, counterarguments, precedent cards, objections, unresolved questions, verifications, jury, analysis, timeline).
- `CaseAnalysis`: the final output (inside `CaseView.analysis`).

JSON Schemas for the same types are in `schemas/*.schema.json`.

## Two ways to use them

1. **Static screens:** load `*.caseview.json` and render it.
2. **Live screens:** stream `*.jsonl` event by event and fold each event into your state. Your client-side fold must produce exactly `*.caseview.json` at the end (PLAN 11.1 tests this). The rules are in `backend/juris/events/fold.py`:
   - apply events in `seq` order;
   - skip an event whose `seq` and `event_id` you have already applied (reconnects re-send events);
   - claims change status only through `claim_status_changed`;
   - objections change only through `objection_answered`.

To simulate the live stream (SSE frames with realistic delays):

```bash
uv run scripts/replay_fixture.py fixtures/runs/coercion_full.jsonl --speed 5
```

## The mock case: coercion

**Question:** "Can a contract be enforced when consent was obtained through coercion?" A contractor signed a reduced "full and final" settlement after the developer withheld dues it had admitted.

- **Issues:** `I-1` (was consent caused by coercion?) and `I-2` (can the settlement still be avoided after accepting payment and waiting?), each with positions `.A` and `.B`.
- **Evidence:** `E-001`–`E-020`, drawn from 9 mock judgments (Supreme Court and High Court) and 3 paraphrased sections of the Indian Contract Act.
- **Claims:** `C-001`–`C-010`. Across the run they pass through every status the UI must handle. `proposed`, `verified` and `unsupported` appear mid-run (see `claim_status_changed` in the timeline). The final statuses in `caseview.json` are:

  | Final status | Claims |
  |---|---|
  | `survives` | `C-003`, `C-005`, `C-009`, `C-010` |
  | `contested` | `C-001`, `C-004`, `C-007` (open objections) |
  | `weak` | `C-006` |
  | `falls` | `C-002` (unsupported, then withdrawn), `C-008` (invalid quote) |

- **Verifications:** include `verified`, `weak`, `unsupported` (`C-002`/`E-006`) and `invalid` (`C-008`/`E-018`, a quote that isn't in the source).
- **Precedent cards:** 8. `MOCK-SC-2003-04` has `treatment_verified: false`, and the analysis lists it under `limitations`. Show unverified treatment clearly.
- **Objections:** `O-001`–`O-004`. One is resolved; three are still open and produced unresolved questions.
- **Counterarguments:** `R-001` and `R-002` (rebuttal round, stage `S5`).
- **Jury:** 3 ballots and an aggregate with mean and spread per rubric criterion.
- **Analysis:** leaning per issue (`balanced`, `leaning_A`) with confidence and reasons, key conflicts, sources, limitations, and the fixed disclaimer. It never gives a verdict.

## Stages

`stage` is `S0`–`S10` (null for run-level events). Suggested display names:

| Code | Name | Code | Name |
|---|---|---|---|
| S0 | Issue framing | S6 | Cross-examination |
| S1 | Research | S7 | Verification #2 |
| S2 | Opening arguments | S8 | Jury |
| S3 | Precedent analysis | S9 | Judge |
| S4 | Verification #1 | S10 | Final verification |
| S5 | Rebuttal | | |

## Regenerating

The fixtures are generated from `scripts/build_fixtures.py`. After changing the models or the story:

```bash
uv run scripts/build_fixtures.py
```

A test fails if the committed files differ from what the script produces, or if they stop validating against `schemas/`.
