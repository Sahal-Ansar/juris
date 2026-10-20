# Counsel: citation validity after repair (PLAN 7.5)

Code: `backend/juris/agents/counsel.py`, `backend/juris/pipeline/stages/opening.py`, prompt
`backend/juris/prompts/counsel.md` v1. Decision: D-052.

## What S2 does

- **One Counsel per position of every issue.** All the calls run at once (A and B in parallel).
  The outputs are written issue by issue and position by position, so IDs do not depend on
  timing.
- **Same evidence, same prompt.** Each counsel sees the issue, its own position, the other
  positions, S0's facts and assumptions, and the issue's evidence pool. The pool is cut to
  `max_evidence_chars` for a small-context model. The prompt is the same for every counsel but
  the assigned position.
- **Output:** up to 6 claims, each with one sentence of reasoning and citations (evidence ID +
  verbatim quote). The pinpoint comes from the chunk.
- **Checks** (7.2's loop):
  - a claim needs text, reasoning and a citation;
  - a citation must name an item counsel was shown, and quote it exactly.

  The problems go back once, and what still fails is dropped.

## Result (Juris-Eval dev, 18 items, 2026-10-20)

`uv run scripts/eval_counsel.py run` takes each item through S0, S1 and S2 with the local
`juris-qwen2.5-7b`, with counsel's evidence cut to 28,000 characters. `report` pools the
counts:

| | citations | failing existence or quote | validity |
|---|---|---|---|
| first outputs | 195 | 119 | 0.390 |
| **after the one repair** (7.5's measure) | 127 | 62 | **0.512** |

- **The bar of at least 0.95 is not met on the local 7B.** As with B1 and B2 (D-044), the model
  paraphrases instead of copying. 7.5 stays open until it is re-measured with the intended
  model.
- **Repair helps here, unlike B1's.** Counsel gets each failing citation named back, and its
  validity rises from 0.39 to 0.51. B1's repair, which got only the failing quotes, fixed 6 of
  103.
- **What reached the record is clean:** 61 claims (27 for A, 34 for B) with 65 citations, all
  passing invariants 1-2. Every item has claims; 51 of 60 counsel outputs needed a repair.
- **Cost:** about 57,500 tokens per case for S0-S2 (30 issues, 60 counsel calls plus repairs).
