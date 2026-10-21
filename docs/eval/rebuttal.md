# Rebuttal round: targets and new evidence (PLAN 7.7)

Code: `backend/juris/agents/rebuttal.py`, `backend/juris/pipeline/stages/rebuttal.py`, prompts
`backend/juris/prompts/counsel_search.md` v1 and `counsel_rebuttal.md` v1. Decisions: D-055
(the rebuttal) and D-056 (keeping the better attempt after a repair).

## What S5 does

For every counsel (issue and position) that has opposing claims to answer, per round:

- **Searches.** Counsel sees the opposing claims and the titles already in its issue's pool, and
  asks for up to 3 searches. Each goes through the Researcher's `targeted_search`, which
  registers up to 3 new chunks for the issue (at most 3 per judgment, counting what is held).
- **Counterarguments.** Counsel sees the opposing claims not yet answered from its position,
  strongest first (verified, weak, proposed, unsupported). For each target it sees the
  reasoning and, per citation, the S4 result and the S3 precedent fit. The evidence is ordered
  as: its own new finds, what the targets cite, then the rest of the pool (cut to 28,000
  characters). It answers up to 3 targets, one counterargument each, each with citations.
- **In the record:** a claim for counsel's position, an argument with its citations, and a
  `Counterargument` pointing the argument at its target. S7 verifies the citations.
- **Checks:** S2's citation checks, plus: the target must be one shown, one counterargument per
  target, at most 3. The problems go back once, and what still fails is dropped.
- **Rounds:** 1 by default. A second round answers the first round's new claims; it is skipped
  with a `budget_warning` when the budget left will not cover what round 1 spent.

## Result (Juris-Eval dev, 18 items, 2026-10-21)

`uv run scripts/eval_rebuttal.py run` takes each item through S0-S5 and the S7 gate with the
local `juris-qwen2.5-7b`. `report` pools the counts:

| | value |
|---|---|
| issues | 31 |
| searches / new evidence registered | 87 / 258 |
| counterarguments in the record | 28 (15 by A, 13 by B; 2 items have none) |
| citations after the one repair: failing quote / total | 42 / 66 (validity **0.364**) |
| citations in the record / to evidence found in S5 | 30 / 8 |
| **S7 results** | **30 / 30** (8 / 8 for the new evidence): 5 verified, 5 weak, 20 unsupported |
| targets by status | 12 verified, 2 weak, 14 unsupported |
| S5 tokens per item | mean 48,700, max 107,600 |

- **7.7's checks hold.** Every counterargument in the record targets an opposing claim of its
  issue: the model never named an invalid target, and the record would have rejected one. Every
  rebuttal citation, including all 8 to evidence the searches found, has an S7 result.
- **Quotes are the weak point on the 7B.** All 69 validation errors were quotes not in their
  chunk (40), or counterarguments left without a citation once those were dropped (29).
  Validity after repair is 0.364, lower than S2's 0.512: the model quotes from memory (the text
  of s. 16, for instance) more often when it argues against a claim. As D-044 and D-052, this
  is re-measured with the intended model.
- **Searches find a lot and little is cited.** 258 new chunks, 8 cited. The new chunks come
  first in counsel's evidence, so it sees them; the 7B rarely quotes them correctly.
- **Spot check (`sample --n 10 --seed 0`):** 9 of 10 counterarguments argue against their
  target; one (JE-021 R-002) restates its target's point. Most are on point but cite weakly: 7
  of the 10 claims are unsupported at S7.
- **Cost.** S5 is the most expensive stage so far, about 48,700 tokens per case on average:
  the evidence and targets go to two calls per counsel, plus repairs (40). 7.12 recalibrates B2
  on the full pipeline's tokens.
- **The repair rule changed during this step (D-056).** The smoke run showed a repair that broke
  a valid quote and lost the whole counterargument. Agents now keep the attempt that keeps more
  after dropping. This also applies to S0-S3, so this run's S2 record can differ slightly from
  the one measured for 7.5 (`counsel.md`).
