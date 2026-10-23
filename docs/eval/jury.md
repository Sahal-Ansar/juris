# Jury: ballots and the aggregate (PLAN 7.10)

Code: `backend/juris/agents/juror.py`, `backend/juris/pipeline/stages/jury.py`, prompt
`backend/juris/prompts/juror.md` v2. Decision: D-058.

## What S8 does

- **Three jurors** (`jurors`), all at once, differing only by sampling seed. Each sees the same
  frozen record, issue by issue: the positions; every claim with its status, its arguments and
  each citation's quote with its verification result and precedent fit; the precedent cards;
  the objections with counsel's replies; the unresolved questions. No retrieval scores, no
  evidence pool.
- **A ballot** scores every position 1-5 on the four rubric criteria (higher is better for the
  position) with a justification citing record IDs. The output has no evidence fields, so a
  juror cannot add evidence; cited IDs must exist. A ballot is all or nothing: a position left
  unscored drops it.
- **The aggregate** is code: per position and criterion, the mean and the spread (max minus
  min), plus `overall` (each ballot's mean of the four). An issue is contested when a position's
  `overall` spread is more than 1.0.

## Result (Juris-Eval dev, 18 items, 2026-10-23)

`uv run scripts/eval_jury.py run` takes each item through S0-S8 with the local
`juris-qwen2.5-7b` (three jurors). `report` pools the results:

| | value |
|---|---|
| ballots cast | 54 / 54 |
| aggregate reproduced from the ballots in reverse order | 18 / 18 |
| contested issues | 9 / 31 |
| mean per position (spread): evidence / precedent fit / counterarguments / unresolved risk / overall | 2.97 (0.68) / 2.84 (1.40) / 2.73 (1.05) / 3.13 (0.87) / 2.92 (0.68) |
| jury's higher `overall` = the side with more standing claims | 19 / 22 (9 issues tied) |
| record IDs cited per justification | 4.7 |
| S8 tokens per item | mean 13,100, max 19,400 |

- **7.10's checks hold.** Every ballot validates: 4 outputs needed an unknown ID dropped
  (`AC-001`, or a claim of another case's numbering), and none was lost. The aggregate is
  pure code, the same in any ballot order. Jurors have no way to add evidence.
- **The scores follow the record.** Where one side has more claims still standing (not fallen,
  not unsupported), the jury rates it higher on 19 of 22 issues.
- **Jurors disagree about precedent fit most** (mean spread 1.4), the criterion the record
  says least about directly.
- **Two changes after the first dev run (D-058).** Contested was first any criterion spread of
  2 or more: every issue (31/31) qualified, so it said nothing; the `overall` spread flags 9. One
  ballot (of 54) was lost to a `precedent_fit` of 0 for a position relying on no judgment;
  prompt v2 asks for 3 there and never 0.
- **Cost:** about 13,100 tokens per case for the three jurors.
