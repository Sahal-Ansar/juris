# Judge and final verification (PLAN 7.11)

Code: `backend/juris/agents/judge.py`, `backend/juris/pipeline/stages/judge.py` (S9),
`backend/juris/pipeline/stages/final_verify.py` (S10), prompt `backend/juris/prompts/judge.md`
v1. Decision: D-059.

## What S9 and S10 do

- **S9, the Judge.** One call over the whole frozen record: per issue the claims with their
  statuses, quotes, verification and fit, the precedent cards, the objections with replies, the
  jury's scores, the **citable evidence** (verified or weak, with a quote that verified) and the
  unresolved questions (U1, U2, ...). It writes, per issue, a summary per position citing
  evidence inline (`[E-012]`) with its key evidence, key conflicts, a leaning and a confidence
  with a reason, and an answer to each question; then an overall summary (at most 250 words)
  and its limitations.
- **The analysis is built in code.** Inline references become one `[E-nnn]` per evidence (a
  claim or argument cited inline becomes the evidence it cites); pinpoints come from the
  chunks; supporting authorities are the judgments cited; every judgment relied on gets a
  "treatment not verified" limitation (D-013); assumptions and the corpus's coverage are listed;
  sources give court, date, citation and pinpoints. The Case Record checks invariants 3 and 5.
- **Gaps are filled, not fatal.** A missing issue, position or answer, or a summary too long,
  goes back for the one repair as a soft problem. If it is still missing, the analysis gets an
  explicit placeholder (an unanalysed issue is `insufficient_evidence`, `low`) and a limitation
  says so. Every case gets an analysis.
- **S10.** The last verification gate, then the Judge's citations: each sentence citing
  evidence is checked against each evidence it cites. Evidence that is not citable fails in
  code; the rest goes to the entailment judge with the sentence as the claim. A failing
  citation is removed, a sentence left without one is listed in the limitations, and the
  analysis is written again.

## Result (Juris-Eval dev, 18 items, 2026-10-23)

`uv run scripts/eval_judge.py run` takes each item through the whole pipeline, S0-S10, with the
local `juris-qwen2.5-7b`. `report` pools the results:

| | value |
|---|---|
| analyses written | **18 / 18** (no placeholder needed) |
| evidence cited in the final analyses (key and inline) | 29: 24 verified, 5 weak, **0 invalid or unsupported** |
| issues with a leaning and a confidence | **31 / 31** |
| leanings | 11 leaning_B, 9 balanced, 7 leaning_A, 2 strongly_A, 1 strongly_B, 1 insufficient |
| confidence | 21 moderate, 7 low, 3 high |
| leaning towards the jury's higher-scored position | 11 agree, 9 not, 11 balanced or a jury tie |
| S10: Judge citations checked / removed (not citable, unsupported) | 88 / 34 (19, 15) |
| sentences flagged as stated without support | 30 |
| unresolved questions answered / key conflicts / sources | 26 / 5 / 27 |
| tokens per case: S9 / whole pipeline | 11,800 / 158,700 (max 286,100) |

- **7.11's checks hold.** Every case ends with a valid analysis; no citation in a final
  analysis is to evidence that is not verified or weak; every issue has a leaning and a
  confidence.
- **The 7B cites thinly.** Only 54 of 208 sentences in the final summaries cite evidence, and
  a third of the inline citations it gives fail S10 (not citable, or the sentence says more
  than the quote). IDEA_final §10 asks for a citation on every sentence with a legal assertion:
  not met on the 7B, to be re-measured with the intended model.
- **The smoke run shaped three fixes (D-059).** The 7B cited claim and argument IDs as
  evidence, numbered the questions per issue itself, and once lost a position in its repair,
  which dropped the whole analysis. Claim and argument IDs are now read as the evidence they
  cite, answers are collected by label from any issue, and gaps are soft problems filled in
  code.
- **The Judge does not simply follow the jury** (11 agree, 9 not): its leaning comes from its
  own reading of the record. Whether that is better is for the 7.12 evaluation.
- **Cost.** The whole pipeline is about 158,700 tokens per case on the 7B; B2 spent about 159k
  against a 170k target. 7.12 recalibrates B2 on these numbers.
