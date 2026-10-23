# Cross-examination: objections, replies, open questions (PLAN 7.8)

Code: `backend/juris/agents/cross_examiner.py`, `backend/juris/agents/counsel_reply.py`,
`backend/juris/pipeline/stages/cross_exam.py`, prompts `backend/juris/prompts/cross_examiner.md`
v1 and `counsel_reply.md` v1. Decision: D-057.

## What S6 does

- **Objections.** One Cross-Examiner call per issue sees both sides' claims, each with its
  arguments: reasoning, every citation's quote, its S4 result and S3 fit, and which claim a
  counterargument answers, plus the cited judgments' precedent cards. It raises up to 4 typed
  objections (IDEA_final §6.1's 8 types), each against a claim or argument ID, with a severity.
  The most severe are kept across issues, at most 8 per case.
- **Contested.** Every claim an objection targets (directly or through its argument), or a
  counterargument answers, becomes contested. That now includes claims from S5, which are not
  verified until S7.
- **Replies.** Each targeted counsel gets one call with all its objections and replies once to
  each: `maintain` (with why) or `concede`. It sees what the objection targets, not the pool,
  and cannot add citations.
- **Outcome.** A conceded objection is resolved and its claims fall. A maintained or unanswered
  one becomes an unresolved question for the Jury and the Judge.

## Result (Juris-Eval dev, 18 items, 2026-10-23)

`uv run scripts/eval_cross_exam.py run` takes each item through S0-S6 with the local
`juris-qwen2.5-7b`. `report` pools the counts:

| | value |
|---|---|
| issues / claims | 31 / 105 |
| objections proposed / raised (cap of 8 hit on 2 items) | 84 / 81, every item has some |
| types | 33 citation_mismatch, 22 logical_gap, 14 unsupported_claim, 4 distinguishable_precedent, 4 missing_evidence, 2 overruled_or_doubted, 2 wrong_statute_or_version |
| against A / B; claims / arguments | 45 / 36; 58 / 23 |
| replies: conceded / maintained / none | 55 / 25 / 1 |
| unresolved questions | 26 |
| claims contested / fell | 84 / 50 |
| S6 tokens per item | mean 7,700, max 13,700 |

- **7.8's checks hold.** All 81 objections target a claim or argument of their issue. The only
  rejection on the 7B was one output naming arguments of another issue (A-001, A-002): both
  were dropped after the repair. Every reply is recorded once, against the objection it names,
  and the open objections are in the record as unresolved questions.
- **The objections are mostly sound.** Spot check (`sample --n 10 --seed 0`): 9 of 10 point at
  a real weakness, usually a quote that does not carry the claim, which the S4 result had
  already flagged. The cross-examiner objects to both sides (45 against A, 36 against B).
- **Counsel concedes a lot, sometimes inconsistently.** 55 of 80 replies concede (the prompt
  says conceding a real flaw is better than defending it), and 50 of 105 claims fall. In the
  sample, two concessions then argue for the claim anyway (JE-031 O-001, JE-019 O-003): the
  stance is taken at its word. A stronger model should maintain more points it can defend.
- **A first run found a lifecycle gap.** S5's claims are only verified at S7, and the lifecycle
  had no proposed -> contested step. 17 conceded objections against them changed nothing.
  D-057 allows that step. S7 still records the results for their citations.
- **Cost:** about 7,700 tokens per case, small next to S5.
