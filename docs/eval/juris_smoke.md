# Juris end to end: smoke run of juris_full and the ablations (PLAN 7.12)

Code: `backend/juris/pipeline/system.py` (`JurisSystem`, registered as `kind: juris` in the
experiment runner), `scripts/run_case.py`, profiles `configs/pipeline/juris_full.yaml` and
`no_*.yaml`. Decision: D-060.

## The profiles

| profile | what changes from juris_full |
|---|---|
| `juris_full` | nothing: S0-S10, 1 rebuttal round, 3 searches per counsel, 3 jurors, counsel's evidence cut to 28,000 characters for the local model |
| `no_opponent` | one counsel per issue (position A; labels are seeded at random), no rebuttal |
| `no_precedent_analyst` | S3 as a passthrough: precedent cards from code facts only, no model |
| `no_rebuttal` | S5 off |
| `no_cross_exam` | S6 off: no objections, replies or unresolved questions |
| `no_jury` | S8 off: the Judge reads the record without jury scores |
| `no_verifier_gates` | S4 and S7 off: counsel, the cross-examiner and the jury see no verdicts; the citations are verified once, just before the Judge (by S10's gate), and the Judge's own at S10 |

`uv run scripts/run_case.py "question"` (or `--item JE-001`, `--config no_jury`) runs one case
and prints the issues with leanings and confidence, the summary and the limitations; it writes
the manifest, event log, folded record, `CaseAnalysis` and calls.

## Result (Juris-Eval dev, first 5 items, local 7B, 2026-10-23)

`uv run scripts/run_experiment.py --config <profile> --split dev --limit 5`, scored by the
answer metrics (5.4) without a grader. Every run of every profile completed (35/35) with an
analysis. The runs used the 7.12 code before it was committed (`results.csv` says `3640c87`,
dirty).

| profile | citation validity | faithfulness | unsupported claims | authority recall | tokens per case |
|---|---|---|---|---|---|
| **juris_full** | 1.000 | 0.407 | 0.569 | 0.267 | 146,898 |
| no_opponent | 1.000 | 0.333 | 0.860 | 0.000 | 64,973 |
| no_precedent_analyst | 1.000 | 0.373 | 0.627 | 0.267 | 137,348 |
| no_rebuttal | 1.000 | 0.507 | 0.561 | 0.267 | 104,422 |
| no_cross_exam | 1.000 | 0.407 | 0.522 | 0.267 | 138,403 |
| no_jury | 1.000 | 0.407 | 0.490 | 0.267 | 132,930 |
| no_verifier_gates | 1.000 | 0.367 | 0.587 | 0.267 | 153,571 |
| B1 (v2), same 5 items | 0.500 | 0.292 | 0.880 | 0.200 | 17,597 |
| B2, same 5 items | 0.133 | 0.000 | 0.960 | 0.267 | 139,889 |

- **7.12's check holds:** juris_full and every ablation run over 5 dev items without crashing.
- **Validity is 1.0 for every Juris profile by construction:** S10 removes any citation that is
  not verified or weak. B1 and B2 keep what they write, so theirs is the share of their own
  citations that check out. Faithfulness and the unsupported-claim rate are the comparable
  numbers: Juris 0.41 and 0.57 against B1 0.29 and 0.88.
- **Five items say little about the ablations;** the confidence intervals overlap everywhere.
  Two signals worth checking on the full split: without the opponent, unsupported claims rise to
  0.86 and authority recall falls to 0; without the jury or the cross-examination, the
  unsupported rate falls a little, which the full evaluation should explain or refute.
- **The five juris_full analyses** lean (per issue, with confidence): JE-001 leaning_B
  (moderate), leaning_A (moderate); JE-003 leaning_B (moderate), balanced (low); JE-005
  leaning_B (low); JE-007 leaning_B (moderate), strongly_B (high); JE-009 leaning_A (moderate).
  Each lists its unverified treatment and corpus coverage among the limitations.

## Cost and latency

- **Tokens:** juris_full 146,900 per case on these 5 (92,600 to 206,800); 158,700 over all 18
  dev items (`docs/eval/judge.md`). By stage on dev: S5 about 48,700, S8 13,100, S9 11,800,
  S6 7,700; the rest is S0-S4 and the verifier.
- **B2 is compute-matched:** it spends 159,000 per case over the 18 dev items against
  juris_full's 158,700 (ratio 1.00), and 139,900 against 146,900 on these 5 (0.95), within the
  ±20% PLAN 6.4 asks. `target_tokens` stays at 170,000 (D-060).
- **Latency:** one juris_full case with the cache off (JE-009) took 345 s on the local 7B
  (39 calls, 144,100 tokens). The juris_full rows above replayed the cache (about 40 s a case);
  the ablations' wall times (114 to 332 s) mix cached and new calls. USD is 0: the local model
  has no price.
