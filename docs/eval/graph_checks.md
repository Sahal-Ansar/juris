# Graph integrity checks (PLAN 8.2)

Code: `backend/juris/graph/checks.py` (`run_checks`, `hints`), metrics
`backend/juris/eval/metrics/graph.py`, the Judge's record checks in `backend/juris/agents/judge.py`,
`scripts/graph_checks.py`. Tests: `backend/tests/test_graph_checks.py`. Decision: D-062. The
graph itself: [graph.md](graph.md).

## The checks

All five read the evidence graph (8.1), which is built from the Case Record, so they can be run
on any run, finished or not, with no model.

| Check | What it finds | Read from |
|---|---|---|
| orphan claims | no citation of the claim verified or was weak: all unsupported, invalid or unchecked, or none at all | `cites` edges (best result over every gate) |
| weak-only claims | supported, but only by weak (partial) citations, none verified | `cites` edges |
| positions with no verified support | no standing claim of the position (status not `falls` or `unsupported`) cites verified evidence | claim `supports` position, then `cites` |
| contested issues | two or more positions of the issue have verified support: the verified evidence pulls both ways | the previous check, by issue |
| circular support | a loop in the support relation | `supports` edges; `cites` and `quotes` read backwards (the evidence supports the claim citing it, the source supports its evidence) |

"Orphan" means no support that held, not a missing edge: in every saved run each claim has an
argument and at least one citation (counsel cannot make a claim without one), so a structural
definition would never fire. The builder's edges cannot close a support loop (documents and
evidence only support, positions are only supported), so circular support is 0 by construction
today; the check guards the graph against new edge kinds (claim-to-claim support, say) and is
tested on hand-made loops.

## Where they appear

- **Metrics** (5.4 suite, every config, every run): `orphan_claim_rate`, `weak_only_claim_rate`
  (over claims), `unverified_position_rate` (over positions), `contested_issue_rate` (over
  issues), `circular_support` (a count). The IDs found are in each metric's `details` in
  `scores.json`; the means are headline values in `summary.md` and `results.csv`. A rate is
  n/a when there is nothing to count (B0 records no claims). Lower is better, except the
  contested rate, which describes the case.
- **The Judge** (S9): before it writes, the checks become sentences per issue
  (`hints`), shown in each issue's block as "Record checks" and listed in the analysis's
  `limitations` as "Record check: ...". Claims that fell or were found unsupported are left out
  of the hints, since the record already says they count for nothing. The prompt (judge v2) says
  to let them temper the confidence and not to repeat them. S9 records how many it gave
  (`s9.record_checks`).

## On the saved dev runs (2026-10-24)

`uv run scripts/graph_checks.py <eval/results>/*-dev-n5-s1 C:/juris-data/runs/cases`: the 35
runs of the 7.12 smoke test (5 dev items, local 7B) and the two `run_case.py` cases, all run
before this step, so their Judge did not see the checks. Counts over the runs:

| experiment | runs | claims | orphan | weak-only | positions | unverified | issues | contested | loops |
|---|---|---|---|---|---|---|---|---|---|
| juris_full | 5 | 24 | 12 | 1 | 16 | 10 | 8 | 1 | 0 |
| no_cross_exam | 5 | 24 | 12 | 1 | 16 | 9 | 8 | 1 | 0 |
| no_jury | 5 | 24 | 12 | 1 | 16 | 10 | 8 | 1 | 0 |
| no_opponent | 5 | 7 | 5 | 0 | 16 | 16 | 8 | 0 | 0 |
| no_precedent_analyst | 5 | 24 | 12 | 3 | 16 | 12 | 8 | 0 | 0 |
| no_rebuttal | 5 | 16 | 8 | 0 | 16 | 12 | 8 | 0 | 0 |
| no_verifier_gates | 5 | 24 | 16 | 0 | 16 | 10 | 8 | 0 | 0 |
| run_case.py cases | 2 | 11 | 7 | 0 | 4 | 1 | 2 | 1 | 0 |

The metrics (mean over the runs):

| experiment | orphan claims | weak-only claims | unverified positions | contested issues | circular |
|---|---|---|---|---|---|
| juris_full | 0.440 | 0.067 | 0.550 | 0.200 | 0 |
| no_cross_exam | 0.440 | 0.067 | 0.500 | 0.200 | 0 |
| no_jury | 0.440 | 0.067 | 0.550 | 0.200 | 0 |
| no_opponent | 0.500 | 0 | 1.000 | 0 | 0 |
| no_precedent_analyst | 0.393 | 0.183 | 0.700 | 0 | 0 |
| no_rebuttal | 0.410 | 0 | 0.800 | 0 | 0 |
| no_verifier_gates | 0.593 | 0 | 0.600 | 0 | 0 |
| B0 (18 items) | n/a | n/a | 1.000 | 0 | n/a |
| B1 v2 (18 items) | 0.796 | 0.059 | 0.861 | 0.028 | 0 |
| B2 (18 items) | 0.826 | 0.025 | 0.833 | 0 | 0 |

What they show:

- **Half of Juris's claims have no support that held.** In juris_full 12 of 24 claims are
  orphans; most of them fell or were found unsupported, which the record already says. The
  checks add three that **still stand** with no support: C-001 on JE-003 (contested) and
  C-005 and C-006 on JE-007. Before this step the Judge had no signal about them.
- **More than half the positions rest on nothing verified** (10 of 16). On 3 of 5 items a whole
  issue has no verified position, which is when "insufficient evidence" or low confidence is
  right. Only JE-009 is contested (both sides verified).
- **Without an opponent nothing is verified on either side** (16/16 positions): the single
  counsel makes 7 claims over 5 items and none that stands has a verified citation.
- **Juris's claims hold far better than the baselines'**: 0.44 orphan against 0.80 (B1) and 0.83
  (B2), on different item sets (5 against 18), so read it as a direction, not a measurement.
- No run has a support loop, as expected.

## A live run with the record checks

`scripts/run_case.py --item JE-007` (juris_full, local 7B; 49 calls, 42 from the cache, so only
the Judge and S10 ran anew; 193 s): the analysis lists four record checks, exactly the ones the
saved run predicts (no verified position on I-1, I-2.A without verified support, C-005 and
C-006 standing with no support). The Judge still gave I-1 `leaning_B`, moderate confidence,
against its own record check: the 7B model does not yet let the checks temper the leaning. The
limitations make the gap visible to the reader; a rule in code (for example capping confidence
when no position is verified) is left for when the intended model has been measured.

The mock runs: `coercion_full` has two orphan claims (both fell), one weak-only claim (C-006,
still standing), every position verified and both issues contested; `failure_case` has one
orphan and no verified position.
