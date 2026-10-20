# Researcher: evidence pool recall (PLAN 7.4)

How much of the gold authority the S1 evidence pool holds, compared with B1's pool. Code:
`backend/juris/agents/researcher.py`, `backend/juris/pipeline/stages/research.py`. Decision:
D-051.

## What S1 does

Per issue framed in S0 (7.3, D-050):

1. **Search.** All of the issue's S0 search queries go to hybrid search together: lexical,
   bge-m3 and multilingual-e5, fused by RRF, no rerank (D-037). It returns 60 candidates.
2. **Provisions.** S0's statute references are hints. A section is kept when the search found
   one of its chunks, or when at least 3 of the candidate judgments cite it (citation graph,
   3.9). A kept section's chunks go first.
3. **Select** up to 25 chunks in rank order, at most 3 from one judgment. Statute sections are
   not capped. If that leaves fewer than 10, the cap is relaxed.
4. **Register** the chunks for the issue; a chunk shared by two issues keeps one `E-` ID.

The Researcher is code. It makes no model call, and it never says which side an item helps.

## Result (Juris-Eval dev, 18 items, 2026-10-20)

`uv run scripts/eval_researcher.py run` frames each item with the Issue Framer (local
`juris-qwen2.5-7b`, prompt v2; 1.7 issues per item on average) and builds the pool. `report`
compares it with the 20-chunk pools of B1's dev run (`b1-v2-dev-nall-s1`).

Recall is per item, averaged over the 18 items:

- supporting: gold supporting judgments in the pool;
- all-gold: supporting and contrary judgments;
- sections: gold statute sections with a chunk in the pool.

| Pool | chunks | documents | supporting recall | all-gold recall | section recall |
|---|---|---|---|---|---|
| B1 (one search, 20 chunks) | 20.0 | 11.8 | 0.676 | 0.662 | 0.046 |
| **S1 Researcher** | 40.2 | 23.8 | **0.750** | **0.750** | **0.287** |

**S1's judgment recall is at least B1's**, which is 7.4's bar. The pool is twice B1's size:
1.7 issues at up to 25 chunks each, against B1's single 20. The two are not size-matched, and
the plan does not ask for that.

| Item | B1 supporting | S1 supporting | B1 sections | S1 sections |
|---|---|---|---|---|
| JE-001 | 1.00 | 1.00 | 0.50 | 0.50 |
| JE-003 | 1.00 | 1.00 | 0.33 | 0.67 |
| JE-005 | 1.00 | 1.00 | 0.00 | 1.00 |
| JE-007 | 0.33 | 0.33 | 0.00 | 0.50 |
| JE-009 | 0.50 | 0.50 | 0.00 | 0.50 |
| JE-011 | 0.33 | 0.67 | 0.00 | 0.00 |
| JE-013 | 0.50 | 1.00 | 0.00 | 0.00 |
| JE-015 | 0.00 | 0.00 | 0.00 | 0.00 |
| JE-017 | 0.50 | 1.00 | 0.00 | 0.00 |
| JE-019 | 1.00 | 0.00 | 0.00 | 1.00 |
| JE-021 | 1.00 | 1.00 | 0.00 | 0.00 |
| JE-026 | 1.00 | 1.00 | 0.00 | 0.50 |
| JE-031 | 1.00 | 1.00 | 0.00 | 0.00 |
| JE-036 | 0.00 | 0.00 | 0.00 | 0.00 |
| JE-041 | 1.00 | 1.00 | 0.00 | 0.50 |
| JE-046 | 1.00 | 1.00 | 0.00 | 0.00 |
| JE-051 | 1.00 | 1.00 | 0.00 | 0.00 |
| JE-056 | 0.00 | 1.00 | 0.00 | 0.00 |

- **Gains** come from more queries per issue: JE-011, -013, -017 and -056 find gold judgments
  that B1's single query set missed.
- **One loss: JE-019.** S0 framed one issue there, and its queries led away from the gold
  judgment that B1's queries found. S1 is only as good as S0's framing.
- **JE-015 and JE-036 fail in both pools.** These are retrieval misses, not framing misses.

## Statute references: confirming the hints

S0's 33 references on dev name 6 gold sections. Two tests were compared for deciding which
references to keep:

| Test | kept | gold among them |
|---|---|---|
| the search found the section's chunk | 3 | 1 of 6 |
| ≥ 1 candidate judgment cites it | 13 | 6 of 6 |
| **≥ 3 candidate judgments cite it** (chosen) | 10 | 6 of 6 |
| ≥ 3 of the top 25 cite it | 6 | 4 of 6 |

Statute chunks seldom rank in hybrid search against judgment text, so the search test missed
most of the right sections (s. 74 for liquidated damages, s. 56 for frustration). The citation
graph keeps those, and still drops guesses such as s. 146 for undue influence or s. 111 for a
guarantee, which no candidate judgment cites. S1 uses either test. With it, section recall
rose from 0.148 to 0.287, and judgment recall did not change.

The threshold was chosen on these same 18 items, so the figures above are optimistic for
unseen items. Re-check on test in 10.1.
