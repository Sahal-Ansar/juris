# Baseline results on Juris-Eval dev (PLAN 6.5)

The numbers Juris has to beat, and a check that the metrics tell the systems apart. Systems
and their design: `docs/eval/baselines.md`. Decisions: D-041 (local model), D-044 (B1),
D-045 (B2).

## Setup

| | B0: LLM only | B1: single-agent RAG | B2: compute-matched |
|---|---|---|---|
| Experiment | `b0-dev-nall-s1` | `b1-v2-dev-nall-s1` | `b2-dev-nall-s1` |
| Profile | `b0.yaml` | `b1.yaml` (answer prompt v2, quote repair) | `b2.yaml` (8 rounds, 170k budget) |
| Run | 2026-10-01 | 2026-10-02 / 2026-10-18 | 2026-10-19 |

- All three: the 18 Juris-Eval dev items, seed 0, every run completed. The model is the local
  `juris-qwen2.5-7b` (Ollama, 16k context, D-041) for answering and for the verifier.
- **These are provisional numbers.** The intended model is not available yet, and 6.3 stays open
  until it is re-measured (D-044).
- No grader model is approved (D-038), so key-point coverage and the quality rubric are empty.
- Juris-Eval and the verifier pairs have not had their human review yet.
- Reproduce the tables with
  `uv run scripts/compare_experiments.py b0-dev-nall-s1 b1-v2-dev-nall-s1 b2-dev-nall-s1`,
  and `--paired` for two of them.

## Comparison

Means over items, with 95% bootstrap CIs. "n" is the number of items with a value: citation
validity and faithfulness are undefined for an answer with no citations.

| Metric | B0 | B1 | B2 |
|---|---|---|---|
| citation validity | n/a (no evidence) | 0.291 (0.152-0.433), n 10 | 0.263 (0.141-0.386), n 18 |
| citation faithfulness | n/a | 0.140 (0.067-0.213), n 10 | 0.148 (0.054-0.252), n 18 |
| unsupported-claim rate | 1.000 (1.000-1.000) | 0.893 (0.820-0.960) | 0.866 (0.744-0.956) |
| authority recall (gold judgments) | 0.000 (0.000-0.000) | 0.278 (0.111-0.463) | 0.185 (0.056-0.343) |
| section recall | 0.028 (0.000-0.083) | 0.000 (0.000-0.000) | 0.028 (0.000-0.083) |
| **hallucinated-authority rate** (B0) | **0.889** (0.667-1.000), n 9 | — | — |
| tokens per case | 1,559 (1,426-1,670) | 20,454 (17,661-23,067) | 158,971 (146,761-172,015) |
| wall-clock per case | 488 s (68-1,312) | 193 s (149-237) | 890 s (716-1,041) |
| key-point coverage, rubric | n/a (no grader) | n/a | n/a |

- **B0's hallucinated-authority rate** is (mismatched + unverified) / judgments cited. It is
  0.889: 8 of the 9 judgments B0 named were found nowhere, neither in the corpus slice nor in its
  citation graph. It is an upper bound (`docs/eval/baselines.md`).
- Pooled over citations: B1 has 34 exact quotes of 131 (26%), B2 25 of 90 (28%). The
  verifier rated B1's 34 as 15 verified, 6 weak and 13 unsupported, and B2's 25 as 12, 2 and 11.

### B2 against B1 (paired, same items)

| Metric | n | B1 | B2 | B2 − B1 | 95% CI | p |
|---|---|---|---|---|---|---|
| citation validity | 10 | 0.291 | 0.240 | −0.051 | −0.283 to +0.173 | 0.67 |
| citation faithfulness | 10 | 0.140 | 0.138 | −0.001 | −0.149 to +0.155 | 1.00 |
| unsupported-claim rate | 18 | 0.893 | 0.866 | −0.027 | −0.182 to +0.099 | 0.73 |
| authority recall | 18 | 0.278 | 0.185 | −0.093 | −0.278 to +0.083 | 0.41 |
| tokens per case | 18 | 20,454 | 158,971 | +138,518 | +126,550 to +150,569 | < 0.001 |

## Do the metrics discriminate?

- **Retrieval against no retrieval: yes.** Authority recall is 0 for B0 and 0.28 for B1. B0
  cannot cite evidence, so its unsupported-claim rate is 1 by construction. B1 and B2 lower it.
  B0's hallucinated-authority rate shows the failure the citation rules exist to prevent.
- **B1 against B2: not on this model.** None of the quality differences is significant. 7.8
  times the tokens bought nothing measurable, which is a useful baseline result in itself (R1).
  With 18 items and single seeds, CIs are wide: a difference under about 0.2 would not show.
- **Citation validity is dominated by quoting ability.** On the 7B, most citations fail the
  exact-quote check before the verifier judges support, so faithfulness is capped by validity.
  This metric will separate configs only once the model copies text reliably. That is the
  reason for re-measuring with the intended model (D-044).
- Key-point coverage and the rubric, which should separate B1 from B2 on content, wait for an
  approved grader (D-038).

## Typical failures

1. **B0 invents authorities.**
   - JE-005 (non-compete after employment) cites "Bharat Steel Ltd. v. N. K. Ghosh".
   - JE-009 cites "Mumbai Electric Supply & Development Co. Ltd. v M.A. Murtuza".
   - JE-041 cites "Lackington, Holman and Co. Ltd. v. Lord Advocate, (1819) 176 ER 1360".
   - JE-019 lists "National Labor Laws, 1976" as a judgment.

   None exists in the corpus or in its citation graph. Only one of B0's nine judgments was real:
   *Satyabrata Ghose* in JE-017.
2. **B0 reaches for the wrong statute.** 7 of 18 answers cite s. 56 of the Contract Act
   (frustration). They include JE-001 (a gift deed and undue influence, where s. 16 is the
   provision) and JE-005 (restraint of trade, s. 27). Section recall is 0.03.
3. **The "quote" is the model's own claim.** In B1 on JE-013 (is time of the essence in a sale
   of a house?), a claim says failing the payment dates lets the vendor cancel. Its quote, "Clause
   4 requires the balance consideration to be paid in three instalments on specific dates...", is
   the claim reworded. The cited passage says the opposite: the payment terms were "delinked" from
   completion. All 14 of B1's citations on JE-013 failed the quote check. Of B1's 97 failing
   quotes, 39 share under 40% of their text with the chunk.
4. **Uncited claims leave empty answers.** B1 must cite every claim (prompt v2). When the 7B
   wrote claims without citations even after the schema retry, they were dropped, leaving 8 of
   18 answers with no claims (JE-001 lost 16). B2's revisions removed the empty answers, but they
   also shrank the rest: JE-017 went from 38 claims to 2, and JE-021 from 16 to 5.
5. **Revision drops gold authorities.** B2's lower authority recall comes from items where the
   revision cut the claims that cited the gold judgment. JE-021 (acceptance by encashing a
   cheque) and JE-031 (refund of tax paid under a void law) fall from recall 1.0 in B1 to 0 in B2.
   JE-001 and JE-013 move the other way, from 0 to 0.67 and 0.5.
6. **Counsel's argument cited as law.** B1 on JE-041 quotes "Mr. Shyam Diwan, learned senior
   counsel, appearing on behalf of the Remi Metals, submitted that..." for a proposition of law.
   B2 on JE-026 quotes "the next submission of the learned senior counsel for KIDC". The verifier
   rated these weak and unsupported. It was not fooled here, but this is the failure mode it
   missed twice in validation (D-043).
7. **Non-substantive text cited as support.**
   - B1 on JE-036 cites the heading of the Contract Act's s. 121, three times. That section was
     repealed in 1930 ("Rep. by the Indian Sale of Goods Act"), yet the search still returned it.
     B1's search sets no in-force filter.
   - B2 on JE-031 cites a headnote's reference line ("D. Cawasji & Co. v. State of Mysore ...
     referred to").

   The quotes are exact, so the citations count as valid. The verifier then marks them
   unsupported.
8. **Near-verbatim quotes that "clean up" the corpus.** In B2's smoke runs the model fixed an
   OCR typo (the chunk reads "undue influene"), turned `"fraud"` into `'fraud'`, and dropped a
   `***` elision mark in s. 19. Each counts as invalid under the exact-quote rule (D-038).
   - Of the failing quotes, 16 (B1) and 9 (B2) share at least 80% of their text with the chunk.
   - The rule is right to reject edits. But part of this gap is corpus noise, not fabrication.
   - Worth a look before the re-measure, as a cleaner fix in the corpus or a logged change to
     the normalisation.

## What Juris has to beat (provisional, local 7B)

- Hallucinated authorities: B0's 0.89, against none possible for evidence-bound configs.
- Citation validity of at least 0.95 (the 6.3 / 7.5 bar), against B1's 0.29 and B2's 0.26.
- Unsupported-claim rate below B2's 0.87, and authority recall above B1's 0.28, at a token
  budget comparable to B2's 159k per case.
