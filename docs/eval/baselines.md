# Baselines

PLAN 6.2-6.4, IDEA_final §5. Results tables come in PLAN 6.5 (`docs/eval/baselines_dev.md`).

## Which model

Every config uses the settings default model for its role (`.env`). There is no paid API key
yet (D-007), so the default is a local model served by Ollama (D-041):

- `juris-qwen2.5-7b`: Qwen 2.5 7B Instruct (Q4, ~4.7 GB) with a 16k-token context, on the RTX
  4070 laptop GPU. Set up with `scripts/ollama_setup.ps1`; `.env.example` has the four lines
  that point the `openai` provider at Ollama's OpenAI-compatible endpoint.
- To use Claude later: delete those lines and set `ANTHROPIC_API_KEY`. The default model is
  then `claude-opus-5-5` again; no code changes.

A 7B local model is far weaker than the intended model. Results with it test the pipeline
and give a floor, not the comparison the research questions need (IDEA_final §4).

## B0: LLM only (PLAN 6.2)

`backend/juris/baselines/b0.py`, prompt `backend/juris/prompts/baseline_b0.md`,
profile `configs/pipeline/b0.yaml`, tests `backend/tests/test_b0.py`.

- One structured call (`B0Answer`): facts, assumptions, 1-4 issues with positions A and B,
  leaning and confidence, numbered authorities in free text (case name + reporter citation,
  or Act + section), an overall summary and limitations. No retrieval.
- Post-processor (`backend/juris/baselines/authorities.py`): each authority is looked up in
  the corpus. Judgments by citation (4.5's normaliser, aliases and citation graph), else by a
  close title match (trigram score at least 0.75). Sections by `parse_provision`. Status:
  `corpus`, `outside_corpus` (a real judgment known from the citation graph but outside the
  slice), `mismatch` (the citation names a corpus judgment with different parties),
  `unverified` (found nowhere) or `unknown_act` (a statute outside the three Acts).
- The `CaseAnalysis` names only authorities found in the corpus (judgments by document ID;
  sections as sources whose pinpoint is the section ID, which `authority_recall.sections`
  counts). The others are listed in its limitations.
- Recorded with the scores (`system` metric): `b0.authorities_cited`, `b0.judgments_cited`,
  `b0.judgments_in_corpus`, `b0.judgments_outside_corpus`, `b0.judgments_mismatched`,
  `b0.judgments_unverified`, `b0.statutes_cited`, `b0.statutes_unverified`, and
  **`b0.unverified_judgment_rate`** = (mismatched + unverified) / judgments cited: the
  hallucinated-authority rate. It is an upper bound, since a real judgment outside the slice
  and never cited by a corpus judgment also counts as unverified.
- B0 registers no evidence, so citation validity and faithfulness do not apply (`None`).

Run: `uv run scripts/run_experiment.py --config b0 --split dev`.

### First run (local model, Juris-Eval dev, 2026-10-02)

Experiment `b0-dev-nall-s1`: 18 dev items, seed 0, `juris-qwen2.5-7b`, no grader. All 18
runs completed and every output validated as a `CaseAnalysis`.

| Value | Mean | 95% CI | n |
|---|---|---|---|
| authorities cited per answer | 1.83 | 1.28-2.33 | 18 |
| judgments cited per answer | 0.50 | 0.28-0.72 | 18 |
| statutes cited per answer | 1.33 | 0.89-1.83 | 18 |
| **unverified-judgment rate** (hallucinated, upper bound) | **0.889** | 0.667-1.000 | 9 |
| authority recall (gold judgments) | 0.000 | 0.000-0.000 | 18 |
| section recall | 0.028 | 0.000-0.083 | 18 |
| unsupported-claim rate | 1.000 | | 18 |
| tokens per case | 1,559 | 1,426-1,670 | 18 |

- The 7B model names few cases: 9 of 18 answers cite one judgment, the others none. 8 of
  those 9 judgments were found nowhere (e.g. "Bharat Steel Ltd. v. N. K. Ghosh", with no
  citation); one resolved to the corpus. The rate rests on only 9 judgments.
- It never named a gold authority, and its sections were often off-point (s. 56 for an
  undue-influence question).
- The unsupported-claim rate is 1 by construction: B0 cites no evidence.
- Wall-clock per case (mean 488 s, very uneven) reflects a shared laptop GPU, not B0.
- Graders were off (no approved grader model), so key-point coverage and the rubric are empty.
- Re-run with the intended model once an API key is set; nothing in B0 changes.

## B1: single-agent RAG (PLAN 6.3)

`backend/juris/baselines/b1.py`, prompts `backend/juris/prompts/baseline_b1_queries.md`,
`baseline_b1.md` and `baseline_b1_repair.md`, profile `configs/pipeline/b1.yaml`, tests
`backend/tests/test_b1.py`.

Question -> hybrid retrieval -> one LLM call -> cited answer, with the same retriever, model,
citation format and Citation Verifier as every config (IDEA_final §5).

- **Queries (S0).** `queries: llm`: one structured call writes 3 to `max_queries` short
  issue-style queries (the legal questions, doctrines and statute sections, no party names;
  D-037 found these retrieve better than the pasted fact pattern). `queries: question`: the
  question alone, no call. Recorded as `query_issued` events.
- **Retrieval (S1).** `RetrievalService.search(queries, k=evidence_k, rerank=rerank)`. The
  hits are shown to the model as `[E-001] <header>` + text, in hit order, and registered
  through `EvidenceRegistry` in that order, so the model's IDs are the registry's. The
  encoders run on the CPU (`RetrievalService.local(device="cpu")`), because the local LLM
  holds the 8 GB GPU; a few queries per case do not need it.
- **Answer (S9).** One structured call (`B1Answer`, `max_tokens` 6000): facts, assumptions,
  1-4 issues with positions A and B, each position's claims (one legal proposition each) with
  citations (an evidence ID and a verbatim quote of at most 40 words), a leaning and
  confidence per issue, an overall summary citing `[E-###]` inline, and limitations. Every claim
  must cite at least one evidence item (prompt v2); a point with no support goes to the issue's
  unresolved questions or the limitations. The schema requires a citation per claim; if the
  output is still invalid after the gateway's retry, claims with none are dropped
  (`b1.uncited_claims_dropped`).
- **Record.** `issue_framed`, `evidence_registered`, `claim_created` (`C-001` ...,
  `author_agent` b1) and one `argument_created` per claim, whose citations carry a pinpoint
  derived from the chunk (`s. 74`, `para 12`, `paras 3-5`, `p. 7`). A citation to an ID that
  was not presented is recorded and fails the verifier's existence check; one whose ID or
  quote cannot be a `Citation` at all (a malformed ID, an empty quote) is dropped and counted.
- **Verification (S10).** `CitationVerifier` with `LLMEntailmentJudge` (the VERIFIER role,
  one pair per call). The `CaseAnalysis` lists as key evidence only the verified and weak
  citations of each position (§8 invariant 3); its limitations say, per issue, how many
  citations failed verification. The overall summary is left as written.
- **Repair** (`repair_quotes`, on in `b1.yaml`). One call lists each citation
  whose quote is not in its chunk and asks for a corrected verbatim quote, or null to drop it
  (IDEA_final §8: one repair attempt). A quote that is still wrong stays in and the verifier
  marks it invalid.

Config knobs (`baseline:` in the profile, `BaselineParams` in `config.py`): `queries`
(`llm` or `question`), `max_queries` (5), `evidence_k` (20), `rerank` (false, D-037),
`repair_quotes` (true in the profile; the schema default is false).

Recorded with the scores (`system` metric): `b1.queries`, `b1.evidence`, `b1.claims`,
`b1.citations`, `b1.invalid_citations`, `b1.share_verified`, `b1.share_weak`,
`b1.share_unsupported`, `b1.share_invalid` (shares of all citations),
`b1.malformed_citations_dropped`, `b1.uncited_claims_dropped`, `b1.ids_normalised` (IDs like `E-3` or `[E-003]` put in the `E-003` form), and with repair on `b1.citation_failures_before_repair`,
`b1.citation_failures_after_repair`, `b1.citations_dropped_by_repair`. The per-citation
outcomes and the queries are in the run's `scores.json` details.

Run: `uv run scripts/run_experiment.py --config b1 --split dev`.

### Dev runs (local model, Juris-Eval dev)

18 dev items, seed 0, `juris-qwen2.5-7b`, no grader. Every run completed and validated.

| Metric | v1 `b1-dev-nall-s1` | v2 `b1-v2-dev-nall-s1` | n (v2) |
|---|---|---|---|
| **citation validity** (bar: ≥ 0.95) | 0.120 (0.016-0.208) | **0.291** (0.152-0.433) | 10 |
| citation faithfulness | 0.042 | 0.140 (0.067-0.213) | 10 |
| unsupported-claim rate | 0.989 | 0.893 (0.820-0.960) | 18 |
| authority recall | 0.361 | 0.278 (0.111-0.463) | 18 |
| tokens per case | 12,568 | 20,454 | 18 |

- **v1** (answer prompt v1, no repair): 14 of 18 answers cited nothing, because the prompt
  allowed claims without citations. v2 makes every claim cite, as the spec requires (IDEA_final
  §6.1, §10), and turns on the one quote-repair call (§8). Both changes were made on dev.
- **v2, pooled:** 131 citations in 10 answers. Every evidence ID was one the model was shown.
  Only 34 quotes are exact substrings of their chunk. The verifier rated those 34 as
  15 verified, 6 weak and 13 unsupported.
- **The 97 failing quotes:** 39 are invented (under 40% of the quote appears in the chunk) and 30
  are partial. 16 are near-verbatim, 8 are real text from another evidence item, and 4 differ
  only in case. Repair fixed 6 of 103.
- **Empty answers:** 8 of 18 answers have no claims. In 6, every claim the model wrote lacked a
  citation and was dropped (62 claims in all). In 2, the model wrote no claims at all.
- **Reading:** the 7B model paraphrases instead of copying. That is a limit of the model,
  not of the pipeline. The bar is not met, and 6.3 stays open until it is re-measured with the
  intended model (D-044). The citation format is unchanged, so every config is measured the same
  way.

## B2: compute-matched single agent (PLAN 6.4)

`backend/juris/baselines/b2.py` (`B2System`, a subclass of `B1System`), prompts
`backend/juris/prompts/baseline_b2_critique.md` and `baseline_b2_revise.md`, profile
`configs/pipeline/b2.yaml`, tests `backend/tests/test_b2.py`. Decision: D-045.

B2 answers IDEA_final R1: it gets roughly Juris's token budget with one agent, so a Juris gain
over B2 is not just "more tokens". It is B1 with a longer S9:

- **Draft.** B1's answer, with the same queries, evidence and prompt.
- **Critique.** One call lists the draft's problems: unsupported claims, missing arguments,
  unframed issues, unjustified leanings. The quote errors are found in code with the verifier's
  exact-quote rule and handed to it, as Juris's agents get them from the Case Record.
- **Second search** (first round only). The first critique also writes up to
  `follow_up_queries` searches. Their results add up to `follow_up_k` new chunks, numbered after
  the first search's (`E-021` ...). The new queries are recorded as `query_issued` at S1.
- **Revise.** One call rewrites the whole answer from the draft, the critique and the evidence.
  If the output never validates, the last valid answer stands (`b2.revise_failures`).
- **Rounds.** Critique and revise repeat up to `max_rounds`. A round starts only while the
  estimated generation spend is under `target_tokens`. The spend counts prompt plus output
  characters / 4 over every BASELINE call. It is an estimate, not the gateway's usage, because a
  cache hit reports zero tokens and would make a cached re-run take more rounds. A critique with
  no problems (and no new evidence) ends the rounds.
- **Context.** Critique and revise see at most `revise_evidence_k` (12) chunks: the cited ones,
  then the new ones, then the rest by rank. With the draft in the prompt, all 28 would not fit
  the local model's 16k window.
- After the rounds, B1's quote repair, record, verifier and analysis run unchanged. Values are
  named `b2.*`: B1's set plus `b2.critiques`, `b2.revisions`, `b2.revise_failures`,
  `b2.follow_up_queries`, `b2.follow_up_evidence` and `b2.tokens_estimate`. The critiques and
  follow-up queries are in the run's details.

The budget (`target_tokens` 170,000, `max_rounds` 8) is a first estimate of `juris_full`, at
about 185k tokens per case:

| Stage | Estimate |
|---|---|
| Framer and researcher | 5k |
| Two counsel | 30k |
| Precedent analyst | 20k |
| Verifier gates | 30k |
| Rebuttal | 30k |
| Cross-examination and replies | 25k |
| Three jurors | 30k |
| Judge | 15k |

PLAN 7.12 measures `juris_full` and resets the budget to within ±20% of it.

Run: `uv run scripts/run_experiment.py --config b2 --split dev`.

### Dev run (local model, Juris-Eval dev, 2026-10-19)

Experiment `b2-dev-nall-s1`: 18 dev items, seed 0, `juris-qwen2.5-7b`, no grader. All 18 runs
completed, and none needed a revise fallback. Token usage, with B1's next to it:

| Metric | B1 (`b1-v2-dev-nall-s1`) | B2 (`b2-dev-nall-s1`) |
|---|---|---|
| **tokens per case** | **20,454** (17,661-23,067) | **158,971** (146,761-172,015) |
| tokens per case, range | | 123,902-206,986 |
| wall-clock per case | 193 s | 890 s |
| citation validity (bar ≥ 0.95) | 0.291 (0.152-0.433), n 10 | 0.263 (0.141-0.386), n 18 |
| citation faithfulness | 0.140 (0.067-0.213) | 0.148 (0.054-0.252) |
| unsupported-claim rate | 0.893 (0.820-0.960) | 0.866 (0.744-0.956) |
| authority recall | 0.278 (0.111-0.463) | 0.185 (0.056-0.343) |
| answers with no claims | 8 of 18 | 0 of 18 |
| claims / citations (pooled) | 126 / 131 | 84 / 90 |
| exact quotes (pooled) | 34 (26%) | 25 (28%) |

- **Budget.** B2 spends 7.8 times B1's tokens per case: 0.86 of the 185k estimate for
  `juris_full`, so within ±20% of it. Every case ran all 8 rounds: the critique never returned an
  empty problem list, and the estimate stayed under `target_tokens`. The estimate (149k per case
  on average) undercounts the measured total: it leaves out the verifier calls and the gateway's
  retries.
- **Quality on the 7B.** The extra compute bought no clear gain. Validity, faithfulness and the
  unsupported-claim rate are within each other's CIs. Revision removed the empty answers (B1's 8
  became 0), but it also cut the claims. Authority recall fell: revised answers cite fewer gold
  judgments.
- **Failing quotes** (65 of 90): 28 invented, 21 partial, 9 near-verbatim, 5 with the wrong ID,
  2 differing only in case. Repair fixed 8 of 73. Each critique was given the quote errors the answer still had,
  but the revisions still did not copy the text verbatim.
- **Near misses are often the corpus's noise.** In the smoke run, 4 of 7 quotes were the passage
  "cleaned up": an OCR typo in the chunk (`influene`), an em dash, `"` written as `'`. The
  exact-quote rule counts these as invalid. This is worth a look before the re-measure in 6.3
  and 6.5, but it does not change the rule (D-038).
- The 2-item smoke run (`b2-smoke-n2`) is not in `results.csv`: the machine slept during it,
  so its wall-clock is meaningless.
