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
