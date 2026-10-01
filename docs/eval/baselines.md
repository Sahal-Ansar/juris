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
