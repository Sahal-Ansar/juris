# Citation verifier

PLAN 6.1, IDEA_final §6.1 (S4/S7/S10). Decision: D-043.

## What it does

`backend/juris/verify/` runs three checks per citation, cheapest first:

| Check | Code | Fails as |
|---|---|---|
| 1. Existence: the evidence ID is registered for the run | `check_citation` | `invalid` (`existence`) |
| 2. Exact quote: the quote is a normalised substring of the chunk (`verify/quote.py`: ASCII quote marks, collapsed whitespace, case kept unless `casefold=True`) | `check_citation` | `invalid` (`quote`) |
| 3. Entailment: an LLM judge labels the pair | `LLMEntailmentJudge` | see below |

- Entailment labels map to statuses: `supports` is `verified`, `partially_supports` is `weak`, and `does_not_support` and `contradicts` are `unsupported`. Each result carries the label, a one-sentence justification and the model name.
- Only citations that pass checks 1-2 reach the model. Checks 1-2 are code and are shared with the answer metrics (`eval/metrics/citations.py`), so both apply one rule.
- `CitationVerifier.verify(claims, citations, stage=...)` emits one `verification_result` per citation, in order. It then emits `claim_status_changed` for each claim whose status follows from its best citation: any verified makes the claim `verified`, else any weak makes it `weak`, else it is `unsupported`. Claims that later stages own (contested, survives, falls) are left alone.
- The evidence source is the run's `EvidenceRegistry`, or a folded run (`ViewEvidence`).
- Prompt: `backend/juris/prompts/verifier_entailment.md` (v1). The model is the `verifier` role's (settings).
- `LLMEntailmentJudge` also satisfies the metrics' `EntailmentJudge`, so `citation_faithfulness` can judge pairs a run never verified with the same code.

### Batching

The judge can send several pairs per call (`batch_size`), but the default is 1. In batches of four, the local 7B model mixed up pairs: VP-005's justification described VP-006's claim, and VP-018's repeated VP-019's. It also returned an empty list for a one-pair batch.

A single pair is therefore asked with a one-judgement schema. A bad batch falls back to one call per pair. Batching should be re-measured before it is raised for another model.

Tests: `backend/tests/test_verifier.py` covers checks 1-2 (unknown evidence, missing chunk, one changed word, typographic quotes and whitespace, case only with `casefold`), the label-to-status mapping, events and folding, claim statuses, batching, deduplication and fallbacks. They need no model or database.

## Evaluation

```bash
uv run scripts/eval_verifier.py run --set juris
```

```bash
uv run scripts/eval_verifier.py run --set coliee4 --n 100
```

```bash
uv run scripts/eval_verifier.py run --set coliee2 --n 100
```

```bash
uv run scripts/eval_verifier.py report
```

Results are written to `eval/results/verifier/` (not tracked). The metric functions are in `backend/juris/eval/verifier_eval.py` (tests: `backend/tests/test_verifier_eval.py`).

- **Juris pairs:** the 50 hand-labelled claim-passage pairs in `eval/verifier/juris_pairs.yaml`, built from Juris-Eval dev authorities (see `eval/verifier/README.md`).
  - Labels: 20 `supports`, 9 `partially_supports`, 9 `does_not_support`, 12 `contradicts`.
  - The headline is **status agreement**: gold and predicted labels are mapped to verified / weak / unsupported, since that status is what the pipeline acts on. It was fixed before the first run.
- **COLIEE** (not Indian law, a sanity check only):
  - Task 4 (statute entailment, English): 100 pairs, seeded and label-balanced. Claim = question, passage = articles.
  - Task 2 (case-law entailment): 100 paragraphs, 50 entailing and 50 not, with the negatives drawn from the same cases. Claim = entailed fragment, passage = paragraph.
  - Scored as yes/no. Strict: only `supports` counts as yes. Lenient: `partially_supports` counts too.

## Results (2026-10-02)

The model is `juris-qwen2.5-7b`: the local Qwen 2.5 7B through Ollama (D-041), with prompt v1, batch size 1, temperature 0 and seed 0.

### Juris pairs (n = 50)

| Metric | Value |
|---|---|
| **Status agreement** | **0.84** (95% CI 0.74-0.94) |
| Status kappa | 0.75 |
| Label accuracy (4-way) | 0.68 |
| Binary accuracy (supports vs rest) | 0.94 |
| `supports` precision / recall / F1 | 0.90 / 0.95 / 0.93 |

Confusion matrix (rows: gold, columns: predicted):

| | supports | partially | does_not | contradicts |
|---|---|---|---|---|
| supports | 19 | 0 | 1 | 0 |
| partially_supports | 0 | 7 | 2 | 0 |
| does_not_support | 2 | 1 | 6 | 0 |
| contradicts | 0 | 2 | 8 | 2 |

Status agreement by construction:

| Construction | gold | paraphrase | negated | dropped_condition | overstated | argument_only | wrong_passage | other |
|---|---|---|---|---|---|---|---|---|
| n | 13 | 7 | 9 | 5 | 4 | 6 | 3 | 3 |
| Agreement | 0.92 | 1.0 | 0.78 | 1.0 | 0.5 | 0.5 | 1.0 | 1.0 |

- The target (0.8 agreement) is met. The lower CI bound is below it, since n = 50.
- **The costly errors are false `verified`.** Two of these occurred, VP-026 and VP-037. In each, the quote reports a party's contention and the model took it as the court's holding. A quoted argument in a passage that does not decide it is the main way a citation could pass the gate wrongly.
- The model rarely says `contradicts` (2 of 12). It labels most contradictions `does_not_support`, which gives the same status, so the gate is unaffected. Two contradictions became `partially_supports` (VP-006, VP-033), which would wrongly mark them weak.
- **Batching:** the same prompt in batches of four scored 0.72 status agreement and 0.52 label accuracy, with 13 calls against 50.
- **Cost:** about 50 calls, 1-2 s each, $0.

### COLIEE (n = 100 each, balanced)

| Set | Strict accuracy | Precision | Recall | F1 | Lenient accuracy | Lenient F1 |
|---|---|---|---|---|---|---|
| Task 2 (case law) | 0.73 | 0.96 | 0.48 | 0.64 | 0.75 | 0.68 |
| Task 4 (statutes) | 0.60 | 0.63 | 0.50 | 0.56 | 0.59 | 0.57 |

- **Task 2:** the judge is conservative. When it says `supports` it is almost always right, but it misses about half the entailing paragraphs. For a gate this is the safer error: a missed support turns a claim `unsupported`, which the Judge must then say. It does not let a wrong citation through.
- **Task 4:** questions on the Japanese Civil Code need multi-step application of articles to facts. The 7B model is barely above chance there. This is outside what the verifier is asked to do: check one quoted passage against one claim.

## Limits

- The 50 labels were drafted and checked in this project, but no lawyer has reviewed them (`reviewed_by: null`). The 0.84 figure is provisional until a human review (10.4).
- The pairs are built from dev authorities, and the prompt was written before the first run and not tuned on them. The only change after the first run was the batch size.
- A 7B local model is a floor. The verifier should be re-measured with the intended model, or a cheaper verifier-role model if its agreement holds (IDEA_final §15), before 10.1.
