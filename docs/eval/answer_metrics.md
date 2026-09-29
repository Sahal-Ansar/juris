# Answer metrics

PLAN 5.4, D-038. Code: `backend/juris/eval/metrics/`. Tests: `backend/tests/test_metrics.py`.

The suite scores one config's answer to one Juris-Eval item. The answer is the folded run (`CaseView`): the `CaseAnalysis` plus the record behind it (evidence, chunks, claims, arguments with their citations, verification results). Baselines produce the same view, so every config is scored the same way.

```python
from juris.eval.metrics import Answer, Grader, score_answers

result = await score_answers([Answer(item, view)], grader=Grader(gateway, grader_model))
result.items[0].values   # per item: {"citation_validity": 0.933, ...}
result.table()           # Markdown: mean, 95% bootstrap CI, n
```

## Metrics

| Metric | Value | Sub-values |
|---|---|---|
| `answered` | 1 if the run produced an analysis, else 0 | |
| `citation_validity` | Share of the record's citations whose evidence ID is registered and whose quote is an exact normalised substring of the chunk | `.existence` (ID registered); `.analysis_refs` (evidence IDs the analysis cites are registered) |
| `citation_faithfulness` | Share of claim-citation pairs labelled `supports`. Invalid citations count as not supporting | `.lenient` (also `partially_supports`) |
| `unsupported_claim_rate` | Share of the final output's statements without verified support | `.uncited` (share with no citation); `.record` (share of the record's claims with no `verified`/`weak` support) |
| `authority_recall` | Share of gold judgments (supporting and contrary) the analysis relies on | `.supporting`, `.contrary`, `.sections` |
| `key_point_coverage` | LLM grader: mean credit over the key points (covered 1, partially 0.5, missing 0) | `.full` (share fully covered) |
| `quality_rubric` | LLM grader: mean of four 1-5 scores | `.correctness`, `.completeness`, `.reasoning_consistency`, `.uncertainty_calibration` |
| `cost_latency` | (no headline value) | `.llm_calls`, `.cached_calls`, `.input_tokens`, `.output_tokens`, `.total_tokens`, `.usd`, `.wall_seconds` |

Details behind each number (which citations are invalid, which statements are unsupported, which gold authorities are missing, the graders' reasons) are in `ItemScores.to_json()`.

### Definitions

- **Quote rule** (`juris/verify/quote.py`, shared with the verifier in 6.1): typographic quote marks become ASCII and whitespace runs become one space, in both the quote and the chunk. Case is kept (optional casefold). Nothing else is forgiven.
- **Faithfulness labels** come from the run's own latest verification of each (claim, evidence) pair, since every config uses the same verifier (IDEA_final §5). Pairs the run never verified go to an `EntailmentJudge` (6.1 plugs its entailment check in). Without one they are left out as "unjudged" and counted in the details.
- **Statements of the final output**: each position summary (cited through its key evidence and any inline `E-###`), and each sentence of the overall summary (cited inline). A statement is supported when at least one evidence item it cites has a `verified` or `weak` verification anywhere in the record. An uncited position summary is unsupported. An uncited summary sentence counts only if the assertion grader says it makes a legal assertion (IDEA_final §10). Without the grader, every uncited sentence counts.
- **What the analysis relies on**: positions' supporting authorities, both sides of key conflicts, sources, and the documents of the evidence it cites. Sections are the statute sections (`contract_act:15`) whose chunks it cites as evidence.
- **Cost and latency** (changed in D-039): from the run's LLM calls (`CallRecord`s or `llm_calls` rows) when given, else from the `run_completed` totals (live calls only). With calls, tokens count every call, cache hits included, so a warm cache doesn't make a config look cheaper. Dollars are the list price of every call when prices are given (the runner passes them), else the live spend. `.cached_calls` counts the hits. Wall-clock is the first-to-last event span unless given, and cache hits shorten it.

### Graders

- Three versioned prompts in `backend/juris/prompts/`: `grader_key_points`, `grader_quality_rubric` (fixed 1/3/5 descriptors per criterion, gold as a reference, not the only right answer) and `grader_assertions`. One call each per answered item, concurrently, through the gateway (cached and costed), tagged `agent=<prompt>`, `prompt=<id>@v<version>`, `item=<id>`.
- The grader model is a parameter (`Grader(gateway, model)`). IDEA_final §11.3 asks for a model other than the generator's; choosing it belongs to the experiment (5.5), with the author's approval.
- Grades are structured output. Key points and sentences must come back numbered 1..n, or `GraderOutputError` is raised.

### Missing values

A value is `None` where it does not apply: no citations, no contrary gold, no grader. Aggregates leave `None` out and report `n`. A run with no analysis scores 0 on key-point coverage and authority recall, `None` on the rubric, and `answered` = 0, so failures count against a config without inventing rubric scores.

## On the mock run (2.4)

`fixtures/runs/coercion_full` scored against the mock gold `backend/tests/golden/metrics/coercion_gold.yaml`. The mock gold names one supporting authority, one contrary authority and one section the analysis leaves out. The grader answers are canned (FakeProvider; no paid API, D-007): 3 key points covered, 1 partially, 1 covered; rubric 4/4/5/4; one of the three uncited summary sentences is a legal assertion.

| Metric | Value | Why |
|---|---|---|
| citation_validity | 0.933 | 14/15: C-008's quote is not in E-018 |
| citation_faithfulness | 0.667 | 10/15 `supports`; `.lenient` 0.867 (+3 partial) |
| unsupported_claim_rate | 0.143 | 1/7: "The outcome turns on whether the Rs 48 lakh was admitted..." has no citation; 0.333 (3/9) without the assertion grader |
| unsupported_claim_rate.record | 0.200 | C-002 (unsupported) and C-008 (invalid quote) |
| authority_recall | 0.667 | 4/6; supporting 3/4, contrary 1/2, sections 2/3 |
| key_point_coverage | 0.900 | canned grades |
| quality_rubric | 4.25 | canned grades |
| cost_latency | 19 calls, 442,800 tokens, $2.36, 495 s | from the run's totals |

`fixtures/runs/failure_case` (no analysis): `answered` 0, key-point coverage 0, authority recall 0, rubric `None`, and no grader calls.

## Not covered here

- Counterargument coverage (IDEA_final §11.2) is only partly measured, by `authority_recall.contrary`.
- B0's free-text citations need 6.2's resolver before citation validity means "hallucinated authorities".
- The graders have not been run against a live model or checked against human grades (10.4).
