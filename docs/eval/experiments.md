# Running experiments

PLAN 5.5, D-039. Code: `backend/juris/eval/{runner,compare}.py`, `scripts/{run_experiment,compare_experiments}.py`. Tests: `backend/tests/test_runner.py`.

## Run a config

```bash
uv run scripts/run_experiment.py --config b1 --split dev --limit 20 --seeds 1
```

- `--config` is a profile in `configs/pipeline/`. Its `kind` picks the system: `dummy` and `b0` (PLAN 6.2, `docs/eval/baselines.md`) work now; `b1`, `b2` and `juris` arrive with PLAN 6.3, 6.4 and 7.12.
- The model is the settings default (`.env`). Without a paid key it is the local Ollama model (D-041, `scripts/ollama_setup.ps1`).
- Items are the split's first `--limit` items by ID. Seeds are the profile's seed plus 0..N-1.
- `--grader PROVIDER:MODEL` turns on the LLM graders (key points, rubric, uncited sentences). Off by default. Pick a model other than the generator's, with the author's approval.
- `--experiment-id` defaults to `<config>-<split>-n<limit|all>-s<seeds>`. Running the same ID again resumes: finished runs are skipped, and `--retry-failed` redoes failed ones. An existing ID with a different spec is refused.
- `--concurrency` sets the runs in flight (default 2). Each run gets its own gateway (own cost ledger, shared response cache), limited by `llm_max_concurrency` inside.

## What is stored

Under `eval/results/` (git ignores the experiment folders; `results.csv` can be committed once it holds real results):

```text
<experiment_id>/experiment.json          spec: profile, kind, split, item IDs, seeds, grader
<experiment_id>/runs/<item>-s<seed>/
    manifest.json                        run manifest (config, git, models, prompt hashes, totals)
    events.jsonl                         every event of the run, in order
    caseview.json                        the folded run
    calls.json                           every LLM call (tokens, cost, cached)
    scores.json                          metric values and details (written last)
<experiment_id>/summary.json, summary.md per-item values (seeds averaged) and means with 95% CIs
results.csv                              one row per experiment (upserted), headline means
```

The runner opens each run (`case_created`, manifest) and closes it: `run_completed` with the call totals, or `run_failed` with the error if the system raised. A failed run is still scored (`answered` 0).

## Compare

```bash
uv run scripts/compare_experiments.py b0-dev-n10-s1 b1-dev-n10-s1 --paired
```

- One row per experiment: each headline metric's mean over items, with its 95% bootstrap CI.
- `--paired`: B against A on the items both have. Shows the mean difference per item, a bootstrap CI of it, and a two-sided sign-flip randomisation p-value (exact up to 16 items, else 20,000 seeded flips).

## The dummy config end to end

`configs/pipeline/dummy.yaml`, over the first 3 dev items. It builds an answer from each item's gold and makes one call to the offline `dummy` provider, so its scores only show that the plumbing works:

| Metric | Mean | 95% CI | n |
|---|---|---|---|
| answered | 1.000 | 1.000-1.000 | 3 |
| citation_validity | 1.000 | 1.000-1.000 | 3 |
| citation_faithfulness | 1.000 | 1.000-1.000 | 3 |
| unsupported_claim_rate | 0.557 | 0.500-0.600 | 3 |
| authority_recall | 0.778 | 0.667-1.000 | 3 |
| key_point_coverage | n/a | n/a | 0 |
| quality_rubric | n/a | n/a | 0 |
| cost_latency.total_tokens | 115.667 | 97.000-144.000 | 3 |

It cites evidence only for position I-1.A (so other positions are unsupported) and uses only supporting authorities (so contrary recall is 0). Running it again skipped all 3 runs. A 2-seed run whose seed-0 calls came from the cache reported the same tokens as the 1-seed run.
