# Precedent Analyst: cards and fits (PLAN 7.6)

Code: `backend/juris/agents/precedent_analyst.py`, `backend/juris/pipeline/stages/precedent.py`,
prompt `backend/juris/prompts/precedent_analyst.md` v1. Decision: D-053.

## What S3 does

For each judgment cited in S2, in order of first citation:

- **Code facts.** Court, level and its weight, bench, date and citations come from the
  document. The treatment cues come from the citation graph: later judgments citing it with
  "followed", "overruled", ... as hints. `treatment_verified` is always False (D-013).
- **What the model reads:** the dispute, the issues the citing arguments are about, those
  facts and cues, the passages (the cited chunks first, then up to 3 headnote chunks, then the
  cited chunks' neighbours, within 12,000 characters), and every citing argument with its
  claim and the quotes taken from this judgment. The prompt carries IDEA_final §9's rules of
  weight.
- **What the model gives:** the ratio, the material facts, the distinguishing factors, and a
  fit label with a reason for each citing argument.
- **Checks:** the principle and material facts are required; a fit must name a citing
  argument, once. With nothing usable left, a code-only fallback card is written, so every
  cited judgment has one.

## Result (Juris-Eval dev, 18 items, local `juris-qwen2.5-7b`, 2026-10-20)

`uv run scripts/eval_precedent.py run` (S0-S3; S0-S2 come from the cache of the counsel run).

| | |
|---|---|
| cited judgments | 34 |
| **cards** | **34 (100%), none a fallback** |
| fits | 46 of 46 citing arguments |
| fit labels | directly_supports 27, supports_by_analogy 11, does_not_support 5, contrary 3 |
| treatment cues on the cards | 35 |

## Spot check: 10 fits

`uv run scripts/eval_precedent.py sample --n 10 --seed 0`, each fit read against the claim, the
quote and what the judgment decides.

| Item, argument | Judgment | Fit given | Check |
|---|---|---|---|
| JE-026 A-002 | Satyabrata Ghose | directly_supports | ✗ The quote is counsel's argument as the court restates it ("it is said"). The court held the contract was *not* frustrated, and the claim's "void ab initio" misstates the doctrine. |
| JE-026 A-005 | Naihati Jute Mills | directly_supports | ✓ for the contract-terms point. The principle restates facts, not the ratio (an absolute duty to furnish the licence). |
| JE-003 A-002 | Shanti Budhiya Vesta Patel | directly_supports | ✓ |
| JE-019 A-002 | DTC v. Mazdoor Congress | does_not_support | ½ The judgment struck the clause down, so `contrary` is the right label; the model's own reason says "contrary". |
| JE-036 A-001 | Man Kaur | supports_by_analogy | ✗ The passage is about specific performance and damages, not a seller's right to rescind. |
| JE-031 A-005 | Suri Prithvi Cotton Mills | directly_supports | ✓ (the claim overstates it slightly) |
| JE-026 A-004 | Naihati Jute Mills | directly_supports | ✓ |
| JE-021 A-003 | Bhagwati Prasad Pawan Kumar | directly_supports | ✓ |
| JE-031 A-004 | State of Karnataka v. State of Meghalaya | supports_by_analogy | ✓ The fit is right, but the principle line contradicts itself. |
| JE-021 A-006 | Bhagwati Prasad Pawan Kumar | directly_supports | ✓ |

**7 of 10 are reasonable.** The failure modes:

- **Over-credit.** 59% of all fits are `directly_supports`.
- **The argument taken for the holding.** A view the court states only to reject is credited
  as the holding (the same mode the verifier missed in D-043).
- **`does_not_support` given where `contrary` is meant.**

Principles often restate the quoted passage or the facts rather than the ratio, even when a
headnote was shown. S5 and S6 (rebuttal and cross-examination) exist to challenge exactly
such reliance, and the Judge weighs cards together with verification results. Re-check with
the intended model before 10.1.
