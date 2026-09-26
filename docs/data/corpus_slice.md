# MVP corpus slice: Indian contract law (PLAN 1.3)

The slice is fully specified in [`configs/corpus/mvp_contract.yaml`](../../configs/corpus/mvp_contract.yaml) (schema: `juris.ingest.slice.SliceConfig`). This page explains the rules and the size estimate. Selection itself happens in PLAN 3.1, using the same matchers.

## Rules

### Supreme Court (core of the corpus)

1. **Core:** every English SC judgment from 1950–2026 whose text, with the same Act mentioned at least twice, cites the Indian Contract Act 1872 or the Specific Relief Act (1963, or its 1877 predecessor), or refers to a specific section of either at least once.
   - Mentions: `\b(?:Indian\s+)?Contract\s+Act\b` and `\bSpecific\s+Relief\s+Act\b` (case-insensitive). "Contract Labour (Regulation and Abolition) Act" does not match.
   - Section references: "Section 73", "Sections 73 and 74", "S. 16(c)", "u/s 10", "Ss. 55, 56" immediately before the Act name ("… of the Contract Act").
2. **One hop:** SC judgments cited by at least 2 core judgments, resolved through SCR (`[1964] 1 S.C.R. 515`) or neutral (`2023 INSC 1043`) citations. Our metadata has no SCC or AIR mapping, so those citations can't be resolved yet. Ranked by the number of citing core judgments and capped at 1,000.
3. **Distractors:** a seeded random sample of non-matching SC judgments, 10% of the selected set (at least 200), so retrieval faces realistic noise.
4. **Size guard:** the total must fall in 2,000–5,000.
   - If the core set is too large: raise `min_mentions` to 3, then also require 2 section references.
   - If it is too small: add the Sale of Goods Act 1930 as a matching Act, then raise the one-hop cap to 2,000.
5. **Old judgments:** all years are kept (1950s–60s contract precedents matter). Judgments before about 1970 have no numbered paragraphs and noisy OCR, so they get page-level pinpoints and are flagged (see `docs/data/sources.md`).

### High Courts (a few hundred)

- Delhi (`7_26`) and Bombay (`27_1`) High Courts, 2018–2025 (after the Commercial Courts Act; modern formatting).
- Candidates are chosen by case type in the metadata title (`CS(COMM)`, `RFA(COMM)`, `FAO(COMM)`, `O.M.P.(COMM)`, `RFA`, `FA`, …). The Delhi/Bombay title formats haven't been sampled yet, so verify in 3.1.
- Keep those that pass the same core rule and are reasoned judgments (≥ 4 pages and ≥ 5 numbered paragraphs), which drops procedural orders. Cap at 400.

### Statutes (section-wise, PLAN 3.5)

| Act | No. | Notes |
|---|---|---|
| The Indian Contract Act, 1872 | 9 of 1872 | |
| The Specific Relief Act, 1963 | 47 of 1963 | Amended 2018 (Act 18 of 2018): track `in_force` dates for ss. 10, 14, 16, 20 |
| The Sale of Goods Act, 1930 | 3 of 1930 | Optional; also the size-guard fallback |

Source: `KanoonGPT/indian-legal-documents` (English principal Acts, deduplicated), used locally only (D-010). Spot-checked by hand against India Code.

## Size estimate

From `uv run scripts/estimate_slice.py`, using the PLAN 1.1 samples (55 SC PDFs: 50 from 2023, 5 from 1960) and per-year counts read from the AWS metadata:

| | Estimate |
|---|---|
| SC judgments 1950–2026 (population) | 43,547 |
| Core matches in sample | 3 / 55 = 5.5% (95% CI 1.9–14.9%) |
| **Core** | **≈ 2,375** (CI 815–6,468) |
| One hop | 1,000 (the cap binds: ~7 resolvable citations per judgment) |
| Distractors | 338 |
| **Total SC** | **≈ 3,713**, within 2,000–5,000 |
| High Courts | ≤ 400 |

**Uncertainty:**
- The sample is small and mostly from 2023, so the core estimate is rough.
- The size guard keeps the final total in range across the whole interval. At the low end (815), adding the Sale of Goods Act and a 2,000 one-hop cap gives ≈ 3,100; at the high end (6,468), stricter thresholds apply.
- PLAN 3.1 computes the exact counts.

**Measured along the way:** the sampled 2023 judgments cite a median of about 6.5 distinct SC authorities. About 40% of distinct citation strings are SCR/INSC (resolvable) and the rest SCC/AIR. Many cases are cited with SCC and SCR side by side, so more cases are resolvable than that 40% suggests.
