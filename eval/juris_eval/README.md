# Juris-Eval

The end-to-end benchmark for Juris (PLAN 5.1, IDEA_final §11.1): Indian contract-law fact
patterns, each with the issues, authorities, statute sections and key points a good analysis
should reach.

**Status:** 20 seed items drafted on 2026-09-28, **not yet reviewed** (`reviewed_by: null`).
Grow to 60–100 items later (PLAN 5.6 / 10.4).

## Files

| File | What it is |
|---|---|
| `items.yaml` | The source of truth. Edit this. |
| `dev.jsonl` | Generated. 10 items for development and tuning. |
| `test.jsonl` | Generated. 10 items for **reporting only**: never tune prompts, retrieval or thresholds on them. |
| `schema.json` | Generated. JSON Schema of one line of `dev.jsonl` / `test.jsonl`. |
| `REVIEW.md` | Generated. Review sheet: every item with the corpus text of each pinpoint. |

## An item

| Field | Meaning |
|---|---|
| `id` | `JE-001` … |
| `topic` | The doctrine tested (one per item in the seed set). |
| `question`, `facts` | The question put to the system and the fact pattern. |
| `gold_issues` | The legal questions a good analysis frames. |
| `gold_supporting_authorities` | Judgments supporting the expected answer. |
| `gold_contrary_authorities` | Judgments the other side relies on, or that limit the rule; a good answer deals with them. |
| `gold_sections` | Statute sections, as `act_id:section` (e.g. `contract_act:74`). |
| `key_points` | What a good answer must cover. |
| `difficulty` | `easy`, `medium` or `hard`. |
| `notes` | Caveats, including points for the reviewer. |
| `reviewed_by` | Name and date of the human reviewer, or `null`. |

Each authority gives the corpus `doc_id`, a reporter `citation` of that judgment, its `title`,
the `pinpoints` (paragraph numbers in the corpus text) and the `proposition` it stands for.

Pinpoint labels follow the corpus segmentation (D-021):
- plain numbers are the Court's paragraph numbers;
- `p-N` is an unnumbered body paragraph;
- `h-N` is a headnote paragraph and `f-N` front matter (both are the reporter's editorial text).

Every authority except Percept D'Mark (an interlocutory decision cited through its headnote)
has at least one pinpoint in the Court's own text. Pinpoints were re-mapped by text after the
2026-09-28 segmentation fix (D-035), which moved some labels.

## How the items were made

- Principles were chosen from well-documented leading Supreme Court decisions on the topics
  in PLAN 5.1: coercion, undue influence, fraud, minors, restraint of trade, wagering,
  frustration, damages and specific performance. Formation, novation, guarantee, quasi-contract,
  public policy and mistake were added to reach 20.
- Every authority was found in the corpus by title and checked by reading the judgment text in
  the database. Pinpoints are paragraphs that were read. A scan flagged pinpoints quoting
  counsel or a lower court, and those were replaced with the Court's own paragraphs.
- Authorities that are not in the corpus (e.g. Mohori Bibee, a Privy Council decision) are not
  gold authorities.
- The split alternates by ID (odd IDs are dev, even IDs are test). Each broad area has items on
  both sides. It is fixed: `backend/tests/test_juris_eval.py` fails if it changes.

## Commands

```bash
uv run scripts/juris_eval.py build          # regenerate dev/test/schema from items.yaml
uv run scripts/juris_eval.py build --check  # fail if the generated files are stale
uv run scripts/juris_eval.py check          # every authority, citation, pinpoint and section resolves in the corpus DB
uv run scripts/juris_eval.py review         # regenerate REVIEW.md
```

## Reviewing

Read `REVIEW.md` item by item and check three things:
- the question and facts raise the stated issues;
- the key points are right and complete;
- each proposition is supported by its quoted paragraphs.

Edit `items.yaml` where needed and set `reviewed_by` (e.g. `"A. Reviewer, 2026-10-02"`). Then
run `build`, `check` and `review`. Items flagged in `notes` for the reviewer: JE-004 (a minor
as mortgagee), JE-005 (Percept D'Mark is interlocutory), JE-006 (possible State wagering laws),
JE-012 and JE-014 (the 2018 amendment to the Specific Relief Act).
