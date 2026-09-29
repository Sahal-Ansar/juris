# Juris-Eval

The end-to-end benchmark for Juris (PLAN 5.1, 5.6, IDEA_final §11.1): Indian contract-law
fact patterns, each with the issues, authorities, statute sections and key points a good
analysis should reach.

**Status:** 60 items: 20 seed items drafted on 2026-09-28 (PLAN 5.1) and 40 added on
2026-09-30 (PLAN 5.6, D-040). 83 distinct authorities, 257 pinpoints, 65 sections.

**Reviewed: 0 of 60** (`reviewed_by: null` on every item). Update this count when items are
reviewed.

## Files

| File | What it is |
|---|---|
| `items.yaml` | The source of truth. Edit this. |
| `dev.jsonl` | Generated. 18 items for development and tuning. |
| `test.jsonl` | Generated. 42 items for **reporting only**: never tune prompts, retrieval or thresholds on them. |
| `schema.json` | Generated. JSON Schema of one line of `dev.jsonl` / `test.jsonl`. |
| `REVIEW.md` | Generated. Review sheet: every item with the corpus text of each pinpoint. |
| `search_queries.yaml` | Hand-written Issue-Framer-style search queries for the dev items, used by the retrieval evaluation (PLAN 5.3, D-037). They don't name the gold cases. Test items get theirs only when results are reported on them. |

## An item

| Field | Meaning |
|---|---|
| `id` | `JE-001` … |
| `topic` | The doctrine tested (distinct for every item). |
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
- The 40 items added in PLAN 5.6 cover doctrines the seed set left out: formation (acceptance
  by conduct, concluded contracts, certainty), privity, consideration and time-barred debts,
  illegality and jurisdiction clauses, frustration of leases and licences, discharge
  (substitution, accord and satisfaction, extension of time), damages and time as the essence,
  quasi-contract, bailment, pledge and lien, guarantee, agency, government contracts,
  insurance, and specific relief (determinable contracts, readiness and willingness,
  discretion, rectification, cancellation). Candidates were found by title and through the
  citation graph (judgments that discuss a section most), then read in the corpus like the
  seed set. Several items pair a leading case with a later or contrary one (e.g. Swastik Gases
  and A.B.C. Laminart, Sushila Devi and Raja Dhruv Dev Chand).
- The split is fixed: `backend/tests/test_juris_eval.py` fails if it changes. The seed set
  alternates by ID (odd IDs are dev); from JE-021 on, IDs with n % 5 == 1 are dev, so the
  whole set is 18 dev / 42 test (30% / 70%, PLAN 5.6).
- `search_queries.yaml` has hand queries for every dev item, so the retrieval evaluation can be
  re-run on the larger dev set.

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
run `build`, `check` and `review`, and update the review count above. Items flagged in `notes`
for the reviewer: JE-004 (a minor as mortgagee), JE-005 (Percept D'Mark is interlocutory),
JE-006 (possible State wagering laws), JE-012, JE-014, JE-035, JE-043 and JE-044 (the 2018
amendment to the Specific Relief Act), JE-023 (Trimex is a single-judge arbitration order),
JE-028 (Swastik Gases' two opinions share paragraph numbers), JE-032 (arbitration clauses now
also turn on s. 16 of the 1996 Act), JE-049 (Chatturbhuj against the later Article 299 cases),
and items whose law partly lies outside the corpus (Transfer of Property Act, Limitation Act,
Insurance Act, Registration Act: JE-025, JE-036, JE-042, JE-045, JE-046, JE-051, JE-057,
JE-059).
