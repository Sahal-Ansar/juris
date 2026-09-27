# Statute ingestion: method and India Code check (PLAN 3.5)

`backend/juris/ingest/statutes.py` splits each Act into sections, and `scripts/ingest_statutes.py` writes them to `<data_dir>/processed/statutes/<act_id>.jsonl`. The output stays out of the repository because the statute text isn't redistributable (see [DATA_NOTICE](../DATA_NOTICE.md)).

Each section record carries:

- `act`, `act_year`, `section`, `title`, `text` (sub-sections kept in), `char_start`/`char_end` (offsets into the 3.2 clean text), pages and `chapter`;
- `repealed`, `sub_sections`, `amendment_notes` (the footnotes on the section) and `amended_by` (the Acts named in them);
- `in_force_from`, `in_force_to` (null = in force, or repeal date unknown), `amendments_curated` and `curation_source`.

## Method

1. **Arrangement first.** The "arrangement of sections" gives the expected section numbers and titles in order. A body line starts a section only if it carries the next expected number (looking up to three entries ahead). Numbered clauses, illustrations and footnotes therefore can't pose as sections.
   - The line may carry an amendment marker (`1 [10.`), a stray dash (`— 3.`) or no space (`20.Agreement`).
   - Gaps in the arrangement are filled: the Contract Act's arrangement omits ss. 113–114, which the body has.
2. **Pages and footnotes.** A page ends at a bare page number. Everything from a bare `N.` line up to that page number is the page's footnotes.
   - Footnote markers in the body (`[1]`, `1 [`, `2* * *`) are resolved against the same page's footnotes into `amendment_notes`.
   - Markers are removed from `text`; the amendment brackets `[...]` are kept.
   - `amended_by` lists the Acts named after an amendment verb (`Subs. by Act 18 of 2018`).
3. **Boundaries.**
   - A section runs to the next section start.
   - Chapter and part headings go into `chapter`.
   - Trailing cross-headings, whether in capitals or run-in, are dropped, and the last section stops at the Schedule.
   - Repealed sections that the body shows only as `* * *` get a stub (`43. [Repealed.]`).
4. **Temporal validity (D-014).**
   - `in_force_from` defaults to the Act's commencement from `configs/statutes/acts.yaml`.
   - `configs/statutes/amendments.yaml` overrides it per section. There, `in_force_from` is the date the current text applies from, and `in_force_to` is the last day a repealed section was in force. Only sections with an entry get `amendments_curated: true`.
5. **Aliases.** `resolve_act()` maps names such as "ICA", "Contract Act" or "Indian Contract Act, 1872" to the canonical Act ID, using `acts.yaml`.

## Result

| Act | Sections | Repealed | Curated | India Code lists |
|---|---|---|---|---|
| Indian Contract Act, 1872 (`contract_act`) | 268 (ss. 1–266, 19A, 178A) | 76 (ss. 76–123, 239–266) | 76 | 268 |
| Specific Relief Act, 1963 (`specific_relief_act`) | 48 | 2 (ss. 43–44) | 17 | 48 |
| Sale of Goods Act, 1930 (`sale_of_goods_act`) | 67 | 1 (s. 65) | 0 | 67 |

Every section of the Indian Contract Act, 1872 is present: ss. 1–238, the repealed ss. 239–266 and the inserted ss. 19A and 178A.

## Spot-check against India Code (indiacode.gov.in, read manually on 2026-09-28)

Our extracted text was compared word for word with India Code's section pages. Case and punctuation were normalised and footnote markers ignored. The only differences are that our text starts with the section heading (India Code shows it apart) and the footnote markers. For ss. 73–74 the opening and closing words and every illustration label were checked.

| Section | Title | Result |
|---|---|---|
| ICA 1 | Short title; extent, commencement; saving | match (ratio 0.97, heading only) |
| ICA 5 | Revocation of proposals and acceptances (with the U.P. state amendment) | match (0.99) |
| ICA 10 | What agreements are contracts | match (0.96) |
| ICA 11 | Who are competent to contract | match (0.93) |
| ICA 16 | "Undue influence" defined | match (1.00) |
| ICA 19A | Power to set aside contract induced by undue influence | match (1.00) |
| ICA 73 | Compensation for loss or damage caused by breach of contract | match: start, end, illustrations (a)–(r) |
| ICA 74 | Compensation for breach of contract where penalty stipulated for | match: start, end, illustrations (a)–(g) |
| ICA 76 | ['Goods' defined] (repealed) | match (0.94) |
| ICA 124 | "Contract of indemnity" defined | match (0.97) |
| ICA 126 | "Contract of guarantee", "surety", "principal debtor" and "creditor" | match (0.94) |
| ICA 128 | Surety's liability | match (0.98) |
| ICA 178A | Pledge by person in possession under voidable contract | match (0.93) |
| ICA 238 | Effect, on agreement, of misrepresentation or fraud by agent | match (0.97) |

That is 14 sections (PLAN: at least 10). The section counts of all three Acts also match India Code (268, 48 and 67).

### Curated dates, checked on the same pages

- **Specific Relief (Amendment) Act, 2018 (18 of 2018), w.e.f. 1-10-2018:** ss. 6, 10, 11, 14, 14A, 15, 16, 19, 20, 20A, 20B, 20C, 21, 25 and 41. Each was read on India Code with its footnote (for example s. 10: "Subs. by Act 18 of 2018, s 3, for section 10 (w.e.f. 1-10-2018)").
- **SRA ss. 43–44:** repealed by the Repealing and Amending Act, 1974 (56 of 1974), w.e.f. 20-12-1974.
- **ICA ss. 76–123:** repealed by the Sale of Goods Act, 1930, s. 65, which came into force on 1 July 1930 (SoGA s. 1(3)).
- **ICA ss. 239–266:** repealed by the Indian Partnership Act, 1932, s. 73, which came into force on 1 October 1932 (Partnership Act s. 1(3)).
- **Commencements:** ICA 1-9-1872 (s. 1); SRA 1-3-1964 (S.O. 189 of 13-1-1964); SoGA 1-7-1930 (s. 1(3)).

## Known limitations

- **Old versions.** Only the current text of an amended section is available, so `in_force_from` marks when that text applies. The pre-2018 SRA wording isn't in the corpus.
- **Uncurated amendments.** Sections amended by other Acts (for example ICA s. 10 by Act 3 of 1951, the SRA 1964 amendments, the 2019 Jammu and Kashmir omissions) keep the Act's commencement date and `amendments_curated: false`. `amended_by` still names the amending Act from the footnote.
- **Unknown repeal date.** SoGA s. 65 ("Repeal", itself repealed by Act 1 of 1938) is `repealed: true` with no `in_force_to`; the repeal date isn't curated.
- **Note scope.** A footnote attaches to the section whose text carries its marker. A substitution covering several sections (SRA s. 14 → ss. 14, 14A) is noted on the first only; the curation file covers the rest.
- **State amendments** printed with a section (for example ICA s. 5, Uttar Pradesh) are part of its text, as on India Code.
