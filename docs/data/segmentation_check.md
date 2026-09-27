# Paragraph segmentation: method and hand check (PLAN 3.3)

`backend/juris/ingest/segment.py` splits each cleaned judgment into `Paragraph(no, text,
char_start, char_end, section, numbering, page_start, page_end, part, markers, contested)`.
Offsets index the clean text from PLAN 3.2; the parse record's offset runs map them back to the
raw PDF text. Corpus-wide numbers: [segmentation_report.md](segmentation_report.md).

## Method

1. **Candidates.** Every line that opens with a paragraph number: `12.`, `12)`, `(12)`, ranges
   (`38-39.`), a bare `12.` line, `10.In`, and the same after a one-character SCR margin token
   (`A 36.`, `8 30.`, `. 24.`). A judge's line that carries paragraph 1 (`S.B. SINHA, J. 1. Leave
   granted.`) is split so that paragraph 1 is a candidate too.
2. **Best chain.** Dynamic programming over candidates in document order picks the chain whose
   numbers rise by one. It scores +1 per member, −0.5 per skipped number (up to 3), +0.25 for
   keeping the predecessor's punctuation, and −1 for punctuation that differs from the chain's
   first member (one OCR misread is tolerated; a run of quoted `N.` paragraphs inside an `N)`
   judgment can't win). There are further penalties for list-like members: −0.3 within two lines
   of the predecessor, −0.6 for quoted issues and clauses (`3. Whether …`, `10. That …`, elided
   `23. XXX`), and −0.4 after a lead-in colon (`… as under:-`). A chain starts at 1–6 (−0.5 per
   number above 1).
3. **Validity.** A chain needs at least 3 members, and bracket chains at least 20 (shorter
   `(1) (2)` runs are footnotes, charges or tables). Chains under 10 members must spread over at
   least half of their region. Candidates inside a reporter's headnote (`HELD: 1. …` up to
   `CIVIL APPELLATE JURISDICTION`) are excluded.
4. **Opinions.** After the main chain, the text left and right of it is searched again. A further
   chain counts as a separate opinion (`part` 2, 3, …) only when an opinion header (`X, J.`,
   `X, J. (concurring)`, `JUDGMENT`, `ORDER`) appears within 10 lines before it. Numbered
   directions at the end of a judgment are therefore not an opinion.
5. **Flags.**
   - Numbers skipped inside a chain go to `missing_numbers`.
   - A paragraph is `contested` when another line within two paragraphs carries the same number
     (a quoted paragraph, issue or clause), so the choice may be wrong.
6. **Fallback.** With no valid chain (most SC judgments before about 2000), body paragraphs are
   inferred from layout: a short line that ends a sentence closes a paragraph. They are numbered
   `p-1`, `p-2` … with `numbering="inferred"` and never passed off as the court's own.
   - Front matter before the body becomes `f-N`.
   - A detectable SCR headnote becomes `h-N`, with `section="headnote"`.
   - Sub-markers (`(iii)`, `(a)`) at line starts are recorded with their offsets.

## Hand-checked sample

There are 30 judgments. They are drawn with `random.Random(33)` from the parsed snapshot
`mvp_contract-1f53c208a8`, stratified by glob: `HC-*` 6, `SC-201*` 6, `SC-202*` 5, `SC-200*`
5, `SC-199*` 3, `SC-198*` 2, `SC-197*` 1, `SC-196*` 1 and `SC-195*` 1. Each numbered paragraph
was checked against the judgment text: is the chosen line the court's own paragraph N, or a
quoted or listed item with the same number? The check also looked for real paragraph numbers
present in the text that the chain missed.

| # | Judgment | Numbering | Paras | Wrong | Missed | Notes |
|---|---|---|---|---|---|---|
| 1 | HC-DLHC011567682016_1_2018-08-16 | explicit | 9 | 0 | 0 | |
| 2 | HC-DLHC010192042020_1_2023-12-20 | explicit | 41 | 1 | 0 | 15 taken from a quoted judgment's run 14–16 (flagged) |
| 3 | HC-DLHC010057542019_1_2021-06-01 | explicit | 73 | 0 | 0 | |
| 4 | HC-DLHC010187952020_1_2023-12-20 | explicit | 41 | 1 | 0 | companion suit, same judgment as #2 |
| 5 | HC-DLHC013386372017_1_2018-02-27 | explicit | 19 | 4 | 0 | 1–2 from a quoted earlier order, 8–9 from a quoted SC judgment (all flagged) |
| 6 | HC-DLHC010119632016_1_2024-08-21 | explicit | 36 | 0 | 0 | para 1 unnumbered in the source |
| 7 | SC-2010_12_772_798 | explicit | 30 | 0 | 0 | |
| 8 | SC-2016_4_763_798 | explicit | 44 | 0 | 1 | para 1 printed `I.` (OCR); 35 and 38 absent from the text |
| 9 | SC-2015_7_631_688 | explicit | 31 | 0 | 0 | `N)` numbering; quoted `N.` paragraphs skipped |
| 10 | SC-2012_8_100_117 | explicit | 31 | 0 | 0 | 1 on the judge's line; `. 24.` margin token |
| 11 | SC-2014_12_360_369 | explicit | 10 | 0 | 0 | |
| 12 | SC-2010_3_1121_1127 | explicit | 8 | 0 | 0 | |
| 13 | SC-2023_12_421_432 | explicit | 15 | 0 | 0 | |
| 14 | SC-2024_10_708_735 | explicit | 51 | 0 | 0 | issue list skipped |
| 15 | SC-2023_8_379_451 | explicit | 40 | 6 | 0 | 1–5 are the award's operative items (real 2–3 lie earlier; 2 flagged); 12 from a quoted judgment (flagged) |
| 16 | SC-2023_4_140_187 | explicit | 31 | 0 | 0 | elided `23. XXX` quote skipped |
| 17 | SC-2022_3_1_4 | explicit | 7 | 0 | 0 | |
| 18 | SC-2009_1_735_747 | explicit | 26 | 0 | 0 | 4 absent from the text |
| 19 | SC-2009_4_1197_1212 | explicit | 21 | 1 | 0 | OCR prints the real 15 as a second `14)`; the later one was taken (flagged) |
| 20–30 | SC-2002_2_31_36, SC-2003_2_1068_1084, SC-2002_2_37_48, SC-1995_2_260_275, SC-1998_1_342_357, SC-1990_1_884_908, SC-1986_3_866_904, SC-1984_1_184_210, SC-1975_3_1_20, SC-1964_1_752_765, SC-1952_1_179_193 | inferred | — | — | — | the source has no paragraph numbers; flagged `inferred` |

**Result.**

- **Explicit judgments:** 564 numbered paragraphs were assigned and 551 are correct. 13 are
  wrong: 9 of them are flagged `contested` and 4 are silent (all in #15, whose text layer puts
  the award's items before the judgment's opening). 1 real number was missed.
- **Paragraph numbers matching the source:** 551 / 565 = **97.5%** (target ≥ 90%).
- **Correct or flagged:** 560 / 565 = 99.1%. Of the correct paragraphs, 52 are flagged
  `contested` as a precaution.
- **Inferred judgments:** all 11 are correctly flagged `inferred`. None has numbering that was
  missed.

## Known limitations

- **Quoted paragraphs in the same style.** A cited judgment quoted with its own `14. 15. 16.`
  between the court's 14 and 16, or an earlier order reproduced before paragraph 1, can take the
  slot. Most such cases are flagged `contested`; pinpoint citations to a contested paragraph
  should be checked against its text.
- **Scrambled text layers.** In some SCR PDFs (e.g. SC-2023_8_379_451) the text layer puts quoted
  items before the opening paragraphs. Nothing flags these silent errors.
- **Lost or misread numbers.** Numbers missing from the text layer become `missing_numbers` and
  their text joins the previous paragraph. A number OCR'd as a letter (`I.` for 1) is missed. A
  chain whose numbers 1–5 are all lost can start at up to 6.
- **Opinion headers.** An opinion whose header runs straight into text (`SHARMA, CJ. We have
  had the benefit …`) is not split off as a separate part.
- **Old reports.** Almost all SC judgments before about 2000 are unnumbered, and their inferred
  paragraphs follow layout (short sentence-final lines), so they are approximate. Headnote
  holdings are excluded from chains only when the headnote's end marker is found.
- **Sub-markers.** `(iii)`, `(a)` are recorded only at line starts.
