# Golden parse excerpts

Raw PDF text layers (a few pages each) of real judgments, used by `backend/tests/test_clean.py` and `backend/tests/test_segment.py` to guard the cleaning and segmentation rules against regressions (PLAN 3.2, 3.3).

Source: *Indian Supreme Court Judgments* and *Indian High Court Judgments* (Dattam Labs, Registry of Open Data on AWS), licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/); see `docs/DATA_NOTICE.md`. Each file records its `source_url` and the pages taken. The text is unmodified output of pdfium's text extraction.

| File | Why it's here |
|---|---|
| `SC-2023_12_979_1033` | Modern SCR: repeated running heads, split ligatures ("fi xed"), curly quotes |
| `SC-1963_3_22_183` | Old OCR text layer: damaged running heads, debris lines, line-break hyphens |
| `SC-S_1996_7_641_643` | Supplementary volume: margin letters on their own lines and at line ends |
| `SC-2015_13_1_1056` | Margin letters at line starts |
| `SC-S_2006_1_587_602` | Margin letters at line starts and ends, SCR heads |
| `HC-DLHC010792242017_1_2018-03-22` | Delhi HC: "RFA 421/2017 Page n of 60" header |
| `HC-HCBM020032982020_1_2024-04-10` | Bombay HC: file-name header, digital-signature block, "n/26" counter |
| `SC-1966_1_656_682` | Split-bench attribution ("The Judgment of Wanchoo and Shah, JJ. was delivered by"); body start in an unnumbered judgment (D-035) |
