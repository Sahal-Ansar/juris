# Data sources (PLAN 1.1)

Reconnaissance of every source in IDEA_final §7.1, done on 2026-09-26 from small samples.
Samples are re-created with `uv run scripts/fetch_samples.py` into `data/raw/samples/` (not committed).

This is a research reading of published licence terms, not legal advice. Where terms are unclear, the rule is **don't redistribute**.

## Summary

| Source | What it really gives us | Licence (as published) | Store locally | Redistribute derived chunks |
|---|---|---|---|---|
| AWS: Indian Supreme Court Judgments | Judgment PDFs with a real text layer, plus good metadata | CC-BY-4.0 | Yes | Yes, with attribution (see `docs/DATA_NOTICE.md`) |
| AWS: Indian High Court Judgments | Judgment/order PDFs with text, plus thinner metadata | CC-BY-4.0 | Yes | Yes, with attribution |
| HF `KanoonGPT/indian-case-laws` | **Cleaned metadata only**: no judgment text, no cited-case lists | Apache-2.0 | Yes | Yes, with Apache-2.0 notice plus the CC-BY attribution of the AWS data it is built from |
| HF `KanoonGPT/indian-legal-documents` | Statute and gazette text (OCR), 35,845 documents | Apache-2.0 on the card; upstream provenance withheld | Yes (research use) | **Unclear, so don't redistribute** |
| India Code (indiacode.gov.in) | Authoritative bare Acts | Restrictive copyright policy | Only single copies for manual reference; no scraping | **No** (needs written permission) |

## 1. AWS Open Data: Indian Supreme Court Judgments

- **Where:** `s3://indian-supreme-court-judgments` (ap-south-1), public, no credentials. Maintained by Dattam Labs ([github.com/vanga/indian-supreme-court-judgments](https://github.com/vanga/indian-supreme-court-judgments)); scraped from the SC eCourts site (scr.sci.gov.in). Updated twice a month.
- **Size:** ~35k judgments, 1950–2025, ~52 GB. 2023 alone: 856 judgments, 854 English PDFs, 399 MB.
- **Layout:** `data/pdf/year=YYYY/{english,regional}/<path>_EN.pdf`, `data/tar/...` (bulk), `metadata/json/year=YYYY/<path>.json`, `metadata/parquet/year=YYYY/metadata.parquet`.
- **Sample:** 2023 metadata parquet, 50 English PDFs from 2023 and 5 from 1960 (with their JSONs).

| Field (metadata.parquet) | Example | Notes |
|---|---|---|
| `title` | `ELDECO HOUSING AND INDUSTRIES LIMITED versus ASHOK VIDYARTHI AND OTHERS` | |
| `petitioner`, `respondent` | | |
| `judge` / `author_judge` | `VIKRAM NATH` / `None` | `author_judge` often the string `"None"` |
| `citation` | `[2023] 16 S.C.R. 872` | SCR law-report citation |
| `case_id` / `nc_display` | `2023 INSC 1043` / `2023INSC1043` | Neutral citation |
| `cnr` | `ESCR010008242023` | eCourts case number, stable join key |
| `decision_date` | `30-11-2023` | **String, DD-MM-YYYY** |
| `disposal_nature` | `Appeal(s) allowed` | |
| `available_languages` | `ENG,PUN` | Regional-language versions exist |
| `path` | `2023_16_872_887` | Links metadata ↔ PDF (`<path>_EN.pdf`) ↔ JSON |
| `description`, `raw_html`, `scraped_at`, `year` | | `description` usually empty; `raw_html` is scraper residue |

- **Per-judgment JSON:** only `raw_html`, `path`, `citation_year`, `nc_display`, `scraped_at`, so nothing beyond the parquet.
- **Text quality:**
  - **2023:** all 50 PDFs have a text layer (median 16 pages, ~2,200 chars/page). They are the **SCR reprint** with a headnote at the top, SCR page numbers, and the margin letters `A`–`H` extracted as separate lines. pypdf splits some words (`chit fund s`).
  - **1960:** OCR text layer (~2,300 chars/page) with errors (`z96o`, `Ga;endrngadkar`) and marginal notes interleaved with the body.
- **Paragraph numbering:** 47/50 of the 2023 judgments have sequential `1.`, `2.`, … paragraphs, so paragraph pinpoints work. **0/5 of the 1960 judgments are numbered**; old judgments need page-level pinpoints.
- **Citations inside the text:** present as free text (SCR/SCC/AIR styles). No structured cited-case list anywhere; PLAN 3.9 must extract it.
- **Language:** English PDFs used; regional versions exist separately.

## 2. AWS Open Data: Indian High Court Judgments

- **Where:** `s3://indian-high-court-judgments` (ap-south-1), public. Same maintainer; scraped from judgments.ecourts.gov.in, some courts backfilled from the eCourts mobile API. Updated daily.
- **Size:** 25 High Courts (45 benches), ~17.8M judgments/orders, ~1.25 TiB. Metadata parquet for 2023 alone is 837 MB across 54 bench files.
- **Layout:** `.../year=YYYY/court=<code>/bench=<bench>/...` for PDFs, JSON, parquet and tar; court codes in `high_courts.csv` (`~` in the code becomes `_` in paths).
- **Sample:** Meghalaya HC 2023 metadata (1,476 rows) and 20 PDFs.

| Field | Example | Notes |
|---|---|---|
| `court_code` | `17~21` | |
| `title` | `MC(WPC)/77/2017 of Sebastian P. George Vs The Director General Assam Rifles` | Case number plus parties in one string |
| `description` | cause-list text | Noisy |
| `judge` | `HON'BLE MR. JUSTICE W. DIENGDOH` | Free text |
| `cnr` | `MLHC010003152017` | Join key; dedupe on `(cnr, decision_date, order_number)` per maintainers |
| `date_of_registration` | `18-05-2017` | String, DD-MM-YYYY |
| `decision_date` | `2023-09-15 00:00:00` | **Timestamp** (different type from SC) |
| `disposal_nature`, `court`, `pdf_link`, `raw_html` | | |
| `pdf_exists` | `False` | **Unreliable**: False for all 1,476 rows, yet the PDFs download fine |

- **No citation fields** (no neutral or law-report citation) in HC metadata.
- **Text quality:** all 20 PDFs have a text layer (median 3 pages, ~1,200 chars/page). Many are **short orders** (listings, adjournments, lists of connected matters), not reasoned judgments.
- **Paragraph numbering:** 9/20 have ≥5 sequential numbered paragraphs, i.e. mostly the real judgments.
- **Metadata quality varies by court** (per maintainers); mixed web/mobile sources can use different filenames for the same order.

## 3. Hugging Face `KanoonGPT/indian-case-laws`

- **Where:** [huggingface.co/datasets/KanoonGPT/indian-case-laws](https://huggingface.co/datasets/KanoonGPT/indian-case-laws); not gated. Parquet, `sample/v1/` (3,640 rows, 2.9 MB) and `structured/v1/year=YYYY/` (10.9 GB total; 2023 alone 1.1 GB).
- **Provenance:** built from the two AWS datasets above. Each row carries `source_json_s3_url` and `source_pdf_s3_url` pointing back to them.
- **Finding:** despite IDEA_final §7.1 ("indexable text", "fast text access"), **there is no judgment text**. `indexable_text` (median 423 chars) is a one-line metadata summary ("Case title: … Parties: … Court: …"). The citation fields are the case's own citations; **there is no list of cases it cites**.

| Field | Filled (sample) | Notes |
|---|---|---|
| `case_title`, `party_petitioner`, `party_respondent`, `party_caption` | ~100% | Cleaner than raw AWS |
| `docket_number`, `cnr_number`, `court_name`, `court_code`, `bench_name` | ~100% | |
| `presiding_judge`, `coram_members` (list), `coram_members_text` | 92–95% | Parsed bench lists: useful for bench strength |
| `decision_date`, `registration_date` | 100% / 96% | **ISO `YYYY-MM-DD`** |
| `neutral_citation`, `law_report_citation` | 139/3,640 | SC rows only (e.g. `2020INSC484`, `[2020] 7 S.C.R. 941`) |
| `headnote_text` | 140/3,640 | SC rows only: SCR headnote, handy for summaries |
| `disposition_text` | 95% | |
| `language_codes` | 3% | |
| `indexable_text` | 100% | Metadata summary, not the judgment |
| `normalized_record_json`, `parser_json`, `quality_json` | 100% | Parser warnings and quality flags (e.g. `has_decision_before_registration`) |
| `source_*` | 100% | Provenance back to AWS S3 |

- **Use for Juris:** a cleaner metadata join layer for the AWS data (parties, bench, dates, quality flags, SC headnotes). Not a text source.

## 4. Hugging Face `KanoonGPT/indian-legal-documents`

- **Where:** [huggingface.co/datasets/KanoonGPT/indian-legal-documents](https://huggingface.co/datasets/KanoonGPT/indian-legal-documents); not gated. 35,845 rows in 8 parquet shards (0.47 GB).
- **Sample:** the viewer's first 100 rows, plus the title/type/jurisdiction columns of all rows via HTTP range reads (1.6 MB).

| Field | Example | Notes |
|---|---|---|
| `doc_id` | sha256 hex | |
| `document_title` | `THE SPECIFIC RELIEF ACT, 1963` | Case and format vary; duplicates and Hindi editions exist |
| `document_type` | `Act` | Rules 11,121, Notification 9,103, Act 8,342, Legislation 1,756, … (not normalised: `Act`, `Legislation (Act)`, `Bare Act`) |
| `document_jurisdiction` | `Central` | Central 14,943; the rest per state |
| `issuing_authority` | `Legislative Assembly of Goa` | |
| `issue_date` | `1963-12-13` | ISO, often `Unknown` |
| `short_description` | | Generated summary |
| `text` | median 6.6k chars, max 76k | OCR/PDF-derived markdown from gazettes; boilerplate and `==> picture … omitted <==` markers |

- **Coverage check:** principal central Acts are present, e.g. Indian Contract Act 1872, CPC 1908, Specific Relief Act 1963, Transfer of Property Act 1882, Limitation Act 1963, Indian Evidence Act 1872, Sale of Goods Act 1930, Consumer Protection Act 2019, BNS 2023, BSA 2023. Each often appears 2–6 times (duplicates, Hindi editions, amendment Acts).
- **No section structure and no version/amendment history.** Section splitting (PLAN 3.5) and deciding which text is current are our job.
- **Provenance:** the card says source URLs and scraping provenance are "intentionally not included". We can't tell whether a text came from the Gazette, India Code, or elsewhere.

## 5. India Code (indiacode.gov.in)

- The old domain `indiacode.nic.in` now redirects to `indiacode.gov.in`.
- **Copyright Policy** (last updated 2026-06-22), in short:
  - all Portal content is the property of the Legislative Department, Ministry of Law and Justice;
  - without written permission it prohibits reproduction and distribution, derivative works, and "systematic extraction, scraping, or harvesting";
  - it permits limited personal, non-commercial use for individual research and study, educational use, and viewing or downloading single copies for reference;
  - required attribution: "IndiaCode Portal, Legislative Department, Ministry of Law and Justice, Government of India."
- **Use for Juris:** manual spot checks of individual sections against our statute text. **No scraping, no bulk download, nothing redistributed.**

## Consequences for later steps

- **Case text comes from the AWS PDFs** (3.1/3.2). KanoonGPT case laws is a metadata layer only.
- **Citation graph (3.9)** needs our own extraction from judgment text: no source provides cited-case lists.
- **Paragraph pinpoints (3.3):** modern SC judgments are numbered; pre-~1970 SC judgments are not, so fall back to page pinpoints. PDF cleanup must drop SCR margin letters (`A`–`H`), page headers and repeated SCR page numbers.
- **Dates:** normalise three formats (DD-MM-YYYY strings, timestamps, ISO).
- **HC data:** don't trust `pdf_exists`; filter out short procedural orders before indexing.
- **Statutes (3.5):** dedupe titles, prefer English principal Acts, strip gazette boilerplate, split into sections, and track which version we hold.
