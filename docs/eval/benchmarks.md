# External benchmarks (PLAN 1.2)

Checked 2026-09-26. Decides which external benchmarks Juris uses, for what, and how to get them. The main end-to-end benchmark is our own Juris-Eval (PLAN 5.1); these cover components.

## Decisions

| Benchmark | Task | Access | Licence | Decision |
|---|---|---|---|---|
| **AILA 2019**: precedent retrieval (Task 1) | Fact scenario → relevant SC judgments | Open, Zenodo | CC-BY-4.0 | **Use for retrieval eval** (primary) |
| **AILA 2019**: statute retrieval (Task 2) | Fact scenario → relevant statutes | Open, Zenodo | CC-BY-4.0 | **Use for retrieval eval** (statute retriever) |
| **AILA 2020** Task 1 | Same pools as 2019 plus 10 new test queries | Encrypted archive; key via registration ⚑ | Not stated | **Optional**: only if the key is easy to get |
| **AILA 2020** Task 2 | Rhetorical role labelling | Same as above ⚑ | Not stated | **Skip** |
| **IL-PCR** (IL-TUR `pcr`) | Full SC judgment → prior SC cases it relies on | HF, manual gate ⚑ | CC-BY-NC-SA-4.0 | **Use for retrieval eval** (secondary, long-document queries) |
| **IL-TUR** other tasks (`lsi`, `rr`, `cjpe`, `lner`, `bail`, `summ`, `lmt`) | Various | HF, manual gate ⚑ | CC-BY-NC-SA-4.0 | **Skip for the MVP**; `lsi` revisited in PLAN 12.2 (criminal law) |
| **COLIEE** Task 2 (case-law entailment, English) | Which paragraph of a case entails a given decision | Signed memorandum ⚑ | Memorandum terms | **Use for verifier sanity check** |
| **COLIEE** Task 4 (statute entailment) | Yes/no: do these articles entail the query? | Signed memorandum ⚑; Japanese from 2026, English only as the 2025 archive | Memorandum terms | **Use for verifier sanity check** (English archive) |
| **COLIEE** Tasks 1, 3, pilots | Canadian case retrieval; Japanese statute QA; tort prediction | ⚑ | | **Skip**: not Indian law and not a verifier fit (IDEA_final R11) |

⚑ = needs the author to register or sign something (see "Actions for the author" below).

## AILA 2019 (FIRE 2019)

- **Access:** [zenodo.org/records/4063986](https://zenodo.org/records/4063986), open, **CC-BY-4.0**, one file `AILA_2019_dataset.zip` (20.7 MB).
- **Content:** 50 queries (descriptions of legal situations); 2,914 Supreme Court case documents for precedent retrieval; 197 statutes (title plus description) for statute retrieval; gold relevance judgments for both.
- **Format:** text files for queries, case documents and statutes, plus gold relevance annotations (the zip's README documents the exact format; the 2020 track scored with `trec_eval`: P, R, MAP, DCG, MRR).
- **Fit:** closest public match to the Researcher (S1): a situation description in, relevant authorities out.
- **How Juris uses it:** index AILA's own document pools with our retrieval stack (lexical, dense, hybrid, rerank) and report Recall@{10,25,50}, P@10, MRR and nDCG@10 (IDEA_final §11.2, RQ5). It tests retrieval **methods**, not our corpus. AILA doc IDs (`C1…`) don't map to CNRs, and that's fine.
- **Caveats:** only 50 queries (use cross-validation or report all 50 as a test set; no tuning on them without a split). The 197-statute pool is fixed and wasn't chosen for contract law (MVP domain), so the statute numbers measure the retriever, not our domain coverage. Check its composition in 5.2.

## AILA 2020 (FIRE 2020)

- **Access:** [track site](https://sites.google.com/view/aila-2020/dataset-evaluation-plan). Train and test archives on Dropbox are **GPG-encrypted**; the key comes from registering for the (2020) task, or by writing to `aila-fire@googlegroups.com`.
- **Content:** Task 1 uses the same 2,914 case docs and 197 statutes as 2019, with **10 additional test queries**. Task 2 is rhetorical role labelling (50 labelled judgments, 7 roles).
- **Decision:** optional. The 10 extra queries would bring AILA to 60, but aren't worth a long chase. Task 2 is skipped (no rhetorical-role component in the plan).

## IL-TUR / IL-PCR

- **Access:** [huggingface.co/datasets/Exploration-Lab/IL-TUR](https://huggingface.co/datasets/Exploration-Lab/IL-TUR), **manually gated**. The request form asks for name, affiliation, position, country and intended use, and requires agreeing to **research-only, non-commercial use and no re-sharing or uploading**. Licence **CC-BY-NC-SA-4.0**. Load with `load_dataset("Exploration-Lab/IL-TUR", "pcr", revision="script")`.
- **PCR content:** ~8k Indian Supreme Court documents (IDs are Indian Kanoon IDs). Splits: `train_queries` 827, `dev_queries` 118, `test_queries` 237, `train_candidates` 4.3k, `dev_candidates` 1k, `test_candidates` 1.7k. Each query has `relevant_candidates`; text is given as sentences.
- **Fit:** document-to-document retrieval (a whole judgment as the query), unlike Juris's short fact-scenario queries. It's a larger, harder second retrieval test, useful for comparing dense and lexical retrieval on long legal text.
- **How Juris uses it:** retrieval eval on the `test_*` splits with our stack; `dev` for tuning. Nothing from it is redistributed (NC-SA plus the no-sharing clause).
- **Other IL-TUR tasks:**
  - `lsi` (100 IPC sections from facts) matters for the criminal-law extension (12.2);
  - `rr` (rhetorical roles) could later help evaluate the Precedent Analyst's ratio/obiter split (7.6);
  - `cjpe` predicts outcomes, which Juris deliberately doesn't do.
  - All are skipped for the MVP.

## COLIEE

- **Access:** [coliee.org/data-request](https://coliee.org/data-request). The 2026 datasets include all previous years' train and test data. Each dataset needs its own **signed memorandum waiver**: name, group, email, position, department, address, and a signature. **Students need a supervisor's signature**; researchers who don't compete submit only the signed memoranda (no registration form), and an independent researcher signs them personally. Signing adds you to the COLIEE mailing list.
- **Tasks (2026):**
  - Task 1: Canadian case retrieval (English)
  - Task 2: case-law entailment, i.e. which paragraph entails a given decision (English)
  - Task 3: statute retrieval plus yes/no entailment (Japanese)
  - Task 4: statute entailment for yes/no questions given the relevant articles (Japanese)
  - Pilot tasks: tort prediction and rationale extraction (Japanese)
  - English translations for Tasks 3/4 are no longer provided, apart from the "Statute Law Dataset (English, 2025 archive only)".
- **Fit (IDEA_final R11):** not Indian law, so used only to sanity-check the entailment step of the Citation Verifier (6.1): does a quoted passage support a claim?
  - Task 2 is the closer fit (English case text, paragraph-level support).
  - Task 4's English archive is a cleaner yes/no entailment check.
- **Fallback if COLIEE access doesn't come through:** the hand-audited verifier subset that IDEA_final §11.2 already requires (claim–citation pairs from Juris runs, labelled by a human).

## Actions for the author

1. **IL-TUR (for IL-PCR):** request access on Hugging Face with your own account (the form above). Needed by PLAN 5.2/5.3.
2. **COLIEE:** sign the memorandum waivers for the *Case Law Dataset* (Task 2) and the *Statute Law Dataset (English, 2025 archive)* as an independent researcher. Juris is a personal project, so there is no supervisor. The supervisor signature applies only to students. If in doubt, email the organisers first. Needed by PLAN 6.1. It is optional: without it, the verifier is validated on the hand-checked subset (IDEA_final §11.2).
3. **AILA 2020 (optional):** register or email `aila-fire@googlegroups.com` for the decryption key.

AILA 2019 needs nothing: it's open and will be downloaded in PLAN 5.2.

## Loaders (PLAN 5.2)

Code: `backend/juris/eval/datasets/` (D-036). Retrieval benchmarks load as a `RetrievalDataset`
(queries, relevant IDs, the benchmark's own candidate pool); entailment benchmarks as an
`EntailmentDataset` (premise, hypothesis, yes/no). Data lives in `<data_dir>/raw/benchmarks/`
and is never committed. `uv run scripts/benchmarks.py summary` loads and validates everything
on disk.

| Benchmark | Loader | Status (2026-09-28) |
|---|---|---|
| AILA 2019 Task 1 (precedents) | `load_aila2019("precedents")` | **Loaded and valid**: 50 queries (median 529 words), 2,914 case documents, 195 relevant pairs (1-22 per query, median 3) |
| AILA 2019 Task 2 (statutes) | `load_aila2019("statutes")` | **Loaded and valid**: 50 queries, 197 statutes, 221 relevant pairs (2-5 per query) |
| IL-PCR (IL-TUR `pcr`) | `load_il_pcr(split)`, `fetch_il_pcr()` | **Waiting for access** (author action 1). Reads the dataset's parquet files at revision `d16219ad`; tested on a sample in the same layout |
| COLIEE Task 2 (case entailment) | `load_task2(files_dir, labels_json)` | **Waiting for the memorandum** (author action 2). Tested on a sample in the published layout |
| COLIEE Task 4 (statute entailment, English 2025) | `load_task4(xml_dir)` | **Waiting for the memorandum** (author action 2). Tested on a sample in the published XML format |
| AILA 2020 | — | Optional, not pursued (no key) |

### AILA 2019 findings

- Fetched from Zenodo record 4063986, MD5 `07f9621e385ff0d4540ce8dfd76b0c21` checked (`uv run scripts/benchmarks.py fetch aila2019`).
- The README says statutes are `S1`…`S197`. The released files are `S1`…`S200` with `S32`, `S58` and `S162` missing.
  - `S58` is judged relevant for Q2, Q33, Q49 and Q50, but no system can retrieve it.
  - The loader drops those four judgments and records them in the dataset's `notes`. Every query keeps at least one relevant statute.
- The statute pool is mostly constitutional, penal and procedural provisions. Only a handful of the 197 titles mention contracts, agreements or sale, so Task 2 measures the retriever, not contract-law coverage.
- The queries are long fact descriptions (about 530 words), much longer than an agent's search queries. This matters for tuning in 5.3.
- All 50 queries are one split. 5.3 must choose how to use them (e.g. cross-validation, or report-only) without tuning on reported queries.

### Adding IL-PCR and COLIEE when access arrives

- **IL-PCR:** request access at huggingface.co/datasets/Exploration-Lab/IL-TUR. Then `huggingface-cli login` (or set `HF_TOKEN`) and run `uv run scripts/benchmarks.py fetch il-pcr`. It downloads the six `pcr/*.parquet` files at the pinned revision.
- **COLIEE:** after the memoranda, unpack the data as follows, then run `uv run scripts/benchmarks.py summary`. The loaders follow the published task formats; confirm them on the real files, since they were written without access.
  - Task 2: case directories into `<data_dir>/raw/benchmarks/coliee/task2/files/`, and the labels JSON as `task2/labels.json`.
  - Task 4: XML files into `coliee/task4/`.
