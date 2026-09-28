# Dense retrieval latency (PLAN 4.2)

Made by `scripts/bench_dense.py` on 2026-09-28. Corpus: 124,996 chunks; BAAI/bge-m3 on cuda; ef_search 400; exact scan at or below 5,000 allowed chunks. Prewarm read 209,756 blocks in 0.2 s. k = 50; median of 3 runs per query after a warm-up pass. Target: < 300 ms.

| | p50 ms | p95 ms | max ms |
|---|---|---|---|
| encode | 33 | 39 | 42 |
| search | 43 | 67 | 93 |
| total | 77 | 99 | 123 |

HNSW recall@50 against an exact scan, 25 unfiltered queries: mean 0.978, min 0.86.

| Kind | Query | Path | Encode ms | Search ms | Total ms | Hits | First hit |
|---|---|---|---|---|---|---|---|
| term | undue influence | hnsw | 42 | 32 | 74 | 50 | SHIV KIRPAL SINGH v. SHRI V. V. GIRI — Supreme Court of India — 1970 — |
| term | frustration of contract | hnsw | 35 | 49 | 84 | 50 | RAJA DHRUV DEV CHAND v. HARMOHINDER SINGH & ANR. — Supreme Court of In |
| term | liquidated damages | hnsw | 33 | 39 | 72 | 50 | ARBP/891/2019 of MERCATOR OIL AND GAS LTD. AND ANOTHER Vs OIL AND NATU |
| term | readiness and willingness | hnsw | 37 | 39 | 77 | 50 | M/S J.P. BUILDERS & ANR. v. A. RAMADAS RAO & ANR. — Supreme Court of I |
| term | quantum meruit | hnsw | 31 | 37 | 67 | 50 | M/S. ALOPI PARSHAD & SONS, LTD. v. THE UNION OF INDIA — Supreme Court  |
| term | privity of contract | hnsw | 34 | 54 | 87 | 50 | COX AND KINGS LTD. v. SAP INDIA PVT. LTD. & ANR. — Supreme Court of In |
| term | restraint of trade | hnsw | 34 | 41 | 74 | 50 | SUPERINTENDENCE COMPANY OF INDIA (P) LTD. v. KRISHAN MURGAI — Supreme  |
| section | Section 74 | hnsw | 39 | 48 | 87 | 50 | M/S. KAILASH NATH ASSOCIATES v. DELHI DEVELOPMENT AUTHORITY & ANR. — S |
| section | s. 16(2) undue influence | hnsw | 37 | 46 | 82 | 50 | CENTRAL ORGANISATION FOR RAILWAY ELECTRIFICATION v. M/S ECI SPIC SMO M |
| section | Section 20 of the Specific Relief Act | hnsw | 32 | 38 | 70 | 50 | Specific Relief Act, 1963 — s. 19 — Relief against parties and persons |
| section | u/s 73 compensation for breach | hnsw | 31 | 30 | 60 | 50 | PHULCHAND EXPORTS LTD v. O.O.O. PATRIOT — Supreme Court of India — 201 |
| section | Section 56 impossibility of performance | hnsw | 32 | 38 | 70 | 50 | GOVINDBHAI GORDHANBHAI PATEL & ORS. v. GULAM ABBAS MULLA ALLIBHAI & OR |
| section | Section 19A | hnsw | 37 | 57 | 94 | 50 | M/S LOHIA MACHINES LIMITED AND ANR. v. UNION OF INDIA AND ORS. — Supre |
| case | Satyabrata Ghose v. Mugneeram Bangur | hnsw | 31 | 40 | 71 | 50 | SATYABRATA GROSE v. MUGNEERAM BANGUR & CO., AND ANOTHER. — Supreme Cou |
| case | Fateh Chand v. Balkishan Dass | hnsw | 34 | 44 | 78 | 50 | FATEH CHAND v. BALKLSHAN DAS — Supreme Court of India — 1963 — headnot |
| case | ONGC v. Saw Pipes | hnsw | 34 | 38 | 72 | 50 | OIL & NATURAL GAS CORPORATION LTD. v. SAW PIPES LTD. — Supreme Court o |
| case | Kailash Nath Associates | hnsw | 29 | 43 | 71 | 50 | M/S. KAILASH NATH ASSOCIATES v. DELHI DEVELOPMENT AUTHORITY & ANR. — S |
| case | Mohori Bibee | hnsw | 32 | 54 | 86 | 50 | MATHAI MATHAI v. JOSEPH MARY @ MARYKKUTIY JOSEPH & ORS. — Supreme Cour |
| question | Is a contract entered into under coercion by a threat to a third party voidable at the option of the coerced party? | hnsw | 30 | 41 | 71 | 50 | M/S N. N. GLOBAL MERCANTILE PRIVATE LIMITED v. M/S INDO UNIQUE FLAME L |
| question | Can earnest money be forfeited without proof of actual loss? | hnsw | 32 | 29 | 61 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| question | When is time of the essence in a contract for sale of immovable property? | hnsw | 39 | 38 | 77 | 50 | SRI BABU RAM @ DURGA PRASAD v. SRI INDRA PAL SINGH (DEAD) BY LRS. — Su |
| question | Does a minor's agreement bind the minor or can it be ratified on majority? | hnsw | 32 | 44 | 76 | 50 | MANIK CHAND AND ANR. v. RAMACHANDRA SON OF CHAWRIRAJ — Supreme Court o |
| common | contract | hnsw | 36 | 37 | 73 | 50 | FIRM OF PRATAPCHAND NOPAJI v. FIRM OF KOTRIKE VENKATTA SETTY & SONS ET |
| common | court | hnsw | 36 | 50 | 86 | 50 | NAHAR INDUSTRIAL ENTERPRISES LTD. v. HONG KONG & SHANGHAI BANKING CORP |
| common | agreement party | hnsw | 33 | 39 | 72 | 50 | ASF BUILDTECH PRIVATE LIMITED v. SHAPOORJI PALLONJI AND COMPANY PRIVAT |
| filtered | specific performance [{'court_levels': ['supreme_court']}] | hnsw | 32 | 67 | 99 | 50 | Specific Relief Act, 1963 — s. 10 — Specific performance in respect of |
| filtered | liquidated damages [{'date_from': '2000-01-01', 'date_to': '2020-12-31'}] | hnsw | 33 | 66 | 99 | 50 | ARBP/891/2019 of MERCATOR OIL AND GAS LTD. AND ANOTHER Vs OIL AND NATU |
| filtered | specific performance discretion [{'acts': ['SRA']}] | hnsw | 33 | 52 | 85 | 50 | JAI NARAIN PARASRAMPURIA (DEAD) AND ORS. v. PUSHPA DEVI SARAF AND ORS. |
| filtered | compensation for loss [{'doc_kinds': ['statute']}] | exact | 38 | 23 | 60 | 50 | Indian Contract Act, 1872 — s. 73 — Compensation for loss or damage ca |
| filtered | contract [{'court_levels': ['high_court'], 'acts': ['ICA']}] | hnsw | 30 | 60 | 90 | 50 | CARBP/135/2022 of OCEAN SPARKLE LIMITED Vs OIL AND NATURAL GAS CORPORA |
| filtered | penalty clause reasonable compensation [{'date_from': '2010-01-01', 'date_to': '2012-12-31'}] | hnsw | 30 | 92 | 122 | 50 | PHULCHAND EXPORTS LTD v. O.O.O. PATRIOT — Supreme Court of India — 201 |
| filtered | agreement void for uncertainty [{'date_to': '1959-12-31'}] | exact | 31 | 93 | 123 | 50 | Indian Contract Act, 1872 — s. 29 — Agreements void for uncertainty |
