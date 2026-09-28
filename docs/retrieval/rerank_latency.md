# Reranking latency (PLAN 4.4)

Made by `scripts/bench_rerank.py` on 2026-09-28. Hybrid search (lexical + bge-m3, RRF, depth 100) with and without BAAI/bge-reranker-v2-m3@953dc6f6 on cuda, reranking the fused top 50; k = 10; median of 3 runs per query after a warm-up pass. 'New in top 10' counts hits reranking brought in from below the fused top k.

| | p50 ms | p95 ms | max ms |
|---|---|---|---|
| hybrid | 78 | 151 | 181 |
| hybrid + rerank | 672 | 747 | 753 |
| rerank added | 587 | 637 | 665 |

| Kind | Query | Hybrid ms | + rerank ms | New in top 10 | First hit before | First hit after |
|---|---|---|---|---|---|---|
| term | undue influence | 74 | 656 | 5 | Indian Contract Act, 1872 — s. 16 — "Undue influence" d | Indian Contract Act, 1872 — s. 16 — "Undue influence" d |
| term | frustration of contract | 76 | 643 | 5 | RAJA DHRUV DEV CHAND v. HARMOHINDER SINGH & ANR. — Supr | NATIONAL AGRICULTURAL COOPERATIVE MARKETING FEDERATION  |
| term | liquidated damages | 81 | 652 | 3 | MAYA DEVI v. LALTA PRASAD — Supreme Court of India — 20 | MAYA DEVI v. LALTA PRASAD — Supreme Court of India — 20 |
| term | readiness and willingness | 76 | 676 | 7 | M/S J.P. BUILDERS & ANR. v. A. RAMADAS RAO & ANR. — Sup | RAMATHAL v. MARUTHATHAL & ORS. — Supreme Court of India |
| term | quantum meruit | 58 | 618 | 3 | M/S. ALOPI PARSHAD & SONS, LTD. v. THE UNION OF INDIA — | CARBP/738/2019 of PRABHAT AGRI BIOTECH LTD. Vs MAHYCO M |
| term | privity of contract | 63 | 623 | 6 | COX AND KINGS LTD. v. SAP INDIA PVT. LTD. & ANR. — Supr | RAJEEV SURI v. DELHI DEVELOPMENT AUTHORITY & ORS — Supr |
| term | restraint of trade | 80 | 672 | 5 | SUPERINTENDENCE COMPANY OF INDIA (P) LTD. v. KRISHAN MU | SUPERINTENDENCE COMPANY OF INDIA (P) LTD. v. KRISHAN MU |
| section | Section 74 | 70 | 650 | 6 | VEENA SINGH (DEAD) THROUGH LR v. THE DISTRICT REGISTRAR | PHULCHAND EXPORTS LTD v. O.O.O. PATRIOT — Supreme Court |
| section | s. 16(2) undue influence | 72 | 638 | 4 | CENTRAL ORGANISATION FOR RAILWAY ELECTRIFICATION v. M/S | AFSAR SHEIKH AND ANR. v. SOLEMAN BIBI AND ORS. — Suprem |
| section | Section 20 of the Specific Relief Act | 68 | 637 | 5 | C. HARIDASAN v. ANAPPATH PARAKKATTU VASUDEVA KURUP & OT | JANARDAN DAS & ORS. v. DURGA PRASAD AGARWALLA & ORS. —  |
| section | u/s 73 compensation for breach | 48 | 669 | 5 | PHULCHAND EXPORTS LTD v. O.O.O. PATRIOT — Supreme Court | CONSOLIDATED CONSTRUCTION CONSORTIUM LIMITED v. SOFTWAR |
| section | Section 56 impossibility of performance | 75 | 661 | 4 | CS(COMM)/531/2020 of ROHIT MALHOTRA Vs GURVINDER SINGH  | NATIONAL AGRICULTURAL COOPERATIVE MARKETING FEDERATION  |
| section | Section 19A | 73 | 664 | 10 | ENERCON (INDIA) LTD. & ORS. v. ENERCON GMBH & ANR. — Su | RFA/167/2017 of NORTH DELHI MUNICIPAL CORPORATION AND A |
| case | Satyabrata Ghose v. Mugneeram Bangur | 63 | 605 | 5 | SATYABRATA GROSE v. MUGNEERAM BANGUR & CO., AND ANOTHER | SATYABRATA GROSE v. MUGNEERAM BANGUR & CO., AND ANOTHER |
| case | Fateh Chand v. Balkishan Dass | 85 | 602 | 6 | AUTHORISED OFFICER, CENTRAL BANK OF INDIA v. SHANMUGAVE | FATEH CHAND v. BALKLSHAN DAS — Supreme Court of India — |
| case | ONGC v. Saw Pipes | 64 | 674 | 6 | SSANGYONG CONSTRUCTION CO. LTD. v. NATIONAL HIGHWAYS AU | OIL & NATURAL GAS CORPORATION LTD. v. SAW PIPES LTD. —  |
| case | Kailash Nath Associates | 73 | 701 | 7 | M/S. KAILASH NATH ASSOCIATES v. DELHI DEVELOPMENT AUTHO | M/S. KAILASH NATH ASSOCIATES v. DELHI DEVELOPMENT AUTHO |
| case | Mohori Bibee | 66 | 654 | 4 | MATHAI MATHAI v. JOSEPH MARY @ MARYKKUTIY JOSEPH & ORS. | MANIK CHAND AND ANR. v. RAMACHANDRA SON OF CHAWRIRAJ —  |
| question | Is a contract entered into under coercion by a threat to a third party voidable at the option of the coerced party? | 112 | 743 | 3 | Indian Contract Act, 1872 — s. 19 — Voidability of agre | ASSISTANT GENERAL MANAGER, STATE BANK OF INDIA & ORS. v |
| question | Can earnest money be forfeited without proof of actual loss? | 97 | 709 | 8 | RFA/678/2006 of PRAVEEN TALWAR Vs NARESH KUMAR MITTAL & | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DEC |
| question | When is time of the essence in a contract for sale of immovable property? | 82 | 726 | 6 | K.S. VIDYANADAM AND ORS. v. VAIRAVAN — Supreme Court of | MRS. SARADAMANI KANDAPPAN v. MRS. S. RAJALAKSHMI & ORS. |
| question | Does a minor's agreement bind the minor or can it be ratified on majority? | 87 | 753 | 5 | GOPAL PRASAD v. BIHAR SCHOOL EXAMINATION BOARD AND OTHE | GOPAL PRASAD v. BIHAR SCHOOL EXAMINATION BOARD AND OTHE |
| common | contract | 98 | 624 | 6 | Indian Contract Act, 1872 — s. 75 — Party rightfully re | FIRM OF PRATAPCHAND NOPAJI v. FIRM OF KOTRIKE VENKATTA  |
| common | court | 116 | 744 | 8 | CARAP/115/2022 of PHTHALO COLOURS AND CHEMICALS (INDIA) | STATE OF WEST BENGAL & ORS. v. ASSOCIATED CONTRACTORS — |
| common | agreement party | 117 | 753 | 5 | ASF BUILDTECH PRIVATE LIMITED v. SHAPOORJI PALLONJI AND | OIL AND NATURAL GAS CORPORATION LTD. v. M/S DISCOVERY E |
| filtered | specific performance [{'court_levels': ['supreme_court']}] | 105 | 717 | 6 | Specific Relief Act, 1963 — s. 15 — Who may obtain spec | RAMATHAL v. MARUTHATHAL & ORS. — Supreme Court of India |
| filtered | liquidated damages [{'date_from': '2000-01-01', 'date_to': '2020-12-31'}] | 91 | 703 | 3 | MAYA DEVI v. LALTA PRASAD — Supreme Court of India — 20 | MAYA DEVI v. LALTA PRASAD — Supreme Court of India — 20 |
| filtered | specific performance discretion [{'acts': ['SRA']}] | 78 | 699 | 7 | ZARINA SIDDIQUI v. A. RAMALINGAMALIAS R.AMARNATHAN — Su | HER HIGHNESS MAHARANI SHANTIDEVI P. GAIKWAD v. SAVJIBHA |
| filtered | compensation for loss [{'doc_kinds': ['statute']}] | 51 | 489 | 3 | Indian Contract Act, 1872 — s. 73 — Compensation for lo | Indian Contract Act, 1872 — s. 73 — Compensation for lo |
| filtered | contract [{'court_levels': ['high_court'], 'acts': ['ICA']}] | 181 | 699 | 7 | Indian Contract Act, 1872 — s. 75 — Party rightfully re | Indian Contract Act, 1872 — s. 65 — Obligation of perso |
| filtered | penalty clause reasonable compensation [{'date_from': '2010-01-01', 'date_to': '2012-12-31'}] | 161 | 747 | 2 | PHULCHAND EXPORTS LTD v. O.O.O. PATRIOT — Supreme Court | Indian Contract Act, 1872 — s. 74 — Compensation for br |
| filtered | agreement void for uncertainty [{'date_to': '1959-12-31'}] | 151 | 725 | 6 | Indian Contract Act, 1872 — s. 29 — Agreements void for | Indian Contract Act, 1872 — s. 29 — Agreements void for |
