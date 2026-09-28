# Lexical retrieval latency (PLAN 4.1)

Made by `scripts/bench_lexical.py` on 2026-09-28. Corpus: 124,996 chunks. k = 50; median of 5 runs per query after a warm-up pass. Target: < 300 ms per query. Empty = no hits (phrase mode on a question, for example).

| Mode | p50 ms | p95 ms | max ms | Queries | Empty |
|---|---|---|---|---|---|
| any | 30 | 80 | 96 | 32 | 0 |
| all | 11 | 67 | 119 | 32 | 2 |
| phrase | 10 | 55 | 82 | 32 | 7 |

| Mode | Kind | Query | ms | Hits | First hit |
|---|---|---|---|---|---|
| any | term | undue influence | 23 | 50 | Indian Contract Act, 1872 — s. 16 — "Undue influence" defined (part 1/ |
| any | term | frustration of contract | 17 | 50 | THE NAIHATI JUTE MILLS LTD. v. KHYALIRAM JAGANNATH — Supreme Court of  |
| any | term | liquidated damages | 34 | 50 | RFA/1014/2018 of ARUN KHANNA Vs SUMIT — High Court of Delhi — 2018 — ¶ |
| any | term | readiness and willingness | 21 | 50 | JASWINDER KAUR (NOW DECEASED) THROUGH . v. GURMEET SINGH AND ORS — Sup |
| any | term | quantum meruit | 17 | 50 | MAHANAGAR TELEPHONE NIGAM LTD. v. TATA COMMUNICATIONS LTD. — Supreme C |
| any | term | privity of contract | 13 | 50 | CS(COMM)/1280/2016 of M/S KASHYAPI INFRASTRUCTURE PVT LTD Vs NATIONAL  |
| any | term | restraint of trade | 38 | 50 | M/S. GUJARAT BOTTLING COMPANY LTD. AND ORS. v. THE COCA COLA CO. AND O |
| any | section | Section 74 | 19 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| any | section | s. 16(2) undue influence | 25 | 50 | SUBHAS CHANDRA DAS MUSHIB v. GANGA PROSAD DAS MUSHIB AND ORS. — Suprem |
| any | section | Section 20 of the Specific Relief Act | 34 | 50 | RFA/535/2005 of BHASKAR SAINI Vs SATBIR SINGH — High Court of Delhi —  |
| any | section | u/s 73 compensation for breach | 41 | 50 | PHULCHAND EXPORTS LTD v. O.O.O. PATRIOT — Supreme Court of India — 201 |
| any | section | Section 56 impossibility of performance | 27 | 50 | LOOP TELECOM AND TRADING LIMITED v. UNION OF INDIA AND ANR. — Supreme  |
| any | section | Section 19A | 11 | 50 | EXECUTIVE ENGINEER, DHENKANAL MINOR IRRIGATION DIVISION, ORISSA, ETC.  |
| any | case | Satyabrata Ghose v. Mugneeram Bangur | 12 | 50 | GOVINDBHAI GORDHANBHAI PATEL & ORS. v. GULAM ABBAS MULLA ALLIBHAI & OR |
| any | case | Fateh Chand v. Balkishan Dass | 30 | 50 | K.R. SURESH v. R. POORNIMA & ORS. — Supreme Court of India — 2025 — ¶  |
| any | case | ONGC v. Saw Pipes | 22 | 50 | SSANGYONG CONSTRUCTION CO. LTD. v. NATIONAL HIGHWAYS AUTHORITY OF INDI |
| any | case | Kailash Nath Associates | 26 | 50 | M/S. KAILASH NATH ASSOCIATES v. DELHI DEVELOPMENT AUTHORITY & ANR. — S |
| any | case | Mohori Bibee | 11 | 12 | CS(COMM)/1155/2018 of SU-KAM POWER SYETEMS LTD. Vs MR. KUNWER SACHDEV  |
| any | question | Is a contract entered into under coercion by a threat to a third party voidable at the option of the coerced party? | 84 | 50 | NATIONAL INSURANCE CO. LTD. v. M/S. BOGHARA POLYFAB PVT. LTD. — Suprem |
| any | question | Can earnest money be forfeited without proof of actual loss? | 68 | 50 | RFA/780/2017 of M C LUTHRA Vs ASHOK KUMAR KHANNA — High Court of Delhi |
| any | question | When is time of the essence in a contract for sale of immovable property? | 39 | 50 | P. PURUSHOTTAM REDDY AND ANR. v. M/S PRATAP STEELS LTD. — Supreme Cour |
| any | question | Does a minor's agreement bind the minor or can it be ratified on majority? | 54 | 50 | GOPAL PRASAD v. BIHAR SCHOOL EXAMINATION BOARD AND OTHERS — Supreme Co |
| any | common | contract | 49 | 50 | ACE PIPELINE CONTRACTS PRIVATE LIMITED v. BHARAT PETROLEUM CORPORATION |
| any | common | court | 78 | 50 | RFA/248/2022 of KIRAN Vs VIRENDER KUMAR JAIN — High Court of Delhi — 2 |
| any | common | agreement party | 96 | 50 | IN RE: INTERPLAY BETWEEN ARBITRATION AGREEMENTS UNDER THE ARBITRATION  |
| any | filtered | specific performance [{'court_levels': ['supreme_court']}] | 42 | 50 | Specific Relief Act, 1963 — s. 12 — Specific performance of part of co |
| any | filtered | liquidated damages [{'date_from': '2000-01-01', 'date_to': '2020-12-31'}] | 26 | 50 | RFA/1014/2018 of ARUN KHANNA Vs SUMIT — High Court of Delhi — 2018 — ¶ |
| any | filtered | specific performance discretion [{'acts': ['SRA']}] | 30 | 50 | PARSWANATH SAHA v. BANDHANA MODAK (DAS) AND ANR. — Supreme Court of In |
| any | filtered | compensation for loss [{'doc_kinds': ['statute']}] | 19 | 50 | Indian Contract Act, 1872 — s. 73 — Compensation for loss or damage ca |
| any | filtered | contract [{'court_levels': ['high_court'], 'acts': ['ICA']}] | 59 | 50 | Indian Contract Act, 1872 — s. 73 — Compensation for loss or damage ca |
| any | filtered | penalty clause reasonable compensation [{'date_from': '2010-01-01', 'date_to': '2012-12-31'}] | 33 | 50 | PHULCHAND EXPORTS LTD v. O.O.O. PATRIOT — Supreme Court of India — 201 |
| any | filtered | agreement void for uncertainty [{'date_to': '1959-12-31'}] | 80 | 50 | Indian Contract Act, 1872 — s. 29 — Agreements void for uncertainty |
| all | term | undue influence | 15 | 50 | BELLACHI (DEAD) BY LR v. PAKEERAN — Supreme Court of India — 2009 — he |
| all | term | frustration of contract | 13 | 50 | RAJA DHRUV DEV CHAND v. HARMOHINDER SINGH & ANR. — Supreme Court of In |
| all | term | liquidated damages | 14 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| all | term | readiness and willingness | 18 | 50 | RAVI SETIA v. MADAN LAL AND OTHERS — Supreme Court of India — 2019 — ¶ |
| all | term | quantum meruit | 8 | 30 | MAHANAGAR TELEPHONE NIGAM LTD. v. TATA COMMUNICATIONS LTD. — Supreme C |
| all | term | privity of contract | 9 | 50 | THE CORRESPONDENCE, RBANMS EDUCATIONAL INSTITUTION v. B. GUNASHEKAR &  |
| all | term | restraint of trade | 9 | 50 | M/S. GUJARAT BOTTLING COMPANY LTD. AND ORS. v. THE COCA COLA CO. AND O |
| all | section | Section 74 | 17 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| all | section | s. 16(2) undue influence | 10 | 1 | SUBHAS CHANDRA DAS MUSHIB v. GANGA PROSAD DAS MUSHIB AND ORS. — Suprem |
| all | section | Section 20 of the Specific Relief Act | 20 | 50 | RFA/535/2005 of BHASKAR SAINI Vs SATBIR SINGH — High Court of Delhi —  |
| all | section | u/s 73 compensation for breach | 12 | 50 | DWARAKA DAS v. STATE OF MADHYA PRADESH AND ANR. — Supreme Court of Ind |
| all | section | Section 56 impossibility of performance | 11 | 50 | THE NAIHATI JUTE MILLS LTD. v. KHYALIRAM JAGANNATH — Supreme Court of  |
| all | section | Section 19A | 9 | 50 | EXECUTIVE ENGINEER, DHENKANAL MINOR IRRIGATION DIVISION, ORISSA, ETC.  |
| all | case | Satyabrata Ghose v. Mugneeram Bangur | 8 | 27 | GOVINDBHAI GORDHANBHAI PATEL & ORS. v. GULAM ABBAS MULLA ALLIBHAI & OR |
| all | case | Fateh Chand v. Balkishan Dass | 8 | 18 | K.R. SURESH v. R. POORNIMA & ORS. — Supreme Court of India — 2025 — ¶  |
| all | case | ONGC v. Saw Pipes | 9 | 50 | PSA SICAL TERMINALS PVT. LTD. v. THE BOARD OF TRUSTEES OF V.O. CHIDAMB |
| all | case | Kailash Nath Associates | 9 | 50 | RFA/404/2018 of RAJBIR SINGH & ANR Vs JASWANT YAVDAV — High Court of D |
| all | case | Mohori Bibee | 6 | 8 | CS(COMM)/1155/2018 of SU-KAM POWER SYETEMS LTD. Vs MR. KUNWER SACHDEV  |
| all | question | Is a contract entered into under coercion by a threat to a third party voidable at the option of the coerced party? | 8 | 0 |  |
| all | question | Can earnest money be forfeited without proof of actual loss? | 8 | 0 |  |
| all | question | When is time of the essence in a contract for sale of immovable property? | 11 | 50 | P. PURUSHOTTAM REDDY AND ANR. v. M/S PRATAP STEELS LTD. — Supreme Cour |
| all | question | Does a minor's agreement bind the minor or can it be ratified on majority? | 9 | 50 | I. C. GOLAK NATH & ORS. v. STA TE OF PUNJAB & ANRS. — Supreme Court of |
| all | common | contract | 52 | 50 | CS(COMM)/377/2020 of KNOWLEDGE PODIUM SYSTEMS PVT. LTD. Vs S M PROFESS |
| all | common | court | 78 | 50 | RFA/248/2022 of KIRAN Vs VIRENDER KUMAR JAIN — High Court of Delhi — 2 |
| all | common | agreement party | 119 | 50 | GREAT OFFSHORE LTD. v. IRANIAN OFFSHORE ENGINEERING & CONSTRUCTION COM |
| all | filtered | specific performance [{'court_levels': ['supreme_court']}] | 62 | 50 | MAN KAUR (DEAD) BY LRS. v. HARTAR SINGH SANGHA — Supreme Court of Indi |
| all | filtered | liquidated damages [{'date_from': '2000-01-01', 'date_to': '2020-12-31'}] | 14 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| all | filtered | specific performance discretion [{'acts': ['SRA']}] | 24 | 50 | BAL KRISHNA AND ANR. v. BHAGWAN DAS (DEAD) AND ORS . — Supreme Court o |
| all | filtered | compensation for loss [{'doc_kinds': ['statute']}] | 14 | 17 | Indian Contract Act, 1872 — s. 56 — Agreement to do impossible act. Co |
| all | filtered | contract [{'court_levels': ['high_court'], 'acts': ['ICA']}] | 67 | 50 | CS(COMM)/377/2020 of KNOWLEDGE PODIUM SYSTEMS PVT. LTD. Vs S M PROFESS |
| all | filtered | penalty clause reasonable compensation [{'date_from': '2010-01-01', 'date_to': '2012-12-31'}] | 11 | 8 | B.S.N.L v. RELIANCE COMMUNICATION LTD. — Supreme Court of India — 2010 |
| all | filtered | agreement void for uncertainty [{'date_to': '1959-12-31'}] | 8 | 4 | Indian Contract Act, 1872 — s. 29 — Agreements void for uncertainty |
| phrase | term | undue influence | 14 | 50 | SHIV KIRPAL SINGH v. SHRI V. V. GIRI — Supreme Court of India — 1970 — |
| phrase | term | frustration of contract | 10 | 50 | RAJA DHRUV DEV CHAND v. HARMOHINDER SINGH & ANR. — Supreme Court of In |
| phrase | term | liquidated damages | 14 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| phrase | term | readiness and willingness | 18 | 50 | MAN KAUR (DEAD) BY LRS. v. HARTAR SINGH SANGHA — Supreme Court of Indi |
| phrase | term | quantum meruit | 7 | 30 | MAHANAGAR TELEPHONE NIGAM LTD. v. TATA COMMUNICATIONS LTD. — Supreme C |
| phrase | term | privity of contract | 10 | 50 | M/S ARIF AZIM CO. LTD. v. M/S MICROMAX INFORMATICS FZE — Supreme Court |
| phrase | term | restraint of trade | 10 | 50 | M/S. GUJARAT BOTTLING COMPANY LTD. AND ORS. v. THE COCA COLA CO. AND O |
| phrase | section | Section 74 | 17 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| phrase | section | s. 16(2) undue influence | 10 | 0 |  |
| phrase | section | Section 20 of the Specific Relief Act | 18 | 50 | RFA/535/2005 of BHASKAR SAINI Vs SATBIR SINGH — High Court of Delhi —  |
| phrase | section | u/s 73 compensation for breach | 9 | 1 | DWARAKA DAS v. STATE OF MADHYA PRADESH AND ANR. — Supreme Court of Ind |
| phrase | section | Section 56 impossibility of performance | 8 | 1 | THE NAIHATI JUTE MILLS LTD. v. KHYALIRAM JAGANNATH — Supreme Court of  |
| phrase | section | Section 19A | 9 | 50 | EXECUTIVE ENGINEER, DHENKANAL MINOR IRRIGATION DIVISION, ORISSA, ETC.  |
| phrase | case | Satyabrata Ghose v. Mugneeram Bangur | 8 | 21 | GOVINDBHAI GORDHANBHAI PATEL & ORS. v. GULAM ABBAS MULLA ALLIBHAI & OR |
| phrase | case | Fateh Chand v. Balkishan Dass | 8 | 14 | K.R. SURESH v. R. POORNIMA & ORS. — Supreme Court of India — 2025 — ¶  |
| phrase | case | ONGC v. Saw Pipes | 9 | 13 | ASSOCIATE BUILDERS v. DELHI DEVELOPMENT AUTHORITY — Supreme Court of I |
| phrase | case | Kailash Nath Associates | 10 | 50 | RFA/404/2018 of RAJBIR SINGH & ANR Vs JASWANT YAVDAV — High Court of D |
| phrase | case | Mohori Bibee | 6 | 8 | CS(COMM)/1155/2018 of SU-KAM POWER SYETEMS LTD. Vs MR. KUNWER SACHDEV  |
| phrase | question | Is a contract entered into under coercion by a threat to a third party voidable at the option of the coerced party? | 7 | 0 |  |
| phrase | question | Can earnest money be forfeited without proof of actual loss? | 7 | 0 |  |
| phrase | question | When is time of the essence in a contract for sale of immovable property? | 8 | 0 |  |
| phrase | question | Does a minor's agreement bind the minor or can it be ratified on majority? | 7 | 0 |  |
| phrase | common | contract | 52 | 50 | CS(COMM)/377/2020 of KNOWLEDGE PODIUM SYSTEMS PVT. LTD. Vs S M PROFESS |
| phrase | common | court | 82 | 50 | RFA/248/2022 of KIRAN Vs VIRENDER KUMAR JAIN — High Court of Delhi — 2 |
| phrase | common | agreement party | 29 | 16 | STATE OF PUNJAB (NOW HARYANA) AND ORS. v. AMAR SINGH AND ANOTHER — Sup |
| phrase | filtered | specific performance [{'court_levels': ['supreme_court']}] | 55 | 50 | MAN KAUR (DEAD) BY LRS. v. HARTAR SINGH SANGHA — Supreme Court of Indi |
| phrase | filtered | liquidated damages [{'date_from': '2000-01-01', 'date_to': '2020-12-31'}] | 14 | 50 | RFA/231/2018 of KAMAL JEET Vs SNEH LATA CHATURVEDI (DECEASED THR HER L |
| phrase | filtered | specific performance discretion [{'acts': ['SRA']}] | 11 | 0 |  |
| phrase | filtered | compensation for loss [{'doc_kinds': ['statute']}] | 13 | 7 | Indian Contract Act, 1872 — s. 56 — Agreement to do impossible act. Co |
| phrase | filtered | contract [{'court_levels': ['high_court'], 'acts': ['ICA']}] | 69 | 50 | CS(COMM)/377/2020 of KNOWLEDGE PODIUM SYSTEMS PVT. LTD. Vs S M PROFESS |
| phrase | filtered | penalty clause reasonable compensation [{'date_from': '2010-01-01', 'date_to': '2012-12-31'}] | 8 | 0 |  |
| phrase | filtered | agreement void for uncertainty [{'date_to': '1959-12-31'}] | 7 | 1 | Indian Contract Act, 1872 — s. 29 — Agreements void for uncertainty |
