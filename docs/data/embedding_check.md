# Embeddings: coverage and neighbour check (PLAN 3.8)

## Setup

| | |
|---|---|
| Model | `BAAI/bge-m3`, revision `5617a9f61b028005a4858fdac845db406aefb181` (weights SHA-256 `b5e0ce34…6aad38`, checked against Hugging Face) |
| Vector | dense [CLS] vector, 1,024 dimensions, L2-normalised; cosine distance (`<=>`) |
| Input | `context_header + "\n" + text` (at most 505 tokens; the limit is 512, D-025) |
| Hardware | local RTX 4070 Laptop GPU (8 GB), fp16, batches of 32 sorted by length |
| Run | `uv run --group embed scripts/embed_corpus.py`: resumable, committed per 2,048 chunks |
| Throughput | ~35–55 chunks/s; the whole corpus took about 50 minutes |
| Index | partial HNSW (`vector_cosine_ops`, pgvector defaults m=16, ef_construction=64) on `(embedding::vector(1024)) WHERE model = 'BAAI/bge-m3'`, built after the bulk load: 976 MB |
| Record | `snapshots.embeddings['BAAI/bge-m3']` = revision, dim, 124,996 of 124,996 chunks, time; `embedding_models.revision` |

**Coverage: 124,996 of 124,996 chunks (100%) have a bge-m3 vector.** `EXPLAIN` shows neighbour queries use the HNSW index (`Index Scan using ix_chunk_embeddings_hnsw_baai_bge_m3`). The alternative candidate (multilingual-e5-large) is registered with an empty index and waits for the retrieval bake-off (PLAN 4.x).

## Neighbour check: 5 hand-picked chunks

`uv run scripts/embed_corpus.py --check <chunk_id> ...` lists the 5 nearest other chunks. The number is the cosine distance.

**1. Indian Contract Act s. 73, part 1 (compensation for breach): sensible.**

The nearest chunks are three other parts of s. 73 (its illustrations), a Delhi HC judgment on damages for breach (*Ajay Kalra v. DDA*, ¶ 148) and a Supreme Court judgment on damages for breach of contract (*Phulchand Exports v. O.O.O. Patriot*, ¶ 26):

| Distance | Chunk |
|---|---|
| 0.116 | ICA s. 73 (part 5/6) |
| 0.135 | *Ajay Kalra v. DDA* (Del HC 2023), ¶ 148 |
| 0.142 | ICA s. 73 (part 4/6) |
| 0.142 | *Phulchand Exports v. O.O.O. Patriot* (SC 2011), ¶ 26 |
| 0.143 | ICA s. 73 (part 2/6) |

**2. Specific Relief Act s. 10 (specific performance in respect of contracts): sensible.**

The nearest chunks are the neighbouring sections of the same chapter on specific performance, including s. 14 (contracts not specifically enforceable) and s. 16 (personal bars):

| Distance | Chunk |
|---|---|
| 0.150 | SRA s. 11 |
| 0.164 | SRA s. 12 (part 1/2) |
| 0.172 | SRA s. 14 |
| 0.179 | SRA s. 16 |
| 0.188 | SRA s. 15 |

**3. Delhi HC, *Kamal Jeet v. Sneh Lata Chaturvedi* (2018), ¶ 7 (forfeiture of earnest money): sensible.**

The nearest chunks are four other Delhi HC appeals on forfeiture of earnest money (the same line of cases applying *Fateh Chand* and *Kailash Nath*) and the same judgment's ¶ 7:

| Distance | Chunk |
|---|---|
| 0.073 | *M C Luthra v. Ashok Kumar Khanna* (2018), ¶ 7 (part 2/5) |
| 0.087 | *Kamal Jeet*, ¶ 17 (part 27/34) |
| 0.089 | *Arun Khanna v. Sumit* (2018), ¶ 7 (part 2/4) |
| 0.106 | *Rajbir Singh v. Jaswant Yadav* (2018), ¶ 7 (part 2/5) |
| 0.110 | *Kamal Jeet*, ¶ 7 (part 1/5) |

**4. *MMTC v. Vedanta* (SC 2019), headnote (interference with an arbitral award under ss. 34/37): sensible.**

The nearest chunks are the same judgment and two other Supreme Court judgments on setting aside arbitral awards (*American Metallurgical Coal v. MMTC*, *PSA SICAL v. V.O. Chidambranar Port Trust*):

| Distance | Chunk |
|---|---|
| 0.039 | *MMTC v. Vedanta*, ¶ 11 |
| 0.085 | *MMTC v. Vedanta*, headnote |
| 0.134 | *American Metallurgical Coal v. MMTC* (2020), ¶ 45 |
| 0.134 | *MMTC v. Vedanta*, ¶¶ 12–14 |
| 0.141 | *PSA SICAL Terminals v. Board of Trustees* (2021), ¶ 40 |

**5. *Jaswinder Kaur v. Gurmeet Singh* (SC 2017), ¶ 22 (quotes *Surjit Kaur v. Naurata Singh* on specific performance): sensible.**

The paragraph quotes *Surjit Kaur*, and all five nearest chunks are from *Surjit Kaur v. Naurata Singh* (SC 2000) itself:

| Distance | Chunk |
|---|---|
| 0.085 | *Surjit Kaur v. Naurata Singh* (SC 2000), part of p-22 |
| 0.102 | *Surjit Kaur*, headnote |
| 0.110 | *Surjit Kaur*, part of p-22 |
| 0.161 | *Surjit Kaur*, p-19–p-20 |
| 0.176 | *Surjit Kaur*, headnote |

All 5 checks returned neighbours on the same legal question: statute sections next to their neighbouring sections and applying judgments, and judgments next to the cases they cite or the same line of authority.

## Notes

- **Same-document neighbours.** Neighbours often come from the same document, whose chunks share a header. Retrieval (PLAN 4.x) should diversify by document where that matters.
- **Shared memory.** The HNSW build uses parallel workers. Docker's default 64 MB `/dev/shm` failed the first build, so `docker-compose.yml` now sets `shm_size: 2gb`, and the index is built with `maintenance_work_mem = 1GB`.
