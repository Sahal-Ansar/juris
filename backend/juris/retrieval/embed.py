"""Dense embeddings for chunks, written to pgvector (PLAN 3.8).

The embedding input is ``context_header + "\\n" + text`` (the header is for embedding only,
D-025). bge-m3's dense vector is the [CLS] hidden state, L2-normalised, so cosine distance
(``<=>``) ranks neighbours. Encoding runs in fp16 on CUDA when available, with batches sorted
by length to limit padding.

``embed_missing`` is resumable: it only embeds chunks without a row for the model in
``chunk_embeddings``. For a bulk load it drops the model's HNSW index first and rebuilds it
at the end (much faster than updating the graph on every insert).

PyTorch and transformers are in the ``embed`` dependency group (``uv sync --group embed``)
and are imported only when an encoder is created.
"""

import datetime as dt
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from juris.db.embeddings import index_name, register_model

SEPARATOR = "\n"


class Encoder(Protocol):
    model: str
    revision: str
    dim: int

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def embedding_input(context_header: str | None, body: str) -> str:
    return f"{context_header}{SEPARATOR}{body}" if context_header else body


@dataclass
class TransformerEncoder:
    """Dense vectors from a local Hugging Face encoder: [CLS] or mean pooling, L2-normalised.

    ``passage_prefix`` is what the model expects before a passage ("passage: " for e5); the
    query side's prefix is ``query_prefix`` in ``configs/embeddings.yaml`` (``retrieval.dense``).
    """

    path: Path
    revision: str
    model: str = "BAAI/bge-m3"
    dim: int = 1024
    pooling: str = "cls"  # "cls" | "mean"
    passage_prefix: str = ""
    max_length: int = 512
    batch_size: int = 32
    device: str | None = None
    truncated: int = 0  # inputs longer than max_length (should stay 0: chunks fit 512, D-025)

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(str(self.path))
        self._net = AutoModel.from_pretrained(str(self.path), dtype=dtype).to(self.device).eval()

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        torch = self._torch
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: list[list[float]] = [[] for _ in texts]
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = self._tok(
                    [texts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_overflowing_tokens=False,
                    return_length=True,
                )
                self.truncated += int((batch.pop("length") >= self.max_length).sum())
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self._net(**batch).last_hidden_state
                if self.pooling == "mean":
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                else:
                    pooled = hidden[:, 0]
                vecs = torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().tolist()
                for i, vec in zip(idx, vecs, strict=True):
                    out[i] = vec
        return out


@dataclass
class BgeM3Encoder(TransformerEncoder):
    """BAAI/bge-m3 ([CLS] pooling) from ``<data_dir>/models/BAAI__bge-m3`` (D-026)."""


E5_MODEL = "intfloat/multilingual-e5-large"
E5_REVISION = "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
E5_FILES: dict[str, str | None] = {
    "config.json": None,
    "special_tokens_map.json": None,
    "tokenizer_config.json": None,
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "tokenizer.json": "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
    "model.safetensors": "020afdebf2762b29fcaf286629a96c3b3b65af241f6a08226b1cfee60a21def6",
}


@dataclass
class E5Encoder(TransformerEncoder):
    """intfloat/multilingual-e5-large (MIT): mean pooling, "passage: " before each passage."""

    model: str = E5_MODEL
    pooling: str = "mean"
    passage_prefix: str = "passage: "


def _literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def missing_chunks(conn: Connection, model: str, limit: int) -> list[tuple[str, str, str]]:
    rows = conn.execute(
        text(
            "SELECT c.chunk_id, c.context_header, c.text FROM chunks c "
            "LEFT JOIN chunk_embeddings e ON e.chunk_id = c.chunk_id AND e.model = :m "
            "WHERE e.chunk_id IS NULL ORDER BY c.chunk_id LIMIT :n"
        ),
        {"m": model, "n": limit},
    )
    return [(r[0], r[1], r[2]) for r in rows]


def count_missing(conn: Connection, model: str) -> tuple[int, int]:
    total: int = conn.execute(text("SELECT count(*) FROM chunks")).scalar_one()
    done: int = conn.execute(
        text("SELECT count(*) FROM chunk_embeddings WHERE model = :m"), {"m": model}
    ).scalar_one()
    return total, total - done


@dataclass
class EmbedReport:
    model: str
    embedded: int
    seconds: float
    index_rebuilt: bool
    truncated: int


def embed_missing(
    engine: Engine,
    encoder: Encoder,
    fetch: int = 2048,
    bulk_threshold: float = 0.1,
    progress: Callable[[int, int], None] | None = None,
) -> EmbedReport:
    """Embed every chunk that has no vector for ``encoder.model``; commit per fetched batch."""
    with engine.begin() as conn:
        register_model(conn, encoder.model, encoder.dim, encoder.revision)
        total, missing = count_missing(conn, encoder.model)
        bulk = total > 0 and missing / total >= bulk_threshold
        if bulk:  # rebuilt at the end; inserts into an HNSW index are slow
            conn.execute(text(f"DROP INDEX IF EXISTS {index_name(encoder.model)}"))
    started, done = time.monotonic(), 0
    while True:
        with engine.begin() as conn:
            rows = missing_chunks(conn, encoder.model, fetch)
            if not rows:
                break
            prefix = getattr(encoder, "passage_prefix", "")
            vectors = encoder.encode([prefix + embedding_input(h, b) for _, h, b in rows])
            conn.execute(
                text(
                    "INSERT INTO chunk_embeddings (chunk_id, model, embedding) "
                    "VALUES (:c, :m, CAST(:v AS vector)) ON CONFLICT DO NOTHING"
                ),
                [
                    {"c": cid, "m": encoder.model, "v": _literal(vec)}
                    for (cid, _, _), vec in zip(rows, vectors, strict=True)
                ],
            )
        done += len(rows)
        if progress:
            progress(done, missing)
    with engine.begin() as conn:
        # memory for the HNSW build (a no-op if the index exists); parallel workers need
        # /dev/shm at least this large in the container (docker-compose.yml shm_size)
        conn.execute(text("SET LOCAL maintenance_work_mem = '1GB'"))
        register_model(conn, encoder.model, encoder.dim, encoder.revision)  # (re)creates index
    return EmbedReport(
        encoder.model,
        done,
        time.monotonic() - started,
        bulk,
        getattr(encoder, "truncated", 0),
    )


def record_in_snapshot(conn: Connection, snapshot_id: str, encoder: Encoder) -> dict[str, Any]:
    """Store model, revision, dimension and coverage under ``snapshots.embeddings``."""
    total, missing = count_missing(conn, encoder.model)
    entry = {
        "revision": encoder.revision,
        "dim": encoder.dim,
        "chunks": total - missing,
        "of": total,
        "embedded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
    conn.execute(
        text(
            "UPDATE snapshots SET embeddings = embeddings || jsonb_build_object(CAST(:m AS text), "
            "CAST(:e AS jsonb)) WHERE snapshot_id = :s"
        ),
        {"m": encoder.model, "e": json.dumps(entry), "s": snapshot_id},
    )
    return entry


def nearest(
    conn: Connection, model: str, dim: int, chunk_id: str, k: int = 5
) -> list[tuple[str, str, float]]:
    """(chunk_id, context_header, cosine distance) of the k nearest other chunks (uses HNSW)."""
    rows = conn.execute(
        text(
            f"SELECT c.chunk_id, c.context_header, "
            f"(e.embedding::vector({dim})) <=> (q.embedding::vector({dim})) AS d "
            f"FROM chunk_embeddings q, chunk_embeddings e JOIN chunks c USING (chunk_id) "
            f"WHERE q.chunk_id = :c AND q.model = :m AND e.model = :m AND e.chunk_id <> :c "
            f"ORDER BY (e.embedding::vector({dim})) <=> (q.embedding::vector({dim})) LIMIT :k"
        ),
        {"c": chunk_id, "m": model, "k": k},
    )
    return [(r[0], r[1], float(r[2])) for r in rows]
