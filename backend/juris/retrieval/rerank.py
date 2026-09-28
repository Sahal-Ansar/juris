"""Cross-encoder reranking of retrieved chunks (PLAN 4.4, D-031).

The fused candidates (``retrieval.hybrid``) are re-scored by a cross-encoder that reads the
query and each chunk together, which is more precise than either retriever. Only the top
``top_n`` (about 50) candidates are scored; the reranked list replaces them, in their new order.

The model is ``BAAI/bge-reranker-v2-m3`` (Apache-2.0), pinned to a revision and fetched once
into ``<data_dir>/models/BAAI__bge-reranker-v2-m3`` with SHA-256 checks (about 2.3 GB). The
passage is the chunk's context header + text, as for embedding. Scores are the sigmoid of the
model's logit, in (0, 1). It runs in fp16 on CUDA when available, else on the CPU.

PyTorch and transformers are in the ``embed`` dependency group and are imported only when a
``BgeReranker`` is created; tests use a fake cross-encoder.
"""

import hashlib
import math
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from sqlalchemy import text
from sqlalchemy.engine import Engine

from juris.config import get_settings
from juris.retrieval.embed import embedding_input

MODEL = "BAAI/bge-reranker-v2-m3"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
# file -> SHA-256 (large files, from the Hub's LFS metadata) or None (small config files)
FILES: dict[str, str | None] = {
    "config.json": None,
    "special_tokens_map.json": None,
    "tokenizer_config.json": None,
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "tokenizer.json": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
    "model.safetensors": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
}


def reranker_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or get_settings().data_dir) / "models" / MODEL.replace("/", "__")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_reranker(data_dir: Path | None = None) -> Path:
    """Download the pinned model files that are missing, checking hashes; returns the dir."""
    target = reranker_dir(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        path = target / name
        if path.exists() and (digest is None or sha256(path) == digest):
            continue
        part = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        urllib.request.urlretrieve(url, part)
        if digest is not None and sha256(part) != digest:
            part.unlink()
            raise ValueError(f"{name}: SHA-256 mismatch")
        part.replace(path)
    return target


class CrossEncoder(Protocol):
    model: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]: ...


@dataclass
class BgeReranker:
    """bge-reranker-v2-m3 from local weights; ``score`` returns sigmoid(logit) per passage."""

    path: Path
    model: str = MODEL
    max_length: int = 512
    batch_size: int = 16
    device: str | None = None

    def __post_init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self._tok = AutoTokenizer.from_pretrained(str(self.path))
        self._net = (
            AutoModelForSequenceClassification.from_pretrained(str(self.path), dtype=dtype)
            .to(self.device)
            .eval()
        )

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        torch = self._torch
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))
        out = [0.0] * len(passages)
        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = self._tok(
                    [query] * len(idx),
                    [passages[i] for i in idx],
                    padding=True,
                    truncation="only_second",  # keep the whole query, trim the passage
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self._net(**batch).logits.view(-1).float().cpu().tolist()
                for i, logit in zip(idx, logits, strict=True):
                    out[i] = 1 / (1 + math.exp(-logit))
        return out


class Candidate(Protocol):
    @property
    def chunk_id(self) -> str: ...
    @property
    def doc_id(self) -> str: ...
    @property
    def score(self) -> float: ...
    @property
    def rank(self) -> int: ...


@dataclass(frozen=True)
class RerankedHit:
    chunk_id: str
    doc_id: str
    score: float  # the cross-encoder's score
    rank: int  # 1-based, after reranking
    retrieval_score: float  # the candidate's score before reranking
    retrieval_rank: int


class Reranker:
    """Re-scores the top ``top_n`` candidates for a query with a cross-encoder."""

    def __init__(self, engine: Engine, encoder: CrossEncoder, top_n: int = 50) -> None:
        self.engine = engine
        self.encoder = encoder
        self.top_n = top_n

    def passages(self, chunk_ids: Sequence[str]) -> dict[str, str]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT chunk_id, context_header, text FROM chunks WHERE chunk_id = ANY(:ids)"
                ),
                {"ids": list(chunk_ids)},
            )
            return {r[0]: embedding_input(r[1], r[2]) for r in rows}

    def rerank(
        self, query: str, candidates: Sequence[Candidate], k: int | None = None
    ) -> list[RerankedHit]:
        """The top ``top_n`` candidates in cross-encoder order (at most ``k`` of them)."""
        pool = list(candidates)[: self.top_n]
        if not pool:
            return []
        texts = self.passages([c.chunk_id for c in pool])
        missing = [c.chunk_id for c in pool if c.chunk_id not in texts]
        if missing:
            raise KeyError(f"chunks not in the database: {missing[:3]}")
        scores = self.encoder.score(query, [texts[c.chunk_id] for c in pool])
        order = sorted(range(len(pool)), key=lambda i: (-scores[i], pool[i].rank))
        hits = [
            RerankedHit(pool[i].chunk_id, pool[i].doc_id, scores[i], n, pool[i].score, pool[i].rank)
            for n, i in enumerate(order, 1)
        ]
        return hits[:k] if k is not None else hits
