"""Embedding models and their per-model HNSW indexes (PLAN 3.6, used by 3.8).

``chunk_embeddings.embedding`` is an unsized ``vector``: pgvector can only index a fixed
dimension, so each registered model gets a partial HNSW index over its own rows, cast to its
dimension. Queries must use the same cast and filter to hit it::

    ORDER BY (embedding::vector(1024)) <=> :q  ...  WHERE model = 'BAAI/bge-m3'
"""

import re
from pathlib import Path

import yaml
from sqlalchemy import text
from sqlalchemy.engine import Connection

from juris.config import REPO_ROOT

EMBEDDINGS_CONFIG = REPO_ROOT / "configs" / "embeddings.yaml"


def model_dims(path: Path | None = None) -> dict[str, int]:
    """{model name: dimension} for the candidate models in ``configs/embeddings.yaml``."""
    data = yaml.safe_load((path or EMBEDDINGS_CONFIG).read_text(encoding="utf-8"))
    return {name: int(m["dim"]) for name, m in data["models"].items()}


def index_name(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"ix_chunk_embeddings_hnsw_{slug}"[:63]


def vector_expr(model_dim: int) -> str:
    """The indexed expression to order by: ``(embedding::vector(1024))``."""
    return f"(embedding::vector({int(model_dim)}))"


def register_model(conn: Connection, model: str, dim: int, revision: str | None = None) -> str:
    """Record ``model`` and create its HNSW (cosine) index if missing; returns the index name.

    Registering an existing model with a different dimension is an error: its stored vectors
    would no longer match the index. A given ``revision`` (the weights' commit) is recorded.
    """
    if dim < 1 or dim > 16000:
        raise ValueError(f"unsupported dimension {dim}")
    existing = conn.execute(
        text("SELECT dim FROM embedding_models WHERE model = :m"), {"m": model}
    ).scalar()
    if existing is not None and existing != dim:
        raise ValueError(f"{model} is registered with dim {existing}, not {dim}")
    name = index_name(model)
    conn.execute(
        text(
            "INSERT INTO embedding_models (model, dim, index_name) VALUES (:m, :d, :i) "
            "ON CONFLICT (model) DO NOTHING"
        ),
        {"m": model, "d": dim, "i": name},
    )
    if revision:
        conn.execute(
            text("UPDATE embedding_models SET revision = :r WHERE model = :m"),
            {"r": revision, "m": model},
        )
    # identifiers can't be bound; the name is a slug and the model is bound in a literal below
    literal = model.replace("'", "''")
    conn.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {name} ON chunk_embeddings "
            f"USING hnsw ({vector_expr(dim)} vector_cosine_ops) WHERE model = '{literal}'"
        )
    )
    return name
