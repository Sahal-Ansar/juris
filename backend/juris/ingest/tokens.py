"""Token counting with the embedding models' own tokenizer (PLAN 3.7).

The candidate models (``configs/embeddings.yaml``) share XLM-RoBERTa's sentencepiece
vocabulary, so bge-m3's tokenizer counts for both. Its files are fetched once, pinned to a
revision, into ``<data_dir>/models/BAAI__bge-m3``.
"""

import urllib.request
from functools import lru_cache
from pathlib import Path

from tokenizers import Tokenizer

from juris.config import get_settings

MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
SPECIAL_TOKENS = 2  # <s> ... </s> added around every input


def tokenizer_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or get_settings().data_dir) / "models" / MODEL.replace("/", "__")


def fetch_tokenizer(data_dir: Path | None = None) -> Path:
    """Download the tokenizer files if missing (about 17 MB); returns their directory."""
    target = tokenizer_dir(data_dir)
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = target / name
        if not path.exists():
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            urllib.request.urlretrieve(url, path.with_suffix(".part"))
            path.with_suffix(".part").replace(path)
    return target


@lru_cache
def load_tokenizer(directory: str | None = None) -> Tokenizer:
    path = Path(directory) if directory else tokenizer_dir()
    tok = Tokenizer.from_file(str(path / "tokenizer.json"))
    tok.no_truncation()
    tok.no_padding()
    return tok


def count_tokens(texts: list[str], tokenizer: Tokenizer | None = None) -> list[int]:
    """Tokens per text, without the special tokens the embedding model adds."""
    tok = tokenizer or load_tokenizer()
    return [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
