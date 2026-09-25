"""Versioned prompt templates.

Each template is a file in this folder (``*.md`` or ``*.j2``) that starts with a
header::

    ---
    id: issue_framer
    version: 3
    ---
    <template body>

The hash covers the whole file (header and body) with line endings normalised,
so it is the same on every OS and changes whenever the prompt does.
"""

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

PROMPTS_DIR = Path(__file__).resolve().parent
TEMPLATE_SUFFIXES = (".md", ".j2")


class PromptTemplate(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    version: int
    body: str
    sha256: str


class PromptFormatError(ValueError):
    pass


def parse_prompt(text: str, source: str = "<string>") -> PromptTemplate:
    text = text.replace("\r\n", "\n")
    if not text.startswith("---\n"):
        raise PromptFormatError(f"{source}: missing '---' header")
    header, sep, body = text[4:].partition("\n---\n")
    if not sep:
        raise PromptFormatError(f"{source}: header is not closed with '---'")
    meta = yaml.safe_load(header) or {}
    if not isinstance(meta, dict) or "id" not in meta or "version" not in meta:
        raise PromptFormatError(f"{source}: header needs 'id' and 'version'")
    return PromptTemplate(
        id=str(meta["id"]),
        version=int(meta["version"]),
        body=body,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def load_prompt(name: str, prompts_dir: Path = PROMPTS_DIR) -> PromptTemplate:
    for suffix in TEMPLATE_SUFFIXES:
        path = prompts_dir / f"{name}{suffix}"
        if path.exists():
            return parse_prompt(path.read_text(encoding="utf-8"), str(path))
    raise FileNotFoundError(f"no prompt template {name!r} in {prompts_dir}")


def prompt_hashes(prompts_dir: Path = PROMPTS_DIR) -> dict[str, str]:
    """``{"<id>@v<version>": sha256}`` for every template, for the run manifest."""
    hashes: dict[str, str] = {}
    for path in sorted(prompts_dir.iterdir()):
        if path.suffix in TEMPLATE_SUFFIXES and path.name != "README.md":
            template = parse_prompt(path.read_text(encoding="utf-8"), str(path))
            hashes[f"{template.id}@v{template.version}"] = template.sha256
    return hashes
