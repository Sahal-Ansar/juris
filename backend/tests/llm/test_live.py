"""Live smoke test against whatever provider the default model points at.

Skipped unless that provider has a key. It works with a free or local
OpenAI-compatible endpoint (D-007), e.g. in ``.env``:

    JURIS_MODELS__DEFAULT__PROVIDER=openai
    JURIS_MODELS__DEFAULT__NAME=<model name>
    OPENAI_BASE_URL=http://localhost:11434/v1
    OPENAI_API_KEY=<any non-empty value for a local server>

Run it explicitly with ``uv run pytest -m live``.
"""

import pytest
from pydantic import BaseModel

from juris.config import Settings
from juris.llm import ChatMessage, LLMGateway

settings = Settings()
MODEL = settings.models.default
KEYS = {"anthropic": settings.anthropic_api_key, "openai": settings.openai_api_key}

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(KEYS.get(MODEL.provider) is None, reason=f"no API key for {MODEL.provider}"),
]


class Capital(BaseModel):
    country: str
    capital: str


async def test_live_structured_output() -> None:
    gateway = LLMGateway.from_settings(settings.model_copy(update={"llm_cache": "off"}))
    result = await gateway.complete(
        [ChatMessage(role="user", content="What is the capital of India?")],
        model=MODEL,
        response_model=Capital,
        max_tokens=2000,
        tags={"agent": "smoke"},
    )

    assert result.parsed is not None
    assert "delhi" in result.parsed.capital.lower()
    assert result.usage.input_tokens > 0
    # Priced models must report a cost; local/free models have no price entry.
    if MODEL.name in settings.llm_prices:
        assert result.usage.cost_usd is not None and result.usage.cost_usd > 0
