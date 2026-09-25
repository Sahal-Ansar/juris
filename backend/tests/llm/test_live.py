"""Live smoke test against the real Anthropic API. Skipped without ANTHROPIC_API_KEY.

Costs a fraction of a cent per run. Run it explicitly with ``uv run pytest -m live``.
"""

import pytest
from pydantic import BaseModel

from juris.config import Settings
from juris.llm import ChatMessage, LLMGateway

settings = Settings()

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(settings.anthropic_api_key is None, reason="ANTHROPIC_API_KEY not set"),
]


class Capital(BaseModel):
    country: str
    capital: str


async def test_anthropic_structured_output() -> None:
    gateway = LLMGateway.from_settings(settings.model_copy(update={"llm_cache": "off"}))
    result = await gateway.complete(
        [ChatMessage(role="user", content="What is the capital of India?")],
        model=settings.models.default,
        response_model=Capital,
        max_tokens=2000,
        tags={"agent": "smoke"},
    )

    assert result.parsed is not None
    assert "delhi" in result.parsed.capital.lower()
    assert result.usage.input_tokens > 0
    assert result.usage.cost_usd is not None and result.usage.cost_usd > 0
