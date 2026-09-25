"""Adapter request/response mapping, with the SDK client stubbed out (no network)."""

from types import SimpleNamespace
from typing import Any

import anthropic
from pydantic import SecretStr

from juris.config import ModelSpec
from juris.llm.providers import AnthropicProvider, OpenAICompatProvider
from juris.llm.types import ChatMessage, ProviderRequest

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": False}


class Recorder:
    def __init__(self, response: Any) -> None:
        self.kwargs: dict[str, Any] = {}
        self._response = response

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        return self._response


def request(model: ModelSpec, **extra: Any) -> ProviderRequest:
    return ProviderRequest(
        model=model,
        messages=(ChatMessage(role="user", content="hi"),),
        system="be brief",
        temperature=0.3,
        seed=5,
        json_schema=SCHEMA,
        **extra,
    )


async def test_anthropic_request_mapping() -> None:
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text='{"x": "a"}'),
        ],
        usage=SimpleNamespace(input_tokens=12, output_tokens=34),
        stop_reason="end_turn",
    )
    recorder = Recorder(response)
    provider = AnthropicProvider(SecretStr("k"))
    provider._client = SimpleNamespace(messages=recorder)  # type: ignore[assignment]

    result = await provider.complete(request(ModelSpec()))

    sent = recorder.kwargs
    assert sent["model"] == "claude-opus-5-5"
    assert sent["system"] == "be brief"
    assert sent["messages"] == [{"role": "user", "content": "hi"}]
    assert sent["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    # Opus 5.5 rejects sampling params and the API has no seed.
    assert "temperature" not in sent and "seed" not in sent
    # Thinking blocks are skipped; only text is returned.
    assert result.text == '{"x": "a"}'
    assert (result.input_tokens, result.output_tokens) == (12, 34)


async def test_anthropic_omits_optional_fields() -> None:
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    recorder = Recorder(response)
    provider = AnthropicProvider(SecretStr("k"))
    provider._client = SimpleNamespace(messages=recorder)  # type: ignore[assignment]

    await provider.complete(
        ProviderRequest(
            model=ModelSpec(effort=None), messages=(ChatMessage(role="user", content="hi"),)
        )
    )
    assert recorder.kwargs["system"] is anthropic.omit
    assert recorder.kwargs["output_config"] is anthropic.omit


async def test_openai_request_mapping() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"x": "a"}'), finish_reason="stop")
        ],
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=8),
    )
    recorder = Recorder(response)
    provider = OpenAICompatProvider(SecretStr("k"))
    provider._client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=recorder)
    )

    result = await provider.complete(request(ModelSpec(provider="openai", name="some-model")))

    sent = recorder.kwargs
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
    ]
    assert sent["temperature"] == 0.3 and sent["seed"] == 5
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA
    assert result.text == '{"x": "a"}'
    assert (result.input_tokens, result.output_tokens, result.stop_reason) == (7, 8, "stop")
