"""Any OpenAI-compatible chat-completions endpoint (OpenAI, a local server, a gateway)."""

from typing import Any, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import SecretStr

from juris.llm.errors import ProviderError, TransientProviderError
from juris.llm.types import ProviderRequest, ProviderResponse


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, api_key: SecretStr | None, base_url: str | None = None) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._client: openai.AsyncOpenAI | None = None

    def _get_client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            if self._api_key is None:
                raise ProviderError("OPENAI_API_KEY is not set")
            self._client = openai.AsyncOpenAI(
                api_key=self._api_key.get_secret_value(), base_url=self._base_url, max_retries=0
            )
        return self._client

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if request.system is not None:
            messages.append({"role": "system", "content": request.system})
        messages += [{"role": m.role, "content": m.content} for m in request.messages]
        kwargs: dict[str, Any] = {}
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.seed is not None:
            kwargs["seed"] = request.seed
        if request.json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": request.json_schema, "strict": False},
            }

        try:
            response = await client.chat.completions.create(
                model=request.model.name,
                messages=cast(list[ChatCompletionMessageParam], messages),
                max_tokens=request.max_tokens,
                **kwargs,
            )
        except openai.RateLimitError as exc:
            raise TransientProviderError(f"rate limited: {exc.message}") from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientProviderError(f"server error {exc.status_code}") from exc
            raise ProviderError(f"{exc.status_code}: {exc.message}") from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise TransientProviderError(f"connection problem: {exc}") from exc

        choice = response.choices[0]
        usage = response.usage
        return ProviderResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            stop_reason=choice.finish_reason,
        )
