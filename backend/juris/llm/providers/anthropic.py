"""Claude via the official Anthropic SDK."""

import anthropic
from anthropic.types import MessageParam, OutputConfigParam
from pydantic import SecretStr

from juris.llm.errors import ProviderError, TransientProviderError
from juris.llm.types import ProviderRequest, ProviderResponse


class AnthropicProvider:
    """Structured output uses ``output_config.format`` (JSON schema).

    Current Claude models reject sampling parameters and forced tool use, and the
    API has no seed, so ``temperature`` and ``seed`` are not sent. The seed still
    distinguishes cache entries, which is how independent jurors get fresh samples.
    """

    name = "anthropic"

    def __init__(self, api_key: SecretStr | None) -> None:
        self._api_key = api_key
        self._client: anthropic.AsyncAnthropic | None = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            if self._api_key is None:
                raise ProviderError("ANTHROPIC_API_KEY is not set")
            # Retries happen in the gateway, so the SDK must not retry as well.
            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key.get_secret_value(), max_retries=0
            )
        return self._client

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        client = self._get_client()
        messages: list[MessageParam] = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        output_config: OutputConfigParam = {}
        if request.model.effort is not None:
            output_config["effort"] = request.model.effort
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}

        try:
            response = await client.messages.create(
                model=request.model.name,
                max_tokens=request.max_tokens,
                messages=messages,
                system=request.system if request.system is not None else anthropic.omit,
                output_config=output_config or anthropic.omit,
            )
        except anthropic.RateLimitError as exc:
            raise TransientProviderError(
                f"rate limited: {exc.message}", _retry_after(exc.response.headers)
            ) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise TransientProviderError(f"server error {exc.status_code}") from exc
            raise ProviderError(f"{exc.status_code}: {exc.message}") from exc
        except (anthropic.APITimeoutError, anthropic.APIConnectionError) as exc:
            raise TransientProviderError(f"connection problem: {exc}") from exc

        text = "".join(block.text for block in response.content if block.type == "text")
        return ProviderResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=response.stop_reason,
        )


def _retry_after(headers: object) -> float | None:
    value = getattr(headers, "get", lambda _key: None)("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None
