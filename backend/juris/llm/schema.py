"""Turn a Pydantic model into a JSON schema that structured-output APIs accept."""

from typing import Any

from pydantic import BaseModel

# Constraints the Claude structured-output API does not accept. They are dropped
# from the schema sent to the provider; Pydantic still enforces them on the result.
_UNSUPPORTED = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def output_schema(model: type[BaseModel]) -> dict[str, Any]:
    result = _clean(model.model_json_schema())
    assert isinstance(result, dict)
    return result


def _clean(node: Any) -> Any:
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {key: _clean(value) for key, value in node.items() if key not in _UNSUPPORTED}
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
    return out
