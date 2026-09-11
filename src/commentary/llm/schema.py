"""Pydantic models to the strict JSON Schema the API's structured outputs want.

Structured outputs require every object to be closed (``additionalProperties:
false``) and every property listed in ``required``. Pydantic emits neither by
default: fields with defaults drop out of ``required``, and objects stay open.
Rather than hand-write a second copy of each schema, derive it.

Optionality survives the trip: a field with a default becomes required but
nullable, so the model must decide and say so instead of quietly omitting it.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

#: Validation keywords the structured-outputs subset rejects outright — a
#: schema carrying any of them is a 400, e.g. ``For 'number' type, properties
#: maximum, minimum are not supported`` for ``confidence: float = Field(ge=0,
#: le=1)``. They are dropped from what we send, not from the models: the
#: backend validates the response with ``model_validate`` (see
#: ``anthropic_backend.parse``), so the bounds are still enforced, just on our
#: side of the wire.
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


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON Schema for ``model``, closed, fully required, no constraints."""
    schema = model.model_json_schema()
    _tighten(schema)
    for definition in schema.get("$defs", {}).values():
        _tighten(definition)
    return schema


def _tighten(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _tighten(item)
        return
    if not isinstance(node, dict):
        return

    for key in _UNSUPPORTED & node.keys():
        del node[key]

    if node.get("type") == "object" and "properties" in node:
        properties = node["properties"]
        node["additionalProperties"] = False
        node["required"] = list(properties)
        for prop in properties.values():
            _nullable_if_defaulted(prop)

    for key, value in node.items():
        if key != "properties":
            _tighten(value)
        else:
            for prop in value.values():
                _tighten(prop)


def _nullable_if_defaulted(prop: Any) -> None:
    """A field the model may legitimately not know becomes explicitly nullable."""
    if not isinstance(prop, dict) or "default" not in prop:
        return
    if prop["default"] is not None:
        return
    if "type" in prop and prop["type"] != "null":
        prop["type"] = [prop["type"], "null"]
