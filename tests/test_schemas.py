"""Strict structured-output schemas for the retained recorded workflow."""

import pytest
from pydantic import BaseModel, ValidationError

from commentary.llm.schema import _UNSUPPORTED, strict_schema
from commentary.recorded_demo import JointScript, ObservedActor, ObserverResult
from commentary.recorded_research import ResearchDraft


def test_observed_actor_requires_its_declared_readable_identity() -> None:
    with pytest.raises(ValidationError, match="shirt_number source requires shirt_number"):
        ObservedActor(
            side="home",
            identity_source="shirt_number",
            action_role="passer",
        )


def test_schema_references_have_no_sibling_keywords() -> None:
    schema = strict_schema(ObserverResult)

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            if "$ref" in node:
                assert list(node) == ["$ref"]
            for value in node.values():
                walk(value)

    walk(schema)


@pytest.mark.parametrize("model", [ObserverResult, JointScript, ResearchDraft])
def test_schema_drops_api_unsupported_constraints_and_closes_objects(
    model: type[BaseModel],
) -> None:
    """Pydantic keeps validation constraints; the API schema omits rejected keywords."""
    schema = strict_schema(model)

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            assert not _UNSUPPORTED & node.keys(), f"unsupported keyword in {node}"
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
            for value in node.values():
                walk(value)

    walk(schema)
