import pytest
from pydantic import ValidationError

from commentary.llm.schema import _UNSUPPORTED, strict_schema
from commentary.schemas import AnalystLine, BoardRead, CallerLine, Event, Scene


def test_silence_is_a_valid_caller_answer():
    call = CallerLine(scene=Scene.REPLAY, event=Event.NONE, confidence=0.9, speak=False)
    assert call.line == ""
    assert call.names_read == []


def test_confidence_is_bounded():
    with pytest.raises(ValidationError):
        CallerLine(scene=Scene.LIVE_PLAY, event=Event.SHOT, confidence=1.4, speak=True)


def test_line_is_capped_short():
    with pytest.raises(ValidationError):
        CallerLine(
            scene=Scene.LIVE_PLAY, event=Event.SHOT, confidence=0.5, speak=True, line="x" * 201
        )


@pytest.mark.parametrize("model", [CallerLine, BoardRead, AnalystLine])
def test_schema_drops_constraints_the_api_rejects(model):
    """Structured outputs 400 on minimum/maxLength and friends; Pydantic emits them."""
    schema = strict_schema(model)

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        assert not _UNSUPPORTED & node.keys(), f"unsupported keyword in {node}"
        if node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False
            assert node["required"] == list(node["properties"])
        for value in node.values():
            walk(value)

    walk(schema)
