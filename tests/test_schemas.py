import pytest
from pydantic import ValidationError

from commentary.schemas import CallerLine, Event, Scene


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
