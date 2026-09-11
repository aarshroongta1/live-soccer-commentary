import numpy as np
import pytest

from commentary.agents.caller import Caller, RepetitionGate, clean_line, similarity, trim_words
from commentary.capture.buffer import DelayBuffer, Frame
from commentary.config import CALLER_MODEL, CallerConfig
from commentary.llm.base import Block, LLMError
from commentary.llm.fake import ScriptedBackend
from commentary.schemas import CallerLine, Event, Scene, Side, Trigger

CONFIG = CallerConfig()


def buffer_with(seconds: float = 16.0, fps: int = 15, delay_s: float = 8.0) -> DelayBuffer:
    buf = DelayBuffer(fps=fps, delay_s=delay_s)
    for i in range(int(fps * seconds)):
        image = np.full((90, 160, 3), (i * 3) % 256, dtype=np.uint8)
        buf.append(Frame(ts=i / fps, image=image))
    return buf


def line(
    text: str = "Saka drives at the full-back and wins the corner.",
    *,
    scene: Scene = Scene.LIVE_PLAY,
    confidence: float = 0.8,
    speak: bool = True,
) -> CallerLine:
    return CallerLine(
        scene=scene,
        event=Event.BUILD_UP,
        side=Side.HOME,
        team="Arsenal",
        confidence=confidence,
        speak=speak,
        line=text,
    )


def backend_saying(*lines: CallerLine) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("caller", list(lines))
    return backend


async def test_the_model_is_shown_the_cursor_frames_and_the_lookahead():
    backend = backend_saying(line())
    caller = Caller(backend, CONFIG)

    result = await caller.call(buffer_with(), "Arsenal 0-0 Chelsea, 12:04", [Trigger.ROAR])

    assert result is not None and result.speak
    calls = backend.calls_tagged("caller")
    assert len(calls) == 1
    assert calls[0].images == CONFIG.frames_at_cursor + CONFIG.frames_lookahead
    assert calls[0].model == CALLER_MODEL
    assert calls[0].output_format is CallerLine
    assert "SECONDS AFTER" in calls[0].text
    assert "Arsenal 0-0 Chelsea, 12:04" in calls[0].text
    assert "roar" in calls[0].text


async def test_the_system_prompt_is_the_same_object_on_every_call():
    backend = backend_saying(line("A shot from distance."), line("A corner to Arsenal."))
    caller = Caller(backend, CONFIG)
    buf = buffer_with()

    await caller.call(buf, "0-0", [])
    await caller.call(buf, "0-0", [])

    first, second = backend.calls_tagged("caller")
    assert first.system == second.system == caller.system


async def test_a_long_line_is_trimmed_to_the_word_cap():
    long_line = (
        "Arsenal are moving it left and right and left again as they look for a way "
        "through the Chelsea block that has sat deep all day and shows no sign of coming out"
    )
    assert len(long_line.split()) > CONFIG.max_words
    caller = Caller(backend_saying(line(long_line)), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert len(result.line.split()) == CONFIG.max_words
    assert result.speak


async def test_low_confidence_forces_silence():
    caller = Caller(backend_saying(line(confidence=CONFIG.min_confidence - 0.05)), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert result.speak is False
    assert caller.suppressed["low_confidence"] == 1
    assert "confidence" in caller.last_reason


async def test_a_replay_is_never_called_as_live():
    caller = Caller(backend_saying(line(scene=Scene.REPLAY)), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert result.speak is False
    assert caller.suppressed["replay"] == 1


async def test_an_empty_line_with_speak_set_is_suppressed():
    caller = Caller(backend_saying(line("   ")), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert result.speak is False
    assert caller.suppressed["empty"] == 1


async def test_preamble_and_quotes_are_stripped():
    caller = Caller(backend_saying(line('Commentary: "Saka wins the corner."')), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert result.line == "Saka wins the corner."


async def test_model_silence_is_left_alone_and_not_remembered():
    caller = Caller(backend_saying(line("", speak=False)), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert result.speak is False
    assert caller.gate.recent == []
    assert caller.last_reason == "the model chose silence"


async def test_a_near_repeat_is_suppressed_on_the_next_call():
    backend = backend_saying(
        line("Arsenal pushing forward on the left"),
        line("Arsenal push forward down the left"),
    )
    caller = Caller(backend, CONFIG)
    buf = buffer_with()

    first = await caller.call(buf, "0-0", [])
    second = await caller.call(buf, "0-0", [])

    assert first is not None and first.speak
    assert second is not None and second.speak is False
    assert caller.suppressed["repetition"] == 1
    assert caller.last_similarity > CONFIG.repetition_threshold


async def test_spoken_lines_are_shown_back_to_the_model():
    backend = backend_saying(
        line("Arsenal pushing forward on the left"),
        line("Sanchez claims the cross at the near post"),
    )
    caller = Caller(backend, CONFIG)
    buf = buffer_with()

    await caller.call(buf, "0-0", [])
    second = await caller.call(buf, "0-0", [])

    assert second is not None and second.speak
    assert "Arsenal pushing forward on the left" in backend.calls_tagged("caller")[1].text


async def test_a_failed_model_call_is_a_missed_line_not_a_crash():
    def explode(_blocks: list[Block], _fmt: type) -> CallerLine:
        raise LLMError("timed out")

    backend = ScriptedBackend()
    backend.register("caller", explode)
    caller = Caller(backend, CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is None
    assert caller.suppressed["llm_error"] == 1
    assert len(backend.calls_tagged("caller")) == 1


async def test_an_unscripted_backend_also_returns_none():
    caller = Caller(ScriptedBackend(), CONFIG)
    assert await caller.call(buffer_with(), "0-0", []) is None


async def test_an_empty_buffer_makes_no_model_call():
    backend = backend_saying(line())
    caller = Caller(backend, CONFIG)

    result = await caller.call(DelayBuffer(fps=15, delay_s=8.0), "0-0", [])

    assert result is None
    assert backend.calls_tagged("caller") == []
    assert caller.suppressed["no_frames"] == 1


def test_gate_rejects_the_near_miss_and_accepts_a_new_line():
    gate = RepetitionGate(CONFIG)
    gate.accept("Arsenal pushing forward on the left")

    allowed, score = gate.judge("Arsenal push forward down the left")
    assert not allowed
    assert score > CONFIG.repetition_threshold

    allowed, score = gate.judge("Sanchez turns it round the post")
    assert allowed
    assert score < CONFIG.repetition_threshold


def test_gate_does_not_confuse_two_different_teams_doing_the_same_thing():
    gate = RepetitionGate(CONFIG)
    gate.accept("Arsenal break down the right")
    allowed, _ = gate.judge("Chelsea break down the left")
    assert allowed


def test_gate_only_remembers_the_configured_number_of_lines():
    config = CallerConfig(recent_lines=2)
    gate = RepetitionGate(config)
    gate.accept("one shot from Saka")
    gate.accept("two headers from Havertz")
    gate.accept("three corners for Chelsea")

    assert gate.recent == ["two headers from Havertz", "three corners for Chelsea"]
    assert gate.judge("one shot from Saka")[0]


def test_gate_never_passes_an_empty_line():
    assert RepetitionGate(CONFIG).judge("   ") == (False, 0.0)


def test_similarity_is_symmetric_and_bounded():
    a, b = "Saka wins the corner", "The corner is won by Saka"
    assert similarity(a, b) == pytest.approx(similarity(b, a))
    assert 0.0 <= similarity(a, b) <= 1.0
    assert similarity(a, a) == pytest.approx(1.0)


def test_clean_line_leaves_a_clean_line_alone():
    text = "Saka drives at the full-back and wins the corner."
    assert clean_line(text) == text


def test_trim_words_does_not_leave_a_dangling_comma():
    assert trim_words("one two three, four five", 3) == "one two three"
