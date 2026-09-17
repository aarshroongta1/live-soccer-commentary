import numpy as np
import pytest

from commentary.agents.caller import Caller, RecentLines, clean_line, similarity, trim_words
from commentary.capture.buffer import DelayBuffer, Frame
from commentary.config import CALLER_MODEL, CallerConfig
from commentary.llm.base import Block, LLMError
from commentary.llm.fake import ScriptedBackend
from commentary.schemas import (
    Action,
    ActionBeat,
    CallerLine,
    Event,
    IdentitySource,
    PlayerIdentity,
    Scene,
    Side,
    Trigger,
)

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

    result = await caller.call(buffer_with(), "Arsenal 0-0 Chelsea, 12:04", [Trigger.CAMERA_CUT])

    assert result is not None and result.speak
    calls = backend.calls_tagged("caller")
    assert len(calls) == 1
    assert calls[0].images == CONFIG.frames_at_cursor + CONFIG.frames_lookahead
    assert calls[0].model == CALLER_MODEL
    assert calls[0].output_format is CallerLine
    assert "SECONDS AFTER" in calls[0].text
    assert "Arsenal 0-0 Chelsea, 12:04" in calls[0].text
    assert "camera_cut" in calls[0].text


async def test_one_window_can_return_multiple_chronological_action_beats():
    proposed = line("Koundé receives, plays inside, then crosses.")
    proposed.actions = [
        ActionBeat(
            frame_index=7,
            action=Action.CROSS,
            actor=PlayerIdentity(
                name="Jules Koundé",
                number=23,
                side=Side.AWAY,
                confidence=0.94,
                source=IdentitySource.SHIRT_NUMBER,
            ),
            confidence=0.96,
        ),
        ActionBeat(frame_index=2, action=Action.RECEIVE, confidence=0.9),
        ActionBeat(frame_index=4, action=Action.PASS, confidence=0.92),
    ]
    buf = buffer_with()
    cursor_frames = buf.at_cursor(CONFIG.frames_at_cursor, CONFIG.cursor_spacing_s)
    caller = Caller(backend_saying(proposed), CONFIG)

    result = await caller.call(buf, "0-0", [])

    assert result is not None
    assert [beat.action for beat in result.actions] == [
        Action.RECEIVE,
        Action.PASS,
        Action.CROSS,
    ]
    assert [beat.video_ts for beat in result.actions] == [
        cursor_frames[1].ts,
        cursor_frames[3].ts,
        cursor_frames[6].ts,
    ]


async def test_an_unframed_or_invalid_action_uses_the_exact_call_cursor():
    proposed = line("A clear pass by an unknown midfielder.")
    proposed.actions = [
        ActionBeat(frame_index=None, action=Action.PASS, confidence=0.95),
        ActionBeat(frame_index=99, action=Action.RECEIVE, confidence=0.8),
    ]
    caller = Caller(backend_saying(proposed), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [], cursor_ts=5.25)

    assert result is not None
    assert [beat.video_ts for beat in result.actions] == [5.25, 5.25]
    assert [beat.frame_index for beat in result.actions] == [None, None]


async def test_low_identity_confidence_does_not_erase_a_clear_action():
    proposed = line("The right-back crosses low.")
    proposed.actions = [
        ActionBeat(
            frame_index=8,
            action=Action.CROSS,
            actor=PlayerIdentity(
                name=None,
                role="right-back",
                side=Side.AWAY,
                confidence=0.3,
                source=IdentitySource.VISIBLE_ROLE,
                evidence="role and away kit are visible; no number is legible",
            ),
            delivery="low",
            confidence=0.95,
            evidence="the ball travels low across the penalty area",
        )
    ]
    caller = Caller(backend_saying(proposed), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None and result.speak
    assert len(result.actions) == 1
    assert result.actions[0].action is Action.CROSS
    assert result.actions[0].confidence == pytest.approx(0.95)
    assert result.actions[0].actor is not None
    assert result.actions[0].actor.name is None


def test_the_prompt_asks_for_actions_before_prose_and_keeps_evidence_separate():
    caller = Caller(ScriptedBackend(), CONFIG)
    system = " ".join(caller.system.split())

    assert "Fill actions BEFORE writing the line" in system
    assert "A window may contain several beats" in system
    assert "their OWN confidence" in system
    assert "Do not create action beats from THE NEAR FUTURE" in system


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
    """The scene is no longer the refusal; the words are, and not here.

    This agent used to veto every replay form before the gate saw it, and
    that is why ``runs/trigger/mbappe/file-20260913-185228.jsonl`` runs from
    12.9 s to 49.0 s without a word: four accurate replay lines died in this
    method. A replay form now comes back with ``speak`` intact and no
    suppression recorded, and whether it may be said is
    :class:`commentary.gate.FactGate`'s question — it can read the line,
    which this agent cannot.
    """
    caller = Caller(backend_saying(line(scene=Scene.REPLAY)), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert result.speak is True
    assert result.scene is Scene.REPLAY
    assert caller.suppressed["replay"] == 0
    assert caller.last_reason == ""


async def test_a_replay_the_model_passes_over_is_still_silence():
    """The model keeps the other half of the decision: nothing new to show."""
    caller = Caller(backend_saying(line("", scene=Scene.REPLAY, speak=False)), CONFIG)

    result = await caller.call(buffer_with(), "0-0", [])

    assert result is not None
    assert result.speak is False
    assert caller.last_reason == "the model chose silence"


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


def test_gate_only_remembers_the_configured_number_of_lines():
    config = CallerConfig(recent_lines=2)
    gate = RecentLines(config)
    gate.accept("one shot from Saka")
    gate.accept("two headers from Havertz")
    gate.accept("three corners for Chelsea")

    assert gate.recent == ["two headers from Havertz", "three corners for Chelsea"]


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


def test_the_caller_asks_for_low_effort():
    """Opus 5 thinks by default and the caller answers against an 8 s timeout.

    Thinking is not disabled — without it Opus 5 writes tool calls into
    visible text — so effort is the only lever.
    """
    from commentary.agents.caller import Caller
    from commentary.llm.fake import ScriptedBackend

    assert Caller(ScriptedBackend()).effort == "low"
