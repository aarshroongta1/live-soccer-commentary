"""The curve: what turns one number on a beat into a way of speaking.

Every beat has carried an ``excitement`` since the phrasing stage landed and
nothing read it, so a goal went out at the same library defaults as a
throw-in. What is checked here is the arithmetic and the switches — that the
line between the two ends is a line, that it clamps, that the environment can
move every point on it, and that ``VOICE_CURVE=off`` really does send
nothing. Whether the numbers are the right numbers is a question for a pair
of ears and ``scripts/voice_sweep.py``, not for a test.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from commentary.bus import Bus, Message, Topic
from commentary.config import VoiceConfig
from commentary.director import Director
from commentary.schemas import Beat, Voice
from commentary.voice.shaping import shape_for_voice
from commentary.voice.speaker import Utterance


@pytest.fixture(autouse=True)
def _no_inherited_curve(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's own .env must not decide what these assert.

    :class:`VoiceConfig` reads the environment when it is built, which is the
    whole point of it — a sweep sets a variable and makes a new one — and
    that cuts both ways.
    """
    for name in list(os.environ):
        if name.startswith("VOICE_"):
            monkeypatch.delenv(name, raising=False)


def caller(excitement: float) -> dict[str, float | bool]:
    settings = VoiceConfig().settings_for("caller", excitement)
    assert settings is not None
    return settings


# -- the curve ----------------------------------------------------------


def test_a_goal_and_a_throw_in_are_not_said_the_same_way() -> None:
    quiet = caller(0.0)
    loud = caller(1.0)

    # Less stability is more freedom to break pitch, which is what a goal
    # sounds like; more style is more of the voice's own performance.
    assert loud["stability"] < quiet["stability"]
    assert loud["style"] > quiet["style"]
    assert loud["speed"] > quiet["speed"]


def test_the_curve_between_the_two_ends_is_a_straight_line() -> None:
    curve = VoiceConfig()
    middle = caller(0.5)

    assert middle["stability"] == pytest.approx(
        (curve.caller.stability_low + curve.caller.stability_high) / 2
    )
    assert middle["style"] == pytest.approx((curve.caller.style_low + curve.caller.style_high) / 2)
    assert middle["speed"] == pytest.approx((curve.caller.speed_low + curve.caller.speed_high) / 2)


def test_an_excitement_off_the_end_of_the_scale_is_clamped_and_not_extrapolated() -> None:
    # Nothing produces these today, and a stability of -0.3 is a 422 from the
    # API rather than a very excited caller.
    assert caller(4.0) == caller(1.0)
    assert caller(-1.0) == caller(0.0)


def test_the_analyst_never_goes_as_loud_as_the_caller() -> None:
    curve = VoiceConfig()
    loud_caller = curve.settings_for("caller", 1.0)
    loud_analyst = curve.settings_for("analyst", 1.0)
    assert loud_caller is not None and loud_analyst is not None

    # The second seat exists in order to not sound like the first one.
    assert loud_analyst["stability"] > loud_caller["stability"]
    assert loud_analyst["style"] < loud_caller["style"]
    assert loud_analyst["speed"] < loud_caller["speed"]


def test_similarity_and_the_speaker_boost_do_not_move_with_the_excitement() -> None:
    # A caller who stops sounding like himself when he shouts is a different
    # caller, not an excited one.
    assert caller(0.0)["similarity_boost"] == caller(1.0)["similarity_boost"] == 0.8
    assert caller(0.0)["use_speaker_boost"] is True


def test_every_point_on_the_curve_can_be_moved_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VOICE_CALLER_STABILITY_LOW", "0.9")
    monkeypatch.setenv("VOICE_CALLER_STABILITY_HIGH", "0.1")
    monkeypatch.setenv("VOICE_CALLER_STYLE_HIGH", "0.75")
    monkeypatch.setenv("VOICE_CALLER_SPEED_HIGH", "1.2")
    monkeypatch.setenv("VOICE_ANALYST_STYLE_LOW", "0.05")
    monkeypatch.setenv("VOICE_SIMILARITY_BOOST", "0.55")
    monkeypatch.setenv("VOICE_SPEAKER_BOOST", "off")

    curve = VoiceConfig()
    quiet = curve.settings_for("caller", 0.0)
    loud = curve.settings_for("caller", 1.0)
    analyst = curve.settings_for("analyst", 0.0)
    assert quiet is not None and loud is not None and analyst is not None

    assert quiet["stability"] == 0.9
    assert loud["stability"] == 0.1
    assert loud["style"] == 0.75
    assert loud["speed"] == 1.2
    assert analyst["style"] == 0.05
    assert loud["similarity_boost"] == 0.55
    assert loud["use_speaker_boost"] is False


def test_the_curve_switched_off_sends_no_settings_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Not zeroed settings, and not defaults written out longhand: nothing, so
    # the voice plays exactly as it did before any of this existed.
    monkeypatch.setenv("VOICE_CURVE", "off")
    curve = VoiceConfig()

    assert curve.settings_for("caller", 1.0) is None
    assert curve.settings_for("analyst", 0.0) is None


def test_the_say_rate_rises_with_the_excitement_and_clamps_the_same_way() -> None:
    curve = VoiceConfig()

    assert curve.say_rate(0.0) == curve.say_rate_low
    assert curve.say_rate(1.0) == curve.say_rate_high
    assert curve.say_rate(0.5) == round((curve.say_rate_low + curve.say_rate_high) / 2)
    assert curve.say_rate(9.0) == curve.say_rate_high


# -- shaping ------------------------------------------------------------


def test_a_goal_line_gets_the_punctuation_of_a_goal() -> None:
    # The name is its own utterance, which is what a commentator does at the
    # moment the ball crosses the line.
    assert shape_for_voice("Mbappé, buried.", 1.0) == "Mbappé! Buried!"


def test_a_loud_line_ends_on_an_exclamation_mark_however_it_arrived() -> None:
    assert shape_for_voice("It is in.", 0.9) == "It is in!"
    assert shape_for_voice("It is in", 0.9) == "It is in!"
    assert shape_for_voice("It is in!", 0.9) == "It is in!"


def test_a_question_and_a_line_trailing_off_are_both_left_alone() -> None:
    assert shape_for_voice("How did that stay out?", 0.95) == "How did that stay out?"
    assert shape_for_voice("And he has time…", 0.95) == "And he has time…"


def test_build_up_loses_an_exclamation_mark_it_was_given() -> None:
    # A voice that shouts at a throw-in is a voice nobody believes at a goal.
    assert shape_for_voice("France push forward down the left!", 0.2) == (
        "France push forward down the left."
    )


def test_the_middle_of_the_scale_is_left_exactly_as_it_came() -> None:
    for excitement in (0.4, 0.5, 0.75, 0.84):
        assert shape_for_voice("Mbappé has the ball back.", excitement) == (
            "Mbappé has the ball back."
        )


def test_a_clause_is_not_mistaken_for_a_name() -> None:
    # Only a bare leading word before the comma is a name worth breaking on.
    assert shape_for_voice("Down the left, and it is worked inside.", 1.0) == (
        "Down the left, and it is worked inside!"
    )


def test_shaping_never_changes_a_word() -> None:
    for line, excitement in (
        ("Mbappé, buried.", 1.0),
        ("France push forward down the left!", 0.1),
        ("Álvarez, again.", 0.95),
    ):
        shaped = shape_for_voice(line, excitement)
        assert [w.strip(".!?,").lower() for w in shaped.split()] == [
            w.strip(".!?,").lower() for w in line.split()
        ]


# -- the sweep script ---------------------------------------------------


def load_sweep() -> ModuleType:
    """The sweep is a script, not a module, so it is loaded by path.

    It goes into ``sys.modules`` before it is executed because its dataclasses
    resolve their own annotations through there, and it is only executed once
    because a second copy of the same module is two of every class.
    """
    if "voice_sweep" in sys.modules:
        return sys.modules["voice_sweep"]
    path = Path(__file__).resolve().parents[1] / "scripts" / "voice_sweep.py"
    spec = importlib.util.spec_from_file_location("voice_sweep", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["voice_sweep"] = module
    spec.loader.exec_module(module)
    return module


def test_the_default_grid_is_priced_before_a_character_of_it_is_spent() -> None:
    sweep = load_sweep()
    settings = sweep.grid(
        sweep.DEFAULT_STABILITIES, sweep.DEFAULT_STYLES, sweep.DEFAULT_SPEEDS
    )
    clips = sweep.plan(["a-voice"], sweep.DEFAULT_LINES, settings)
    cost = sweep.estimate(clips, sweep.MODEL_ID)

    assert len(settings) == 12
    assert cost.clips == 12 * len(sweep.DEFAULT_LINES) == 60
    assert cost.characters == 12 * sum(len(line.text) for line in sweep.DEFAULT_LINES)
    # Flash bills at half a credit a character, so the estimate is half the
    # character count rather than equal to it.
    assert cost.credits == cost.characters / 2


def test_an_unrecognised_model_is_priced_at_the_higher_rate() -> None:
    # An estimate that comes in under is a nicer surprise than one that does not.
    sweep = load_sweep()
    assert sweep.credits_per_character("eleven_flash_v2_5") == 0.5
    assert sweep.credits_per_character("eleven_multilingual_v2") == 1.0
    assert sweep.credits_per_character("something_new") == 1.0


def test_nothing_is_rendered_without_yes(capsys: pytest.CaptureFixture[str]) -> None:
    sweep = load_sweep()
    calls: list[str] = []

    code = sweep.main(
        ["--voice-id", "a-voice"],
        fetch=lambda voice_id, body, key: calls.append(voice_id) or b"",
    )

    assert code == 2
    assert calls == [], "it spent credits it had not been given permission to spend"
    out = capsys.readouterr().out
    assert "--yes" in out
    assert "60 clips" in out and "credits" in out


def test_a_run_writes_every_clip_and_an_index_that_labels_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = load_sweep()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    sent: list[dict[str, object]] = []

    def fetch(voice_id: str, body: dict[str, object], api_key: str) -> bytes:
        sent.append(body)
        return b"ID3-not-really-an-mp3"

    code = sweep.main(
        [
            "--voice-id",
            "a-voice",
            "--stability",
            "0.2,0.5",
            "--style",
            "0.6",
            "--speed",
            "1.0",
            "--out",
            str(tmp_path),
            "--yes",
        ],
        fetch=fetch,
    )

    assert code == 0
    assert len(sent) == 2 * len(sweep.DEFAULT_LINES)
    assert len(list(tmp_path.glob("*.mp3"))) == 2 * len(sweep.DEFAULT_LINES)

    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    for clip_file in tmp_path.glob("*.mp3"):
        assert clip_file.name in index
    # Labelled by the settings that made it, or there is nothing to compare.
    assert "0.20" in index and "0.50" in index and "stability" in index
    assert "Mbappé on the spot." in index


def test_a_run_with_no_key_says_so_instead_of_writing_empty_clips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = load_sweep()
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    code = sweep.main(["--voice-id", "a-voice", "--out", str(tmp_path), "--yes"])

    assert code == 1
    assert list(tmp_path.glob("*")) == []


# -- the trace row ------------------------------------------------------


@pytest.mark.asyncio
async def test_the_settings_a_line_was_said_with_reach_the_spoken_trace_row() -> None:
    """Beside ``seconds`` and ``first_audio_s``, and for the same reason.

    The eval does not re-run the match, it reads the trace. A line that came
    out flat has to be diagnosable from the file afterwards, and that needs
    the numbers that were sent rather than the excitement they came from.
    """
    bus = Bus()
    director = Director(speaker=SaySettings(), bus=bus)
    seen: list[Message] = []

    async def watch() -> None:
        async for message in bus.subscribe():
            seen.append(message)

    watching = asyncio.create_task(watch())
    running = asyncio.create_task(director.run())
    director.submit(
        Beat(
            id="b1",
            voice=Voice.CALLER,
            text="Mbappé! Buried!",
            video_ts=0.0,
            created_ts=time.monotonic(),
            excitement=1.0,
        )
    )
    await director.drain()
    director.stop()
    for task in (running, watching):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    spoken = [m for m in seen if m.topic is Topic.SPOKEN]
    assert spoken, "the line was never reported as spoken"
    assert spoken[0].payload["voice_settings"] == {"stability": 0.2}
    # And it survives being turned into a line of the trace file.
    assert '"stability": 0.2' in spoken[0].to_json()


class SaySettings:
    """A speaker that reports settings and makes no sound."""

    async def say(self, beat: Beat, cancel: asyncio.Event) -> Utterance:
        return Utterance(
            beat=beat,
            spoken=beat.text,
            seconds=0.0,
            completed=True,
            first_audio_s=0.0,
            voice_settings={"stability": 0.2},
        )

    async def aclose(self) -> None:
        return None
