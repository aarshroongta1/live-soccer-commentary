"""The phrasing stage: the caller's form, said the way a commentator says it.

The thing being protected here is not the register — a prompt cannot be
unit-tested into sounding right, and what it sounds like is in
``runs/rephrased/``. It is everything around the register: that the gate sees
what is actually going to the speaker, that the stage can be switched off and
leave nothing behind, and that a phraser which falls over costs a rewrite and
never a line.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from commentary.agents.phraser import Phraser, phraser_enabled
from commentary.capture.buffer import Frame
from commentary.config import (
    CallerConfig,
    CaptureConfig,
    DirectorConfig,
    PhraserConfig,
    PredictorConfig,
    Settings,
)
from commentary.llm.base import Block
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.commentary_examples import EXAMPLES, KINDS
from commentary.prompts.phraser import phraser_blocks, phraser_system
from commentary.rephrase import rephrase
from commentary.runtime import Runtime
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    KnowledgePack,
    PhrasedLine,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
    Trigger,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.trace import read_trace
from commentary.voice import LogSpeaker

# -- fixtures ----------------------------------------------------------------


def a_form(
    line: str = "Molina moves forward down the right and looks for a way inside.",
    *,
    event: Event = Event.CARRY,
    sightings: list[Sighting] | None = None,
) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=event,
        side=Side.HOME,
        team="Argentina",
        sightings=sightings if sightings is not None else [Sighting(number=26, name="Molina")],
        confidence=0.7,
        speak=True,
        line=line,
    )


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            demonym="Argentine",
            starters=[
                Player(name="Nahuel Molina", number=26),
                Player(name="Lionel Messi", number=10),
            ],
        ),
        away=TeamSheet(
            name="France",
            short="FRA",
            demonym="French",
            starters=[Player(name="Kylian Mbappé", number=10)],
        ),
    )


def saying(*lines: PhrasedLine) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("phraser", list(lines))
    return backend


def a_phraser(backend: ScriptedBackend) -> Phraser:
    return Phraser(
        backend,
        config=PhraserConfig(model="claude-haiku-4-5"),
        home="Argentina",
        away="France",
    )


def text_of(blocks: list[Block]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


# -- the prompt --------------------------------------------------------------


def test_the_system_prompt_carries_real_utterances_from_every_kind() -> None:
    system = phraser_system()
    for kind in KINDS:
        assert EXAMPLES[kind], f"{kind} has no examples"
        assert EXAMPLES[kind][0] in system, f"{kind} is not shown in the prompt"


def test_the_system_prompt_is_the_same_bytes_every_time() -> None:
    assert phraser_system() == phraser_system()


def test_the_call_shows_the_form_the_state_and_what_was_just_said() -> None:
    body = text_of(
        phraser_blocks(
            a_form(),
            "Argentina 2-0 France\n78:09",
            ["Fernández.", "Now De Paul."],
            home="Argentina",
            away="France",
        )
    )
    assert "Molina" in body
    assert "carry" in body
    assert "Argentina 2-0 France" in body
    assert "Now De Paul." in body
    assert "Molina moves forward down the right" in body


def test_the_call_carries_no_pictures() -> None:
    """A phraser with a frame describes the frame, which is the bug."""
    blocks = phraser_blocks(a_form(), "", [], home="Argentina", away="France")
    assert all(block.get("type") == "text" for block in blocks)


def test_a_form_that_named_nobody_says_so_rather_than_saying_nothing() -> None:
    body = text_of(
        phraser_blocks(
            a_form(sightings=[]), "", [], home="Argentina", away="France"
        )
    )
    assert "none read off the picture" in body


def test_the_name_carried_from_the_last_line_is_offered_as_usable() -> None:
    body = text_of(
        phraser_blocks(
            a_form(sightings=[]),
            "",
            [],
            home="Argentina",
            away="France",
            on_the_ball="Messi",
        )
    )
    assert "Messi" in body


def test_every_example_is_shorter_than_a_commentator_ever_goes() -> None:
    """Twenty-eight words was the longest thing said in half an hour."""
    for kind in KINDS:
        for text in EXAMPLES[kind]:
            assert len(text.split()) < 30, f"{kind}: {text!r}"
            assert text.strip() == text


def test_the_example_set_is_big_enough_to_be_a_register_and_small_enough_to_cache() -> None:
    total = sum(len(EXAMPLES[kind]) for kind in KINDS)
    assert 150 <= total <= 300, total


# -- the agent ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_phraser_returns_the_line_the_model_wrote() -> None:
    phraser = a_phraser(saying(PhrasedLine(line="Molina, forward.", excitement=0.3)))
    phrased = await phraser.phrase(a_form(), "")
    assert phrased is not None
    assert phrased.line == "Molina, forward."
    assert phrased.excitement == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_a_phrased_line_is_stripped_and_capped_like_the_callers() -> None:
    phraser = a_phraser(saying(PhrasedLine(line='"Commentary: Molina."', excitement=0.2)))
    phrased = await phraser.phrase(a_form(), "")
    assert phrased is not None
    assert phrased.line == "Molina."


@pytest.mark.asyncio
async def test_a_failed_call_is_a_reason_not_an_exception() -> None:
    phraser = a_phraser(ScriptedBackend())  # no handler registered for "phraser"
    assert await phraser.phrase(a_form(), "") is None
    assert "model call failed" in phraser.last_reason


@pytest.mark.asyncio
async def test_what_was_said_is_shown_back_so_the_voice_does_not_repeat_itself() -> None:
    backend = saying(PhrasedLine(line="Molina.", excitement=0.2))
    phraser = a_phraser(backend)
    phraser.accept("Fernández.")
    await phraser.phrase(a_form(), "")
    assert "Fernández." in backend.calls_tagged("phraser")[0].text


def test_the_stage_is_off_when_the_model_says_off() -> None:
    assert phraser_enabled("claude-haiku-4-5")
    assert not phraser_enabled("off")
    assert not phraser_enabled("OFF")
    assert not phraser_enabled("")


# -- the runtime -------------------------------------------------------------


def fast_settings(**kw: Any) -> Settings:
    return Settings(
        capture=CaptureConfig(
            width=640, height=360, fps=8, delay_s=4.0, history_s=3.0, present_offset_s=3.0
        ),
        caller=CallerConfig(min_gap_s=2.0),
        predictor=PredictorConfig(tick_s=0.05),
        director=DirectorConfig(max_beat_age_s=30.0),
        **kw,
    )


def a_runtime(settings: Settings | None = None) -> Runtime:
    """A runtime holding the sim's teams, with frames in the buffer and no clock.

    Built rather than run. ``Runtime.run`` stops on a wall-clock deadline
    while the sim source pours frames in as fast as the machine will take
    them, so two runs of the same seed with the same settings produce
    different commentary — measured at nineteen beats against nine. That is
    fine for the tests that ask "did anything reasonable happen" and useless
    for one that asks "is this identical", so here the frames go in by hand
    and the calls are made one at a time.
    """
    sim = MatchSim(seed=5, duration_s=120.0)
    chosen = settings or fast_settings()
    runtime = Runtime(
        source=SimSource(sim, chosen.capture, realtime=False),
        backend=SimOracle(sim=sim),
        pack=sim.knowledge_pack,
        settings=chosen,
        speaker=LogSpeaker(words_per_second=120),
    )
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for index in range(96):
        runtime.buffer.append(Frame(ts=index / chosen.capture.fps, image=blank))
    return runtime


def caught_beats(runtime: Runtime) -> list[Beat]:
    """Wrap the director so a test can see what it was handed."""
    beats: list[Beat] = []
    original = runtime.director.submit

    def spy(beat: Beat) -> None:
        beats.append(beat)
        original(beat)

    runtime.director.submit = spy  # type: ignore[method-assign]
    return beats


def caught_gate(runtime: Runtime) -> list[CallerLine]:
    judged: list[CallerLine] = []
    original = runtime.gate.judge

    def spy(line: CallerLine, *args: Any, **kw: Any) -> Any:
        judged.append(line)
        return original(line, *args, **kw)

    runtime.gate.judge = spy  # type: ignore[method-assign]
    return judged


def speaking(runtime: Runtime, line: CallerLine) -> None:
    """Make the caller return this form and nothing else."""

    async def call(*_args: Any, **_kw: Any) -> CallerLine:
        return line

    runtime.caller.call = call  # type: ignore[method-assign]


@pytest.mark.asyncio
async def test_the_gate_judges_the_phrased_line_not_the_callers() -> None:
    runtime = a_runtime()
    assert runtime.pack is not None
    player = runtime.pack.home.starters[0]
    form = a_form(
        line=f"{player.surname} carries it forward past two challenges towards the box.",
        sightings=[Sighting(number=player.number, name=player.name, side=Side.HOME)],
    )
    speaking(runtime, form)
    runtime.phraser = a_phraser(
        saying(PhrasedLine(line=f"{player.surname}, away.", excitement=0.4))
    )
    judged, beats = caught_gate(runtime), caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert [line.line for line in judged] == [f"{player.surname}, away."]
    assert [beat.text for beat in beats] == [f"{player.surname}, away."]
    assert beats[0].excitement == pytest.approx(0.4)


@pytest.mark.asyncio
async def test_a_phraser_that_fails_costs_a_rewrite_and_never_a_line() -> None:
    runtime = a_runtime()
    assert runtime.pack is not None
    player = runtime.pack.home.starters[0]
    form = a_form(
        line=f"{player.surname} carries it forward past two challenges towards the box.",
        sightings=[Sighting(number=player.number, name=player.name, side=Side.HOME)],
    )
    speaking(runtime, form)
    runtime.phraser = a_phraser(ScriptedBackend())
    beats = caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert [beat.text for beat in beats] == [form.line]
    assert beats[0].excitement == 0.0


@pytest.mark.asyncio
async def test_an_empty_phrasing_falls_back_rather_than_dropping_the_line() -> None:
    runtime = a_runtime()
    assert runtime.pack is not None
    player = runtime.pack.home.starters[0]
    form = a_form(
        line=f"{player.surname} carries it forward past two challenges towards the box.",
        sightings=[Sighting(number=player.number, name=player.name, side=Side.HOME)],
    )
    speaking(runtime, form)
    runtime.phraser = a_phraser(saying(PhrasedLine(line="   ", excitement=0.9)))
    beats = caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert [beat.text for beat in beats] == [form.line]


@pytest.mark.asyncio
async def test_the_stage_switched_off_leaves_the_runtime_exactly_as_it_was() -> None:
    """``PHRASER_MODEL=off`` is not a degraded mode; it is the old runtime.

    The same sim forms go through two runtimes that differ only in the
    setting. The one with the stage on has the sim's oracle behind it, and
    the oracle answers by decoding a timestamp out of a picture — the phraser
    sends no pictures — so every call fails and every line falls back. That
    is the second half of what this asserts: a stage that is absent and a
    stage that has fallen over produce the same commentary, down to the
    event, the excitement and whether the director may cut it off.
    """
    arms: list[list[tuple[str, Event, float, bool]]] = []
    for model in ("off", "claude-haiku-4-5"):
        runtime = a_runtime(fast_settings(phraser=PhraserConfig(model=model)))
        assert (runtime.phraser is None) == (model == "off")
        beats = caught_beats(runtime)
        for form in sim_forms(runtime):
            speaking(runtime, form)
            await runtime._call([Trigger.SCHEDULED])
        arms.append([(b.text, b.event, b.excitement, b.preemptable) for b in beats])

    assert arms[0], "nothing was spoken, so nothing was compared"
    assert arms[0] == arms[1]


def sim_forms(runtime: Runtime) -> list[CallerLine]:
    """A handful of caller forms off the sim's own team sheets."""
    assert runtime.pack is not None
    home = runtime.pack.home.starters[0]
    away = runtime.pack.away.starters[0]
    return [
        a_form(
            line=f"{home.surname} collects it inside his own half and turns upfield.",
            sightings=[Sighting(number=home.number, name=home.name, side=Side.HOME)],
        ),
        a_form(
            line=f"{away.surname} closes him down and wins it back near the touchline.",
            event=Event.TACKLE,
            sightings=[Sighting(number=away.number, name=away.name, side=Side.AWAY)],
        ),
        a_form(
            line=f"{home.surname} is away again down the right with space to run into.",
            sightings=[Sighting(number=home.number, name=home.name, side=Side.HOME)],
        ),
    ]


# -- rephrase ----------------------------------------------------------------


def a_trace() -> list[dict[str, Any]]:
    """Four rows of a run: one state, one spoken caller line, one analyst beat."""
    return [
        {
            "topic": "state",
            "ts": 0.5,
            "home": "Argentina",
            "away": "France",
            "home_score": 1,
            "away_score": 0,
            "clock": "78:09",
            "period": 2,
        },
        {"topic": "board", "ts": 1.0, "bug_visible": True, "confidence": 0.9},
        {
            "topic": "caller",
            "ts": 10.0,
            "scene": "live_play",
            "event": "carry",
            "side": "home",
            "team": "Argentina",
            "sightings": [{"number": 26, "name": "Molina", "side": "home"}],
            "confidence": 0.7,
            "speak": True,
            "line": "Molina drives forward down the right, looking for a way inside.",
        },
        {
            "topic": "gate",
            "ts": 10.0,
            "passed": True,
            "reasons": [],
            "line": "Molina drives forward down the right, looking for a way inside.",
            "event": "carry",
        },
        {
            "topic": "beat",
            "ts": 10.0,
            "id": "b1",
            "voice": "caller",
            "text": "Molina drives forward down the right, looking for a way inside.",
            "video_ts": 10.0,
            "created_ts": 1.0,
            "live_ts": 13.0,
            "event": "carry",
            "preemptable": True,
        },
        {
            "topic": "spoken",
            "ts": 10.0,
            "id": "b1",
            "voice": "caller",
            "text": "Molina drives forward down the right, looking for a way inside.",
            "video_ts": 10.0,
            "created_ts": 1.0,
            "spoken": "Molina drives forward down the right, looking for a way inside.",
            "seconds": 3.4,
        },
        {
            "topic": "caller",
            "ts": 20.0,
            "scene": "live_play",
            "event": "shot",
            "side": "home",
            "team": "Argentina",
            "sightings": [{"number": 10, "name": "Messi", "side": "home"}],
            "confidence": 0.8,
            "speak": True,
            "line": "Messi shapes and strikes it towards the near post from the edge.",
        },
        {
            "topic": "beat",
            "ts": 20.0,
            "id": "b2",
            "voice": "caller",
            "text": "Messi shapes and strikes it towards the near post from the edge.",
            "video_ts": 20.0,
            "created_ts": 1.0,
            "live_ts": 23.0,
            "event": "shot",
            "preemptable": True,
        },
        {
            "topic": "beat",
            "ts": 30.0,
            "id": "b3",
            "voice": "analyst",
            "text": "Argentina have found the overload down that right side all evening.",
            "video_ts": 30.0,
            "created_ts": 1.0,
            "live_ts": 33.0,
            "event": "none",
            "preemptable": True,
        },
        {"topic": "cost", "ts": 30.0, "total_usd": 0.41},
    ]


def rows_of(rows: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("topic") == topic]


@pytest.mark.asyncio
async def test_rephrasing_rewrites_the_caller_beats_and_leaves_the_analyst_alone() -> None:
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())

    beats = {row["id"]: row["text"] for row in rows_of(result.rows, "beat")}
    assert beats["b1"] == "Molina, down the right."
    assert beats["b2"] == "Messi strikes."
    assert beats["b3"] == "Argentina have found the overload down that right side all evening."


@pytest.mark.asyncio
async def test_rephrasing_drops_the_speaking_rows_and_keeps_everything_else() -> None:
    backend = saying(PhrasedLine(line="Molina, down the right.", excitement=0.3))
    result = await rephrase(a_trace(), backend, pack=a_pack())

    assert rows_of(result.rows, "spoken") == []
    assert rows_of(result.rows, "preempted") == []
    assert rows_of(result.rows, "board") == rows_of(a_trace(), "board")
    assert rows_of(result.rows, "state") == rows_of(a_trace(), "state")
    assert rows_of(result.rows, "cost") == rows_of(a_trace(), "cost")
    assert len(rows_of(result.rows, "phrased")) == 2


@pytest.mark.asyncio
async def test_a_phrased_line_that_claims_the_wrong_score_never_reaches_a_beat() -> None:
    """The gate is downstream of the phraser, and it is the same gate.

    This is the rule that came out of the first phantom goals on air: an
    ordinal is a scoreline with a number left out, and a side that has
    scored one may be said to have scored one.
    """
    backend = saying(
        PhrasedLine(line="Molina! Argentina's third.", excitement=1.0),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())

    beats = {row["id"] for row in rows_of(result.rows, "beat")}
    assert "b1" not in beats
    assert "b2" in beats

    refused = [row for row in rows_of(result.rows, "gate") if row.get("where") == "rephrase"]
    assert len(refused) == 1
    assert any("score_claim" in reason for reason in refused[0]["reasons"])
    assert result.rejected == 1


@pytest.mark.asyncio
async def test_a_rephrase_that_fails_keeps_the_line_the_run_paid_for() -> None:
    result = await rephrase(a_trace(), ScriptedBackend(), pack=a_pack())

    beats = {row["id"]: row["text"] for row in rows_of(result.rows, "beat")}
    assert beats["b1"].startswith("Molina drives forward")
    assert len(rows_of(result.rows, "error")) == 2
    assert all(line.fallback for line in result.lines)


@pytest.mark.asyncio
async def test_the_table_shows_both_versions_of_every_line() -> None:
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi strikes.", excitement=0.75),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())
    table = result.table()

    assert "Molina drives forward down the right" in table
    assert "Molina, down the right." in table
    assert "Messi strikes." in table


@pytest.mark.asyncio
async def test_a_rephrased_trace_is_a_trace(tmp_path: Path) -> None:
    """It has to load through the same reader the watch page replays from."""
    backend = saying(PhrasedLine(line="Molina, down the right.", excitement=0.3))
    result = await rephrase(a_trace(), backend, pack=a_pack())
    path = result.write(tmp_path / "phrased.jsonl")

    from commentary.replay import cues

    rows = read_trace(path)
    assert rows == result.rows
    assert any(cue.topic.value == "beat" for cue in cues(rows, delay_s=8.0))


# -- the form's event is binding ---------------------------------------------


def test_the_call_makes_the_forms_event_the_thing_the_line_must_be_about() -> None:
    """Every invented outcome on the first two traces reached past this word.

    A penalty being waited on was written as a penalty struck; a foul given
    was written as a booking. The event is in the body twice now, once in
    capitals beside the instruction and once in the line that hands over the
    description.
    """
    body = text_of(
        phraser_blocks(
            a_form(event=Event.PENALTY), "", [], home="Argentina", away="France"
        )
    )
    assert "EVENT: penalty" in body
    assert "only about a penalty" in body


def test_the_rules_forbid_promoting_the_event_and_implying_the_score() -> None:
    rules = phraser_system()
    for forbidden in ("in the book", "levels it", "the equaliser", "YOU COMPRESS"):
        assert forbidden in rules, forbidden


@pytest.mark.asyncio
async def test_the_table_shows_the_event_and_says_when_the_words_moved_it() -> None:
    backend = saying(
        PhrasedLine(line="Molina, down the right.", excitement=0.3),
        PhrasedLine(line="Messi scores.", excitement=1.0),
    )
    result = await rephrase(a_trace(), backend, pack=a_pack())

    events = [line.events for line in result.lines]
    assert events[0] == "carry"
    # The form said shot; the words say the ball went in, which is the
    # reading the director uses to decide whether it may be cut off.
    assert events[1] == "shot->goal"
    assert "shot->goal" in result.table()

    rows = [row for row in result.rows if row.get("topic") == "phrased"]
    assert [row["form_event"] for row in rows] == ["carry", "shot"]
    assert [row["event"] for row in rows] == ["carry", "goal"]
