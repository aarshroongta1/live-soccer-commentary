"""The two halves of a dead ball: nothing at all, or the long line.

``docs/research/real-commentary-corpus.md`` section 2.3 measures the phase and
finds the opposite of a fixed rate. In an attacking move the commentator
speaks half again as fast and says less each time — median gap 2.8 s, median 7
words. At a restart the gap opens to 4.5 s and the lines get *longer*: median
10 words, one in five over sixteen, because that is where the researched
clause and the storyline go. And section 3 counts how often a restart is not
called at all: 43% of goal kicks, 37% of throw-ins, 33% of free-kick
deliveries, 31% of kickoffs.

This system did neither. It chose silence once in 35 calls and wrote the
jingle: "Through the middle at speed." / "Through midfield now." / "Wide on
the right now." / "Into the corner now." — four lines with nobody in them,
three of them ending on the same word.

So the silence is decided in code before the model is asked, the tail repeat
is checked the way the opening word already was, and a restart with something
to say is told so and given the words to do it in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from commentary.agents.phraser import Phraser, nameless_build_up
from commentary.capture.buffer import Frame
from commentary.config import (
    CallerConfig,
    CaptureConfig,
    DeadBallConfig,
    DirectorConfig,
    PhraserConfig,
    PredictorConfig,
    Settings,
    SilenceConfig,
)
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.phraser import (
    LONG_LINE_EVENTS,
    is_long_line,
    looks_like_a_goal_kick,
    notes_allowed,
    phraser_blocks,
)
from commentary.rephrase import rephrase
from commentary.runtime import Runtime
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    Note,
    PhrasedLine,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
    Trigger,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.voice import LogSpeaker

# -- fixtures ---------------------------------------------------------------


def a_form(
    line: str = "Argentina work it across the halfway line with nobody closing.",
    *,
    event: Event = Event.BUILD_UP,
    sightings: list[Sighting] | None = None,
    detail: str | None = None,
    scene: Scene = Scene.LIVE_PLAY,
) -> CallerLine:
    return CallerLine(
        scene=scene,
        event=event,
        side=Side.HOME,
        team="Argentina",
        sightings=sightings if sightings is not None else [],
        confidence=0.7,
        speak=True,
        line=line,
        detail=detail,
    )


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina", short="ARG", starters=[Player(name="Nahuel Molina", number=26)]
        ),
        away=TeamSheet(
            name="France", short="FRA", starters=[Player(name="Kylian Mbappé", number=10)]
        ),
    )


def a_phraser(
    backend: ScriptedBackend | None = None, *, silence: SilenceConfig | None = None
) -> Phraser:
    return Phraser(
        backend or ScriptedBackend(),
        config=PhraserConfig(model="claude-haiku-4-5"),
        silence=silence or SilenceConfig(),
        home="Argentina",
        away="France",
    )


def saying(*lines: PhrasedLine) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("phraser", list(lines))
    return backend


def text_of(blocks: list[dict[str, Any]]) -> str:
    return "\n".join(str(b["text"]) for b in blocks if b.get("type") == "text")


# -- the predicate ----------------------------------------------------------


def test_a_form_with_nobody_and_nothing_on_it_is_nameless_build_up() -> None:
    assert nameless_build_up(a_form())
    assert nameless_build_up(a_form(event=Event.PASS))
    assert nameless_build_up(a_form(event=Event.CARRY))
    assert nameless_build_up(a_form(event=Event.NONE))


def test_a_bound_name_a_carried_name_or_a_detail_all_keep_the_call() -> None:
    assert not nameless_build_up(a_form(sightings=[Sighting(number=26, name="Molina")]))
    assert not nameless_build_up(a_form(), on_the_ball="Nahuel Molina")
    assert not nameless_build_up(a_form(detail="nutmegged on the touchline"))


def test_a_sighting_with_a_number_and_no_name_is_still_nobody() -> None:
    """A shirt the caller could not put a name to names nobody."""
    assert nameless_build_up(a_form(sightings=[Sighting(number=26)]))


def test_anything_that_is_not_build_up_is_called() -> None:
    for event in (Event.SHOT, Event.CORNER, Event.FOUL, Event.THROW_IN, Event.GOAL):
        assert not nameless_build_up(a_form(event=event)), event


def test_a_replay_is_never_passed_over() -> None:
    """It is on the screen to be talked over, and its form carries the incident."""
    assert not nameless_build_up(a_form(event=Event.NONE, scene=Scene.REPLAY))


# -- the decision, and the memory behind it ---------------------------------


def test_the_first_nameless_line_is_ordinary_commentary() -> None:
    phraser = a_phraser()
    phraser.accept("Molina drives forward.", Event.CARRY, ts=10.0, nameless=False)

    assert phraser.passes_over(a_form(), ts=12.0) is None


def test_the_second_one_running_is_the_silence() -> None:
    phraser = a_phraser()
    phraser.accept("Through midfield.", Event.BUILD_UP, ts=10.0, nameless=True)

    assert phraser.passes_over(a_form(), ts=12.0) == (
        "silence: nameless build-up after nameless build-up"
    )


def test_a_name_on_this_form_is_enough_to_be_worth_a_line() -> None:
    phraser = a_phraser()
    phraser.accept("Through midfield.", Event.BUILD_UP, ts=10.0, nameless=True)

    assert phraser.passes_over(a_form(), ts=12.0, on_the_ball="Nahuel Molina") is None


def test_a_nameless_line_from_long_ago_does_not_hold_the_voice_quiet() -> None:
    """Otherwise a quiet passage is answered with more quiet, and the system goes mute."""
    phraser = a_phraser(silence=SilenceConfig(within_s=15.0))
    phraser.accept("Through midfield.", Event.BUILD_UP, ts=10.0, nameless=True)

    assert phraser.passes_over(a_form(), ts=40.0) is None


def test_nothing_is_passed_over_with_the_rule_switched_off() -> None:
    phraser = a_phraser(silence=SilenceConfig(enabled=False))
    phraser.accept("Through midfield.", Event.BUILD_UP, ts=10.0, nameless=True)

    assert phraser.passes_over(a_form(), ts=12.0) is None


def test_an_older_caller_that_stamps_nothing_still_asks_the_model() -> None:
    """``accept`` without a timestamp is every caller of this before the rule."""
    phraser = a_phraser()
    phraser.accept("Through midfield.", Event.BUILD_UP)

    assert phraser.passes_over(a_form(), ts=12.0) is None


@pytest.mark.asyncio
async def test_the_silence_costs_no_model_call() -> None:
    backend = saying(PhrasedLine(line="Through midfield.", excitement=0.2))
    phraser = a_phraser(backend)
    phraser.accept("Through midfield.", Event.BUILD_UP, ts=10.0, nameless=True)

    assert phraser.passes_over(a_form(), ts=12.0) is not None
    assert backend.calls == []


# -- the same decision, offline ---------------------------------------------


def a_build_up_row(ts: float, line: str) -> list[dict[str, Any]]:
    """One nameless build-up call as a trace records it: a form and a beat."""
    return [
        {
            "topic": "caller",
            "ts": ts,
            "scene": "live_play",
            "event": "build_up",
            "side": "home",
            "team": "Argentina",
            "sightings": [],
            "confidence": 0.7,
            "speak": True,
            "line": line,
        },
        {
            "topic": "gate",
            "ts": ts,
            "passed": True,
            "reasons": [],
            "line": line,
            "event": "build_up",
        },
        {
            "topic": "beat",
            "ts": ts,
            "id": f"b{ts:.0f}",
            "voice": "caller",
            "text": line,
            "video_ts": ts,
            "created_ts": 1.0,
            "live_ts": ts + 3.0,
            "event": "build_up",
            "preemptable": True,
        },
    ]


@pytest.mark.asyncio
async def test_the_rephrase_passes_over_the_second_nameless_form_too() -> None:
    """Both paths ask the same object, so a silence offline is a silence live."""
    rows = [
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
        *a_build_up_row(10.0, "Argentina move it across the halfway line."),
        *a_build_up_row(14.0, "The ball goes wide with nobody closing it down."),
    ]
    backend = saying(PhrasedLine(line="Through midfield.", excitement=0.2))

    result = await rephrase(rows, backend, pack=a_pack(), colour=False)

    phrased = [row for row in result.rows if row.get("topic") == "phrased"]
    assert [row["line"] for row in phrased] == ["Through midfield.", ""]
    assert phrased[1]["reason"] == "silence: nameless build-up after nameless build-up"
    assert phrased[1]["usd"] == 0.0
    assert [line.silent for line in result.lines] == [False, True]
    # One call, not two: the second form never reached the model.
    assert len(backend.calls) == 1
    # And no second beat, because nothing was said.
    beats = [row for row in result.rows if row.get("topic") == "beat"]
    assert [row["ts"] for row in beats] == [10.0]


# -- the long line ----------------------------------------------------------


def a_note() -> Note:
    return Note(
        about="Nahuel Molina",
        text="has not been beaten one-on-one in three matches",
        kind="stat",
        checked=True,
    )


def body_for(line: CallerLine, notes: tuple[Note, ...] = ()) -> str:
    return text_of(
        phraser_blocks(
            line,
            "Argentina 1-0 France",
            [],
            home="Argentina",
            away="France",
            notes=notes,
        )
    )


def test_a_restart_with_a_clause_is_told_it_is_the_long_one() -> None:
    body = body_for(a_form("Argentina wait to take the throw.", event=Event.THROW_IN), (a_note(),))

    assert "THIS IS THE LONG ONE" in body
    assert f"{DeadBallConfig().min_words} to {DeadBallConfig().target_words} words" in body
    assert "the restart in a\nclause, then the storyline" in body


def test_a_restart_with_nothing_on_it_is_told_to_say_nothing() -> None:
    body = body_for(a_form("Argentina wait to take the throw.", event=Event.THROW_IN))

    assert "THIS IS A RESTART WITH NOTHING ON IT" in body
    assert "43% of goal kicks" in body
    assert "THIS IS THE LONG ONE" not in body


def test_a_detail_is_material_enough_for_the_long_line() -> None:
    body = body_for(
        a_form(
            "Argentina wait to take the throw.",
            event=Event.THROW_IN,
            detail="taken quickly down the line",
        )
    )

    assert "THIS IS THE LONG ONE" in body


def test_a_goalkeeper_with_the_ball_is_a_goal_kick_however_the_form_is_filed() -> None:
    """There is no ``Event.GOAL_KICK``; the caller files one as build-up."""
    assert looks_like_a_goal_kick(a_form("The goalkeeper waits to restart it."))
    assert looks_like_a_goal_kick(a_form("A goal kick for Argentina.", event=Event.NONE))
    assert not looks_like_a_goal_kick(
        a_form("The keeper gets a hand to it.", event=Event.SAVE)
    ), "a save is not a restart"

    body = body_for(a_form("The goalkeeper waits to restart it."), (a_note(),))
    assert "THIS IS THE LONG ONE" in body


def test_the_goal_kick_slot_is_offered_the_clause_as_well_as_the_length() -> None:
    """43% silent, 1% say "goal kick", and what goes there instead is the storyline.

    A form filed ``none`` is the emptiest picture the caller has a word for,
    and it is where a goalkeeper waiting to restart lands. Offered nothing to
    say, the line is "Goal kick." or silence.
    """
    form = a_form("The goalkeeper waits to restart it.", event=Event.NONE)
    assert notes_allowed(form)

    body = body_for(form, (a_note(),))
    assert "has not been beaten one-on-one in three matches" in body
    assert "THIS IS THE LONG ONE" in body


def test_a_live_ball_is_never_told_to_write_the_long_one() -> None:
    body = body_for(a_form("Molina drives at the defender.", event=Event.CARRY), (a_note(),))

    assert "THIS IS THE LONG ONE" not in body
    assert "THIS IS A RESTART" not in body


def test_the_goal_follow_up_keeps_its_own_block() -> None:
    """Two length instructions in one body is the model choosing which to obey."""
    body = text_of(
        phraser_blocks(
            a_form("The whistle goes for the restart.", event=Event.KICKOFF),
            "Argentina 1-0 France",
            [],
            home="Argentina",
            away="France",
            notes=(a_note(),),
            followup="BEAT 2 — THE CELEBRATION",
        )
    )

    assert "BEAT 2 — THE CELEBRATION" in body
    assert "THIS IS THE LONG ONE" not in body


# -- the cap that goes with it ----------------------------------------------


def test_the_word_cap_is_lifted_at_a_restart_and_after_a_goal_and_nowhere_else() -> None:
    phraser = a_phraser()
    ordinary = phraser.config.max_words
    lifted = max(ordinary, phraser.dead_ball.max_words)

    assert phraser._word_cap(a_form(event=Event.CARRY), None) == ordinary
    assert phraser._word_cap(a_form(event=Event.THROW_IN), None) == lifted
    assert phraser._word_cap(a_form(event=Event.CARRY), 4) == lifted
    assert lifted > ordinary, "a lift that lifts nothing is not a lift"


def test_every_restart_kind_the_corpus_measures_gets_the_long_line() -> None:
    for event in LONG_LINE_EVENTS:
        assert is_long_line(a_form(event=event)), event
    assert not is_long_line(a_form(event=Event.PENALTY)), "about to be the loudest moment"
    assert not is_long_line(
        a_form(event=Event.THROW_IN, scene=Scene.REPLAY)
    ), "the pictures change before a long line finishes"


# -- the tail ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_line_ending_on_a_word_one_of_the_last_five_ended_on_is_asked_again() -> None:
    backend = saying(
        PhrasedLine(line="Wide on the right now.", excitement=0.3),
        PhrasedLine(line="Out towards the touchline.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("Through midfield now.", Event.BUILD_UP, ts=10.0)

    phrased = await phraser.phrase(a_form(), "", on_the_ball="Nahuel Molina")

    assert phrased is not None
    assert phrased.line == "Out towards the touchline."
    assert phrased.closer_retry
    assert not phrased.opener_retry
    note = text_of(backend.calls[-1].blocks)
    assert 'it ENDED on "now"' in note


@pytest.mark.asyncio
async def test_the_tail_check_asks_once_and_then_takes_the_tail_off() -> None:
    """One re-ask, and what comes back with the tail again loses it in code.

    Four of twenty-one lines on the offside clip ended a clause on "now", and
    three of the four end on another word entirely — "Through midfield now,
    halfway line reached." — so the closer check never saw them.
    """
    backend = saying(
        PhrasedLine(line="Wide on the right now.", excitement=0.3),
        PhrasedLine(line="Into the corner now.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("Through midfield now.", Event.BUILD_UP, ts=10.0)

    phrased = await phraser.phrase(a_form(), "", on_the_ball="Nahuel Molina")

    assert phrased is not None
    assert phrased.line == "Into the corner."
    assert phrased.now_stripped
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_one_re_ask_covers_both_ends_of_the_sentence() -> None:
    backend = saying(
        PhrasedLine(line="Through the middle now.", excitement=0.3),
        PhrasedLine(line="Out towards the touchline.", excitement=0.3),
    )
    phraser = a_phraser(backend)
    phraser.accept("Through midfield now.", Event.BUILD_UP, ts=10.0)

    phrased = await phraser.phrase(a_form(), "", on_the_ball="Nahuel Molina")

    assert phrased is not None
    assert phrased.opener_retry and phrased.closer_retry
    assert len(backend.calls) == 2, "one line, one re-ask"
    note = text_of(backend.calls[-1].blocks)
    assert 'it OPENED on "Through"' in note and 'it ENDED on "now"' in note


@pytest.mark.asyncio
async def test_a_goal_is_shouted_in_repeats_and_the_tail_rule_is_off() -> None:
    backend = saying(PhrasedLine(line="Mbappé! Buried it! Buried!", excitement=1.0))
    phraser = a_phraser(backend)
    phraser.accept("And it is buried!", Event.GOAL, ts=10.0)

    phrased = await phraser.phrase(a_form(event=Event.GOAL), "")

    assert phrased is not None
    assert not phrased.closer_retry
    assert len(backend.calls) == 1


# -- the same decision, live ------------------------------------------------


def a_runtime() -> Runtime:
    """A runtime with frames in the buffer, built rather than run."""
    sim = MatchSim(seed=5, duration_s=120.0)
    settings = Settings(
        capture=CaptureConfig(
            width=640, height=360, fps=8, delay_s=4.0, history_s=3.0, present_offset_s=3.0
        ),
        caller=CallerConfig(min_gap_s=2.0),
        predictor=PredictorConfig(tick_s=0.05),
        director=DirectorConfig(max_beat_age_s=30.0),
    )
    runtime = Runtime(
        source=SimSource(sim, settings.capture, realtime=False),
        backend=SimOracle(sim=sim),
        pack=sim.knowledge_pack,
        settings=settings,
        speaker=LogSpeaker(words_per_second=120),
    )
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for index in range(96):
        runtime.buffer.append(Frame(ts=index / settings.capture.fps, image=blank))
    return runtime


def watching(runtime: Runtime) -> list[tuple[str, dict[str, Any]]]:
    """Every ``_publish`` the runtime makes, topic and keywords."""
    seen: list[tuple[str, dict[str, Any]]] = []
    original = runtime._publish

    def spy(topic: Any, ts: float, value: Any = None, **extra: Any) -> None:
        seen.append((str(getattr(topic, "value", topic)), extra))
        original(topic, ts, value, **extra)

    runtime._publish = spy  # type: ignore[method-assign]
    return seen


@pytest.mark.asyncio
async def test_the_runtime_passes_over_the_second_nameless_form_too() -> None:
    """The live half of the pair above, through the runtime's own call path.

    Both paths ask the same object the same question, so a silence offline is
    a silence on air.
    """
    runtime = a_runtime()
    form = a_form("Argentina move it across the halfway line with nobody closing.")

    async def call(*_args: Any, **_kw: Any) -> CallerLine:
        return form

    runtime.caller.call = call  # type: ignore[method-assign]
    backend = saying(PhrasedLine(line="Through midfield.", excitement=0.2))
    runtime.phraser = Phraser(
        backend,
        config=PhraserConfig(model="claude-haiku-4-5"),
        home=runtime.home,
        away=runtime.away,
    )
    beats: list[Any] = []
    original = runtime.director.submit

    def spy(beat: Any) -> None:
        beats.append(beat)
        original(beat)

    runtime.director.submit = spy  # type: ignore[method-assign]
    published = watching(runtime)

    await runtime._call([Trigger.SCHEDULED])
    await runtime._call([Trigger.SCHEDULED])

    assert [beat.text for beat in beats] == ["Through midfield."]
    assert len(backend.calls) == 1, "the second form never reached the model"
    phrased = [extra for topic, extra in published if topic == "phrased"]
    assert phrased[-1]["line"] == ""
    assert phrased[-1]["reason"] == "silence: nameless build-up after nameless build-up"
    assert not [topic for topic, _ in published if topic == "gate"][1:], "one gate row, one line"
