"""The numbers, written by code, and the thirty seconds after a goal.

Two modules and the two places that drive them. ``commentary.scoreline``
composes the scoreline and takes the model's out; ``commentary.goalfollow``
says which of the corpus's beats is due and where a missing one goes. The
offline rephrase is exercised in ``test_phraser.py``, where the trace fixture
lives; what is here is the arithmetic itself and the live runtime, which is
the half the two must not drift apart on.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from commentary.agents.phraser import Phraser
from commentary.capture.buffer import Frame
from commentary.config import (
    CallerConfig,
    CaptureConfig,
    DirectorConfig,
    PhraserConfig,
    PredictorConfig,
    RestatementConfig,
    Settings,
)
from commentary.goalfollow import MAX_SYNTH, GoalFollowup
from commentary.llm.fake import ScriptedBackend
from commentary.runtime import Runtime
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    KnowledgePack,
    MatchState,
    Note,
    PhrasedLine,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
    Trigger,
)
from commentary.scoreline import (
    Restatements,
    effective_score,
    in_words,
    restatement,
    say_score,
    settle_numbers,
    strip_score,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.voice import LogSpeaker

# -- fixtures ----------------------------------------------------------------


def a_state(home: int = 2, away: int = 1, **kw: Any) -> MatchState:
    return MatchState(home="Argentina", away="France", home_score=home, away_score=away, **kw)


def a_form(line: str = "Mbappé hooks it in.", event: Event = Event.GOAL, **kw: Any) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=event,
        side=kw.pop("side", Side.AWAY),
        sightings=kw.pop(
            "sightings", [Sighting(number=10, name="Kylian Mbappé", side=Side.AWAY)]
        ),
        confidence=0.9,
        speak=True,
        line=line,
        **kw,
    )


# -- the number as a commentator says it -------------------------------------


def test_a_number_is_a_word_and_nothing_is_two_different_words() -> None:
    """"Nil minutes gone" is not English, which is why zero is a parameter."""
    assert in_words(0) == "nil"
    assert in_words(0, zero="no") == "no"
    assert in_words(3) == "three"
    assert in_words(21) == "twenty-one"
    assert in_words(40) == "forty"


def test_the_score_a_line_is_about_is_the_board_plus_the_goal_arriving() -> None:
    """The board lags the ball by about the length of a celebration."""
    state = a_state(2, 1)
    assert effective_score(state, Side.AWAY, goal_in_state=False) == (2, 2)
    assert effective_score(state, Side.HOME, goal_in_state=False) == (3, 1)
    # Once the graphic has caught up, the board is the whole answer.
    assert effective_score(state, Side.AWAY, goal_in_state=True) == (2, 1)


def test_a_goal_by_nobody_over_a_board_that_has_not_moved_is_not_guessed_at() -> None:
    assert effective_score(a_state(2, 1), Side.UNKNOWN, goal_in_state=False) is None
    assert say_score(a_state(2, 1), Side.UNKNOWN) == ""


def test_the_leading_side_is_named_first_and_the_shapes_rotate() -> None:
    """Section 8.4 slot 4: a small closed set, and not the same one all evening."""
    state = a_state(2, 1)
    said = {say_score(state, Side.AWAY, goal_in_state=True, index=i) for i in range(3)}
    assert said == {"Two-one to Argentina.", "Two-one, Argentina.", "Two-one."}


def test_a_level_score_takes_the_two_shapes_the_corpus_uses_for_a_draw() -> None:
    state = a_state(2, 1)
    assert say_score(state, Side.AWAY, index=0) == "Two-two."
    assert say_score(state, Side.AWAY, index=1) == "Two-all."


def test_a_board_nobody_can_see_says_no_score_at_all() -> None:
    """``bug_visible`` false is the bug gone longer than any replay lasts."""
    assert say_score(a_state(2, 1, bug_visible=False), Side.AWAY) == ""


# -- taking the model's number out -------------------------------------------


def test_the_score_comes_out_and_the_commentary_stays() -> None:
    stripped = strip_score("Mbappé! On the volley! Two-two.")
    assert stripped.text == "Mbappé! On the volley!"
    assert stripped.removed == ("Two-two",)


def test_a_fragment_that_is_nothing_but_a_number_goes_entirely() -> None:
    """"Mbappé! Into the net! And that's." is worse than losing the fragment."""
    assert strip_score("Mbappé! Into the net! And that's two-two.").text == (
        "Mbappé! Into the net!"
    )
    assert strip_score("It's two-one.").text == ""


def test_what_leans_on_the_number_goes_with_it() -> None:
    """A sentence that lost its object reads worse than one that lost a clause."""
    assert strip_score("Mbappé makes it three-two.").text == "Mbappé."
    assert strip_score("Argentina take a two-nil lead.").text == "Argentina."
    assert strip_score("Two-one to France, and the place erupts.").text == "And the place erupts."


def test_a_score_hiding_in_ordinary_words_comes_out_too() -> None:
    assert strip_score("Mbappé! He levels it!").removed
    assert strip_score("Molina! Argentina's third!").removed == ("Argentina's third",)


def test_the_scorers_own_tally_is_left_alone_because_it_is_the_third_beat() -> None:
    """"His third of the campaign" is a pack claim, and beat 3 of a goal."""
    said = "Mbappé! His third of the tournament."
    assert strip_score(said).text == said
    assert strip_score(said).removed == ()


def test_a_line_with_no_number_in_it_is_returned_untouched() -> None:
    said = "Molina, down the right."
    assert strip_score(said) == strip_score(said)
    assert strip_score(said).text == said
    assert not strip_score(said).changed


def test_the_goal_line_gets_one_number_and_the_celebration_gets_none() -> None:
    """The single rule this module exists for, at its single call site."""
    state = a_state(2, 1)
    call = settle_numbers(
        "Mbappé! On the volley! Three-two.", state=state, side=Side.AWAY, append=True
    )
    assert call.line == "Mbappé! On the volley! Two-two."
    assert call.stripped == ("Three-two",)
    assert call.appended == "Two-two."

    after = settle_numbers("Mbappé! Three-two.", state=state, side=Side.AWAY, append=False)
    assert after.line == "Mbappé!"
    assert after.appended == ""


# -- the score and the clock, on a timer -------------------------------------


def test_the_restatement_is_the_clock_then_the_score() -> None:
    state = a_state(2, 1, clock_s=1200.0)
    assert restatement(state, index=0) == "Twenty minutes gone, two-one to Argentina."
    assert restatement(state, index=1) == "Still two-one to Argentina, twenty minutes played."


def test_a_restatement_needs_a_clock_a_bug_and_no_replay() -> None:
    assert restatement(a_state(2, 1)) == ""
    assert restatement(a_state(2, 1, clock_s=600.0, in_replay=True)) == ""
    assert restatement(a_state(2, 1, clock_s=600.0, bug_visible=False)) == ""


def test_the_first_clock_the_timer_sees_is_where_the_schedule_starts() -> None:
    """A match joined at 78 minutes owes fifteen of these on the arithmetic."""
    timer = Restatements(every_s=300.0)
    timer.note(a_state(clock_s=4680.0))
    assert not timer.pending


def test_a_period_comes_due_once_and_is_spent_once() -> None:
    timer = Restatements(every_s=300.0)
    timer.note(a_state(clock_s=60.0))
    timer.note(a_state(clock_s=240.0))
    assert not timer.pending
    timer.note(a_state(clock_s=320.0))
    assert timer.pending

    said = timer.take(a_state(2, 1, clock_s=320.0))
    assert "five minutes" in said.lower()
    assert "two-one" in said.lower()
    assert not timer.pending
    assert timer.take(a_state(2, 1, clock_s=320.0)) == ""


def test_a_timer_set_to_zero_never_comes_due() -> None:
    timer = Restatements(every_s=0.0)
    timer.note(a_state(clock_s=60.0))
    timer.note(a_state(clock_s=900.0))
    assert not timer.enabled
    assert not timer.pending
    assert timer.take(a_state(2, 1, clock_s=900.0)) == ""


# -- which beat is due -------------------------------------------------------


def test_a_goal_arms_the_window_and_the_beats_run_in_order() -> None:
    follow = GoalFollowup()
    assert follow.is_the_call(10.0), "no goal is live, so this is the call"
    follow.arm(10.0, a_form(), "Mbappé! On the volley!")
    assert not follow.is_the_call(12.0), "the score has already gone out"
    assert follow.beat(12.0) == 2
    follow.said(12.0)
    assert follow.beat(15.0) == 3
    follow.said(15.0)
    assert follow.beat(18.0) == 4
    follow.said(18.0)
    assert follow.beat(21.0) is None, "four beats plus the colour seat is the corpus's seven"


def test_the_window_closes_after_thirty_seconds() -> None:
    follow = GoalFollowup()
    follow.arm(10.0, a_form(), "Mbappé!")
    assert follow.active(39.0)
    assert not follow.active(41.0)
    assert follow.beat(41.0) is None
    assert follow.is_the_call(41.0), "a goal called now is a new goal"


def test_a_beat_waits_the_corpus_gap_before_it_is_due() -> None:
    follow = GoalFollowup()
    follow.arm(10.0, a_form(), "Mbappé!")
    assert not follow.due(11.0), "the corpus's own gap is a median two seconds"
    assert follow.due(12.5)


def test_a_gap_the_caller_is_about_to_fill_is_left_alone() -> None:
    follow = GoalFollowup()
    follow.arm(10.0, a_form(), "Mbappé!")
    assert follow.synth_times(10.0, until=14.0) == []
    assert follow.synth_times(10.0, until=None) == [14.0, 18.0]
    assert len(follow.synth_times(10.0, until=None)) <= MAX_SYNTH


def test_nothing_is_synthesised_past_the_end_of_the_window() -> None:
    follow = GoalFollowup()
    follow.arm(10.0, a_form(), "Mbappé!")
    # Six seconds from the last line to the end of the window is not a gap
    # this fills: the corpus's own longest gap inside one is 7.1 s.
    assert follow.synth_times(35.0, until=None) == []
    assert follow.synth_times(10.0, until=None)[-1] <= 40.0


def test_the_scorer_is_the_name_the_spoken_line_actually_used() -> None:
    follow = GoalFollowup()
    follow.arm(
        10.0,
        a_form(
            sightings=[
                Sighting(number=5, name="Aurélien Tchouaméni", side=Side.AWAY),
                Sighting(number=10, name="Kylian Mbappé", side=Side.AWAY),
            ]
        ),
        "Mbappé! On the volley!",
    )
    assert follow.scorer == "Kylian Mbappé"


def test_the_block_names_the_beat_and_puts_the_scorers_clause_in_front_of_it() -> None:
    follow = GoalFollowup()
    follow.arm(10.0, a_form(), "Mbappé! On the volley!")
    follow.said(12.0)
    pack = KnowledgePack(
        home=TeamSheet(name="Argentina", short="ARG", demonym="Argentine", starters=[]),
        away=TeamSheet(
            name="France",
            short="FRA",
            demonym="French",
            starters=[Player(name="Kylian Mbappé", number=10)],
        ),
        notes=[
            Note(about="Kylian Mbappé", text="Three in the tournament already.", kind="stat")
        ],
    )
    block = follow.block(15.0, pack)
    assert "BEAT 3 — ONE NUMBER ABOUT THE SCORER" in block
    assert "Three in the tournament already." in block
    assert "Do not say it again" in block


def test_a_form_the_caller_never_said_is_still_the_account_of_the_move() -> None:
    """After the Mbappé penalty four forms went unspoken. Beat 4 is built of them."""
    follow = GoalFollowup()
    follow.arm(10.0, a_form(line="Mbappé hooks it in."), "Mbappé!")
    follow.saw_form(a_form(line="Dembélé stood it up at the back post."))
    follow.said(12.0)
    follow.said(15.0)
    block = follow.block(18.0)
    assert "BEAT 4 — REBUILD THE MOVE" in block
    assert "Dembélé stood it up at the back post." in block


def test_a_synthesised_form_carries_nothing_the_gate_has_not_seen() -> None:
    follow = GoalFollowup()
    original = a_form()
    follow.arm(10.0, original, "Mbappé! On the volley!")
    made = follow.synthetic()
    assert made.event is Event.GOAL
    assert [s.name for s in made.sightings] == [s.name for s in original.sightings]
    assert made.line == original.line


# -- the live runtime --------------------------------------------------------


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
    """A runtime with frames in the buffer, built rather than run.

    The same shape as ``test_phraser.a_runtime``: ``Runtime.run`` stops on a
    wall clock while the sim pours frames in as fast as the machine takes
    them, so a test that asks "is this exactly the line" has to make the
    calls one at a time instead.
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


def phraser_saying(runtime: Runtime, *lines: PhrasedLine) -> None:
    backend = ScriptedBackend()
    if lines:
        backend.queue("phraser", list(lines))
    runtime.phraser = Phraser(
        backend,
        config=PhraserConfig(model="claude-haiku-4-5"),
        home=runtime.home,
        away=runtime.away,
    )


def caught_beats(runtime: Runtime) -> list[Beat]:
    beats: list[Beat] = []
    original = runtime.director.submit

    def spy(beat: Beat) -> None:
        beats.append(beat)
        original(beat)

    runtime.director.submit = spy  # type: ignore[method-assign]
    return beats


def calling(runtime: Runtime, line: CallerLine) -> None:
    async def call(*_args: Any, **_kw: Any) -> CallerLine:
        return line

    runtime.caller.call = call  # type: ignore[method-assign]


def a_goal_form(runtime: Runtime) -> CallerLine:
    """A goal the runtime's own roster and board will both stand behind."""
    assert runtime.pack is not None
    player = runtime.pack.away.starters[0]
    # The board has already counted it, so the scoreline is the board's and
    # the team names of a simulated match never come into it: one each.
    runtime.state.home_score, runtime.state.away_score = 1, 1
    runtime._last_goal_ts = 0.0
    return a_form(
        line=f"{player.surname} turns it in from six yards and the net bulges.",
        side=Side.AWAY,
        sightings=[Sighting(number=player.number, name=player.name, side=Side.AWAY)],
    )


@pytest.mark.asyncio
async def test_the_runtime_appends_the_score_to_the_goal_line_and_strips_the_models() -> None:
    """Point 1 of the rule, live. The refusals this replaces were both goals."""
    runtime = a_runtime()
    form = a_goal_form(runtime)
    surname = form.sightings[0].name.rsplit(" ", 1)[-1]
    calling(runtime, form)
    phraser_saying(runtime, PhrasedLine(line=f"{surname}! Three-nil!", excitement=1.0))
    beats = caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])

    assert [beat.text for beat in beats] == [f"{surname}! One-one."]
    assert runtime.follow.active(runtime.cursor_ts), "the window did not open"


@pytest.mark.asyncio
async def test_a_celebration_after_the_same_goal_loses_its_number_and_gains_none() -> None:
    """Point 2. Both of the lines the gate used to refuse were this line."""
    runtime = a_runtime()
    form = a_goal_form(runtime)
    surname = form.sightings[0].name.rsplit(" ", 1)[-1]
    calling(runtime, form)
    phraser_saying(
        runtime,
        PhrasedLine(line=f"{surname}! Off the ground!", excitement=1.0),
        PhrasedLine(line=f"{surname}! Two-one!", excitement=0.9),
    )
    beats = caught_beats(runtime)

    await runtime._call([Trigger.SCHEDULED])
    await runtime._call([Trigger.SCHEDULED])

    assert [beat.text for beat in beats] == [
        f"{surname}! Off the ground! One-one.",
        f"{surname}!",
    ]


@pytest.mark.asyncio
async def test_a_followup_call_goes_out_on_the_beat_that_is_due() -> None:
    """Point 3, live: the caller has gone quiet and the window has not."""
    runtime = a_runtime()
    form = a_goal_form(runtime)
    surname = form.sightings[0].name.rsplit(" ", 1)[-1]
    runtime.follow.arm(runtime.cursor_ts - 3.0, form, f"{surname}!")
    phraser_saying(runtime, PhrasedLine(line="And the bench is up. Two-one.", excitement=0.9))
    beats = caught_beats(runtime)

    assert await runtime._say_followup(runtime.cursor_ts)

    assert [beat.text for beat in beats] == ["And the bench is up."]
    assert beats[0].event is Event.GOAL
    assert not beats[0].preemptable
    assert runtime.follow.synthesised == 1
    assert runtime.stats.followups == 1


@pytest.mark.asyncio
async def test_a_followup_the_phraser_has_nothing_for_simply_does_not_happen() -> None:
    """There is no caller line underneath this one to fall back to."""
    runtime = a_runtime()
    form = a_goal_form(runtime)
    runtime.follow.arm(runtime.cursor_ts - 3.0, form, "Goal!")
    phraser_saying(runtime)
    beats = caught_beats(runtime)

    assert not await runtime._say_followup(runtime.cursor_ts)
    assert beats == []


def test_the_runtime_says_the_score_and_the_clock_when_the_period_comes_due() -> None:
    """Point 4, live. No model is asked and no gate is needed: it is all state."""
    runtime = a_runtime(fast_settings(restatement=RestatementConfig(every_s=300.0)))
    runtime.state.home_score, runtime.state.away_score = 2, 0
    runtime.state.clock_s = 60.0
    beats = caught_beats(runtime)

    assert not runtime._maybe_restate(), "the first clock is where the schedule starts"
    runtime.state.clock_s = 620.0
    assert runtime._maybe_restate()

    assert [beat.text for beat in beats] == [
        f"Ten minutes gone, two-nil to {runtime.home}."
    ]
    assert beats[0].excitement == pytest.approx(0.1)
    assert beats[0].preemptable
    assert runtime.stats.restatements == 1
    assert not runtime._maybe_restate(), "one period, one line"


def test_a_restatement_waits_rather_than_speaking_over_the_game() -> None:
    """Filler that is late is better than filler that talks across a goal."""
    runtime = a_runtime(fast_settings(restatement=RestatementConfig(every_s=300.0)))
    runtime.state.home_score, runtime.state.away_score = 2, 0
    runtime.state.clock_s = 60.0
    runtime._maybe_restate()
    runtime.state.clock_s = 620.0
    runtime._last_spoken_video_ts = runtime.cursor_ts - 0.5
    beats = caught_beats(runtime)

    assert not runtime._maybe_restate()
    assert beats == []
    assert runtime.restatements.pending, "the period is owed, not lost"

    runtime._last_spoken_video_ts = runtime.cursor_ts - 20.0
    assert runtime._maybe_restate()


def test_the_restatement_can_be_turned_off_for_a_feed_that_carries_a_bug() -> None:
    runtime = a_runtime(fast_settings(restatement=RestatementConfig(every_s=0.0)))
    runtime.state.clock_s = 60.0
    runtime._maybe_restate()
    runtime.state.clock_s = 3000.0
    assert not runtime._maybe_restate()
