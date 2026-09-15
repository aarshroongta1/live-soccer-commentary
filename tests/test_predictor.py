"""The predictor is what silence sounds like. These pin down when it breaks."""

import pytest

from commentary.config import SETTINGS
from commentary.predictor import SpeakPredictor
from commentary.schemas import Event, Trigger

MIN_GAP = SETTINGS.caller.min_gap_s
ATTACKING_GAP = SETTINGS.caller.min_gap_attacking_s
BUILD_UP_GAP = SETTINGS.caller.min_gap_build_up_s
DEAD_BALL_GAP = SETTINGS.caller.min_gap_dead_ball_s
GAP_FLOOR = SETTINGS.caller.min_gap_floor_s
GAP_PAD = SETTINGS.caller.gap_after_line_s
FORCES_AT = SETTINGS.predictor.silence_forces_at_s


def test_the_rate_cap_holds_against_a_loud_trigger():
    predictor = SpeakPredictor()
    decision = predictor.decide(100.0 + MIN_GAP / 2, [Trigger.CAMERA_CUT], last_spoken_ts=100.0)
    assert not decision.should_call
    assert decision.reason.startswith("rate_cap")
    assert "camera_cut" in decision.reason


def test_a_board_change_beats_the_rate_cap():
    predictor = SpeakPredictor()
    decision = predictor.decide(100.5, [Trigger.BOARD_CHANGE], last_spoken_ts=100.0)
    assert decision.should_call
    assert decision.urgency == pytest.approx(1.0)
    assert "board_change" in decision.reason


def test_silence_eventually_forces_a_line():
    predictor = SpeakPredictor()
    quiet = predictor.decide(100.0 + FORCES_AT - 1.0, [], last_spoken_ts=100.0)
    assert not quiet.should_call

    forced = predictor.decide(100.0 + FORCES_AT + 0.5, [], last_spoken_ts=100.0)
    assert forced.should_call
    assert forced.urgency == pytest.approx(1.0)
    assert Trigger.SILENCE_PRESSURE in forced.triggers
    assert "silence_pressure" in forced.reason


def test_nothing_spoken_yet_counts_as_maximum_silence():
    predictor = SpeakPredictor()
    decision = predictor.decide(0.0, [], last_spoken_ts=None)
    assert decision.should_call
    assert "nothing spoken yet" in decision.reason


def test_the_reason_names_the_winning_trigger():
    predictor = SpeakPredictor()
    decision = predictor.decide(
        108.0, [Trigger.CAMERA_CUT, Trigger.BOARD_CHANGE], last_spoken_ts=100.0
    )
    assert decision.should_call
    assert "board_change" in decision.reason
    assert "8.0s since last line" in decision.reason


def test_no_trigger_means_no_line():
    predictor = SpeakPredictor()
    decision = predictor.decide(105.0, [], last_spoken_ts=100.0)
    assert not decision.should_call
    assert decision.reason.startswith("no trigger")


def test_pressure_lifts_a_weak_trigger_as_the_silence_runs_on():
    predictor = SpeakPredictor()
    early = predictor.decide(105.0, [Trigger.CAMERA_CUT], last_spoken_ts=100.0)
    late = predictor.decide(111.0, [Trigger.CAMERA_CUT], last_spoken_ts=100.0)
    assert early.should_call and late.should_call
    assert late.urgency > early.urgency


def test_a_trigger_trace_replays_identically():
    trace = [
        (100.0, [Trigger.CAMERA_CUT]),
        (101.0, [Trigger.CAMERA_CUT]),
        (106.0, [Trigger.BOARD_CHANGE]),
        (119.0, []),
    ]

    def replay() -> list[tuple[bool, float]]:
        predictor = SpeakPredictor()
        spoken: float | None = None
        out: list[tuple[bool, float]] = []
        for ts, triggers in trace:
            decision = predictor.decide(ts, triggers, spoken)
            if decision.should_call:
                spoken = ts
            out.append((decision.should_call, decision.urgency))
        return out

    assert replay() == replay()


def test_a_goal_beats_the_rate_cap() -> None:
    """The four seconds after a goal are the four that matter most.

    A goal is the one thing that is never too soon: the scorer's name, the
    celebration and the replay all arrive inside the window the cap was
    keeping quiet.
    """
    from commentary.predictor import SpeakPredictor
    from commentary.schemas import Trigger as T

    predictor = SpeakPredictor()
    blocked = predictor.decide(now_ts=10.0, triggers=[T.SCHEDULED], last_spoken_ts=8.0)
    assert not blocked.should_call
    assert "rate_cap" in blocked.reason

    allowed = predictor.decide(
        now_ts=10.0, triggers=[T.SCHEDULED], last_spoken_ts=8.0, after_goal=True
    )
    assert allowed.should_call
    assert "a goal beats the rate cap" in allowed.reason


def test_a_short_line_buys_a_short_silence() -> None:
    """"De Paul." takes under a second to say and does not earn four of quiet.

    This is the fragment rhythm the caller was rewritten for: real commentary
    speaks a median 2.4s apart, and the cap as it was made that impossible
    whatever the caller wrote.
    """
    predictor = SpeakPredictor()
    short = 0.6  # two words at the speaker's own rate
    decision = predictor.decide(
        100.0 + GAP_FLOOR + 0.1,
        [Trigger.CAMERA_CUT],
        last_spoken_ts=100.0,
        last_spoken_seconds=short,
    )
    assert decision.should_call
    # Where the old cap would still have been holding it back.
    assert GAP_FLOOR + 0.1 < MIN_GAP

    too_soon = predictor.decide(
        100.5, [Trigger.CAMERA_CUT], last_spoken_ts=100.0, last_spoken_seconds=short
    )
    assert not too_soon.should_call
    assert f"min gap {GAP_FLOOR:.1f}s" in too_soon.reason


def test_a_long_line_still_buys_the_old_four_seconds() -> None:
    """A 20-word sentence takes six or seven seconds; the cap is the cap."""
    predictor = SpeakPredictor()
    blocked = predictor.decide(
        103.9, [Trigger.CAMERA_CUT], last_spoken_ts=100.0, last_spoken_seconds=6.5
    )
    assert not blocked.should_call
    assert f"min gap {MIN_GAP:.1f}s" in blocked.reason

    allowed = predictor.decide(
        104.1, [Trigger.CAMERA_CUT], last_spoken_ts=100.0, last_spoken_seconds=6.5
    )
    assert allowed.should_call


def test_the_gap_is_the_line_plus_a_breath_between_the_floor_and_the_cap() -> None:
    predictor = SpeakPredictor()
    assert predictor.gap_after(None) == pytest.approx(MIN_GAP)
    assert predictor.gap_after(0.0) == pytest.approx(GAP_FLOOR)
    assert predictor.gap_after(0.3) == pytest.approx(GAP_FLOOR)
    assert predictor.gap_after(2.0) == pytest.approx(2.0 + GAP_PAD)
    assert predictor.gap_after(30.0) == pytest.approx(MIN_GAP)


def test_an_unmeasured_line_keeps_the_old_fixed_cap() -> None:
    """Nobody recorded a length, so nothing may be assumed about it."""
    predictor = SpeakPredictor()
    decision = predictor.decide(102.0, [Trigger.CAMERA_CUT], last_spoken_ts=100.0)
    assert not decision.should_call
    assert f"min gap {MIN_GAP:.1f}s" in decision.reason


def test_a_board_change_still_beats_the_shortened_gap() -> None:
    predictor = SpeakPredictor()
    decision = predictor.decide(
        100.2, [Trigger.BOARD_CHANGE], last_spoken_ts=100.0, last_spoken_seconds=0.6
    )
    assert decision.should_call
    assert "board_change beats the rate cap" in decision.reason


def test_a_goal_still_beats_the_shortened_gap() -> None:
    predictor = SpeakPredictor()
    decision = predictor.decide(
        100.2,
        [Trigger.SCHEDULED],
        last_spoken_ts=100.0,
        last_spoken_seconds=0.6,
        after_goal=True,
    )
    assert decision.should_call


def test_the_fixed_cadence_baseline_ignores_the_line_length() -> None:
    """The naive timer is the thing being compared against; it does not move."""
    from commentary.grading.baselines import FixedCadence

    plain = FixedCadence()
    lengths = FixedCadence()
    for ts in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 9.0):
        left = plain.decide(ts, [Trigger.CAMERA_CUT], last_spoken_ts=None)
        right = lengths.decide(
            ts, [Trigger.CAMERA_CUT], last_spoken_ts=ts - 0.5, last_spoken_seconds=0.4
        )
        assert left.should_call == right.should_call
        assert left.reason == right.reason


# -- two rates, not one ------------------------------------------------------


def test_the_cap_is_the_phase_the_last_line_was_about() -> None:
    """Study section 2.3: 2.8s in the box, 4.2s in build-up, 4.5s at a restart.

    One rate through all three is the corpus study's Gap 3. A long line still
    cannot buy more than its phase allows, which is what these three assert.
    """
    predictor = SpeakPredictor()
    assert predictor.gap_after(30.0, Event.SHOT) == pytest.approx(ATTACKING_GAP)
    assert predictor.gap_after(30.0, Event.BUILD_UP) == pytest.approx(BUILD_UP_GAP)
    assert predictor.gap_after(30.0, Event.THROW_IN) == pytest.approx(DEAD_BALL_GAP)
    assert predictor.gap_after(30.0, None) == pytest.approx(MIN_GAP)


def test_a_replay_takes_the_dead_ball_rate_whatever_it_is_a_replay_of() -> None:
    """The broadcast's own dead ball.

    A replay of a shot is not an attacking move: the ball is not in play
    behind the picture and nobody is about to score. At the 2.5 s attacking
    cap the voice would be back over the next camera angle before the last
    line had landed, so the replay outranks the event and takes the restart's
    rate instead.
    """
    predictor = SpeakPredictor()
    assert predictor.gap_after(30.0, Event.SHOT) == pytest.approx(ATTACKING_GAP)
    assert predictor.gap_after(30.0, Event.SHOT, last_was_replay=True) == pytest.approx(
        DEAD_BALL_GAP
    )
    assert predictor.gap_after(30.0, Event.BUILD_UP, last_was_replay=True) == pytest.approx(
        DEAD_BALL_GAP
    )


def test_the_voice_comes_back_faster_in_the_box_than_on_the_halfway_line() -> None:
    predictor = SpeakPredictor()
    after_a_shot = predictor.decide(
        103.0, [Trigger.CAMERA_CUT], last_spoken_ts=100.0, last_spoken_seconds=4.0,
        last_event=Event.SHOT,
    )
    after_build_up = predictor.decide(
        103.0, [Trigger.CAMERA_CUT], last_spoken_ts=100.0, last_spoken_seconds=4.0,
        last_event=Event.BUILD_UP,
    )
    assert after_a_shot.should_call
    assert not after_build_up.should_call


def test_a_chosen_silence_holds_the_cap_without_easing_the_pressure() -> None:
    """The phraser passing over a moment must not send the caller straight back.

    Real commentary says nothing at a quarter of build-up touches. The tick
    loop runs twice a second, so without this the next tick calls Opus again
    half a second after the phraser decided there was nothing to say — and
    the silence itself still has to force a line eventually, which is why the
    pressure clock is left alone.
    """
    predictor = SpeakPredictor()
    straight_after = predictor.decide(
        100.5, [Trigger.CAMERA_CUT], last_spoken_ts=None, last_quiet_ts=100.0
    )
    assert not straight_after.should_call
    assert straight_after.reason.startswith("rate_cap")

    later = predictor.decide(
        100.0 + FORCES_AT + 1.0, [Trigger.SCHEDULED], last_spoken_ts=None, last_quiet_ts=100.0
    )
    assert later.should_call
    assert "silence_pressure forces a line" in later.reason
