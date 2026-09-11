"""The predictor is what silence sounds like. These pin down when it breaks."""

import pytest

from commentary.config import SETTINGS
from commentary.predictor import SpeakPredictor
from commentary.schemas import Trigger

MIN_GAP = SETTINGS.caller.min_gap_s
FORCES_AT = SETTINGS.predictor.silence_forces_at_s


def test_the_rate_cap_holds_against_a_loud_trigger():
    predictor = SpeakPredictor()
    decision = predictor.decide(100.0 + MIN_GAP / 2, [Trigger.ROAR], last_spoken_ts=100.0)
    assert not decision.should_call
    assert decision.reason.startswith("rate_cap")
    assert "roar" in decision.reason


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
        108.0, [Trigger.CAMERA_CUT, Trigger.WHISTLE, Trigger.ROAR], last_spoken_ts=100.0
    )
    assert decision.should_call
    assert "roar" in decision.reason
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
        (100.0, [Trigger.WHISTLE]),
        (101.0, [Trigger.CAMERA_CUT]),
        (106.0, [Trigger.ROAR]),
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
