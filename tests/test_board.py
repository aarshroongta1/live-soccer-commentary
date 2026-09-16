import numpy as np
import pytest

from commentary.capture import Frame
from commentary.config import BOARD_MODEL, BoardConfig
from commentary.llm.fake import ScriptedBackend
from commentary.perception import BoardReader, BoardTracker, crop_score_bug
from commentary.perception.board import BUG_GONE_S
from commentary.schemas import BoardRead
from commentary.state import MatchStateTracker

CONFIG = BoardConfig()


def read(
    home: int | None = 0,
    away: int | None = 0,
    *,
    clock: str | None = "12:30",
    confidence: float = 0.9,
    visible: bool = True,
) -> BoardRead:
    return BoardRead(
        bug_visible=visible,
        home_score=home,
        away_score=away,
        clock=clock,
        confidence=confidence,
    )


def test_crop_takes_the_fractional_box() -> None:
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = crop_score_bug(image, (0.0, 0.0, 0.42, 0.16))
    assert crop.shape == (115, 538, 3)


def test_crop_keeps_the_pixels_inside_the_box() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[10:20, 30:40] = 255
    crop = crop_score_bug(image, (0.15, 0.1, 0.2, 0.2))
    assert crop.shape == (10, 10, 3)
    assert int(crop.min()) == 255


def test_crop_rejects_a_box_that_is_not_a_box() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_score_bug(image, (0.6, 0.0, 0.4, 0.5))


@pytest.mark.asyncio
async def test_reader_makes_one_tagged_call_with_the_crop() -> None:
    scripted = ScriptedBackend()
    scripted.always("board", read(1, 0, clock="37:12"))
    reader = BoardReader(scripted)

    got = await reader.read(Frame(ts=4.0, image=np.zeros((720, 1280, 3), dtype=np.uint8)))

    assert got.home_score == 1
    calls = scripted.calls_tagged("board")
    assert len(calls) == 1
    assert calls[0].model == BOARD_MODEL
    assert calls[0].images == 1
    assert calls[0].output_format is BoardRead


@pytest.mark.asyncio
async def test_reader_system_prompt_tells_the_model_to_admit_an_absent_bug() -> None:
    scripted = ScriptedBackend()
    scripted.always("board", read())
    await BoardReader(scripted).read(
        Frame(ts=0.0, image=np.zeros((720, 1280, 3), dtype=np.uint8))
    )

    system = scripted.calls_tagged("board")[0].system
    assert "bug_visible" in system
    assert "45+2" in system


def test_a_complete_read_updates_the_score_immediately() -> None:
    tracker = BoardTracker(CONFIG)

    initial = tracker.update(read(0, 0), 10.0)
    goal = tracker.update(read(1, 0, clock="37:12"), 63.7)

    assert initial is not None and not initial.is_goal
    assert goal is not None and goal.is_goal
    assert goal.ts == 63.7
    assert (tracker.home_score, tracker.away_score) == (1, 0)


def test_a_lower_read_is_ignored_without_moving_the_score_backwards() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(2, 1), 10.0)

    assert tracker.update(read(1, 1), 20.0) is None
    assert tracker.update(read(2, 0), 22.0) is None
    assert (tracker.home_score, tracker.away_score) == (2, 1)


def test_partial_and_low_confidence_reads_are_ignored() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(0, 0), 10.0)

    assert tracker.update(read(1, None), 20.0) is None
    assert tracker.update(read(9, 9, confidence=CONFIG.min_confidence - 0.01), 22.0) is None
    assert (tracker.home_score, tracker.away_score) == (0, 0)


def test_negative_scores_are_malformed_and_ignored() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(0, 0), 10.0)

    assert tracker.update(read(-1, 0), 20.0) is None
    assert (tracker.home_score, tracker.away_score) == (0, 0)


def test_one_absent_read_does_not_flip_into_a_replay() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(0, 0), 0.0)

    tracker.update(read(None, None, clock=None, visible=False), 10.0)
    assert not tracker.in_replay


def test_two_absent_reads_are_a_replay_and_the_bug_coming_back_ends_it() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(1, 0), 0.0)
    absent = read(None, None, clock=None, visible=False)

    tracker.update(absent, 10.0)
    tracker.update(absent, 12.0)
    assert tracker.in_replay

    tracker.update(read(1, 0), 14.0)
    assert not tracker.in_replay
    assert (tracker.home_score, tracker.away_score) == (1, 0)


def test_a_replay_does_not_destroy_the_latest_score() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(1, 0), 0.0)
    tracker.update(read(2, 0), 62.7)
    absent = read(None, None, clock=None, visible=False)
    for i in range(11):
        tracker.update(absent, 91.0 + i * 3.8)

    assert (tracker.home_score, tracker.away_score) == (2, 0)
    assert tracker.bug_missing
    assert not tracker.in_replay


def test_the_clock_tracks_every_confident_read_but_not_low_confidence() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(0, 0, clock="12:30"), 0.0)

    tracker.update(read(0, 0, clock="12:32"), 20.0)
    tracker.update(read(0, 0, clock="12:34", confidence=0.1), 22.0)
    assert tracker.clock == "12:32"


def test_the_period_changes_with_a_complete_read() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(0, 0, clock="44:50"), 0.0)
    tracker.update(read(0, 0, clock="46:01"), 60.0)
    assert tracker.period == 2


def test_first_half_stoppage_time_is_still_the_first_half() -> None:
    tracker = BoardTracker(CONFIG)
    tracker.update(read(0, 0, clock="45+1"), 0.0)
    assert tracker.period == 1


def test_a_bug_that_never_comes_back_stops_being_a_replay() -> None:
    tracker = BoardTracker(CONFIG)
    state = MatchStateTracker("Arsenal", "Madrid")
    absent = read(None, None, clock=None, visible=False)

    tracker.update(absent, 0.0)
    tracker.update(absent, CONFIG.interval_s)
    assert tracker.in_replay and not tracker.bug_missing
    state.apply_board(tracker)
    assert "replay" in state.summary()

    ts = 2 * CONFIG.interval_s
    while ts <= BUG_GONE_S:
        tracker.update(absent, ts)
        ts += CONFIG.interval_s

    assert tracker.bug_missing
    assert not tracker.in_replay
    state.apply_board(tracker)
    summary = state.summary()
    assert "no score bug visible" in summary
    assert "replay" not in summary


def test_the_bug_coming_back_after_a_long_absence_clears_both_flags() -> None:
    tracker = BoardTracker(CONFIG)
    absent = read(None, None, clock=None, visible=False)
    for i in range(40):
        tracker.update(absent, i * CONFIG.interval_s)
    assert tracker.bug_missing

    tracker.update(read(0, 0), 80.0)
    assert not tracker.bug_missing
    assert not tracker.in_replay
