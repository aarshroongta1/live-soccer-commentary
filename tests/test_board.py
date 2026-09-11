import numpy as np
import pytest

from commentary.capture import Frame
from commentary.config import BOARD_MODEL, BoardConfig
from commentary.llm.fake import ScriptedBackend
from commentary.perception import BoardReader, BoardTracker, crop_score_bug
from commentary.schemas import BoardRead, Side

CONFIG = BoardConfig()


@pytest.fixture
def scripted() -> ScriptedBackend:
    return ScriptedBackend()


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


def settle(tracker: BoardTracker, value: BoardRead, start: float = 0.0) -> None:
    """Push enough agreeing reads to confirm whatever `value` says."""
    for i in range(CONFIG.confirmations):
        tracker.update(value, start + i * CONFIG.interval_s)


def test_crop_takes_the_fractional_box():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    crop = crop_score_bug(image, (0.0, 0.0, 0.42, 0.16))
    assert crop.shape == (115, 538, 3)


def test_crop_keeps_the_pixels_inside_the_box():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[10:20, 30:40] = 255
    crop = crop_score_bug(image, (0.15, 0.1, 0.2, 0.2))
    assert crop.shape == (10, 10, 3)
    assert int(crop.min()) == 255


def test_crop_rejects_a_box_that_is_not_a_box():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        crop_score_bug(image, (0.6, 0.0, 0.4, 0.5))


async def test_reader_makes_one_tagged_call_with_the_crop(scripted):
    scripted.always("board", read(1, 0, clock="37:12"))
    reader = BoardReader(scripted)

    frame = Frame(ts=4.0, image=np.zeros((720, 1280, 3), dtype=np.uint8))
    got = await reader.read(frame)

    assert got.home_score == 1
    calls = scripted.calls_tagged("board")
    assert len(calls) == 1
    assert calls[0].model == BOARD_MODEL
    assert calls[0].images == 1
    assert calls[0].output_format is BoardRead


async def test_reader_system_prompt_tells_the_model_to_admit_an_absent_bug(scripted):
    scripted.always("board", read())
    reader = BoardReader(scripted)
    await reader.read(Frame(ts=0.0, image=np.zeros((720, 1280, 3), dtype=np.uint8)))

    system = scripted.calls_tagged("board")[0].system
    assert "bug_visible" in system
    assert "45+2" in system


def test_two_agreeing_reads_do_not_move_the_score():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    for i in range(CONFIG.confirmations - 1):
        assert tracker.update(read(1, 0), 10.0 + i * 2.0) is None
    assert tracker.home_score == 0
    assert tracker.pending is not None
    assert tracker.pending.home_score == 1
    assert tracker.pending.count == CONFIG.confirmations - 1


def test_three_agreeing_reads_move_the_score():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    change = None
    for i in range(CONFIG.confirmations):
        change = tracker.update(read(1, 0, clock="12:36"), 10.0 + i * 2.0)

    assert change is not None
    assert (tracker.home_score, tracker.away_score) == (1, 0)
    assert change.is_goal
    assert change.scoring_side is Side.HOME
    assert tracker.pending is None


def test_the_change_is_stamped_at_the_first_agreeing_read():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    first_ts = 100.0
    change = None
    for i in range(CONFIG.confirmations):
        change = tracker.update(read(0, 1), first_ts + i * CONFIG.interval_s)

    assert change is not None
    assert change.ts == first_ts
    assert change.scoring_side is Side.AWAY
    # The goal is reported no later than the confirmation window, which is what
    # the day-2 gate ("within 3 s of every goal") is actually measuring.
    assert (first_ts + (CONFIG.confirmations - 1) * CONFIG.interval_s) - change.ts <= 4.0


def test_a_single_outlier_never_moves_the_score():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    for i in range(12):
        value = read(3, 2) if i % 4 == 0 else read(0, 0)
        assert tracker.update(value, 20.0 + i * 2.0) is None
    assert (tracker.home_score, tracker.away_score) == (0, 0)


def test_a_low_confidence_read_is_no_evidence_either_way():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    tracker.update(read(1, 0), 10.0)
    tracker.update(read(9, 9, confidence=CONFIG.min_confidence - 0.1), 12.0)
    change = tracker.update(read(1, 0), 14.0)
    assert change is None

    change = tracker.update(read(1, 0), 16.0)
    assert change is not None
    assert change.ts == 10.0


def test_one_absent_read_does_not_flip_into_a_replay():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    tracker.update(read(None, None, clock=None, visible=False), 10.0)
    assert not tracker.in_replay


def test_two_absent_reads_are_a_replay_and_the_bug_coming_back_ends_it():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    absent = read(None, None, clock=None, visible=False)
    tracker.update(absent, 10.0)
    tracker.update(absent, 12.0)
    assert tracker.in_replay

    tracker.update(read(0, 0), 14.0)
    assert not tracker.in_replay


def test_a_replay_discards_half_formed_evidence():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    tracker.update(read(1, 0), 10.0)
    tracker.update(read(1, 0), 12.0)
    tracker.update(read(None, None, clock=None, visible=False), 14.0)
    assert tracker.pending is None

    assert tracker.update(read(1, 0), 16.0) is None
    assert tracker.home_score == 0


def test_the_clock_tracks_every_confident_read():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0, clock="12:30"))

    tracker.update(read(0, 0, clock="12:32"), 20.0)
    assert tracker.clock == "12:32"
    tracker.update(read(0, 0, clock="12:34", confidence=0.1), 22.0)
    assert tracker.clock == "12:32"


def test_the_first_confirmed_board_is_not_reported_as_a_goal():
    tracker = BoardTracker(CONFIG)
    change = None
    for i in range(CONFIG.confirmations):
        change = tracker.update(read(0, 0), i * 2.0)

    assert change is not None
    assert not change.is_goal
    assert change.scoring_side is Side.UNKNOWN


def test_the_period_is_confirmed_like_the_score():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0, clock="44:50"))
    assert tracker.period == 1

    for i in range(CONFIG.confirmations):
        tracker.update(read(0, 0, clock=f"46:0{i}"), 60.0 + i * 2.0)
    assert tracker.period == 2


def test_first_half_stoppage_time_is_still_the_first_half():
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0, clock="45+1"))
    assert tracker.period == 1
