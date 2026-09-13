import numpy as np
import pytest

from commentary.capture import Frame
from commentary.config import BOARD_MODEL, BoardConfig
from commentary.llm.fake import ScriptedBackend
from commentary.perception import BoardReader, BoardTracker, crop_score_bug
from commentary.schemas import BoardRead

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


def test_a_replay_does_not_destroy_the_evidence_of_the_goal():
    """This test used to assert the opposite, and the opposite was wrong.

    It read "a replay interrupts the evidence": an absent read cleared the
    half-formed score. But the bug is pulled for the replay of *the goal that
    just went in*, every time, on every broadcast — so the reads that had seen
    the score move were thrown away exactly when they mattered. On the second
    real run the bug went 1-0 to 2-0, was read twice, and then vanished for a
    minute behind the celebration and the replay; the change had to start
    again from nothing, and lines about the goal were rejected as phantoms
    while the state still said 1-0.

    The pending change now survives an absent run and confirms on the next
    agreeing read, stamped at the first read that saw it.
    """
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(1, 0))

    # The reads of that run, at their own times: 2-0 twice, then eleven absent
    # looks from 91.0 to 132.4, then the bug back at 136.1.
    assert tracker.update(read(2, 0), 62.7) is None
    assert tracker.update(read(2, 0), 66.3) is None
    absent = read(None, None, clock=None, visible=False)
    for i in range(11):
        assert tracker.update(absent, 91.0 + i * 3.8) is None
    assert tracker.pending is not None
    assert tracker.pending.count == 2
    assert tracker.pending.first_ts == 62.7

    change = tracker.update(read(2, 0), 136.1)
    assert change is not None
    assert change.is_goal
    assert change.ts == 62.7
    assert (tracker.home_score, tracker.away_score) == (2, 0)


def test_a_visible_board_showing_something_else_does_discard_the_evidence():
    """The one thing that is a reason to stop believing: a look that disagrees."""
    tracker = BoardTracker(CONFIG)
    settle(tracker, read(0, 0))

    tracker.update(read(1, 0), 10.0)
    tracker.update(read(1, 0), 12.0)
    tracker.update(read(0, 0), 14.0)
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


def test_a_bug_that_never_comes_back_stops_being_a_replay():
    """The checkpoint's unverified claim, verified: it was true.

    Two minutes of absent reads used to leave ``in_replay`` set, so every
    prompt for the rest of the match carried "screen: replay, not live play"
    — and the caller is told never to call a replay as live, so it went
    quiet. A bug missing for longer than any replay lasts is a crop pointed
    at the wrong corner or a broadcast with no bug at all, which is a
    different thing and calls for the opposite behaviour.
    """
    from commentary.perception.board import BUG_GONE_S
    from commentary.state import MatchStateTracker

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


def test_the_bug_coming_back_after_a_long_absence_clears_both_flags():
    tracker = BoardTracker(CONFIG)
    absent = read(None, None, clock=None, visible=False)
    for i in range(40):
        tracker.update(absent, i * CONFIG.interval_s)
    assert tracker.bug_missing

    tracker.update(read(0, 0), 80.0)
    assert not tracker.bug_missing
    assert not tracker.in_replay


def test_pending_goal_is_a_score_increase_and_nothing_else():
    tracker = BoardTracker(CONFIG)
    # Nothing settled yet: the first board ever seen is not a goal.
    tracker.update(read(1, 0), 0.0)
    assert tracker.pending is not None
    assert tracker.pending_goal is None

    settle(tracker, read(1, 0), 10.0)
    # A new period at the same score is pending, but not a goal.
    tracker.update(read(1, 0, clock="46:01"), 20.0)
    assert tracker.pending is not None
    assert tracker.pending_goal is None
    # A read that lowers a score is a misread, not a goal.
    tracker.update(read(0, 0), 24.0)
    assert tracker.pending_goal is None
    # One read with the score up is a goal in the making.
    tracker.update(read(1, 1), 28.0)
    assert tracker.pending_goal is not None
    assert tracker.pending_goal.count == 1
    assert tracker.pending_goal.first_ts == 28.0
