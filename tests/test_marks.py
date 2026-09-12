"""Names drawn on the picture, which is how the language model reads one.

The chain under test is short: the sim's tracker says who is where and which
shirts were legible, and ``draw_marks`` puts a surname over each body the
system can actually name. The two things that must never happen are a name
over a player nobody identified, and a mark on a frame in the buffer.
"""

from __future__ import annotations

import numpy as np
import pytest

from commentary.capture.buffer import Frame
from commentary.perception.players import NullTracker, Track
from commentary.prompts.caller import caller_blocks, draw_marks, no_tracks
from commentary.schemas import KnowledgePack, Player, Side, TeamSheet
from commentary.sim import BroadcastRenderer, MatchSim, SimTracker
from commentary.sim.render import NUMBER_LEGIBLE_RADIUS
from commentary.state import EntityRegistry


def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina", short="ARG", starters=[Player(name="Ángel Di María", number=11)]
        ),
        away=TeamSheet(name="France", short="FRA"),
    )


def blank() -> np.ndarray:
    return np.full((200, 300, 3), 90, dtype=np.uint8)


# -- the marks themselves ----------------------------------------------------


def test_a_named_track_gets_its_surname():
    image = blank()
    track = Track(id=0, side=Side.HOME, box=(40, 60, 70, 130), number=11, name="Ángel Di María")
    marked = draw_marks(image, [track], pack())
    assert not np.array_equal(marked, image)
    # Drawn on a copy: the frame handed in is the frame in the buffer, and the
    # board reader and the analyst must see the broadcast, not our annotations.
    assert int(image.min()) == int(image.max()) == 90


def test_a_number_with_nobody_on_the_sheet_gets_the_team_and_the_number():
    image = blank()
    track = Track(id=0, side=Side.HOME, box=(40, 60, 70, 130), number=14, name=None)
    assert not np.array_equal(draw_marks(image, [track], pack()), image)


@pytest.mark.parametrize(
    "track",
    [
        Track(id=0, side=Side.HOME, box=(40, 60, 70, 130)),
        Track(id=1, side=Side.UNKNOWN, box=(40, 60, 70, 130), number=11, name="Ángel Di María"),
    ],
)
def test_an_unidentified_body_is_left_unmarked(track: Track):
    """No "?" labels. An unlabelled body means unknown, and a frame full of
    question marks is noise the model has to reason past."""
    image = blank()
    assert np.array_equal(draw_marks(image, [track], pack()), image)


# -- the caller's blocks -----------------------------------------------------


def frames(count: int, start: float = 0.0) -> list[Frame]:
    return [Frame(ts=start + i, image=blank()) for i in range(count)]


def test_caller_blocks_leave_the_buffer_frames_byte_identical():
    cursor = frames(2)
    before = [f.image.copy() for f in cursor]
    track = Track(id=0, side=Side.HOME, box=(40, 60, 70, 130), number=11, name="Ángel Di María")

    caller_blocks(cursor, frames(1, 5.0), "state", [], [], lambda ts: [track], pack())

    for frame, original in zip(cursor, before, strict=True):
        assert np.array_equal(frame.image, original)


def test_the_marks_change_the_pictures_and_nothing_else():
    cursor, lookahead = frames(2), frames(1, 5.0)
    track = Track(id=0, side=Side.HOME, box=(40, 60, 70, 130), number=11, name="Ángel Di María")

    plain = caller_blocks(cursor, lookahead, "state", [], [], no_tracks, pack())
    marked = caller_blocks(cursor, lookahead, "state", [], [], lambda ts: [track], pack())

    assert [b["type"] for b in plain] == [b["type"] for b in marked]
    texts = [b for b in plain if b["type"] == "text"]
    assert texts == [b for b in marked if b["type"] == "text"], "the rule text must not move"
    images = [(a, b) for a, b in zip(plain, marked, strict=True) if a["type"] == "image"]
    assert all(a != b for a, b in images), "every picture should carry the mark"


# -- the simulator's tracker -------------------------------------------------


def test_the_sim_tracker_names_the_dots_whose_numbers_are_drawn():
    sim = MatchSim(seed=11, duration_s=120.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=1280, height=720)
    registry = EntityRegistry()
    tracker = SimTracker(sim, renderer, registry=registry)

    named: list[Track] = []
    for i in range(600):
        ts = i / 15.0
        tracks = tracker.update(Frame(ts=ts, image=np.zeros((4, 4, 3), dtype=np.uint8)))
        named = [t for t in tracks if t.name is not None]
        if len(named) >= 5:
            break

    assert named, "no dot was ever legible enough to name"
    sheets = (sim.knowledge_pack.home, sim.knowledge_pack.away)
    squad = {p.name for sheet in sheets for p in sheet.squad}
    assert all(t.name in squad for t in named)
    assert all(t.side is not Side.UNKNOWN for t in named)
    assert registry.identified(40.0), "nothing reached the registry"


def test_the_sim_tracker_leaves_a_distant_dot_unread():
    """Legibility is the renderer's decision, not the tracker's.

    A number the picture never printed must not be readable off the picture,
    or the sim's marks row measures the ground truth rather than the chain.
    """
    sim = MatchSim(seed=11, duration_s=120.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=1280, height=720)
    tracker = SimTracker(sim, renderer)

    for i in range(300):
        ts = i / 15.0
        state = sim.at(ts)
        visible = renderer.dots_on_screen(state, renderer.camera(state))
        small = [d for d, _c, r in visible if r < NUMBER_LEGIBLE_RADIUS]
        if not small:
            continue
        tracks = tracker.update(Frame(ts=ts, image=np.zeros((4, 4, 3), dtype=np.uint8)))
        unread = {(t.side, t.number) for t in tracks if t.number is None}
        assert any((d.side, None) in unread for d in small)
        return
    pytest.skip("no dot was ever too small to read in this clip")


def test_a_cut_starts_the_ids_again_and_the_shirts_have_to_be_read_again():
    sim = MatchSim(seed=11, duration_s=120.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=1280, height=720)
    registry = EntityRegistry()
    tracker = SimTracker(sim, renderer, registry=registry)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(200):
        tracker.update(Frame(ts=i / 15.0, image=image))
    before = registry.identified(13.0)
    assert before, "nothing was identified before the cut"

    tracker.reset()
    tracks = tracker.update(Frame(ts=13.5, image=image))
    assert min((t.id for t in tracks), default=0) == 0, "ids did not restart"
    # The name survives the cut even though the track does not.
    assert registry.identified(13.5) == before


# -- the substitution --------------------------------------------------------


def test_the_null_tracker_is_the_same_shape_and_finds_nobody():
    tracker = NullTracker()
    assert tracker.update(Frame(ts=1.0, image=blank())) == []
    tracker.reset()
