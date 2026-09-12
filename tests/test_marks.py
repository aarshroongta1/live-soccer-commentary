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


def test_an_unidentified_body_gets_a_handle_and_not_a_guess():
    """``#0`` claims nothing about who it is, which is the point of it.

    The caller cannot report "I can read an eleven on that one" without a way
    to say which one, and a tag it can point at is that way. It is not a
    shirt number and the rules say so.
    """
    image = blank()
    track = Track(id=0, side=Side.HOME, box=(40, 60, 70, 130))

    marked = draw_marks(image, [track], pack())
    named = draw_marks(
        image,
        [Track(id=0, side=Side.HOME, box=(40, 60, 70, 130), number=11, name="Ángel Di María")],
        pack(),
    )

    assert not np.array_equal(marked, image)
    assert not np.array_equal(marked, named), "a handle and a name must not look the same"


def test_a_body_on_neither_team_is_tagged_like_any_other():
    """It used to be left alone, and that cost 94% of the close-ups.

    Of 784 bodies on the real clip tall enough for a shirt number to be
    legible, 44 carried a tag: a close-up crop looks nothing like the
    wide-shot crops the kit split was fitted on, so the split called them
    referees. The bodies whose numbers can be read were exactly the bodies
    with nothing to read them against. A letter over the actual referee is
    harmless — nobody reports a shirt number off him.
    """
    image = blank()
    track = Track(id=1, side=Side.UNKNOWN, box=(40, 60, 70, 130))
    assert not np.array_equal(draw_marks(image, [track], pack()), image)


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
    tracker = SimTracker(sim, renderer)

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
    assert all(t.number is not None for t in named)


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
    tracker = SimTracker(sim, renderer)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    for i in range(200):
        for track in tracker.update(Frame(ts=i / 15.0, image=image)):
            if track.number is not None and track.name is not None:
                registry.believe(track.number, track.name, i / 15.0, side=track.side)
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


async def test_the_runtime_puts_what_the_tracker_read_into_the_state():
    """The tracker perceives; the registry remembers; the runtime joins them.

    Without this the marks are drawn on the frames and the state's
    "identified" line stays empty, so a name leaves with the camera.
    """
    from commentary.config import CaptureConfig, PredictorConfig, Settings
    from commentary.runtime import Runtime
    from commentary.sim import MatchSim, SimOracle, SimSource

    sim = MatchSim(seed=11, duration_s=120.0)
    settings = Settings(
        capture=CaptureConfig(width=640, height=360, fps=8, delay_s=2.0, history_s=3.0),
        predictor=PredictorConfig(tick_s=0.05),
    )
    source = SimSource(sim, settings.capture, realtime=False)
    runtime = Runtime(
        source=source,
        backend=SimOracle(sim=sim, error_rate=0.0),
        pack=sim.knowledge_pack,
        settings=settings,
        tracker=SimTracker(sim, source.renderer, pack=sim.knowledge_pack),
    )
    await runtime.run(seconds=3.0)

    identified = runtime.state_tracker.registry.identified(runtime.cursor_ts)
    assert identified, "the tracker read shirts and none of them reached the state"
    assert "identified:" in runtime.state_tracker.summary(runtime.cursor_ts)


def test_a_label_with_no_room_above_the_player_is_not_drawn():
    """Clipped at the top edge it is unreadable, and it would sit over
    whatever the broadcaster has up there — a score bug, a clock, a strap."""
    image = blank()
    high = Track(id=0, side=Side.HOME, box=(40, 2, 70, 90), number=11, name="Ángel Di María")
    assert np.array_equal(draw_marks(image, [high], pack()), image)
