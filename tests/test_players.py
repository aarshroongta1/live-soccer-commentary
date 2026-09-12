import numpy as np

from commentary.capture import Frame
from commentary.perception import NullTracker, PlayerTracker
from commentary.schemas import KnowledgePack, Player, Side, TeamSheet
from commentary.state import EntityRegistry

Box = tuple[int, int, int, int]

#: Solid BGR per kit. The split is a histogram of the real pixels now, so
#: these are the actual evidence rather than a stand-in for it.
KIT_BGR = {"red": (0, 0, 220), "blue": (220, 0, 0), "yellow": (0, 230, 230)}

#: Box heights in the 640-wide working image, so twice these in the frame.
TALL = 60
SHORT = 50

#: Four of each kit and a referee. Only the first is close enough to read.
SQUAD = [
    ("red", TALL),
    ("red", SHORT),
    ("red", SHORT),
    ("red", SHORT),
    ("blue", SHORT),
    ("blue", SHORT),
    ("blue", SHORT),
    ("blue", SHORT),
    ("yellow", SHORT),
]

PACK = KnowledgePack(
    home=TeamSheet(
        name="Arsenal",
        short="ARS",
        kit="red and white",
        starters=[Player(name="Bukayo Saka", number=7)],
    ),
    away=TeamSheet(
        name="Chelsea",
        short="CHE",
        kit="blue",
        starters=[Player(name="Cole Palmer", number=20)],
    ),
)


def scene(kits: list[tuple[str, int]], ts: float = 0.0) -> tuple[Frame, list[Box]]:
    """A 720p frame of solid-colour players, and the boxes for them.

    The boxes are in the 640-wide working image the tracker downscales to
    before it detects, which is why the frame rectangles are painted at twice
    these coordinates.
    """
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    boxes: list[Box] = []
    for i, (colour, height) in enumerate(kits):
        box = (10 + i * 60, 20, 34 + i * 60, 20 + height)
        boxes.append(box)
        x0, y0, x1, y1 = (v * 2 for v in box)
        image[y0:y1, x0:x1] = KIT_BGR[colour]
    return Frame(ts=ts, image=image), boxes


class FakeDetector:
    """Boxes from a script, one list per frame, holding the last after that."""

    def __init__(self, script: list[list[Box]]) -> None:
        self.script = script
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[Box]:
        boxes = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return list(boxes)


def build(script: list[list[Box]]) -> PlayerTracker:
    return PlayerTracker(FakeDetector(script), pack=PACK, fit_samples=len(script[0]))


def believe(registry: EntityRegistry, tracks: list, ts: float) -> None:
    """What the runtime does with what the tracker found.

    The tracker perceives and the registry remembers, and the runtime is the
    one place that joins them — so the tests join them the same way rather
    than handing the tracker a registry it would be the second writer of.
    """
    for track in tracks:
        if track.number is not None and track.name is not None:
            registry.believe(track.number, track.name, ts, side=track.side)


def test_the_kits_split_in_two_and_the_referee_belongs_to_neither():
    frame, boxes = scene(SQUAD)
    tracker = build([boxes])

    sides = [track.side for track in tracker.update(frame)]

    assert sides[:4] == [Side.HOME] * 4
    assert sides[4:8] == [Side.AWAY] * 4
    assert sides[8] is Side.UNKNOWN


def test_a_pass_is_the_detection_and_nothing_else():
    """Fifteen bodies, well under the frame budget, or the tags mean nothing.

    The run this replaced took 1.6 s a pass with SigLIP in it, and at that
    rate a track id is a different body every time it is looked at. Nothing
    that scales with the number of bodies belongs in here.
    """
    import time

    crowd = [("red", SHORT)] * 7 + [("blue", SHORT)] * 7 + [("yellow", SHORT)]
    frame, boxes = scene(crowd)
    tracker = build([boxes])
    tracker.update(frame)

    started = time.perf_counter()
    for i in range(5):
        tracks = tracker.update(scene(crowd, ts=float(i))[0])
    elapsed = (time.perf_counter() - started) / 5

    assert len(tracks) == 15
    assert elapsed < 0.2, f"{elapsed * 1000:.0f} ms a pass for fifteen bodies"


def test_a_sighting_names_a_track_and_it_keeps_the_name():
    """The caller read the shirt; the tracker's job is to hold on to it."""
    frame, boxes = scene(SQUAD, ts=1.0)
    tracker = build([boxes])
    first = tracker.update(frame)
    assert first[0].name is None

    tracker.identify(first[0].id, Side.HOME, 7, "Bukayo Saka", 1.0)
    assert first[0].name == "Bukayo Saka"

    later = tracker.update(scene(SQUAD, ts=2.0)[0])
    assert later[0].id == first[0].id
    assert (later[0].number, later[0].name) == (7, "Bukayo Saka")


def test_a_sighting_overrides_the_kit_split():
    """Whatever read the number saw more than a histogram of a torso did."""
    frame, boxes = scene(SQUAD, ts=1.0)
    tracker = build([boxes])
    referee = tracker.update(frame)[8]
    assert referee.side is Side.UNKNOWN

    tracker.identify(referee.id, Side.AWAY, 20, "Cole Palmer", 1.0)
    after = tracker.update(scene(SQUAD, ts=2.0)[0])[8]

    assert after.side is Side.AWAY
    assert after.name == "Cole Palmer"


def test_a_confirmed_sighting_is_believed_as_a_name():
    registry = EntityRegistry()
    frame, boxes = scene(SQUAD, ts=1.0)
    tracker = build([boxes])
    tracks = tracker.update(frame)

    tracker.identify(tracks[0].id, Side.HOME, 7, "Bukayo Saka", 1.0)
    believe(registry, tracker.update(scene(SQUAD, ts=2.0)[0]), 2.0)

    assert registry.name_for(7, Side.HOME) == "Bukayo Saka"
    assert registry.on_pitch(2.0) == {"7": "Bukayo Saka"}


def test_a_cut_starts_the_ids_again_but_the_registry_keeps_the_names():
    registry = EntityRegistry()
    first_frame, boxes = scene(SQUAD, ts=1.0)
    joined_frame, joined_boxes = scene([*SQUAD, ("blue", SHORT)], ts=3.0)
    after_frame, _ = scene(SQUAD, ts=40.0)
    tracker = build([boxes, joined_boxes, boxes])

    tracks = tracker.update(first_frame)
    tracker.identify(tracks[0].id, Side.HOME, 7, "Bukayo Saka", 1.0)
    joined = tracker.update(joined_frame)
    believe(registry, joined, 3.0)
    assert max(track.id for track in joined) == 9

    tracker.reset()
    after = tracker.update(after_frame)

    assert [track.id for track in after] == list(range(9))
    # The name belonged to the old id, and an id is not a person.
    assert after[0].name is None
    assert registry.name_for(7, Side.HOME) == "Bukayo Saka"


def test_the_null_tracker_sees_nobody():
    frame, _ = scene(SQUAD)
    tracker = NullTracker()

    assert tracker.update(frame) == []
    assert tracker.reset() is None
    assert tracker.identify(0, Side.HOME, 7, "Bukayo Saka", 1.0) is None
