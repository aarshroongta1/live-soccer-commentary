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
    # Nine of the ten: a body is tagged from the second pass it is seen on,
    # so a one-pass false positive never gets a tag the caller could point at.
    assert len(joined) == 9

    before = {track.id for track in joined}
    tracker.reset()
    after = tracker.update(after_frame)

    # No id is ever reused. A sighting written before the cut and delivered
    # after it names a body that has gone, rather than whoever inherited it.
    assert not before & {track.id for track in after}
    # The name belonged to the old id, and an id is not a person.
    assert all(track.name is None for track in after)
    assert registry.name_for(7, Side.HOME) == "Bukayo Saka"


def test_the_null_tracker_sees_nobody():
    frame, _ = scene(SQUAD)
    tracker = NullTracker()

    assert tracker.update(frame) == []
    assert tracker.reset() is None
    assert tracker.identify(0, Side.HOME, 7, "Bukayo Saka", 1.0) is None


def test_a_tag_is_letters_and_never_a_number():
    """Three of six sightings on the real clip came back with the mark equal
    to the shirt number, which is a reading of the tag and not of the shirt."""
    from commentary.perception.players import id_of, mark_of

    assert [mark_of(n) for n in (0, 1, 25, 26, 27, 701, 702)] == [
        "A",
        "B",
        "Z",
        "AA",
        "AB",
        "ZZ",
        "AAA",
    ]
    assert all(id_of(mark_of(n)) == n for n in range(2000))
    assert id_of("#4") is None
    assert id_of("11") is None
    assert id_of("") is None
    assert id_of(" aa ") == 26


def test_a_body_keeps_its_id_while_it_moves():
    """What the greedy matcher could not do: a pan is not a new set of people.

    On the real clip the median overlap between consecutive passes is 0.35
    against a threshold of 0.3, so half the bodies were renamed every pass.
    ByteTrack matches against where a body is predicted to be instead.
    """
    tracker = build([scene(SQUAD)[1]])
    tracker.update(scene(SQUAD, ts=0.0)[0])

    ids = []
    for step in range(1, 6):
        drifted = [(colour, height) for colour, height in SQUAD]
        frame, boxes = scene(drifted, ts=float(step))
        # The whole shot slides sideways, the way a camera follows play.
        slid = [(x0 + 6 * step, y0, x1 + 6 * step, y1) for x0, y0, x1, y1 in boxes]
        tracker.detector.script = [slid]
        tracker.detector.calls = 0
        ids.append([track.id for track in tracker.update(frame)])

    assert ids[0] == ids[-1], f"ids churned across a pan: {ids[0]} -> {ids[-1]}"


# -- the gallery, through the tracker ----------------------------------------


class _Recorder:
    """A gallery that records what it was asked, with no embedder at all."""

    def __init__(self, answer=None) -> None:
        self.learned: list[tuple] = []
        self.asked: list[int] = []
        self.answer = answer
        self.players = 0

    def learn(self, crop, side, number, name) -> None:
        self.learned.append((side, number, name, crop.shape[0]))
        self.players = len({(s, n) for s, n, _, _ in self.learned})

    def classify(self, crops):
        self.asked.append(len(crops))
        return [self.answer] * len(crops)

    def drain(self) -> dict:
        return {"players": self.players, "classified": 0, "hits": 0}


def test_a_sighting_is_the_only_thing_the_gallery_learns_from():
    """Bootstrapped from certainty: nothing else calls learn."""
    recorder = _Recorder()
    frame, boxes = scene(SQUAD, ts=1.0)
    tracker = PlayerTracker(FakeDetector([boxes]), pack=PACK, fit_samples=len(boxes))
    tracker.gallery = recorder
    tracks = tracker.update(frame)
    assert recorder.learned == []

    tracker.identify(tracks[0].id, Side.HOME, 7, "Bukayo Saka", 1.0)

    assert [(s, n, name) for s, n, name, _ in recorder.learned] == [
        (Side.HOME, 7, "Bukayo Saka")
    ]


def test_the_gallery_is_asked_about_unnamed_bodies_once_a_second_at_most():
    """It is the expensive part of the chain and the detector must not wait."""
    from commentary.perception.gallery import CLASSIFY_EVERY_S, CROPS_PER_PASS

    recorder = _Recorder()
    frame, boxes = scene(SQUAD, ts=0.0)
    tracker = PlayerTracker(FakeDetector([boxes]), pack=PACK, fit_samples=len(boxes))
    tracker.gallery = recorder

    tracker.update(frame)
    for step in (0.2, 0.4, 0.6):
        tracker.update(scene(SQUAD, ts=step)[0])
    assert len(recorder.asked) == 1, "one pass a second, not one a frame"
    assert recorder.asked[0] <= CROPS_PER_PASS

    tracker.update(scene(SQUAD, ts=CLASSIFY_EVERY_S + 0.1)[0])
    assert len(recorder.asked) == 2


def test_a_gallery_hit_names_the_track_and_says_where_the_name_came_from():
    from commentary.perception.gallery import Match

    recorder = _Recorder(Match(7, "Bukayo Saka", Side.HOME, 0.9, 0.2))
    frame, boxes = scene(SQUAD, ts=0.0)
    tracker = PlayerTracker(FakeDetector([boxes]), pack=PACK, fit_samples=len(boxes))
    tracker.gallery = recorder

    tracks = tracker.update(frame)
    named = [t for t in tracks if t.name is not None]

    assert named, "the gallery answered and nothing was named"
    assert all(t.from_gallery for t in named)
    assert all((t.number, t.name) == (7, "Bukayo Saka") for t in named)


def test_a_read_shirt_beats_a_gallery_guess_on_the_same_track():
    from commentary.perception.gallery import Match

    recorder = _Recorder(Match(7, "Bukayo Saka", Side.HOME, 0.9, 0.2))
    frame, boxes = scene(SQUAD, ts=0.0)
    tracker = PlayerTracker(FakeDetector([boxes]), pack=PACK, fit_samples=len(boxes))
    tracker.gallery = recorder
    tracks = tracker.update(frame)

    tracker.identify(tracks[0].id, Side.AWAY, 20, "Cole Palmer", 1.0)
    after = tracker.update(scene(SQUAD, ts=2.0)[0])[0]

    assert (after.number, after.name) == (20, "Cole Palmer")
    assert not after.from_gallery
