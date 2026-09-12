import numpy as np

from commentary.capture import Frame
from commentary.perception import NullTracker, PlayerTracker
from commentary.schemas import KnowledgePack, Player, Side, TeamSheet
from commentary.state import EntityRegistry

Box = tuple[int, int, int, int]

#: Solid BGR per kit, and the vector the fake embedder answers with. Separable
#: by construction, because what is under test is the split and not SigLIP.
KIT_BGR = {"red": (0, 0, 220), "blue": (220, 0, 0), "yellow": (0, 230, 230)}
KIT_VECTOR = {"red": (1.0, 0.0, 0.0), "blue": (0.0, 1.0, 0.0), "yellow": (0.0, 0.0, 1.0)}
BY_BGR = {bgr: name for name, bgr in KIT_BGR.items()}

#: Box heights in the 640-wide working image, so twice these in the frame: one
#: side of the 110 px the number reader needs, and one side the other.
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


class FakeEmbedder:
    """One fixed vector per kit, looked up by the crop's mean colour."""

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        vectors = []
        for crop in crops:
            mean = tuple(int(round(v)) for v in crop.reshape(-1, 3).mean(axis=0))
            vectors.append(KIT_VECTOR[BY_BGR[mean]])
        return np.asarray(vectors, dtype=np.float32)


class FakeNumberReader:
    """Scripted (text, confidence), holding the last, and keeps every crop."""

    def __init__(self, script: list[tuple[str, float]]) -> None:
        self.script = script
        self.crops: list[np.ndarray] = []

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        self.crops.append(crop)
        return self.script[min(len(self.crops) - 1, len(self.script) - 1)]


def build(
    script: list[list[Box]],
    reads: list[tuple[str, float]],
) -> tuple[PlayerTracker, FakeNumberReader]:
    reader = FakeNumberReader(reads)
    tracker = PlayerTracker(
        FakeDetector(script),
        FakeEmbedder(),
        reader,
        pack=PACK,
        fit_samples=len(script[0]),
    )
    return tracker, reader


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
    tracker, _ = build([boxes], [("", 0.0)])

    sides = [track.side for track in tracker.update(frame)]

    assert sides[:4] == [Side.HOME] * 4
    assert sides[4:8] == [Side.AWAY] * 4
    assert sides[8] is Side.UNKNOWN


def test_a_number_confirms_on_the_second_agreeing_read():
    first_frame, boxes = scene(SQUAD, ts=1.0)
    second_frame, _ = scene(SQUAD, ts=2.0)
    tracker, _ = build([boxes], [("7", 0.95)])

    first = tracker.update(first_frame)
    assert first[0].number is None

    second = tracker.update(second_frame)
    assert second[0].id == first[0].id
    assert second[0].number == 7
    assert second[0].name == "Bukayo Saka"


def test_a_low_confidence_read_is_not_evidence():
    frames = [scene(SQUAD, ts=float(i)) for i in range(4)]
    tracker, _ = build([frames[0][1]], [("7", 0.5), ("7", 0.5), ("7", 0.95), ("7", 0.95)])

    numbers = [tracker.update(frame)[0].number for frame, _ in frames]

    assert numbers == [None, None, None, 7]


def test_a_three_digit_read_is_not_a_shirt_number():
    frames = [scene(SQUAD, ts=float(i)) for i in range(4)]
    tracker, _ = build([frames[0][1]], [("777", 0.95), ("777", 0.95), ("7", 0.95), ("7", 0.95)])

    numbers = [tracker.update(frame)[0].number for frame, _ in frames]

    assert numbers == [None, None, None, 7]


def test_a_confirmed_number_is_believed_as_a_name():
    registry = EntityRegistry()
    first_frame, boxes = scene(SQUAD, ts=1.0)
    second_frame, _ = scene(SQUAD, ts=2.0)
    tracker, _ = build([boxes], [("7", 0.95)])

    believe(registry, tracker.update(first_frame), 1.0)
    believe(registry, tracker.update(second_frame), 2.0)

    assert registry.name_for(7, Side.HOME) == "Bukayo Saka"
    assert registry.on_pitch(2.0) == {"7": "Bukayo Saka"}


def test_a_player_too_far_away_is_never_sent_to_the_number_reader():
    distant = [(colour, SHORT) for colour, _ in SQUAD]
    frame, boxes = scene(distant)
    tracker, reader = build([boxes], [("7", 0.95)])

    tracker.update(frame)

    assert reader.crops == []


def test_a_cut_starts_the_ids_again_but_the_registry_keeps_the_names():
    registry = EntityRegistry()
    first_frame, boxes = scene(SQUAD, ts=1.0)
    second_frame, _ = scene(SQUAD, ts=2.0)
    joined_frame, joined_boxes = scene([*SQUAD, ("blue", SHORT)], ts=3.0)
    after_frame, _ = scene(SQUAD, ts=40.0)
    tracker, _ = build([boxes, boxes, joined_boxes, boxes], [("7", 0.95)])

    believe(registry, tracker.update(first_frame), 1.0)
    believe(registry, tracker.update(second_frame), 2.0)
    joined = tracker.update(joined_frame)
    believe(registry, joined, 3.0)
    assert max(track.id for track in joined) == 9

    tracker.reset()
    after = tracker.update(after_frame)

    assert [track.id for track in after] == list(range(9))
    # The votes belonged to the old ids, so the number has to be earned again.
    assert after[0].number is None
    assert registry.name_for(7, Side.HOME) == "Bukayo Saka"


def test_the_null_tracker_sees_nobody():
    frame, _ = scene(SQUAD)
    tracker = NullTracker()

    assert tracker.update(frame) == []
    assert tracker.reset() is None
