"""Holding on to a body while the camera stays on it.

A commentator never says "the number seven". They say the name, because they
watched the teams run out and they can read a shirt. This used to try to do
the whole of that job locally — detect, embed the kit with SigLIP, read the
number with PARSeq — and on three minutes of real broadcast it confirmed no
shirt number at all, while the caller read nine correct number-and-name pairs
off the very same frames.

So the division of labour changed. Claude reads the numbers, because Claude
can. This finds the bodies, decides which team each is in, and gives each one
a tag the caller can point at — and then holds that tag on that body while the
camera stays on it, which is the one thing a language model looking at single
frames cannot do. A name arrives through :meth:`PlayerTracker.identify` from
something that actually read it, and rides the track until the next cut.

What is left is a detector and arithmetic. A pass is the detection and
nothing else, which is what lets it run often enough for a tag to mean the
same body from one pass to the next — the thing SigLIP's second per pass took
away.
"""

from __future__ import annotations

import importlib
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from commentary.capture.buffer import Frame
from commentary.schemas import KnowledgePack, Side

log = logging.getLogger(__name__)

#: Detection runs on a 640-wide copy — the boxes land within a pixel of where
#: they would on the full frame and cost a quarter as much.
WORK_WIDTH = 640

#: How permissive ByteTrack's association is, as its own "matching threshold"
#: — a distance, so 0.9 accepts a match at an overlap of 0.1 or better against
#: the *predicted* box. Tuned on 300 passes of the real clip at 5.8 passes a
#: second, scored on the only thing that matters here: the caller points at a
#: body, and four seconds later its line comes back — is the id still there?
#:
#:   matcher                     ids   median life   still there at +4 s
#:   greedy IoU >= 0.3 (before) 1966         0.00 s                   3 %
#:   ByteTrack 0.7               392         1.50 s                  17 %
#:   ByteTrack 0.8               274         2.42 s                  28 %
#:   ByteTrack 0.9               202         2.67 s                  42 %
#:   ByteTrack 0.95              177         3.17 s                  48 %
#:
#: 0.95 keeps going up and is where it stops being tuning: an overlap of 0.05
#: is barely a claim that this is the same body, and a body that inherits a
#: track inherits a name. 0.9 is the last value that still requires the boxes
#: to actually overlap.
MATCH_THRESHOLD = 0.9

#: Passes a body may be missing — occluded, out of shot, missed by the
#: detector — before its id is given up. Thirty of them, about five seconds at
#: the rate the detector manages.
LOST_PASSES = 30

#: What the tracker expects to run at, which is all ByteTrack uses the frame
#: rate for: it scales the lost-track buffer by it.
TRACK_RATE_HZ = 5.8

#: Bins of the kit histogram: hue, then saturation, then value. Hue is what
#: separates red from blue; saturation and value are what separate white from
#: black, which have no hue to speak of and are half the kits in football.
KIT_BINS = (12, 3, 3)

#: Crops to fit the two-team split on before it is believed, and how often to
#: fit it again. A hundred rather than the two hundred the embedding version
#: wanted, because a histogram costs microseconds and the run that measured
#: this only ever gathered a few crops a pass.
KIT_SAMPLES = 100
KIT_REFIT_S = 300.0

#: How far a unit-length kit histogram may sit from the nearer centroid and
#: still belong to a team. Past it are the referee, the physio and the ball boy,
#: who wear neither kit and must not be called as players.
KIT_DISTANCE = 0.5

#: The dozen words a kit is actually described with, as BGR — the colour space
#: the captured frames are already in.
KIT_COLOURS: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 220),
    "blue": (220, 0, 0),
    "white": (240, 240, 240),
    "black": (20, 20, 20),
    "green": (0, 128, 0),
    "yellow": (0, 230, 230),
    "orange": (0, 140, 255),
    "purple": (128, 0, 128),
    "claret": (49, 30, 122),
    "sky": (235, 206, 135),
    "navy": (128, 0, 0),
    "maroon": (0, 0, 128),
    "gold": (55, 175, 212),
    "grey": (128, 128, 128),
}


class Detector(Protocol):
    """Person boxes, xyxy, in the coordinates of the image handed in."""

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]: ...


class Tracker(Protocol):
    """What the runtime holds. Two implementations, and they are substituted.

    The ablation that measures what the marks buy does not set a flag: it
    puts a :class:`NullTracker` where the real one goes, so the match loop
    has no branch in it and cannot drift between the two configurations.
    """

    def update(self, frame: Frame) -> list[Track]: ...

    def reset(self) -> None: ...

    def identify(
        self, mark: int, side: Side, number: int, name: str, ts: float
    ) -> None: ...


@dataclass(frozen=True)
class _Identity:
    """A name bound to a track id, and the moment somebody read it."""

    side: Side
    number: int
    name: str
    ts: float


@dataclass
class Track:
    """One body, followed across frames, named once somebody reads a shirt."""

    id: int
    side: Side
    box: tuple[int, int, int, int]
    number: int | None = None
    name: str | None = None


def mark_of(track_id: int) -> str:
    """The tag printed above a body: A, B, ... Z, AA, AB, ...

    Letters, because the first version printed the track id as "#4" and the
    caller read it as a shirt number — three of six sightings on the real
    clip came back with the mark equal to the number, which is a reading of
    the tag and not of the shirt. A letter cannot be a shirt number, so the
    ambiguity is gone rather than argued with in the prompt.
    """
    letters = ""
    n = track_id
    while True:
        letters = chr(ord("A") + n % 26) + letters
        n = n // 26 - 1
        if n < 0:
            return letters


def id_of(mark: str) -> int | None:
    """The track a tag refers to, or None if that is not a tag."""
    text = mark.strip().upper()
    if not text or not text.isalpha() or not text.isascii():
        return None
    value = 0
    for char in text:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(w, box[2]), min(h, box[3])
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=image.dtype)
    return np.ascontiguousarray(image[y0:y1, x0:x1])


def _torso(crop: np.ndarray) -> np.ndarray:
    """The upper half of a body, which is where the shirt is.

    The lower half is shorts, socks and grass, and the grass is the same for
    both teams — including it is handing the split a dimension that carries
    no information and a lot of variance.
    """
    return crop[: max(1, crop.shape[0] // 2)]


def _kit_histogram(crop: np.ndarray) -> np.ndarray:
    """What colour this body is, as one small unit-length vector.

    This replaced a SigLIP embedding, which was the right tool for telling
    two people apart and much too slow to ask about every body on every pass
    — a second for fifteen crops, against 92 ms for the whole detection. The
    question here is only "red or blue", a histogram answers it in
    microseconds, and being able to ask it every pass is worth more than any
    sharpness a learned embedding would have added.
    """
    import cv2

    hsv = cv2.cvtColor(_torso(crop), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(KIT_BINS), [0, 180, 0, 256, 0, 256])
    return np.asarray(hist, dtype=np.float64).reshape(-1)


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Length-one rows, so one distance threshold means the same thing always."""
    points = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return np.asarray(points / np.where(norms == 0.0, 1.0, norms))


def _kmeans(points: np.ndarray, *, iterations: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Two clusters, seeded from the data rather than from a random state.

    A dozen lines beats a scikit-learn dependency for k=2, and seeding from the
    first point and the point farthest from it makes the split reproducible:
    the same crops give the same two teams on every run, which matters when an
    eval is comparing two runs of the same match.
    """
    first = points[0]
    second = points[int(np.argmax(np.linalg.norm(points - first, axis=1)))]
    centroids = np.stack([first, second]).astype(np.float64)
    labels = np.zeros(len(points), dtype=np.int64)
    for _ in range(iterations):
        gaps = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.asarray(np.argmin(gaps, axis=1))
        for k in (0, 1):
            members = points[labels == k]
            if len(members):
                centroids[k] = members.mean(axis=0)
    return labels, centroids


def _kit_colour(kit: str) -> tuple[int, int, int] | None:
    """The first colour word in a kit description, as BGR.

    "red and white stripes" is red: the first word is the one a viewer would
    name the team by, and the stripe is detail the mean colour cannot see.
    """
    for word in re.findall(r"[a-z]+", kit.lower()):
        if word in KIT_COLOURS:
            return KIT_COLOURS[word]
    return None


def _colour_gap(mean: np.ndarray, bgr: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(mean - np.asarray(bgr, dtype=np.float64)))


class PlayerTracker:
    """Bodies in, tagged tracks out.

    Ids come from ByteTrack, which the plan always called for and which the
    first version stood in for with greedy IoU against the previous pass. That
    stand-in did not survive contact with a broadcast: on the real clip the
    median best overlap between consecutive passes is 0.35 against a threshold
    of 0.3, so about half of every frame's bodies were handed a new id every
    pass — 1966 ids in 300 passes, 1539 of them lasting a single pass. A tag
    that means a different body each time it is drawn is worse than no tag.

    The difference is not the threshold, it is the Kalman filter: ByteTrack
    matches against where a body is *predicted* to be, so a camera pan stops
    counting against every overlap in the frame. See ``MATCH_THRESHOLD`` for
    what that is worth, measured.
    """

    def __init__(
        self,
        detector: Detector,
        *,
        pack: KnowledgePack | None = None,
        fit_samples: int = KIT_SAMPLES,
        refit_s: float = KIT_REFIT_S,
    ) -> None:
        self.detector = detector
        self.pack = pack
        self.fit_samples = fit_samples
        #: Kits change at half time and light changes all match, so the split
        #: is re-fitted rather than decided once at kickoff.
        self.refit_s = refit_s
        self._samples: deque[np.ndarray] = deque(maxlen=fit_samples)
        self._colours: deque[np.ndarray] = deque(maxlen=fit_samples)
        self._centroids: np.ndarray | None = None
        self._home_cluster = 0
        self._fit_ts = 0.0
        self._bytetrack = _new_bytetrack()
        #: ByteTrack counts from one again on a fresh instance, so ids are
        #: shifted past everything the last camera used. A sighting written
        #: before a cut and delivered after it then names a body that does not
        #: exist, which is the right answer — the alternative is that it names
        #: whoever inherited the number.
        self._base = 0
        self._high = 0
        self._live: dict[int, Track] = {}
        #: Names bound to track ids by :meth:`identify`, kept until a cut.
        self._named: dict[int, _Identity] = {}
        self._next_id = 0

    def update(self, frame: Frame) -> list[Track]:
        """Every body in this frame, carrying whatever is known about each."""
        found = self._assign_ids(self._detect(frame.image))
        if not found:
            self._live = {}
            return []

        ids = [tid for tid, _ in found]
        boxes = [box for _, box in found]
        crops = [_crop(frame.image, box) for box in boxes]
        sides = self._sides(crops, frame.ts)

        tracks: list[Track] = []
        live: dict[int, Track] = {}
        for tid, box, side in zip(ids, boxes, sides, strict=True):
            held = self._live.get(tid)
            if side is Side.UNKNOWN and held is not None:
                # A track that has been on a team keeps it: one crop half full
                # of grass is not a change of shirt.
                side = held.side
            known = self._named.get(tid)
            if known is not None:
                side = known.side
            track = Track(
                id=tid,
                side=side,
                box=box,
                number=None if known is None else known.number,
                name=None if known is None else known.name,
            )
            live[tid] = track
            tracks.append(track)

        self._live = live
        return tracks

    def identify(self, mark: int, side: Side, number: int, name: str, ts: float) -> None:
        """Bind a name to a tracked body, from something that actually read it.

        The one way a track ever gets a name. What did the reading is not this
        module's business — on the real system it is the caller, which can see
        a shirt number in a frame far better than anything that will run
        locally — and the tracker's contribution is that the name then stays
        on that body while the camera does, over frames where the number is
        turned away or too small to read.

        It overrides the kit split, because whatever read the number saw more
        than a histogram of a torso did.
        """
        self._named[mark] = _Identity(side=side, number=number, name=name, ts=ts)
        track = self._live.get(mark)
        if track is not None:
            track.side, track.number, track.name = side, number, name

    def reset(self) -> None:
        """Start the ids again at a cut.

        The picture after a cut is a different camera on a different part of the
        pitch, so no box in it continues a box from before. The names go with
        the ids — they are bound to a track and a name bound to one body must
        never be inherited by another. The registry keeps its names, because a
        name learned before the cut is still true after it, and so does the kit
        split, because the teams did not change shirts.
        """
        self._bytetrack = _new_bytetrack()
        self._base = self._high + 1
        self._live = {}
        self._named = {}

    def _detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        scale = WORK_WIDTH / w if w > WORK_WIDTH else 1.0
        if scale < 1.0:
            size = (WORK_WIDTH, max(1, int(round(h * scale))))
            import cv2

            small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        else:
            # A copy even when nothing is resized, so a detector that writes on
            # what it is given cannot scribble on the delay buffer's frame.
            small = image.copy()
        boxes: list[tuple[int, int, int, int]] = []
        for x0, y0, x1, y1 in self.detector.detect(small):
            boxes.append(
                (
                    int(round(x0 / scale)),
                    int(round(y0 / scale)),
                    int(round(x1 / scale)),
                    int(round(y1 / scale)),
                )
            )
        return boxes

    def _assign_ids(
        self, boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Which body is which, across passes.

        ByteTrack answers this, and answers it about where each body is going
        rather than where it was, which is the whole difference on footage
        where the camera moves. Detections arrive without a score — the
        detector has already applied its own threshold — so they all go in as
        high-confidence, and ByteTrack's second pass over low-scoring boxes
        does nothing. That pass is for recovering a body the detector nearly
        missed; recovering it wrongly would put a name on it.

        A box ByteTrack will not take responsibility for is dropped rather
        than given an id of our own: an id nothing is tracking is exactly the
        tag the caller must not be shown.
        """
        import numpy as np
        import supervision as sv

        if not boxes:
            self._bytetrack.update_with_detections(sv.Detections.empty())
            return []
        detections = sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            confidence=np.full(len(boxes), 0.9, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
        )
        tracked = self._bytetrack.update_with_detections(detections)
        if tracked.tracker_id is None:
            return []
        found: list[tuple[int, tuple[int, int, int, int]]] = []
        for box, tid in zip(tracked.xyxy, tracked.tracker_id, strict=True):
            x0, y0, x1, y1 = (int(round(float(v))) for v in box)
            mine = self._base + int(tid)
            self._high = max(self._high, mine)
            found.append((mine, (x0, y0, x1, y1)))
        return found

    def _sides(self, crops: list[np.ndarray], ts: float) -> list[Side]:
        """Which team each body is in, from the colour of its shirt.

        Every crop, every pass. The embedding version had to ration this —
        SigLIP cost more than the rest of the chain put together, so a track
        was embedded once and the vector carried for ten passes — and a
        histogram costs microseconds, so the rationing and the cache it
        needed are gone.
        """
        points = _unit(np.stack([_kit_histogram(crop) for crop in crops]))
        for point, crop in zip(points, crops, strict=True):
            self._samples.append(point)
            self._colours.append(_torso(crop).reshape(-1, 3).mean(axis=0))

        if self._due(ts):
            self._fit(ts)
        if self._centroids is None:
            return [Side.UNKNOWN] * len(crops)

        gaps = np.linalg.norm(points[:, None, :] - self._centroids[None, :, :], axis=2)
        sides: list[Side] = []
        for row in gaps:
            near = int(np.argmin(row))
            if float(row[near]) > KIT_DISTANCE:
                sides.append(Side.UNKNOWN)
            else:
                sides.append(Side.HOME if near == self._home_cluster else Side.AWAY)
        return sides

    def _due(self, ts: float) -> bool:
        if self._centroids is None:
            return len(self._samples) >= self.fit_samples
        return ts - self._fit_ts >= self.refit_s

    def _fit(self, ts: float) -> None:
        points = np.stack(list(self._samples))
        colours = np.stack(list(self._colours))
        labels, centroids = _kmeans(points)
        self._centroids = centroids
        self._fit_ts = ts
        means = [
            colours[labels == k].mean(axis=0) if bool(np.any(labels == k)) else colours.mean(axis=0)
            for k in (0, 1)
        ]
        self._home_cluster = self._pick_home(means)
        log.info(
            "kit split over %d crops: home is cluster %d (mean BGR %s), away cluster %d (%s)",
            len(points),
            self._home_cluster,
            means[self._home_cluster].round().astype(int).tolist(),
            1 - self._home_cluster,
            means[1 - self._home_cluster].round().astype(int).tolist(),
        )

    def _pick_home(self, means: list[np.ndarray]) -> int:
        """Which of the two clusters is the home team.

        The weakest link in the whole chain: a mean crop colour against the
        first colour word of a kit description a language model wrote. Stripes,
        floodlights and a keeper in a third colour all attack it, and when it is
        wrong it is wrong for every player at once — the names come out swapped
        between the teams. It survives because the alternative is asking a
        vision model per fit, and because the pack's kit words are checked by a
        human before kickoff.
        """
        home = _kit_colour(self.pack.home.kit) if self.pack is not None else None
        away = _kit_colour(self.pack.away.kit) if self.pack is not None else None
        if home is None or away is None:
            return 0
        straight = _colour_gap(means[0], home) + _colour_gap(means[1], away)
        swapped = _colour_gap(means[1], home) + _colour_gap(means[0], away)
        return 0 if straight <= swapped else 1


class NullTracker:
    """No marks at all: the ablation, as a substitution rather than a flag.

    The point of the ablation is that everything downstream runs exactly as it
    does with marks on, and sees no players. A flag would put an ``if`` in the
    runtime and the ablation would then be measuring that ``if`` as much as it
    measures the tracker.
    """

    def update(self, frame: Frame) -> list[Track]:
        return []

    def reset(self) -> None:
        return None

    def identify(self, mark: int, side: Side, number: int, name: str, ts: float) -> None:
        return None


def _new_bytetrack() -> Any:
    """A fresh ByteTrack, which is also how the tracker forgets a cut.

    The deprecation warning is swallowed here and nowhere else. supervision
    moves ByteTrack to a separate ``trackers`` package in 0.31 and we are
    pinned below that; what the warning has to say is in the pin's comment in
    pyproject, and it would otherwise print on every camera cut of every run.
    """
    import warnings

    import supervision as sv

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ByteTrack.*deprecated.*")
        return sv.ByteTrack(
            minimum_matching_threshold=MATCH_THRESHOLD,
            lost_track_buffer=int(round(LOST_PASSES * 30.0 / TRACK_RATE_HZ)),
            frame_rate=TRACK_RATE_HZ,
        )


class VisionExtraMissing(ImportError):
    """The local vision models are not installed."""


def _require(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:  # pragma: no cover - needs the extra to not exist
        raise VisionExtraMissing(
            f"player marks need {name}, which is in the optional vision extra: "
            "install it with `uv sync --extra vision`"
        ) from exc


@dataclass
class _RfDetr:
    """RF-DETR boxes, kept to the person class.

    RF-DETR rather than Ultralytics YOLO because RF-DETR is Apache-2.0 and YOLO
    is AGPL, which is not a licence this project can ship under.

    The nano variant, which is what ``default_tracker`` builds. ``RFDETRBase``
    is a deprecation proxy in current rfdetr and 355 MB of weights; nano is 62
    MB and found the same fifteen bodies on the wide shots of the test clip.
    Measured at 640 wide on this machine, MPS, warm: 92 ms a frame against
    144 ms for ``RFDETRSmall``.
    """

    model: Any
    threshold: float = 0.5
    #: RF-DETR labels with COCO's 91-class ids, where person is 1.
    person_class: int = 1

    def detect(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        found = self.model.predict(image, threshold=self.threshold)
        boxes: list[tuple[int, int, int, int]] = []
        for box, class_id in zip(found.xyxy, found.class_id, strict=True):
            if int(class_id) != self.person_class:
                continue
            x0, y0, x1, y1 = (int(round(float(v))) for v in box)
            boxes.append((x0, y0, x1, y1))
        return boxes


def default_tracker(
    pack: KnowledgePack | None = None,
) -> PlayerTracker:
    """The detector, imported here and nowhere else.

    Marks are off in most runs, and an import at module scope would make every
    one of them pay several seconds and a gigabyte of torch for a feature they
    are not using. Weights land in the library's own cache on first use, which
    is why no test may ever call this.

    One model now. PARSeq and SigLIP were here and are gone: on three minutes
    of real broadcast PARSeq confirmed no shirt number at all, and SigLIP's
    second per pass was what made the passes too far apart for a tag to mean
    the same body twice. rfdetr picks its own device and resolves to MPS on
    this machine, so there is nothing to pass.
    """
    rfdetr = _require("rfdetr")
    return PlayerTracker(_RfDetr(rfdetr.RFDETRNano()), pack=pack)
