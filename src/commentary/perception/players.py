"""Putting names to the bodies on the pitch.

A commentator never says "the number seven". They say the name, because they
watched the teams run out and they can read a shirt. This does the same job the
only way a machine can: track every body, split the bodies by kit, read a shirt
number once while it happens to be legible, and look that number up in the team
sheet the researcher prepared.

None of that is Claude's work. A vision call per player per frame would cost
more than the rest of the system together and still be slower than a detector
built for the job, so the "is that a 7 or a 1" layer is small open models and
Claude stays the writer.

Every read is held cheaply and confirmed by agreement, the way the score bug is,
for the same reason: a number the system is wrong about becomes a name the
system is wrong about, and a confident wrong name is the worst thing either
voice can say.
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
#: they would on the full frame and cost a quarter as much. Crops go to the
#: number reader at full resolution, which is the only place the digits exist.
WORK_WIDTH = 640

#: Overlap at which a box is the same body as last frame, and keeps its id.
IOU_KEEP = 0.3

#: A box shorter than this is too far from the camera for a shirt number to
#: survive the crop. Pixels of the captured frame, set against a 720p capture.
MIN_NUMBER_HEIGHT_PX = 110

#: A read below this is thrown away rather than voted on. With the agreement
#: rule below it stands in for a legibility classifier: a model that is unsure
#: twice about the same digits is telling us the shirt was never readable.
NUMBER_MIN_CONFIDENCE = 0.8
NUMBER_VOTES = 5
NUMBER_AGREEMENTS = 2

#: How far a unit-length kit embedding may sit from the nearer centroid and
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


class Embedder(Protocol):
    """One (n, d) float32 block of appearance vectors for n crops."""

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


class NumberReader(Protocol):
    """Whatever text is on this crop, and how sure the model is of it."""

    def read(self, crop: np.ndarray) -> tuple[str, float]: ...


class Tracker(Protocol):
    """What the runtime holds. Two implementations, and they are substituted.

    The ablation that measures what the marks buy does not set a flag: it
    puts a :class:`NullTracker` where the real one goes, so the match loop
    has no branch in it and cannot drift between the two configurations.
    """

    def update(self, frame: Frame) -> list[Track]: ...

    def reset(self) -> None: ...


@dataclass
class Track:
    """One body, followed across frames, named once the evidence allows."""

    id: int
    side: Side
    box: tuple[int, int, int, int]
    number: int | None = None
    name: str | None = None


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    if overlap == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return overlap / float(area_a + area_b - overlap)


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0 = max(0, box[0]), max(0, box[1])
    x1, y1 = min(w, box[2]), min(h, box[3])
    if x1 <= x0 or y1 <= y0:
        return np.zeros((1, 1, 3), dtype=image.dtype)
    return np.ascontiguousarray(image[y0:y1, x0:x1])


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
    """Bodies in, named tracks out.

    The id assignment here is greedy IoU against the previous frame and nothing
    more: nearest box over a threshold keeps its id, everything else is new. A
    real pipeline would hand this to ByteTrack (via ``supervision``), which
    survives the occlusions a penalty box is made of. This is deliberately the
    simple version — it is honest about losing an id behind a crowd of players,
    and a lost id costs one re-read of a shirt, not a wrong name.
    """

    def __init__(
        self,
        detector: Detector,
        embedder: Embedder,
        reader: NumberReader,
        *,
        pack: KnowledgePack | None = None,
        fit_samples: int = 200,
        refit_s: float = 300.0,
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.reader = reader
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
        self._previous: list[tuple[int, tuple[int, int, int, int]]] = []
        self._live: dict[int, Track] = {}
        self._votes: dict[int, deque[int]] = {}
        self._next_id = 0

    def update(self, frame: Frame) -> list[Track]:
        """Every body in this frame, carrying whatever is known about each."""
        boxes = self._detect(frame.image)
        if not boxes:
            self._previous = []
            self._live = {}
            return []

        crops = [_crop(frame.image, box) for box in boxes]
        sides = self._sides(crops, frame.ts)
        ids = self._assign_ids(boxes)

        tracks: list[Track] = []
        live: dict[int, Track] = {}
        for tid, box, side, crop in zip(ids, boxes, sides, crops, strict=True):
            held = self._live.get(tid)
            if side is Side.UNKNOWN and held is not None:
                # A track that has been on a team keeps it: one crop half full
                # of grass is not a change of shirt.
                side = held.side
            track = Track(
                id=tid,
                side=side,
                box=box,
                number=None if held is None else held.number,
                name=None if held is None else held.name,
            )
            if track.number is None:
                self._read_number(track, crop, frame.ts)
            live[tid] = track
            tracks.append(track)

        self._live = live
        self._previous = list(zip(ids, boxes, strict=True))
        return tracks

    def reset(self) -> None:
        """Start the ids again at a cut.

        The picture after a cut is a different camera on a different part of the
        pitch, so no box in it continues a box from before. The number votes go
        with the ids — they are keyed by track id and a vote gathered on one
        body must never be counted towards another. The registry keeps its
        names, because a name learned before the cut is still true after it, and
        so does the kit split, because the teams did not change shirts.
        """
        self._previous = []
        self._live = {}
        self._votes = {}
        self._next_id = 0

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

    def _assign_ids(self, boxes: list[tuple[int, int, int, int]]) -> list[int]:
        matched: list[int | None] = [None] * len(boxes)
        taken: set[int] = set()
        pairs = sorted(
            (_iou(box, old), i, tid)
            for i, box in enumerate(boxes)
            for tid, old in self._previous
        )
        for overlap, i, tid in reversed(pairs):
            if overlap < IOU_KEEP or matched[i] is not None or tid in taken:
                continue
            matched[i] = tid
            taken.add(tid)
        ids: list[int] = []
        for kept in matched:
            if kept is None:
                kept = self._next_id
                self._next_id += 1
            ids.append(kept)
        return ids

    def _sides(self, crops: list[np.ndarray], ts: float) -> list[Side]:
        points = _unit(self.embedder.embed(crops))
        colours = np.stack([crop.reshape(-1, 3).mean(axis=0) for crop in crops])
        for point, colour in zip(points, colours, strict=True):
            self._samples.append(point)
            self._colours.append(colour)
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

    def _read_number(self, track: Track, crop: np.ndarray, ts: float) -> None:
        if track.box[3] - track.box[1] < MIN_NUMBER_HEIGHT_PX:
            return
        torso = crop[: max(1, crop.shape[0] // 2)]
        text, confidence = self.reader.read(torso)
        if confidence < NUMBER_MIN_CONFIDENCE:
            return
        text = text.strip()
        if not text.isdigit() or len(text) > 2:
            return
        number = int(text)
        if not 1 <= number <= 99:
            return

        votes = self._votes.setdefault(track.id, deque(maxlen=NUMBER_VOTES))
        votes.append(number)
        if votes.count(number) < NUMBER_AGREEMENTS:
            return

        track.number = number
        track.name = self._name_for(number, track.side)

    def _name_for(self, number: int, side: Side) -> str | None:
        """The squad's name for that number, or none and the caller says "#14"."""
        sheet = self.pack.team(side) if self.pack is not None else None
        if sheet is None:
            return None
        for player in sheet.squad:
            if player.number == number:
                return player.name
        return None


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


@dataclass
class _SigLip:
    """SigLIP image embeddings, which separate two kits without being told to."""

    model: Any
    preprocess: Any

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = importlib.import_module("torch")
        image_mod = importlib.import_module("PIL.Image")
        batch = torch.stack(
            [self.preprocess(image_mod.fromarray(crop[:, :, ::-1])) for crop in crops]
        )
        with torch.inference_mode():
            features = self.model.encode_image(batch)
        return np.asarray(features.cpu().numpy(), dtype=np.float32)


@dataclass
class _Parseq:
    """PARSeq on a 32x128 torso crop, which is the shape it was trained on."""

    model: Any

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        torch = importlib.import_module("torch")
        import cv2

        resized = cv2.resize(crop, (128, 32), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
        tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)
        tensor = tensor.float().div(255.0).sub(0.5).div(0.5).unsqueeze(0)
        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(-1)
        text, confidence = self.model.tokenizer.decode(probabilities)
        return str(text[0]), float(confidence[0].mean())


def default_tracker(
    pack: KnowledgePack | None = None,
) -> PlayerTracker:
    """The real models, imported here and nowhere else.

    Marks are off in most runs, and an import at module scope would make every
    one of them pay several seconds and a gigabyte of torch for a feature they
    are not using. Weights land in each library's own cache on first use, which
    is why no test may ever call this.
    """
    torch = _require("torch")
    rfdetr = _require("rfdetr")
    open_clip = _require("open_clip")
    siglip, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16-SigLIP", pretrained="webli"
    )
    return PlayerTracker(
        _RfDetr(rfdetr.RFDETRBase()),
        _SigLip(siglip.eval(), preprocess),
        _Parseq(torch.hub.load("baudm/parseq", "parseq", pretrained=True).eval()),
        pack=pack,
    )
