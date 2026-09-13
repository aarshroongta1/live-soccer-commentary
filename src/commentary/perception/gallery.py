"""Keeping a name on a player after the camera has cut away from them.

A track id dies at every cut, and a broadcast cuts every few seconds, so a
name read off a shirt is worth about one shot. The reason that is fixable
here and not in general is that this is a **closed set**: the sheet says who
the twenty-two are, and the picture has already said which eleven a body
belongs to. Identity across a cut is an eleven-way classification against
samples collected during this match, not open-set re-identification of a
stranger.

The samples come from certainty and nowhere else. Every sighting the caller
read and the roster confirmed is a labelled picture of that player, in this
match's kit and this stadium's light, and that is what the gallery keeps. A
body with no number is then compared against the eleven, and named only when
one of them is both close enough and clearly closer than the next — a rule
with a margin in it, because the failure that matters is not missing a name,
it is calling somebody by a team-mate's.

What it cannot do, stated here so nobody expects it to. A thirty-pixel body
in a wide shot has nothing in it to embed and stays team-level. Two players
of a build in the same kit will fall inside the margin and stay unnamed,
which is the right answer. And on a three-minute clip the gallery has barely
warmed up: it is thin at minute five and full by half time, so the numbers a
short clip produces understate it.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from commentary.schemas import Side

log = logging.getLogger(__name__)

#: Samples kept per player. Eight is enough to average out one bad crop and
#: short enough that the gallery follows the light as the half wears on.
KEEP_PER_PLAYER = 8

#: A body must be at least this tall before it is worth embedding at all.
#: Below it the crop is a smudge of kit colour and the match is a coin toss
#: dressed up as a measurement.
MIN_BODY_PX = 90

#: How close the best match has to be, and how far clear of the second best.
#: Both are cosine similarity between unit vectors. The pair is the whole
#: safety argument: the threshold says "this is one of the players we have
#: seen", the margin says "and it is not the other ten".
MATCH_THRESHOLD = 0.75
MATCH_MARGIN = 0.08

#: Crops embedded on one classification pass. The embedder is the slowest
#: thing available to this system — about 70 ms a crop — and the tracker's
#: own pass is 114 ms, so an unbounded classification pass would halve the
#: rate at which tags mean anything. The tallest unnamed bodies go first
#: because they are the ones a viewer is being shown.
CROPS_PER_PASS = 4

#: Seconds between classification passes. Identity does not change between
#: frames and this is the one part of the chain that is expensive.
CLASSIFY_EVERY_S = 1.0

#: What a gallery name is worth against a number somebody actually read.
#: Lower, so a later number read overrides it rather than arguing with it.
GALLERY_STRENGTH = 0.6


class Embedder(Protocol):
    """One (n, d) float32 block of appearance vectors for n crops."""

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


@dataclass(frozen=True)
class Match:
    """One body, recognised, with the evidence that it was not a guess."""

    number: int
    name: str
    side: Side
    similarity: float
    margin: float


@dataclass
class Decision:
    """One classification, hit or miss, for the trace to report."""

    best: float
    margin: float
    matched: str | None


@dataclass
class Gallery:
    """Who each player has looked like so far, and who a body looks like now."""

    embedder: Embedder
    threshold: float = MATCH_THRESHOLD
    margin: float = MATCH_MARGIN
    keep: int = KEEP_PER_PLAYER
    _samples: dict[tuple[Side, int], deque[np.ndarray]] = field(default_factory=dict)
    _names: dict[tuple[Side, int], str] = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)

    @property
    def players(self) -> int:
        """How many players the gallery could recognise if it saw them now."""
        return len(self._samples)

    def learn(self, crop: np.ndarray, side: Side, number: int, name: str) -> None:
        """Keep what this player looked like, from a sighting that was read.

        The only way anything enters the gallery. A guess would poison the
        centroid it was added to and every body compared against it after.
        """
        vector = self._embed_one(crop)
        if vector is None:
            return
        key = (side, number)
        held = self._samples.setdefault(key, deque(maxlen=self.keep))
        held.append(vector)
        self._names[key] = name

    def classify(self, crops: list[tuple[np.ndarray, Side]]) -> list[Match | None]:
        """Who these bodies are, where the gallery is sure enough to say.

        One embedder call for the batch, because that is where the cost is.
        A body whose side is known is compared against that side's eleven; a
        body the kit split could not place is compared against everybody,
        which is a harder question and answered by the same two rules.
        """
        usable = [i for i, (crop, _) in enumerate(crops) if _tall_enough(crop)]
        if not usable or not self._samples:
            return [None] * len(crops)
        vectors = _unit(self.embedder.embed([crops[i][0] for i in usable]))
        found: list[Match | None] = [None] * len(crops)
        for vector, i in zip(vectors, usable, strict=True):
            found[i] = self._nearest(vector, crops[i][1])
        return found

    def _nearest(self, vector: np.ndarray, side: Side) -> Match | None:
        scored: list[tuple[float, tuple[Side, int]]] = []
        for key, held in self._samples.items():
            if side is not Side.UNKNOWN and key[0] is not side:
                continue
            centroid = np.mean(np.stack(list(held)), axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm == 0.0:
                continue
            scored.append((float(vector @ (centroid / norm)), key))
        if not scored:
            return None
        scored.sort(reverse=True)
        best, key = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        gap = best - second
        matched = best >= self.threshold and gap >= self.margin
        self.decisions.append(
            Decision(round(best, 4), round(gap, 4), self._names[key] if matched else None)
        )
        if not matched:
            return None
        return Match(key[1], self._names[key], key[0], best, gap)

    def _embed_one(self, crop: np.ndarray) -> np.ndarray | None:
        if not _tall_enough(crop):
            return None
        return np.asarray(_unit(self.embedder.embed([crop]))[0])

    def drain(self) -> dict[str, Any]:
        """The decisions since the last look, as one trace row's worth."""
        made = self.decisions
        self.decisions = []
        hits = [d for d in made if d.matched is not None]
        return {
            "players": self.players,
            "classified": len(made),
            "hits": len(hits),
            "margins": [d.margin for d in made],
            "named": [d.matched for d in hits],
        }


def _tall_enough(crop: np.ndarray) -> bool:
    return bool(crop.shape[0] >= MIN_BODY_PX)


def _unit(vectors: np.ndarray) -> np.ndarray:
    points = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return np.asarray(points / np.where(norms == 0.0, 1.0, norms))
