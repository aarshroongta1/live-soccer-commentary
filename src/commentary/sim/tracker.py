"""The simulator's player tracker: the same type, fed by the truth.

RF-DETR will not find a coloured dot on a green rectangle, so the real
tracker sees nothing at all on this broadcast and the marks row of the
ablation table would be empty forever. This reads the same tracks straight
off the sim's ground truth instead, and it is a substitution rather than a
flag: the runtime holds a :class:`~commentary.perception.players.Tracker` and
cannot tell which one it has.

What it does not do is cheat about legibility. A number is carried on a track
only once the renderer has actually drawn it large enough to read, by the same
constant the renderer decides that with. A player in the far corner of a wide
shot is a body with no name here exactly as they would be on real footage,
which is what makes the sim's marks row worth reading at all.
"""

from __future__ import annotations

from commentary.capture.buffer import Frame
from commentary.perception.players import Track
from commentary.schemas import KnowledgePack, Side
from commentary.sim.match import MatchSim
from commentary.sim.render import NUMBER_LEGIBLE_RADIUS, BroadcastRenderer


class SimTracker:
    """Tracks from the sim's own state, legible only where the picture is."""

    def __init__(
        self,
        sim: MatchSim,
        renderer: BroadcastRenderer,
        *,
        pack: KnowledgePack | None = None,
    ) -> None:
        self.sim = sim
        self.renderer = renderer
        self.pack = pack if pack is not None else sim.knowledge_pack
        self._ids: dict[tuple[Side, int], int] = {}
        self._read: set[tuple[Side, int]] = set()
        self._named: dict[int, tuple[Side, int]] = {}

    def update(self, frame: Frame) -> list[Track]:
        state = self.sim.at(frame.ts)
        cam = self.renderer.camera(state)
        tracks: list[Track] = []
        for dot, (cx, cy), radius in self.renderer.dots_on_screen(state, cam):
            key = (dot.side, dot.number)
            if radius >= NUMBER_LEGIBLE_RADIUS:
                self._read.add(key)
            legible = key in self._read or self._ids.get(key) in self._named
            number = dot.number if legible else None
            name = self._name_for(dot.side, dot.number) if legible else None
            tracks.append(
                Track(
                    id=self._id_for(key),
                    side=dot.side,
                    box=(cx - radius, cy - radius, cx + radius, cy + radius),
                    number=number,
                    name=name,
                )
            )
        return tracks

    def reset(self) -> None:
        """New ids after a cut, and the shirts have to be read again.

        A name already learned stays in the registry the runtime owns, which
        is the point of keeping the two apart: the identity survives the cut,
        the track does not.
        """
        self._ids = {}
        self._read = set()
        self._named = {}

    def identify(self, mark: int, side: Side, number: int, name: str, ts: float) -> None:
        """The caller read a shirt the renderer drew too small to be legible.

        On the sim that is not a contradiction worth fighting: something read
        it, and on the real system the something is the caller looking at a
        frame. The dot keeps its number from here on, which is what the real
        tracker does too.
        """
        self._named[mark] = (side, number)

    def _id_for(self, key: tuple[Side, int]) -> int:
        if key not in self._ids:
            self._ids[key] = len(self._ids)
        return self._ids[key]

    def _name_for(self, side: Side, number: int) -> str | None:
        sheet = self.pack.team(side)
        if sheet is None:
            return None
        for player in sheet.squad:
            if player.number == number:
                return player.name
        return None
