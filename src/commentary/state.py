"""What the system believes about the match, and where each belief came from.

Two sources feed this and nothing else does: the board reader, which owns the
score and the clock, and the caller, which owns what is happening on the pitch.
Keeping the two separate is the point of the design — a vision model that
thinks it saw a goal must not be able to move the scoreline, because a model
that can move the scoreline can talk itself into anything afterwards.

The entity registry is the same idea applied to names. A shirt number is only
worth a name for as long as the sighting is fresh: the number 9 that a graphic
named at minute four belongs to somebody else after the substitution at minute
seventy, and nothing on screen announces the handover.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from commentary.schemas import (
    BoardRead,
    CallerLine,
    Event,
    KnowledgePack,
    MatchState,
    Scene,
    Side,
)

#: "37:12", "45", "45+2", "45+2:13", with an optional broadcast apostrophe.
_CLOCK_RE = re.compile(r"^(\d{1,3})(?:\s*\+\s*(\d{1,2}))?(?::(\d{1,2}))?$")

#: Longer than any match clock ever reaches, so anything past it is junk.
_MAX_MINUTES = 130

PERIOD_NAMES = {
    1: "first half",
    2: "second half",
    3: "extra time, first half",
    4: "extra time, second half",
}


def _clock_parts(text: str | None) -> tuple[int, int, int] | None:
    if text is None:
        return None
    cleaned = text.strip().strip("'’\"").strip()
    match = _CLOCK_RE.match(cleaned)
    if match is None:
        return None
    minutes = int(match.group(1))
    added = int(match.group(2)) if match.group(2) else 0
    seconds = int(match.group(3)) if match.group(3) else 0
    if minutes > _MAX_MINUTES or seconds >= 60 or added > 30:
        return None
    return minutes, added, seconds


def parse_clock(text: str | None) -> float | None:
    """Seconds played, from the clock exactly as the broadcast prints it.

    Stoppage time adds on top of the base minute rather than replacing it, so
    "45+2" is two minutes past the forty-five and sorts after it — which is
    what the eval needs when it aligns spoken lines to a feed by match time.
    Anything that does not look like a clock comes back as None rather than as
    a plausible-looking number.
    """
    parts = _clock_parts(text)
    if parts is None:
        return None
    minutes, added, seconds = parts
    return float((minutes + added) * 60 + seconds)


def period_for_clock(text: str | None) -> int | None:
    """Which period the clock is in, read from the text and not just the total.

    The "+" matters: 45+2 is still the first half even though it is past
    forty-five minutes, and 90+3 is still the second. Without looking at the
    raw form, every match would gain a half during first-half stoppage.
    """
    parts = _clock_parts(text)
    if parts is None:
        return None
    minutes, added, _ = parts
    if minutes < 45:
        return 1
    if minutes == 45 and added:
        return 1
    if minutes < 90:
        return 2
    if minutes == 90 and added:
        return 2
    if minutes < 105:
        return 3
    if minutes == 105 and added:
        return 3
    return 4


class ConfirmedBoard(Protocol):
    """The part of a board tracker that match state is allowed to read.

    Stated structurally so state does not import perception: the dependency
    runs the other way, and a test can hand in any object with these four
    attributes.
    """

    @property
    def home_score(self) -> int | None: ...

    @property
    def away_score(self) -> int | None: ...

    @property
    def clock(self) -> str | None: ...

    @property
    def in_replay(self) -> bool: ...


@dataclass(frozen=True)
class Belief:
    """One "that number is that player", and how good the evidence was."""

    number: int
    name: str
    ts: float
    side: Side = Side.UNKNOWN
    #: A roster is a list of who might play; a graphic is a sighting of who is
    #: playing. They should not be believed equally.
    strength: float = 1.0


#: Squad lists say who is available, not who is on the pitch right now.
ROSTER_STRENGTH = 0.7

#: Twenty-five minutes for a sighting to be worth half of what it was.
DEFAULT_HALF_LIFE_S = 1500.0


class EntityRegistry:
    """Shirt numbers to names, held with a confidence that fades.

    Seeded from the knowledge pack at kickoff and then corrected all match by
    whatever the caller can actually read off a graphic. A later sighting
    overwrites an earlier one for the same number without ceremony, because
    that is precisely what a substitution looks like from the outside: the same
    number, a different player, no announcement.
    """

    def __init__(self, *, half_life_s: float = DEFAULT_HALF_LIFE_S) -> None:
        self.half_life_s = half_life_s
        self._beliefs: dict[tuple[Side, int], Belief] = {}

    def seed(self, pack: KnowledgePack, ts: float = 0.0) -> None:
        """Take the rosters as a starting point, held loosely."""
        for side in (Side.HOME, Side.AWAY):
            sheet = pack.team(side)
            if sheet is None:
                continue
            for player in sheet.squad:
                if player.number is None:
                    continue
                self.believe(
                    player.number, player.name, ts, side=side, strength=ROSTER_STRENGTH
                )

    def believe(
        self,
        number: int,
        name: str,
        ts: float,
        *,
        side: Side = Side.UNKNOWN,
        strength: float = 1.0,
    ) -> None:
        self._beliefs[(side, number)] = Belief(
            number=number, name=name, ts=ts, side=side, strength=strength
        )

    def _candidates(self, number: int, side: Side) -> list[Belief]:
        found = [b for b in self._beliefs.values() if b.number == number]
        if side is not Side.UNKNOWN:
            found = [b for b in found if b.side in (side, Side.UNKNOWN)]
        found.sort(key=lambda b: b.ts, reverse=True)
        return found

    def _resolve(self, number: int, side: Side = Side.UNKNOWN) -> Belief | None:
        """The freshest belief for a number, unless two sides tie for it.

        Both teams have a number 9. Asking without naming a side is answerable
        only when one sighting is more recent than the other; two roster seeds
        from the same kickoff are a genuine tie and get no answer at all.
        """
        found = self._candidates(number, side)
        if not found:
            return None
        if len(found) > 1 and found[0].ts == found[1].ts and found[0].name != found[1].name:
            return None
        return found[0]

    def name_for(self, number: int, side: Side = Side.UNKNOWN) -> str | None:
        belief = self._resolve(number, side)
        return None if belief is None else belief.name

    def number_for(self, name: str, side: Side = Side.UNKNOWN) -> int | None:
        """The number wearing this name, matched on the full name or a surname."""
        wanted = name.strip().lower()
        if not wanted:
            return None
        found = [
            b
            for b in self._beliefs.values()
            if b.name.lower() == wanted or b.name.rsplit(" ", 1)[-1].lower() == wanted
        ]
        if side is not Side.UNKNOWN:
            found = [b for b in found if b.side in (side, Side.UNKNOWN)]
        if not found:
            return None
        return max(found, key=lambda b: b.ts).number

    def confidence(self, number: int, ts: float, side: Side = Side.UNKNOWN) -> float:
        """How much that name is still worth, given how long ago it was read."""
        belief = self._resolve(number, side)
        if belief is None:
            return 0.0
        age = max(0.0, ts - belief.ts)
        decay = math.exp(-math.log(2.0) * age / self.half_life_s)
        return belief.strength * decay

    def on_pitch(self, ts: float, *, min_confidence: float = 0.35) -> dict[str, str]:
        """Numbers still worth naming, in the shape ``MatchState`` wants them."""
        named: dict[str, str] = {}
        for number in sorted({b.number for b in self._beliefs.values()}):
            if self.confidence(number, ts) >= min_confidence:
                name = self.name_for(number)
                if name is not None:
                    named[str(number)] = name
        return named


#: "9", "#9", "9 Haaland", "Haaland (9)", "Haaland #9".
_SIGHTING_RES = (
    re.compile(r"^#?(\d{1,2})\s*[-—:.]?\s*([A-Za-z][\w'’.\- ]*)$"),
    re.compile(r"^([A-Za-z][\w'’.\- ]*?)\s*[(#]\s*(\d{1,2})\s*\)?$"),
)


def parse_sighting(text: str) -> tuple[int, str] | None:
    """Pull a number and a name out of whatever the caller wrote down.

    The caller reports names_read as free text because that is what it can
    honestly produce; a bare name or a bare number carries no pairing and is
    not a sighting at all.
    """
    cleaned = text.strip()
    for number_first, pattern in zip((True, False), _SIGHTING_RES, strict=True):
        match = pattern.match(cleaned)
        if match is None:
            continue
        number_s = match.group(1) if number_first else match.group(2)
        name = (match.group(2) if number_first else match.group(1)).strip()
        if not name:
            return None
        return int(number_s), name
    return None


class MatchStateTracker:
    """The single writable copy of what we believe, with the sources separated."""

    def __init__(
        self,
        home: str,
        away: str,
        *,
        registry: EntityRegistry | None = None,
        max_events: int = 8,
    ) -> None:
        self.state = MatchState(home=home, away=away)
        self.registry = registry if registry is not None else EntityRegistry()
        self.max_events = max_events

    @classmethod
    def from_pack(cls, pack: KnowledgePack, *, kickoff_ts: float = 0.0) -> MatchStateTracker:
        registry = EntityRegistry()
        registry.seed(pack, kickoff_ts)
        return cls(pack.home.name, pack.away.name, registry=registry)

    def apply_board(self, source: BoardRead | ConfirmedBoard) -> None:
        """Take the score, clock and replay flag from the board. Only from the board.

        A raw :class:`BoardRead` is trusted as given, which is fine in a test
        and wrong in a match: pass the tracker, so that three reads have had to
        agree before anything here moves.
        """
        if isinstance(source, BoardRead):
            home_score, away_score = source.home_score, source.away_score
            clock, in_replay = source.clock, not source.bug_visible
        else:
            home_score, away_score = source.home_score, source.away_score
            clock, in_replay = source.clock, source.in_replay

        if home_score is not None:
            self.state.home_score = home_score
        if away_score is not None:
            self.state.away_score = away_score
        if clock is not None:
            self.state.clock = clock
            self.state.clock_s = parse_clock(clock)
            period = period_for_clock(clock)
            if period is not None:
                self.state.period = period
        self.state.in_replay = in_replay

    def apply_caller(self, line: CallerLine, ts: float | None = None) -> None:
        """Take possession, events and name sightings from the caller.

        Deliberately unable to touch the scoreline. Replays are dropped rather
        than recorded: the second goal of a match should not become the third
        because the broadcast showed the first one again from behind the net.
        """
        if line.scene is Scene.REPLAY:
            return

        if line.side is not Side.UNKNOWN and line.scene is Scene.LIVE_PLAY:
            self.state.possession = line.side

        if line.event is not Event.NONE:
            self.state.last_events.append(line.event)
            del self.state.last_events[: -self.max_events]

        sighting_ts = ts if ts is not None else (self.state.clock_s or 0.0)
        seen = False
        for raw in line.names_read:
            sighting = parse_sighting(raw)
            if sighting is None:
                continue
            number, name = sighting
            self.registry.believe(number, name, sighting_ts, side=line.side)
            seen = True
        if seen:
            self.state.on_pitch = self.registry.on_pitch(sighting_ts)

    def summary(self) -> str:
        """A few lines of state for the top of a prompt. It goes in every call."""
        state = self.state
        lines = [state.scoreline]
        clock = state.clock or "clock unseen"
        lines.append(f"{clock} ({PERIOD_NAMES.get(state.period, f'period {state.period}')})")
        if state.in_replay:
            lines.append("screen: replay, not live play")
        if state.possession is not Side.UNKNOWN:
            holder = state.home if state.possession is Side.HOME else state.away
            lines.append(f"possession: {holder}")
        if state.last_events:
            recent = ", ".join(e.value for e in state.last_events[-5:])
            lines.append(f"recent: {recent}")
        return "\n".join(lines)
