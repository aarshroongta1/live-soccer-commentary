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
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    MatchState,
    NamedEvent,
    Possession,
    Scene,
    Side,
    WireEvent,
)

#: "37:12", "45", "45+2", "45+2:13", with an optional broadcast apostrophe.
_CLOCK_RE = re.compile(r"^(\d{1,3})(?:\s*\+\s*(\d{1,2}))?(?::(\d{1,2}))?$")

#: Longer than any match clock ever reaches, so anything past it is junk.
_MAX_MINUTES = 130

#: How long a possession is worth reporting. Past it the caller is being told
#: who had the ball in a passage of play that has since ended.
BALL_FRESH_S = 6.0

#: How long a named event stays in "just now". A foul is news for about this
#: long and then it is history.
JUST_NOW_S = 20.0

#: The last few named events, so a quiet spell does not scroll the prompt.
MAX_NAMED = 6

#: A wire goal this close to a board goal on the same side is the same goal,
#: and the wire is naming it rather than reporting a second one.
SAME_GOAL_S = 15.0

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
    runs the other way, and a test can hand in any object with these five
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

    @property
    def bug_missing(self) -> bool: ...


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

    def confidence(self, number: int, ts: float, side: Side = Side.UNKNOWN) -> float:
        """How much that name is still worth, given how long ago it was read."""
        belief = self._resolve(number, side)
        if belief is None:
            return 0.0
        age = max(0.0, ts - belief.ts)
        decay = math.exp(-math.log(2.0) * age / self.half_life_s)
        return belief.strength * decay

    def identified(
        self, ts: float, *, min_confidence: float = 0.35
    ) -> dict[Side, list[tuple[int, str]]]:
        """Numbers a *sighting* has put a name to, by side, lowest number first.

        Roster seeds are left out. A squad list says who might play, and the
        caller already has both of them printed in full above the frames; a
        line naming all forty would tell it nothing and cost it attention.
        What belongs here is the handful of shirts something actually read.
        """
        found: dict[Side, list[tuple[int, str]]] = {}
        for belief in self._beliefs.values():
            if belief.strength <= ROSTER_STRENGTH or belief.side is Side.UNKNOWN:
                continue
            if self.confidence(belief.number, ts, belief.side) < min_confidence:
                continue
            found.setdefault(belief.side, []).append((belief.number, belief.name))
        for numbered in found.values():
            numbered.sort()
        return found

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
        pack: KnowledgePack | None = None,
    ) -> None:
        self.state = MatchState(home=home, away=away)
        self.registry = registry if registry is not None else EntityRegistry()
        self.max_events = max_events
        self.pack = pack
        #: What each side is called in one word, for the lines of summary
        #: where the full name would be most of the line.
        self.shorts = {
            Side.HOME: (pack.home.short or pack.home.name) if pack is not None else home,
            Side.AWAY: (pack.away.short or pack.away.name) if pack is not None else away,
        }
        #: A pass names the passer now and the recipient when the ball gets
        #: there. Held rather than applied, so possession is right at every
        #: cursor time in between instead of jumping forward.
        self._handover: tuple[float, Possession] | None = None

    @classmethod
    def from_pack(cls, pack: KnowledgePack, *, kickoff_ts: float = 0.0) -> MatchStateTracker:
        registry = EntityRegistry()
        registry.seed(pack, kickoff_ts)
        return cls(pack.home.name, pack.away.name, registry=registry, pack=pack)

    def apply_board(self, source: ConfirmedBoard) -> None:
        """Take the score, clock and replay flag from the board. Only from the board.

        Takes the confirmed tracker rather than a single :class:`BoardRead`,
        so three reads have had to agree before anything here moves.
        """
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
        self.state.bug_visible = not source.bug_missing

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

    # -- the wire ------------------------------------------------------

    def apply_wire(self, event: WireEvent, ts: float) -> str | None:
        """Take one statistician's event. Returns what changed, for the trace.

        This is the only path other than the board that may move the score,
        and it may because a feed is not a model: it is not guessing at the
        picture, it is reporting the match. Everything else it carries is
        what the picture cannot give — who passed to whom, who was fouled,
        whose card it is — and none of it touches the scoreline.

        Possession answers ``None``. It changes twice a second and a trace
        row for each would bury the corrections that matter.
        """
        self._settle_ball(ts)
        if event.event is Event.PASS:
            self._pass(event, ts)
            return None
        if event.event in (Event.CARRY, Event.SHOT):
            self._carry(event, ts)
            return None
        if event.event is Event.GOAL:
            return self._wire_goal(event, ts)
        if event.event in (Event.CARD, Event.SUBSTITUTION):
            return self._wire_incident(event, ts)
        self._wire_named(event, ts)
        return None

    def _pass(self, event: WireEvent, ts: float) -> None:
        if event.player is None:
            return
        self.state.ball = Possession(player=event.player, side=event.side, since_ts=ts)
        if event.recipient is not None:
            self._handover = (
                ts + event.duration_s,
                Possession(
                    player=event.recipient,
                    side=event.side,
                    since_ts=ts + event.duration_s,
                    from_player=event.player,
                ),
            )

    def _carry(self, event: WireEvent, ts: float) -> None:
        if event.player is None:
            return
        held = self.state.ball
        if held is None or held.player != event.player:
            self.state.ball = Possession(player=event.player, side=event.side, since_ts=ts)
        if event.event is Event.SHOT:
            self._note_event(Event.SHOT)

    def _wire_goal(self, event: WireEvent, ts: float) -> str | None:
        """The score, and a name on a goal the board may already have seen.

        The board sees a graphic change; the wire saw the ball cross the
        line. When both report the same goal the second one to arrive is not
        news, it is the missing half of the first — so it names the existing
        incident instead of inventing a second goal.
        """
        before = (self.state.home_score, self.state.away_score)
        self.state.home_score, self.state.away_score = event.home_score, event.away_score
        self._note_event(Event.GOAL)

        for incident in reversed(self.state.incidents):
            if (
                incident.event is Event.GOAL
                and incident.side is event.side
                and abs(incident.video_ts - ts) <= SAME_GOAL_S
            ):
                if incident.player is None:
                    incident.player = event.player
                    return f"goal {event.side.value} is {event.player}"
                return None

        self.state.incidents.append(
            Incident(
                event=Event.GOAL,
                side=event.side,
                player=event.player,
                video_ts=ts,
                source="wire",
            )
        )
        after = (self.state.home_score, self.state.away_score)
        return f"score {before[0]}-{before[1]} -> {after[0]}-{after[1]}"

    def _wire_incident(self, event: WireEvent, ts: float) -> str:
        self.state.incidents.append(
            Incident(
                event=event.event,
                side=event.side,
                player=event.player,
                video_ts=ts,
                source="wire",
            )
        )
        self._note_event(event.event)
        if event.event is Event.SUBSTITUTION and event.player is not None:
            self._believe_by_name(event.player, event.side, ts)
        detail = f" {event.detail}" if event.detail else ""
        return f"{event.event.value}{detail} {event.side.value} {event.player or 'unknown'}"

    def _wire_named(self, event: WireEvent, ts: float) -> None:
        self.state.named.append(
            NamedEvent(
                event=event.event,
                side=event.side,
                player=event.player,
                recipient=event.recipient,
                detail=event.detail,
                video_ts=ts,
            )
        )
        del self.state.named[:-MAX_NAMED]
        self._note_event(event.event)

    def _believe_by_name(self, name: str, side: Side, ts: float) -> None:
        """A substitute whose number is in the pack is a shirt we can now read."""
        sheet = self.pack.team(side) if self.pack is not None else None
        if sheet is None:
            return
        for player in sheet.squad:
            if player.name == name and player.number is not None:
                self.registry.believe(player.number, player.name, ts, side=side)
                return

    def _note_event(self, event: Event) -> None:
        self.state.last_events.append(event)
        del self.state.last_events[: -self.max_events]

    def _settle_ball(self, ts: float) -> None:
        """Let a pass reach its recipient, once the cursor has reached them."""
        if self._handover is None:
            return
        when, possession = self._handover
        if ts >= when:
            self.state.ball = possession
            self._handover = None

    def summary(self, ts: float = 0.0) -> str:
        """A few lines of state for the top of a prompt. It goes in every call.

        ``ts`` is video time, and it is here for the identity line: a shirt
        read twenty minutes ago is worth less than one read twenty seconds
        ago, and the registry needs to know when "now" is to say so.
        """
        self._settle_ball(ts)
        state = self.state
        lines = [state.scoreline]
        if holder := self._ball_line(ts):
            lines.append(holder)
        clock = state.clock or "clock unseen"
        lines.append(f"{clock} ({PERIOD_NAMES.get(state.period, f'period {state.period}')})")
        if state.in_replay:
            lines.append("screen: replay, not live play")
        elif not state.bug_visible:
            # Said plainly rather than left as a silent replay, because a
            # caller told it is watching a replay says nothing at all.
            lines.append("screen: no score bug visible, so the score and clock may be stale")
        if state.possession is not Side.UNKNOWN:
            holder = state.home if state.possession is Side.HOME else state.away
            lines.append(f"possession: {holder}")
        identified = self._identified_line(ts)
        if identified:
            lines.append(identified)
        lines.extend(self._just_now(ts))
        lines.extend(self._so_far())
        if state.last_events:
            recent = ", ".join(e.value for e in state.last_events[-5:])
            lines.append(f"recent: {recent}")
        return "\n".join(lines)

    def _ball_line(self, ts: float) -> str:
        """``on the ball: Mac Allister (ARG), from Otamendi``, while it is true.

        Stale possession is worse than none: a caller told who has the ball
        will say so, and a name six seconds old belongs to a passage of play
        that has already ended.
        """
        ball = self.state.ball
        if ball is None or ts - ball.since_ts > BALL_FRESH_S:
            return ""
        side = self.shorts.get(ball.side, ball.side.value)
        line = f"on the ball: {ball.player} ({side})"
        return f"{line}, from {ball.from_player}" if ball.from_player else line

    def _just_now(self, ts: float) -> list[str]:
        """The named events of the last twenty seconds, newest last."""
        recent = [e for e in self.state.named if 0.0 <= ts - e.video_ts <= JUST_NOW_S]
        if not recent:
            return []
        return ["just now:", *(f"  {self._named_line(e)}" for e in recent)]

    def _named_line(self, event: NamedEvent) -> str:
        side = self.shorts.get(event.side, event.side.value)
        who = f"{event.player} ({side})" if event.player else side
        if event.event is Event.FOUL:
            line = f"foul by {who}"
            return f"{line} on {event.recipient}" if event.recipient else line
        if event.detail:
            return f"{event.detail} {event.event.value} {who}"
        return f"{event.event.value} {who}"

    def _so_far(self) -> list[str]:
        """Goals, cards and subs, one line each, for as long as the match lasts."""
        lines: list[str] = []
        kinds = (("goals", Event.GOAL), ("cards", Event.CARD), ("subs", Event.SUBSTITUTION))
        for label, kind in kinds:
            found = [i for i in self.state.incidents if i.event is kind]
            if not found:
                continue
            who = ", ".join(
                f"{i.player or 'unknown'} ({self.shorts.get(i.side, i.side.value)})" for i in found
            )
            lines.append(f"{label}: {who}")
        return lines

    def _identified_line(self, ts: float) -> str:
        """``identified: ARG 11 Di María, 7 De Paul · FRA 10 Mbappé``.

        The labels drawn on the frames go off screen the moment the camera
        moves, and the caller is asked about a moment several seconds later.
        This is the same knowledge in a form that survives the cut.
        """
        by_side = self.registry.identified(ts)
        if not by_side:
            return ""
        parts: list[str] = []
        for side in (Side.HOME, Side.AWAY):
            numbered = by_side.get(side)
            if not numbered:
                continue
            who = ", ".join(f"{number} {name}" for number, name in numbered)
            parts.append(f"{self.shorts.get(side, side.value)} {who}")
        return "identified: " + " · ".join(parts) if parts else ""
