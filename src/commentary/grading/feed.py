"""The play-by-play feed, read from a file, used only to grade.

This is the module the project's central claim rests on. The feed is the one
source of truth about the match that did not come off the screen, so it is
kept behind the grading wall and nothing in the runtime may import it. That
is also why there is no scraper here and nothing is fetched at import time:
a loader that reads a file somebody saved after the match cannot leak into a
live run, however badly the rest of the code is wired. A scraper could.

Two jobs. The first is reading a saved play-by-play into
:class:`~commentary.schemas.GroundTruthEvent`. The shape this project writes
is small and documented below; ESPN's own export is accepted too where doing
so costs a line, because that is what people actually have on disk.

The second is alignment, which is the part that quietly ruins evals. The feed
counts in match time and a trace counts in video time, and the only bridge
between them is the board reader's own clock readings. Fit that offset badly
and every recall number is wrong while the table still looks fine — so the
fit is a median rather than a least-squares line, one misread board cannot
move it, and the residual is reported next to the answer.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.schemas import Event, GroundTruthEvent, Side

#: ``37``, ``37'``, ``37:12``, ``45+2``, ``45'+2'``, ``90 + 4:30``.
CLOCK = re.compile(
    r"^\s*(\d{1,3})(?::(\d{1,2}))?\s*'?"
    r"(?:\s*\+\s*(\d{1,3})(?::(\d{1,2}))?\s*'?)?\s*$"
)

#: Rows whose type is one of these are dropped without complaint. They are
#: restarts and bookkeeping, not things a commentator is graded on missing.
IGNORED = frozenset(
    {"goal kick", "kick off", "kickoff", "half time", "full time", "end of match", "delay"}
)

#: Matched in order against the lowercased type text, first hit wins. Order is
#: load-bearing: "Goalkeeper Save" and "Goal Kick" both contain "goal", so the
#: narrower phrases have to be tried before the broad ones.
EVENT_WORDS: tuple[tuple[str, Event], ...] = (
    ("own goal", Event.GOAL),
    ("penalty - scored", Event.GOAL),
    ("penalty scored", Event.GOAL),
    ("penalty - missed", Event.PENALTY),
    ("penalty - saved", Event.PENALTY),
    ("save", Event.SAVE),
    ("goal", Event.GOAL),
    ("penalty", Event.PENALTY),
    ("yellow card", Event.CARD),
    ("red card", Event.CARD),
    ("booking", Event.CARD),
    ("card", Event.CARD),
    ("substitution", Event.SUBSTITUTION),
    ("sub ", Event.SUBSTITUTION),
    ("corner", Event.CORNER),
    ("offside", Event.OFFSIDE),
    ("free kick", Event.FREE_KICK),
    ("freekick", Event.FREE_KICK),
    ("throw", Event.THROW_IN),
    ("foul", Event.FOUL),
    ("shot", Event.SHOT),
    ("attempt", Event.SHOT),
    ("header", Event.SHOT),
)


class FeedError(ValueError):
    """The saved feed is not something this loader can read."""


@dataclass(frozen=True)
class Clock:
    """A match clock reading, with the half it belongs to.

    The half cannot be derived from the seconds alone: first-half stoppage
    reads ``45+2``, which is 2820 seconds played and would otherwise sort
    itself into the second half, where the alignment offset is different by
    the length of the interval.
    """

    seconds: float
    period: int
    text: str


@dataclass(frozen=True)
class FeedEvent:
    """One row of the play-by-play, normalised onto our own vocabulary."""

    clock: Clock
    event: Event
    side: Side = Side.UNKNOWN
    player: str | None = None
    home_score: int = 0
    away_score: int = 0
    text: str = ""


@dataclass
class Feed:
    """A saved play-by-play, plus what was thrown away reading it.

    ``skipped`` is here so a feed whose vocabulary this loader does not know
    reports itself instead of silently producing a short event list and a
    flattering recall number.
    """

    events: list[FeedEvent] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    source: str = ""

    def __len__(self) -> int:
        return len(self.events)


def parse_clock(text: str | None) -> Clock | None:
    """A printed match clock to seconds played. ``None`` if it is not one."""
    if not text:
        return None
    match = CLOCK.match(str(text))
    if match is None:
        return None
    minutes, seconds, added, added_seconds = match.groups()
    base = int(minutes)
    total = base * 60 + int(seconds or 0)
    if added is not None:
        total += int(added) * 60 + int(added_seconds or 0)
    stoppage = added is not None
    period = 1 if base < 45 or (base == 45 and stoppage) else 2
    return Clock(seconds=float(total), period=period, text=str(text).strip())


def parse_event(text: str | None) -> Event | None:
    """A feed's word for what happened to one of ours. ``None`` means drop it."""
    if not text:
        return None
    lowered = " ".join(str(text).lower().split())
    if lowered in IGNORED:
        return None
    for needle, event in EVENT_WORDS:
        if needle in lowered:
            return event
    return None


# -- reading a saved file ----------------------------------------------


def load_feed(path: Path) -> Feed:
    """Read a play-by-play somebody saved earlier. Never fetches anything.

    The shape this project writes is::

        {"events": [{"clock": "37:12", "type": "goal", "team": "home",
                     "player": "Bukayo Saka", "home_score": 1,
                     "away_score": 0, "text": "..."}]}

    ``clock`` and ``type`` are the only required fields. Everything else is
    filled in where it can be: scores run forward from the goals if the file
    does not carry them, and an absent team leaves the side unknown rather
    than guessing, because a guessed side is worse than no side at all.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FeedError(f"{path}: not JSON: {exc}") from exc
    feed = parse_feed(document)
    feed.source = str(path)
    return feed


def parse_feed(document: Any) -> Feed:
    """The body of :func:`load_feed`, for a document already in memory."""
    rows, context = _rows_and_context(document)
    feed = Feed()
    home = away = 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        clock = parse_clock(
            _scalar(_first(row, "clock", "time", "minute", "displayClock"), "displayValue", "value")
        )
        raw_type = _scalar(_first(row, "type", "event", "kind", "play_type"), "text", "name")
        event = parse_event(raw_type)
        if clock is None or event is None:
            label = " ".join(str(raw_type or "no type").lower().split())
            feed.skipped[label] = feed.skipped.get(label, 0) + 1
            continue

        side = _side(row, context)
        said_home = _int(_first(row, "home_score", "homeScore"))
        said_away = _int(_first(row, "away_score", "awayScore"))
        if said_home is None or said_away is None:
            if event is Event.GOAL and side is Side.HOME:
                home += 1
            elif event is Event.GOAL and side is Side.AWAY:
                away += 1
        else:
            home, away = said_home, said_away

        feed.events.append(
            FeedEvent(
                clock=clock,
                event=event,
                side=side,
                player=_player(row),
                home_score=home,
                away_score=away,
                text=str(_first(row, "text", "shortText", "description") or ""),
            )
        )

    feed.events.sort(key=lambda e: (e.clock.period, e.clock.seconds))
    return feed


def _rows_and_context(document: Any) -> tuple[list[Any], dict[str, str]]:
    """The event rows, and whatever the file says about which team is which."""
    if isinstance(document, list):
        return document, {}
    if not isinstance(document, dict):
        raise FeedError("feed must be a JSON object or a list of events")

    rows = _first(document, "events", "plays", "commentary", "items")
    if not isinstance(rows, list):
        raise FeedError("feed has no 'events' list")

    context: dict[str, str] = {}
    for key, side in (("home", "home"), ("away", "away")):
        named = _first(document, f"{key}_team", f"{key}Team", key)
        if isinstance(named, str):
            context[named.strip().lower()] = side
        elif isinstance(named, dict):
            for label in ("id", "name", "displayName", "abbreviation", "short"):
                value = named.get(label)
                if value:
                    context[str(value).strip().lower()] = side
        ident = _first(document, f"{key}_id", f"{key}TeamId")
        if ident:
            context[str(ident).strip().lower()] = side
    return rows, context


def _side(row: dict[str, Any], context: dict[str, str]) -> Side:
    raw = _first(row, "team", "side", "homeAway")
    if isinstance(raw, dict):
        raw = _first(raw, "homeAway", "id", "abbreviation", "displayName", "name")
    if raw is None:
        return Side.UNKNOWN
    token = str(raw).strip().lower()
    if token in ("home", "away"):
        return Side(token)
    resolved = context.get(token)
    return Side(resolved) if resolved else Side.UNKNOWN


def _player(row: dict[str, Any]) -> str | None:
    raw = _first(row, "player", "athlete", "scorer", "participants")
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    if isinstance(raw, dict):
        raw = _first(raw, "displayName", "fullName", "name", "athlete")
    if isinstance(raw, dict):
        raw = _first(raw, "displayName", "fullName", "name")
    if raw is None:
        return None
    name = str(raw).strip()
    return name or None


def _scalar(value: Any, *keys: str) -> str | None:
    """Unwrap the little objects ESPN wraps its scalars in.

    A clock arrives as ``{"displayValue": "37'"}`` and a type as
    ``{"text": "Goal - Header"}`` often enough that stringifying the dict
    and hoping is not good enough — it would turn "Goal Kick" into a goal.
    """
    if isinstance(value, dict):
        value = _first(value, *keys)
    if value is None:
        return None
    return str(value)


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# -- putting the feed on the video clock -------------------------------


@dataclass(frozen=True)
class Fix:
    """One board reading turned into a (match clock, video time) pair."""

    video_ts: float
    clock_s: float
    period: int

    @property
    def offset_s(self) -> float:
        return self.video_ts - self.clock_s


@dataclass
class Alignment:
    """Feed events on video time, and how much to believe them.

    ``residual_s`` is the median distance between a board reading and where
    the fitted offset says it should have been. It is the number to look at
    before any recall figure: an alignment four seconds out still produces a
    complete, plausible table in which every line has missed its event.
    """

    events: list[GroundTruthEvent] = field(default_factory=list)
    offsets: dict[int, float] = field(default_factory=dict)
    residual_s: float = 0.0
    worst_s: float = 0.0
    n: int = 0
    unusable: int = 0
    tolerance_s: float = 2.0

    @property
    def ok(self) -> bool:
        """Enough readings to fit at all, and they agree with each other."""
        return self.n >= 2 and self.residual_s <= self.tolerance_s

    @property
    def offset_s(self) -> float:
        """The first half's offset, or the only one there is."""
        if not self.offsets:
            return 0.0
        return self.offsets.get(1, next(iter(self.offsets.values())))

    def offset_for(self, period: int) -> float:
        return self.offsets.get(period, self.offset_s)

    def summary(self) -> str:
        if self.n == 0:
            return "alignment FAILED: no usable board readings; events left on match time"
        offsets = ", ".join(f"h{p} {o:+.1f}s" for p, o in sorted(self.offsets.items()))
        verdict = "ok" if self.ok else "SUSPECT"
        return (
            f"alignment {verdict}: {offsets} from {self.n} board reads "
            f"({self.unusable} unusable), residual {self.residual_s:.2f}s, "
            f"worst {self.worst_s:.2f}s"
        )


def align(
    events: list[FeedEvent],
    board_reads: list[dict[str, Any]],
    *,
    tolerance_s: float = 2.0,
    min_reads_per_period: int = 3,
) -> Alignment:
    """Restamp feed events onto video time using the board reader's clock.

    The offset is the median of ``video_ts - clock_s`` over every readable
    board row, fitted per half because the interval sits between them and a
    single offset would split the difference and be wrong in both. Median
    rather than a least-squares fit: the board reader misreads, and one
    reading of ``07:12`` as ``87:12`` would drag a fitted line far enough to
    move every event by several seconds while leaving no visible trace.

    Readings with no clock — the score bug is hidden during replays — are
    counted in ``unusable`` and otherwise ignored.
    """
    fixes: list[Fix] = []
    unusable = 0
    for row in board_reads:
        if not row.get("bug_visible", True):
            unusable += 1
            continue
        clock = parse_clock(row.get("clock"))
        ts = row.get("ts")
        if clock is None or not isinstance(ts, int | float):
            unusable += 1
            continue
        fixes.append(Fix(video_ts=float(ts), clock_s=clock.seconds, period=clock.period))

    if not fixes:
        return Alignment(
            events=[_truth(e, 0.0) for e in events],
            unusable=unusable,
            tolerance_s=tolerance_s,
        )

    overall = statistics.median(f.offset_s for f in fixes)
    offsets: dict[int, float] = {}
    for period in sorted({f.period for f in fixes}):
        same = [f.offset_s for f in fixes if f.period == period]
        offsets[period] = statistics.median(same) if len(same) >= min_reads_per_period else overall

    residuals = [abs(f.offset_s - offsets.get(f.period, overall)) for f in fixes]
    alignment = Alignment(
        offsets=offsets,
        residual_s=statistics.median(residuals),
        worst_s=max(residuals),
        n=len(fixes),
        unusable=unusable,
        tolerance_s=tolerance_s,
    )
    alignment.events = [_truth(e, alignment.offset_for(e.clock.period)) for e in events]
    return alignment


def shift(events: list[FeedEvent], offset_s: float) -> list[GroundTruthEvent]:
    """Restamp by hand, for a clip whose offset you already know."""
    return [_truth(event, offset_s) for event in events]


def _truth(event: FeedEvent, offset_s: float) -> GroundTruthEvent:
    return GroundTruthEvent(
        video_ts=event.clock.seconds + offset_s,
        event=event.event,
        side=event.side,
        player=event.player,
        home_score=event.home_score,
        away_score=event.away_score,
    )
