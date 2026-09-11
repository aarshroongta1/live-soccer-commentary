"""The play-by-play feed, read from a file, used only to grade.

This is the module the project's central claim rests on. The feed is the one
source of truth about the match that did not come off the screen, so it is
kept behind the grading wall and nothing in the runtime may import it. That
is also why there is no scraper here and nothing is fetched at import time:
a loader that reads a file somebody saved after the match cannot leak into a
live run, however badly the rest of the code is wired. A scraper could.

Two jobs. The first is reading a saved play-by-play into
:class:`~commentary.schemas.GroundTruthEvent`. One shape is read, the one
this project writes, and it is documented on :func:`load_feed`. A feed that
arrives in somebody else's shape is converted to ours before it gets here,
where the conversion is visible, rather than inside a loader whose tolerance
would have to be trusted.

The second is alignment, which is the part that quietly ruins evals. The feed
counts in match time and a trace counts in video time, and the only bridge
between them is the board reader's own clock readings. Fit that offset badly
and every recall number is wrong while the table still looks fine — so the
fit is a median rather than a least-squares line, one misread board cannot
move it, and the residual is reported next to the answer.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary import state
from commentary.schemas import Event, GroundTruthEvent, Side

#: The vocabulary a row's ``type`` may use: our own event names, plus the two
#: card colours, because a saved feed writes those rather than "card".
#: Anything else is counted in :attr:`Feed.skipped` rather than guessed at.
EVENT_WORDS: dict[str, Event] = {
    "goal": Event.GOAL,
    "own goal": Event.GOAL,
    "shot": Event.SHOT,
    "save": Event.SAVE,
    "corner": Event.CORNER,
    "free kick": Event.FREE_KICK,
    "penalty": Event.PENALTY,
    "foul": Event.FOUL,
    "offside": Event.OFFSIDE,
    "throw in": Event.THROW_IN,
    "card": Event.CARD,
    "yellow card": Event.CARD,
    "red card": Event.CARD,
    "substitution": Event.SUBSTITUTION,
}


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
    """A printed match clock to seconds played. ``None`` if it is not one.

    The reading itself is :func:`commentary.state.parse_clock`, the same one
    the board reader's clock goes through at runtime — a feed and a board
    print the clock the same way, and two parsers would be two chances to
    disagree about what "45+2" means. What is added here is the half, which
    the seconds alone cannot give.
    """
    if not text:
        return None
    cleaned = str(text)
    seconds = state.parse_clock(cleaned)
    period = state.period_for_clock(cleaned)
    if seconds is None or period is None:
        return None
    return Clock(seconds=seconds, period=period, text=cleaned.strip())


def parse_event(text: str | None) -> Event | None:
    """A feed's word for what happened to one of ours. ``None`` means drop it."""
    if not text:
        return None
    return EVENT_WORDS.get(" ".join(str(text).lower().split()))


# -- reading a saved file ----------------------------------------------


def load_feed(path: Path) -> Feed:
    """Read a play-by-play somebody saved earlier. Never fetches anything.

    The shape this project writes is::

        {"home_team": "Arsenal", "away_team": "Real Madrid",
         "events": [{"clock": "37:12", "type": "goal", "team": "home",
                     "player": "Bukayo Saka", "home_score": 1,
                     "away_score": 0, "text": "..."}]}

    ``clock`` and ``type`` are the only required fields. Everything else is
    filled in where it can be: scores run forward from the goals if the file
    does not carry them, and an absent team leaves the side unknown rather
    than guessing, because a guessed side is worse than no side at all.
    ``team`` may be "home"/"away" or either of the two team names.
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
    if not isinstance(document, dict):
        raise FeedError("feed must be a JSON object with an 'events' list")
    rows = document.get("events")
    if not isinstance(rows, list):
        raise FeedError("feed has no 'events' list")
    sides = _sides(document)

    feed = Feed()
    home = away = 0
    for row in rows:
        raw_type = row.get("type")
        event = parse_event(raw_type)
        clock = parse_clock(row.get("clock"))
        if clock is None or event is None:
            label = " ".join(str(raw_type or "no type").lower().split())
            feed.skipped[label] = feed.skipped.get(label, 0) + 1
            continue

        side = _side(row.get("team"), sides)
        said_home, said_away = row.get("home_score"), row.get("away_score")
        if isinstance(said_home, int) and isinstance(said_away, int):
            home, away = said_home, said_away
        elif event is Event.GOAL and side is Side.HOME:
            home += 1
        elif event is Event.GOAL and side is Side.AWAY:
            away += 1

        player = str(row.get("player") or "").strip()
        feed.events.append(
            FeedEvent(
                clock=clock,
                event=event,
                side=side,
                player=player or None,
                home_score=home,
                away_score=away,
                text=str(row.get("text") or ""),
            )
        )

    feed.events.sort(key=lambda e: (e.clock.period, e.clock.seconds))
    return feed


def _sides(document: dict[str, Any]) -> dict[str, Side]:
    """Team name to side, so a row may name the team instead of saying "home"."""
    named: dict[str, Side] = {}
    for key, side in (("home_team", Side.HOME), ("away_team", Side.AWAY)):
        value = document.get(key)
        if isinstance(value, str) and value.strip():
            named[value.strip().lower()] = side
    return named


def _side(raw: Any, named: dict[str, Side]) -> Side:
    if raw is None:
        return Side.UNKNOWN
    token = str(raw).strip().lower()
    if token in ("home", "away"):
        return Side(token)
    return named.get(token, Side.UNKNOWN)


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
