"""Reading the scoreboard the way a human does: by glancing at it.

A commentator does not receive the score over a wire. They look at the corner
of the screen. So does this: a crop of the score bug goes to a cheap vision
model every couple of seconds, and three agreeing reads are required before
anything is believed. One hallucinated digit should never invent a goal.

The absence of the bug is information too. Broadcasters pull the score bug
during replays, and that is how the system knows not to call a replay as live
play — no replay detector, just a graphic that went away.

But only for as long as a replay lasts. A bug that is absent for minutes is
not a very long replay, it is a broadcast that does not carry one or a crop
pointed at the wrong corner, and reading that as a replay tells the caller to
stay quiet for the rest of the match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from commentary.capture.buffer import Frame
from commentary.config import BOARD_MODEL, SETTINGS, BoardConfig
from commentary.llm.base import Block, LLMBackend, encode_frame, image_block, text_block
from commentary.schemas import BoardRead, Side
from commentary.state import period_for_clock

#: Longer than any replay a broadcast cuts to. Past this the bug is not
#: pulled, it is not there: the wrong crop, or a feed that carries no score
#: graphic at all. The two call for opposite behaviour, so they are not
#: allowed to share a flag.
BUG_GONE_S = 30.0

BOARD_SYSTEM = """\
You read the score bug on a live soccer broadcast: the small graphic, usually \
in a corner, carrying the two team abbreviations, the score, and the match \
clock. You are shown a tight crop of where that graphic sits.

Report only what is printed in the image. Never infer the score from the play, \
never carry a number over from a previous look, never fill a gap with what \
would be plausible.

Rules:
- bug_visible: false whenever the crop does not contain a score bug. During a \
replay, a close-up, a full-screen graphic, or an ad break the bug is pulled, \
and its absence is exactly the signal this system needs. A wrong "true" with \
guessed numbers is far worse than an honest "false", so when the graphic is \
absent, obscured, or half-faded in a transition, say false and leave the \
other fields null.
- home_score / away_score: the two numbers as printed, home (the team listed \
first, on the left) first. Null if you cannot read them.
- clock: exactly as shown, with no reformatting. Keep stoppage-time forms as \
printed: "45+2", "45+2:13", "90+3". Keep "37:12" as "37:12". Null if the \
clock is not shown or is unreadable.
- confidence: how sure you are of this whole read, 0 to 1. Low confidence is \
useful — a read below the threshold is discarded rather than acted on.
"""

BOARD_QUESTION = "Read this score bug."


def crop_score_bug(image: np.ndarray, crop: tuple[float, float, float, float]) -> np.ndarray:
    """Cut the score-bug box out of a frame.

    The box is fractional rather than pixel-based so a preset survives a change
    of capture resolution: the same four numbers work on a 720p and a 1080p
    grab of the same broadcast.
    """
    x0, y0, x1, y1 = crop
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"crop must be an ordered box inside the unit square, got {crop!r}")
    h, w = image.shape[:2]
    px0, px1 = int(round(x0 * w)), int(round(x1 * w))
    py0, py1 = int(round(y0 * h)), int(round(y1 * h))
    px1 = max(px1, px0 + 1)
    py1 = max(py1, py0 + 1)
    return np.ascontiguousarray(image[py0:py1, px0:px1])


class BoardReader:
    """One vision call per look at the board.

    The crop is small, so it is encoded at a higher resolution and quality than
    the wide shots the caller sees: the whole job is telling a 3 from an 8, and
    a downscale that costs nothing on a pitch full of players is fatal here.
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: BoardConfig = SETTINGS.board,
        *,
        model: str = BOARD_MODEL,
        max_width: int = 1024,
        quality: int = 88,
    ) -> None:
        self.backend = backend
        self.config = config
        self.model = model
        self.max_width = max_width
        self.quality = quality

    async def read(self, frame: Frame) -> BoardRead:
        crop = crop_score_bug(frame.image, self.config.crop)
        jpeg = encode_frame(crop, quality=self.quality, max_width=self.max_width)
        blocks: list[Block] = [image_block(jpeg), text_block(BOARD_QUESTION)]
        parsed = await self.backend.parse(
            model=self.model,
            system=BOARD_SYSTEM,
            blocks=blocks,
            output_format=BoardRead,
            max_tokens=256,
            cache_system=True,
            tag="board",
        )
        return parsed.value


@dataclass(frozen=True)
class BoardPending:
    """Evidence being accumulated for a board state that is not believed yet."""

    home_score: int
    away_score: int
    period: int
    clock: str | None
    count: int
    first_ts: float

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.home_score, self.away_score, self.period)


@dataclass(frozen=True)
class BoardChange:
    """The board settled on something new.

    ``ts`` is the video time of the *first* of the agreeing reads, not the last.
    The goal happened when the board first showed it; confirmation is our
    caution, not the match's. Report the third read's time instead and every
    goal trigger fires late and the eval's lag numbers are wrong by a whole
    confirmation window.
    """

    ts: float
    home_score: int
    away_score: int
    clock: str | None
    period: int
    previous: tuple[int, int] | None

    @property
    def is_goal(self) -> bool:
        if self.previous is None:
            return False
        return (self.home_score, self.away_score) != self.previous

    @property
    def scoring_side(self) -> Side | None:
        """Whose goal it was, when the board can say.

        Both numbers moving between two reads is a board we cannot reason
        about — two goals in one confirmation window, or a misread — and
        naming a side for it would be inventing one.
        """
        if self.previous is None:
            return None
        home, away = self.previous
        if self.home_score > home and self.away_score == away:
            return Side.HOME
        if self.away_score > away and self.home_score == home:
            return Side.AWAY
        return None


class BoardTracker:
    """Turns a stream of single reads into something worth believing.

    Scores and periods move only on ``confirmations`` consecutive agreeing
    reads. The clock is different: it changes between every read by
    construction, so it can never be confirmed that way and is instead taken
    from the most recent read the model was confident about.
    """

    def __init__(self, config: BoardConfig = SETTINGS.board, *, replay_reads: int = 2) -> None:
        self.config = config
        #: Absent-bug reads needed before we believe a replay. Two, so a single
        #: dropped or mid-transition read cannot make the flag flap.
        self.replay_reads = replay_reads
        self._confirmed: tuple[int, int, int] | None = None
        self._pending: BoardPending | None = None
        self._clock: str | None = None
        self._absent_run = 0
        self._absent_since: float | None = None
        self._last_ts = 0.0
        self.last_change: BoardChange | None = None

    @property
    def home_score(self) -> int | None:
        return None if self._confirmed is None else self._confirmed[0]

    @property
    def away_score(self) -> int | None:
        return None if self._confirmed is None else self._confirmed[1]

    @property
    def period(self) -> int | None:
        return None if self._confirmed is None else self._confirmed[2]

    @property
    def clock(self) -> str | None:
        return self._clock

    @property
    def in_replay(self) -> bool:
        """The bug has been pulled, the way it is pulled for a replay."""
        return self._absent_run >= self.replay_reads and not self.bug_missing

    @property
    def bug_missing(self) -> bool:
        """The bug has been gone longer than any replay lasts.

        Measured from the first absent read rather than counted in reads, so
        a slower board interval does not move the threshold.
        """
        if self._absent_since is None:
            return False
        return self._last_ts - self._absent_since >= BUG_GONE_S

    @property
    def pending(self) -> BoardPending | None:
        """What the tracker is gathering evidence for right now, if anything."""
        return self._pending

    def update(self, read: BoardRead, ts: float) -> BoardChange | None:
        """Feed one read taken from the frame at video time ``ts``.

        Returns the change if this read is the one that confirmed it, otherwise
        None. A read the model was unsure about is discarded rather than
        counted against the run: an uncertain look is no evidence, in either
        direction.
        """
        if read.confidence < self.config.min_confidence:
            return None
        self._last_ts = ts

        if not read.bug_visible:
            self._absent_run += 1
            if self._absent_since is None:
                self._absent_since = ts
            # A replay interrupts the evidence: the half-formed score we were
            # accumulating belongs to a board we can no longer see.
            self._pending = None
            return None

        self._absent_run = 0
        self._absent_since = None
        if read.clock is not None:
            self._clock = read.clock
        if read.home_score is None or read.away_score is None:
            return None

        period = period_for_clock(read.clock)
        if period is None:
            period = self.period if self.period is not None else 1
        key = (read.home_score, read.away_score, period)

        if key == self._confirmed:
            self._pending = None
            return None

        if self._pending is not None and self._pending.key == key:
            pending = BoardPending(
                home_score=read.home_score,
                away_score=read.away_score,
                period=period,
                clock=read.clock or self._pending.clock,
                count=self._pending.count + 1,
                first_ts=self._pending.first_ts,
            )
        else:
            pending = BoardPending(
                home_score=read.home_score,
                away_score=read.away_score,
                period=period,
                clock=read.clock,
                count=1,
                first_ts=ts,
            )
        self._pending = pending

        if pending.count < self.config.confirmations:
            return None

        previous = None if self._confirmed is None else (self._confirmed[0], self._confirmed[1])
        change = BoardChange(
            ts=pending.first_ts,
            home_score=pending.home_score,
            away_score=pending.away_score,
            clock=pending.clock,
            period=pending.period,
            previous=previous,
        )
        self._confirmed = key
        self._pending = None
        self.last_change = change
        return change
