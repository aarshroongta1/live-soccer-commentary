"""The match loop: everything wired together, running against a clock.

Five things happen at once. Frames and sound pour into the ring buffers at
the live edge. The board reader glances at the score bug every couple of
seconds. The speak predictor ticks, and when it says so the caller looks at
the cursor and the near future and fills in a form. The fact gate judges that
form. The director decides who says what, and cuts someone off when a goal
goes in.

One subtlety governs the whole file. The board reader reads at the LIVE edge,
because that is how the system can know a goal went in before the narration
cursor has reached it. But a board change is not applied to match state until
the cursor passes the moment it happened, so the commentary can never
announce something the viewer has not seen yet. The early read is used only
as evidence: when the caller claims a goal, the gate asks whether the board
moved around that moment, which is exactly the question a human commentator
answers by glancing up at the graphic.

"Around that moment" is a fixed window, not the buffer depth. Tying it to
the buffer made the gate more permissive the longer we chose to wait, which
is a strange thing for a safety check to do and undid most of what the delay
was for.

The glance is not only at the settled board. Confirmation takes three
agreeing reads and lands well after the goal, so the gate also accepts a
board change the tracker is still gathering evidence for, and accepts a goal
the state has already taken in as cover for the lines that follow one. The
first read that disagrees with the settled score in a goal's direction is
also enough to *prompt* the caller, which is how the kick gets called as it
happens rather than seven seconds on. What the score itself is allowed to
move on does not change: three reads.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from commentary.agents.analyst import Analyst
from commentary.agents.caller import Caller
from commentary.agents.colour import (
    EXCITEMENT,
    ColourSeat,
    judge_utterance,
    mentions,
    one_subject,
    space_out,
)
from commentary.agents.phraser import Phraser, nameless_build_up, roster_names
from commentary.bus import Bus, Topic
from commentary.capture.audio import CutDetector
from commentary.capture.buffer import DelayBuffer, Frame
from commentary.config import SETTINGS, ReplayTalkConfig, Settings
from commentary.director import Director, next_beat_id
from commentary.gate import FactGate, claims_goal, facts_used, fold, is_the_same_name
from commentary.goalfollow import MAX_SYNTH, SYNTH_GAP_S, GoalFollowup
from commentary.ledger import CONTEXT_FACTS, Ledger
from commentary.ledger import Fact as LedgerFact
from commentary.llm.base import LLMBackend, Usage
from commentary.perception.board import BoardChange, BoardReader, BoardTracker
from commentary.predictor import SpeakPredictor
from commentary.schemas import (
    Beat,
    BoardRead,
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    MatchState,
    Player,
    Scene,
    Side,
    Sighting,
    SpeakDecision,
    Trigger,
    Voice,
)
from commentary.scoreline import Restatements, settle_numbers
from commentary.state import MatchStateTracker, parse_clock, period_for_clock
from commentary.threads import Offered, Threads
from commentary.tools import MatchTools
from commentary.trace import RunTrace
from commentary.voice.speaker import WORDS_PER_SECOND, LogSpeaker, Speaker
from commentary.wire import Wire, WireSync

#: How long after the ball crosses the line a broadcaster's score bug
#: catches up. A property of television, not of our buffer, which is the
#: whole point: the window the gate will accept a board change in must not
#: grow when we choose to wait longer.
#:
#: Measured on the first run against real footage (Argentina v France 2022):
#: the ball crossed the line at video 58 (StatsBomb 35:22) and the FIFA bug
#: went 1-0 to 2-0 at video 63.7, 5.7 s later. At the old 5.0 the window
#: [cursor - 2, cursor + 5] closed at 61.5 and missed it, and the caller's
#: correct goal call was rejected as unconfirmed.
GOAL_GRAPHIC_LAG_S = 10.0

#: The backstop on how long a goal stays a thing worth talking about, and
#: only the backstop: what really ends it is play restarting.
#:
#: This was 45 seconds of cursor time and that was wrong in kind. A
#: broadcaster spends 60 to 90 seconds after a goal on the celebration, the
#: replays, the scorer's face and the walk back, and every line about the
#: goal in that stretch is a line about something the state holds. On the
#: second real run the celebration lines at 128.1 and 140.1 were rejected as
#: phantom goals for being more than 45 s past a goal applied at 66.8 — while
#: the picture was still showing it, and the kickoff was still 13 s away.
#:
#: The cap exists because the restart can be missed: no kickoff line
#: written. Past it the picture has moved on whatever the caller says.
GOAL_TALK_CAP_S = 150.0

@dataclass
class ReplaySequence:
    """How many lines this run of replay pictures has had, and when.

    Two things the model cannot know and a rate cap cannot decide. Whether
    the replay has already been named as one — "as we see it again" belongs
    to the first line of a sequence and nowhere else
    (``docs/research/real-commentary-corpus.md`` section 3.2: "Watch this." /
    "Rakitic into Messi." / "Brilliant touch … and a fine finish" is one
    sequence at 37:24, and only the first of the three says it is a replay) —
    and how many angles of the same tackle the voice has already talked over.
    Three is the longest run in the corpus.

    Fed every replay *look*, spoken or not, because the sequence is a fact
    about the pictures rather than about this system: a broadcaster cutting
    back to the game and then to another replay has started a second one.
    """

    cfg: ReplayTalkConfig = field(default_factory=ReplayTalkConfig)
    #: When the last replay form arrived, which is what separates sequences.
    last_look: float | None = None
    #: When the last replay line was *spoken*, for the spacing.
    last_said: float | None = None
    #: How many have been spoken in this sequence.
    said: int = 0

    def look(self, ts: float) -> None:
        """A replay form arrived. Starts a new sequence if the gap is long."""
        if self.last_look is None or ts - self.last_look > self.cfg.sequence_gap_s:
            self.said = 0
            self.last_said = None
        self.last_look = ts

    @property
    def first(self) -> bool:
        """Is the next line the first of this sequence, and so allowed to name it?"""
        return self.said == 0

    def may_speak(self, ts: float) -> bool:
        """Is there room in this sequence for a line at ``ts``?"""
        if self.said >= self.cfg.max_lines:
            return False
        return self.last_said is None or ts - self.last_said >= self.cfg.min_gap_s

    def spoke(self, ts: float) -> None:
        """A replay line went out."""
        self.said += 1
        self.last_said = ts


#: How long a name stays on the ball. The caller reads a shirt, the player
#: turns, and the number is gone while the move it is part of is still going
#: on: Molina was named twice on his run and anonymous when he finished it.
#: Eight seconds is the length of a run, not of a passage of play — past it
#: the name has to be read again.
CARRY_NAME_S = 8.0


def _player_named(pack: KnowledgePack, name: str) -> tuple[Side, Player] | None:
    """Which player on either sheet this is, if it is one of them.

    One matcher, shared with the gate: a read the gate accepts and the
    runtime refuses to bind is a name in a line with nothing holding it up.
    """
    if not fold(name):
        return None
    for side in (Side.HOME, Side.AWAY):
        sheet = pack.team(side)
        if sheet is None:
            continue
        for player in sheet.squad:
            if is_the_same_name(name, player.name):
                return side, player
    return None


def _player_numbered(pack: KnowledgePack, side: Side, number: int) -> Player | None:
    sheet = pack.team(side)
    if sheet is None:
        return None
    for player in sheet.squad:
        if player.number == number:
            return player
    return None


@dataclass
class RuntimeStats:
    frames: int = 0
    board_reads: int = 0
    caller_calls: int = 0
    analyst_calls: int = 0
    ticks: int = 0
    gated_out: int = 0
    sightings: int = 0
    sightings_dropped: int = 0
    spoken: int = 0
    #: Extra phraser calls made in the quiet after a goal, and lines written
    #: by code off the state with no model in them at all.
    followups: int = 0
    restatements: int = 0
    cost_stopped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Runtime:
    """One match, from first frame to final whistle."""

    source: Any
    backend: LLMBackend
    pack: KnowledgePack | None = None
    settings: Settings = SETTINGS
    speaker: Speaker = field(default_factory=LogSpeaker)
    trace: RunTrace | None = None
    home: str = "Home"
    away: str = "Away"
    #: The second voice. Off by one flag, because "single voice" is one of the
    #: ablations the results table has to report.
    with_analyst: bool = True
    #: A statistician's feed. ``None`` in the default runtime and in every
    #: row of the results table but one: the thesis is the picture, the sound
    #: and notes, and this is here to say what a feed would have bought.
    wire: Wire | None = None

    bus: Bus = field(default_factory=Bus)
    stats: RuntimeStats = field(default_factory=RuntimeStats)

    def __post_init__(self) -> None:
        cap = self.settings.capture
        if self.pack is not None:
            self.home, self.away = self.pack.home.name, self.pack.away.name

        self.buffer = DelayBuffer(cap.fps, cap.delay_s, cap.history_s)

        self.board_reader = BoardReader(self.backend, config=self.settings.board)
        self.board_tracker = BoardTracker(self.settings.board)
        self.state_tracker = (
            MatchStateTracker.from_pack(self.pack)
            if self.pack is not None
            else MatchStateTracker(home=self.home, away=self.away)
        )
        self.caller = Caller(
            self.backend,
            config=self.settings.caller,
            pack=self.pack,
        )
        #: The speaking half of the play-by-play voice, or ``None`` when
        #: ``PHRASER_MODEL=off``. None is not a degraded mode: it is exactly
        #: the runtime that existed before the split, and a test asserts that
        #: the spoken lines are identical with it off.
        phraser = Phraser(
            self.backend,
            config=self.settings.phraser,
            silence=self.settings.silence,
            dead_ball=self.settings.dead_ball,
            home=self.home,
            away=self.away,
        )
        self.phraser: Phraser | None = phraser if phraser.enabled else None
        self.analyst = Analyst(
            self.backend,
            config=self.settings.analyst,
            tools=MatchTools(state=self.state_tracker.state, pack=self.pack),
            pack=self.pack,
        )
        #: The pack's notes with a memory and a running count behind them.
        #: Seeded at kickoff, and built ahead of the colour seat below so
        #: that seat can share its tallies: one match, one running count,
        #: whichever voice is reading a note off it.
        self.threads = Threads.from_pack(self.pack)
        #: What this broadcast has counted for itself: corners, fouls, shots,
        #: who has had how many. Built here, beside the threads and sharing
        #: their tallies so a player's goals are counted once, and fed from
        #: :meth:`_call` with every form and every state change. The colour
        #: seat below reads it rather than keeping its own.
        self.ledger = Ledger.from_pack(self.pack, tallies=self.threads.tallies)
        #: The second seat, event-driven and text-only. When it is enabled
        #: the old silence-timer analyst is not asked at all; see
        #: :meth:`_maybe_colour` and ``docs/HANDOFF.md`` section 3e.
        self.colour = ColourSeat(
            self.backend,
            config=self.settings.colour,
            pack=self.pack,
            tallies=self.threads.tallies,
            ledger=self.ledger,
        )
        self.gate = FactGate(self.settings.gate)
        self.predictor = SpeakPredictor(self.settings.predictor, self.settings.caller)
        self.director = Director(speaker=self.speaker, cfg=self.settings.director, bus=self.bus)

        self.cut = CutDetector(self.settings.predictor)

        self._pending: list[Trigger] = []
        self._board_changes: list[BoardChange] = []
        self._cuts: list[float] = []
        #: Cursor time at which the state last took in a board goal — not the
        #: time of the board change itself. See ``_apply_due_board_changes``.
        self._last_goal_ts: float | None = None
        #: Cursor time at which play was seen to restart after that goal, and
        #: the end of talking about it. ``None`` until it is seen.
        self._restart_ts: float | None = None
        self._last_spoken_video_ts: float | None = None
        #: How long that line takes to say. The rate cap is a debt the last
        #: line ran up, and a fragment runs up less of one than a sentence,
        #: so the predictor needs the length and not just the timestamp.
        self._last_spoken_seconds: float | None = None
        #: What the last line was about, which picks which of the three rates
        #: the cap uses. See :func:`commentary.predictor.phase_of`.
        self._last_spoken_event: Event | None = None
        #: Whether that last line was about a replay, which takes the
        #: dead-ball rate whatever the event was. See :meth:`_mark_spoken`.
        self._last_spoken_replay: bool = False
        #: The run of replay pictures the broadcast is on, and how much of it
        #: has been talked over.
        self.replays = ReplaySequence(cfg=self.settings.replay_talk)
        #: When the phraser last chose to say nothing. It holds the rate cap
        #: off so the next tick does not send the caller straight back out,
        #: and it deliberately does not touch the silence pressure: a chosen
        #: silence is still silence to whoever is listening.
        self._last_quiet_ts: float | None = None
        #: Whether the last line the caller got past the gate claimed a goal.
        #: The four seconds after one are the scorer's name, the celebration
        #: and the replay arriving together, and the rate cap spent them
        #: silent.
        self._said_a_goal = False
        #: Which side the player on the ball plays for, so a carried name
        #: cannot cross to the other team on the next line.
        self._carry_side = Side.UNKNOWN
        self._last_analyst_ts: float = 0.0
        #: The thirty seconds after a goal: which of the corpus's beats is
        #: due, and what the caller has said about the move. Driven from
        #: :meth:`_call` exactly as the offline rephrase drives it, so a beat
        #: that exists in one exists in the other.
        #: ``self.threads`` is built above, ahead of the colour seat; shared
        #: with the goal follow-up here so that beat 3 and a clause dropped
        #: into a lull are one selection.
        self.follow = GoalFollowup(threads=self.threads)
        #: The score-and-clock line, on the match clock. Off with
        #: ``RESTATEMENT_EVERY_S=0``, which is what a feed with a permanent
        #: score bug wants.
        self.restatements = Restatements(every_s=self.settings.restatement.every_s)
        #: The colour seat's turn in flight, so a second one is never started
        #: on top of it and the final whistle can cancel it.
        self._colour_turn: asyncio.Task[None] | None = None
        #: The follow-up filler in flight, for the same two reasons.
        self._goal_turn: asyncio.Task[None] | None = None
        #: Lead beats submitted, counted so a colour turn already in the air
        #: stops the moment the caller has something.
        self._lead_beats = 0
        self._recent_event: tuple[Event, float] | None = None
        self._sync = WireSync(self.wire) if self.wire is not None else None
        self._stop = asyncio.Event()

    # -- what the web layer is allowed to see ----------------------------

    @property
    def state(self) -> MatchState:
        return self.state_tracker.state

    @property
    def usage(self) -> Usage:
        return self.backend.total

    @property
    def present_offset_s(self) -> float:
        """How far behind the cursor the viewer's picture is held."""
        return self.settings.capture.present_offset_s

    def status(self) -> dict[str, Any]:
        stats = self.gate.stats
        return {
            **self.stats.as_dict(),
            **self.director.stats.as_dict(),
            "buffered_frames": len(self.buffer),
            "cursor_ts": self.buffer.cursor_ts,
            "live_ts": self.buffer.live_ts,
            # The ratio governor's own number: what share of the last five
            # minutes of utterances was the second voice, against club
            # football's 31%. ``None`` means too few lines to say.
            "colour_share": self.colour.share.share(self.cursor_ts),
            "colour_share_target": self.settings.colour.colour_share_target,
            "gate": {
                "judged": stats.judged,
                "passed": stats.passed,
                "trimmed": stats.trimmed,
                "rejected": stats.rejected,
                "by_reason": dict(stats.by_reason),
            },
        }

    # -- the loops -------------------------------------------------------

    async def run(self, seconds: float | None = None) -> RuntimeStats:
        async with self.source:
            # The recorder subscribes before anything can publish, so the
            # trace starts at the first frame rather than the first race.
            recorder = asyncio.create_task(self._record(), name="trace")
            await asyncio.sleep(0)
            tasks = [
                recorder,
                asyncio.create_task(self._ingest_frames(), name="frames"),
                asyncio.create_task(self._read_board(), name="board"),
                asyncio.create_task(self._tick(), name="tick"),
                asyncio.create_task(self.director.run(), name="director"),
            ]
            if seconds is not None:
                tasks.append(asyncio.create_task(self._deadline(seconds), name="deadline"))

            try:
                await self._stop.wait()
            finally:
                # Let whatever is mid-sentence finish, then close the books
                # while the recorder is still listening, so the final tallies
                # land in the trace rather than after it.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.director.drain(timeout=2.0), timeout=2.5)
                self._publish(Topic.STATUS, self.live_ts, **self.status())
                self._publish(
                    Topic.COST, self.live_ts, total_usd=round(self.backend.total.cost_usd, 4)
                )
                await asyncio.sleep(0)
                self.director.stop()
                # A colour turn is a run of utterances with sleeps between
                # them, so at the whistle there may be one still waiting to
                # say its third thing. It goes with everything else.
                if self._colour_turn is not None:
                    tasks.append(self._colour_turn)
                if self._goal_turn is not None:
                    tasks.append(self._goal_turn)
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
        return self.stats

    async def _deadline(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
        self.stop()

    def stop(self) -> None:
        self._stop.set()

    @property
    def live_ts(self) -> float:
        return self.buffer.live_ts or 0.0

    @property
    def cursor_ts(self) -> float:
        return self.buffer.cursor_ts or 0.0

    async def _ingest_frames(self) -> None:
        async for frame in self.source.frames():
            self.buffer.append(frame)
            self.stats.frames += 1
            self._apply_due_board_changes()
            self._apply_due_wire()
            if (cut := self.cut.feed(frame)) is not None:
                # Kept with its timestamp: a cut is both a reason to consider
                # speaking and the edge of what counts as "next" for the
                # caller, and only the second of those needs to know when.
                self._cuts.append(cut.ts)
                self._fire(Trigger.CAMERA_CUT)
        # The source ran out: a clip ended, or the stream died. Either way the
        # match is over as far as this process is concerned.
        self.stop()

    async def _read_board(self) -> None:
        """Glance at the score bug, at the live edge, on a fixed interval."""
        while True:
            await asyncio.sleep(self.settings.board.interval_s)
            frame = self._live_frame()
            if frame is None:
                continue
            try:
                read = await self.board_reader.read(frame)
            except Exception as exc:  # a missed glance is not a dead match
                self._publish(Topic.ERROR, frame.ts, where="board", detail=str(exc))
                continue
            self.stats.board_reads += 1
            self._publish(Topic.BOARD, frame.ts, read)
            self._take_board_read(read, frame.ts)

    def _take_board_read(self, read: BoardRead, ts: float) -> None:
        """One glance at the bug, and what the rest of the system makes of it.

        A confirmed change fires the trigger, as it always did. So does the
        *first* read that disagrees with the settled score in a goal's
        direction. The board reader runs a buffer's length ahead of the
        cursor and a broadcaster's graphic runs six seconds behind the ball,
        so that first read lands at the cursor about when the ball is
        crossing the line, with the finish inside the caller's lookahead.
        Waiting for the third agreeing read fired it seven seconds after the
        kick instead — on the Mbappé penalty, that was a line about the
        run-up at 76.6, a lull handed to the analyst at 80.8, and the goal
        called at 87.9. Only the trigger moves early; the score still waits
        for three reads, and the gate already treated one agreeing read as
        corroboration.
        """
        self._observe_clock(read, ts)
        change = self.board_tracker.update(read, ts)
        if change is not None:
            self._board_changes.append(change)
            self._fire(Trigger.BOARD_CHANGE)
        elif self._first_sight_of_a_goal(ts):
            self._fire(Trigger.BOARD_CHANGE)
        self._note_screen()

    def _first_sight_of_a_goal(self, ts: float) -> bool:
        """Did the read just taken open a goal-shaped disagreement with the board?

        Once per pending change: the second and third agreeing reads add
        evidence, not news, and the confirmation fires on its own.
        """
        pending = self.board_tracker.pending_goal
        return pending is not None and pending.count == 1 and pending.first_ts == ts

    def _observe_clock(self, read: BoardRead, ts: float) -> None:
        """Tell the sync where the match clock and the video clock meet.

        The feed counts in match time and everything here counts in video
        time, and the board reader's own glance at the graphic is the only
        bridge between them that exists at runtime. This is the whole of the
        board's involvement: when an event is *released* is a question about
        the live edge, and the frame loop answers that one.
        """
        if self._sync is None or not read.bug_visible or read.clock is None:
            return
        clock_s = parse_clock(read.clock)
        period = period_for_clock(read.clock)
        if clock_s is None or period is None:
            return
        self._sync.observe_clock(clock_s, period, ts)

    def _live_frame(self) -> Frame | None:
        live = self.buffer.live_ts
        return self.buffer.nearest(live) if live is not None else None

    def _note_screen(self) -> None:
        """Push a change in what the board looks like, replay or gone entirely."""
        tracker = self.board_tracker
        seen = (tracker.in_replay, not tracker.bug_missing)
        if seen != (self.state.in_replay, self.state.bug_visible):
            self.state_tracker.apply_board(tracker)
            self._publish(Topic.STATE, self.cursor_ts, self.state)

    def _apply_due_board_changes(self) -> None:
        """Let the state catch up to the cursor, never to the live edge.

        This is what keeps the system from announcing a goal the viewer has
        not been shown. The evidence arrives early; the belief arrives on time.
        """
        cursor = self.buffer.cursor_ts
        if cursor is None:
            return
        due = [c for c in self._board_changes if c.ts <= cursor]
        if not due:
            return
        self._board_changes = [c for c in self._board_changes if c.ts > cursor]
        # Stamped with the cursor, not with the change's own ts. "The score
        # moved recently" is a fact about the viewer's scoreboard, and the
        # viewer's scoreboard moves when the state takes the change in. The
        # two are the same moment when confirmation is prompt and a long way
        # apart when it is not: a change first seen at 62.7 and confirmed
        # only when the bug came back after the replay at 136 is news at 136,
        # and the lines about the goal come after that, not after 62.7.
        if any(c.is_goal for c in due):
            self._last_goal_ts = cursor
            self._restart_ts = None
        for change in due:
            if change.is_goal and change.scoring_side is not None:
                self.state.incidents.append(
                    Incident(
                        event=Event.GOAL,
                        side=change.scoring_side,
                        player=None,
                        video_ts=change.ts,
                        source="board",
                    )
                )
        self.state_tracker.apply_board(self.board_tracker)
        self._publish(Topic.STATE, cursor, self.state)

    def _apply_due_wire(self) -> None:
        """Let the statistician catch up to the cursor, never to the live edge.

        Same rule as the board, and for the same reason: the feed knows
        things before the viewer has seen them, and commentary that used them
        would be describing a match nobody is watching yet.
        """
        cursor = self.buffer.cursor_ts
        if self._sync is None or cursor is None:
            return
        # Polled here rather than off the back of a board read. What the feed
        # has said is a fact about the live edge, which this loop knows
        # exactly and the board reader only samples every couple of seconds;
        # tying release to that sampling made a goal's arrival depend on when
        # the score bug was last glanced at, which is nothing to do with it.
        self._sync.poll(self.live_ts)
        for correction in self._sync.apply_due(cursor, self.state_tracker):
            self._publish(
                Topic.CORRECTION,
                correction.video_ts,
                what=correction.what,
                event=correction.event.event.value,
            )
            self._publish(Topic.STATE, cursor, self.state)

    # -- deciding to speak -----------------------------------------------

    async def _tick(self) -> None:
        interval = self.settings.predictor.tick_s
        while True:
            await asyncio.sleep(interval)
            self.stats.ticks += 1
            if not self.buffer.ready:
                continue
            triggers = self._drain_triggers()
            if not triggers:
                triggers = [Trigger.SCHEDULED]
            decision = self.predictor.decide(
                now_ts=self.cursor_ts,
                triggers=triggers,
                last_spoken_ts=self._last_spoken_video_ts,
                after_goal=self._said_a_goal,
                last_spoken_seconds=self._last_spoken_seconds,
                last_event=self._last_spoken_event,
                last_was_replay=self._last_spoken_replay,
                last_quiet_ts=self._last_quiet_ts,
                # The lead's build-up cap opens while the second voice is
                # short of its share of the channel. See
                # ``SpeakPredictor.cap_for`` and ``colour.Share``.
                colour_stretch=self.colour.share.stretch(self.cursor_ts),
            )
            self._publish(Topic.TRIGGER, self.cursor_ts, decision)
            if self._over_budget():
                continue

            # A lull belongs to the analyst, and it has to be offered one
            # first. The predictor's job is to never let the broadcast go
            # mute, so left alone it will always send the caller to fill a
            # silence — and the analyst, which by design only speaks into
            # silences, would never once get a turn.
            if self.settings.colour.enabled:
                if await self._maybe_colour():
                    continue
            elif self._is_a_lull(decision) and await self._maybe_analyst():
                continue
            # And a period of the clock may be owed to whoever has just
            # joined. It is the flattest thing anybody says in a match and it
            # costs nothing, so it is offered a moment that is already clear
            # rather than one the caller wants.
            if self._maybe_restate():
                continue
            if decision.should_call:
                await self._call(decision.triggers)

    def _is_a_lull(self, decision: SpeakDecision) -> bool:
        """Nothing has happened; the only reason to speak is that nobody has."""
        real = set(decision.triggers) - {Trigger.SCHEDULED, Trigger.SILENCE_PRESSURE}
        return not real

    async def _maybe_colour(self) -> bool:
        """Offer the colour seat a turn. Returns whether it took one.

        The phase gate is deterministic and free, so this can be asked every
        tick and most ticks end on the first line. When it does say yes the
        turn is a run of two to four utterances, and they are spoken from a
        background task rather than submitted together: the director's queue
        is three deep and a turn that filled it would push the lead out.

        Everything about the timing is in
        :func:`commentary.agents.colour.may_speak`, with the corpus numbers
        beside it. Nothing about it is here.
        """
        if not self.colour.enabled or self._colour_turn is not None:
            return False
        cursor = self.cursor_ts
        offer = self.colour.offer(cursor)
        if not offer.allowed:
            return False

        turn = await self.colour.turn(offer, self.state_tracker.summary(cursor))
        self.colour.answered(cursor)
        if turn is None:
            self._publish(Topic.ERROR, cursor, where="colour", detail=self.colour.last_reason)
            return False
        self._publish(
            Topic.COLOUR,
            cursor,
            share=self.colour.share.share(cursor),
            share_target=self.settings.colour.colour_share_target,
            situation=offer.situation,
            reason=offer.reason,
            speak=turn.speak,
            angle=turn.angle.value,
            cites=turn.cites,
            utterances=turn.utterances,
        )
        if not turn.speak or not turn.utterances:
            return False

        spacing = space_out(
            cursor,
            len(turn.utterances),
            gap=self.settings.colour.utterance_gap_s,
        )
        self._colour_turn = asyncio.create_task(
            self._say_colour(turn.utterances, spacing, offer.situation)
        )
        self.stats.analyst_calls += 1
        return True

    async def _say_colour(
        self, utterances: list[str], spacing: list[float], situation: str
    ) -> None:
        """Put one turn on the channel, an utterance at a time, and stop early.

        The corpus's colour voice "hands back by stopping mid-thought when
        the ball moves" (section 4.6) — there is no verbal hand-back anywhere
        in it — so the moment the lead submits a beat, the rest of the turn
        is dropped rather than queued behind it.

        The three arguments after ``said_before`` are the attribution check,
        and they are what makes it work at all. ``attributed`` is who the
        caller's own forms put on each event; ``named_before`` is the one man
        this turn has already named, so that a bare "he" is resolved to
        somebody before it is checked; ``after`` is the utterance of this turn
        that actually reached air, which is what lets a continuation say "he"
        without being struck out as filler. They are computed exactly as
        :func:`commentary.agents.colour.colour_pass` computes them — off what
        passed the gate rather than off what the model wrote, because an
        utterance nobody heard is no antecedent for a pronoun — and without
        them ``misattributes`` returns early on every line and the whole check
        is inert on the live path while the offline one has it.
        """
        started = self._lead_beats
        attributed = self.colour.attributed(self.cursor_ts)
        #: What has actually gone out of this turn, and the one man it named.
        spoken = ""
        referent = ""
        try:
            for index, (text, at) in enumerate(zip(utterances, spacing, strict=False)):
                if index:
                    await asyncio.sleep(max(0.0, at - spacing[index - 1]))
                    if self._lead_beats != started:
                        break
                verdict = judge_utterance(
                    text,
                    self.state,
                    self.pack,
                    self.gate,
                    goal_in_state=self._score_counts_the_goal(self.cursor_ts),
                    at=self.cursor_ts,
                    # Its own last ten, updated as each one passes, so a
                    # phrase the seat has settled into is struck out
                    # whether it read it in the prompt or wrote it itself.
                    said_before=self.colour.history,
                    attributed=attributed,
                    named_before=[referent] if referent else (),
                    after=spoken,
                )
                self._publish(
                    Topic.GATE, self.cursor_ts, verdict, event=Event.NONE.value, where="colour"
                )
                if not verdict.passed:
                    continue
                # Counted against the same threads the lead's lines are:
                # a note the colour seat has just said is a note that has
                # been said, whichever voice said it.
                self._publish_threads(
                    self.cursor_ts,
                    self.threads.said(verdict.line, ts=self.cursor_ts, pack=self.pack),
                    "used",
                )
                self.director.submit(
                    Beat(
                        id=next_beat_id("c"),
                        voice=Voice.ANALYST,
                        text=verdict.line,
                        video_ts=self.cursor_ts,
                        created_ts=time.monotonic(),
                        live_ts=self.live_ts,
                        excitement=EXCITEMENT.get(situation, 0.2),
                        preemptable=True,
                    )
                )
                spoken = verdict.line
                # A continuation that names nobody keeps the man it inherited,
                # and one that names two people hands on nobody.
                named_here = any(
                    mentions(verdict.line, name) for name in roster_names(self.pack)
                )
                referent = one_subject(verdict.line, self.pack) or (
                    "" if named_here else referent
                )
                self.colour.accept([verdict.line])
                self.colour.spoke_colour(self.cursor_ts)
                self._mark_spoken(self.cursor_ts, verdict.line)
        finally:
            self._colour_turn = None

    async def _fill_the_goal_window(self) -> None:
        """Talk into the silence after a goal, up to twice, and stop early.

        The half-minute after a goal is real commentary's fastest sustained
        talking: a median seven utterances and sixty words, with no internal
        gap longer than 7.1 s (``docs/research/real-commentary-corpus.md``
        section 2.4). This system said two lines of seven words and went
        quiet, because the caller is only asked for a line when the picture
        gives it one and after a goal the picture is replays, faces and a
        bench.

        So the beats are asked for rather than waited for. Four seconds after
        the last line, if the caller has not come back with one of its own,
        one extra phraser call goes out; then, at most, one more. The moment
        the lead speaks the rest is dropped — the same rule the colour seat
        follows, and for the same reason: two voices on one moment is worse
        than one.
        """
        started = self._lead_beats
        try:
            for _ in range(MAX_SYNTH):
                await asyncio.sleep(SYNTH_GAP_S)
                if self._lead_beats != started:
                    return
                cursor = self.cursor_ts
                if not self.follow.due(cursor) or self._over_budget():
                    return
                if not await self._say_followup(cursor):
                    return
                started = self._lead_beats
        finally:
            self._goal_turn = None

    async def _say_followup(self, cursor: float) -> bool:
        """One phraser call the caller never asked for. Returns whether it spoke.

        Nothing in it is new perception. The form carries the goal line's own
        sightings — already judged once — and the caller's own words about the
        move, and the block tells the model which beat is due. It goes through
        the same gate and the same strip as any other line, and a call that
        comes back empty or refused simply does not happen: there is no
        caller line here to fall back to.
        """
        if self.phraser is None:
            return False
        form = self.follow.synthetic()
        self.stats.followups += 1
        self.threads.see_state(self.state)
        offered = (
            self.threads.offer([self.follow.scorer], ts=cursor, payoff=True)
            if self.follow.scorer
            else []
        )
        self._publish_threads(cursor, offered, "offered")
        phrased = await self.phraser.phrase(
            form,
            self.state_tracker.summary(cursor),
            notes=[item.note for item in offered],
            callbacks=[item.callback for item in offered],
            followup=self.follow.block(cursor, self.pack),
            goal_beat=self.follow.beat(cursor),
            scorer=self.follow.scorer,
            roster=roster_names(self.pack),
            said_of_the_goal=self.follow.spoken,
        )
        if phrased is None or not phrased.line.strip():
            # A beat asked for and not written. The reason is worth a row —
            # a dropped repeat and a failed call are different things, and
            # this is the only record either leaves.
            self._publish(
                Topic.PHRASED,
                cursor,
                original=form.line,
                line="",
                excitement=0.0,
                reason=self.phraser.last_reason or "the phraser wrote no follow-up",
                synthetic=True,
            )
            return False
        settled = settle_numbers(
            phrased.line,
            state=self.state,
            side=form.side,
            goal_in_state=self._score_counts_the_goal(cursor),
            # The score went out on the call. This is a line after it.
            append=False,
        )
        if not settled.line.strip():
            return False
        self._publish(
            Topic.PHRASED,
            cursor,
            original=form.line,
            line=settled.line,
            excitement=phrased.excitement,
            opener_retry=phrased.opener_retry,
            closer_retry=phrased.closer_retry,
            name_retry=phrased.name_retry,
            shout_retry=phrased.shout_retry,
            shout_rewritten=phrased.shout_rewritten,
            score_stripped=list(settled.stripped),
            synthetic=True,
        )
        verdict = self.gate.judge(
            form.model_copy(update={"line": settled.line}),
            self.state,
            self.pack,
            # The goal this line is about has already passed the gate once,
            # which is the whole of what a board change is evidence for.
            board_changed=True,
            wire_confirmed=self._wire_confirms_goal(cursor),
            goal_in_state=self._score_counts_the_goal(cursor),
            at=cursor,
            notes=self.threads.notes(),
            described=form.line,
        )
        self._publish(Topic.GATE, cursor, verdict, event=Event.GOAL.value, where="followup")
        if not verdict.passed:
            self.stats.gated_out += 1
            return False
        self.director.submit(
            Beat(
                id=next_beat_id("g"),
                voice=Voice.CALLER,
                text=verdict.line,
                video_ts=cursor,
                created_ts=time.monotonic(),
                live_ts=self.live_ts,
                event=Event.GOAL,
                excitement=phrased.excitement,
                preemptable=False,
            )
        )
        self._lead_beats += 1
        said = self.threads.said(verdict.line, ts=cursor, pack=self.pack)
        self._publish_threads(cursor, said, "used")
        self.follow.said(cursor, verdict.line)
        self.follow.synthesised += 1
        self.colour.saw_lead_line(cursor, verdict.line)
        self.phraser.accept(verdict.line, Event.GOAL, ts=cursor)
        self._mark_spoken(cursor, verdict.line, Event.GOAL)
        self.stats.spoken += 1
        return True

    def _maybe_restate(self) -> bool:
        """Say the score and the clock, in code, on a timer. Returns whether it did.

        Gap 8 item 4 and section 5.3 of the corpus study: a club-channel feed
        restates both every few minutes for whoever has just joined, always
        clock then score, out of a vocabulary of about ten phrasings. No
        model is needed to say "ten minutes gone, two-nil to Barcelona", and
        by the standing rule — code writes numbers, the model writes words —
        no model should be asked to.

        It is filler and it never speaks over the game: a period that comes
        due while the lead is talking waits, and goes out at the first moment
        that is clear. Nothing here is a claim the gate could check, because
        every word of it is the state's own.
        """
        self.restatements.note(self.state)
        if not self.restatements.pending:
            return False
        cursor = self.cursor_ts
        clear_of = self.settings.restatement.clear_of_a_beat_s
        last = self._last_spoken_video_ts
        if last is not None and cursor - last < clear_of:
            return False
        if self.follow.blocks_restatement(cursor):
            return False
        if self._colour_turn is not None or self._goal_turn is not None:
            return False
        text = self.restatements.take(self.state)
        if not text:
            return False
        self.director.submit(
            Beat(
                id=next_beat_id("r"),
                voice=Voice.CALLER,
                text=text,
                video_ts=cursor,
                created_ts=time.monotonic(),
                live_ts=self.live_ts,
                event=Event.NONE,
                excitement=self.settings.restatement.excitement,
                preemptable=True,
            )
        )
        self._lead_beats += 1
        self._mark_spoken(cursor, text)
        self.stats.restatements += 1
        self.stats.spoken += 1
        return True

    async def _maybe_analyst(self) -> bool:
        """Offer the analyst a turn. Returns whether it took one."""
        if not self.with_analyst:
            return False
        cursor = self.cursor_ts
        silence = (
            float("inf")
            if self._last_spoken_video_ts is None
            else cursor - self._last_spoken_video_ts
        )
        allowed, reason = self.analyst.should_speak(
            silence_s=silence,
            last_analyst_ts=self._last_analyst_ts,
            now_ts=cursor,
            last_event=self._recent_event_within(6.0),
        )
        if not allowed:
            return False

        line = await self.analyst.call(self.buffer, self.state_tracker.summary(cursor), reason)
        if line is None:
            self._publish(Topic.ERROR, cursor, where="analyst", detail=self.analyst.last_reason)
            return False
        self._publish(Topic.ANALYST, cursor, line)
        if not line.speak or not line.line.strip():
            return False

        self.director.submit(
            Beat(
                id=next_beat_id("a"),
                voice=Voice.ANALYST,
                text=line.line,
                video_ts=cursor,
                created_ts=time.monotonic(),
                live_ts=self.live_ts,
                preemptable=True,
            )
        )
        self._last_analyst_ts = cursor
        self._mark_spoken(cursor, line.line)
        self.stats.analyst_calls += 1
        return True

    def _recent_event_within(self, seconds: float) -> Event | None:
        """Whatever just happened, while it is still 'just'."""
        if self._recent_event is None:
            return None
        event, ts = self._recent_event
        return event if self.cursor_ts - ts <= seconds else None

    def _over_budget(self) -> bool:
        spent = self.backend.total.cost_usd
        if spent < self.settings.cost.max_usd_per_match:
            return False
        if not self.stats.cost_stopped:
            self.stats.cost_stopped = True
            self._publish(Topic.STATUS, self.cursor_ts, reason="cost_cap", spent_usd=spent)
        return True

    async def _call(self, triggers: list[Trigger]) -> None:
        cursor = self.cursor_ts
        self.stats.caller_calls += 1
        line = await self.caller.call(
            self.buffer,
            self.state_tracker.summary(cursor),
            triggers,
            lookahead_until=self._next_cut_after(cursor),
        )
        if line is None:
            self._publish(Topic.ERROR, cursor, where="caller", detail=self.caller.last_reason)
            return

        self._publish(Topic.CALLER, cursor, line)
        # The colour seat reads the phase off the forms, spoken or not: a
        # form the caller filled in and chose not to say is still the best
        # evidence there is about what the picture was.
        self.ledger.saw_form(cursor, line)
        self.colour.saw_form(cursor, line)
        # And so does the follow-up, for the same reason and one more: after
        # a goal the caller fills in form after form over the replays and
        # says none of them, and those forms are the only account anywhere of
        # how the goal was scored. Beat 4 rebuilds the move out of them.
        self.follow.saw_form(line)
        # A replay moves nothing. ``apply_caller`` and ``Ledger.saw_form``
        # each drop a replay form of their own accord — the second goal of a
        # match must not become the third because the broadcast showed the
        # first one again — and the restart is the third of the same kind: the
        # caller wrote "kickoff, live play" over a replay of a goal on the
        # second real run and ended goal talk sixty seconds early. The
        # sightings are the exception and stay: a shirt number legible in a
        # replay is a number that was legible, and the registry is a map from
        # numbers to names rather than a record of what has happened.
        replay = line.scene is Scene.REPLAY
        if replay:
            # Every look, spoken or not: the sequence is a fact about what
            # the broadcast is showing, not about what this system said.
            self.replays.look(cursor)
        self.state_tracker.apply_caller(line, cursor)
        if not replay:
            self._note_restart(line, cursor)
        self._bind_sightings(line, cursor)
        if not line.speak or not line.line.strip():
            return
        if replay and not self.replays.may_speak(cursor):
            # Three angles of the same tackle is where a commentator stops,
            # and two lines four seconds apart is as fast as the corpus's own
            # replay runs go. Refused here rather than after the phraser,
            # because the model call is the expensive half.
            self._publish(
                Topic.STATUS,
                cursor,
                reason="replay_spent",
                said=self.replays.said,
            )
            return

        # Seeing is done; speaking is a separate call. What the gate judges
        # is whatever is actually going to the speaker, so the phrased line
        # is checked against the roster and the scoreline exactly as the
        # caller's would have been. Nothing the phraser writes gets past a
        # check the caller's line had to pass.
        said = await self._phrase(line, cursor)
        if said is None:
            # The phraser chose silence. No beat, no gate row, and the
            # `phrased` row it published is the record that it was a choice.
            return
        judged, excitement = said

        verdict = self.gate.judge(
            judged,
            self.state,
            self.pack,
            board_changed=self._board_supports_goal(cursor),
            wire_confirmed=self._wire_confirms_goal(cursor),
            # The narrow half of the three, passed on its own: a goal the
            # state already holds is a goal the score already counts, and the
            # gate's arithmetic needs to know that the number is settled
            # rather than arriving.
            goal_in_state=self._score_counts_the_goal(cursor),
            carried=self._carried_name(line, cursor),
            at=cursor,
            notes=self.threads.notes(),
            # Every count about anybody this line names, not only the two the
            # phraser was shown: the check is whether the number is one the
            # match holds, and the match holds all of them.
            ledger=self._counts_for(judged, cursor),
            # The caller's own account, for the one rule that asks whose the
            # thing was rather than whether the name is real.
            described=line.line,
        )
        self._publish(Topic.GATE, cursor, verdict, event=line.event.value)
        if not verdict.passed:
            self.stats.gated_out += 1
            return

        self.caller.gate.accept(verdict.line)
        # What the line says outranks what the caller filed it under. Ronaldo's
        # free kick was called correctly — "curls it over the wall and into the
        # top corner" — tagged `free_kick`, and dropped by the director on the
        # camera cut that every broadcaster makes the instant a goal goes in.
        # A goal is a goal whatever put the ball there.
        event = Event.GOAL if claims_goal(verdict.line, line.event) else line.event
        # A replay is not an event happening. It does not open the four
        # seconds after a goal that the rate cap is told never to hold, and
        # it never holds the channel against live football: the whole of what
        # makes a replay line safe to say is that the moment the game is back
        # on the screen, the line about the last one can be dropped.
        self._said_a_goal = event is Event.GOAL and not replay
        beat = Beat(
            id=next_beat_id(),
            voice=Voice.CALLER,
            text=verdict.line,
            video_ts=cursor,
            created_ts=time.monotonic(),
            live_ts=self.live_ts,
            event=event,
            excitement=excitement,
            triggers=triggers,
            preemptable=replay or event not in (Event.GOAL, Event.PENALTY),
        )
        self.director.submit(beat)
        if replay:
            self.replays.spoke(cursor)
        self._lead_beats += 1
        self.colour.saw_lead_line(cursor, verdict.line)
        if self.phraser is not None:
            self.phraser.accept(
                verdict.line,
                line.event,
                ts=cursor,
                nameless=nameless_build_up(line, on_the_ball=self._carried_name(line, cursor)),
            )
        if not replay:
            # Who was on the ball in a replay is who was on the ball a minute
            # ago. Carrying that name into the next live line would name the
            # wrong man for the phase that is actually on the screen.
            self._remember_on_the_ball(line, verdict.line, cursor)
        self._mark_spoken(cursor, verdict.line, line.event, replay=replay)
        self._publish_threads(
            cursor, self.threads.said(verdict.line, ts=cursor, pack=self.pack), "used"
        )
        self._publish_ledger(cursor, self._counts_said(verdict.line, cursor), "used")
        # -- the thirty seconds after a goal -----------------------------
        # A goal line that got through opens the window; any line inside it
        # spends one of its beats. Then, if the caller leaves the kind of
        # silence it left after the Mbappé penalty — 24 seconds, because
        # every picture in between was a replay — the gap is filled.
        if event is Event.GOAL and self.follow.is_the_call(cursor) and not replay:
            self.follow.arm(cursor, line, verdict.line, self.pack)
            # Whose goal it was is the one thing the board never knows, and
            # every running count about him is wrong from this second on.
            self.threads.credit_goal(self.follow.scorer, cursor)
            self.ledger.credit_goal(self.follow.scorer, cursor, line.side)
        elif self.follow.active(cursor):
            # A replay line *is* beat 4, whatever beat the counter is on:
            # both are the past-tense account of how the goal was scored, and
            # the replay has the pictures behind it. Every other line inside
            # the window spends the next beat in order.
            if replay:
                self.follow.rebuilt(cursor, verdict.line)
            else:
                self.follow.said(cursor, verdict.line)
        if self.follow.active(cursor) and self._goal_turn is None:
            self._goal_turn = asyncio.create_task(self._fill_the_goal_window())
        # -- end of the goal window --------------------------------------
        if line.event is not Event.NONE and not replay:
            # "Whatever just happened" is a question about the live picture.
            # A replay of the foul is not a second foul, and the ten seconds
            # in which a penalty's kick and its goal sit apart do not restart
            # because the broadcast showed the kick again.
            self._recent_event = (line.event, cursor)
        self.stats.spoken += 1
        self._publish(Topic.COST, cursor, total_usd=round(self.backend.total.cost_usd, 4))

    async def _phrase(self, line: CallerLine, cursor: float) -> tuple[CallerLine, float] | None:
        """Say the caller's form the way a commentator would, or keep its words.

        Returns the form the gate should judge and the excitement to hang on
        the beat. With the stage off, or when it fails, that is the caller's
        own line untouched: a line the caller wrote and the gate has yet to
        see is worth more spoken badly than not spoken at all, so a phraser
        that errors is an error row on the bus and never a silent drop.

        ``None`` is the one case that is a silent drop, and it is the
        phraser *choosing* one: a quarter of build-up touches and 43% of
        goal kicks pass with nothing said in real commentary
        (``docs/research/real-commentary-corpus.md`` section 3), and the
        phraser's prompt now names the moments to pass over. A chosen
        silence publishes a `phrased` row with an empty line and a reason,
        so the trace can be counted, and no beat and no gate row, because
        nothing was said.
        """
        if self.phraser is None:
            return line, 0.0
        carried = self._carried_name(line, cursor)
        passed_over = self.phraser.passes_over(line, ts=cursor, on_the_ball=carried)
        if passed_over is not None:
            # No model call at all. The same row a chosen silence publishes,
            # with the reason code decided it rather than the model's.
            self._publish(
                Topic.PHRASED,
                cursor,
                original=line.line,
                line="",
                excitement=0.0,
                reason=passed_over,
            )
            self._last_quiet_ts = cursor
            return None
        followup = self.follow.block(cursor, self.pack)
        names = self._named_by(line, carried)
        if followup and self.follow.scorer:
            # Beat 3 is a number about the scorer, and the notes are capped,
            # so the scorer's clauses go to the front and the cap falls off
            # the far end.
            names = [self.follow.scorer] + names
        # The man on the ball, then the teams, the way the tracker ordered
        # them — and then the threads decide which of the clauses about those
        # people is worth saying now. ``docs/research/real-commentary-corpus.md``
        # section 7: a callback beats a fresh fact.
        ball = self.state.ball
        wanted = ([ball.player] if ball is not None else []) + names
        wanted += [self.state.home, self.state.away]
        self.threads.see_state(self.state)
        self.ledger.see_state(self.state)
        offered = self.threads.offer(wanted, ts=cursor, payoff=bool(followup))
        self._publish_threads(cursor, offered, "offered")
        # The same people, the same order, asked of the other source of
        # numbers. Two clauses, after the notes: the block is an offer and a
        # menu of six is not one.
        counts = self.ledger.facts(cursor, wanted, limit=CONTEXT_FACTS)
        self._publish_ledger(cursor, counts, "offered")
        phrased = await self.phraser.phrase(
            line,
            self.state_tracker.summary(cursor),
            on_the_ball=carried,
            notes=[item.note for item in offered],
            callbacks=[item.callback for item in offered],
            ledger=counts,
            followup=followup,
            goal_beat=self.follow.beat(cursor),
            scorer=self.follow.scorer,
            roster=roster_names(self.pack),
            said_of_the_goal=self.follow.spoken,
            replay_first=self.replays.first,
        )
        if phrased is not None and not phrased.line.strip() and self.phraser.chose_silence:
            self._publish(
                Topic.PHRASED,
                cursor,
                original=line.line,
                line="",
                excitement=0.0,
                reason=self.phraser.last_reason or "the phraser chose silence",
                # A silence the retry chose is the retry working: it is
                # offered "open differently or say nothing" and took the
                # second.
                opener_retry=phrased.opener_retry,
                closer_retry=phrased.closer_retry,
                name_retry=phrased.name_retry,
            )
            self._last_quiet_ts = cursor
            return None
        if phrased is None or not phrased.line.strip():
            self._publish(
                Topic.ERROR,
                cursor,
                where="phraser",
                detail=self.phraser.last_reason or "the phraser returned nothing",
            )
            return line, 0.0
        # Numbers by code. Whatever score the model wrote comes out, and the
        # one the state supports goes on — once, on the line that calls the
        # goal, and never on the celebration after it. The same call the
        # offline rephrase makes, so the two cannot drift.
        # ``docs/HANDOFF.md`` section 3d.
        #
        # The how gets the same treatment beside a penalty: this form's own
        # event, or the last spoken form's within ten seconds — the gap a
        # penalty's kick and its goal sit apart — is what "82.5 Mbappé! Over
        # the wall!" was missing a check on.
        penalty = line.event is Event.PENALTY or self._recent_event_within(10.0) is Event.PENALTY
        settled = settle_numbers(
            phrased.line,
            state=self.state,
            side=line.side,
            goal_in_state=self._score_counts_the_goal(cursor),
            append=self.follow.is_the_call(cursor) and claims_goal(phrased.line, line.event),
            description=line.line,
            penalty=penalty,
        )
        if settled.how_removed:
            self._publish(
                Topic.STATUS,
                cursor,
                reason="how_not_in_form",
                removed=list(settled.how_removed),
            )
        self._publish(
            Topic.PHRASED,
            cursor,
            original=line.line,
            line=settled.line,
            excitement=phrased.excitement,
            opener_retry=phrased.opener_retry,
            closer_retry=phrased.closer_retry,
            name_retry=phrased.name_retry,
            shout_retry=phrased.shout_retry,
            shout_rewritten=phrased.shout_rewritten,
            replay_marker_stripped=phrased.replay_marker_stripped,
            score_appended=settled.appended,
            score_stripped=list(settled.stripped),
            how_stripped=list(settled.how_removed),
        )
        return line.model_copy(update={"line": settled.line}), phrased.excitement

    def _publish_threads(self, cursor: float, offered: list[Offered], action: str) -> None:
        """Put a callback on the bus, offered or spoken, so it can be counted.

        Only callbacks are worth a row on the offer side — an unused note is
        the ordinary case and there are three of them on every call — but every
        note that reaches air gets one, because "how many facts were said more
        than once" is the whole measurement section 7 asks for.
        """
        for item in offered:
            if action == "offered" and not item.callback:
                continue
            self._publish(
                Topic.THREAD,
                cursor,
                action=action,
                thread=item.index,
                subject=item.subject,
                note=item.note.text,
                times_said=item.times_said,
                callback=item.callback,
            )


    def _publish_ledger(self, cursor: float, facts: Sequence[LedgerFact], action: str) -> None:
        """Put a count on the bus, offered or spoken, so it can be counted.

        Every offer, unlike :meth:`_publish_threads`, which only rows a
        callback: a count offered and not taken is the measurement this is
        for. Gap 1 of the corpus study is that numbers barely reach air, and
        the only way to tell a voice that will not say them from a system
        that never gives it one is to trace both ends.
        """
        for fact in facts:
            self._publish(
                Topic.LEDGER,
                cursor,
                action=action,
                about=fact.about,
                kind=fact.kind,
                count=fact.count,
                text=fact.text,
            )

    def _counts_for(self, line: CallerLine, cursor: float) -> list[LedgerFact]:
        """Every count about anybody this line could be talking about."""
        ball = self.state.ball
        wanted = ([ball.player] if ball is not None else []) + self._named_by(
            line, self._carried_name(line, cursor)
        )
        wanted += [self.state.home, self.state.away]
        return self.ledger.facts(cursor, wanted)

    def _counts_said(self, text: str, cursor: float) -> list[LedgerFact]:
        """The counts this line actually put on air."""
        facts = self._counts_for(
            CallerLine(
                scene=Scene.LIVE_PLAY, event=Event.NONE, confidence=1.0, speak=True, line=text
            ),
            cursor,
        )
        return [facts[index] for index in facts_used(text, facts, self.pack)]

    @staticmethod
    def _named_by(line: CallerLine, carried: str | None) -> list[str]:
        """Who this moment is about, most relevant first.

        The carried name leads because it is the player the last line was
        already about and the one still on the ball; the sightings follow in
        the order the caller wrote them. Both teams are added by the tracker,
        at the end, where they lose to anything more specific.
        """
        names = [carried] if carried else []
        names += [s.name for s in line.sightings if s.name]
        return names

    def _mark_spoken(
        self, cursor: float, text: str, event: Event | None = None, *, replay: bool = False
    ) -> None:
        """Remember when the voice was last given a line, and how long a one.

        The length is the speaker's own arithmetic — words over
        ``WORDS_PER_SECOND`` — rather than the ``seconds`` the director
        records once the line has been said. It has to be: the next tick can
        come half a second after the beat is submitted, before a word of it
        has been spoken, and a cap that waits for the measurement would spend
        every short line's window using the previous line's length.

        ``replay`` picks the rate rather than the event does. A replay is the
        broadcast's own dead ball — the ball is not in play behind it and the
        corpus's gaps over one run with the restarts, not with the move being
        shown (``docs/research/real-commentary-corpus.md`` section 2.3) — so a
        line about a replayed shot takes the 5.0 s dead-ball cap and not the
        2.5 s of an attacking move.
        """
        self._last_spoken_video_ts = cursor
        self._last_spoken_event = event
        self._last_spoken_replay = replay
        self._last_quiet_ts = None
        words = len(text.split())
        self._last_spoken_seconds = words / WORDS_PER_SECOND if words else None

    def _next_cut_after(self, cursor: float) -> float | None:
        """Where the near future stops being the same passage of play.

        A broadcast cuts away every few seconds, and everything past the cut
        belongs to a different picture: a replay, the bench, a face in the
        crowd. Those frames are not what happens next, so the caller does not
        get to see them as though they were.
        """
        ahead = [ts for ts in self._cuts if ts > cursor]
        return min(ahead) if ahead else None

    def _board_supports_goal(self, cursor: float) -> bool:
        """Is there anything on the scoreboard behind a goal claimed here?

        Three things count, and they answer three different questions the
        first real run asked in its first three minutes.
        """
        return (
            self._board_changed_near(cursor)
            or self._board_pending_near(cursor)
            or self._goal_already_in_the_state(cursor)
        )

    def _board_pending_near(self, cursor: float) -> bool:
        """Is the board *in the middle of* agreeing with this claim?

        The tracker needs three agreeing reads and they land every three and
        a half seconds, so on the real run a goal the bug showed at 63.7 was
        not confirmed until 70.0 — thirteen seconds after the ball crossed
        the line, which no delay we would run at covers.

        One board read agreeing with a caller that has independently claimed
        a goal is two sources, and that is what the gate wants. It is not
        enough to move the score: the state still waits for three reads. The
        difference matters because a single misread digit that corroborates
        nothing decays away, while one that happens to line up with a real
        claim was probably not a misread.
        """
        pending = self.board_tracker.pending_goal
        if pending is None:
            return False
        window = min(GOAL_GRAPHIC_LAG_S, self.settings.capture.delay_s)
        return cursor - 2.0 <= pending.first_ts <= cursor + window

    def _bind_sightings(self, line: CallerLine, cursor: float) -> None:
        """Put the numbers and names the caller read into the registry.

        The team sheets are the check: a number has to be in that side's
        squad and a name has to be on the roster, because a hallucinated pair
        here would not just be one wrong line, it would be carried into the
        next one. What survives goes to the registry, which is what a later
        line reads a name back out of.
        """
        if not line.sightings:
            return
        seen: list[dict[str, Any]] = []
        for sighting in line.sightings:
            found = self._roster_check(sighting, cursor)
            row: dict[str, Any] = {
                "number": sighting.number,
                "name": sighting.name,
                "side": sighting.side.value,
                "bound": found is not None,
            }
            if found is not None:
                side, number, name = found
                self.state_tracker.registry.believe(number, name, cursor, side=side)
                row["as"] = f"{number} {name}"
                self.stats.sightings += 1
            else:
                self.stats.sightings_dropped += 1
            seen.append(row)
        self._publish(Topic.SIGHTING, cursor, sightings=seen)

    def _roster_check(self, sighting: Sighting, cursor: float) -> tuple[Side, int, str] | None:
        """The player this sighting is about, or None if it does not stand up.

        A name settles which side it is and which number goes with it, so a
        number given alongside has to agree.

        A number needs a side, because both squads wear a 5, a 7 and an 11.
        The side comes from the caller: it is looking at the kit and it has
        both kit strings in its team sheets, and a model reading the picture
        does not confuse white stripes with navy. It used to come from the
        kit split, which is the weakest link in the chain by its own
        docstring: on the real clip it put an Argentina body on France, and a
        sighting of "26" on it became Marcus Thuram in a passage Argentina
        played the whole of. A number with no side still names nobody — that
        is the case that was throwing away half of every read, and the answer
        to it is for the caller to say which kit, not for anything here to
        guess.
        """
        if self.pack is None:
            return None
        if sighting.name:
            found = _player_named(self.pack, sighting.name)
            if found is None:
                return None
            side, player = found
            if player.number is None:
                return None
            if sighting.number is not None and sighting.number != player.number:
                return None
            if sighting.side is not Side.UNKNOWN and sighting.side is not side:
                # The name and the kit disagree, so one of them was misread
                # and there is no way to tell which.
                return None
            return side, player.number, player.name
        if sighting.number is None:
            return None
        if sighting.side is not Side.UNKNOWN:
            wearer = _player_numbered(self.pack, sighting.side, sighting.number)
            return (sighting.side, sighting.number, wearer.name) if wearer else None
        wearing = [
            (side, wearer)
            for side in (Side.HOME, Side.AWAY)
            if (wearer := _player_numbered(self.pack, side, sighting.number)) is not None
        ]
        if len(wearing) != 1:
            return None
        one_side, wearer = wearing[0]
        return one_side, sighting.number, wearer.name

    def _remember_on_the_ball(self, line: CallerLine, spoken: str, cursor: float) -> None:
        """Record who this line was about, so the next one may keep the name.

        The name has to be in the line that was actually spoken: a sighting
        the caller reported and did not say is not what the commentary was
        about, and carrying it would be inventing a subject rather than
        keeping one.
        """
        registry = self.state_tracker.registry
        dead = line.event in (Event.PENALTY, Event.FREE_KICK, Event.CORNER, Event.THROW_IN)
        for sighting in line.sightings:
            name = sighting.name
            if not name:
                continue
            if fold(name.rsplit(" ", 1)[-1]) in fold(spoken):
                registry.name_on_the_ball(name, cursor, dead_ball=dead)
                self._carry_side = sighting.side
                return

    def _carried_name(self, line: CallerLine, cursor: float) -> str | None:
        """The name the last line had on the ball, if this line may keep it.

        One case only. The last line named the player on the ball within
        :data:`CARRY_NAME_S`, the phase of play has not changed — no restart
        and no change of possession — and this line is about the same run or
        shot. Then the name is still the name of the man being described, and
        the caller does not have to read a number that is facing away.

        The dead-ball case was specified and is not implemented. On the
        Netherlands clip the one name the caller had at a penalty was de Jong
        and the taker was van Dijk: a sighting says "I read this number on
        somebody in this picture", never "this is the man on the ball".
        """
        registry = self.state_tracker.registry
        held = registry.on_the_ball
        if held is None or held.dead_ball:
            return None
        if line.event in (Event.KICKOFF, Event.THROW_IN, Event.CORNER, Event.FREE_KICK):
            return None
        if line.side is not Side.UNKNOWN and self._carry_side not in (Side.UNKNOWN, line.side):
            return None
        return registry.carried_name(cursor, window_s=CARRY_NAME_S)

    def _note_restart(self, line: CallerLine, cursor: float) -> None:
        """Has the game gone again since the goal the state is holding?

        The caller writing a kickoff. There used to be a second way, a
        referee's whistle heard since the goal and then a live picture, and
        on real broadcast the whistle detector fired once in 58 runs, so it
        never was a way. Past the cap the talk ends anyway.

        A kickoff does not count while the score bug is away or the board
        reader thinks we are in a replay. That is the broadcaster's own answer to "has the
        game started again", it does not depend on the caller getting the
        scene right, and the caller does not: it wrote a kickoff over a replay
        of the goal.
        """
        if self._last_goal_ts is None or self._restart_ts is not None:
            return
        if not self.state.bug_visible or self.state.in_replay:
            # The broadcaster pulls the score bug for the replays and brings
            # it back when the game does. Until it is back we are watching
            # the goal, whatever the caller calls the scene — on the second
            # real run it wrote "kickoff, live play" over a replay at cursor
            # 89.5 and ended goal talk sixty seconds early, which cost the
            # celebration lines that followed.
            return
        if line.event is Event.KICKOFF:
            self._restart_ts = cursor

    def _goal_already_in_the_state(self, cursor: float) -> bool:
        """Is this a line about a goal the state has already taken in?

        The celebration, the replay, the scorer's face, the walk back: all of
        them are lines about a goal, and none of them is near the moment the
        board moved. Asking only whether the board moved *at the cursor*
        rejected four correct lines about one goal over eighty seconds.

        It runs until the game does. A fixed window cannot be the answer
        because a broadcaster's celebration is not a fixed length — the
        second real run lost two correct lines to a 45-second one while the
        picture was still on the scorer — and the thing that ends it is not
        a clock but the ball being kicked off again. The cap is for the
        restart nobody saw.
        """
        if self._last_goal_ts is None:
            return False
        since = cursor - self._last_goal_ts
        # A line from just before the state caught up is a line about the same
        # goal: the caller watched the ball cross the line and the graphic
        # followed it, which is the whole of what GOAL_GRAPHIC_LAG_S measures.
        # The third run lost a correct goal call at cursor 59.3 to a state
        # that applied the board at 62.2.
        if not -GOAL_GRAPHIC_LAG_S <= since <= GOAL_TALK_CAP_S:
            return False
        return self._restart_ts is None or cursor < self._restart_ts

    def _score_counts_the_goal(self, cursor: float) -> bool:
        """Does the score the gate is about to read already include this goal?

        A narrower question than :meth:`_goal_already_in_the_state`, and the
        two were one method until a line said a score out loud. That one runs
        from ten seconds *before* the state caught up, because a caller who
        watched the ball cross the line is talking about the same goal the
        graphic is about to show, and a line in that gap deserves its cover.

        The arithmetic cannot use the same window. In that gap the state still
        reads 2-0 while the ball is in the net for 2-1, and answering
        "settled" there is what struck out two correct scorelines on the
        Mbappé trace — "Mbappé! The volley, buried! Two-one." against a state
        row that still said 2-0 — as scoreline_mismatch, for being one goal
        ahead of a board that had not moved. One goal ahead of a board that
        has not moved is precisely the latitude the gate grants, and it was
        granting it nowhere.

        So: a goal change still queued for the state is a goal the score does
        not count, whatever earlier goals it does count. ``_board_changes``
        holds only what has not been applied — :meth:`_apply_due_board_changes`
        takes each one out as the cursor reaches it — so a queued goal within
        the graphic's own lag is this goal, arriving.
        """
        arriving = any(
            change.is_goal and change.ts - cursor <= GOAL_GRAPHIC_LAG_S
            for change in self._board_changes
        )
        if arriving:
            return False
        return self._last_goal_ts is not None and cursor >= self._last_goal_ts

    def _board_changed_near(self, cursor: float) -> bool:
        """Did the scoreboard move around the moment being called?

        A goal shows on the graphic a beat after the ball crosses the line, so
        the change sits slightly ahead of the cursor, and seeing it early is
        what the delay buys. But the window is a fact about broadcast
        graphics, not about our buffer: it must not widen just because we
        chose to wait longer.

        The first version used ``delay_s`` as the window, and so made the gate
        more permissive the deeper the buffer got — at eight seconds it would
        accept a board change eight seconds after the cursor as proof of a
        goal being called now, which is often a different passage of play
        entirely. Measured across the sweep, phantom goals reaching air went
        1, 1, 4, 6 as the buffer deepened. The delay was buying the caller
        information and paying for it by loosening the gate, which is most of
        why the delay chart showed nothing.
        """
        window = min(GOAL_GRAPHIC_LAG_S, self.settings.capture.delay_s)
        recent = [c for c in self._board_changes if c.is_goal]
        return any(cursor - 2.0 <= c.ts <= cursor + window for c in recent)

    def _wire_confirms_goal(self, cursor: float) -> bool:
        """Has the statistician said a goal went in around this moment?

        Known rather than applied: the feed's own latency is the thing being
        modelled, and a goal it has reported is evidence from the instant it
        reports it even though the state waits for the cursor.
        """
        if self._sync is None:
            return False
        return any(
            event.event is Event.GOAL
            and event.video_ts is not None
            and cursor - GOAL_GRAPHIC_LAG_S <= event.video_ts <= cursor + 2.0
            for event in self._sync.known
        )

    # -- plumbing --------------------------------------------------------

    def _fire(self, trigger: Trigger) -> None:
        """Park a trigger until the next tick reads them all at once.

        Triggers arrive on the frame loop, which runs far faster than the
        system can speak. Collecting them and letting the tick decide is what
        keeps a burst of three cuts from becoming three separate attempts to
        say something.
        """
        self._pending.append(trigger)

    def _drain_triggers(self) -> list[Trigger]:
        drained = list(dict.fromkeys(self._pending))
        self._pending.clear()
        return drained

    def _publish(self, topic: Topic, ts: float, value: Any = None, **extra: Any) -> None:
        self.bus.publish(topic, ts, value, **extra)

    async def _record(self) -> None:
        """Write everything on the bus to the trace.

        The trace subscribes rather than being written to directly, because
        the director publishes on its own — a line that was spoken, or cut off
        mid-word, is the director's news, not the runtime's. Anything written
        on a second path is a thing the eval would silently never see.
        """
        if self.trace is None:
            return
        async for message in self.bus.subscribe():
            self.trace.write(message)


def trace_path(root: Path, name: str) -> Path:
    return root / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
