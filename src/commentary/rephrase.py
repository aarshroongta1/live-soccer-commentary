"""Say a finished run's lines again, in the register, for a fraction of a cent.

A run costs real money and happens once. The traces in ``runs/`` are the
whole evidence base for this system and every one of them was paid for on
Opus at about thirty cents a minute of video, so the way to find out whether
a phrasing stage helps is emphatically not to run the match again.

Everything the phraser needs is already in the trace. The caller's form is a
``caller`` row. The state it was looking at is the last ``state`` row before
it. What had just been said is the ``beat`` rows above it. So this reads a
trace, calls the phraser once per spoken caller line at Haiku prices, puts
each rewrite back through :class:`~commentary.gate.FactGate`, and writes a
new trace that :func:`commentary.replay.from_files` will play through the
watch page and a real voice exactly like any other.

**Three things are approximations, and they are all in the gate's cover
flags.** A live runtime hands the gate three facts that live in the runtime's
own head rather than on the bus:

``board_changed``
    Whether the scoreboard supports a goal claimed at this moment. Not in the
    trace as such, but implied by it: the original gate would have rejected a
    goal claim with ``unconfirmed_goal`` if nothing had supported it, so a
    goal claim that *passed* proves the board was behind it. That is what is
    reconstructed here — true when the original line claimed a goal and its
    gate row passed, false otherwise. A phrased line that invents a goal
    claim where the caller made none therefore meets ``board_changed`` false
    and is rejected, which is the direction to be wrong in.

``goal_in_state``
    Whether the score already counts the goal this line is about. Rebuilt
    from the ``incidents`` list on the ``state`` rows, which does carry it:
    the first state row whose incidents include a goal is the moment the
    state took that goal in. A goal taken in within ``GOAL_GRAPHIC_LAG_S``
    after the line is the goal the line is about, arriving, and the number is
    not settled yet; anything earlier is. The same question
    ``Runtime._score_counts_the_goal`` answers, asked of the very state row
    the gate is handed here, so the two cannot disagree.

``carried``
    The name the previous line had on the ball. The registry is not in the
    trace, so it is rebuilt the way the runtime builds it: the last sighting
    name that actually appeared in the previous spoken line, inside
    ``CARRY_NAME_S``, not from a dead ball. Without it a phrased line that
    keeps a name the caller carried would be trimmed for a name the roster
    check cannot see the source of.

``wire_confirmed`` is always false: no trace on disk was run with a wire.

What the new trace contains: every row of the old one, in order, plus a
``phrased`` row per rewritten line carrying the original words, the new ones,
the excitement and what the call cost; caller ``beat`` rows with their text
replaced; a ``gate`` row in place of any beat the gate now refuses; and no
``spoken`` or ``preempted`` rows at all, because those are last time's
account of the speaking and ``replay --voice`` writes this time's.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.agents.colour import ColourPass, _merge, colour_pass
from commentary.agents.phraser import Phraser, nameless_build_up, roster_names
from commentary.bus import Topic
from commentary.config import SETTINGS, Settings
from commentary.gate import FactGate, claims_goal, facts_used, fold
from commentary.goalfollow import FOLLOWUP_S, GoalFollowup, blocks_restatement
from commentary.ledger import CONTEXT_FACTS, Ledger
from commentary.ledger import Fact as LedgerFact
from commentary.llm.base import LLMBackend
from commentary.runtime import CARRY_NAME_S, GOAL_GRAPHIC_LAG_S, ReplaySequence
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    GateVerdict,
    KnowledgePack,
    MatchState,
    PhrasedLine,
    Scene,
    Voice,
)
from commentary.scoreline import Numbers, Restatements, settle_numbers
from commentary.threads import Offered, Threads
from commentary.trace import read_trace

#: Rows the replay regenerates for itself when a voice is attached. Dropped
#: rather than rewritten: they are an account of an evening's audio, and the
#: lines they are about are not the lines any more.
DROPPED = frozenset({Topic.SPOKEN.value, Topic.PREEMPTED.value})

#: Two rows are "the same moment" if their timestamps are this close. They
#: are written from the same float in the same process, so this only guards
#: against JSON round-tripping.
SAME_TS = 1e-6


@dataclass
class Line:
    """One caller beat, before and after. The row of the printed table."""

    ts: float
    original: str
    phrased: str
    excitement: float
    passed: bool
    reason: str
    usd: float = 0.0
    #: The event the phrased line amounts to — the form's, unless the words
    #: promote it to a goal, which is the same reading the director uses.
    #: Side by side with the form's own event it is where a rewrite that
    #: invented an outcome shows up.
    event: Event = Event.NONE
    form_event: Event = Event.NONE
    #: The phraser had nothing and the caller's own line went through.
    fallback: bool = False
    #: The scoreline code appended to this line, or "" if it appended none.
    #: Exactly one line per goal should carry one.
    appended: str = ""
    #: What :func:`commentary.scoreline.strip_score` took out of the model's
    #: words before the gate saw them. Every one of these used to be a line
    #: refused whole.
    stripped: tuple[str, ...] = ()
    #: What :func:`commentary.scoreline.strip_how_not_in_form` took out
    #: beside a penalty: a how the model reached for that this goal's own
    #: form never gave it. "82.5 Mbappé! Over the wall!" is why this exists.
    how_stripped: tuple[str, ...] = ()
    #: The phraser was never called for this line: code wrote it, off the
    #: state. The score-and-clock restatement, and nothing else so far.
    written_by_code: bool = False
    #: An extra call made in a gap the caller left in the thirty seconds
    #: after a goal.
    synthetic: bool = False
    #: A line said over a replay, out of a caller form the original run
    #: never spoke. See :func:`rephrase`'s replay handling.
    replay: bool = False
    #: The phraser chose to say nothing, which is a line in itself. Real
    #: commentary passes over 43% of goal kicks and a quarter of build-up
    #: touches (the corpus study, section 3), and counting the ones this
    #: system chooses is the only way to know whether it has learned to.
    silent: bool = False

    @property
    def verdict(self) -> str:
        if self.silent:
            return "chose silence"
        if self.fallback:
            return "fell back"
        if not self.passed:
            return self.reason or "rejected"
        marks = []
        if self.appended:
            marks.append("+score")
        if self.stripped:
            marks.append("-score")
        if self.written_by_code:
            marks.append("by code")
        if self.synthetic:
            marks.append("extra")
        if self.replay:
            marks.append("replay")
        return "passed " + " ".join(marks) if marks else "passed"

    @property
    def events(self) -> str:
        """``penalty`` or ``penalty->goal`` when the words moved it."""
        if self.event is self.form_event:
            return self.form_event.value
        return f"{self.form_event.value}->{self.event.value}"


@dataclass
class Rephrased:
    """What came back: the new rows, the table, and what it cost."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    cost_usd: float = 0.0
    #: What the colour seat did on this pass, or ``None`` when it was not
    #: run. Its rows are already in :attr:`rows`; this is the account of
    #: them, for the printed table and the counts.
    colour: ColourPass | None = None

    @property
    def rejected(self) -> int:
        return sum(1 for line in self.lines if not line.passed)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row) + "\n")
        return path

    def table(self) -> str:
        """Side by side, one line per rewritten beat."""
        head = f"{'ts':>7}  {'exc':>4}  {'event':<16}  {'verdict':<24}  original / phrased"
        out = [head, "-" * len(head)]
        for line in self.lines:
            out.append(
                f"{line.ts:>7.1f}  {line.excitement:>4.2f}  {line.events:<16}  "
                f"{line.verdict:<24}  {line.original}"
            )
            out.append(f"{'':>7}  {'':>4}  {'':<16}  {'':<24}  -> {line.phrased}")
        return "\n".join(out)


def _rows_by_ts(rows: list[dict[str, Any]], topic: str) -> dict[float, dict[str, Any]]:
    """The first row of this topic at each timestamp.

    First rather than last: a caller row and its gate row share a timestamp
    with nothing else, and a duplicate would be a second call at the same
    cursor, which the runtime cannot produce.
    """
    found: dict[float, dict[str, Any]] = {}
    for row in rows:
        if row.get("topic") == topic:
            found.setdefault(float(row.get("ts", 0.0)), row)
    return found


def _nearest(index: dict[float, dict[str, Any]], ts: float) -> dict[str, Any] | None:
    row = index.get(ts)
    if row is not None:
        return row
    for key, value in index.items():
        if abs(key - ts) <= SAME_TS:
            return value
    return None


def _caller_line(payload: dict[str, Any]) -> CallerLine | None:
    try:
        return CallerLine.model_validate(payload)
    except Exception:
        return None


def _state_at(states: list[tuple[float, MatchState]], ts: float) -> MatchState | None:
    """The latest state row at or before ``ts``."""
    found: MatchState | None = None
    for row_ts, state in states:
        if row_ts <= ts + SAME_TS:
            found = state
        else:
            break
    return found


def _goal_taken_in(states: list[tuple[float, MatchState]]) -> list[float]:
    """When the state first held each goal, in cursor time.

    A ``state`` row is published when something changes, and its
    ``incidents`` list grows by one when a goal is applied. So the timestamp
    of the row on which the list first grows is the runtime's own
    ``_last_goal_ts`` — the moment the state took the goal in, not the moment
    the ball crossed the line.
    """
    moments: list[float] = []
    held = 0
    for ts, state in states:
        goals = sum(1 for incident in state.incidents if incident.event is Event.GOAL)
        if goals > held:
            moments.extend([ts] * (goals - held))
            held = goals
    return moments


class Cover:
    """The three facts the gate needs that the runtime kept in its head.

    See this module's docstring for what each one is and how faithfully it
    can be rebuilt. Kept together in one object so that the approximations
    are in one place rather than spread through the loop.

    A fourth fact lives here too, for the how-backstop rather than the gate:
    the last spoken form's event and when, mirroring ``Runtime._recent_event``
    so that :func:`commentary.scoreline.strip_how_not_in_form` sees the same
    "was the form just before this one a penalty" that the live runtime does.
    """

    def __init__(self, states: list[tuple[float, MatchState]]) -> None:
        self._goals = _goal_taken_in(states)
        #: The last name a spoken line put on the ball, and when.
        self._held: tuple[str, float] | None = None
        #: The last spoken form's event, and when. Set after a line is
        #: judged, not before, so the line being judged still sees whatever
        #: was true of the one before it.
        self._recent_event: tuple[Event, float] | None = None

    def goal_in_state(self, ts: float) -> bool:
        """Does the score the gate is about to read already include this goal?

        Mirror of ``Runtime._score_counts_the_goal``, asked of the very state
        row the gate is handed. A goal the state takes in within the graphic's
        own lag *after* ``ts`` is the goal this line is about, arriving: the
        row in the gate's hand still says 2-1 and the ball is in the net for
        2-2. Any earlier goal is settled and stays settled.

        Asking the wider "is this a line about a goal" — the cover question,
        which runs from ten seconds before the state catches up — told the
        gate the number was settled in exactly that gap, and struck out two
        correct scorelines on the Mbappé trace.
        """
        if any(0.0 < at - ts <= GOAL_GRAPHIC_LAG_S for at in self._goals):
            return False
        return any(at <= ts for at in self._goals)

    def board_changed(self, line: CallerLine, gate_row: dict[str, Any] | None) -> bool:
        """Did the scoreboard back a goal here? Only a passed goal claim says so."""
        if not claims_goal(line.line, line.event):
            return False
        return bool(gate_row is None or gate_row.get("passed", False))

    def carried(self, line: CallerLine, ts: float) -> str | None:
        """The name the last line had on the ball, if this line may keep it."""
        if self._held is None:
            return None
        if line.event in (Event.KICKOFF, Event.THROW_IN, Event.CORNER, Event.FREE_KICK):
            return None
        name, at = self._held
        return name if 0.0 <= ts - at <= CARRY_NAME_S else None

    def remember(self, line: CallerLine, spoken: str, ts: float) -> None:
        """Mirror of ``Runtime._remember_on_the_ball``, minus the registry."""
        dead = line.event in (Event.PENALTY, Event.FREE_KICK, Event.CORNER, Event.THROW_IN)
        for sighting in line.sightings:
            name = sighting.name
            if not name:
                continue
            if fold(name.rsplit(" ", 1)[-1]) in fold(spoken):
                self._held = None if dead else (name, ts)
                return

    def recent_event_within(self, ts: float, seconds: float) -> Event | None:
        """Mirror of ``Runtime._recent_event_within``: whatever was just spoken."""
        if self._recent_event is None:
            return None
        event, at = self._recent_event
        return event if ts - at <= seconds else None

    def note_event(self, line: CallerLine, ts: float) -> None:
        """Mirror of the line at the end of ``Runtime._call``: remember what aired."""
        if line.event is not Event.NONE:
            self._recent_event = (line.event, ts)


async def rephrase(
    rows: list[dict[str, Any]],
    backend: LLMBackend,
    *,
    pack: KnowledgePack | None = None,
    settings: Settings = SETTINGS,
    model: str | None = None,
    colour: bool = True,
    colour_model: str | None = None,
) -> Rephrased:
    """Rewrite every spoken caller line in a trace, and judge the rewrites.

    Nothing here touches the clip, the board reader or the caller. Two models
    are called: the phraser, once per line that was actually spoken, and —
    unless ``colour`` is false — the colour seat, once at each moment its
    phase gate allows, which on a three-minute trace is a handful of times.
    """
    states: list[tuple[float, MatchState]] = []
    for row in rows:
        if row.get("topic") == Topic.STATE.value:
            payload = {k: v for k, v in row.items() if k not in ("topic", "ts")}
            try:
                states.append((float(row.get("ts", 0.0)), MatchState.model_validate(payload)))
            except Exception:
                continue

    callers = _rows_by_ts(rows, Topic.CALLER.value)
    gates = _rows_by_ts(rows, Topic.GATE.value)
    teams = states[0][1] if states else MatchState(home="Home", away="Away")

    phraser = Phraser(
        backend,
        config=settings.phraser,
        silence=settings.silence,
        dead_ball=settings.dead_ball,
        model=model,
        home=teams.home,
        away=teams.away,
    )
    gate = FactGate(settings.gate)
    cover = Cover(states)
    #: The pack's notes with a memory: what has been said, when, and what the
    #: match has done to any number in them. The follow-up shares it, so beat
    #: 3 of a goal and a clause dropped into a lull are one selection and one
    #: tally. ``docs/research/real-commentary-corpus.md`` section 7.
    threads = Threads.from_pack(pack)
    #: And what the broadcast counted for itself as the walk goes past it:
    #: corners, fouls, shots, who has had how many. Sharing the threads'
    #: tallies, so a player's goals in this match are counted in one place.
    #: ``docs/research/real-commentary-corpus.md`` section 5.2.
    ledger = Ledger.from_pack(pack, tallies=threads.tallies)
    follow = GoalFollowup(threads=threads)
    out = Rephrased()
    #: When each goal's follow-up window opened, for the restatement pass:
    #: it must stay out of every one of these, not just the last one a match
    #: with more than one goal has armed by the time the main loop is done.
    goal_calls: list[float] = []

    def thread_rows(at: float, offered: list[Offered], action: str) -> None:
        """Trace what was picked back up, so that it can be counted."""
        for item in offered:
            if not item.callback and action == "offered":
                continue
            out.rows.append(
                {
                    "topic": Topic.THREAD.value,
                    "ts": at,
                    "action": action,
                    "thread": item.index,
                    "subject": item.subject,
                    "note": item.note.text,
                    "times_said": item.times_said,
                    "callback": item.callback,
                }
            )

    def ledger_rows(at: float, facts: Sequence[LedgerFact], action: str) -> None:
        """Trace every count offered and every count said, so both can be counted.

        Both ends, unlike ``thread_rows``, which only records an offer that
        was a callback. Gap 1 of the corpus study is that numbers barely
        reach air, and the only way to tell a voice that will not say one
        from a system that never offers one is to have both rows.
        """
        for fact in facts:
            out.rows.append(
                {
                    "topic": Topic.LEDGER.value,
                    "ts": at,
                    "action": action,
                    "about": fact.about,
                    "kind": fact.kind,
                    "count": fact.count,
                    "text": fact.text,
                }
            )

    def counts_for(at: float, names: Sequence[str]) -> list[LedgerFact]:
        return ledger.facts(at, names)

    def counts_said(text: str, at: float, names: Sequence[str]) -> list[LedgerFact]:
        facts = counts_for(at, names)
        return [facts[index] for index in facts_used(text, facts, pack)]

    #: Every caller form in the trace, spoken or not, oldest first. The
    #: follow-up state is fed all of them: after the Mbappé penalty the caller
    #: wrote four forms in a row and said none of them, and those four are the
    #: only account anywhere of how the goal was scored.
    forms = sorted(
        (
            (ts, form)
            for ts, row in callers.items()
            if (form := _caller_line({k: v for k, v in row.items() if k not in ("topic", "ts")}))
        ),
        key=lambda pair: pair[0],
    )
    fed = 0

    def feed(upto: float) -> None:
        """Hand the follow-up every form the caller filled in up to now."""
        nonlocal fed
        while fed < len(forms) and forms[fed][0] <= upto + SAME_TS:
            ledger.saw_form(forms[fed][0], forms[fed][1])
            follow.saw_form(forms[fed][1])
            fed += 1

    #: Every replay form with words in it, in order. These are the lines the
    #: original run never said: the caller filled the form in and three
    #: layers — its own veto, the gate's ``scene_replay`` and the runtime's
    #: ``speak`` check — threw it away. Replay mode says them.
    replay_times = [
        ts for ts, form in forms if form.scene is Scene.REPLAY and form.line.strip()
    ]

    #: The run of replay pictures, and how much of it has been talked over.
    #: The same object the runtime uses, so live and offline agree about what
    #: a sequence is and how many lines it gets.
    replays = ReplaySequence(cfg=settings.replay_talk)

    #: When the lead spoke in the original run. What a replay line has to
    #: stay clear of: everything in this list is already in the trace.
    lead_times = sorted(
        float(row.get("ts", 0.0))
        for row in rows
        if row.get("topic") == Topic.BEAT.value and row.get("voice") == Voice.CALLER.value
    )

    #: When the lead speaks, for the gap arithmetic that decides whether a
    #: follow-up beat has to be synthesised. The replay candidates are in
    #: here too, and that is how the goal window learns to prefer a replay
    #: rebuild over a synthesised one: ``GoalFollowup.synth_times`` takes the
    #: next lead line as its ``until``, and a replay four seconds away closes
    #: the gap the synthesiser was going to fill. Two voices rebuilding the
    #: same move is worse than one, and the replay is the one with a picture
    #: behind it.
    beat_times = sorted([*lead_times, *replay_times])

    async def extra(at: float, state: MatchState) -> bool:
        """One phraser call the caller never asked for, inside a goal window.

        Same phraser, same gate, same strip. What is different is the form:
        it carries the goal line's own sightings and the caller's own words
        about the move, so nothing about the picture is being claimed twice.
        A call that comes back empty, or is refused, simply does not happen —
        there is no fallback here, because there was no caller line to fall
        back to.
        """
        feed(at)
        threads.see_state(state)
        ledger.see_state(state)
        form = follow.synthetic()
        offered = (
            threads.offer([follow.scorer], ts=at, payoff=True) if follow.scorer else []
        )
        thread_rows(at, offered, "offered")
        scorer_names = [follow.scorer] if follow.scorer else []
        counts = counts_for(at, scorer_names)[:CONTEXT_FACTS]
        ledger_rows(at, counts, "offered")
        phrased = await phraser.phrase(
            form,
            _summary(state),
            notes=[item.note for item in offered],
            callbacks=[item.callback for item in offered],
            ledger=counts,
            followup=follow.block(at, pack),
            goal_beat=follow.beat(at),
            scorer=follow.scorer,
            roster=roster_names(pack),
        )
        usd = phraser.last_usage.cost_usd
        out.cost_usd += usd
        if phrased is None or not phrased.line.strip():
            return False
        settled = settle_numbers(
            phrased.line, state=state, side=form.side, goal_in_state=True, append=False
        )
        if not settled.line.strip():
            return False
        verdict = gate.judge(
            form.model_copy(update={"line": settled.line}),
            state,
            pack,
            # The goal this line is about has already passed the gate once,
            # which is the whole of what a board change is evidence for.
            board_changed=True,
            goal_in_state=cover.goal_in_state(at),
            at=at,
            notes=threads.notes(),
            ledger=counts_for(at, scorer_names),
        )
        out.rows.append(
            {
                "topic": Topic.PHRASED.value,
                "ts": at,
                "original": form.line,
                "line": settled.line,
                "excitement": phrased.excitement,
                "event": Event.GOAL.value,
                "form_event": Event.GOAL.value,
                "usd": round(usd, 6),
                "synthetic": True,
                "opener_retry": phrased.opener_retry,
                "closer_retry": phrased.closer_retry,
                "name_retry": phrased.name_retry,
                "shout_retry": phrased.shout_retry,
                "shout_rewritten": phrased.shout_rewritten,
                "score_stripped": list(settled.stripped),
                "tokens_in": phraser.last_usage.input_tokens,
                "cache_read": phraser.last_usage.cache_read_tokens,
                "cache_write": phraser.last_usage.cache_write_tokens,
            }
        )
        if not verdict.passed:
            out.rows.append(
                {
                    "topic": Topic.GATE.value,
                    "ts": at,
                    "passed": False,
                    "reasons": verdict.reasons,
                    "line": settled.line,
                    "event": Event.GOAL.value,
                    "where": "rephrase",
                }
            )
        else:
            out.rows.append(
                {
                    "topic": Topic.BEAT.value,
                    "ts": at,
                    "id": f"synth-{at:.1f}",
                    "voice": Voice.CALLER.value,
                    "text": verdict.line,
                    "video_ts": at,
                    "created_ts": 0.0,
                    "live_ts": at,
                    "event": Event.GOAL.value,
                    "excitement": phrased.excitement,
                    "preemptable": False,
                }
            )
            phraser.accept(verdict.line, Event.GOAL, ts=at)
            thread_rows(at, threads.said(verdict.line, ts=at, pack=pack), "used")
            ledger_rows(at, counts_said(verdict.line, at, scorer_names), "used")
            follow.said(at)
            follow.synthesised += 1
        out.lines.append(
            Line(
                ts=at,
                original=form.line,
                phrased=verdict.line if verdict.passed else settled.line,
                excitement=phrased.excitement,
                passed=verdict.passed,
                reason="; ".join(verdict.reasons)[:60],
                usd=usd,
                event=Event.GOAL,
                form_event=Event.GOAL,
                stripped=settled.stripped,
                synthetic=True,
            )
        )
        return verdict.passed

    async def replay_line(row: dict[str, Any]) -> None:
        """Say a replay form the original run threw away, in the past tense.

        This is the offline half of replay mode. On
        ``runs/trigger/mbappe/file-20260913-185228.jsonl`` the penalty is
        conceded at 12.9 s and the next spoken line is at 49.0 s, and in
        between sit four accurate replay forms — "The replay: driving across,
        the leg in behind him, and down he goes." — with ``speak`` false,
        because the caller vetoed its own replays before the gate ever saw
        them. Every trace on disk was recorded that way, so eligibility here
        deliberately ignores ``speak``: the form is the evidence, and the flag
        is a record of a rule that no longer exists.

        Three things bound it, and none of them is the gate's job:

        * one line per :attr:`~commentary.config.ReplayTalkConfig.min_gap_s`
          and at most ``max_lines`` per sequence, which is
          :class:`~commentary.runtime.ReplaySequence` — the same object the
          live runtime counts with, so the two agree about what a sequence is;
        * clear of a line the lead already has, by ``clear_of_a_beat_s``: a
          rewritten trace already has a beat at that second, and two voices on
          one moment is worse than one;
        * and the sequence is fed every look, spoken or not, because a
          broadcaster cutting back to the game and then to another angle has
          started a second replay.

        What comes out is an ordinary lead beat — ``voice: caller``, at the
        form's own timestamp — so the register measurement and the colour pass
        see it as the lead line it is. The ``replay`` flag on the beat and on
        the phrased row is for the trace printer and the counts; nothing
        downstream reads it.
        """
        ts = float(row.get("ts", 0.0))
        form = _caller_line({k: v for k, v in row.items() if k not in ("topic", "ts")})
        if form is None or form.scene is not Scene.REPLAY:
            return
        replays.look(ts)
        if not form.line.strip() or not replays.may_speak(ts):
            return
        if any(abs(ts - at) <= settings.replay_talk.clear_of_a_beat_s for at in lead_times):
            return

        state = _state_at(states, ts) or teams
        threads.see_state(state)
        ledger.see_state(state)
        feed(ts)
        names = [s.name for s in form.sightings if s.name]
        names += [state.home, state.away]
        phrased = await phraser.phrase(
            form,
            _summary(state),
            # No carry, no notes, no counts. A replay line is a past-tense
            # account of one concrete thing on the picture: who is on the ball
            # is a fact about live play, and a number is the one shape
            # ``replay_block`` forbids outright.
            replay_first=replays.first,
        )
        usd = phraser.last_usage.cost_usd
        out.cost_usd += usd
        if phrased is None or not phrased.line.strip():
            return
        # The score never goes on a replay line, so the strip runs with
        # nothing to append: whatever number the model reached for comes out
        # before the gate reads the line.
        settled = settle_numbers(
            phrased.line,
            state=state,
            side=form.side,
            goal_in_state=cover.goal_in_state(ts),
            append=False,
        )
        if not settled.line.strip():
            return
        verdict = gate.judge(
            form.model_copy(update={"line": settled.line, "speak": True}),
            state,
            pack,
            # Nothing outside the replay is cover for it. The gate's replay
            # rule reads ``goal_in_state`` on its own: the score already
            # counting the goal is the only thing that lets a replay line
            # mention one at all.
            board_changed=False,
            goal_in_state=cover.goal_in_state(ts),
            at=ts,
            notes=threads.notes(),
            ledger=counts_for(ts, names),
        )
        out.rows.append(
            {
                "topic": Topic.PHRASED.value,
                "ts": ts,
                "original": form.line,
                "line": settled.line,
                "excitement": phrased.excitement,
                "event": form.event.value,
                "form_event": form.event.value,
                "usd": round(usd, 6),
                "replay": True,
                "opener_retry": phrased.opener_retry,
                "closer_retry": phrased.closer_retry,
                "name_retry": phrased.name_retry,
                "replay_marker_stripped": phrased.replay_marker_stripped,
                "score_stripped": list(settled.stripped),
                "tokens_in": phraser.last_usage.input_tokens,
                "cache_read": phraser.last_usage.cache_read_tokens,
                "cache_write": phraser.last_usage.cache_write_tokens,
            }
        )
        if verdict.passed:
            out.rows.append(
                {
                    "topic": Topic.BEAT.value,
                    "ts": ts,
                    "id": f"replay-{ts:.1f}",
                    "voice": Voice.CALLER.value,
                    "text": verdict.line,
                    "video_ts": ts,
                    "created_ts": 0.0,
                    "live_ts": ts,
                    # The form's own event, never promoted to a goal by the
                    # words, and always preemptable: the whole of what makes a
                    # replay line safe to say is that the moment the game is
                    # back on the screen it can be dropped.
                    "event": form.event.value,
                    "excitement": phrased.excitement,
                    "preemptable": True,
                    "replay": True,
                }
            )
            phraser.accept(verdict.line, form.event, ts=ts)
            replays.spoke(ts)
            lead_times.append(ts)
            if follow.active(ts):
                # Inside a goal window the replay line *is* the rebuild, so it
                # spends that beat and not merely the next one: the shout and
                # the tally are still owed, and the synthesiser will not write
                # a second past-tense account of the same move.
                follow.rebuilt(ts)
        else:
            out.rows.append(
                {
                    "topic": Topic.GATE.value,
                    "ts": ts,
                    "passed": False,
                    "reasons": verdict.reasons,
                    "line": settled.line,
                    "event": form.event.value,
                    "where": "replay",
                }
            )
        out.lines.append(
            Line(
                ts=ts,
                original=form.line,
                phrased=verdict.line if verdict.passed else settled.line,
                excitement=phrased.excitement,
                passed=verdict.passed,
                reason="; ".join(verdict.reasons)[:60],
                usd=usd,
                event=form.event,
                form_event=form.event,
                stripped=settled.stripped,
                replay=True,
            )
        )

    for row in rows:
        topic = row.get("topic", "")
        if topic in DROPPED:
            continue
        if topic != Topic.BEAT.value or row.get("voice") != Voice.CALLER.value:
            out.rows.append(row)
            if topic == Topic.CALLER.value:
                await replay_line(row)
            continue

        ts = float(row.get("ts", 0.0))
        caller_row = _nearest(callers, ts)
        form = (
            _caller_line({k: v for k, v in caller_row.items() if k not in ("topic", "ts")})
            if caller_row is not None
            else None
        )
        if form is None:
            # A beat with no form behind it: an older trace, or a row the
            # writer lost. Nothing to rephrase from, so it goes through as
            # the caller said it.
            out.rows.append(row)
            continue

        state = _state_at(states, ts) or teams
        threads.see_state(state)
        ledger.see_state(state)
        fell_back = False
        carried = cover.carried(form, ts)
        names = [carried] if carried else []
        names += [s.name for s in form.sightings if s.name]
        names += [state.home, state.away]
        feed(ts)
        passed_over = phraser.passes_over(form, ts=ts, on_the_ball=carried)
        if passed_over is not None:
            # No model call at all — the nameless build-up the corpus passes
            # over a quarter of the time. Same row shape as a chosen silence,
            # with the reason code decided it rather than the model's, and no
            # beat and no gate row because nothing was said.
            out.rows.append(
                {
                    "topic": Topic.PHRASED.value,
                    "ts": ts,
                    "original": form.line,
                    "line": "",
                    "excitement": 0.0,
                    "event": form.event.value,
                    "form_event": form.event.value,
                    "reason": passed_over,
                    "usd": 0.0,
                }
            )
            out.lines.append(
                Line(
                    ts=ts,
                    original=form.line,
                    phrased="",
                    excitement=0.0,
                    passed=True,
                    reason="",
                    usd=0.0,
                    event=form.event,
                    form_event=form.event,
                    silent=True,
                )
            )
            continue
        followup = follow.block(ts, pack)
        if followup and follow.scorer:
            # The scorer's clauses go to the front, because beat 3 is a number
            # about him and the cap falls off the end.
            names = [follow.scorer] + names
        offered = threads.offer(names, ts=ts, payoff=bool(followup))
        thread_rows(ts, offered, "offered")
        counts = counts_for(ts, names)[:CONTEXT_FACTS]
        ledger_rows(ts, counts, "offered")
        phrased = await phraser.phrase(
            form,
            _summary(state),
            on_the_ball=carried,
            notes=[item.note for item in offered],
            callbacks=[item.callback for item in offered],
            ledger=counts,
            followup=followup,
            goal_beat=follow.beat(ts),
            scorer=follow.scorer,
            roster=roster_names(pack),
        )
        usd = phraser.last_usage.cost_usd
        out.cost_usd += usd
        if phrased is not None and not phrased.line.strip() and phraser.chose_silence:
            # Chosen silence: the `phrased` row records it so the judge can
            # count it, and there is no beat and no gate row because nothing
            # was said. The same shape the runtime publishes.
            out.rows.append(
                {
                    "topic": Topic.PHRASED.value,
                    "ts": ts,
                    "original": form.line,
                    "line": "",
                    "excitement": 0.0,
                    "event": form.event.value,
                    "form_event": form.event.value,
                    "reason": phraser.last_reason or "the phraser chose silence",
                    "usd": round(usd, 6),
                    # A silence the retry chose is the retry working: it is
                    # offered "open differently or say nothing" and took the
                    # second.
                    "opener_retry": phrased.opener_retry,
                    "closer_retry": phrased.closer_retry,
                    "name_retry": phrased.name_retry,
                    "tokens_in": phraser.last_usage.input_tokens,
                    "cache_read": phraser.last_usage.cache_read_tokens,
                    "cache_write": phraser.last_usage.cache_write_tokens,
                }
            )
            out.lines.append(
                Line(
                    ts=ts,
                    original=form.line,
                    phrased="",
                    excitement=0.0,
                    passed=True,
                    reason="",
                    usd=usd,
                    event=form.event,
                    form_event=form.event,
                    silent=True,
                )
            )
            continue
        if phrased is None or not phrased.line.strip():
            # Same rule as the runtime: never a silent drop. The caller's own
            # words go out and the trace says the phraser had nothing.
            out.rows.append(
                {
                    "topic": Topic.ERROR.value,
                    "ts": ts,
                    "where": "phraser",
                    "detail": phraser.last_reason or "the phraser returned nothing",
                }
            )
            out.rows.append(row)
            phrased = PhrasedLine(line=form.line, excitement=0.0)
            verdict = GateVerdict(passed=True, line=form.line)
            fell_back = True
            settled = Numbers(line=form.line)
        else:
            # Numbers by code. The model's words keep the name and the how;
            # any score in them comes out, and the one the state supports goes
            # on — once, on the line that calls the goal, and never on the
            # celebration after it. ``docs/HANDOFF.md`` section 3d.
            is_call = follow.is_the_call(ts) and claims_goal(phrased.line, form.event)
            # The how gets the same treatment beside a penalty: this form's
            # own event, or the last spoken form's within the ten seconds a
            # penalty's kick and its goal sit apart — the check "82.5 Mbappé!
            # Over the wall!" went out without.
            penalty = form.event is Event.PENALTY or (
                cover.recent_event_within(ts, 10.0) is Event.PENALTY
            )
            settled = settle_numbers(
                phrased.line,
                state=state,
                side=form.side,
                goal_in_state=cover.goal_in_state(ts),
                append=is_call,
                description=form.line,
                penalty=penalty,
            )
            # A line that was nothing but a number now has nothing in it, and
            # the gate refuses it as empty. That is the right answer: the model
            # was asked for words and wrote arithmetic.
            phrased = phrased.model_copy(update={"line": settled.line})
            verdict = gate.judge(
                form.model_copy(update={"line": phrased.line}),
                state,
                pack,
                board_changed=cover.board_changed(form, _nearest(gates, ts)),
                goal_in_state=cover.goal_in_state(ts),
                carried=cover.carried(form, ts),
                at=ts,
                notes=threads.notes(),
                # Every count about anybody the line names, not only the two
                # the phraser was shown: the check is whether the number is
                # one the match holds, and the match holds all of them.
                ledger=counts_for(ts, names),
            )
            out.rows.append(
                {
                    "topic": Topic.PHRASED.value,
                    "ts": ts,
                    "original": form.line,
                    "line": phrased.line,
                    "excitement": phrased.excitement,
                    "event": _event_of(phrased.line, form.event).value,
                    "form_event": form.event.value,
                    "usd": round(usd, 6),
                    "opener_retry": phrased.opener_retry,
                    "closer_retry": phrased.closer_retry,
                    "name_retry": phrased.name_retry,
                    "shout_retry": phrased.shout_retry,
                    "shout_rewritten": phrased.shout_rewritten,
                    "score_appended": settled.appended,
                    "score_stripped": list(settled.stripped),
                    "how_stripped": list(settled.how_removed),
                    # Per call, because the aggregate cannot say whether the
                    # cached prefix was ever read: a run where every call
                    # shows cache_read zero is paying full price for two
                    # hundred utterances on every line.
                    "tokens_in": phraser.last_usage.input_tokens,
                    "cache_read": phraser.last_usage.cache_read_tokens,
                    "cache_write": phraser.last_usage.cache_write_tokens,
                }
            )
            if verdict.passed:
                out.rows.append(_rewritten(row, verdict.line, phrased.excitement))
            else:
                out.rows.append(
                    {
                        "topic": Topic.GATE.value,
                        "ts": ts,
                        "passed": False,
                        "reasons": verdict.reasons,
                        "line": phrased.line,
                        "event": form.event.value,
                        "where": "rephrase",
                    }
                )

        out.lines.append(
            Line(
                ts=ts,
                original=form.line,
                phrased=verdict.line if verdict.passed else phrased.line,
                excitement=phrased.excitement,
                passed=verdict.passed,
                reason="; ".join(verdict.reasons)[:60],
                usd=usd,
                event=_event_of(phrased.line, form.event),
                form_event=form.event,
                fallback=fell_back,
                appended=settled.appended,
                stripped=settled.stripped,
                how_stripped=settled.how_removed,
            )
        )
        if not verdict.passed:
            continue
        phraser.accept(
            verdict.line,
            form.event,
            ts=ts,
            nameless=nameless_build_up(form, on_the_ball=carried),
        )
        cover.remember(form, verdict.line, ts)
        cover.note_event(form, ts)
        thread_rows(ts, threads.said(verdict.line, ts=ts, pack=pack), "used")
        ledger_rows(ts, counts_said(verdict.line, ts, names), "used")

        # -- the thirty seconds after a goal -----------------------------
        # A goal line that got through opens the window; a line inside it
        # spends one of its beats. Then, if the caller is about to leave a
        # gap longer than any gap in the corpus, the gap is filled.
        if not fell_back and claims_goal(verdict.line, form.event) and follow.is_the_call(ts):
            follow.arm(ts, form, verdict.line, pack)
            goal_calls.append(ts)
            # Whose goal it was is the one thing the board never knows, and
            # every running count about him is wrong from this second on.
            threads.credit_goal(follow.scorer, ts)
            ledger.credit_goal(follow.scorer, ts, form.side)
        elif follow.active(ts):
            follow.said(ts)
        if follow.active(ts):
            after = next((at for at in beat_times if at > ts + SAME_TS), None)
            for at in follow.synth_times(ts, after):
                await extra(at, _state_at(states, at) or state)
        # -- end of the goal window --------------------------------------

    # -- the score and the clock, said by code ---------------------------
    # Gap 8 item 4. A club feed restates both every few minutes for viewers
    # joining late, out of a closed vocabulary, and no model is needed to say
    # "ten minutes gone, two-nil to Barcelona". It runs over the rewritten
    # rows so that it can see where the lead actually speaks now.
    out.rows, restated = restatement_pass(
        out.rows, settings, goal_calls=goal_calls, window_s=follow.window_s
    )
    out.lines.extend(restated)
    # -- end of the restatement ------------------------------------------

    # -- the colour seat -------------------------------------------------
    # The second seat runs over the rewritten trace, not the original one:
    # what it observes off is what the lead actually says now. Everything
    # about it — when it is offered a turn, how the turn is spaced, how each
    # utterance is judged — is in ``commentary.agents.colour``, and this is
    # the whole of the hook.
    if colour:
        out.colour = await colour_pass(
            out.rows,
            backend,
            pack=pack,
            settings=settings,
            model=colour_model,
            goal_in_state=cover.goal_in_state,
            state_summary=_summary,
        )
        out.rows = out.colour.rows
        out.cost_usd += out.colour.cost_usd
    # -- end of the colour seat ------------------------------------------

    return out


#: How finely the restatement hunts for a clear moment once one is owed.
RESTATE_STEP_S = 0.5

#: How long it will keep hunting before giving the period up. Longer than any
#: run of lead lines in the corpus, short enough that "twenty minutes gone"
#: never goes out at twenty-two.
RESTATE_PATIENCE_S = 30.0

def restatement_pass(
    rows: list[dict[str, Any]],
    settings: Settings = SETTINGS,
    *,
    goal_calls: Sequence[float] = (),
    window_s: float = FOLLOWUP_S,
) -> tuple[list[dict[str, Any]], list[Line]]:
    """Put the score-and-clock line into the trace on its timer. No model.

    Section 5.3 of the corpus study: a club-channel feed restates the score
    every 150 to 260 seconds and the clock every 300 to 500, always as two
    short utterances, clock then score, out of a vocabulary of about ten
    phrasings. Gap 8 item 4 asks for it as code, and the handoff's standing
    rule — code writes numbers, the model writes words — settles the rest.

    Run over the rewritten rows rather than the original ones, because what it
    has to keep out of the way of is where the lead speaks *now*. ``goal_calls``
    is every moment a goal's follow-up window opened in this trace — not just
    the last one, because a match with more than one goal has moved on from
    the earlier ones by the time this runs — and a candidate is out of bounds
    for ``window_s`` after any of them plus the grace
    :func:`~commentary.goalfollow.blocks_restatement` adds on top.

    The new row is placed in trace order rather than after the state row that
    owed it: :func:`commentary.agents.colour._merge` puts it after the last
    original row stamped at or before it, which is what keeps a restatement
    the clear-moment search pushed forward from landing ahead of a beat that
    was already in the file at an earlier timestamp.

    Returns the rows with the restatements in them and a table row for each.
    """
    cfg = settings.restatement
    timer = Restatements(every_s=cfg.every_s)
    if not timer.enabled:
        return rows, []
    lead = sorted(
        float(row.get("ts", 0.0))
        for row in rows
        if row.get("topic") == Topic.BEAT.value and row.get("voice") == Voice.CALLER.value
    )

    def in_a_goal_window(at: float) -> bool:
        return any(blocks_restatement(at, armed_at, window_s) for armed_at in goal_calls)

    def clear(at: float) -> bool:
        if in_a_goal_window(at):
            return False
        return all(abs(at - beat) > cfg.clear_of_a_beat_s for beat in lead)

    said: list[tuple[float, str]] = []
    for row in rows:
        if row.get("topic") != Topic.STATE.value:
            continue
        payload = {k: v for k, v in row.items() if k not in ("topic", "ts")}
        try:
            state = MatchState.model_validate(payload)
        except Exception:
            continue
        timer.note(state)
        if not timer.pending:
            continue
        at = float(row.get("ts", 0.0))
        stop = at + RESTATE_PATIENCE_S
        while at <= stop and not clear(at):
            at += RESTATE_STEP_S
        if at > stop:
            continue
        text = timer.take(state)
        if text:
            said.append((at, text))
            lead.append(at)

    additions: list[tuple[float, dict[str, Any]]] = []
    lines: list[Line] = []
    for at, text in said:
        additions.append(
            (
                at,
                {
                    "topic": Topic.BEAT.value,
                    "ts": at,
                    "id": f"restate-{at:.1f}",
                    "voice": Voice.CALLER.value,
                    "text": text,
                    "video_ts": at,
                    "created_ts": 0.0,
                    "live_ts": at,
                    "event": Event.NONE.value,
                    "excitement": cfg.excitement,
                    "preemptable": True,
                    "by_code": True,
                },
            )
        )
        lines.append(
            Line(
                ts=at,
                original="(the state)",
                phrased=text,
                excitement=cfg.excitement,
                passed=True,
                reason="",
                written_by_code=True,
            )
        )
    return _merge(rows, additions), lines


def _event_of(text: str, event: Event) -> Event:
    """What the phrased line amounts to, by the director's own reading.

    The same call ``Runtime._call`` makes: a line that says the ball went in
    is a goal whatever the form filed it under, because that is what decides
    whether the director may drop it on a camera cut. Printed beside the
    form's event, it is how a rewrite that turned a penalty being waited on
    into a penalty being scored becomes visible in the table.
    """
    return Event.GOAL if claims_goal(text, event) else event


def _rewritten(row: dict[str, Any], text: str, excitement: float) -> dict[str, Any]:
    """The beat, with new words and nothing else touched.

    ``video_ts``, ``live_ts``, ``event`` and ``preemptable`` are kept exactly
    as the run produced them: the replay schedules on ``live_ts`` and the
    director cuts on ``event``, so a goal has to still be a goal here or the
    preemption this whole design is about stops happening.
    """
    beat = dict(row)
    beat["text"] = text
    beat["excitement"] = excitement
    return beat


def _summary(state: MatchState) -> str:
    """The state as a few lines of prompt, rebuilt from the row.

    Not the tracker's own ``summary``: that leans on the entity registry and
    the rolling "just now" list, neither of which a ``state`` row carries.
    What survives is the part the phraser is actually allowed to use, and it
    is told plainly not to say the score or the clock anyway.
    """
    lines = [state.scoreline, state.clock or "clock unseen"]
    if state.ball is not None:
        lines.append(f"on the ball: {state.ball.player}")
    if state.in_replay:
        lines.append("screen: replay, not live play")
    if state.named:
        lines.append("just now:")
        for event in state.named[-3:]:
            who = event.player or ""
            lines.append(f"  {event.event.value}{f' by {who}' if who else ''}")
    return "\n".join(lines)


def load(path: str | Path) -> list[dict[str, Any]]:
    """A trace off disk, ready for :func:`rephrase`."""
    return read_trace(Path(path))


def beats_of(rows: list[dict[str, Any]]) -> list[Beat]:
    """Every beat in a trace as a model, for tests and for counting."""
    found: list[Beat] = []
    for row in rows:
        if row.get("topic") != Topic.BEAT.value:
            continue
        payload = {k: v for k, v in row.items() if k not in ("topic", "ts")}
        try:
            found.append(Beat.model_validate(payload))
        except Exception:
            continue
    return found
