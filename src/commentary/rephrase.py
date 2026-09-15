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
    state took that goal in, and the runtime's own window runs from
    ``GOAL_GRAPHIC_LAG_S`` before it to ``GOAL_TALK_CAP_S`` after. Two small
    differences from the runtime, both in the permissive direction and both
    only reachable in the seconds around a second goal: the restart cannot be
    rebuilt (``_restart_ts`` is a caller line the runtime noticed and never
    published), so goal talk here is capped by the clock alone and never by
    the kickoff; and every goal's window is checked rather than only the
    latest one's.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from commentary.agents.phraser import Phraser
from commentary.bus import Topic
from commentary.config import SETTINGS, Settings
from commentary.gate import FactGate, claims_goal, fold
from commentary.llm.base import LLMBackend
from commentary.runtime import CARRY_NAME_S, GOAL_GRAPHIC_LAG_S, GOAL_TALK_CAP_S
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    GateVerdict,
    KnowledgePack,
    MatchState,
    PhrasedLine,
    Voice,
)
from commentary.state import notes_for
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

    @property
    def verdict(self) -> str:
        if self.fallback:
            return "fell back"
        return "passed" if self.passed else (self.reason or "rejected")

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
    """

    def __init__(self, states: list[tuple[float, MatchState]]) -> None:
        self._goals = _goal_taken_in(states)
        #: The last name a spoken line put on the ball, and when.
        self._held: tuple[str, float] | None = None

    def goal_in_state(self, ts: float) -> bool:
        return any(-GOAL_GRAPHIC_LAG_S <= ts - at <= GOAL_TALK_CAP_S for at in self._goals)

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


async def rephrase(
    rows: list[dict[str, Any]],
    backend: LLMBackend,
    *,
    pack: KnowledgePack | None = None,
    settings: Settings = SETTINGS,
    model: str | None = None,
) -> Rephrased:
    """Rewrite every spoken caller line in a trace, and judge the rewrites.

    Nothing here touches the clip, the board reader or the caller. The only
    model called is the phraser, once per line that was actually spoken.
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
        model=model,
        home=teams.home,
        away=teams.away,
    )
    gate = FactGate(settings.gate)
    cover = Cover(states)
    out = Rephrased()

    for row in rows:
        topic = row.get("topic", "")
        if topic in DROPPED:
            continue
        if topic != Topic.BEAT.value or row.get("voice") != Voice.CALLER.value:
            out.rows.append(row)
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
        fell_back = False
        carried = cover.carried(form, ts)
        names = [carried] if carried else []
        names += [s.name for s in form.sightings if s.name]
        names += [state.home, state.away]
        phrased = await phraser.phrase(
            form,
            _summary(state),
            on_the_ball=carried,
            notes=notes_for(pack, names),
        )
        usd = phraser.last_usage.cost_usd
        out.cost_usd += usd
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
        else:
            verdict = gate.judge(
                form.model_copy(update={"line": phrased.line}),
                state,
                pack,
                board_changed=cover.board_changed(form, _nearest(gates, ts)),
                goal_in_state=cover.goal_in_state(ts),
                carried=cover.carried(form, ts),
                at=ts,
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
            )
        )
        if verdict.passed:
            phraser.accept(verdict.line)
            cover.remember(form, verdict.line, ts)

    return out


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
