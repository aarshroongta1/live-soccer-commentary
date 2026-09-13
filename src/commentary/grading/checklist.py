"""The twelve-item definition of done, checked against one real trace.

The brief settles what "good commentary" means on a three-minute clip as a
list of twelve things, and until now the list was checked by reading a trace
by hand. That does not survive a second clip: a run on footage nobody has
tried before is only worth its cost if the same twelve questions are asked of
it the same way, and by eye they are not.

So each item is one function over the trace, the aligned truth and the pack,
and each answers with a verdict and one line of evidence — the line that was
spoken, the name that was wrong, the gap that was too long. The evidence
matters more than the verdict: an item that fails on a clip with no goal in
it is telling you about the clip, not the system.

Three of the twelve cannot be fully mechanised and say so in their evidence.
Item 5 asks whether a name was right, and a player standing still is in no
feed, so what is checked is corroboration within three seconds and the rest
is reported for a human. Item 8's "no wrong bind" is checked against the
pack's roster, which catches a number bound to the wrong squad but not a
correct number read off the wrong body. Item 10's "nothing false" is the
score and the clock, not the argument.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any

from commentary.grading import metrics
from commentary.schemas import Event, GroundTruthEvent, KnowledgePack, WireEvent
from commentary.trace import rows_of

#: Item 1: how close to the event a goal line has to land.
GOAL_NEAR_S = 6.0
#: Item 2: the window event recall is measured in, and the bar.
RECALL_WINDOW_S = 10.0
RECALL_TARGET = 0.70
#: Item 3: a claimed event with nothing in the feed this close is a phantom.
PHANTOM_WINDOW_S = 12.0
#: Item 5: how near the feed has to put a player for the name to be corroborated.
NAME_WINDOW_S = 3.0
#: Item 6.
NAME_RATE_TARGET = 0.60
#: Item 7.
OPEN_PLAY_PLAYERS = 3
#: Item 8.
BIND_RATE_TARGET = 0.50
#: Item 9.
SILENCE_CAP_S = 20.0
#: Item 10.
ANALYST_EVERY_S = 40.0
ANALYST_WORDS = 30
#: Item 12.
COST_CAP_USD = 1.20
PASSES_PER_S = 5.0
TRACK_LIFE_S = 2.5

#: Events a line can claim that the feed can be asked about (item 3).
CLAIMABLE: dict[str, Event] = {
    "goal": Event.GOAL,
    "card": Event.CARD,
    "penalty": Event.PENALTY,
    "foul": Event.FOUL,
}

#: How long after the feed's row a line may still be about it, per event.
#:
#: The feed stamps an instant and a broadcast spends much longer than an
#: instant on it, so a flat window scores the coverage as a phantom. A goal
#: gets the celebration and the replays (the same number the gate uses). A
#: penalty is the longest of the lot: the award and the kick were ninety-one
#: seconds apart on the clip this came off, and every line between them —
#: the wall clearing, the keeper alone, the ball on the spot — is about a
#: penalty that is really happening. A card is the walk over and the replay,
#: a foul is however long the free kick takes to set.
TAIL_S: dict[Event, float] = {
    Event.GOAL: metrics.GOAL_TALK_S,
    Event.PENALTY: 150.0,
    Event.CARD: 45.0,
    Event.FOUL: 20.0,
}

#: A scoreline in words, for item 11. The digits are caught by the gate's own
#: regex; a caller that has been told not to say "2-0" says "two-nil" next.
SPELLED_SCORE = re.compile(
    r"\b(nil|none|one|two|three|four|five)[\s-](nil|none|one|two|three|four|five)\b"
)


#: What a line says when it is calling this event, beyond the caller's own
#: ``event`` tag. Recall measures whether a line landed near an event, which
#: is what the brief asked for and is not the same question: the offside clip
#: scored offside 1/1 on a line about a corner that happened to be nearby,
#: and nothing in three minutes ever said the word.
SAYS: dict[Event, tuple[str, ...]] = {
    # "the flag goes up against the runner" is how the Croatia clip called
    # one, and none of the first four phrasings caught it. A flag going up in
    # football is an offside; there is nothing else it can be.
    Event.OFFSIDE: (
        "offside", "flag", "linesman", "assistant", "called back", "pulled back",
    ),
    Event.CARD: ("yellow", "red card", "booked", "booking", "card", "sent off"),
    Event.PENALTY: ("penalty", "the spot", "twelve yards"),
    Event.SUBSTITUTION: (
        "substitut", "comes on", "coming on", "off for", "replaced", "jogs on",
        "board goes up", "fourth official", "afternoon is over", "makes a change",
    ),
    Event.CORNER: ("corner",),
    Event.FREE_KICK: ("free kick", "free-kick"),
    Event.THROW_IN: ("throw",),
    Event.SAVE: ("save", "saved", "gets across it", "beats it away", "palms"),
    Event.FOUL: ("foul", "free kick", "brings him down", "pulls him back", "challenge"),
    Event.SHOT: ("shot", "effort", "strikes it", "drives it", "fires"),
    Event.CLEARANCE: ("clear", "hacked away", "heads it away"),
    Event.TACKLE: ("tackle", "wins it back", "dispossess"),
    Event.KICKOFF: ("kick off", "kickoff", "restart", "under way", "underway"),
    Event.INTERCEPTION: ("intercept", "cuts it out", "reads it"),
}


@dataclass(frozen=True)
class Call:
    """One thing that happened, and what the run did about it."""

    event: Event
    video_ts: float
    player: str | None
    said: bool
    named: bool
    lag_s: float | None
    line: str

    def row(self) -> str:
        if not self.said:
            verdict = "NOT CALLED"
        elif self.player and not self.named:
            verdict = f"called, {self.player.rsplit(' ', 1)[-1]} not named"
        else:
            verdict = "called"
        when = f"{self.lag_s:+.1f}s" if self.lag_s is not None else "   -  "
        return (
            f"  {self.event.value:<13} {self.video_ts:6.1f}s  {when:>7}  "
            f"{verdict:<28} {self.line[:60]}"
        )


def calls(state: Watched, *, window_s: float = RECALL_WINDOW_S) -> list[Call]:
    """Was each event actually called, by name, and how late?

    Distinct from recall, which asks only whether a line landed nearby. A
    line about a corner four seconds before an offside counts for recall and
    tells the listener nothing about the offside.
    """
    out: list[Call] = []
    for event in state.truth:
        near = [
            line for line in state.lines if abs(line.ts - event.video_ts) <= window_s
        ]
        spoke = [line for line in near if _calls_it(line, event.event)]
        surname = _surname(event.player)
        named = [line for line in spoke if surname and surname in _words(line.text)]
        found = named or spoke
        best = found[0] if found else None
        out.append(
            Call(
                event=event.event,
                video_ts=event.video_ts,
                player=event.player,
                said=bool(spoke),
                named=bool(named),
                lag_s=best.ts - event.video_ts if best else None,
                line=best.text if best else "",
            )
        )
    return out


def _calls_it(line: Line, event: Event) -> bool:
    """Does this line call that event — by its tag or in as many words?"""
    if line.event == event.value:
        return True
    if event is Event.GOAL:
        return _claims_goal(line)
    lowered = metrics.normalise(line.text)
    return any(word in lowered for word in SAYS.get(event, ()))


@dataclass(frozen=True)
class Item:
    """One of the twelve, decided, with the evidence that decided it."""

    number: int
    title: str
    ok: bool
    evidence: str

    def row(self) -> str:
        return f"{'PASS' if self.ok else 'FAIL'}  {self.number:>2}. {self.title} — {self.evidence}"


@dataclass(frozen=True)
class Line:
    """A spoken line with what the caller was looking at when it wrote it."""

    ts: float
    voice: str
    text: str
    event: str
    seconds: float = 0.0
    scene: str = ""
    trimmed: bool = False

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class Sighting:
    """One reported read of a shirt, as the runtime resolved it."""

    ts: float
    mark: str | None
    number: int | None
    name: str | None
    bound: bool
    live: bool
    side: str = ""


@dataclass
class Watched:
    """Everything the twelve items are decided from, read out of one trace."""

    rows: list[dict[str, Any]]
    run: metrics.Run
    truth: list[GroundTruthEvent]
    wire: list[WireEvent]
    pack: KnowledgePack
    watched_s: float
    lines: list[Line] = field(default_factory=list)
    sightings: list[Sighting] = field(default_factory=list)


def watched(
    rows: list[dict[str, Any]],
    run: metrics.Run,
    truth: list[GroundTruthEvent],
    wire: list[WireEvent],
    pack: KnowledgePack,
    *,
    watched_s: float,
) -> Watched:
    """Reconstruct the spoken lines and the sightings from the trace rows."""
    scenes = {
        round(float(r.get("ts", 0.0)), 3): str(r.get("scene", ""))
        for r in rows_of(rows, "caller")
    }
    trims = {
        round(float(r.get("ts", 0.0)), 3): any(
            str(reason).startswith("trimmed_name") for reason in r.get("reasons", [])
        )
        for r in rows_of(rows, "gate")
    }
    lines = [
        Line(
            ts=float(row.get("ts", 0.0)),
            voice=str(row.get("voice", "caller")),
            text=str(row.get("spoken", "")),
            event=str(row.get("event", "none")),
            seconds=float(row.get("seconds", 0.0)),
            scene=scenes.get(round(float(row.get("ts", 0.0)), 3), ""),
            trimmed=trims.get(round(float(row.get("ts", 0.0)), 3), False),
        )
        for row in rows_of(rows, "spoken")
    ]
    sightings = [
        Sighting(
            ts=float(row.get("ts", 0.0)),
            mark=_text_or_none(entry.get("mark")),
            number=entry.get("number") if isinstance(entry.get("number"), int) else None,
            name=_text_or_none(entry.get("name")),
            bound=bool(entry.get("bound")),
            live=bool(entry.get("live")),
            side=str(entry.get("side") or ""),
        )
        for row in rows_of(rows, "sighting")
        for entry in row.get("sightings", [])
    ]
    return Watched(
        rows=rows,
        run=run,
        truth=truth,
        wire=wire,
        pack=pack,
        watched_s=watched_s,
        lines=lines,
        sightings=sightings,
    )


def check(state: Watched) -> list[Item]:
    """All twelve, in the brief's order."""
    return [
        _goal_line(state),
        _event_recall(state),
        _phantoms(state),
        _unconfirmed_goal(state),
        _name_precision(state),
        _name_rate(state),
        _open_play_names(state),
        _sightings(state),
        _silence(state),
        _analyst(state),
        _scoreline(state),
        _health(state),
    ]


def report(items: list[Item]) -> str:
    passed = sum(1 for item in items if item.ok)
    return "\n".join(
        [f"Definition of done: {passed} of {len(items)}", "", *(item.row() for item in items)]
    )


# -- events --------------------------------------------------------------


def _goal_line(state: Watched) -> Item:
    title = "the goal is spoken as a goal, named, in 6 s, intact"
    goals = [e for e in state.truth if e.event is Event.GOAL]
    if not goals:
        return Item(1, title, True, "no goal in the watched window")

    verdicts: list[str] = []
    ok = True
    for goal in goals:
        near = [
            line
            for line in state.lines
            if abs(line.ts - goal.video_ts) <= GOAL_NEAR_S and _claims_goal(line)
        ]
        surname = _surname(goal.player)
        named = [line for line in near if surname and surname in _words(line.text)]
        if not near:
            ok = False
            verdicts.append(
                f"{goal.player or 'goal'} at {goal.video_ts:.0f}s: no goal line within 6 s"
            )
        elif not named:
            ok = False
            verdicts.append(
                f"{goal.player or 'goal'} at {goal.video_ts:.0f}s: called, unnamed — "
                f'"{near[0].text}"'
            )
        elif named[0].trimmed:
            ok = False
            verdicts.append(
                f'{goal.player} at {goal.video_ts:.0f}s: named but trimmed — "{named[0].text}"'
            )
        else:
            verdicts.append(
                f'{goal.player} at {goal.video_ts:.0f}s: {named[0].ts - goal.video_ts:+.1f}s '
                f'"{named[0].text}"'
            )
    return Item(1, title, ok, "; ".join(verdicts))


def _event_recall(state: Watched) -> Item:
    title = f"non-goal event recall >= {RECALL_TARGET:.0%} within {RECALL_WINDOW_S:.0f} s"
    rest = [e for e in state.truth if e.event is not Event.GOAL]
    if not rest:
        return Item(2, title, True, "no non-goal events in the watched window")
    recall = metrics.event_recall(state.run, rest, window_s=RECALL_WINDOW_S)
    breakdown = ", ".join(
        f"{kind} {got}/{need}" for kind, (got, need) in sorted(recall.by_event.items())
    )
    return Item(
        2,
        title,
        recall.rate >= RECALL_TARGET,
        f"{recall.matched}/{recall.total} = {recall.rate:.0%} ({breakdown})",
    )


def _phantoms(state: Watched) -> Item:
    title = "zero phantom goals, cards, penalties or fouls"
    claimed = 0
    phantoms: list[str] = []
    for line in state.lines:
        for kind in _claims(line):
            claimed += 1
            if not _feed_has(state.wire, kind, line.ts):
                phantoms.append(f'{line.ts:.0f}s {kind.value}: "{line.text}"')
    if phantoms:
        return Item(3, title, False, f"{len(phantoms)} of {claimed} claims: " + "; ".join(phantoms))
    return Item(3, title, True, f"{claimed} event claims, all in the feed within 12 s")


def _unconfirmed_goal(state: Watched) -> Item:
    title = "zero unconfirmed_goal rejections of a line about a real goal"
    # A rejected line is written empty on the gate row — nothing reached a
    # voice — so the line it killed has to come off the caller row beside it.
    proposed = {
        round(float(row.get("ts", 0.0)), 3): str(row.get("line", ""))
        for row in rows_of(state.rows, "caller")
    }
    hits = [
        (
            float(row.get("ts", 0.0)),
            str(row.get("line", "")) or proposed.get(round(float(row.get("ts", 0.0)), 3), ""),
        )
        for row in rows_of(state.rows, "gate")
        if any(str(reason).startswith("unconfirmed_goal") for reason in row.get("reasons", []))
    ]
    if not hits:
        return Item(4, title, True, "none")
    # A rejection of a goal nobody scored is the gate doing its job, and the
    # offside clip produced one: the caller wrote a goal at 31:15 with the
    # score 2-0 and unchanged. Counting that as a failure scores the system
    # for the one thing it is built to do.
    goals = [e.video_ts for e in state.truth if e.event is Event.GOAL]
    real = [
        (ts, line)
        for ts, line in hits
        if any(-PHANTOM_WINDOW_S <= ts - g <= metrics.GOAL_TALK_S for g in goals)
    ]
    invented = len(hits) - len(real)
    if not real:
        return Item(
            4, title, True, f"none; {invented} rejected a goal that never happened, correctly"
        )
    return Item(
        4,
        title,
        False,
        f"{len(real)}: " + "; ".join(f'{ts:.0f}s "{line}"' for ts, line in real[:3]),
    )


# -- names ---------------------------------------------------------------


def _name_precision(state: Watched) -> Item:
    title = "name_precision = 100%"
    scored = metrics.names(state.run, state.wire, window_s=NAME_WINDOW_S)
    invented = [
        error.detail
        for error in metrics.factual_errors(state.run, state.truth, state.pack)
        if error.kind == "name_off_roster"
    ]
    if scored.names == 0:
        return Item(
            5, title, not invented, f"no squad name in any line; off-roster words {invented}"
        )
    uncorroborated = _uncorroborated(state)
    detail = (
        f"{scored.correct}/{scored.names} names have the feed within {NAME_WINDOW_S:.0f} s"
        f" ({scored.precision:.0%})"
    )
    if uncorroborated:
        detail += "; no feed nearby (check the frame by eye): " + ", ".join(uncorroborated[:8])
    if invented:
        detail += f"; off the roster {invented}"
    # The brief's own wording: a name is right if the feed has the player on
    # the ball *or* they are visibly the subject of a close-up, and the second
    # half is a human's call. So the machine fails this item only on a name
    # nobody in the match is called; the uncorroborated ones are listed for
    # the eye that has to check them.
    return Item(5, title, not invented, detail)


def _name_rate(state: Watched) -> Item:
    title = f"name_rate >= {NAME_RATE_TARGET:.0%} of caller lines in live play"
    live = [line for line in state.lines if line.voice == "caller" and line.scene == "live_play"]
    if not live:
        return Item(6, title, False, "no caller line in live play")
    named = [line for line in live if metrics.named_in(line.text, state.wire)]
    rate = len(named) / len(live)
    overall = metrics.names(state.run, state.wire).rate
    return Item(
        6,
        title,
        rate >= NAME_RATE_TARGET,
        f"{len(named)}/{len(live)} = {rate:.0%} in live play; {overall:.0%} of all spoken lines",
    )


def _open_play_names(state: Watched) -> Item:
    title = f"{OPEN_PLAY_PLAYERS}+ distinct players in open play, one carried on a mark"
    distinct: set[str] = set()
    for line in state.lines:
        if line.scene == "live_play":
            distinct.update(metrics.named_in(line.text, state.wire))
    carried = _carried_on_a_mark(state)
    ok = len(distinct) >= OPEN_PLAY_PLAYERS and carried is not None
    detail = f"{len(distinct)} in open play ({', '.join(sorted(distinct)) or 'none'})"
    detail += f"; carried on a mark: {carried}" if carried else "; nothing carried on a mark"
    return Item(7, title, ok, detail)


def _sightings(state: Watched) -> Item:
    title = f"sightings bound / made >= {BIND_RATE_TARGET:.0%}, none bound wrong"
    made = len(state.sightings)
    if not made:
        return Item(8, title, False, "no sighting reported")
    bound = [s for s in state.sightings if s.bound]
    wrong = [f"{s.number} {s.name} at {s.ts:.0f}s" for s in bound if _contradicts_roster(state, s)]
    rate = len(bound) / made
    detail = (
        f"{len(bound)}/{made} = {rate:.0%} bound, "
        f"{sum(1 for s in bound if s.mark)} on a mark"
    )
    if wrong:
        detail += "; contradicts the roster: " + ", ".join(wrong)
    return Item(8, title, rate >= BIND_RATE_TARGET and not wrong, detail)


# -- voice ---------------------------------------------------------------


def _silence(state: Watched) -> Item:
    title = f"no silence over {SILENCE_CAP_S:.0f} s while the scene is live play"
    edges = [0.0, *(line.ts + line.seconds for line in state.lines)]
    starts = [*(line.ts for line in state.lines), state.watched_s]
    worst = 0.0
    worst_at = 0.0
    bad: list[str] = []
    for end, start in zip(edges, starts, strict=False):
        gap = start - end
        if gap <= 0:
            continue
        live = [
            row
            for row in rows_of(state.rows, "caller")
            if end <= float(row.get("ts", 0.0)) <= start and row.get("scene") == "live_play"
        ]
        if gap > worst and live:
            worst, worst_at = gap, end
        if gap > SILENCE_CAP_S and live:
            bad.append(f"{gap:.0f}s from {end:.0f}s")
    if bad:
        return Item(9, title, False, f"{len(bad)} live-play gaps: " + ", ".join(bad[:4]))
    return Item(9, title, True, f"longest live-play gap {worst:.0f}s at {worst_at:.0f}s")


def _analyst(state: Watched) -> Item:
    title = f"analyst: <= 1 per {ANALYST_EVERY_S:.0f} s, <= {ANALYST_WORDS} words, nothing false"
    lines = [line for line in state.lines if line.voice == "analyst"]
    if not lines:
        return Item(10, title, True, "the analyst said nothing")
    gaps = [b.ts - a.ts for a, b in zip(lines, lines[1:], strict=False)]
    too_close = [g for g in gaps if g < ANALYST_EVERY_S]
    too_long = [line for line in lines if line.words > ANALYST_WORDS]
    texts = {line.text for line in lines}
    false = [
        f"{e.kind} {e.detail}"
        for e in metrics.factual_errors(state.run, state.truth, state.pack)
        if e.line in texts
    ]
    ok = not too_close and not too_long and not false
    detail = f"{len(lines)} lines, max {max(line.words for line in lines)} words"
    if too_close:
        detail += f"; {len(too_close)} closer than {ANALYST_EVERY_S:.0f}s (min {min(gaps):.0f}s)"
    if too_long:
        detail += f"; over-long: \"{too_long[0].text}\""
    if false:
        detail += "; false: " + ", ".join(false)
    return Item(10, title, ok, detail)


def _scoreline(state: Watched) -> Item:
    title = "no line states the scoreline"
    said = [
        f'{line.ts:.0f}s "{line.text}"'
        for line in state.lines
        if metrics.SCORELINE.search(line.text) or SPELLED_SCORE.search(line.text.lower())
    ]
    repetition = metrics.repetition_rate(state.run)
    if said:
        return Item(11, title, False, "; ".join(said[:3]))
    return Item(11, title, True, f"none in {len(state.lines)} lines; repetition {repetition:.0%}")


def _health(state: Watched) -> Item:
    title = (
        f"<= ${COST_CAP_USD:.2f}, zero errors, tracker >= {PASSES_PER_S:.0f}/s, "
        f"median track life >= {TRACK_LIFE_S:.1f}s"
    )
    errors = rows_of(state.rows, "error")
    rate, life = _tracker(state.rows)
    ok = (
        state.run.cost_usd <= COST_CAP_USD
        and not errors
        and rate >= PASSES_PER_S
        and life >= TRACK_LIFE_S
    )
    detail = (
        f"${state.run.cost_usd:.2f}, {len(errors)} errors, {rate:.1f} passes/s, "
        f"median track life {life:.2f}s"
    )
    if errors:
        detail += f"; first error {errors[0].get('where')}: {str(errors[0].get('detail'))[:60]}"
    return Item(12, title, ok, detail)


# -- the small print -----------------------------------------------------


def _tracker(rows: list[dict[str, Any]]) -> tuple[float, float]:
    """Passes a second, and the median life of a track id, from the trace."""
    passes = rows_of(rows, "tracks")
    if len(passes) < 2:
        return 0.0, 0.0
    span = float(passes[-1].get("ts", 0.0)) - float(passes[0].get("ts", 0.0))
    seen: dict[int, list[float]] = {}
    for row in passes:
        for track_id in row.get("ids", []):
            seen.setdefault(int(track_id), []).append(float(row.get("ts", 0.0)))
    lives = [stamps[-1] - stamps[0] for stamps in seen.values()]
    return (len(passes) / span if span > 0 else 0.0, statistics.median(lives) if lives else 0.0)


def _claims_goal(line: Line) -> bool:
    lowered = metrics.normalise(line.text)
    if line.event == Event.GOAL.value:
        return True
    return any(phrase in lowered for phrase in metrics.GOAL_CLAIMS)


def _claims(line: Line) -> set[Event]:
    """Which of the four checkable events this line asserts happened."""
    claimed = {CLAIMABLE[line.event]} if line.event in CLAIMABLE else set()
    if _claims_goal(line):
        claimed.add(Event.GOAL)
    return claimed


def _feed_has(wire: list[WireEvent], kind: Event, ts: float) -> bool:
    """Does the feed have this kind of event near this moment?

    A penalty is two rows in StatsBomb — the award and the kick — and the kick
    arrives as a goal or a shot, so either answers a penalty claim.

    Behind the event each kind gets the tail its coverage really runs to
    (:data:`TAIL_S`). Ahead of it the window stays short for all of them,
    because ahead is the direction inventing an event looks like.
    """
    kinds = {kind}
    if kind is Event.PENALTY:
        kinds |= {Event.GOAL, Event.SHOT}
    behind = TAIL_S.get(kind, PHANTOM_WINDOW_S)
    return any(
        event.event in kinds
        and event.video_ts is not None
        and -PHANTOM_WINDOW_S <= ts - event.video_ts <= behind
        and (kind is not Event.PENALTY or event.event is Event.PENALTY or event.detail == "penalty")
        for event in wire
    )


def _uncorroborated(state: Watched) -> list[str]:
    """Names said with nobody of that name in the feed nearby, for a human."""
    out: list[str] = []
    for line in state.lines:
        near = {
            name
            for event in state.wire
            if event.video_ts is not None and abs(event.video_ts - line.ts) <= NAME_WINDOW_S
            for name in (event.player, event.recipient)
            if name
        }
        out.extend(
            f"{name} at {line.ts:.0f}s"
            for name in metrics.named_in(line.text, state.wire)
            if name not in near
        )
    return out


def _carried_on_a_mark(state: Watched) -> str | None:
    """A sighting bound on a live mark whose name a later line then said."""
    for sighting in state.sightings:
        if not (sighting.bound and sighting.mark):
            continue
        name = sighting.name or _roster_name(state.pack, sighting.number)
        if not name:
            continue
        surname = _surname(name)
        for line in state.lines:
            if line.ts >= sighting.ts and surname and surname in _words(line.text):
                return (
                    f"{name} (mark {sighting.mark} at {sighting.ts:.0f}s, "
                    f"said at {line.ts:.0f}s)"
                )
    return None


def _contradicts_roster(state: Watched, sighting: Sighting) -> bool:
    """A bound sighting the team sheets say cannot be right.

    Both squads wear a 7 and an 11, so a shared number on its own is not a
    contradiction. When the caller said which kit it read the number off, the
    check narrows to that squad, which is the whole point of asking. What is
    checkable otherwise is a number nobody wears, a name nobody is called,
    and a number and a name that belong to two different people.
    """
    sheets = {"home": [state.pack.home], "away": [state.pack.away]}.get(
        sighting.side, [state.pack.home, state.pack.away]
    )
    squad = [player for sheet in sheets for player in sheet.squad]
    if sighting.name:
        # Any word of the name, not the last one: a caller reading KOLO MUANI
        # off a shirt may write either half, and "Di María" arrives as often
        # as "María" does. Matching the last word alone called a correct read
        # of the 12 a contradiction.
        said = _words(sighting.name)
        wearing = [p for p in squad if said & _words(p.name)]
        if not wearing:
            return True
        if sighting.number is not None:
            return all(p.number != sighting.number for p in wearing)
        return False
    if sighting.number is not None:
        return all(p.number != sighting.number for p in squad)
    return False


def _roster_name(pack: KnowledgePack, number: int | None) -> str | None:
    """The one player wearing this number, or None if it is nobody's or shared."""
    if number is None:
        return None
    found = [
        player.name
        for sheet in (pack.home, pack.away)
        for player in sheet.squad
        if player.number == number
    ]
    return found[0] if len(found) == 1 else None


def _surname(name: str | None) -> str:
    return metrics.normalise(name.rsplit(" ", 1)[-1]) if name else ""


def _words(text: str) -> set[str]:
    return {metrics.normalise(word) for word in re.findall(r"[\w'-]+", text)}


def _text_or_none(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None
