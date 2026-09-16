"""The colour seat: offered a turn by a phase gate, not by a silence timer.

``docs/HANDOFF.md`` section 3e asks for two seats. The lead calls the action;
the colour seat "is event-driven: it reacts ... after a big moment, and in
slow build-up it observes off the lead's recent forms and the pack notes. It
never speaks over an action call." What was in the tree was
:class:`~commentary.agents.analyst.Analyst`, which fires when nobody has said
anything for seven seconds, sees six frames, and writes one thirty-word
sentence. On the Mbappé trace it spoke three times in 202 seconds, once eight
seconds after the goal call, and once truncated mid-clause at the word cap.

Four measurements out of ``docs/research/real-commentary-corpus.md`` are what
this module is, and each one is a rule rather than a request in the prompt.

**Phase, not silence** (section 4.2). Colour entries per 100 utterances: 2.1
in an attacking move, 5.9 in build-up, 10.5 at a dead ball, 11.8 at a
stoppage, 15.4 over a replay. The seat is five times less likely to speak
while the ball is live than while it is dead, so :func:`may_speak` reads the
last two caller forms and asks what the picture is, not how long the channel
has been quiet.

**The first twelve seconds belong to the lead** (section 4.3). Pooled over
174 big events the median delay to the first colour entry is 21.4 s and only
7% land inside six seconds. Those seconds are the lead rebuilding the move
and giving the tally (section 8.4, slots 1 to 6; the colour voice is slot 7,
"4 to 26 seconds in"). The one exception is the goal, where 31% of colour
entries do arrive inside six seconds and every one of them is a reaction
fragment: "WELL, it's the first goal of the game." / "Well, well, well."

**A turn, not a line** (section 4.4). Median 4 utterances, mean 4.4, 51% run
four or more, 3 to 12 words each, said in sequence. A 30-word cap that
truncates mid-clause is a worse instrument than a 12-word cap that lets the
voice come back in two seconds.

**The opener is the speaker label** (section 4.1). "Well", "Yeah", "I think"
and the rest open colour turns at three times the base rate on every file
with speaker markers. It is how a listener knows the voice has changed before
the timbre tells them, and it is measured here as a target above 60%.

What this module does not have is the lead's name. The corpus's two voices
address each other by name on every file that has two identified voices
(section 4.1) and it is the most copiable thing in the study — but this
system has one caller with no name, and inventing one is inventing a person.

The seat is text-only. It is never shown a frame, which is the difference
between telling a vision model not to narrate the picture and not giving it
one.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from commentary.agents.analyst import restates_score
from commentary.agents.caller import clean_line, trim_words
from commentary.bus import Topic
from commentary.config import SETTINGS, ColourConfig, Settings
from commentary.gate import FactGate, fold, is_the_same_name, noun_for
from commentary.ledger import Fact, Ledger
from commentary.llm.base import Block, LLMBackend, LLMError, Parsed, Usage, text_block
from commentary.prompts.colour import (
    OPENERS,
    colour_blocks,
    colour_system,
    form_line,
    opens_with_a_cue,
)
from commentary.schemas import (
    CallerLine,
    ColourTurn,
    Event,
    GateVerdict,
    KnowledgePack,
    MatchState,
    Note,
    Scene,
    Sighting,
    Voice,
)
from commentary.state import notes_for
from commentary.tallies import Tallies
from commentary.threads import CALLBACK_QUIET_S, Threads
from commentary.voice.speaker import WORDS_PER_SECOND

#: The value of ``COLOUR_MODEL`` that means "no colour seat".
OFF = "off"

#: The scenes that mean the ball is not being played. Section 4.2: the seat's
#: rate is 10.5 per 100 utterances at a dead ball, 11.8 at a stoppage and
#: 15.4 over a replay, against 2.1 in an attacking move. ``GRAPHIC`` is not
#: here because a full-screen graphic is usually a scoreboard or a lineup and
#: the caller files it with whatever event was last true.
QUIET_SCENES = frozenset({Scene.STOPPAGE, Scene.REPLAY, Scene.CLOSE_UP, Scene.CROWD})

#: The events that mean the ball is dead. The study's list is corner, free
#: kick, throw-in, goal kick and kickoff; :class:`~commentary.schemas.Event`
#: has no ``goal_kick`` member — the caller has never been offered one — so a
#: goal kick reaches here as ``build_up`` or ``none`` and is caught, if at
#: all, by the twenty-second clause instead.
DEAD_BALL_EVENTS = frozenset(
    {
        Event.CORNER,
        Event.FREE_KICK,
        Event.THROW_IN,
        Event.KICKOFF,
        Event.STOPPAGE,
        Event.OFFSIDE,
        Event.FOUL,
        Event.SUBSTITUTION,
        #: A card is a stoppage — section 4.2 counts "card, injury, sub"
        #: together at 11.8 colour entries per 100 utterances — even though
        #: it is a big event for the twenty-second clause.
        Event.CARD,
    }
)

#: The other things a second voice is entitled to an opinion about. A foul, a
#: tackle, an offside: nobody has scored, so none of them is a *big* moment,
#: and every one of them is what section 3.2's booking window is made of —
#: "It's a really poor challenge from Casemiro", "Well, it could be a yellow
#: card for the Frenchman", "every time you look at it, it looks less and
#: less like there was enough contact". The measurement that put them here is
#: on the Mbappé trace: the penalty was conceded at 12.9 s, the caller filed
#: four replay forms of the contact between 21.5 s and 37.8 s, and the seat
#: was never offered a turn over any of them because a foul was not an event
#: this file had heard of. It first spoke at 135.8 s.
INCIDENTS = frozenset({Event.FOUL, Event.OFFSIDE, Event.TACKLE})

#: The events whose first twelve seconds belong to the lead. The four the
#: brief names — a goal, a shot, a save, a penalty — and the incidents, whose
#: numbers in section 4.3 are the strongest case in the table for holding:
#: after a foul the median delay to the first colour entry is 16.7 s and
#: **5%** land inside six seconds, the lowest share of any event kind. A card
#: is the one thing not held: its median is 17.0 s but 29% do land inside six
#: seconds, and a card is a stoppage, which is where this seat is meant to
#: speak.
HELD_FOR_THE_LEAD = frozenset({Event.GOAL, Event.SHOT, Event.SAVE, Event.PENALTY}) | INCIDENTS

#: The big moments: the four the lead is given his window for, plus the card,
#: which is the study's "foul-with-card". What the situation split and the
#: goal reaction read, and unchanged by any of this.
BIG_MOMENTS = frozenset({Event.GOAL, Event.SHOT, Event.SAVE, Event.PENALTY, Event.CARD})

#: Everything the seat may judge, which is what :meth:`ColourSeat.material`
#: reads for its EVENT line and what the rate cap counts as an event worth
#: one turn.
JUDGED = BIG_MOMENTS | INCIDENTS

#: The example groups in :mod:`commentary.prompts.colour`, which are also the
#: situations the gate can hand back.
AFTER_A_GOAL = "after a goal"
AFTER_A_CHANCE = "after a chance"
#: An incident and the pictures of it again. Section 4.2 has the colour voice
#: at 15.4 entries per 100 utterances over a replay, its highest rate of any
#: phase and seven times its rate in an attacking move, and section 4.4 shows
#: what it does with them: it gives a verdict on the thing that has just
#: happened, off the evidence the replay is showing.
OVER_A_REPLAY = "over the replay"
AT_A_DEAD_BALL = "at a dead ball"
IN_BUILD_UP = "in quiet build-up"

#: How long a big event stays the thing the turn is about, for the purpose of
#: choosing which examples to lean on. Past this the situation is ordinary
#: build-up whatever happened a minute ago.
SITUATION_WINDOW_S = 60.0

#: How hard each situation is said. The colour seat never shouts — see
#: ``VoiceConfig.analyst``, whose whole point is that an analyst at a
#: caller's pitch is two voices becoming one — so even a goal reaction sits
#: well below the caller's top end.
EXCITEMENT = {
    AFTER_A_GOAL: 0.45,
    AFTER_A_CHANCE: 0.3,
    OVER_A_REPLAY: 0.25,
    AT_A_DEAD_BALL: 0.2,
    IN_BUILD_UP: 0.15,
}


def colour_enabled(model: str) -> bool:
    """Is there a colour seat at all?"""
    return model.strip().lower() not in (OFF, "", "none")


@dataclass(frozen=True)
class FormAt:
    """One caller form, flattened to what the seat is allowed to know.

    No frame, no confidence, no sighting object: the scene and the event are
    what the phase gate reads, and the team, the names and the words are what
    the prompt shows. Everything else on a :class:`CallerLine` is about
    looking, and this seat does not look.
    """

    ts: float
    scene: Scene
    event: Event
    team: str = ""
    names: tuple[str, ...] = ()
    said: str = ""

    @classmethod
    def of(cls, ts: float, line: CallerLine) -> FormAt:
        return cls(
            ts=ts,
            scene=line.scene,
            event=line.event,
            team=line.team or "",
            names=tuple(s.name for s in line.sightings if s.name),
            said=line.line or "",
        )

    def as_line(self) -> CallerLine:
        """Back to a caller form, for anything that reads one.

        Lossy, and only in the direction that does not matter here: the side
        and the confidence were dropped on the way in and do not come back.
        It exists so a bare list of forms can be handed to a throwaway
        :class:`~commentary.ledger.Ledger` — a test, or a seat with nothing
        shared — without a second ingest path existing anywhere.
        """
        return CallerLine(
            scene=self.scene,
            event=self.event,
            team=self.team or None,
            sightings=[Sighting(name=name) for name in self.names],
            confidence=1.0,
            speak=bool(self.said),
            line=self.said,
        )

    @property
    def quiet(self) -> bool:
        """Is the ball dead, or is the picture something other than the play?"""
        return self.scene in QUIET_SCENES or self.event in DEAD_BALL_EVENTS

    def rendered(self) -> str:
        return form_line(self.scene.value, self.event.value, self.team, self.names, self.said)


@dataclass
class Share:
    """The running lead:colour split, and how far behind the seat is.

    Club football gives the colour voice about 31% of the utterances
    (``ColourConfig.colour_share_target``, which carries the three
    measurements). Nothing in this system aimed at that number and the
    pooled night-one set came out at 80 colour lines against 505, which is
    14%. The phase gate is why: it is a rate limit and a permission, and a
    permission that is never exercised produces silence rather than a ratio.

    So the share is measured over a sliding window of cursor time and read
    back as a *shortfall*, and three things loosen in proportion to it: the
    build-up rate the seat is offered at, the lead's build-up cap, and how
    many utterances a turn may run to. Nothing here is a permission on its
    own — the material gate in :meth:`ColourSeat.offer` is still the hard
    one, and a seat with nothing to say stays quiet at any share.

    Both seats are counted in utterances, which is the unit the corpus
    measures: a lead line and a colour line are one each, whatever their
    length.
    """

    window_s: float = ColourConfig.share_window_s
    target: float = ColourConfig.colour_share_target
    min_sample: int = ColourConfig.share_min_sample
    lead: list[float] = field(default_factory=list)
    colour: list[float] = field(default_factory=list)

    def said(self, ts: float, *, colour: bool) -> None:
        """Record one utterance that actually went out, by seat."""
        (self.colour if colour else self.lead).append(ts)

    def counts(self, now: float) -> tuple[int, int]:
        """Lead and colour utterances inside the window ending at ``now``."""
        floor = now - self.window_s
        return (
            sum(1 for ts in self.lead if ts > floor),
            sum(1 for ts in self.colour if ts > floor),
        )

    def share(self, now: float) -> float | None:
        """The colour seat's share of the window, or ``None`` if too thin.

        ``None`` rather than zero, and everything downstream treats it as
        "not measured": a window with two lines in it is not evidence that
        the second voice is short, and the governor must not stretch the
        lead's cadence off the first line of a match.
        """
        lead, colour = self.counts(now)
        if lead + colour < self.min_sample:
            return None
        return colour / (lead + colour)

    def shortfall(self, now: float) -> float:
        """How far below target the share is, 0 to ``target``."""
        share = self.share(now)
        if share is None:
            return 0.0
        return max(0.0, self.target - share)

    def stretch(self, now: float) -> float:
        """The shortfall as 0 to 1, which is what every loosening is linear in.

        1.0 means the seat has said nothing at all in the window; 0.0 means
        it is at or above its share and every rule is back at its default.
        """
        if self.target <= 0.0:
            return 0.0
        return min(1.0, self.shortfall(now) / self.target)


@dataclass(frozen=True)
class Moment:
    """Everything :func:`may_speak` is allowed to look at, in one object.

    A frozen record rather than five arguments because the gate is the piece
    that has to be testable without a model, a trace or a runtime: a
    synthetic moment is six fields.
    """

    now: float
    #: The most recent caller forms, oldest last. Only the last
    #: ``phase_forms`` of them are read.
    forms: tuple[FormAt, ...] = ()
    #: The last event worth an opinion — one of :data:`JUDGED` — and when it
    #: was. Not only the big ones: a foul, a tackle and an offside are here
    #: too, because the rate cap's "one turn per event" is what lets the seat
    #: speak over the replays of an incident without waiting out the
    #: build-up gap, and an incident nobody has scored off is exactly the
    #: thing the corpus's second voice gives a verdict on.
    last_big: tuple[Event, float] | None = None
    #: How many lead lines have gone out at all, ever.
    lead_lines: int = 0
    #: How many lead lines have gone out since ``last_big``.
    lead_lines_since_big: int = 0
    #: When the seat last answered an offer, spoken or not.
    last_turn_ts: float | None = None
    #: How many turns it has taken since ``last_big``.
    turns_since_big: int = 0
    #: How far short of its share of the channel the seat is, 0 to 1. See
    #: :class:`Share`. Zero is the behaviour this gate had before there was
    #: a governor, and it is what a synthetic moment gets by default.
    stretch: float = 0.0

    @property
    def since_big(self) -> float:
        """Seconds since the last big event; infinite when there has been none."""
        if self.last_big is None:
            return float("inf")
        return self.now - self.last_big[1]


@dataclass(frozen=True)
class Offer:
    """May the seat be asked, which examples apply, and why — in words.

    The reason is not only for the log. On a yes it goes into the prompt as
    the reason this turn is being offered, because "the ball has been dead
    for nine seconds" and "the goal has been called and the lead has had his
    follow-up" want different lines out of the same model.
    """

    allowed: bool
    situation: str = IN_BUILD_UP
    reason: str = ""
    #: This is the short reaction inside the lead's own window, not a turn.
    #: The brief allows "one short reaction line" there and the corpus's four
    #: examples are all one fragment — "Well, well, well." — so the turn is
    #: capped at a single utterance when this is set.
    reaction: bool = False
    #: How many utterances there is actually room for before the lead comes
    #: back, or zero where nobody has worked it out. Offline the lead's next
    #: beats are known, so the model can be asked for the number that will
    #: fit rather than for four of which two are scheduled into his lines and
    #: thrown away — which is what happened to "Argentina two up, and this is
    #: the moment that changes it", paid for and never heard.
    room: int = 0


def may_speak(moment: Moment, cfg: ColourConfig | None = None) -> Offer:
    """The phase gate. Deterministic, free, and the whole of the timing rule.

    In order, and the order matters:

    1. **The seat never opens cold.** Section 4.6: the colour voice comes in
       after the lead and hands back by stopping when the ball moves. Before
       ``min_lead_lines`` lines exist there is nothing to observe off.
    2. **The first ``quiet_after_big_s`` seconds after a goal, shot, save or
       penalty belong to the lead** (section 4.3: median 21.4 s, 7% within
       6 s). The one exception is a goal reaction between
       ``goal_reaction_from_s`` and ``goal_reaction_to_s``, allowed only once
       and only if the lead has already said something since the goal —
       which is this system's stand-in for section 8.4's slots 4 to 6, the
       score and the tally and the rebuild, being done first.
    3. **Rate.** In the aftermath of a big event, one turn per event. In
       build-up, one turn per ``min_gap_s`` — or, when the seat is short of
       its share of the channel, per :func:`gap_when_behind`, which shrinks
       that towards ``min_gap_behind_s`` in proportion to the shortfall.
       One turn per big event is not loosened at any share: the twelve
       seconds after a goal belong to the lead whatever the ratio says.
    4. **Phase.** The last ``phase_forms`` forms are a dead ball, a stoppage,
       a replay, a close-up or a crowd shot — or ``settled_after_big_s``
       have passed since the last big event, which is the clause that lets
       the seat speak in ordinary live build-up at the corpus's 5.9 per 100.

    Everything else is a no, and the reason says which rule said so.
    """
    cfg = cfg or ColourConfig()
    if moment.lead_lines < cfg.min_lead_lines:
        return Offer(
            False,
            reason=(
                f"the lead has said {moment.lead_lines} lines and the seat does not open "
                f"before {cfg.min_lead_lines}"
            ),
        )

    big = moment.last_big
    since = moment.since_big
    if big is not None and big[0] in HELD_FOR_THE_LEAD and since < cfg.quiet_after_big_s:
        reaction = _goal_reaction(moment, cfg)
        if reaction is not None:
            return reaction
        return Offer(
            False,
            reason=(
                f"{since:.1f} s after the {big[0].value.replace('_', ' ')}, and the first "
                f"{cfg.quiet_after_big_s:.0f} s belong to the lead"
            ),
        )

    if moment.last_turn_ts is not None:
        gap = moment.now - moment.last_turn_ts
        if since < cfg.settled_after_big_s:
            # A goal gets two: the reaction fragment inside the lead's window
            # and a turn once he has finished, which is where the corpus's
            # "Well, they've done a Real Madrid" lands (+14 s). Anything
            # smaller gets one.
            allowed = 2 if big is not None and big[0] is Event.GOAL else 1
            if moment.turns_since_big >= allowed:
                return Offer(
                    False,
                    reason=(
                        f"{'two turns' if allowed == 2 else 'one turn'} per big event, and "
                        f"this one has already had {moment.turns_since_big}"
                    ),
                )
        else:
            owed = gap_when_behind(moment.stretch, cfg)
            if gap < owed:
                behind = (
                    ""
                    if owed >= cfg.min_gap_s
                    else f" (shortened from {cfg.min_gap_s:.0f} s: the seat is behind its share)"
                )
                return Offer(
                    False,
                    reason=(
                        f"only {gap:.1f} s since your last turn, and the gap is "
                        f"{owed:.0f} s{behind}"
                    ),
                )

    recent = moment.forms[-max(1, cfg.phase_forms) :]
    situation = _situation(moment, recent)
    if recent and all(form.quiet for form in recent):
        return Offer(True, situation, _phase_reason(recent))
    if since >= cfg.settled_after_big_s:
        return Offer(
            True,
            situation,
            (
                "nothing big has happened for "
                + ("the whole passage" if big is None else f"{since:.0f} s")
                + " and the ball is going sideways"
            ),
        )
    return Offer(
        False,
        reason=(
            "the ball is live and the last big event was "
            + ("never" if big is None else f"{since:.1f} s ago")
        ),
    )


def gap_when_behind(stretch: float, cfg: ColourConfig | None = None) -> float:
    """The build-up rate the seat is offered at, given its shortfall.

    ``min_gap_s`` at the target and ``min_gap_behind_s`` at a shortfall of
    the whole target, linear in between. The brief asks for the next dead
    ball "immediately rather than waiting for the 45 s build-up rate"; the
    floor is what stops "immediately" meaning every tick, because
    :meth:`ColourSeat.answered` spends the rate cap on a call that came back
    silent as well as on one that spoke.
    """
    cfg = cfg or ColourConfig()
    reach = max(0.0, min(1.0, stretch))
    return cfg.min_gap_s - (cfg.min_gap_s - cfg.min_gap_behind_s) * reach


def _goal_reaction(moment: Moment, cfg: ColourConfig) -> Offer | None:
    """The one line allowed inside the lead's window, or None.

    Section 4.3 again: at a goal, 31% of colour entries land within six
    seconds, and all four the study prints are reaction fragments rather than
    analysis. The three conditions are the study's and the brief's: it is a
    goal, the delay is in the window, and the lead has said at least one
    thing since — his follow-up beats — and the seat has not already taken
    this event's turn.
    """
    big = moment.last_big
    if big is None or big[0] is not Event.GOAL:
        return None
    since = moment.since_big
    if not (cfg.goal_reaction_from_s <= since <= cfg.goal_reaction_to_s):
        return None
    if moment.lead_lines_since_big < 1 or moment.turns_since_big >= 1:
        return None
    return Offer(
        True,
        AFTER_A_GOAL,
        (
            f"{since:.1f} s after the goal call and the lead has had his follow-up: ONE "
            "short reaction fragment, not analysis, and nothing else until the next event"
        ),
        reaction=True,
    )


def _situation(moment: Moment, recent: Sequence[FormAt]) -> str:
    """Which of the example groups this moment belongs to.

    :data:`OVER_A_REPLAY` is read off the picture first, before the kind of
    event is looked at, because the broadcast deciding to show a thing again
    is the thing the corpus measures: 15.4 colour entries per 100 utterances
    over a replay against 10.5 at a dead ball (section 4.2). A foul, a card,
    an offside or a tackle gets the same group without the replay, because
    what a second voice says about one of those is a verdict either way —
    section 3.2's booking window is the colour voice arguing about the
    challenge, whether or not the pictures are up.
    """
    big = moment.last_big
    fresh = big is not None and moment.since_big <= SITUATION_WINDOW_S
    if fresh and any(form.scene is Scene.REPLAY for form in recent):
        return OVER_A_REPLAY
    if fresh and big is not None:
        if big[0] is Event.GOAL:
            return AFTER_A_GOAL
        if big[0] in INCIDENTS or big[0] is Event.CARD:
            return OVER_A_REPLAY
        return AFTER_A_CHANCE
    if recent and all(form.quiet for form in recent):
        return AT_A_DEAD_BALL
    return IN_BUILD_UP


def _phase_reason(recent: Sequence[FormAt]) -> str:
    last = recent[-1]
    return (
        f"the ball is dead: the last {len(recent)} looks were "
        f"{last.scene.value.replace('_', ' ')} / {last.event.value.replace('_', ' ')}"
    )


def speaking_for(text: str) -> float:
    """How long the lead takes to say this, by the speaker's own arithmetic.

    ``WORDS_PER_SECOND`` is what ``Runtime._mark_spoken`` charges the rate cap
    with, and the two have to agree: a beat's timestamp is when a line
    *starts*, so anything that treats it as a point rather than as an
    interval will schedule the second voice over the tail of the first.
    """
    return len(text.split()) / WORDS_PER_SECOND


def space_out(
    start: float,
    count: int,
    *,
    gap: float,
    avoid: Sequence[tuple[float, float]] = (),
    clear: float = 0.0,
    span: float | None = None,
    lengths: Sequence[float] = (),
) -> list[float]:
    """When each utterance of one turn goes out, and where the turn stops.

    ``lengths`` is how long each utterance takes to say, so that it is placed
    where it can finish before the lead's next line, not just start.

    ``gap`` apart, never over one of the lead's lines, never out of order.
    ``avoid`` is his beats as ``(when it starts, how long it takes to say)``:
    two voices share one channel and the lead has the ball, so it is the
    colour utterance that moves.

    ``span`` ends the turn. Pushing clear of the lead stretches a turn, and a
    thought that lands fifteen seconds behind the one before it is not part
    of the same turn; the corpus's colour voice simply stops when the ball
    moves. Fewer timestamps come back than were asked for when that happens.

    Offline this can see the lead's future beats, which the live path cannot;
    live, the director's preemption does the same job after the fact.
    """
    blocked = sorted(avoid)
    out: list[float] = []
    at = start
    for index in range(max(0, count)):
        if out:
            at = max(at, out[-1] + gap)
        # The utterance's own length is part of the window it needs: an
        # utterance that starts clear of the lead and is still being said when
        # his next line lands is cut off mid-sentence by the director, and the
        # first listen heard exactly that — "they are interrupting each
        # other's sentences". So it has to end, plus the clearance, before he
        # starts, or it moves past him.
        length = lengths[index] if index < len(lengths) else 0.0
        at = _clear_of(at, blocked, clear, length=length)
        if span is not None and at > start + span:
            break
        out.append(at)
        at += gap
    return out


def _clear_of(
    ts: float, blocked: Sequence[tuple[float, float]], clear: float, *, length: float = 0.0
) -> float:
    """Push ``ts`` past every lead line it would talk over.

    One pass forward is not enough — stepping past one line can land on the
    next — so it walks the sorted list and takes the last push that applies.
    The window round a lead beat is asymmetric: ``clear`` before it starts,
    and ``clear`` after it has finished being said.
    """
    if clear <= 0.0:
        return ts
    at = ts
    for beat, seconds in blocked:
        # ``length`` is how long the utterance itself takes: it overlaps a
        # beat if any of [at, at + length] falls inside the window, not only
        # its first word.
        if beat - clear < at + length and at < beat + seconds + clear:
            at = beat + seconds + clear
    return at


#: Number words, for the rule that this seat does not say numbers. Ordinals
#: and the spelled-out decades are here because a commentator says "eighty
#: six" and "the first", not "86" and "1st".
_NUMBER_WORDS = (
    "two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand|dozen|"
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    "once|twice|double|treble|hat-trick"
)
_A_NUMBER = re.compile(rf"\b(?:\d+|{_NUMBER_WORDS})\b", re.IGNORECASE)

#: "One" is the only number word English also uses as a pronoun, and it is
#: not here with the rest for that reason. Two good lines died on it: "Well,
#: Messi got the better of that one." and "He had the crucial one against the
#: Netherlands." Neither counts anything.
#:
#: So "one" is checked separately: a determiner or an adjective in front of
#: it, or "of" after it, makes it a pronoun; anything else makes it a count.
#: "One-nil", "one in it", "one more" and "one behind" are all still numbers,
#: and the first of those is a scoreline.
_ONE = re.compile(r"\bones?\b", re.IGNORECASE)
_ONE_AS_A_PRONOUN = re.compile(
    r"\b(?:that|this|these|those|the|each|every|which|no|any|another|only|either|neither|"
    r"big|crucial|good|better|best|important|decisive|key|late|early|last|only|real|"
    r"first|second|hard|easy|clever|poor|great|bad|lucky|cheap|soft|silly)\s+ones?\b"
    r"|\bones?\s+of\b",
    re.IGNORECASE,
)


def says_a_number(text: str) -> bool:
    """Does this utterance carry a figure, in digits or in words?

    Section 5.1 is the rule and it is counted rather than argued: one
    utterance in six carries a number across seven matches, and **the share
    of number-carrying lines that open like the colour voice is 3.4-12.8%
    against a base rate of 2.4-14.1% — within noise on every file. Numbers
    are the lead's job**, dropped into the flow of play description, not the
    second seat's set piece. Gap 1 says the same thing from the other end.

    It is enforced here rather than asked for in the prompt because the first
    two measured passes of this seat produced "Back-to-back titles have only
    happened once" (the notes say no side has done it *since* 1962, which is
    not the same claim and is wrong: it has happened twice) and "Argentina
    are forty minutes from their first" with ten minutes left on the clock.
    Neither is a claim the fact gate can check, and both are the failure the
    invention score exists to catch.

    "One" is the exception and :data:`_ONE_AS_A_PRONOUN` is why: every other
    number word only ever counts something, and that one is also English's
    word for "the thing we were just talking about".
    """
    if _A_NUMBER.search(text):
        return True
    counted = {found.span() for found in _ONE.finditer(text)}
    pronouns = {found.span() for found in _ONE_AS_A_PRONOUN.finditer(text)}
    return any(
        not any(start >= low and end <= high for low, high in pronouns) for start, end in counted
    )


#: How long a run of words has to be before saying it twice is repeating
#: yourself. Four: "has been here before" is four, and three would strike
#: out "that is the" in two lines that are otherwise different sentences.
REPEAT_GRAM = 4

#: How many of the seat's own utterances a new one is checked against. Ten,
#: which on a turn of two to four is the last three or four turns — long
#: enough to catch a phrase the seat has settled into and short enough that
#: a line at the end of a match is not held against one in the first
#: minute.
REPEAT_HISTORY = 10

_NOT_A_WORD = re.compile(r"[^a-z0-9' ]+")


def _grams(text: str, n: int = REPEAT_GRAM) -> set[tuple[str, ...]]:
    """Every ``n``-word run in ``text``, folded for punctuation and case."""
    words = _NOT_A_WORD.sub(" ", text.lower()).split()
    if len(words) < n:
        return set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def repeats_itself(text: str, said_before: Sequence[str], n: int = REPEAT_GRAM) -> str:
    """The run of words this utterance shares with an earlier one, or ``""``.

    The colour prompt leaked: of 80 colour lines across the 39 pooled
    night-one traces, 17 carried "has been here before", which is a phrase
    out of the rules' own worked example. Taking it out of the prompt stops
    that one; this stops the next one, whatever it turns out to be, because
    a second voice that has found a sentence it likes will say it again
    whether it read it in the prompt or wrote it itself.

    It is a refusal and not a re-ask. The corpus's colour voice hands back
    by stopping (section 4.6), so a seat with nothing new to say staying
    quiet is the behaviour, not a failure of one.
    """
    seen: set[tuple[str, ...]] = set()
    for earlier in said_before:
        seen |= _grams(earlier, n)
    shared = _grams(text, n) & seen
    if not shared:
        return ""
    return " ".join(sorted(shared)[0])


#: Saying the scores are equal. The fact gate has its own six patterns for
#: this (``gate._LEVEL_CLAIMS``) and the colour seat got past every one of
#: them: on the Mbappé trace it said "Upamecano back in and France level from
#: the spot" at 2-1, and a bare predicative — a side *being* level rather
#: than levelling something — matches none of the gate's six, which all want
#: either a verb with an object ("levels it", "levelled the scores") or a
#: fixed phrase ("all square", "level terms", "it's level").
#:
#: So this seat checks the word itself, with the three things football calls
#: level that are not the score carved out: level **with** a man is an
#: offside, level **at** something is a table or a scoring chart — which is
#: what one of this pack's own notes about Mbappé says — and a level **ball**
#: is a pass. Everything else is the scoreboard, and the scoreboard is not
#: this seat's to read out whether or not it has it right.
#: Narrow on purpose, and narrowed again after it refused "This is what
#: experience at this level looks like" at 1-0. Football calls a great many
#: things level that are not the score: a standard ("at this level", "the top
#: level"), a defensive line, an offside ("level with the last man"), a
#: scoring chart ("level at the top", "level on five"). Only the shapes whose
#: subject can only be the scoreline are here.
_SCORES_LEVEL = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\blevell?(?:s|ed|ing)?\s+(?:it|things|matters|the\s+(?:scores?|game|tie|match))\b",
        r"\b(?:are|is|'re|'s)\s+level\b(?!\s+(?:with|at|on|to)\b)",
        r"\blevel\s+(?:from\s+the\s+spot|again|now)\b",
        r"\blevel\s+(?:terms|pegging)\b",
        r"\bthe\s+scores?\s+(?:are|is)\s+level\b",
        r"\bit(?:'?s|\s+is)\s+(?:all\s+)?level\b(?!\s+with\b)",
        r"\bequali[sz]\w*\b",
        r"\ball\s+square\b",
        r"\b(?:back\s+)?on\s+terms\b",
        r"\bparity\b",
        r"\bpegged\s+(?:them\s+|it\s+)?back\b",
    )
)


def says_the_scores_are_level(text: str) -> str:
    """The words this utterance uses to say the scores are equal, or ``""``.

    A scoreline with both numbers left out is still a scoreline. Section 5.1
    is the standing reason the colour voice does not give one — numbers are
    the lead's job and colour-opener lines carry them no more often than any
    other line — and this is the form of it the arithmetic cannot see.
    """
    for pattern in _SCORES_LEVEL:
        found = pattern.search(text)
        if found:
            return found.group(0)
    return ""


#: The event words that name one particular incident, and the caller form
#: each belongs to. Deliberately not :data:`EVENT_WORDS`, which is a much
#: looser list answering a different question ("is this line about
#: anything?"): these are the words that, said of a man, say he did it.
_EVENT_KINDS: dict[str, Event] = {
    "goal": Event.GOAL,
    "equaliser": Event.GOAL,
    "penalty": Event.PENALTY,
    "spot-kick": Event.PENALTY,
    "foul": Event.FOUL,
    "challenge": Event.FOUL,
    "handball": Event.FOUL,
    "trip": Event.FOUL,
    "card": Event.CARD,
    "booking": Event.CARD,
    "yellow": Event.CARD,
    "red": Event.CARD,
    "save": Event.SAVE,
    "stop": Event.SAVE,
    "tackle": Event.TACKLE,
    "block": Event.TACKLE,
    "offside": Event.OFFSIDE,
    "shot": Event.SHOT,
    "strike": Event.SHOT,
    "header": Event.SHOT,
    "volley": Event.SHOT,
    "finish": Event.SHOT,
    "effort": Event.SHOT,
    "corner": Event.CORNER,
    "cross": Event.CROSS,
}

#: One particular incident, rather than the kind of thing in general. "That
#: penalty", "the challenge", "his header", "Casemiro's foul" — a determiner
#: or a possessive, up to two words of opinion, then the event. The
#: determiner is what separates "he gave that penalty away" from "he has
#: saved penalties before", which is a note about a career and is true.
_ONE_EVENT = re.compile(
    r"(?:\b(?:the|that|this|those|his|her|their|its|a|an)\b|[\w'’-]+['’]s)\s+"
    r"(?:[\w-]+\s+){0,2}?"
    r"\b(" + "|".join(sorted(_EVENT_KINDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

#: A bare third-person subject. The measured fabrication was "Back in and
#: he's just conceded the penalty" — no name in it at all, and the man it
#: meant was the one the seat had named in the utterance before.
_HE = re.compile(r"\b(?:he|him|his|he's|he’s)\b", re.IGNORECASE)


#: The nouns a verdict is given about. Narrower than :data:`EVENT_WORDS`,
#: which answers "is this line about anything"; these are the things a second
#: voice passes judgement on.
_JUDGED_NOUNS = (
    "foul|penalty|penalties|card|booking|yellow|red|offside|save|finish|"
    "challenge|tackle|handball|decision|shout|call"
)

#: "A foul every time", "a penalty all day long". An intensifier that says
#: *how clear* the thing was, in the shape English gives it: a count word
#: doing no counting. The gate reads "every time" and "always" as claims
#: needing a note behind them (``gate._NOTE_CLAIMS``) and it is right to —
#: "always goes to the keeper's left" is a claim about a career. Next to a
#: verdict noun it is not one, and this is where the two are told apart.
_VERDICT_INTENSIFIER = re.compile(
    rf"\b(?:{_JUDGED_NOUNS})\b[^.!?]{{0,24}}?\b(every\s+time|all\s+day(?:\s+long)?|always)\b"
    rf"|\b(every\s+time|all\s+day(?:\s+long)?|always)\b[^.!?]{{0,24}}?\b(?:{_JUDGED_NOUNS})\b",
    re.IGNORECASE,
)

#: Just the intensifier, for taking it back out.
_INTENSIFIER = re.compile(r"\s*\b(?:every\s+time|all\s+day(?:\s+long)?|always)\b", re.IGNORECASE)


def _tidy_spaces(text: str) -> str:
    """Close the hole an elided word leaves, without moving anything else."""
    return re.sub(r"\s+([.,;:!?])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def verdict_intensifier(text: str) -> str:
    """The "every time" in "that is a foul every time", or ``""``.

    Only where a judgement noun is beside it, and only in this seat, and it
    is safe here for a reason that is checked upstream rather than argued:
    :func:`says_a_number` has already refused every utterance carrying a
    digit or a number word, so a colour line reaching the gate cannot hold a
    count. What is left of ``gate._NOTE_CLAIMS`` that this could hide is the
    wordless shapes — unbeaten, has not lost, never won, in a row,
    consecutive — and none of them is touched: only the intensifier itself is
    taken out of the probe, and the rest of the line still goes to the gate.

    It is here because the shape the corpus gives a verdict is exactly this
    one. "It's a ridiculous challenge from the Real Madrid captain. / It
    looks worse every time you see it." is section 3.2's booking window, and
    "That is a foul every time." — the verdict this seat was rebuilt to
    produce — was refused on ``runs/rephrased/r3-colour/mbappe`` as a note
    claim about a pack that has no note about fouls in it.
    """
    found = _VERDICT_INTENSIFIER.search(text)
    if not found:
        return ""
    words = _INTENSIFIER.search(text)
    return words.group(0).strip() if words else ""


def unsourced_names(text: str, sources: Sequence[str], pack: KnowledgePack | None) -> str:
    """The roster name in this line that nothing put in front of the seat.

    A team sheet says who exists; it does not say who is in this passage. The
    seat said "Mbappé took Tagliafico to the cleaners down that left side" off
    a note reading "takes the full-back on down the left" — Tagliafico is a
    real Argentina defender, the gate's roster check passed him, and nothing
    the seat had been given named him. The full-back might have been Molina.

    So every name has to be in the material, in the EVENT or REPLAY lines, or
    in one of the lead's recent lines. ``sources`` is those, already
    assembled; empty means no evidence and the check does not run, the same
    way the attribution check does not.
    """
    if not sources:
        return ""
    for name in roster_names(pack):
        if mentions(text, name) and not any(mentions(source, name) for source in sources):
            return name
    return ""


#: How a thing was done, in words the seat may only use if its material did.
#: After the volley on the Mbappé trace the seat said "Mbappé buried that
#: from the spot": the first goal had been a penalty, and the seat carried
#: its how onto the second. The man was right and the check on names passed.
_HOW_WORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a penalty", re.compile(r"\b(?:penalty|penalties|from the spot|spot[- ]kick)\b", re.I)),
    ("a header", re.compile(r"\b(?:header|headed|with his head)\b", re.I)),
    ("a free kick", re.compile(r"\bfree[- ]kick\b", re.I)),
    ("a volley", re.compile(r"\bvolley(?:ed)?\b", re.I)),
    ("an own goal", re.compile(r"\bown goal\b", re.I)),
)


#: A decision the referee has given, as it appears in the material, and the
#: denial of it. "That's not a penalty" went out at 56.7 s on the Mbappé
#: trace with the referee pointing to the spot; the listener heard the
#: second voice overrule the referee. A pundit may call a given decision
#: soft or harsh; the flat denial is not a verdict, it is a different match.
_DECISIONS: tuple[tuple[str, re.Pattern[str], re.Pattern[str]], ...] = (
    (
        "a penalty",
        re.compile(r"\bpenalt(?:y|ies)\b|\bfrom the spot\b|\bspot[- ]kick\b", re.I),
        re.compile(
            r"\b(?:not|never|no|isn't|is not|wasn't|was not)\s+(?:a\s+)?penalty\b"
            r"|\bno penalty\b|\bnot a spot[- ]kick\b",
            re.I,
        ),
    ),
    (
        "a foul",
        re.compile(r"\bfoul\b|\bfree[- ]kick\b", re.I),
        re.compile(
            r"\b(?:not|never|no|isn't|is not|wasn't|was not)\s+(?:a\s+)?foul\b|\bno foul\b", re.I
        ),
    ),
    (
        "a card",
        re.compile(r"\b(?:yellow|red|booked|booking|sent off)\b", re.I),
        re.compile(
            r"\b(?:not|never|no|isn't|is not|wasn't|was not)\s+(?:a\s+)?"
            r"(?:yellow|red|card|booking)\b",
            re.I,
        ),
    ),
)


#: Which way the keeper went. On the first penalty anybody watched with this
#: system the keeper dived the right way and could not reach it, the caller
#: said he was sent the wrong way, and both seats repeated it for a minute.
#: It is the commonest wrong detail on a penalty and worth nothing when it
#: is right, so neither voice says it.
_KEEPER_DIRECTION = re.compile(
    r"\b(?:sent|sends|went|goes|going|dived|dives|diving|guessed|guesses|guessing|committed)"
    r"\s+(?:him\s+|the\s+keeper\s+|the\s+goalkeeper\s+|[\w'’-]+\s+)?(?:the\s+)?"
    r"(?:wrong|other|right)\s+way\b",
    re.IGNORECASE,
)


def contradicts_decision(text: str, sources: Sequence[str]) -> str:
    """A given decision the utterance denies outright, or ``""``.

    Only against the EVENT lines of the material — a decision the match has
    actually recorded — so a verdict on an incident the referee waved away
    is still free to say it was nothing.
    """
    events = [line for line in sources if line.startswith("EVENT")]
    if not events:
        return ""
    pooled = " ".join(events)
    for label, given, denial in _DECISIONS:
        if given.search(pooled) and denial.search(text):
            return label
    return ""


def unsourced_how(text: str, sources: Sequence[str]) -> str:
    """A how the utterance claims that none of the material carries, or ``""``.

    Only the hows that name a distinct kind of goal or kick; "buried" and
    "finished" are anybody's. With no sources at all there is nothing to
    check against and the utterance passes, like the name check.
    """
    if not sources:
        return ""
    pooled = " ".join(sources)
    for label, pattern in _HOW_WORDS:
        if pattern.search(text) and not pattern.search(pooled):
            return label
    return ""


def _same_man(one: str, other: str) -> bool:
    """One person under two spellings: the surname, folded.

    The caller reads a name off whatever the broadcast put on the screen, so
    the same penalty came back as "Mbappé", "Kylian Mbappé" and "Kylian
    Mbappe" across three looks. The surname is the part every spelling of a
    name agrees on, which is why :func:`one_name_each` groups on it too.
    """
    return fold(one).rsplit(" ", 1)[-1] == fold(other).rsplit(" ", 1)[-1]


@dataclass(frozen=True)
class Attributed:
    """Who the caller's forms put on each event, for :func:`misattributes`.

    Evidence, not inference: ``by_kind`` is the names read off that event's
    own looks and the looks around it, and ``fresh`` is the names read off
    whatever the broadcast is on now, which is what an event this system has
    no form for at all is checked against instead.

    Empty means "no evidence", and the check does not run. A seat with no
    forms has no material either, so there is nothing for it to misattribute.
    """

    by_kind: Mapping[Event, tuple[str, ...]] = field(default_factory=dict)
    fresh: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.by_kind or self.fresh)

    def names_on(self, kind: Event) -> tuple[str, ...]:
        """Who may be credited or blamed for this kind of event."""
        return tuple(dict.fromkeys(self.by_kind.get(kind, ()) + self.fresh))


def misattributes(
    text: str,
    attributed: Attributed,
    pack: KnowledgePack | None,
    *,
    named_before: Sequence[str] = (),
) -> str:
    """Does this utterance hang an event on the wrong man? The reason, or ``""``.

    The predicate, in one sentence: **an utterance that names one particular
    incident — a determiner or a possessive in front of an event word — and
    names a man, or says "he" having named one earlier in the same turn,
    credits or blames that man for it, and he has to be somebody the caller's
    own forms read off that incident.**

    It is here because of two lines this seat really produced on the Mbappé
    trace. At 145.4 s: "Back in and he's just conceded the penalty", about
    Upamecano, whose whole material was one note saying he had missed the
    semi-final ill — Otamendi conceded it, thirty seconds of forms and four
    replays say so, and none of them says Upamecano. At 168.9 s: "Upamecano
    back in and France level from the spot." Neither is a claim the fact gate
    can check: the roster check passes a real player, the note check passes a
    real note, and what is false is the join between them.

    Two things keep it off the lines it should not touch. A note about a man
    is not an attribution — "he's back in the side tonight" names no event —
    and the plural or bare form is a career rather than an incident, so "he
    has saved penalties before" is not read as a claim about this one.

    Refusing costs a line and passing costs a lie, so where it is unsure it
    refuses: an event this system holds no form for at all is checked against
    the names on the pictures now, and a name that is on neither is refused.
    """
    if not attributed:
        return ""
    kinds = {_EVENT_KINDS[found.lower()] for found in _ONE_EVENT.findall(text)}
    if not kinds:
        return ""
    named = [name for name in roster_names(pack) if mentions(text, name)]
    if not named and _HE.search(text) and named_before:
        named = [named_before[0]]
    if not named:
        return ""
    for kind in sorted(kinds, key=lambda event: event.value):
        allowed = attributed.names_on(kind)
        for name in named:
            if not any(_same_man(name, other) for other in allowed):
                return f"attribution: {name} was not the man on that {kind.value.replace('_', ' ')}"
    return ""


def judge_utterance(
    text: str,
    state: MatchState,
    pack: KnowledgePack | None,
    gate: FactGate,
    *,
    goal_in_state: bool = True,
    at: float | None = None,
    said_before: Sequence[str] = (),
    attributed: Attributed | None = None,
    named_before: Sequence[str] = (),
    after: str = "",
    lead_said: Sequence[str] = (),
    only_repeated: bool = False,
    sources: Sequence[str] = (),
) -> GateVerdict:
    """One colour utterance, judged exactly as a phrased caller line is.

    It is wrapped in a form with no sightings on it, so the roster check
    judges the names it says, ``note_claim`` judges any number it gives, and
    a goal claim meets ``board_changed=False`` and is struck out — the second
    voice does not get to announce a goal.

    The scoreline is checked first and separately, by the analyst's own
    ``restates_score``, because the fact gate only rejects a score that
    contradicts the state and this seat may not read out a *correct* one
    either. The board reader owns the score, the graphic is on the screen,
    and section 5.1 is the reason: colour-opener lines are not where the
    numbers live, and the share of number-carrying lines that open like the
    colour voice is within noise of the base rate on every file.

    Then :func:`says_nothing` and :func:`is_filler`, which are the judge's
    5.0 turned into two predicates: a stock phrase is refused wherever it
    sits, and an utterance that names nobody, names no side doing something
    again and names no event is refused as ``colour_filler`` — unless it is a
    continuation and ``after``, the utterance of this turn that has just gone
    out, named exactly one man for it to call "he". They sit here rather than
    in the prompt because the prompt has asked for it three times and been
    given "I think this is what it comes down to", then "And that is the
    price of it right there".

    ``only_repeated`` says the turn's whole material was a count, and then
    the line has to say the thing happened *again*. Without that the count is
    doing no work in the sentence and what comes out is a reading of the
    picture with a number's permission slip: "Well, France keep giving it
    away from the wing", off a ledger line saying Théo Hernández had given
    away two throw-ins.

    Then :func:`repeats_itself` twice over. Against ``said_before`` — the
    seat's own last :data:`REPEAT_HISTORY` utterances — because the prompt is
    shown what the seat has said and said "has been here before" seventeen
    times in eighty lines anyway. And against ``lead_said``, the lead's last
    :data:`LEAD_ECHO_LINES` aired lines, because two voices sharing four
    words in a row eleven seconds apart is one voice: "He knew it from the
    moment it left his boot", then "Yeah, Ronaldo knew that was in the moment
    it left his foot."

    And last of the code checks, :func:`misattributes`, against
    ``attributed`` — who the caller's own forms put on each event — with
    ``named_before`` carrying the men this turn has already named so that a
    bare "he" is resolved to the one it means. Both of the fabrications the
    Mbappé trace produced were joins of two true things, which is the shape
    of claim neither the roster check nor the note check can see.
    """
    if restates_score(text, {"home_score": state.home_score, "away_score": state.away_score}):
        return GateVerdict(
            passed=False,
            reasons=["scoreline: the colour seat does not read the scoreboard back out"],
            line=text,
        )
    level = says_the_scores_are_level(text)
    if level:
        return GateVerdict(
            passed=False,
            reasons=[
                f'level_claim: "{level}" is the scoreline with the figures left out, and '
                f"the seat does not give the score (state {state.home_score}-"
                f"{state.away_score})"
            ],
            line=text,
        )
    if says_a_number(text):
        return GateVerdict(
            passed=False,
            reasons=["number_claim: numbers are the lead's job, not this seat's"],
            line=text,
        )
    meta = says_meta(text)
    if meta:
        return GateVerdict(
            passed=False,
            reasons=[f'meta: "{meta}" is your own briefing, and the listener cannot see it'],
            line=text,
        )
    stock = says_nothing(text)
    if stock:
        return GateVerdict(
            passed=False,
            reasons=[f'colour_filler: "{stock}" would fit any match ever played'],
            line=text,
        )
    bare = "" if after else says_only_a_name(text, pack)
    if bare:
        return GateVerdict(
            passed=False,
            reasons=[f'colour_filler: "{bare}" with nothing said about him is the lead\'s shape'],
            line=text,
        )
    if is_filler(text, pack, after=after):
        return GateVerdict(
            passed=False,
            reasons=[
                "colour_filler: names no player, no side doing it again and no event"
                + (
                    ", and the utterance before it named nobody to call him"
                    if after
                    else ", and it opens the turn"
                )
            ],
            line=text,
        )
    if only_repeated and not SAYS_AGAIN.search(text):
        return GateVerdict(
            passed=False,
            reasons=[
                "pattern_unsaid: the only material is a count, and the line does not say "
                "the thing happened again"
            ],
            line=text,
        )
    shared = repeats_itself(text, said_before)
    if shared:
        return GateVerdict(
            passed=False,
            reasons=[f'colour_repeat: you have already said "{shared}" this match'],
            line=text,
        )
    borrowed = repeats_itself(text, list(lead_said)[-LEAD_ECHO_LINES:])
    if borrowed:
        return GateVerdict(
            passed=False,
            reasons=[f'echoes_lead: your colleague has just said "{borrowed}"'],
            line=text,
        )
    wrong = misattributes(
        text,
        attributed if attributed is not None else Attributed(),
        pack,
        named_before=named_before,
    )
    if wrong:
        return GateVerdict(passed=False, reasons=[wrong], line=text)
    stranger = unsourced_names(text, sources, pack)
    if stranger:
        return GateVerdict(
            passed=False,
            reasons=[
                f"name_not_in_material: nothing you were given named {stranger} in this passage"
            ],
            line=text,
        )
    how = unsourced_how(text, sources)
    if how:
        return GateVerdict(
            passed=False,
            reasons=[f"how_not_in_material: nothing you were given said {how} in this passage"],
            line=text,
        )
    if _KEEPER_DIRECTION.search(text):
        # "Martínez went the other way, no blame there." The seat cannot see
        # the dive, the lead's own line about it was wrong, and the corpus's
        # second voice does not adjudicate a keeper's guess either way.
        return GateVerdict(
            passed=False,
            reasons=["keeper_direction: which way the keeper went is nobody's to say here"],
            line=text,
        )
    denied = contradicts_decision(text, sources)
    if denied:
        return GateVerdict(
            passed=False,
            reasons=[
                f"contradicts_decision: the referee gave {denied}; call it soft or harsh, "
                "never say it was not one"
            ],
            line=text,
        )
    # The verdict's "every time" is taken out of what the gate is shown and
    # put back into what goes to air. See :func:`verdict_intensifier` for why
    # that loosens no real count check, and note the two guards here: the
    # probe still carries the whole rest of the line, and the original is
    # restored only when the gate handed the probe back untouched. A trimmed
    # line is the gate's, verbatim.
    intensifier = verdict_intensifier(text)
    probe = _tidy_spaces(_INTENSIFIER.sub("", text)) if intensifier else text
    verdict = gate.judge(
        CallerLine(
            scene=Scene.STOPPAGE,
            event=Event.NONE,
            confidence=1.0,
            speak=True,
            line=probe,
        ),
        state,
        pack,
        board_changed=False,
        goal_in_state=goal_in_state,
        at=at,
    )
    if intensifier and verdict.passed and verdict.line.strip() == probe.strip():
        verdict.line = text
    # A trimmed colour utterance is refused, never aired with a hole in it.
    # The gate trims a caller's line because a long sentence with one clause
    # cut out of it is still a sentence; a colour utterance is three to
    # twelve words and what comes back is not one. "Well, Tagliafico's come
    # from to this summer" went to air on ``runs/rephrased/r6/mbappe`` with a
    # club name taken out of the middle of it.
    if verdict.passed and verdict.line.strip() != text.strip():
        return GateVerdict(
            passed=False,
            reasons=[*verdict.reasons, f'trimmed: what was left was "{verdict.line.strip()}"'],
            line=text,
        )
    return verdict


#: The counting vocabulary — which events are worth a count, what each is
#: called in the plural, how far back a spell reaches and how many times
#: something has to happen to be a pattern — lives in
#: :mod:`commentary.ledger` now, with the counting itself. There were two
#: copies of it and they were free to disagree.

#: How old the last big event may be and still be worth an opinion. Past
#: this the moment has gone and an opinion about it is a history lesson;
#: section 4.3's colour entries after a big event have a median delay of
#: 21.4 s, so the window is drawn just past that.
EVENT_FRESH_S = 25.0

#: How many of the lead's lines are read for the names a note may be about.
#: Three, because a note is only worth saying while the man it is about is
#: still the man the listener is thinking about.
LEAD_LINES_FOR_NOTES = 3

#: How much of a note two lines have to share before the second one is
#: saying it again. Two content words, stemmed to their first
#: :data:`NOTE_ECHO_PREFIX` letters, and the man's name in the line as well.
#:
#: Looser than :func:`~commentary.gate.notes_used`, which wants *every*
#: content word, and deliberately: that one decides whether a thread has been
#: told, where a wrong yes means the thread never comes back, so it is strict.
#: This one decides whether a note is worth saying for a third time in thirty
#: seconds, where a wrong yes costs one line and a wrong no is what the
#: Mbappé trace produced — the lead's "Upamecano, back in the side tonight"
#: at 133.2 s, then "Upamecano back after missing the semi with illness" at
#: 141.0 s, then "Upamecano missing that semi through illness" at 162.8 s.
#: Not one of those pairs shares a four-word run or every content word.
NOTE_ECHO_WORDS = 2

#: Stemming, such as it is: missed and missing are the same word, and so are
#: ill and illness. Three letters is crude and the short words it would
#: confuse are dropped as stopwords before it is applied.
NOTE_ECHO_PREFIX = 3

#: Words that carry none of a note. Everything else in it is content.
_NOT_CONTENT = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "from",
        "by",
        "as",
        "that",
        "this",
        "it",
        "its",
        "he",
        "his",
        "him",
        "she",
        "her",
        "they",
        "them",
        "their",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "not",
        "no",
        "so",
        "up",
        "out",
        "off",
        "who",
        "what",
        "when",
        "here",
        "there",
        "now",
        "then",
        "back",
        "into",
        "over",
        "after",
        "before",
        "again",
        "down",
        "all",
        "just",
        "still",
        "very",
        "more",
        "most",
        "well",
        "yeah",
        "you",
        "know",
        "think",
        "one",
        "two",
    }
)


def _stems(text: str) -> set[str]:
    """The content words of a line, cut to their first few letters."""
    words = _NOT_A_WORD.sub(" ", fold(text)).split()
    return {
        word[:NOTE_ECHO_PREFIX]
        for word in words
        if word not in _NOT_CONTENT and len(word) > NOTE_ECHO_PREFIX
    }


def echoes(note: Note, line: str) -> bool:
    """Has this line already carried this note?

    Both halves have to hold: the line names the man the note is about, and
    it shares :data:`NOTE_ECHO_WORDS` of the note's own content words with
    it. The name is load-bearing — two men went into that final level at the
    top of the scoring charts, and without it a line about one of them would
    take the other's note off offer.
    """
    if not _names_the_man(line, note.about):
        return False
    wanted = _stems(note.clause or "") | _stems(note.text)
    if len(wanted) < NOTE_ECHO_WORDS:
        return False
    return len(wanted & _stems(line)) >= NOTE_ECHO_WORDS


def _names_the_man(line: str, about: str) -> bool:
    """Is the note's subject in this line, under any of his spellings?"""
    return bool(about) and mentions(line, about)


#: Words that make a claim of repetition. Half of the material check: a line
#: about a team is only specific when it says the team did something *again*.
REPEATS = frozenset(
    {
        "again",
        "another",
        "same",
        "every",
        "each",
        "keeps",
        "keep",
        "still",
        "repeatedly",
        "more",
    }
)

#: The half of :data:`REPEATS` that looks backwards. "Again", "another",
#: "the same man" say a thing has happened before, which is what a count
#: entitles the seat to say. "Keep" and "keeps" do not: they say a thing is
#: happening now and will go on happening, which is a claim about a pitch
#: this seat cannot see. It made it twice on the offside clip off one
#: ledger count of two throw-ins — "Well, France keep giving it away from
#: the wing" and "And Argentina keep finding these set plays" — and both
#: were licensed by this set.
SAID_AGAIN = REPEATS - {"keep", "keeps"}

#: Saying what a side is doing as it is being done. A ledger count is a fact
#: about the past; turned into the present continuous it becomes a reading of
#: the picture, and the picture is the one thing this seat is never shown.
#: "That is where Argentina are finding their space" is the whole failure in
#: one line: true or false, nothing the seat was given could tell it which.
_PRESENT_TACTICAL = re.compile(
    r"\b(?:keep|keeps|are|is|were|was|been)\s+(?:\w+\s+){0,2}?\w+ing\b", re.IGNORECASE
)

#: And the same sentence with the auxiliary left out, which is how a
#: commentator writes it and how it got past the rule above: "Argentina
#: finding their numbers in midfield early on." Seven letters at least,
#: because "wing" and "thing" end in the same three and "France down the wing
#: again" is an observation; the handful of long nouns that end in them are
#: named.
_A_PARTICIPLE = re.compile(
    r"\b(?!everything|anything|nothing|something|morning|evening|meaning|feeling|warning)"
    r"\w{4,}ing\b",
    re.IGNORECASE,
)

#: The words that say a count out loud. An utterance built on nothing but a
#: REPEATED line has to carry one: the count is the only reason the line is
#: allowed, and "again" is the only part of it that reaches air, since
#: :func:`says_a_number` refuses the figure. "Once more" is not here and is
#: not offered in the rules either: "once" is a number word and
#: :func:`says_a_number` would refuse the line two checks earlier.
SAYS_AGAIN = re.compile(
    r"\b(?:again|another|the\s+same|same\s+(?:man|side|flank|end)|still)\b",
    re.IGNORECASE,
)

#: How many of the lead's aired lines a colour utterance is checked against.
#: Five, which at his rate is the last twenty to thirty seconds — everything
#: a listener still has in their head. The rules have forbidden paraphrasing
#: him since the seat existed and the free-kick pass produced "Yeah, Ronaldo
#: knew that was in the moment it left his foot" eleven seconds after he said
#: "He knew it from the moment it left his boot."
LEAD_ECHO_LINES = 5

#: Words that name something that happened rather than something that is
#: happening. The other half of the material check. Deliberately short and
#: concrete: a line with one of these in it is about an event, and a line
#: with none of these, no player and no team-plus-repetition is about
#: nothing.
EVENT_WORDS = frozenset(
    {
        "goal",
        "goals",
        "scored",
        "scores",
        "finish",
        "finished",
        "strike",
        "struck",
        "shot",
        "shots",
        "header",
        "volley",
        "penalty",
        "penalties",
        "spot",
        "save",
        "saved",
        "stop",
        "card",
        "booked",
        "booking",
        "yellow",
        "red",
        "foul",
        "fouled",
        "corner",
        "corners",
        "kick",
        "kicks",
        "offside",
        "throw",
        "substitution",
        "sub",
        "whistle",
        "referee",
        "ref",
        "keeper",
        "post",
        "bar",
        "net",
        # Added after "Every time you see it, there is contact in the box."
        # was refused for naming no event. A verdict on an incident is often
        # about the contact rather than about the award, which is how the
        # corpus's second voice talks over a replay: "every time you look at
        # it, it looks less and less like there was enough contact".
        "contact",
        "challenge",
        "challenges",
        "tackle",
        "tackles",
        "handball",
        "block",
        "blocked",
    }
)


@dataclass(frozen=True)
class Material:
    """What the seat is allowed to build a turn out of, and nothing else.

    The first material gate passed on team-level notes alone, and a team-level
    note is a licence to say anything about a team: what came back was
    "France keeping it simple across the back" and "I think this is what it
    comes down to", about a pitch the seat cannot see. The Opus judge scored
    the second voice 5.0 — "a second voice exists but says little" — and that
    is the same finding from the other end.

    So the material is specific or there is none. Three kinds, and each one
    is anchored to somebody or something a listener can check:

    ``notes``
        A note about a **player the lead has named in his last three lines**.
        Not about a team: the pack's team notes are the lead's to drop into
        a quiet moment, and in this seat's hands they became weather.
    ``patterns``
        Something that happened twice or more in the forms filed since the
        last turn, **with a name on it** — the same player in look after
        look, the same kind of event, one side down one flank. A team simply
        having the ball for a while is not a pattern.
    ``last_event``
        The last event in :data:`JUDGED` — a goal, a shot, a save, a
        penalty, a card, a foul, a tackle, an offside — less than
        :data:`EVENT_FRESH_S` old, **with the player named**. An opinion
        about that is an opinion about something that has finished.
    ``replays``
        What the caller wrote while the pictures were being shown again: his
        replay forms of that same incident, in his words. This is the
        evidence a verdict is given off — "every time you look at it, it
        looks less and less like there was enough contact" is a line about a
        replay and could not be written without one — and while replay forms
        keep arriving the incident stays fresh, because the broadcast is
        still on it.
    ``ledger``
        A count this match has produced: one about a player the lead has
        named in his last three lines, or a side count of two or more that
        has *moved* since this seat's last turn. Section 4.5's first kind of
        colour is "a pattern that has now repeated", and a count that has
        gone up is the plainest version of that there is. A count that has
        not moved since the seat last spoke is not news, and a side's first
        anything is the lead's to say — which is why
        :meth:`~commentary.ledger.Ledger.changed_since` exists and why it
        takes a floor.

    ``about`` is the one person the turn has to be about, set only for the
    sanctioned reaction after a goal, where the corpus's second voice talks
    about the scorer and nothing else.
    """

    notes: tuple[Note, ...] = ()
    patterns: tuple[str, ...] = ()
    last_event: str = ""
    about: str = ""
    ledger: tuple[Fact, ...] = ()
    replays: tuple[str, ...] = ()
    #: The lead's last aired lines. The first listen came back "barely any
    #: comments from the second commentator": the fence above gave the seat
    #: four one-line turns against twenty-seven lead lines, because it could
    #: only speak with a note, a count or an incident in hand. The corpus's
    #: second voice is a third of the words and most of them are an opinion
    #: about what the lead just described — "Spurs might have made more of
    #: that", "you don't want to be giving him a sight of goal" — and an
    #: opinion about the colleague's words is not a claim about the pitch.
    #: Every fabrication check still runs on it.
    lead: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Is there anything specific enough here to make a turn out of?"""
        return bool(
            self.notes
            or self.patterns
            or self.last_event
            or self.ledger
            or self.replays
            or self.lead
        )

    @property
    def only_a_count(self) -> bool:
        """Is a count of something the whole of what this turn may say?

        A turn with a note or a finished event behind it has a subject. A
        turn with nothing but ``patterns`` and ``ledger`` has an arithmetic,
        and the only sentence an arithmetic licenses is that the thing has
        happened again.
        """
        return bool(self.patterns or self.ledger) and not (
            self.notes or self.last_event or self.replays or self.lead
        )

    def lines(self) -> list[str]:
        """The material as the model is shown it: labelled, one item a line.

        A note carrying a figure and a ``clause`` — the researcher's own
        no-number rewrite of it — is shown the clause and nothing else: there
        is no figure left to say by mistake. Older notes with no clause fall
        back to the warning this always used to be, which is what let "a
        goal in the 2018 World Cup final at nineteen" come back as "Scored in
        a final at nineteen" and get struck out whole — a rule the model had
        been told twice, a foot away from the thing it was reading, and
        exactly the failure a pre-written clause is here to stop.

        The EVENT and its REPLAY lines come first. They are the thing the
        turn is for whenever there is one — section 4.2 puts the seat's
        highest rate of the match over a replay — and what is first in the
        list is what the first utterance is about. They used to come last,
        and on the Mbappé trace the seat built both of its turns out of one
        NOTE about a substitute while the penalty it was really about sat at
        the bottom of the block.
        """
        figure = " (has a figure in it: say the fact, never the figure)"
        then = " — that was then, not now"
        out: list[str] = []
        if self.last_event:
            out.append(f"EVENT, finished, speak about it in the past tense: {self.last_event}")
        out.extend(f"REPLAY, what the pictures showed again: {text}" for text in self.replays)
        out += [
            (
                f"NOTE about {note.about}: {note.clause}"
                if says_a_number(note.text) and note.clause
                else (
                    f"NOTE about {note.about}"
                    + (figure if says_a_number(note.text) else "")
                    + f": {note.text}"
                )
            )
            + (then if already_happened(note) else "")
            for note in self.notes
        ]
        out.extend(f"REPEATED: {text}" for text in self.patterns)
        # A count is shown as :attr:`~commentary.ledger.Fact.clause` — the
        # same fact with the figure taken out — and never as its text, for
        # the reason a note with a figure is shown its clause: told "say the
        # fact, never the figure" a foot away from the figure, this seat
        # says the figure. A count with no clause is not offered at all, so
        # there is no figure here to leave out.
        out.extend(f"REPEATED: {fact.clause}" for fact in self.ledger if fact.clause)
        out.extend(f"LEAD, what your colleague has just said: {text}" for text in self.lead)
        return out


#: A note anchored to a time that is not now. A note with one of these in it
#: is about a man as he was, and the seat read one of them as if it were
#: about him tonight: "a goal in a World Cup final, as a teenager" came back
#: as "That is what a teenager dreams of", about a man of twenty-three whose
#: age the lead had given twelve seconds earlier.
_ALREADY_HAPPENED = re.compile(
    r"\bas a (?:teenager|boy|youngster|kid|child)\b"
    r"|\bin (?:19|20)\d{2}\b"
    r"|\bback in\b"
    r"|\blast (?:season|year|time|month|summer)\b"
    r"|\b(?:aged|at) (?:nineteen|eighteen|seventeen|sixteen|twenty)\b",
    re.IGNORECASE,
)


def already_happened(note: Note) -> bool:
    """Is this note about a man as he was, rather than as he is tonight?"""
    return bool(_ALREADY_HAPPENED.search(f"{note.text} {note.clause}"))


#: A note about where a man stands against other men. Unlike a tally, which
#: :class:`~commentary.tallies.Tallies` can advance by counting, a standing
#: cannot be advanced from inside the broadcast: "level at the top of the
#: scoring charts" stops being true the moment either man scores and nothing
#: here knows what the other one has done tonight. So it is not adjusted, it
#: is withdrawn — the seat said "Yeah, Mbappé level with Messi on the charts
#: now" after Mbappé had scored twice in the same trace.
_A_STANDING = re.compile(
    r"\blevel\s+(?:at\s+the\s+top|with)\b"
    r"|\bjoint[\s-]top\b"
    r"|\b(?:one|two)\s+(?:behind|clear|ahead)\b"
    r"|\btop\s+of\s+the\s+(?:scoring\s+)?charts?\b"
    r"|\b(?:leads?|leading)\s+the\s+(?:scoring\s+)?charts?\b"
    r"|\btop\s+scorer\b"
    r"|\bgolden\s+boot\b",
    re.IGNORECASE,
)


def about_a_standing(note: Note) -> bool:
    """Is this note about where a man stands against somebody else?"""
    return bool(_A_STANDING.search(f"{note.text} {note.clause}"))


def _already_said(fact: Fact, patterns: Sequence[str]) -> bool:
    """Is a spell pattern already making this count's point?"""
    subject = fold(fact.about)
    noun = fold(noun_for(fact.kind, plural=True))
    return any(subject in fold(text) and noun in fold(text) for text in patterns)


def roster_names(pack: KnowledgePack | None) -> list[str]:
    """Everybody who exists, both squads, starters before bench."""
    if pack is None:
        return []
    names = [
        player.name
        for team in (pack.home, pack.away)
        for player in [*team.starters, *team.bench]
        if player.name
    ]
    # The managers exist too: "Scaloni got exactly what he demanded there"
    # was refused for naming nobody on a night the lead had just named him.
    names += [team.manager for team in (pack.home, pack.away) if team.manager]
    return names


def team_words(pack: KnowledgePack | None) -> list[str]:
    """What either side gets called: the name, the short form, the demonym."""
    if pack is None:
        return []
    return [
        word
        for team in (pack.home, pack.away)
        for word in (team.name, team.short, team.demonym)
        if word
    ]


def mentions(text: str, name: str) -> bool:
    """Is this name in this sentence, allowing for the surname alone?

    A selection heuristic, not a fact check — the gate's own roster matching
    does that afterwards. Folded on both sides because the caller reads a
    name off a graphic and the pack has it off a team sheet.

    Every run of one to three words in the line is offered to
    :func:`~commentary.gate.is_the_same_name`, which is the gate's own
    matcher and knows the two things a plain comparison does not: a name
    spaced differently — the pack says "Alexis MacAllister" and the line says
    "Mac Allister", which cost the seat a whole turn on
    ``runs/rephrased/r5b`` — and the initial form, "T. Hernández". One matcher
    for both seats, so a name the gate will accept is a name this seat can
    see.
    """
    folded = fold(name)
    if not folded:
        return False
    haystack = f" {fold(text)} "
    surname = folded.rsplit(" ", 1)[-1]
    if f" {folded} " in haystack or f" {surname} " in haystack:
        return True
    words = _NOT_A_WORD.sub(" ", fold(text)).split()
    return any(
        is_the_same_name(" ".join(words[start : start + length]), folded)
        for length in (2, 3)
        for start in range(max(0, len(words) - length + 1))
    )


def one_name_each(names: Sequence[str]) -> list[str]:
    """One spelling per person, the fullest one, in the order first read.

    The caller reads a name off whatever the broadcast put on the screen, so
    one penalty came back as "Mbappé", "Kylian Mbappé" and "Kylian Mbappe"
    across three looks and the material line named him three times. Grouped
    by folded surname, because that is the part every spelling agrees on.
    """
    best: dict[str, str] = {}
    order: list[str] = []
    for name in names:
        folded = fold(name)
        if not folded:
            continue
        key = folded.rsplit(" ", 1)[-1]
        if key not in best:
            order.append(key)
            best[key] = name
        elif len(name) > len(best[key]):
            best[key] = name
    return [best[key] for key in order]


#: Lines that would fit any match ever played. The first eight came out of
#: this seat's own mouth on the pooled night-one traces and have been in the
#: rules ever since; the last three came out of it on
#: ``runs/rephrased/r2-colour/mbappe`` — "And that is the price of it right
#: there", "Now the question is what he can do again", "Yeah, that's a finish
#: at this moment" — which is the same failure wearing a continuation's
#: clothes. In code rather than in the prompt because the prompt has listed
#: them, verbatim, for three passes and the seat wrote them anyway.
ABOUT_NOTHING = (
    "this is what it comes down to",
    "that changes everything",
    "have to find a way through",
    "keeping it simple at the back",
    "sitting deep and letting them have it",
    "this is the moment right here",
    "comes down to this",
    "know what they are protecting",
    "the price of it",
    "the question is what",
    "at this moment",
    "been building to",
    "been building towards",
    "building up to",
    "what it is all about",
)

#: And the shape: a sentence whose whole subject is a pronoun standing in for
#: the occasion. "This is what it has all been building to for him" went out
#: on ``runs/rephrased/r3-colour/mbappe`` and says nothing about the goal, the
#: man or the match; it would have fitted the other three goals equally.
_ALL_ABOUT_NOTHING = re.compile(
    r"\b(?:this|that|it)\s+is\s+what\s+(?:he|she|it|they|we|you)\b", re.IGNORECASE
)

#: And the shape rather than the phrase: a line that ends by pointing at
#: itself. "That is the price of it **right there**", "this is the moment
#: **right here**". The words are only filler at the end of the thought,
#: because "Otamendi's leg was right there" is an observation.
_POINTING_AT_ITSELF = re.compile(r"\bright (?:there|here|now)\b[.!?]*\s*$", re.IGNORECASE)

#: What a continuation is allowed to say instead of a name. Section 4.4's
#: real runs are full of them — "Morris is the man. / He's the man here.",
#: "He's backheeled the ball into the goal. / And another standing ovation."
#: — because the utterances of one turn go out two and a half seconds apart
#: and nobody has to be reintroduced in between.
_CARRIES_ON = re.compile(r"\b(?:he|him|his|she|her|they|them|their|he's|they've)\b", re.IGNORECASE)


#: The contractions a commentator actually says, and their long forms. Folded
#: before the stock-phrase check because "That's what he does in these
#: moments." went out while the rules and the pattern both had "that is what
#: he" in them. Only for matching: nothing here ever reaches air.
_CONTRACTIONS = tuple(
    (re.compile(rf"\b{short}\b", re.IGNORECASE), long)
    for short, long in (
        (r"that['\u2019]s", "that is"),
        (r"it['\u2019]s", "it is"),
        (r"he['\u2019]s", "he is"),
        (r"she['\u2019]s", "she is"),
        (r"here['\u2019]s", "here is"),
        (r"there['\u2019]s", "there is"),
        (r"they['\u2019]re", "they are"),
        (r"we['\u2019]re", "we are"),
        (r"what['\u2019]s", "what is"),
        (r"who['\u2019]s", "who is"),
    )
)


def spelled_out(text: str) -> str:
    """The same line with its contractions opened, for matching only."""
    for pattern, long in _CONTRACTIONS:
        text = pattern.sub(long, text)
    return text


def says_nothing(text: str) -> str:
    """The stock phrase this utterance is built on, or ``""``."""
    text = spelled_out(text)
    folded = " ".join(fold(text).split())
    for phrase in ABOUT_NOTHING:
        if phrase in folded:
            return phrase
    if _POINTING_AT_ITSELF.search(text):
        return "right there"
    shape = _ALL_ABOUT_NOTHING.search(text)
    return shape.group(0) if shape else ""


#: Words that open a phrase rather than a predicate. "Mbappé from the spot"
#: says where he was, not what he did, and it is the lead's shape: he calls
#: build-up in exactly this — "Here's Salah", "Now Griezmann", "Mbappé, off
#: the left".
_A_PHRASE_NOT_A_PREDICATE = frozenset(
    {
        "from",
        "on",
        "in",
        "into",
        "at",
        "off",
        "over",
        "under",
        "down",
        "up",
        "across",
        "through",
        "past",
        "behind",
        "beyond",
        "inside",
        "outside",
        "near",
        "with",
        "without",
        "for",
        "to",
        "by",
        "against",
        "around",
        "after",
    }
)

#: How long a phrase opening on one of those has to be before it is doing a
#: predicate's work. Four words is "from the spot" and "on the ball" and "in
#: the box"; five is "off the ground and buried it", which has a verb in it.
_PHRASE_WORDS = 5


#: The seat talking about its own briefing rather than about the match.
#: "Yeah, Mbappé did exactly what the note said he would do there" went out
#: on ``runs/rephrased/r5b/mbappe``. The notes are things the second voice
#: knows, the way it knows the team sheets; a broadcast in which one of them
#: says "the note" is a broadcast with the working out left in. "My
#: colleague" is here for a different reason and the rules give it: this seat
#: has never been told the lead's name and must not invent one.
_META = re.compile(
    r"\bthe\s+(?:note|notes|brief|briefing|pack|sheet|team\s+sheet|form\s+guide|research)\b"
    r"|\bwhat\s+the\s+note\b"
    r"|\bas\s+noted\b"
    r"|\bon\s+paper\s+(?:it\s+)?says\b"
    r"|\bmy\s+(?:colleague|co-commentator|notes)\b"
    r"|\bthe\s+material\b",
    re.IGNORECASE,
)


def says_meta(text: str) -> str:
    """The words in which this utterance names its own briefing, or ``""``."""
    found = _META.search(spelled_out(text))
    return found.group(0) if found else ""


def _filler_retry_note(first: str) -> str:
    """The opener described the pitch, or named nobody; say the opinion instead."""
    return (
        f'YOUR FIRST LINE WAS "{first}" AND IT IS THROWN AWAY: it describes the play as '
        "it happens, or names nobody, and you cannot see the pitch. Write the turn again. "
        "The first utterance is an OPINION about what your colleague said, in the past "
        "tense, naming the man or the side he named: \"<PLAYER> was slow to see that "
        'coming", "<SIDE> should have done better with that ball", "you would want more '
        'from <PLAYER> there". No "trying to", no "sitting deep", no "looking to", no '
        "present participle about a side. Then the short reactions that follow it."
    )


def _meta_retry_note(named: str) -> str:
    """Name the leak and ask again. One note, one extra call, once a turn."""
    return (
        f'YOU SAID "{named}". The notes are things you know, the way you know the team\n'
        "sheets. They are not things to cite, and the listener has never heard of them.\n"
        "Say the fact itself, as your own opinion, with the man's name on it — and if\n"
        "there is nothing to say without pointing at your own briefing, speak false."
    )


def _with_note(blocks: Sequence[Block], note: str) -> list[Block]:
    """The same call's body with a note on the end of its last text block."""
    again = [dict(block) for block in blocks]
    for block in reversed(again):
        if block.get("type") == "text":
            block["text"] = f"{block['text']}\n\n{note}"
            return again
    again.append(text_block(note))
    return again


def says_only_a_name(text: str, pack: KnowledgePack | None) -> str:
    """The name this utterance is, if it is nothing but a name, else ``""``.

    "Well, Mbappé." went out as a whole colour turn. A name with no predicate
    on it is the *lead's* shape — the corpus is full of "Here's Salah." and
    "Now Griezmann." — and in the second voice it is the sound of a seat that
    has been told to name somebody and has done only that.

    Measured on the opener alone, because "He's the man here." after "Morris
    is the man." is the corpus's own continuation and carries the predicate
    the pair needs between them.
    """
    rest = text.strip()
    cue = cue_of(rest)
    if cue:
        rest = rest[len(cue) :].lstrip(" ,")
    found = [name for name in roster_names(pack) if mentions(rest, name)]
    if not found:
        return ""
    for name in found:
        surname = fold(name).rsplit(" ", 1)[-1]
        rest = re.sub(rf"\b{re.escape(fold(name))}\b|\b{re.escape(surname)}\b", " ", fold(rest))
    left = _NOT_A_WORD.sub(" ", rest).split()
    if len(left) < 2:
        return found[0]
    # "Yeah, Mbappé from the spot." — a cue, a name and a prepositional
    # phrase, and no verb and no adjective anywhere in it. It got past the
    # word count and the judge caught it. Where the phrase runs on it is
    # doing a predicate's work and is left alone.
    if left[0] in _A_PHRASE_NOT_A_PREDICATE and len(left) < _PHRASE_WORDS:
        return found[0]
    return ""


#: The cues a turn may be swapped onto. All four take a comma, which is what
#: makes the swap safe to do in code: "You know, <rest>" becomes "Well,
#: <rest>" and nothing else about the sentence moves. "Well" opens 2.77% of
#: the club corpus's utterances and "Yeah" 2.23% (section 4.1), so a rotation
#: through these is the corpus's own distribution rather than a house style.
SWAPPABLE_CUES = ("Well", "Yeah", "Yes", "Oh")


def cue_of(text: str) -> str:
    """The opener this utterance begins on, folded, or ``""``."""
    head = text.strip().lower().lstrip("\"'")
    for cue in sorted(OPENERS, key=len, reverse=True):
        if head.startswith(cue):
            return cue
    return ""


def swap_cue(text: str, spent: str) -> str:
    """Move this utterance off ``spent`` onto another cue, or leave it alone.

    The rules have asked for a different cue every turn since the seat
    existed and the measured pass opened two turns in a row on "You know,"
    and two more on "Yeah,". A re-ask would cost a second model call for a
    word, so the swap is done here: the opener is the one part of an
    utterance that carries no claim, and replacing it can make nothing false.

    Only an utterance that really opens on the spent cue is touched, and only
    the cue itself: everything after the comma is the model's.
    """
    found = cue_of(text)
    if not found or found != spent:
        return text
    rest = text.strip()[len(found) :].lstrip()
    if rest.startswith(","):
        rest = rest[1:].lstrip()
    if not rest:
        return text
    for cue in SWAPPABLE_CUES:
        if cue.lower() != spent:
            return f"{cue}, {rest}"
    return text


def one_subject(text: str, pack: KnowledgePack | None) -> str:
    """The one man or one side this line is about, or ``""`` if it is not one.

    What the next utterance of the turn is allowed to call "he". Exactly one:
    a line naming two players leaves a listener with no antecedent, and a
    line naming none has nothing to hand on.
    """
    named = one_name_each([name for name in roster_names(pack) if mentions(text, name)])
    if len(named) == 1:
        return named[0]
    if named:
        return ""
    sides = {word for word in team_words(pack) if mentions(text, word)}
    folded = {fold(word) for word in sides}
    return sorted(sides)[0] if len(folded) == 1 else ""


def _names_anybody(text: str, pack: KnowledgePack | None) -> bool:
    """Is anybody on either team sheet in this line?"""
    return any(mentions(text, name) for name in roster_names(pack))


def is_filler(text: str, pack: KnowledgePack | None, *, after: str = "") -> bool:
    """Is this utterance about nothing?

    Three ways to be about something, and the **first** utterance of a turn
    needs one of them:

    - it names somebody on a team sheet;
    - it names a side **and** says they have done it **again** — "France down
      that side again" is an observation, "France keeping it simple" is a
      guess at a picture the seat cannot see, and "France keep giving it away
      from the wing" is the same guess wearing a count's clothes: the word
      that licenses it has to look backwards (:data:`SAID_AGAIN`) and the
      verb must not be the present continuous;
    - it names an event: a goal, a penalty, a save, a card, a foul, a corner.

    ``after`` is the utterance of this turn that has just gone out, and a
    fourth way opens with it. A **continuation** may say "he" or "they" when
    ``after`` named exactly one man or one side, because the two utterances
    are two and a half seconds apart on one held microphone and that is how
    the corpus's second voice talks: "Morris is the man. / He's the man
    here." This rule was written for the case where the lead cuts in between
    utterances, and it cost the seat the second line of every turn it took on
    the Mbappé trace — "That's what he does — he's dangerous the moment he
    comes on" was refused for naming nobody, a second and a half after it had
    named him.

    A continuation still has to be worth hearing. :func:`says_nothing` is
    checked first and on every utterance, first or not: a stock phrase is a
    stock phrase whoever it follows.
    """
    if says_nothing(text):
        return True
    words = set(fold(text).split())
    if not words:
        return True
    if words & EVENT_WORDS:
        return False
    if any(mentions(text, name) for name in roster_names(pack)):
        return False
    # Nobody named, no event named, and a verb saying what is being done as
    # it is done. Whatever this is, it came off the picture, and the picture
    # is the one thing the seat is never shown. Before the continuation
    # allowance rather than after it: "You know, Molina again down that right
    # side." then "That is where Argentina are finding their space." — the
    # first is the count said properly and the second went out on the back of
    # it, because "their" made it a continuation.
    if _PRESENT_TACTICAL.search(text):
        return True
    named_side = any(mentions(text, word) for word in team_words(pack))
    # A side and a bare participle is the same sentence with the auxiliary
    # left out, and it is how the seat actually writes it: "Argentina finding
    # their numbers in midfield early on."
    if named_side and _A_PARTICIPLE.search(text):
        return True
    if named_side and words & SAID_AGAIN:
        return False
    # A side and a judgement is an opinion about what the lead described,
    # not a guess at the picture: "Argentina were slow to react there",
    # "France should have done better with that". The present-tense
    # tactical shapes were refused above; what is left is a verdict.
    if named_side and words & JUDGEMENT_WORDS:
        return False
    if after and one_subject(after, pack) and bool(_CARRIES_ON.search(text)):
        return False
    # A short reaction carries on a turn the way the corpus's does — "I agree
    # with you.", "No.", "What a beauty.", "Be ready for it." — eight words
    # or fewer, after an opener that passed, and not a stock phrase (checked
    # first). The first listen had every second utterance refused for naming
    # nobody, and a median of one utterance a turn against a real four.
    # A pronoun with no referent is the one short shape that stays out: "he"
    # after a line naming two men points at neither, and the attribution
    # check could not read it.
    return not (
        after and len(text.split()) <= SHORT_REACTION_WORDS and not _CARRIES_ON.search(text)
    )


#: Words that make a line about a side an opinion rather than narration.
JUDGEMENT_WORDS = frozenset(
    {
        "poor",
        "good",
        "better",
        "best",
        "worse",
        "slow",
        "sloppy",
        "lucky",
        "brave",
        "clever",
        "naive",
        "careless",
        "wasteful",
        "sharp",
        "nervous",
        "composed",
        "should",
        "shouldn't",
        "could",
        "couldn't",
        "wanted",
        "deserved",
        "deserve",
        "badly",
        "well",
        "right",
        "wrong",
        "fortunate",
        "unlucky",
        "harsh",
        "soft",
        "brilliant",
        "terrific",
        "superb",
        "dreadful",
        "awful",
        "quality",
    }
)

#: A continuation this short is a reaction, and a reaction is allowed.
SHORT_REACTION_WORDS = 8


def patterns_in(
    forms: Sequence[FormAt], ledger: Ledger | None = None, since: float = -1.0
) -> list[str]:
    """What has happened twice or more, with a name on it and the count.

    The counting is :meth:`commentary.ledger.Ledger.patterns` and has been
    since the ledger existed. It moved because there were two counters: this
    one, over a window of forms, told the seat "3 corners in this spell", and
    the ledger told the lead "France's fourth corner" — two arithmetics over
    the same broadcast, free to disagree with each other in front of a
    listener. Now the spell count and the match count are one list of
    occurrences read over two windows.

    ``ledger`` is the match's own, shared with whatever is feeding it, and
    ``since`` is the instant the spell starts — the seat's last turn. Without
    one, a throwaway ledger is built out of the forms given, which is what a
    test does and what a seat with nothing shared falls back to.
    """
    if ledger is not None:
        return ledger.patterns(since, forms[-1].ts if forms else since)
    local = Ledger()
    for form in forms:
        local.saw_form(form.ts, form.as_line())
    return local.patterns(since, forms[-1].ts if forms else since)


class ColourSeat:
    """The second voice: what it has heard, when it may speak, and the call.

    It keeps its own memory of the match — the lead's last lines, the forms
    since its own last turn, when the last big thing happened — because both
    the runtime and the offline pass need exactly that and neither should
    have to rebuild it. Feed it with :meth:`saw_form` and
    :meth:`saw_lead_line`, ask it with :meth:`offer`, and it costs nothing
    until :meth:`turn`.
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: ColourConfig | None = None,
        pack: KnowledgePack | None = None,
        *,
        model: str | None = None,
        tallies: Tallies | None = None,
        ledger: Ledger | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or ColourConfig()
        self.pack = pack
        self.model = model if model is not None else self.config.model
        #: What this match has done to a running-count note, as of right now.
        #: Shared with whatever is crediting goals against the match — the
        #: runtime's own :class:`~commentary.threads.Threads`, or the local
        #: one :func:`colour_pass` builds — so a note that counts a player's
        #: goals reads the number he actually has, not the one researched
        #: before kickoff. A fresh, uncredited one if nothing is shared,
        #: which behaves exactly as this seat did before tallies existed.
        self.tallies = tallies if tallies is not None else Tallies()
        #: The counts this broadcast has made, as of right now. Shared the
        #: same way and for the same reason: one match, one set of numbers,
        #: whichever voice is reading one off it. **Shared means the owner
        #: feeds it** — the runtime and :func:`colour_pass` both walk every
        #: form already — so :meth:`saw_form` only feeds a ledger this seat
        #: made for itself, which is what a seat with nothing shared gets.
        self._owns_ledger = ledger is None
        self.ledger = ledger if ledger is not None else Ledger.from_pack(pack, tallies=self.tallies)
        #: Built once, never rebuilt: identical bytes on every turn is what
        #: makes forty real utterances and two squads affordable to send.
        self.system = colour_system(pack, self.config)
        #: The running lead:colour split over the last
        #: ``share_window_s`` of cursor time, and the whole of the ratio
        #: governor. Fed by :meth:`saw_lead_line` and :meth:`spoke_colour`,
        #: read by :meth:`moment` and by whoever is driving the lead's rate
        #: cap. Shared rather than private so the runtime can ask it what
        #: the lead's build-up cap should be this tick.
        self.share = Share(
            window_s=self.config.share_window_s,
            target=self.config.colour_share_target,
            min_sample=self.config.share_min_sample,
        )
        self._forms: list[FormAt] = []
        self._lead: deque[str] = deque(maxlen=max(1, self.config.lead_lines))
        self._said: deque[str] = deque(maxlen=max(1, self.config.lead_lines))
        #: Everything this seat has got past the gate this match, last
        #: :data:`REPEAT_HISTORY` only, for :func:`repeats_itself`. Separate
        #: from ``_said``, which is what the prompt is shown and is capped
        #: at the prompt's own window.
        self._history: deque[str] = deque(maxlen=REPEAT_HISTORY)
        #: Every line either voice has put out, with its instant, for
        #: :meth:`notes`. Both voices, because a note the lead has just
        #: dropped into a dead ball is as said as one this seat said, and a
        #: listener does not care which mouth it came out of.
        self._carried: list[tuple[float, str]] = []
        #: The cue the last turn opened on, so this one does not open on it
        #: again. The prompt has asked for that since the seat existed and
        #: the measured pass opened two turns in a row on "You know,".
        self._last_cue = ""
        self._last_big: tuple[Event, float] | None = None
        self._lead_lines = 0
        self._lead_since_big = 0
        self._last_turn_ts: float | None = None
        #: The kind of point the last turn made, so the next one can be told
        #: not to make the same kind again.
        self._last_angle = ""
        self._turns_since_big = 0
        #: Forms filed since the last turn, which is what the patterns are
        #: counted over.
        self._since_turn: list[FormAt] = []
        #: The instant of the last offer, so that :meth:`turn` rebuilds the
        #: same material the offer was allowed on without being handed the
        #: clock twice.
        self._offered_at = 0.0
        #: What the last turn was given to make itself out of. Kept so that
        #: the offline pass can print each utterance beside the material it
        #: was supposed to be about, which is the only way to read a turn and
        #: say whether it did what it was asked.
        self.last_material = Material()
        #: Why the last turn produced nothing, in words, for the error row.
        self.last_reason = ""
        #: What the last turn cost, so a pass can price itself per turn.
        self.last_usage = Usage()

    @property
    def enabled(self) -> bool:
        return colour_enabled(self.model)

    @property
    def lead_lines(self) -> list[str]:
        return list(self._lead)

    @property
    def said(self) -> list[str]:
        return list(self._said)

    @property
    def history(self) -> list[str]:
        """The seat's own last utterances, for the repeat check."""
        return list(self._history)

    # -- what it has heard ------------------------------------------------

    def saw_form(self, ts: float, line: CallerLine) -> None:
        """File a caller form, spoken or not.

        Not only the spoken ones: a form the caller filled in and chose not
        to speak is still the best evidence the seat has about what the
        picture was, and the phase gate reads the picture.
        """
        form = FormAt.of(ts, line)
        self._forms.append(form)
        self._since_turn.append(form)
        if self._owns_ledger:
            self.ledger.saw_form(ts, line)
        if form.event in JUDGED:
            # One goal, not five. The caller files the same goal over several
            # looks and then over the replays, and taking each of them as a
            # new big event reset both the rate cap and the once-per-goal
            # reaction: the last pass took two reaction fragments at the same
            # instant and a third off a replay twenty-four seconds later. So
            # a big event of the same kind inside the freshness window is the
            # same event, and it keeps the timestamp of the look the lead
            # actually called it on.
            #
            # An incident inside the window of anything is the same incident
            # too, whatever the caller called it this look. The Mbappé trace
            # files one piece of contact as foul, foul, foul, tackle, foul,
            # foul over twenty-five seconds; taking the change of word as a
            # new event would hand the seat a second turn on the same
            # challenge and would let a tackle filed five seconds after a
            # goal take the goal's place as the thing the turn is about.
            standing = self._last_big
            again = (
                standing is not None
                and ts - standing[1] <= EVENT_FRESH_S
                and (standing[0] is form.event or form.event in INCIDENTS)
            )
            if not again:
                self._last_big = (form.event, ts)
                self._lead_since_big = 0
                self._turns_since_big = 0

    def saw_lead_line(self, ts: float, text: str) -> None:
        """Record a line the lead actually got past the gate."""
        said = text.strip()
        if not said:
            return
        self._lead.append(said)
        self._carried.append((ts, said))
        self._lead_lines += 1
        self._lead_since_big += 1
        self.share.said(ts, colour=False)

    def spoke_colour(self, ts: float) -> None:
        """Record that one of this seat's utterances got past the gate.

        Separate from :meth:`accept`, which takes the text and has no
        clock: the governor counts utterances against cursor time and both
        the runtime and the offline pass know when each one went out.
        """
        self.share.said(ts, colour=True)

    def moment(self, now: float) -> Moment:
        """Everything the phase gate reads, as of ``now``."""
        return Moment(
            now=now,
            forms=tuple(self._forms),
            last_big=self._last_big,
            lead_lines=self._lead_lines,
            lead_lines_since_big=self._lead_since_big,
            last_turn_ts=self._last_turn_ts,
            turns_since_big=self._turns_since_big,
            stretch=self.share.stretch(now),
        )

    def offer(self, now: float) -> Offer:
        """May the seat be asked at this instant? Cheap; no model call.

        Two gates, and both are free. :func:`may_speak` asks whether the
        phase allows a turn; :meth:`material` asks whether there is anything
        specific enough to make one out of. A seat with a dead ball and
        nothing to say produced "Everything turns on what the ref decides
        next", and a seat with a team-level note produced "France keeping it
        simple across the back" — stopping both here rather than in the
        answer saves the call as well as the line.
        """
        self._offered_at = now
        offer = may_speak(self.moment(now), self.config)
        if not offer.allowed:
            return offer
        if not self.material(now, reaction=offer.reaction):
            return Offer(
                False,
                offer.situation,
                (
                    "nothing specific to say: no note about a man the lead has named, no "
                    "pattern with a name on it, no finished event"
                    if not offer.reaction
                    else "the goal reaction has no scorer to be about"
                ),
            )
        return offer

    def answered(self, now: float) -> None:
        """Record that an offer was put to the model, whatever came back.

        Silence costs a call, so silence has to spend the rate cap too;
        otherwise a seat that declines once is asked again half a second
        later, forever.
        """
        self._last_turn_ts = now
        self._turns_since_big += 1
        self._since_turn = []

    def accept(self, utterances: Iterable[str], ts: float | None = None) -> None:
        """Record what actually went out, so it is not said twice.

        ``ts`` is when it went out, for the note check in :meth:`notes`; the
        instant of the last offer is the fallback, which is this turn's own
        start and is within a few seconds of every utterance in it.
        """
        at = self._offered_at if ts is None else ts
        for text in utterances:
            said = text.strip()
            if said:
                self._said.append(said)
                self._history.append(said)
                self._carried.append((at, said))

    # -- the call ---------------------------------------------------------

    def notes(self, now: float | None = None) -> list[Note]:
        """The pack's notes about the players the lead has just named.

        Players, and only players. The old version of this walked the forms
        and then appended both team names, which meant a pack with a note
        about either country handed the seat material at every instant of the
        match — and "Argentina have not lost since that Saudi Arabia game" in
        this seat's mouth came out as "I think this is what it comes down
        to". A team note is the lead's to drop into a quiet moment.

        Most recently named first, which is the ordering
        :func:`~commentary.state.notes_for` caps against, so a note about the
        man still on the screen wins a place over one about the man before
        him. Adjusted by :attr:`tallies` before it goes anywhere: a note that
        counts goals is stale the instant the man it is about scores one, and
        this seat sees the match exactly as long as the lead does.

        And never a note either voice has just said. ``now`` is the instant
        the material is being built for, and a note carried by any line that
        has gone out inside
        :data:`~commentary.threads.CALLBACK_QUIET_S` — the lead's or this
        seat's own — is not offered again. That number is the corpus's, out
        of section 7: its callbacks are minutes apart and the shortest gap
        anywhere that is not one continuous burst is about five minutes. What
        happened without it is on ``runs/rephrased/r2-colour/mbappe``, where
        one note about a substitute was said three times in thirty seconds,
        once by the lead and twice by this seat.
        """
        found = [
            note
            for note in self.tallies.adjusted(notes_for(self.pack, self.lead_named()))
            if not self._overtaken(note)
        ]
        if now is None:
            return found
        recent = [line for ts, line in self._carried if 0.0 <= now - ts <= CALLBACK_QUIET_S]
        return [note for note in found if not any(echoes(note, line) for line in recent)]

    def _overtaken(self, note: Note) -> bool:
        """Has tonight made this standing note false?

        :meth:`~commentary.tallies.Tallies.adjust` can move a note that
        *counts* — a man on five goals is on six when he scores — because the
        arithmetic is inside the broadcast. A note about where he stands
        against somebody else cannot be moved that way: whether he is still
        level at the top depends on what the other man has done, tonight and
        at every other ground, and nothing here knows. So the first goal by
        anybody the note names takes it off offer.
        """
        if not about_a_standing(note):
            return False
        whom = [note.about] + [
            name for name in roster_names(self.pack) if mentions(f"{note.text} {note.clause}", name)
        ]
        return any(self.tallies.count(name, "goals") for name in whom if name)

    def lead_named(self) -> list[str]:
        """Who the lead has named in his last three lines, newest first."""
        found: list[str] = []
        for line in reversed(list(self._lead)[-LEAD_LINES_FOR_NOTES:]):
            for name in roster_names(self.pack):
                if name not in found and mentions(line, name):
                    found.append(name)
        return found

    def material(self, now: float, *, reaction: bool = False) -> Material:
        """The four things the seat may build a turn out of.

        See :class:`Material`. ``reaction`` is the sanctioned fragment after
        a goal, which is not a turn and is about one man.
        """
        if reaction:
            return self._about_the_scorer(now)
        since = self._last_turn_ts if self._last_turn_ts is not None else -1.0
        patterns = tuple(patterns_in(self._since_turn, self.ledger, since))
        return Material(
            notes=tuple(self.notes(now)),
            patterns=patterns,
            last_event=self._last_completed(now),
            ledger=tuple(self._counts(now, since, patterns)),
            replays=tuple(self._replays(now)),
            lead=tuple(list(self._lead)[-3:]),
        )

    def _replays(self, now: float) -> list[str]:
        """What the caller wrote over the pictures being shown again.

        The evidence, in his words, and the reason the seat has anything to
        judge at all: on the Mbappé trace the referee gave the penalty at
        12.9 s and the caller then filed four replay forms of the contact
        between 21.5 s and 37.8 s — "Otamendi's leg in behind him, and down
        he goes", "the contact from Otamendi as the France runner goes down
        inside the box" — every one of them unspoken, and none of them
        reaching this seat. Section 4.2 measures the seat at 15.4 entries
        per 100 utterances over a replay, its busiest phase of the match,
        and here it was empty.

        Only forms with words on them, only while they are fresh, and the
        last ``ColourConfig.replays_shown`` of them: a verdict wants the
        pictures it is a verdict on, not the whole sequence.
        """
        seen = [
            form.said.strip()
            for form in self._forms
            if form.scene is Scene.REPLAY
            and form.said.strip()
            and 0.0 <= now - form.ts <= EVENT_FRESH_S
        ]
        return seen[-max(1, self.config.replays_shown) :]

    def attributed(self, now: float) -> Attributed:
        """Who the caller's own forms put on each kind of event.

        The evidence :func:`misattributes` is checked against. Nothing is
        inferred here: it is the names the caller read off the picture on
        that event's own looks and on the looks around it, which is the only
        record this system has of who a thing happened to.
        """
        by_kind: dict[Event, tuple[str, ...]] = {}
        for form in self._forms:
            if form.event in JUDGED:
                by_kind[form.event] = ()
        for kind in by_kind:
            latest = max(f.ts for f in self._forms if f.event is kind)
            by_kind[kind] = tuple(one_name_each(self._names_around(latest)))
        return Attributed(by_kind=by_kind, fresh=tuple(one_name_each(self._names_around(now))))

    def _names_around(self, ts: float) -> list[str]:
        """Every name the caller read on a judged look within the window of ``ts``.

        Wider than the one event on purpose. A penalty is given for a foul
        and the foul's looks are the ones that named the man who gave it
        away, so an opinion that blames him for the penalty is true and the
        penalty's own form never said his name. Pooling the looks around the
        incident is what lets "<PLAYER> gave that penalty away" through and
        still refuses it about a man who was nowhere near it.
        """
        found: list[str] = []
        for form in self._forms:
            if form.event in JUDGED and abs(form.ts - ts) <= EVENT_FRESH_S:
                found.extend(name for name in form.names if name)
        return found

    def _counts(self, now: float, since: float, patterns: Sequence[str] = ()) -> list[Fact]:
        """The ledger clauses this seat is allowed to build a turn out of.

        Two kinds, and the second is the one that makes this a colour seat's
        material rather than the lead's. A count about a man the lead has
        just named is specific in the way :class:`Material` demands of
        everything. A side count is specific only when it has *moved*: the
        corpus's second voice remarks that something has happened again, and
        "France have had four corners" said at the same four it was said at
        last time is the weather this seat was rebuilt to stop producing.
        """
        named = self.ledger.facts(now, self.lead_named())
        # Only what has happened *again*. A count of one is not a repetition
        # and this seat has nothing to say about one, and — the measured
        # reason — a count of one has no number-free form, so offering it is
        # handing a voice that may not say figures a fact that is nothing
        # but a figure. Five utterances across two traces went out as "his
        # first tackle" and "Otamendi's first foul of the night" and every
        # one was struck by ``says_a_number``.
        # And never the same fact twice. The spell pattern and the match count
        # come out of the same occurrences, so "2 fouls on Otamendi" and
        # "another foul from Otamendi" can both be true of one moment, and
        # showing both spends two of the block's places on one thought — with
        # the numbered one there to be copied.
        found = [fact for fact in named + self.ledger.changed_since(since, now) if fact.clause]
        return [fact for fact in found if not _already_said(fact, patterns)]

    def _about_the_scorer(self, now: float) -> Material:
        """The material for the one line allowed just after a goal.

        The corpus's reaction fragments are about the man or about the move —
        "Morris is the man", "Another example, a pure quality finish and it
        starts outside the post and curls in" — and never about the match in
        general. What this seat produced instead was "I think that changes
        everything now", which would fit any goal ever scored.

        So: a note about the scorer if the pack has one, else the move in the
        lead's own words, else nothing, which means no turn at all.
        """
        scorer, called = self._the_goal(now)
        if not scorer:
            return Material()
        notes = tuple(self.tallies.adjusted(notes_for(self.pack, [scorer])))
        if notes:
            return Material(notes=notes, about=scorer)
        if called:
            return Material(last_event=f"the goal, {scorer} — he called it: {called}", about=scorer)
        return Material()

    def goal_scorer(self, now: float) -> tuple[str, float]:
        """Who scored the goal that has just been called, and when it was called.

        The seat's own reading of the scorer, made public so that a pass
        which has no goal-follow-up to tell it can credit the tallies off the
        same answer the goal reaction is built from. Empty when there is no
        fresh goal or nobody legible on it.
        """
        big = self._last_big
        if big is None or big[0] is not Event.GOAL:
            return "", 0.0
        scorer, _ = self._the_goal(now)
        return scorer, big[1]

    def _the_goal(self, now: float) -> tuple[str, str]:
        """Who scored the goal that has just been called, and how it was called.

        The caller files a goal over several looks and names the scorer on
        only some of them — on the Mbappé trace the form carrying "steps up
        and strikes it" read no name at all and the next one read Mbappé — so
        the run is read as one thing.
        """
        big = self._last_big
        if big is None or big[0] is not Event.GOAL or now - big[1] > EVENT_FRESH_S:
            return "", ""
        names, called = self._run_of(Event.GOAL, big[1])
        return (names[0] if names else ""), called

    def _run_of(self, event: Event, at: float) -> tuple[list[str], str]:
        """Every name read off this event's looks, and the words for it.

        Newest first for the names, because the scorer is whoever the caller
        could read closest to the goal being given; the words are the lead's
        first line about it, which is the one that described the move.

        The run reaches both ways round ``at``, not only backwards. The
        caller reads a name off whatever the broadcast put on the screen and
        that is often the look *after* the one the event was called on: the
        Mbappé penalty was filed at 82.5 s with four unreadable shirts on it
        and at 86.8 s with "Mbappé" on it, so a run that stopped at ``at``
        found no scorer and the goal reaction — 4 to 8 s after the call,
        section 4.3's one sanctioned fragment inside the lead's window —
        was refused for having nobody to be about.
        """
        names: list[str] = []
        called = ""
        for form in reversed(self._forms):
            if form.event is not event or abs(form.ts - at) > EVENT_FRESH_S:
                continue
            names.extend(name for name in form.names if name and name not in names)
            if form.said:
                called = form.said
        return one_name_each(names), called

    def _last_completed(self, now: float) -> str:
        """The last big thing the lead called, if it is fresh and has a name on it.

        Not the ball now — the seat cannot see the ball now, and the whole of
        the last pass's worst material was the present tense. A goal, a shot,
        a save, a penalty, a card, a foul, a tackle, an offside: something
        with a beginning and an end, under :data:`EVENT_FRESH_S` old, with
        the man it happened to named, and the words the lead used for it so
        that an opinion has something to be an opinion about. Anything looser
        was where "I think this is what it comes down to" came from.

        Freshness is measured off the **last look at the event**, replays
        included, not off the instant it happened. While the broadcast keeps
        showing a thing again the thing is still the subject, which is why
        the Mbappé foul is material at 37.8 s — twenty-five seconds after the
        contact and half a second after the fourth replay of it.
        """
        latest: FormAt | None = None
        for form in reversed(self._forms):
            if form.event in JUDGED:
                latest = form
                break
        if latest is None or now - latest.ts > EVENT_FRESH_S:
            return ""
        names, called = self._run_of(latest.event, latest.ts)
        if not names:
            return ""
        said = f" — he called it: {called}" if called else ""
        return f"{latest.event.value.replace('_', ' ')}, {', '.join(names)}{said}"

    async def turn(
        self, offer: Offer, state_summary: str, now: float | None = None
    ) -> ColourTurn | None:
        """Ask for one turn. ``None`` means the call failed, not silence.

        ``now`` defaults to the instant of the last :meth:`offer`, which is
        what both the runtime and the offline pass ask at; it is a parameter
        so that the material can be rebuilt rather than cached between the
        two calls.
        """
        self.last_reason = ""
        self.last_usage = Usage()
        at = self._offered_at if now is None else now
        material = self.material(at, reaction=offer.reaction)
        self.last_material = material
        most = self._how_many(offer, material, stretch=self.share.stretch(at))
        blocks = colour_blocks(
            offer.situation,
            offer.reason,
            state_summary,
            self.lead_lines,
            self.said,
            material.lines(),
            about=material.about,
            last_angle=self._last_angle,
            most=most,
        )
        parsed = await self._ask(blocks)
        if parsed is None:
            return None
        self.last_usage = parsed.usage
        # One re-ask, and only for the fault a rewrite actually fixes. A turn
        # that names its own briefing — "Mbappé did exactly what the note said
        # he would do" — has the right subject and the wrong frame, and the
        # model that wrote it can write it again without the frame. Everything
        # else this seat refuses is refused outright: a line with no material
        # behind it does not become one when asked twice, and a second call
        # doubles what the turn costs.
        named = next((says_meta(text) for text in parsed.value.utterances if says_meta(text)), "")
        if named:
            again = await self._ask(_with_note(blocks, _meta_retry_note(named)))
            if again is not None:
                self.last_usage = self.last_usage + again.usage
                parsed = again
        # And one more, for the opener that reads the picture. With the lead's
        # lines as material the seat's first instinct is "France trying to
        # build something here", which the filler check refuses and which
        # costs the whole turn. The material is right and the tense is wrong;
        # asked again with the shape spelled out, the same model writes the
        # opinion it was meant to.
        first = next((text for text in parsed.value.utterances if text.strip()), "")
        if first and is_filler(first, self.pack):
            again = await self._ask(_with_note(blocks, _filler_retry_note(first)))
            if again is not None:
                self.last_usage = self.last_usage + again.usage
                parsed = again
        return self._settle(parsed.value, most)

    async def _ask(self, blocks: list[Block]) -> Parsed[ColourTurn] | None:
        """The call itself. ``None`` is the model failing, not silence."""
        try:
            return await self.backend.parse(
                model=self.model,
                system=self.system,
                blocks=blocks,
                output_format=ColourTurn,
                max_tokens=self.config.max_tokens,
                effort="low",
                cache_system=True,
                tag="colour",
            )
        except LLMError as exc:
            self.last_reason = f"model call failed: {exc}"
            return None

    def _how_many(self, offer: Offer, material: Material, *, stretch: float = 0.0) -> int:
        """How many utterances this turn is allowed to be.

        Section 4.4 has the colour voice holding the microphone for a median
        of four utterances, but that is a voice with a whole match in its
        head. This seat has whatever the material gate let through, and on
        one item a four-utterance turn is one thought and three restatements
        — which is what the judge heard. So: the sanctioned goal reaction is
        one fragment, a single item of material is worth two utterances (the
        thing, then why it matters), and only real material gets the run.

        ``stretch`` is the ratio governor, and it lifts only the one-item
        clamp: a seat short of its share of the channel gets the corpus's
        full run rather than half of it. It never lifts ``max_utterances``,
        never overrides ``offer.room`` — that is the lead still talking —
        and it does nothing at all to a turn with no material behind it,
        because the material gate has already refused that turn.
        """
        if offer.reaction:
            return 1
        limit = self.config.max_utterances
        if len(material.lines()) <= 1 and stretch <= 0.0:
            limit = min(2, limit)
        if offer.room:
            limit = min(limit, offer.room)
        return max(1, limit)

    def _settle(self, proposed: ColourTurn, limit: int) -> ColourTurn:
        """Trim, cap and count, before anyone sees the turn.

        A single utterance is kept rather than dropped — 20% of the corpus's
        real turns are one utterance — but the prompt asks for two to four
        and the count is one of the measurements.
        """
        kept: list[str] = []
        for text in proposed.utterances:
            cleaned = clean_line(text)
            said = trim_words(cleaned, self.config.max_words)
            # Dropped rather than trimmed. The study's case against the old
            # analyst's cap is that "a 30-word cap that truncates mid-clause
            # is worse than a shorter cap: the corpus's colour entries are 3
            # to 12 words each and simply come in sequence" — and the first
            # measured turn of this seat produced "Three times in quick
            # succession he's been the one France are running", cut exactly
            # there. A turn is a run, so losing one of its utterances costs
            # the listener a pause and not a thought.
            if said and said == cleaned:
                kept.append(said)
            if len(kept) >= limit:
                break
        if not proposed.speak or not kept:
            self.last_reason = self.last_reason or "the seat chose silence"
            return proposed.model_copy(update={"utterances": kept, "speak": False})
        kept[0] = swap_cue(kept[0], self._last_cue)
        self._last_cue = cue_of(kept[0])
        self._last_angle = proposed.angle.value
        return proposed.model_copy(update={"utterances": kept, "speak": True})


# -- the offline pass ---------------------------------------------------


@dataclass
class ColourUtterance:
    """One colour utterance, scheduled, judged, and printable."""

    ts: float
    text: str
    situation: str
    passed: bool
    reasons: tuple[str, ...] = ()
    #: The lead's last line before this one went out, for the printed table.
    after: str = ""
    #: The one man the utterance before it named, and so who this one means
    #: by "he". Empty on the first utterance of a turn, and on any
    #: continuation whose predecessor named nobody or named two people. It is
    #: what lets a continuation carry a pronoun at all and what the
    #: attribution check reads that pronoun as.
    referent: str = ""

    @property
    def cue(self) -> bool:
        return opens_with_a_cue(self.text)

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class ColourTurnRecord:
    """One offered turn: what the gate said, what the model said, what it cost."""

    ts: float
    situation: str
    reason: str
    spoke: bool
    #: Was this the sanctioned one-line goal reaction rather than a turn?
    reaction: bool = False
    #: What the seat was given to make this turn out of, labelled. Printed
    #: beside the utterances, because "is this line about anything?" is not a
    #: question you can answer without knowing what it was handed.
    material: tuple[str, ...] = ()
    angle: str = ""
    cites: tuple[str, ...] = ()
    utterances: list[ColourUtterance] = field(default_factory=list)
    usd: float = 0.0
    since_big_s: float | None = None

    @property
    def spoken(self) -> list[ColourUtterance]:
        return [u for u in self.utterances if u.passed]


@dataclass
class ColourPass:
    """What the offline pass produced: new rows, the turns, and the bill."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    turns: list[ColourTurnRecord] = field(default_factory=list)
    cost_usd: float = 0.0
    #: The window the counts are measured against, carried rather than read
    #: off the global settings so a sweep of it is visible in the numbers.
    quiet_after_big_s: float = ColourConfig.quiet_after_big_s
    #: The lead's own utterances on the trace this pass walked, so the
    #: colour share can be a share rather than a count.
    lead_lines: int = 0
    #: What that share is being held to. See ``ColourConfig``.
    share_target: float = ColourConfig.colour_share_target

    @property
    def spoken(self) -> list[ColourUtterance]:
        return [u for turn in self.turns for u in turn.spoken]

    @property
    def refused(self) -> list[ColourUtterance]:
        return [u for turn in self.turns for u in turn.utterances if not u.passed]

    @property
    def spoke(self) -> list[ColourTurnRecord]:
        return [turn for turn in self.turns if turn.spoke]

    def table(self) -> str:
        """Every colour utterance, with the lead line it came in behind."""
        if not self.turns:
            return "no colour turns"
        out: list[str] = []
        for turn in self.turns:
            head = f"{turn.ts:>7.1f}  {turn.situation:<17}"
            if not turn.spoke:
                out.append(f"{head}  (silent) {turn.reason}")
                continue
            out.append(f"{head}  {turn.angle} <- {turn.reason}")
            for item in turn.material:
                out.append(f"{'':>7}  {'material':<17}  * {item}")
            for utterance in turn.utterances:
                mark = " " if utterance.passed else "x"
                if utterance.after:
                    out.append(f"{'':>7}  {'lead':<17}  | {utterance.after}")
                reason = "" if utterance.passed else f"   [{'; '.join(utterance.reasons)[:60]}]"
                out.append(f"{utterance.ts:>7.1f}  {'':<17}{mark} {utterance.text}{reason}")
        return "\n".join(out)

    def counts(self) -> dict[str, float]:
        """The numbers the study asks for, in one dict."""
        spoke = self.spoke
        runs = [len(turn.spoken) for turn in spoke if turn.spoken]
        spoken = self.spoken
        first = [turn.spoken[0] for turn in spoke if turn.spoken]
        # Two different numbers, and the study wants them apart. The target
        # of zero is turns that landed inside the lead's window *without*
        # being the sanctioned goal reaction; the reactions themselves are
        # the 31% of colour entries section 4.3 says do arrive inside six
        # seconds of a goal, and they are meant to be there.
        inside = sum(
            1
            for turn in spoke
            if turn.since_big_s is not None
            and turn.since_big_s < self.quiet_after_big_s
            and not turn.reaction
        )
        reactions = sum(1 for turn in spoke if turn.reaction)
        heard = self.lead_lines + len(spoken)
        return {
            "turns_offered": float(len(self.turns)),
            "turns_spoken": float(len(spoke)),
            "utterances": float(len(spoken)),
            # What a listener heard, not what was offered: the governor's
            # own number. Section 4's club football is 0.31.
            "colour_share": (len(spoken) / heard) if heard else 0.0,
            "colour_share_target": self.share_target,
            "inside_12s_of_a_big_event": float(inside),
            "sanctioned_goal_reactions": float(reactions),
            "median_utterances_per_turn": _median(runs),
            "median_words": _median([float(u.words) for u in spoken]),
            "cue_share": (sum(1 for u in first if u.cue) / len(first)) if first else 0.0,
            "refused": float(len(self.refused)),
            "usd": self.cost_usd,
        }


def _median(values: Sequence[float] | Sequence[int]) -> float:
    kept = sorted(float(v) for v in values)
    if not kept:
        return 0.0
    middle = len(kept) // 2
    if len(kept) % 2:
        return kept[middle]
    return (kept[middle - 1] + kept[middle]) / 2


async def colour_pass(
    rows: list[dict[str, Any]],
    backend: LLMBackend,
    *,
    pack: KnowledgePack | None = None,
    settings: Settings = SETTINGS,
    model: str | None = None,
    goal_in_state: Callable[[float], bool] = lambda _ts: True,
    state_summary: Callable[[MatchState], str] = lambda state: state.scoreline,
) -> ColourPass:
    """Walk a rephrased trace and give the colour seat its turns.

    The same loop the runtime runs, on a trace instead of a clip: tick every
    ``predictor.tick_s``, hand the seat every caller form and every lead beat
    as its timestamp goes past, ask the phase gate, and call the model only
    where it says yes. Nothing here re-reads the picture, so the whole pass
    is a few cents and needs no clip.

    Two things are better offline than live and both are said out loud here.
    The scheduler can see the lead's *future* beats, so a colour utterance
    never has to be preempted to stay out of the lead's way; live, the
    director does that after the fact. And ``goal_in_state`` is handed in
    rather than guessed — :class:`commentary.rephrase.Cover` rebuilt it from
    the state rows and the same answer should reach both seats' gates.
    """
    cfg = settings.colour
    #: This pass's own memory of what has been credited and what has been
    #: said — the same :class:`~commentary.threads.Threads` the lead pass
    #: uses, but a pass of its own, because the lead's has already reached
    #: the final whistle by the time this runs. Fed one state row at a time
    #: as ``now`` passes it, below, never the final state up front, which
    #: would hand the seat a note advanced by goals that, at that point in
    #: the walk, have not been called yet.
    threads_local = Threads.from_pack(pack)
    #: And this pass's own count of what the broadcast did, for the same
    #: reason and fed the same way: one form at a time as the walk reaches
    #: it, never the finished match up front. Sharing the tallies with the
    #: threads is what keeps a player's goals counted once.
    ledger_local = Ledger.from_pack(pack, tallies=threads_local.tallies)
    seat = ColourSeat(
        backend,
        config=cfg,
        pack=pack,
        model=model,
        tallies=threads_local.tallies,
        ledger=ledger_local,
    )
    out = ColourPass(
        rows=list(rows),
        quiet_after_big_s=cfg.quiet_after_big_s,
        share_target=cfg.colour_share_target,
    )
    if not seat.enabled:
        return out

    states = _states(rows)
    forms = _forms(rows)
    beats = _lead_beats(rows)
    out.lead_lines = len(beats)
    beat_ts = [(ts, speaking_for(text)) for ts, text in beats]
    lag = _lag(rows, settings.capture.delay_s)
    gate = FactGate(settings.gate)
    additions: list[tuple[float, dict[str, Any]]] = []

    events: list[tuple[float, str, Any]] = [
        *((ts, "form", value) for ts, value in forms),
        *((ts, "beat", value) for ts, value in beats),
    ]
    events.sort(key=lambda item: item[0])
    cursor = 0
    tick = max(0.05, settings.predictor.tick_s)
    now = events[0][0] if events else 0.0
    end = max((ts for ts, _, _ in events), default=0.0)

    while now <= end + tick:
        while cursor < len(events) and events[cursor][0] <= now:
            ts, kind, value = events[cursor]
            if kind == "form":
                ledger_local.saw_form(ts, value)
                seat.saw_form(ts, value)
            else:
                seat.saw_lead_line(ts, value)
            cursor += 1

        state = _state_at(states, now)
        if state is not None:
            threads_local.see_state(state)
            ledger_local.see_state(state)
        # Credit the scorer here, because nothing else in this pass will.
        # Live, :class:`~commentary.goalfollow.GoalFollowup` names him and
        # credits the tallies the seat shares with the lead; offline there is
        # no follow-up, and the board's own incident carries ``player=None``
        # — a scoreboard cannot see who scored — so ``Tallies.see_state``
        # credits nobody and every note stayed as researched. It aired "Yeah,
        # Mbappé level at the top of the scoring charts now" fifteen seconds
        # after he had gone one clear of them.
        #
        # Stamped with the goal's own instant rather than with ``now`` so
        # that crediting it on every tick of the freshness window is one
        # goal: ``Tallies`` deduplicates inside eight seconds of an existing
        # credit, and ``now`` would walk out of that window and count it
        # twice.
        scorer, called_at = seat.goal_scorer(now)
        if scorer:
            threads_local.tallies.credit_goal(scorer, called_at)

        offer = seat.offer(now)
        room = _room_for(now, beat_ts, cfg) if offer.allowed else 0
        # The sanctioned reaction after a goal is one fragment, so one hole
        # in the lead's cadence is all it needs. Asking it for two was why
        # it never went out: with the run of looks now reaching past the
        # call, the seat had a scorer at 4.7 s on the Mbappé trace and was
        # turned away at every tick of the window for want of a second slot
        # it was never going to use.
        needed = 1 if offer.reaction else cfg.min_utterances
        if offer.allowed and room >= needed:
            offer = replace(offer, room=room)
        elif offer.allowed:
            # The phase says yes and the lead has not drawn breath. Checked
            # before the call rather than after it, because a turn that ends
            # up entirely inside the lead's lines is a turn that was paid for
            # and thrown away: the first pass with the interval clearance on
            # spent three calls out of five that way. Offline the lead's next
            # beats are known, so this is free; live it is the director's job
            # and is done after the fact, by preemption.
            offer = Offer(
                False,
                offer.situation,
                f"the lead has not stopped talking long enough for {needed} utterances",
            )
        if offer.allowed:
            moment = seat.moment(now)
            turn = await seat.turn(offer, state_summary(state) if state else "")
            usd = seat.last_usage.cost_usd
            out.cost_usd += usd
            seat.answered(now)
            record = ColourTurnRecord(
                ts=now,
                situation=offer.situation,
                reason=offer.reason,
                spoke=bool(turn is not None and turn.speak and turn.utterances),
                reaction=offer.reaction,
                material=tuple(seat.last_material.lines()),
                angle=turn.angle.value if turn is not None else "",
                cites=tuple(turn.cites) if turn is not None else (),
                usd=usd,
                since_big_s=None if moment.last_big is None else moment.since_big,
            )
            if turn is None:
                additions.append(
                    (
                        now,
                        {
                            "topic": Topic.ERROR.value,
                            "ts": now,
                            "where": "colour",
                            "detail": seat.last_reason or "the colour seat returned nothing",
                        },
                    )
                )
            elif record.spoke:
                _schedule(
                    record,
                    turn,
                    seat=seat,
                    gate=gate,
                    state=state,
                    pack=pack,
                    cfg=cfg,
                    beats=beats,
                    beat_ts=beat_ts,
                    lag=lag,
                    goal_in_state=goal_in_state,
                    additions=additions,
                    threads=threads_local,
                )
            additions.append(
                (
                    now,
                    _colour_row(
                        now,
                        offer,
                        turn,
                        record,
                        seat.last_usage,
                        share=seat.share.share(now),
                        target=cfg.colour_share_target,
                    ),
                )
            )
            out.turns.append(record)
        now += tick

    # Two colour seats on one channel is one seat too many. A trace made by
    # the old runtime carries the silence-timer analyst's beats, and leaving
    # them beside this seat's turns would have the judge score both as "the
    # second voice" — one of them being the thing this replaces. They only
    # go when this seat actually said something, because a trace where it
    # stayed quiet is a trace whose only colour is the old lines, and
    # deleting those would leave it with none. The ``analyst`` rows always
    # stay: they are the record of what it said.
    base = _without_the_old_analyst(rows) if out.spoken else list(rows)
    out.rows = _merge(base, additions)
    return out


def _room_for(now: float, beats: Sequence[tuple[float, float]], cfg: ColourConfig) -> int:
    """How many utterances fit in the hole in the lead's cadence.

    ``max_utterances`` asked for, inside ``turn_span_s``, and however many of
    them clear his beats is the answer. Real commentary's colour voice comes
    in when the ball is dead and the lead has stopped, and on a trace where
    the lead speaks every four seconds and takes two of them to say it there
    is simply nowhere to stand — so fewer than ``min_utterances`` means no
    turn at all, and the rest is what the model is asked for.
    """
    return len(
        space_out(
            now,
            cfg.max_utterances,
            gap=cfg.utterance_gap_s,
            avoid=beats,
            clear=cfg.clear_of_caller_s,
            span=cfg.turn_span_s,
        )
    )


def _schedule(
    record: ColourTurnRecord,
    turn: ColourTurn,
    *,
    seat: ColourSeat,
    gate: FactGate,
    state: MatchState | None,
    pack: KnowledgePack | None,
    cfg: ColourConfig,
    beats: Sequence[tuple[float, str]],
    beat_ts: Sequence[tuple[float, float]],
    lag: float,
    goal_in_state: Callable[[float], bool],
    additions: list[tuple[float, dict[str, Any]]],
    threads: Threads,
) -> None:
    """Place the turn's utterances in time, judge each one, and write the rows.

    Every utterance goes through the fact gate exactly as a phrased caller
    line does: it is wrapped in a form with no sightings on it, so the roster
    check judges the names it says, ``note_claim`` judges any number, and
    ``score_claim`` strikes out a scoreline — which the prompt already
    forbids and the gate is what enforces.

    A line that passes and used one of the notes in :attr:`ColourSeat.notes`
    is recorded against ``threads`` exactly as a phrased caller line is, in
    :mod:`commentary.rephrase`: the hook that turns a list of facts into a
    thread does not care which seat said the fact.
    """
    when = space_out(
        record.ts,
        len(turn.utterances),
        gap=cfg.utterance_gap_s,
        avoid=beat_ts,
        clear=cfg.clear_of_caller_s,
        span=cfg.turn_span_s,
        lengths=[speaking_for(text) for text in turn.utterances],
    )
    for text in turn.utterances[len(when) :]:
        # The lead kept talking and the turn ran out of room. Recorded rather
        # than dropped silently: a turn that loses half itself to the lead is
        # a turn that should not have been offered.
        record.utterances.append(
            ColourUtterance(
                ts=record.ts + cfg.turn_span_s,
                text=text,
                situation=record.situation,
                passed=False,
                reasons=("pushed_out: the lead was still talking and the turn ended",),
            )
        )
    attributed = seat.attributed(record.ts)
    # Everything that put a name in front of the seat this turn: the material
    # block, which already carries the EVENT and REPLAY lines, and the lead's
    # recent lines. A name in none of them is a name the seat reached for.
    sources = [*seat.last_material.lines(), *seat.lead_lines[-LEAD_ECHO_LINES:]]
    # The utterance of this turn that has actually gone out, and the one man
    # it named. Both are what a listener has in their head when the next one
    # arrives two and a half seconds later, so both are read off what passed
    # the gate rather than off what the model wrote: an utterance nobody
    # heard is no antecedent for "he".
    #
    # ``spoken`` is what lets a continuation carry a pronoun at all
    # (:func:`is_filler`); ``referent`` is who that pronoun is, for the
    # attribution check — "Back in and he's just conceded the penalty" names
    # nobody and blames Upamecano.
    spoken = ""
    referent = ""
    for at, text in zip(when, turn.utterances, strict=False):
        verdict = judge_utterance(
            text,
            state if state is not None else MatchState(home="Home", away="Away"),
            pack,
            gate,
            goal_in_state=goal_in_state(at),
            at=at,
            # Its own history, updated inside this loop rather than at the
            # end of it, so a turn that says the same thing twice is caught
            # on its second utterance and not on its next turn.
            said_before=seat.history,
            attributed=attributed,
            named_before=[referent] if referent else (),
            after=spoken,
            lead_said=seat.lead_lines,
            only_repeated=seat.last_material.only_a_count,
            sources=sources,
        )
        utterance = ColourUtterance(
            ts=at,
            text=text,
            situation=record.situation,
            passed=verdict.passed,
            reasons=tuple(verdict.reasons),
            after=_lead_before(beats, at),
            referent=referent,
        )
        record.utterances.append(utterance)
        if verdict.passed:
            spoken = verdict.line
            # A continuation that names nobody keeps the man it inherited —
            # "Morris is the man. / He's the man here. / And he has done it
            # again." is one subject over three utterances — and one that
            # names two people hands on nobody.
            referent = one_subject(verdict.line, pack) or (
                referent if not _names_anybody(verdict.line, pack) else ""
            )
            seat.accept([verdict.line], ts=at)
            seat.spoke_colour(at)
            threads.said(verdict.line, ts=at, pack=pack)
            additions.append((at, _beat_row(at, verdict.line, record.situation, lag)))
        else:
            additions.append(
                (
                    at,
                    {
                        "topic": Topic.GATE.value,
                        "ts": at,
                        "passed": False,
                        "reasons": verdict.reasons,
                        "line": text,
                        "event": Event.NONE.value,
                        "where": "colour",
                    },
                )
            )


def _beat_row(ts: float, text: str, situation: str, lag: float) -> dict[str, Any]:
    """A colour utterance as a beat the replay and the register both read.

    ``event`` is ``none`` and ``preemptable`` is true, which together are the
    rule that the colour voice never speaks over an action call: the director
    drops a preemptable beat the moment a goal, a penalty or a card arrives.
    """
    return {
        "topic": Topic.BEAT.value,
        "ts": ts,
        "id": f"c{int(ts * 100)}",
        "voice": Voice.ANALYST.value,
        "text": text,
        "video_ts": ts,
        "created_ts": 0.0,
        "live_ts": ts + lag,
        "event": Event.NONE.value,
        "urgency": 0.0,
        "excitement": EXCITEMENT.get(situation, 0.2),
        "triggers": [],
        "preemptable": True,
    }


def _colour_row(
    ts: float,
    offer: Offer,
    turn: ColourTurn | None,
    record: ColourTurnRecord,
    usage: Usage,
    *,
    share: float | None = None,
    target: float = ColourConfig.colour_share_target,
) -> dict[str, Any]:
    """One ``colour`` trace row per turn, spoken or not.

    ``share`` is the running lead:colour split as this turn was offered,
    and ``target`` what it is being held to, so the register can print the
    ratio the governor was working against rather than only the one the
    whole trace ended on. ``None`` means the window was too thin to say.
    """
    return {
        "topic": Topic.COLOUR.value,
        "ts": ts,
        "share": None if share is None else round(share, 4),
        "share_target": round(target, 4),
        "situation": offer.situation,
        "reason": offer.reason,
        "speak": record.spoke,
        "material": list(record.material),
        "angle": record.angle,
        "cites": list(record.cites),
        "utterances": [u.text for u in record.utterances],
        "scheduled": [round(u.ts, 3) for u in record.utterances],
        "refused": [
            {"ts": round(u.ts, 3), "line": u.text, "reasons": list(u.reasons)}
            for u in record.utterances
            if not u.passed
        ],
        "usd": round(usage.cost_usd, 6),
        "tokens_in": usage.input_tokens,
        "cache_read": usage.cache_read_tokens,
        "cache_write": usage.cache_write_tokens,
    }


# -- reading the trace ---------------------------------------------------


def _without_the_old_analyst(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every row except the last colour pass's, and the old analyst's beats.

    Both, because a trace is usually run through this pass more than once.
    The beats went from the start — two colour voices on one channel is one
    too many — but the ``colour`` rows did not, so a twice-passed trace
    carried both passes' turns and the second reader of it spent ten minutes
    working out why the seat had apparently spoken twice at the same instant.
    The ``analyst`` rows stay: they are the record of what the thing this
    replaces actually said.
    """
    return [
        row
        for row in rows
        if row.get("topic") != Topic.COLOUR.value
        and not (row.get("topic") == Topic.BEAT.value and row.get("voice") == Voice.ANALYST.value)
    ]


def _states(rows: Sequence[dict[str, Any]]) -> list[tuple[float, MatchState]]:
    found: list[tuple[float, MatchState]] = []
    for row in rows:
        if row.get("topic") != Topic.STATE.value:
            continue
        payload = {k: v for k, v in row.items() if k not in ("topic", "ts")}
        try:
            found.append((float(row.get("ts", 0.0)), MatchState.model_validate(payload)))
        except Exception:
            continue
    return found


def _state_at(states: Sequence[tuple[float, MatchState]], ts: float) -> MatchState | None:
    found: MatchState | None = None
    for row_ts, state in states:
        if row_ts <= ts:
            found = state
    return found


def _forms(rows: Sequence[dict[str, Any]]) -> list[tuple[float, CallerLine]]:
    found: list[tuple[float, CallerLine]] = []
    for row in rows:
        if row.get("topic") != Topic.CALLER.value:
            continue
        payload = {k: v for k, v in row.items() if k not in ("topic", "ts")}
        try:
            found.append((float(row.get("ts", 0.0)), CallerLine.model_validate(payload)))
        except Exception:
            continue
    return found


def _lead_beats(rows: Sequence[dict[str, Any]]) -> list[tuple[float, str]]:
    """Every beat the lead actually got out, in cursor order."""
    found: list[tuple[float, str]] = []
    for row in rows:
        if row.get("topic") != Topic.BEAT.value:
            continue
        if row.get("voice") != Voice.CALLER.value:
            continue
        text = str(row.get("text", "")).strip()
        if text:
            found.append((float(row.get("ts", 0.0)), text))
    found.sort(key=lambda item: item[0])
    return found


def _lead_before(beats: Sequence[tuple[float, str]], ts: float) -> str:
    """The last thing the lead said before this instant, for the table."""
    found = ""
    for at, text in beats:
        if at <= ts:
            found = text
        else:
            break
    return found


def _lag(rows: Sequence[dict[str, Any]], fallback: float) -> float:
    """How far the live edge sat ahead of the cursor on this run.

    Taken from the trace rather than assumed: it is the buffer depth plus
    whatever the model took, and ``replay`` schedules on ``live_ts``. The
    median over the lead's beats, so one slow call does not move it.
    """
    gaps = [
        float(row.get("live_ts", 0.0)) - float(row.get("ts", 0.0))
        for row in rows
        if row.get("topic") == Topic.BEAT.value and row.get("live_ts") is not None
    ]
    kept = [gap for gap in gaps if gap > 0]
    return _median(kept) if kept else fallback


def _merge(
    rows: list[dict[str, Any]], additions: Sequence[tuple[float, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Put the new rows into the trace without resorting the old ones.

    A trace is append-only and is **not** sorted by timestamp: the board
    reader publishes a read at cursor 2.0 while a caller call that started at
    cursor 0.27 is still in flight, so a file routinely steps backwards.
    Sorting the whole thing to insert a dozen rows would reorder what the
    replay and the eval both read positionally, so each addition is placed
    after the last original row stamped at or before it and nothing else
    moves.

    The first version of this walked the file once and flushed whenever the
    next row was later than a pending addition, which assumed the file went
    forwards. On the real Mbappé trace that put a colour beat stamped 6.2 s
    ahead of the caller beat stamped 0.27 s, fifty-five rows out of order.
    """
    pending = sorted(additions, key=lambda item: item[0])
    # Where each addition belongs: after every original row at or before it.
    at: list[int] = []
    for ts, _row in pending:
        last = 0
        for index, row in enumerate(rows):
            if float(row.get("ts", 0.0)) <= ts:
                last = index + 1
        at.append(last)

    out: list[dict[str, Any]] = []
    cursor = 0
    for index, row in enumerate(rows):
        while cursor < len(pending) and at[cursor] == index:
            out.append(pending[cursor][1])
            cursor += 1
        out.append(row)
    out.extend(row for _, row in pending[cursor:])
    return out
