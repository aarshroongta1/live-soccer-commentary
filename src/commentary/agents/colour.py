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
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from commentary.agents.analyst import restates_score
from commentary.agents.caller import clean_line, trim_words
from commentary.bus import Topic
from commentary.config import SETTINGS, ColourConfig, Settings
from commentary.gate import FactGate, fold
from commentary.llm.base import LLMBackend, LLMError, Usage
from commentary.prompts.colour import colour_blocks, colour_system, form_line, opens_with_a_cue
from commentary.schemas import (
    CallerLine,
    ColourTurn,
    Event,
    GateVerdict,
    KnowledgePack,
    MatchState,
    Note,
    Scene,
    Voice,
)
from commentary.state import notes_for
from commentary.tallies import Tallies
from commentary.threads import Threads
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

#: The events whose first twelve seconds belong to the lead. Exactly the four
#: the brief names: a goal, a shot, a save, a penalty. A card is not here —
#: the corpus's median delay after a yellow is 17.0 s but 29% of colour
#: entries land inside six seconds of one, and a card is a stoppage, which is
#: where this seat is meant to speak.
HELD_FOR_THE_LEAD = frozenset({Event.GOAL, Event.SHOT, Event.SAVE, Event.PENALTY})

#: What counts as "the last big event" for the twenty-second clause: the four
#: above plus the card, which is the study's "foul-with-card".
BIG_MOMENTS = HELD_FOR_THE_LEAD | {Event.CARD}

#: The four example groups in :mod:`commentary.prompts.colour`, which are
#: also the four situations the gate can hand back.
AFTER_A_GOAL = "after a goal"
AFTER_A_CHANCE = "after a chance"
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

    @property
    def quiet(self) -> bool:
        """Is the ball dead, or is the picture something other than the play?"""
        return self.scene in QUIET_SCENES or self.event in DEAD_BALL_EVENTS

    def rendered(self) -> str:
        return form_line(self.scene.value, self.event.value, self.team, self.names, self.said)


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
    #: The last goal, shot, save, penalty or card, and when it was.
    last_big: tuple[Event, float] | None = None
    #: How many lead lines have gone out at all, ever.
    lead_lines: int = 0
    #: How many lead lines have gone out since ``last_big``.
    lead_lines_since_big: int = 0
    #: When the seat last answered an offer, spoken or not.
    last_turn_ts: float | None = None
    #: How many turns it has taken since ``last_big``.
    turns_since_big: int = 0

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
       build-up, one turn per ``min_gap_s``.
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
            if moment.turns_since_big >= 1:
                return Offer(
                    False,
                    reason=(
                        "one turn per big event, and this one has already had "
                        f"{moment.turns_since_big}"
                    ),
                )
        elif gap < cfg.min_gap_s:
            return Offer(
                False,
                reason=(
                    f"only {gap:.1f} s since your last turn, and the gap is "
                    f"{cfg.min_gap_s:.0f} s"
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
    """Which of the four example groups this moment belongs to."""
    big = moment.last_big
    if big is not None and moment.since_big <= SITUATION_WINDOW_S:
        return AFTER_A_GOAL if big[0] is Event.GOAL else AFTER_A_CHANCE
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
) -> list[float]:
    """When each utterance of one turn goes out, and where the turn stops.

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
    for _ in range(max(0, count)):
        if out:
            at = max(at, out[-1] + gap)
        at = _clear_of(at, blocked, clear)
        if span is not None and at > start + span:
            break
        out.append(at)
        at += gap
    return out


def _clear_of(ts: float, blocked: Sequence[tuple[float, float]], clear: float) -> float:
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
        if beat - clear < at < beat + seconds + clear:
            at = beat + seconds + clear
    return at


#: Number words, for the rule that this seat does not say numbers. Ordinals
#: and the spelled-out decades are here because a commentator says "eighty
#: six" and "the first", not "86" and "1st".
_NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    "fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    "eighty|ninety|hundred|thousand|dozen|"
    "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    "once|twice|double|treble|hat-trick"
)
_A_NUMBER = re.compile(rf"\b(?:\d+|{_NUMBER_WORDS})\b", re.IGNORECASE)


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
    """
    return bool(_A_NUMBER.search(text))


def judge_utterance(
    text: str,
    state: MatchState,
    pack: KnowledgePack | None,
    gate: FactGate,
    *,
    goal_in_state: bool = True,
    at: float | None = None,
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

    Then :func:`is_filler`, which is the judge's 5.0 turned into a predicate:
    an utterance that names nobody, names no side doing something again and
    names no event is refused as ``colour_filler``. It sits here rather than
    in the prompt because the prompt has asked for it twice and been given
    "I think this is what it comes down to" both times.
    """
    if restates_score(text, {"home_score": state.home_score, "away_score": state.away_score}):
        return GateVerdict(
            passed=False,
            reasons=["scoreline: the colour seat does not read the scoreboard back out"],
            line=text,
        )
    if says_a_number(text):
        return GateVerdict(
            passed=False,
            reasons=["number_claim: numbers are the lead's job, not this seat's"],
            line=text,
        )
    if is_filler(text, pack):
        return GateVerdict(
            passed=False,
            reasons=["colour_filler: names no player, no side doing it again, and no event"],
            line=text,
        )
    return gate.judge(
        CallerLine(
            scene=Scene.STOPPAGE,
            event=Event.NONE,
            confidence=1.0,
            speak=True,
            line=text,
        ),
        state,
        pack,
        board_changed=False,
        goal_in_state=goal_in_state,
        at=at,
    )


#: Events that are something happening rather than the ball moving. A
#: pattern worth remarking on is built out of these; "three passes in a row"
#: is not an observation.
NAMED_EVENTS = frozenset(
    {
        Event.GOAL,
        Event.SHOT,
        Event.SAVE,
        Event.CORNER,
        Event.FREE_KICK,
        Event.PENALTY,
        Event.FOUL,
        Event.OFFSIDE,
        Event.THROW_IN,
        Event.CARD,
        Event.SUBSTITUTION,
    }
)

#: How a count reads in English. The seat is shown this text and a model
#: shown "3 penaltys" writes worse English than one shown "3 penalties".
PLURALS = {
    Event.PENALTY: "penalties",
    Event.OFFSIDE: "offside calls",
    Event.THROW_IN: "throw-ins",
    Event.FREE_KICK: "free kicks",
    Event.SUBSTITUTION: "substitutions",
}

#: How many forms back a pattern may be counted over. A ceiling rather than
#: the window: the forms handed to :func:`patterns_in` are the ones filed
#: since the seat's last turn, which on a forty-five second gap is about a
#: dozen anyway.
PATTERN_FORMS = 12

#: How many times something has to happen before it is a pattern. Two is the
#: floor the study's own examples sit on — "unlike the corners there ...
#: they've gone zonally" is about the second corner, not the fifth — and one
#: is not a pattern, it is the thing that just happened.
PATTERN_FLOOR = 2

#: How old the last big event may be and still be worth an opinion. Past
#: this the moment has gone and an opinion about it is a history lesson;
#: section 4.3's colour entries after a big event have a median delay of
#: 21.4 s, so the window is drawn just past that.
EVENT_FRESH_S = 25.0

#: How many of the lead's lines are read for the names a note may be about.
#: Three, because a note is only worth saying while the man it is about is
#: still the man the listener is thinking about.
LEAD_LINES_FOR_NOTES = 3

#: The two sides of the pitch, which is the only geography the seat can get
#: out of the lead's own words. "France down the left again" is the study's
#: shape; there is no zone on a caller form, so the flank is read off what
#: the lead said while the form was filed.
FLANKS = ("left", "right")

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
        The last goal, shot, save, penalty or card, less than
        :data:`EVENT_FRESH_S` old, **with the player named**. An opinion
        about that is an opinion about something that has finished.

    ``about`` is the one person the turn has to be about, set only for the
    sanctioned reaction after a goal, where the corpus's second voice talks
    about the scorer and nothing else.
    """

    notes: tuple[Note, ...] = ()
    patterns: tuple[str, ...] = ()
    last_event: str = ""
    about: str = ""

    def __bool__(self) -> bool:
        """Is there anything specific enough here to make a turn out of?"""
        return bool(self.notes or self.patterns or self.last_event)

    def lines(self) -> list[str]:
        """The material as the model is shown it: labelled, one item a line.

        A note carrying a figure is marked as carrying one. The seat may not
        say a number and the gate throws the utterance away when it does, so
        "a goal in the 2018 World Cup final at nineteen" came back as
        "Scored in a final at nineteen" and was struck out — a whole
        utterance lost to a rule the model had been told twice, a foot away
        from the thing it was reading.
        """
        figure = " (has a figure in it: say the fact, never the figure)"
        out = [
            f"NOTE about {note.about}"
            + (figure if says_a_number(note.text) else "")
            + f": {note.text}"
            for note in self.notes
        ]
        out.extend(f"REPEATED: {text}" for text in self.patterns)
        if self.last_event:
            out.append(f"EVENT, finished, speak about it in the past tense: {self.last_event}")
        return out


def roster_names(pack: KnowledgePack | None) -> list[str]:
    """Everybody who exists, both squads, starters before bench."""
    if pack is None:
        return []
    return [
        player.name
        for team in (pack.home, pack.away)
        for player in [*team.starters, *team.bench]
        if player.name
    ]


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
    """
    haystack = f" {fold(text)} "
    folded = fold(name)
    if not folded:
        return False
    surname = folded.rsplit(" ", 1)[-1]
    return f" {folded} " in haystack or f" {surname} " in haystack


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


def is_filler(text: str, pack: KnowledgePack | None) -> bool:
    """Is this utterance about nothing?

    Three ways to be about something, and a line needs one of them:

    - it names somebody on a team sheet;
    - it names a side **and** claims a repetition — "France down that side
      again" is an observation, "France keeping it simple" is a guess at a
      picture the seat cannot see;
    - it names an event: a goal, a penalty, a save, a card, a foul, a corner.

    Everything else is the failure the judge quoted: "I think this is what it
    comes down to", "that changes everything", "they know what they are
    protecting". Checked in code because asking the prompt nicely did not
    stop it.
    """
    words = set(fold(text).split())
    if not words:
        return True
    if words & EVENT_WORDS:
        return False
    if any(mentions(text, name) for name in roster_names(pack)):
        return False
    named_side = any(mentions(text, word) for word in team_words(pack))
    return not (named_side and bool(words & REPEATS))


def patterns_in(forms: Sequence[FormAt]) -> list[str]:
    """What has happened twice or more, with a name on it and the count.

    The count is in the string because the prompt has to be able to say "that
    is the third one" to itself in order to write "again"; the rules and
    :func:`says_a_number` between them stop the figure reaching air.

    Occurrences, not forms. The caller files the same penalty on five looks
    in a row and counting forms would have handed the seat "5 penalties",
    which is false and is the kind of false a listener notices. So a run of
    consecutive forms carrying the same thing counts once.

    Three kinds, which are the brief's: the same kind of event happening
    again, the same player coming back into the picture, one side going down
    one flank again. A side simply having the ball is not among them — it was
    the whole of the last pass's material and it produced nothing worth
    hearing.
    """
    # A replay is the same thing again, not another one. Counting them gave
    # "2 goals on Mbappé" off one penalty and its replay, which is false and
    # would have reached air as "again".
    kept = [form for form in forms if form.scene is not Scene.REPLAY][-PATTERN_FORMS:]
    if not kept:
        return []
    found: list[tuple[int, str]] = []

    events = _runs(kept, lambda form: form.event if form.event in NAMED_EVENTS else None)
    for event, count in events:
        if count < PATTERN_FLOOR:
            continue
        kind = PLURALS.get(event, f"{event.value.replace('_', ' ')}s")
        who = _who_runs_through(kept, event)
        found.append(
            (count, f"{count} {kind} on {who}" if who else f"{count} {kind} in this spell")
        )

    people: Counter[str] = Counter()
    for name in {name for form in kept for name in form.names if name}:
        people[name] = sum(count for _key, count in _runs(kept, _in_the_picture(name)))
    for name, count in people.most_common():
        if count >= PATTERN_FLOOR:
            found.append((count, f"{name} in the picture on {count} separate looks"))

    for (team, flank), count in _flank_runs(kept).items():
        if count >= PATTERN_FLOOR:
            found.append((count, f"{team} down the {flank} {count} times in this spell"))

    return [text for _count, text in sorted(found, key=lambda item: -item[0])]


def _runs(forms: Sequence[FormAt], key: Callable[[FormAt], Any]) -> list[tuple[Any, int]]:
    """How many separate times each key turns up, counting a run as one.

    A falsy key breaks the run and is never counted: those are the forms this
    key is not about.
    """
    counts: Counter[Any] = Counter()
    last: Any = None
    for form in forms:
        value = key(form)
        if not value:
            last = None
            continue
        if value != last:
            counts[value] += 1
        last = value
    return counts.most_common()


def _who_runs_through(forms: Sequence[FormAt], event: Event) -> str:
    """The one name on every form of this event, or nothing.

    "the third foul on Otamendi" needs Otamendi to have been on all three of
    them. One name across two fouls is an observation; a different name each
    time is a coincidence.
    """
    seen = [set(form.names) for form in forms if form.event is event and form.names]
    if not seen:
        return ""
    shared = set.intersection(*seen)
    return sorted(shared)[0] if shared else ""


def _flank_runs(forms: Sequence[FormAt]) -> dict[tuple[str, str], int]:
    """How many separate spells each side spent down each flank.

    Read off the lead's own words, because a caller form has a team and a
    scene and no geography at all.
    """
    found: dict[tuple[str, str], int] = {}
    for team in {form.team for form in forms if form.team}:
        for flank in FLANKS:
            count = sum(number for _key, number in _runs(forms, _down_the(team, flank)))
            if count:
                found[(team, flank)] = count
    return found


def _in_the_picture(name: str) -> Callable[[FormAt], Any]:
    """A key that is this player's name on the looks he was legible on."""

    def key(form: FormAt) -> Any:
        return name if name in form.names else None

    return key


def _down_the(team: str, flank: str) -> Callable[[FormAt], Any]:
    """A key that is this flank on the looks this side spent going down it."""

    def key(form: FormAt) -> Any:
        if form.team != team:
            return None
        return flank if flank in fold(form.said).split() else None

    return key


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
        #: Built once, never rebuilt: identical bytes on every turn is what
        #: makes forty real utterances and two squads affordable to send.
        self.system = colour_system(pack, self.config)
        self._forms: list[FormAt] = []
        self._lead: deque[str] = deque(maxlen=max(1, self.config.lead_lines))
        self._said: deque[str] = deque(maxlen=max(1, self.config.lead_lines))
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
        if form.event in BIG_MOMENTS:
            # One goal, not five. The caller files the same goal over several
            # looks and then over the replays, and taking each of them as a
            # new big event reset both the rate cap and the once-per-goal
            # reaction: the last pass took two reaction fragments at the same
            # instant and a third off a replay twenty-four seconds later. So
            # a big event of the same kind inside the freshness window is the
            # same event, and it keeps the timestamp of the look the lead
            # actually called it on.
            again = (
                self._last_big is not None
                and self._last_big[0] is form.event
                and ts - self._last_big[1] <= EVENT_FRESH_S
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
        self._lead_lines += 1
        self._lead_since_big += 1

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

    def accept(self, utterances: Iterable[str]) -> None:
        """Record what actually went out, so it is not said twice."""
        for text in utterances:
            said = text.strip()
            if said:
                self._said.append(said)

    # -- the call ---------------------------------------------------------

    def notes(self) -> list[Note]:
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
        """
        return self.tallies.adjusted(notes_for(self.pack, self.lead_named()))

    def lead_named(self) -> list[str]:
        """Who the lead has named in his last three lines, newest first."""
        found: list[str] = []
        for line in reversed(list(self._lead)[-LEAD_LINES_FOR_NOTES:]):
            for name in roster_names(self.pack):
                if name not in found and mentions(line, name):
                    found.append(name)
        return found

    def material(self, now: float, *, reaction: bool = False) -> Material:
        """The three things the seat may build a turn out of.

        See :class:`Material`. ``reaction`` is the sanctioned fragment after
        a goal, which is not a turn and is about one man.
        """
        if reaction:
            return self._about_the_scorer(now)
        return Material(
            notes=tuple(self.notes()),
            patterns=tuple(patterns_in(self._since_turn)),
            last_event=self._last_completed(now),
        )

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
        """
        names: list[str] = []
        called = ""
        for form in reversed(self._forms):
            if form.event is not event or form.ts > at or at - form.ts > EVENT_FRESH_S:
                continue
            names.extend(name for name in form.names if name and name not in names)
            if form.said:
                called = form.said
        return one_name_each(names), called

    def _last_completed(self, now: float) -> str:
        """The last big thing the lead called, if it is fresh and has a name on it.

        Not the ball now — the seat cannot see the ball now, and the whole of
        the last pass's worst material was the present tense. A goal, a shot,
        a save, a penalty, a card: something with a beginning and an end,
        under :data:`EVENT_FRESH_S` old, with the man it happened to named,
        and the words the lead used for it so that an opinion has something
        to be an opinion about. Anything looser was where "I think this is
        what it comes down to" came from.
        """
        latest: FormAt | None = None
        for form in reversed(self._forms):
            if form.event in BIG_MOMENTS:
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
        most = self._how_many(offer, material)
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
        try:
            parsed = await self.backend.parse(
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
        self.last_usage = parsed.usage
        return self._settle(parsed.value, most)

    def _how_many(self, offer: Offer, material: Material) -> int:
        """How many utterances this turn is allowed to be.

        Section 4.4 has the colour voice holding the microphone for a median
        of four utterances, but that is a voice with a whole match in its
        head. This seat has whatever the material gate let through, and on
        one item a four-utterance turn is one thought and three restatements
        — which is what the judge heard. So: the sanctioned goal reaction is
        one fragment, a single item of material is worth two utterances (the
        thing, then why it matters), and only real material gets the run.
        """
        if offer.reaction:
            return 1
        limit = self.config.max_utterances
        if len(material.lines()) <= 1:
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
                out.append(
                    f"{utterance.ts:>7.1f}  {'':<17}{mark} {utterance.text}{reason}"
                )
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
        return {
            "turns_offered": float(len(self.turns)),
            "turns_spoken": float(len(spoke)),
            "utterances": float(len(spoken)),
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
    seat = ColourSeat(backend, config=cfg, pack=pack, model=model, tallies=threads_local.tallies)
    out = ColourPass(rows=list(rows), quiet_after_big_s=cfg.quiet_after_big_s)
    if not seat.enabled:
        return out

    states = _states(rows)
    forms = _forms(rows)
    beats = _lead_beats(rows)
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
                seat.saw_form(ts, value)
            else:
                seat.saw_lead_line(ts, value)
            cursor += 1

        state = _state_at(states, now)
        if state is not None:
            threads_local.see_state(state)

        offer = seat.offer(now)
        room = _room_for(now, beat_ts, cfg) if offer.allowed else 0
        if offer.allowed and room >= cfg.min_utterances:
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
                "the lead has not stopped talking long enough for two utterances",
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
            additions.append((now, _colour_row(now, offer, turn, record, seat.last_usage)))
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
    said: list[str] = []
    for at, text in zip(when, turn.utterances, strict=False):
        verdict = judge_utterance(
            text,
            state if state is not None else MatchState(home="Home", away="Away"),
            pack,
            gate,
            goal_in_state=goal_in_state(at),
            at=at,
        )
        utterance = ColourUtterance(
            ts=at,
            text=text,
            situation=record.situation,
            passed=verdict.passed,
            reasons=tuple(verdict.reasons),
            after=_lead_before(beats, at),
        )
        record.utterances.append(utterance)
        if verdict.passed:
            said.append(verdict.line)
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
    seat.accept(said)


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
) -> dict[str, Any]:
    """One ``colour`` trace row per turn, spoken or not."""
    return {
        "topic": Topic.COLOUR.value,
        "ts": ts,
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
