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
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from commentary.agents.analyst import restates_score
from commentary.agents.caller import clean_line, trim_words
from commentary.bus import Topic
from commentary.config import SETTINGS, ColourConfig, Settings
from commentary.gate import FactGate
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
    ) -> None:
        self.backend = backend
        self.config = config or ColourConfig()
        self.pack = pack
        self.model = model if model is not None else self.config.model
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
        #: Forms filed since the last turn, which is what the prompt shows.
        self._since_turn: list[FormAt] = []
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
        """May the seat be asked at this instant? Cheap; no model call."""
        return may_speak(self.moment(now), self.config)

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
        """The pack's notes about the people this passage has been about.

        Same ordering rule as the phraser's: the names the caller could read,
        most recent first, then both teams at the end where they lose to
        anything more specific.
        """
        names: list[str] = []
        for form in reversed(self._since_turn or self._forms[-6:]):
            names.extend(name for name in form.names if name)
            if form.team:
                names.append(form.team)
        if self.pack is not None:
            names.extend([self.pack.home.name, self.pack.away.name])
        return notes_for(self.pack, names)

    async def turn(self, offer: Offer, state_summary: str) -> ColourTurn | None:
        """Ask for one turn. ``None`` means the call failed, not silence."""
        self.last_reason = ""
        self.last_usage = Usage()
        blocks = colour_blocks(
            offer.situation,
            offer.reason,
            state_summary,
            self.lead_lines,
            [form.rendered() for form in self._since_turn[-6:]],
            self.notes(),
            self.said,
            self._last_angle,
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
        return self._settle(parsed.value, 1 if offer.reaction else self.config.max_utterances)

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
    seat = ColourSeat(backend, config=cfg, pack=pack, model=model)
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

        offer = seat.offer(now)
        if offer.allowed:
            moment = seat.moment(now)
            state = _state_at(states, now)
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
) -> None:
    """Place the turn's utterances in time, judge each one, and write the rows.

    Every utterance goes through the fact gate exactly as a phrased caller
    line does: it is wrapped in a form with no sightings on it, so the roster
    check judges the names it says, ``note_claim`` judges any number, and
    ``score_claim`` strikes out a scoreline — which the prompt already
    forbids and the gate is what enforces.
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
    """Every row except the beats the silence-timer analyst spoke."""
    return [
        row
        for row in rows
        if not (row.get("topic") == Topic.BEAT.value and row.get("voice") == Voice.ANALYST.value)
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
