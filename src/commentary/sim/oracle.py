"""A backend that answers from the sim, and lies on request.

The offline pipeline needs something to stand where Claude stands when there
is no key. A handler that simply returned the sim's state would be cheating,
because it would know things no vision model could know. So this one is held
to the same rules as the real backend: it is handed content blocks and
nothing else, and it recovers which moment it is looking at by decoding the
timestamp burned into the picture. If the frames stop carrying it, the oracle
fails rather than guessing, exactly as the real backend fails on a refusal.

Being wrong is the other half of the job, and it comes in two kinds that the
results table has to be able to tell apart.

``error_rate`` is hallucination: a name off the roster, a scoreline that
contradicts the board, a goal nobody scored. The fact gate exists to stop
exactly these, and it cannot be shown stopping them by a model that is right
by construction. None of it has anything to do with how much of the match the
caller was shown, so the rate does not move with the buffer depth.

``outcome_guess_error`` is the failure the delay buffer was built for.
A caller watching a shot leave a boot does not know whether it is a goal. If
the prompt's lookahead frames reach past the moment it resolves, this oracle
reads the answer off them and is right; if they do not, it guesses and is
wrong this often. That is the only thing in this file allowed to depend on
the delay, and it is the mechanism the headline chart is measuring.
"""

from __future__ import annotations

import base64
import random
from dataclasses import dataclass, field
from typing import TypeVar

import cv2
import numpy as np
from pydantic import BaseModel

from commentary.llm.base import Block, LLMError, Parsed, Usage
from commentary.schemas import (
    AnalystLine,
    Angle,
    BoardRead,
    CallerLine,
    Event,
    Scene,
)
from commentary.sim.match import MatchSim, Outcome, SimState
from commentary.sim.render import decode_ts

T = TypeVar("T", bound=BaseModel)

#: The three ways a caller goes wrong that the fact gate is built to stop.
#: These are hallucinations: they have nothing to do with how much the caller
#: could see, so they stay at the same rate however deep the buffer is.
ERROR_KINDS: tuple[str, ...] = ("fake_name", "wrong_score", "phantom_goal")

#: The other failure, and the one the delay is supposed to fix: committing to
#: how a move ends before the picture has shown it. Recorded separately so the
#: eval can tell a caller that invented a person from one that called a save
#: a goal.
OUTCOME_KIND = "guessed_outcome"
INJECTED_KINDS: tuple[str, ...] = (*ERROR_KINDS, OUTCOME_KIND)

#: How the caller prompt announces its lookahead frames. Matched on the words
#: rather than the whole sentence, because that file is not ours.
_FUTURE_MARKER = "near future"

#: Fraction of unresolved moments a caller with no lookahead calls wrongly.
#: This is a modelling assumption, not a measurement. A coin flip on goal
#: against save would be 0.5; this sits a little under it because the picture
#: carries some signal, and well above the base rate a caller would get by
#: always saying "saved", because a model asked to narrate leans towards the
#: dramatic reading. GetStream measured frame narration wrong more than half
#: the time with no lookahead at all, which is the number this is anchored to.
DEFAULT_OUTCOME_GUESS_ERROR = 0.45

#: What a caller says instead when it guesses the ending wrong.
_WRONG_OUTCOME: dict[Event, Event] = {
    Event.GOAL: Event.SAVE,
    Event.SAVE: Event.GOAL,
    Event.PENALTY: Event.GOAL,
    Event.NONE: Event.GOAL,
}

_OUTCOME_LINES: dict[Event, tuple[str, ...]] = {
    Event.GOAL: (
        "Goal! {who} finds the corner and the net bulges.",
        "That is a goal, {who} gets it and this place goes up.",
    ),
    Event.SAVE: (
        "Saved! The keeper gets across and smothers it.",
        "Held by the keeper, and the chance is gone.",
    ),
    Event.NONE: (
        "{who} looks for the shot, and it comes to nothing.",
        "Half a chance for {team}, and it fizzles out.",
    ),
}

#: Invented people, used only when lying. Checked against the rosters first.
_OFF_ROSTER = (
    "Ronan Velasquez",
    "Kaito Fairweather",
    "Milan Osterhagen",
    "Dexter Almeida",
)

_QUIET_SCENES = frozenset({Scene.REPLAY, Scene.GRAPHIC})
_WORTH_SAYING = frozenset(
    {
        Event.GOAL,
        Event.SHOT,
        Event.SAVE,
        Event.CARD,
        Event.PENALTY,
        Event.CORNER,
        Event.SUBSTITUTION,
        Event.KICKOFF,
        Event.FOUL,
    }
)

_LINES: dict[Event, tuple[str, ...]] = {
    Event.KICKOFF: ("And we are under way again.", "{team} get us going."),
    Event.BUILD_UP: (
        "{team} working it across the back line.",
        "Patient from {team}, probing for a way in.",
        "{team} keep it, looking for the angle.",
    ),
    Event.SHOT: (
        "{who} lets fly from the edge of the box!",
        "That is a strike from {who}, and it is on target!",
    ),
    Event.SAVE: ("Saved, a strong hand from {who}.", "{who} gets down well to keep it out."),
    Event.GOAL: (
        "Goal! {who} finishes it off and the place erupts.",
        "That is a goal, {who} buries it low to the near post.",
    ),
    Event.CORNER: ("Corner to {team}, swung in.", "{team} have it back from the corner."),
    Event.FOUL: ("Whistle goes, free kick {team}.", "That is a foul, and play stops."),
    Event.FREE_KICK: ("{team} have it, free kick in their own half.", "Restart for {team}."),
    Event.CARD: ("Into the book goes {who}.", "Yellow card, {who} has to be careful now."),
    Event.SUBSTITUTION: ("Change for {team}, {who} comes on.", "{who} is on for {team}."),
    Event.OFFSIDE: ("Flag is up, offside.", "Offside against {team}, and the whistle goes."),
    Event.THROW_IN: ("Throw-in {team}.", "{team} take it quickly on the touchline."),
}

_ANALYST_LINES: dict[Angle, str] = {
    Angle.TACTICS: (
        "{home} are pushing both full-backs high, and that is why the space "
        "keeps opening up in behind."
    ),
    Angle.FORM: (
        "Worth remembering {away} arrive in good form, and they have carried "
        "that into this half."
    ),
    Angle.MOMENTUM: (
        "The last ten minutes have been one-way traffic, and {team} look like they know it."
    ),
    Angle.PLAYER: (
        "{who} has been the difference here, dropping off the front and turning every time."
    ),
    Angle.STAKES: (
        "A result here settles the group, so neither of these sides can afford to sit in."
    ),
    Angle.HISTORY: (
        "These two have met three times this season, and every one of them has been tight."
    ),
}


@dataclass(frozen=True)
class Sighting:
    """What one caller call could and could not see, kept for the eval.

    The delay chart is an argument about information, so the argument is
    easier to check if the information itself is recorded rather than inferred
    from the lines afterwards. ``pending`` says the moment was still in the
    balance; ``knew`` says the lookahead reached far enough to settle it.
    """

    cursor_ts: float
    horizon_s: float
    pending: bool
    knew: bool


@dataclass(frozen=True)
class Moment:
    """The two instants one prompt is about.

    ``cursor_ts`` is what the line is describing and what the runtime stamps
    it with; ``live_ts`` is the furthest ahead the prompt let the caller see.
    The gap between them is the delay the whole experiment sweeps.
    """

    cursor_ts: float
    live_ts: float

    @property
    def horizon_s(self) -> float:
        return max(0.0, self.live_ts - self.cursor_ts)


@dataclass
class SimOracle:
    """An :class:`LLMBackend` that reads the sim off the picture it is given.

    Two failure modes, kept apart on purpose, because the delay experiment
    only means something if they can be told apart in the results.

    ``error_rate`` is hallucination: invented names, scorelines that
    contradict the board, goals that never happened. It has nothing to do
    with how much of the match the caller was shown, so it stays flat however
    deep the buffer is. A phantom goal is not injected at a moment when a
    goal genuinely happened, since that would not be an error; one of the
    other two kinds is used instead.

    ``outcome_guess_error`` is the failure the delay exists to fix. When the
    prompt's lookahead frames reach past the moment a move resolves, this
    oracle knows how it ended and says so. When they do not, it has to guess,
    and it guesses wrongly this often: a save called as a goal, a goal called
    as a save, a chance sold that fizzled out. Those are recorded under
    :data:`OUTCOME_KIND` rather than as hallucinations.
    """

    sim: MatchSim
    error_rate: float = 0.0
    outcome_guess_error: float = DEFAULT_OUTCOME_GUESS_ERROR
    seed: int = 17
    injected: list[tuple[float, str]] = field(default_factory=list)
    sightings: list[Sighting] = field(default_factory=list)
    calls: int = 0
    _total: Usage = field(default_factory=Usage)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")
        if not 0.0 <= self.outcome_guess_error <= 1.0:
            raise ValueError("outcome_guess_error must be between 0 and 1")
        self._rng = random.Random(self.seed)

    @property
    def total(self) -> Usage:
        return self._total

    @property
    def injected_kinds(self) -> dict[str, int]:
        counts = {kind: 0 for kind in INJECTED_KINDS}
        for _, kind in self.injected:
            counts[kind] += 1
        return counts

    async def parse(
        self,
        *,
        model: str,
        system: str,
        blocks: list[Block],
        output_format: type[T],
        max_tokens: int = 1024,
        effort: str | None = None,
        cache_system: bool = True,
        tag: str = "",
    ) -> Parsed[T]:
        moment = self.moment_from(blocks)
        if moment is None:
            raise LLMError(
                "no decodable timestamp in the frames: the oracle only knows what it can see"
            )
        self.calls += 1
        value = self._answer(tag, output_format, moment)
        if not isinstance(value, output_format):
            raise LLMError(f"oracle cannot produce {output_format.__name__} for tag {tag!r}")
        usage = Usage(
            input_tokens=_rough_tokens(blocks),
            output_tokens=64,
            latency_s=0.0,
        )
        self._total = self._total + usage
        return Parsed(value=value, usage=usage, model=model)

    # ------------------------------------------------------------ reading

    @staticmethod
    def moment_from(blocks: list[Block]) -> Moment | None:
        """The two instants a prompt is asking about, recovered from the pictures.

        The caller is sent the frames it is calling and then, behind a heading
        announcing the near future, one or two frames from nearer the live
        edge. Taking the last image in the prompt would therefore answer about
        the live edge while the runtime files the line under the cursor, and
        every line would describe something several seconds after the moment
        it is stamped with. So the cursor is the last image *before* that
        heading, and the live edge is the last image of all.

        The heading is matched loosely, on the words rather than the sentence,
        because the prompt is somebody else's file and will be reworded. When
        there is no heading at all — a bare score-bug crop, or the analyst,
        which gets no lookahead — the last image is both.
        """
        cursor: float | None = None
        latest: float | None = None
        future = False
        for block in blocks:
            kind = block.get("type")
            if kind == "text":
                if _FUTURE_MARKER in str(block.get("text", "")).lower():
                    future = True
                continue
            if kind != "image":
                continue
            ts = _image_ts(block)
            if ts is None:
                continue
            latest = ts
            if not future:
                cursor = ts
        if latest is None:
            return None
        if cursor is None:
            cursor = latest
        return Moment(cursor_ts=cursor, live_ts=max(cursor, latest))

    # ------------------------------------------------------------ answers

    def _answer(self, tag: str, output_format: type[BaseModel], moment: Moment) -> BaseModel:
        state = self.sim.at(moment.cursor_ts)
        if tag == "board" or output_format is BoardRead:
            return self._board(state)
        if tag == "analyst" or output_format is AnalystLine:
            return self._analyst(state)
        return self._caller(state, moment)

    def _board(self, state: SimState) -> BoardRead:
        if state.in_replay:
            return BoardRead(bug_visible=False, confidence=0.92)
        return BoardRead(
            bug_visible=True,
            home_score=state.home_score,
            away_score=state.away_score,
            clock=state.clock,
            confidence=0.96,
        )

    def _caller(self, state: SimState, moment: Moment) -> CallerLine:
        rng = self._moment_rng(state.ts)
        pack = self.sim.knowledge_pack
        team = pack.team(state.possession)
        event = state.event
        recent = self.sim.events_near(state.ts, 2.5)
        truth_event = recent[-1].event if recent else event

        who = self._who(state)
        names = self._names_read(state, rng)
        speak = state.scene not in _QUIET_SCENES and (
            event in _WORTH_SAYING or rng.random() < 0.4
        )
        line = self._render_line(event, state, who, rng)

        outcome = self.sim.outcome_at(state.ts)
        self.sightings.append(
            Sighting(
                cursor_ts=state.ts,
                horizon_s=moment.horizon_s,
                pending=outcome.pending,
                knew=outcome.known_by(moment.live_ts),
            )
        )
        if outcome.pending and state.scene not in _QUIET_SCENES:
            event, line, wrong = self._call_the_outcome(outcome, moment, state, who, rng)
            speak = True
            if wrong:
                self.injected.append((state.ts, OUTCOME_KIND))

        injection = self._pick_injection(truth_event)
        if injection == "fake_name":
            fake = self._fake_name(rng)
            names = [*names, fake]
            line = f"{fake} is all over this, and {line[0].lower()}{line[1:]}"
            speak = True
        elif injection == "wrong_score":
            home, away = state.home_score + 2, state.away_score + 1
            line = f"{line} That makes it {home}-{away}."
            speak = True
        elif injection == "phantom_goal":
            event = Event.GOAL
            line = f"Goal! {who} has scored, and this place erupts."
            speak = True
        if injection is not None:
            self.injected.append((state.ts, injection))

        return CallerLine(
            scene=state.scene,
            event=event,
            side=state.possession,
            team=None if team is None else team.name,
            names_read=names,
            confidence=round(rng.uniform(0.62, 0.94), 2),
            speak=speak,
            line=line[:200] if speak else "",
        )

    def _call_the_outcome(
        self,
        outcome: Outcome,
        moment: Moment,
        state: SimState,
        who: str,
        rng: random.Random,
    ) -> tuple[Event, str, bool]:
        """Commit to how this move ends, knowing or guessing.

        The caller prompt tells the model in as many words to use the later
        frames to decide what to say, so when they reach past the resolution
        this reports what actually happened. When they do not, there is
        nothing to read it off and the answer is a guess, wrong
        ``outcome_guess_error`` of the time. That is the entire mechanism the
        delay-versus-error chart is measuring, so it is the one thing in this
        file that must depend on how far ahead the prompt could see.
        """
        called = outcome.event
        wrong = False
        if not outcome.known_by(moment.live_ts) and self._rng.random() < self.outcome_guess_error:
            called = _WRONG_OUTCOME.get(outcome.event, Event.GOAL)
            wrong = True
        options = _OUTCOME_LINES.get(called, _OUTCOME_LINES[Event.NONE])
        pack = self.sim.knowledge_pack
        team = pack.team(state.possession)
        line = options[rng.randrange(len(options))].format(
            who=who,
            team=pack.home.name if team is None else team.name,
        )
        return (called if called is not Event.NONE else Event.SHOT), line, wrong

    def _analyst(self, state: SimState) -> AnalystLine:
        rng = self._moment_rng(state.ts + 0.5)
        pack = self.sim.knowledge_pack
        angle = rng.choice(list(Angle))
        team = pack.team(state.possession)
        line = _ANALYST_LINES[angle].format(
            home=pack.home.name,
            away=pack.away.name,
            team=pack.home.name if team is None else team.name,
            who=self._who(state),
        )
        cites = [pack.storylines[rng.randrange(len(pack.storylines))]] if pack.storylines else []
        if angle is Angle.FORM and pack.form:
            cites.append(f"{pack.away.short} form {pack.form.get(pack.away.short, '')}".strip())
        return AnalystLine(
            angle=angle,
            cites=cites,
            confidence=round(rng.uniform(0.55, 0.9), 2),
            speak=state.scene is not Scene.REPLAY,
            line=line[:280],
        )

    # ------------------------------------------------------------ helpers

    def _moment_rng(self, ts: float) -> random.Random:
        """One stream per instant, so the same frame always reads the same way."""
        return random.Random(self.seed * 1_000_003 + int(round(ts * 1000.0)))

    def _who(self, state: SimState) -> str:
        if state.graphic is not None:
            return state.graphic.surname
        candidates = [d for d in state.players if d.side is state.possession]
        if not candidates:
            return "the man in possession"
        ball = state.ball
        nearest = min(candidates, key=lambda d: (d.x - ball[0]) ** 2 + (d.y - ball[1]) ** 2)
        return f"number {nearest.number}"

    def _names_read(self, state: SimState, rng: random.Random) -> list[str]:
        """Only what a camera could actually have shown: a graphic, or numbers."""
        if state.graphic is not None:
            return [state.graphic.name]
        near = sorted(
            (d for d in state.players if state.zoom > 1.1 or d.side is state.possession),
            key=lambda d: (d.x - state.ball[0]) ** 2 + (d.y - state.ball[1]) ** 2,
        )[: 1 + rng.randrange(2)]
        return [str(d.number) for d in near]

    def _render_line(self, event: Event, state: SimState, who: str, rng: random.Random) -> str:
        pack = self.sim.knowledge_pack
        team = pack.team(state.possession)
        options = _LINES.get(event, _LINES[Event.BUILD_UP])
        return options[rng.randrange(len(options))].format(
            team=pack.home.name if team is None else team.name,
            who=who,
            home=pack.home.short,
            away=pack.away.short,
            h=state.home_score,
            a=state.away_score,
        )

    def _fake_name(self, rng: random.Random) -> str:
        roster = self.sim.roster_names
        options = [n for n in _OFF_ROSTER if n not in roster]
        return options[rng.randrange(len(options))]

    def _pick_injection(self, truth_event: Event) -> str | None:
        if self.error_rate <= 0.0 or self._rng.random() >= self.error_rate:
            return None
        kind = self._rng.choice(ERROR_KINDS)
        if kind == "phantom_goal" and truth_event is Event.GOAL:
            return self._rng.choice(("fake_name", "wrong_score"))
        return kind


def _image_ts(block: Block) -> float | None:
    """The video time burned into one image block, or ``None`` if unreadable."""
    data = block.get("source", {}).get("data")
    if not isinstance(data, str):
        return None
    raw = np.frombuffer(base64.standard_b64decode(data), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        return None
    return decode_ts(np.asarray(image))


def _rough_tokens(blocks: list[Block]) -> int:
    """A believable input-token count, so cost accounting has something to add."""
    total = 0
    for block in blocks:
        if block.get("type") == "image":
            total += 780
        else:
            total += len(str(block.get("text", ""))) // 4
    return total
