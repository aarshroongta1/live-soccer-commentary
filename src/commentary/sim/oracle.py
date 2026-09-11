"""A backend that answers from the sim, and lies on request.

The offline pipeline needs something to stand where Claude stands when there
is no key. A handler that simply returned the sim's state would be cheating,
because it would know things no vision model could know. So this one is held
to the same rules as the real backend: it is handed content blocks and
nothing else, and it recovers which moment it is looking at by decoding the
timestamp burned into the picture. If the frames stop carrying it, the oracle
fails rather than guessing, exactly as the real backend fails on a refusal.

The lying is the other half of the job. The fact gate is only worth having if
we can show it catching errors, and a model that is right by construction can
never demonstrate that. ``error_rate`` injects the three failures that matter
and writes down what it did, so the eval can divide caught by injected.
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
from commentary.sim.match import MatchSim, SimState
from commentary.sim.render import decode_ts

T = TypeVar("T", bound=BaseModel)

#: The three ways a caller goes wrong that the fact gate is built to stop.
ERROR_KINDS: tuple[str, ...] = ("fake_name", "wrong_score", "phantom_goal")

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
        "It is in! {who} finishes, and it is {home} {h}-{a} {away}.",
        "Goal! {who} buries it, {h}-{a}.",
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


@dataclass
class SimOracle:
    """An :class:`LLMBackend` that reads the sim off the picture it is given.

    ``error_rate`` is the fraction of caller and analyst calls that come back
    deliberately wrong. A phantom goal cannot be injected at a moment when a
    goal genuinely happened, since that would not be an error; in that case
    one of the other two kinds is used instead.
    """

    sim: MatchSim
    error_rate: float = 0.0
    seed: int = 17
    injected: list[tuple[float, str]] = field(default_factory=list)
    calls: int = 0
    _total: Usage = field(default_factory=Usage)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError("error_rate must be between 0 and 1")
        self._rng = random.Random(self.seed)

    @property
    def total(self) -> Usage:
        return self._total

    @property
    def injected_kinds(self) -> dict[str, int]:
        counts = {kind: 0 for kind in ERROR_KINDS}
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
        ts = self.timestamp_from(blocks)
        if ts is None:
            raise LLMError(
                "no decodable timestamp in the frames: the oracle only knows what it can see"
            )
        self.calls += 1
        value = self._answer(tag, output_format, self.sim.at(ts))
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
    def timestamp_from(blocks: list[Block]) -> float | None:
        """Video time recovered from the last image in the prompt.

        The last image is the newest one every agent sends, so it is the one
        the answer should be about.
        """
        for block in reversed(blocks):
            if block.get("type") != "image":
                continue
            source = block.get("source", {})
            data = source.get("data")
            if not isinstance(data, str):
                continue
            raw = np.frombuffer(base64.standard_b64decode(data), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None:
                continue
            return decode_ts(np.asarray(image))
        return None

    # ------------------------------------------------------------ answers

    def _answer(self, tag: str, output_format: type[BaseModel], state: SimState) -> BaseModel:
        if tag == "board" or output_format is BoardRead:
            return self._board(state)
        if tag == "analyst" or output_format is AnalystLine:
            return self._analyst(state)
        return self._caller(state)

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

    def _caller(self, state: SimState) -> CallerLine:
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
            line = f"It is in! {who} has scored, and this place erupts."
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


def _rough_tokens(blocks: list[Block]) -> int:
    """A believable input-token count, so cost accounting has something to add."""
    total = 0
    for block in blocks:
        if block.get("type") == "image":
            total += 780
        else:
            total += len(str(block.get("text", ""))) // 4
    return total
