"""The colour voice: when it is allowed to speak, what it looks up, and what
it is not permitted to say once the model has answered.

Three things make this agent different from the caller, and all three are here
rather than in the prompt, because a prompt is a request and this is a rule.

It speaks rarely, and on its own schedule. :meth:`Analyst.should_speak` is
deliberately separate from the model call so the director can ask the question
every tick for nothing: a lull long enough, a decent gap since its own last
line, or the aftermath of something big. Most ticks end there and no tokens
are spent.

It chooses what to look up. :meth:`Analyst.gather` reaches for two or three of
the match tools, not all seven. Dumping every tool's output into every call is
how an analyst ends up saying whatever happens to be at the top of the list
instead of what bears on this moment, and it is how the cheap prefix stops
being the expensive part of the message.

It must not do the caller's job. A second voice that paraphrases the first is
worse than no second voice, so a line too close to what the caller has just
said is killed here, at a tighter threshold than the analyst's own
self-repetition gate — repeating yourself is a lapse, repeating your colleague
is the failure mode that makes two-voice commentary sound wrong.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Mapping
from typing import Any

from commentary.agents.caller import (
    RepetitionGate,
    clean_line,
    flatten,
    similarity,
    trim_words,
)
from commentary.capture.buffer import DelayBuffer
from commentary.config import ANALYST_MODEL, AnalystConfig, CallerConfig
from commentary.llm.base import LLMBackend, LLMError
from commentary.prompts.analyst import analyst_blocks, analyst_system
from commentary.schemas import BIG_EVENTS, AnalystLine, Event, KnowledgePack, MatchState, Side
from commentary.tools import MatchTools

#: Below this the analyst stays quiet. Higher than the caller's floor on
#: purpose: the caller is the only voice that can describe a shot and a hedged
#: description still beats silence, while an unsure aside has no such excuse —
#: nothing is lost by skipping it. ``AnalystConfig`` carries no threshold of
#: its own, so this is the knob, exposed on the constructor for sweeps.
MIN_CONFIDENCE = 0.45

#: How close an analyst line may come to something the caller just said. Well
#: under the caller's own repetition threshold, because the two voices share a
#: channel and a viewer hears the paraphrase immediately.
ECHO_THRESHOLD = 0.45

#: Lines of on-air memory shown back to the model, both voices together.
ON_AIR_LINES = 6

#: How much of the recent event list the analyst is shown.
RECENT_EVENTS = 6

#: Events that mean nothing has happened that the picture explains by itself.
#: Everything else — a corner, a foul, a shot, a substitution — is the match
#: in a pattern, and a pattern is what the pre-match matchups speak to.
_QUIET_EVENTS = frozenset({Event.NONE, Event.BUILD_UP, Event.KICKOFF})

#: Scores as a commentator says them, for catching a scoreline read back out.
_SCORE_WORDS: dict[int, tuple[str, ...]] = {
    0: ("nil", "nought", "zero", "none"),
    1: ("one",),
    2: ("two",),
    3: ("three",),
    4: ("four",),
    5: ("five",),
    6: ("six",),
    7: ("seven",),
    8: ("eight",),
    9: ("nine",),
}

#: Ways of stating a scoreline that do not name the numbers at all.
_SCORE_PHRASES = ("all square", "apiece", "goalless", "the equaliser", "the equalizer")

#: The labels this voice reaches for when it forgets it is talking, not
#: writing. The caller's cleaner knows its own set and is not ours to edit, so
#: these are stripped first and the shared cleaner does the rest.
_ANALYST_LABEL = re.compile(
    r"^\s*(?:analyst|analysis|colou?r|co[- ]commentator|summari[sz]er)\s*[:\-–—]\s*",
    re.IGNORECASE,
)


def strip_label(text: str) -> str:
    """``clean_line``, plus the labels only the second voice writes."""
    return clean_line(_ANALYST_LABEL.sub("", clean_line(text)))


class Analyst:
    """Frames, state and a handful of looked-up facts in; an aside out, rarely.

    Holds its own :class:`RepetitionGate`, separate from the caller's. The two
    voices must not silence each other: the analyst saying "that left side
    again" ten seconds after the caller said "Arsenal down the left" is not a
    repeat, it is the job, and neither gate should be able to veto the other
    voice's line. What the analyst may not do is paraphrase — that is
    :data:`ECHO_THRESHOLD`, checked against on-air memory, not against a gate.
    """

    def __init__(
        self,
        backend: LLMBackend,
        tools: MatchTools,
        config: AnalystConfig | None = None,
        pack: KnowledgePack | None = None,
        *,
        model: str = ANALYST_MODEL,
        max_tokens: int = 640,
        effort: str | None = "low",
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self.backend = backend
        self.config = config or AnalystConfig()
        self.tools = tools
        self.pack = pack
        self.model = model
        self.max_tokens = max_tokens
        #: Low, and not for cost. This is a considered aside, not a hard
        #: reasoning problem, and a lull that has passed by the time the line
        #: arrives is worse than no line at all.
        self.effort = effort
        self.min_confidence = min_confidence
        #: Built once, never rebuilt: identical bytes on every call is what
        #: makes carrying two squads and the pre-match notes affordable.
        self.system = analyst_system(self.pack, self.config)
        #: The analyst's own lines only. See the class docstring.
        self.gate = RepetitionGate(
            CallerConfig(recent_lines=ON_AIR_LINES, repetition_threshold=0.62)
        )
        #: Everything that has gone out, either voice, in the order it went.
        self._on_air: deque[tuple[str, str]] = deque(maxlen=ON_AIR_LINES)
        #: Why the last call did not produce speech, in words, for the log.
        self.last_reason = ""
        #: How close the last candidate came to something already said.
        self.last_similarity = 0.0
        #: Suppressions by cause, for the eval's silence breakdown.
        self.suppressed: Counter[str] = Counter()

    # -- eligibility -----------------------------------------------------

    def should_speak(
        self,
        silence_s: float,
        last_analyst_ts: float,
        now_ts: float,
        last_event: Event | None = None,
    ) -> tuple[bool, str]:
        """May the analyst be asked at all, and why — cheap, no model call.

        ``silence_s`` is how long nobody has said anything;
        ``last_analyst_ts`` is when this voice last spoke, and ``0.0`` for
        never. ``last_event`` is whatever has just happened, or ``None`` once
        it is no longer recent: the director owns the event clock, so "in the
        seconds after" is its judgement to make, not this function's.

        The returned reason is not only for the log — on a yes it is handed
        to the prompt as the reason the analyst is being asked, and a lull and
        an aftermath want different lines out of the same model.
        """
        if last_event is not None and last_event in BIG_EVENTS:
            return True, f"the aftermath of a {last_event.value.replace('_', ' ')}"
        gap = now_ts - last_analyst_ts
        if gap < self.config.min_gap_s:
            return False, (
                f"only {gap:.1f} s since your own last line, and the gap is "
                f"{self.config.min_gap_s:.0f} s"
            )
        if silence_s < self.config.lull_s:
            return False, (
                f"the last line was {silence_s:.1f} s ago, and a lull is {self.config.lull_s:.0f} s"
            )
        return True, f"a lull — nothing has been said for {silence_s:.1f} s"

    # -- what to look up -------------------------------------------------

    @property
    def frame_spacing_s(self) -> float:
        """Seconds between the frames it is shown: the window, spread thin."""
        return self.config.window_s / max(1, self.config.frames - 1)

    def gather(self) -> dict[str, Any]:
        """Pick the few facts that bear on this moment. Never all of them.

        The match facts — score, recent events, who has the ball — are always
        worth the tokens; they are three short dicts and they are what makes
        an aside land in the right match. After that it is a choice, and the
        choice is the point:

        - a name the system can actually read off a shirt is the most
          concrete thing available, so that player gets looked up, preferring
          whoever is on the ball-side;
        - after something big, the season is the frame: what this goal or card
          does to teams in *this* form;
        - in a genuine lull the pre-match notes are all there is, so the
          storylines and the form come across together;
        - otherwise the match is in a pattern, and the duel the researcher
          flagged before kickoff is the thing that explains it.

        ``team_sheet`` is deliberately never gathered: both squads are already
        in the cached system prefix, and paying for them twice buys nothing.
        """
        state = self.tools.state
        facts: dict[str, Any] = {"scoreline": self.tools.scoreline()}
        events = self.tools.recent_events(RECENT_EVENTS)
        if events:
            facts["recent_events"] = events
        if state.possession is not Side.UNKNOWN:
            facts["possession"] = self.tools.possession()

        found = self._player_in_play(state)
        if found is not None:
            facts["player"] = found

        if self._after_big_event(state):
            form = self.tools.form()
            if form:
                facts["form"] = form
        elif self._quiet(state):
            storylines = self.tools.storylines()
            if storylines:
                facts["storylines"] = storylines
            form = self.tools.form()
            if form:
                facts["form"] = form
        else:
            matchups = self.tools.key_matchups()
            if matchups:
                facts["key_matchups"] = matchups
        return facts

    def _player_in_play(self, state: MatchState) -> dict[str, Any] | None:
        """The player worth a lookup, or None when no name is legible.

        ``on_pitch`` holds the numbers the caller has actually read, so a name
        in there is a name the system has earned. Where several are in play,
        the one on the ball-side is the one the aside is likely to be about.
        """
        if not state.on_pitch:
            return None
        found: dict[str, Any] | None = None
        for name in state.on_pitch.values():
            entry = self.tools.player(name)
            if "error" in entry:
                continue
            if found is None:
                found = entry
            if state.possession is Side.UNKNOWN or entry.get("side") == state.possession.value:
                return entry
        return found

    @staticmethod
    def _after_big_event(state: MatchState) -> bool:
        """Has something worth interrupting for happened in the last few beats?"""
        return any(event in BIG_EVENTS for event in state.last_events[-3:])

    @staticmethod
    def _quiet(state: MatchState) -> bool:
        """Nothing has happened lately that the picture explains by itself."""
        return all(event in _QUIET_EVENTS for event in state.last_events[-3:])

    # -- on-air memory ---------------------------------------------------

    def heard(self, line: str) -> None:
        """Record what the other voice just said, so the analyst can avoid it.

        The director calls this when a caller line goes out. It feeds both the
        prompt and the echo veto; without it the analyst's only defence
        against paraphrasing its colleague is the prompt, and a prompt is a
        request.
        """
        text = line.strip()
        if text:
            self._on_air.append(("the caller", text))

    @property
    def on_air(self) -> list[str]:
        """What has gone out lately, labelled by voice, oldest first."""
        return [f"{who}: {text}" for who, text in self._on_air]

    @property
    def caller_lines(self) -> list[str]:
        """Just the other voice's lines — what the echo check measures against."""
        return [text for who, text in self._on_air if who == "the caller"]

    # -- the call --------------------------------------------------------

    async def call(
        self,
        buffer: DelayBuffer,
        state_summary: str,
        reason: str,
    ) -> AnalystLine | None:
        """Look at a wide slice of the recent past and decide whether to speak.

        No lookahead, unlike the caller: the analyst is not committing to how
        a move ends, so peeking past the cursor would only tempt it into
        narrating something the viewer has not seen yet.

        Returns ``None`` only when there was nothing to look at or the model
        call failed — a dropped aside is a missed line, not a crashed match.
        """
        frames = buffer.at_cursor(self.config.frames, self.frame_spacing_s)
        if not frames:
            self.last_reason = "no frames at the cursor"
            self.suppressed["no_frames"] += 1
            return None

        facts = self.gather()
        blocks = analyst_blocks(frames, state_summary, facts, self.on_air, reason)

        try:
            parsed = await self.backend.parse(
                model=self.model,
                system=self.system,
                blocks=blocks,
                output_format=AnalystLine,
                max_tokens=self.max_tokens,
                effort=self.effort,
                cache_system=True,
                tag="analyst",
            )
        except LLMError as exc:
            self.last_reason = f"model call failed: {exc}"
            self.suppressed["llm_error"] += 1
            return None

        return self._settle(parsed.value, facts)

    def _settle(self, proposed: AnalystLine, facts: Mapping[str, Any]) -> AnalystLine:
        """Apply the post-conditions and record why, before anyone sees the line."""
        text = trim_words(strip_label(proposed.line), self.config.max_words)
        self.last_similarity = 0.0

        if not proposed.speak:
            self.last_reason = "the model chose silence"
            return proposed.model_copy(update={"line": text, "speak": False})

        code, reason, score = self._veto(proposed, text, facts)
        self.last_similarity = score
        if code:
            self.last_reason = reason
            self.suppressed[code] += 1
            return proposed.model_copy(update={"line": text, "speak": False})

        self.last_reason = ""
        self.gate.accept(text)
        self._on_air.append(("you", text))
        return proposed.model_copy(update={"line": text, "speak": True})

    def _veto(
        self,
        proposed: AnalystLine,
        text: str,
        facts: Mapping[str, Any],
    ) -> tuple[str, str, float]:
        """``(code, reason, similarity)``; an empty code means the line may go."""
        if not text:
            return "empty", "the model set speak but wrote nothing", 0.0
        if proposed.confidence < self.min_confidence:
            return (
                "low_confidence",
                f"confidence {proposed.confidence:.2f} is under {self.min_confidence:.2f}",
                0.0,
            )
        if restates_score(text, facts.get("scoreline")):
            return "scoreline", "the line reads the scoreboard back out", 0.0
        echo = max((similarity(text, said) for said in self.caller_lines), default=0.0)
        if echo >= ECHO_THRESHOLD:
            return "echo", f"too close to what the caller just said ({echo:.2f})", echo
        allowed, score = self.gate.judge(text)
        if not allowed:
            return "repetition", f"too close to its own recent line ({score:.2f})", score
        return "", "", max(echo, score)


def restates_score(text: str, scoreline: Any = None) -> bool:
    """Is this line just the scoreboard, in figures or in words?

    The board reader owns the score and the graphic on screen shows it, so an
    analyst line that spends itself announcing it is both redundant and a way
    to contradict the one component that actually read the thing. Figures are
    only caught when they match the live score, because "4-3-3" is a formation
    and a commentator is allowed to say it.
    """
    flat = flatten(text)
    if any(phrase in flat for phrase in _SCORE_PHRASES):
        return True
    if not isinstance(scoreline, Mapping):
        return False
    home, away = scoreline.get("home_score"), scoreline.get("away_score")
    if not isinstance(home, int) or not isinstance(away, int):
        return False
    for first, second in ((home, away), (away, home)):
        for left in (str(first), *_SCORE_WORDS.get(first, ())):
            for right in (str(second), *_SCORE_WORDS.get(second, ())):
                if f"{left} {right}" in flat:
                    return True
    return False
