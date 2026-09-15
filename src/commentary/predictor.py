"""When to consider speaking at all.

worldcupvoice speaks every four seconds whatever is happening, which is why it
talks over nothing and misses everything. Here silence is a decision with two
forces acting on it. Triggers push towards speech, each carrying its own
weight: a board change is the game changing and outranks everything, a camera
cut is worth a glance. Pushing back is a rate cap, because a voice that lands
every two seconds sounds panicked no matter how good the lines are.

The cap is measured off the last line rather than off the clock. Real
commentary calls build-up in fragments — "De Paul." "Messi, Álvarez." — and a
fragment takes under a second to say; holding the next one for four seconds
because a sentence would have taken four is three seconds of dead air the
broadcast never has. So the gap a line earns is how long it took to say plus a
breath, floored so two lines cannot tread on each other and capped.

**The cap is two rates, not one.** ``docs/research/real-commentary-corpus.md``
section 2.3 measures the gap between utterances by the phase of play: 2.8 s in
an attacking move, 4.2 s in build-up, 4.5 s at a dead ball, 4.6 s at a
stoppage — and the words move the other way, 7 to a line in the box against 10
at a restart. One rate through both sounds too slow in the box and too busy on
the halfway line, which is that study's Gap 3, so the cap is chosen by what
the last line was about. :func:`phase_of` is the whole of the mapping.

**The build-up cap opens when the other seat is short of the channel.**
Section 4 of the same study gives the colour voice about 31% of the
utterances in club football and this system was giving it 14%. The lead is
not wrong to speak, but in build-up — 54.6% of the match — there is no hole
for a second voice to stand in, so the build-up cap alone stretches from
4.5 s towards 6.0 s in proportion to the shortfall (``colour_stretch``,
computed by :class:`commentary.agents.colour.Share`). It is a ceiling, not
a wait: a fragment still buys a fragment's silence.

**A chosen silence is a decision and the cap counts it.** The phraser may now
return an empty line, which is what a quarter of build-up touches and 43% of
goal kicks get in real commentary. Nothing is spoken, so the silence that
builds pressure keeps building — a broadcast can be sparse and cannot go mute
— but the rate cap measures from the moment the decision was made rather than
from the last line, or the next tick calls the caller again half a second
later and pays Opus for the privilege.

Between those two, pressure. Nothing said for long enough starts lifting the
urgency, and past ``silence_forces_at_s`` it lifts it all the way: broadcast
commentary can be sparse but it cannot go mute, and a system that only speaks
on triggers goes mute the moment its detectors get unlucky.

``decide`` is a pure function of its arguments plus a little bookkeeping, so
the eval can replay a recorded trigger trace through it offline and get the
same answer it got live.
"""

from __future__ import annotations

from collections.abc import Sequence

from commentary.config import SETTINGS, CallerConfig, PredictorConfig
from commentary.schemas import Event, SpeakDecision, Trigger

#: The ball is moving at goal. Study section 2.3's "attacking move" is ±10 s
#: around a shot or ±5 s around a cross or a corner delivery, and it is the
#: one phase where the gap closes rather than opens.
#:
#: This used to carry ``Event.CLEARANCE`` in ``Event.CROSS``'s place, a
#: defensive action standing in for the one the corpus actually means, because
#: the caller's vocabulary had no cross yet. Gap 8 adds it; this is the other
#: half of using it.
ATTACKING = frozenset(
    {Event.GOAL, Event.SHOT, Event.SAVE, Event.PENALTY, Event.CORNER, Event.CROSS}
)

#: The ball is dead, or the game is. The slowest of the three: 4.5 s at a
#: restart, 4.6 s at a stoppage, and the longest lines in the corpus.
DEAD_BALL = frozenset(
    {
        Event.FREE_KICK,
        Event.THROW_IN,
        Event.KICKOFF,
        Event.FOUL,
        Event.CARD,
        Event.OFFSIDE,
        Event.SUBSTITUTION,
        Event.STOPPAGE,
    }
)


def phase_of(event: Event | None, *, replay: bool = False) -> str:
    """Which of the three rates a line about ``event`` earns.

    ``None`` is a line filed under no phase — the colour seat's — and takes
    the default cap rather than a guess.

    ``replay`` is the broadcast's own dead ball and outranks the event. A
    replay of a shot is not an attacking move: the ball is not in play behind
    it, nobody is about to score, and the 2.5 s cap an attacking move earns
    would have the voice back over the next replay angle before the last line
    had landed. Study section 2.3's dead-ball gap is 4.5 s and a stoppage's
    is 4.6, and a replay is what a broadcast shows during both.
    """
    if replay:
        return "dead_ball"
    if event is None or event is Event.NONE:
        return "none"
    if event in ATTACKING:
        return "attacking"
    if event in DEAD_BALL:
        return "dead_ball"
    return "build_up"


class SpeakPredictor:
    """Decides, on each tick, whether this is a moment for a line."""

    def __init__(
        self,
        cfg: PredictorConfig = SETTINGS.predictor,
        caller: CallerConfig = SETTINGS.caller,
    ) -> None:
        self.cfg = cfg
        self.caller = caller
        self.ticks = 0
        self.calls = 0
        self.last_decision: SpeakDecision | None = None

    def _weight(self, trigger: Trigger) -> float:
        return float(self.cfg.urgency_by_trigger.get(trigger.value, 0.0))

    def _pressure(self, silence_s: float) -> float:
        """0 to 1, how badly the silence itself now wants a line."""
        start = self.cfg.silence_pressure_after_s
        forces = self.cfg.silence_forces_at_s
        if silence_s >= forces:
            return 1.0
        if silence_s <= start or forces <= start:
            return 0.0
        return (silence_s - start) / (forces - start)

    def cap_for(
        self,
        last_event: Event | None,
        colour_stretch: float = 0.0,
        *,
        last_was_replay: bool = False,
    ) -> float:
        """The longest the voice ever waits after a line about ``last_event``.

        ``colour_stretch`` is the second voice's shortfall, 0 to 1, out of
        :class:`commentary.agents.colour.Share`. It opens the build-up cap
        and nothing else: from ``min_gap_build_up_s`` at 0 towards
        ``min_gap_build_up_stretched_s`` at 1, linear in between. The other
        two phases do not move — an attacking move is the lead's and a
        restart already gives the colour seat five times the rate.

        This is a ceiling and not a wait. The gap actually owed is what the
        last line took plus a breath (:meth:`gap_after`), so stretching the
        cap only changes anything where a long line has already earned more
        than 4.5 s of silence — which is exactly the hole the second voice
        needs and cannot currently find.
        """
        caps = {
            "attacking": self.caller.min_gap_attacking_s,
            "build_up": self._build_up_cap(colour_stretch),
            "dead_ball": self.caller.min_gap_dead_ball_s,
        }
        return caps.get(phase_of(last_event, replay=last_was_replay), self.caller.min_gap_s)

    def _build_up_cap(self, colour_stretch: float) -> float:
        reach = max(0.0, min(1.0, colour_stretch))
        base = self.caller.min_gap_build_up_s
        return base + (self.caller.min_gap_build_up_stretched_s - base) * reach

    def gap_after(
        self,
        last_spoken_seconds: float | None,
        last_event: Event | None = None,
        colour_stretch: float = 0.0,
        *,
        last_was_replay: bool = False,
    ) -> float:
        """How long the voice owes the last line before the next one.

        ``None`` for the length means nobody recorded how long it took, and
        the honest answer then is the phase's cap on its own.
        """
        cap = self.cap_for(last_event, colour_stretch, last_was_replay=last_was_replay)
        if last_spoken_seconds is None:
            return cap
        earned = last_spoken_seconds + self.caller.gap_after_line_s
        return min(cap, max(self.caller.min_gap_floor_s, earned))

    def decide(
        self,
        now_ts: float,
        triggers: Sequence[Trigger],
        last_spoken_ts: float | None,
        *,
        after_goal: bool = False,
        last_spoken_seconds: float | None = None,
        last_event: Event | None = None,
        last_was_replay: bool = False,
        last_quiet_ts: float | None = None,
        colour_stretch: float = 0.0,
    ) -> SpeakDecision:
        """One tick's verdict, with a reason a human reading a trace can act on.

        ``last_event`` is what the last line was about and picks the rate,
        unless ``last_was_replay`` says the last line was said over a replay,
        which takes the dead-ball rate whatever the event was.
        ``last_quiet_ts`` is when the phraser last chose to say nothing: it
        holds the rate cap off without touching the pressure that stops the
        broadcast going mute. ``colour_stretch`` is how far the second voice
        is behind its share of the channel and opens the build-up cap; see
        :meth:`cap_for`.
        """
        self.ticks += 1
        silence_s = float("inf") if last_spoken_ts is None else now_ts - last_spoken_ts
        since = (
            "nothing spoken yet" if last_spoken_ts is None else f"{silence_s:.1f}s since last line"
        )

        fired = list(dict.fromkeys(triggers))
        best = max(fired, key=self._weight, default=None)
        base = self._weight(best) if best is not None else 0.0

        pressure = self._pressure(silence_s)
        if pressure > 0.0 and Trigger.SILENCE_PRESSURE not in fired:
            fired.append(Trigger.SILENCE_PRESSURE)
        # Pressure lifts whatever the triggers already asked for towards 1.0,
        # so a weak trigger late in a silence outranks the same trigger early.
        urgency = min(1.0, base + pressure * (1.0 - base))

        forced = silence_s >= self.cfg.silence_forces_at_s
        gap = self.gap_after(
            last_spoken_seconds, last_event, colour_stretch, last_was_replay=last_was_replay
        )
        # The cap runs from whichever came later, the last line or the last
        # decision to pass over a moment. Pressure, above, runs from the last
        # line only: a chosen silence is still silence to a listener.
        since_cap = silence_s
        if last_quiet_ts is not None:
            since_cap = min(since_cap, now_ts - last_quiet_ts)
        capped = (last_spoken_ts is not None or last_quiet_ts is not None) and since_cap < gap
        # A goal is the one thing that is never too soon. The four seconds
        # after one are when the scorer's name, the celebration and the
        # replay all arrive at once, and they were the four seconds the cap
        # spent silent.
        override = Trigger.BOARD_CHANGE in fired or after_goal
        winner = best.value if best is not None else "silence_pressure" if pressure else "none"

        if capped and not override:
            return self._record(
                SpeakDecision(
                    should_call=False,
                    triggers=fired,
                    urgency=urgency,
                    reason=(
                        f"rate_cap: {winner} {urgency:.2f} but {since}, "
                        f"min gap {gap:.1f}s"
                    ),
                )
            )
        if capped and override:
            return self._record(
                SpeakDecision(
                    should_call=True,
                    triggers=fired,
                    urgency=urgency,
                    reason=(
                        f"{'a goal' if after_goal else 'board_change'} beats the rate cap, {since}"
                    ),
                )
            )
        if forced:
            return self._record(
                SpeakDecision(
                    should_call=True,
                    triggers=fired,
                    urgency=urgency,
                    reason=f"silence_pressure forces a line: {since}, {winner} {urgency:.2f}",
                )
            )
        # Pressure on its own never speaks before the forcing point; it only
        # makes the next real trigger count for more. Speaking at six seconds
        # because six seconds have passed is the timer this replaces.
        if best is None:
            return self._record(
                SpeakDecision(
                    should_call=False,
                    triggers=fired,
                    urgency=urgency,
                    reason=f"no trigger, {since}",
                )
            )
        return self._record(
            SpeakDecision(
                should_call=True,
                triggers=fired,
                urgency=urgency,
                reason=f"{winner} {urgency:.2f}, {since}",
            )
        )

    def _record(self, decision: SpeakDecision) -> SpeakDecision:
        self.last_decision = decision
        if decision.should_call:
            self.calls += 1
        return decision
