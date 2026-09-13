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
breath, floored so two lines cannot tread on each other and capped at the old
four seconds, which is what a full sentence earns anyway.

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
from commentary.schemas import SpeakDecision, Trigger


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

    def gap_after(self, last_spoken_seconds: float | None) -> float:
        """How long the voice owes the last line before the next one.

        ``None`` means nobody recorded how long it took, and the honest answer
        then is the old fixed cap.
        """
        if last_spoken_seconds is None:
            return self.caller.min_gap_s
        earned = last_spoken_seconds + self.caller.gap_after_line_s
        return min(self.caller.min_gap_s, max(self.caller.min_gap_floor_s, earned))

    def decide(
        self,
        now_ts: float,
        triggers: Sequence[Trigger],
        last_spoken_ts: float | None,
        *,
        after_goal: bool = False,
        last_spoken_seconds: float | None = None,
    ) -> SpeakDecision:
        """One tick's verdict, with a reason a human reading a trace can act on."""
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
        gap = self.gap_after(last_spoken_seconds)
        capped = last_spoken_ts is not None and silence_s < gap
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
