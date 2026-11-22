"""Opt-in speech-floor experiment; no model, observer, or production default changes."""

from commentary.replay.runtime import ReplayRuntime

# This is an authored editorial policy, not a learned reward or a detector.
MAJOR_ACTIONS = frozenset(
    {
        "save",
        "block",
        "tackle",
        "interception",
        "penalty_awarded",
        "red_card",
    }
)


class YieldingReplayRuntime(ReplayRuntime):
    """Yield older analysis only when a fresh major-action caller is ready.

    A ready line is essential: interrupting on observation arrival alone creates
    silence while the caller is still generating. Existing validity, goal,
    duplicate, and freshness rules continue to govern the eventual admission.
    Speech interruption here is simulated; a real audio player would need safe
    word-boundary cancellation and should be evaluated separately.
    """

    def _speech(self) -> None:
        active = self._active
        caller = self._queue.get("caller")
        if (
            active is not None
            and active.line.voice == "analyst"
            and self._now() < active.ended_s
            and caller is not None
            and caller.result.line is not None
            and not self._is_candidate_stale(caller.result, caller.snapshot)
            and self._spoken_duplicate(caller.result.line) is None
        ):
            cited = set(caller.result.line.evidence_ids)
            newer_major_action = any(
                item.id in cited
                and item.at_s > (active.event_at_s if active.event_at_s is not None else -1)
                and item.kind == "play"
                and item.action is not None
                and item.action.certainty == "confirmed"
                and item.action.type in MAJOR_ACTIONS
                for item in caller.snapshot.observations
            )
            if newer_major_action:
                active.ended_s = self._now()
                active.interrupted = True
                self._trace(
                    "speech_interrupted",
                    "analyst",
                    speech_id=active.id,
                    reason="ready caller cites a newer confirmed major action",
                )
                self._active = None
                self._next_speech_s = self._now()
        super()._speech()
