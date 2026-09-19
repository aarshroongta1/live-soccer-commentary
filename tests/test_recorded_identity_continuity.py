"""Focused identity and bounded observer-context contracts for the recorded demo."""

from pathlib import Path

import pytest

from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import (
    Frame,
    Observation,
    ObservedActor,
    ObserverResult,
    RecordedDemoConfig,
    _observe,
    _resolve_observation_actors,
    dry_run_estimate,
    safe_pack,
)


def test_multiword_shirt_name_matches_contiguous_accented_roster_name() -> None:
    pack = safe_pack({"away": {"starters": [{"name": "Ángel Di María", "number": 11}]}})
    resolved = _resolve_observation_actors(
        [
            Observation(
                id="di-maria",
                at_s=1,
                kind="play",
                description="A player carries the ball.",
                actor=ObservedActor(
                    side="away",
                    shirt_number=11,
                    shirt_name="DI MARIA",
                    identity_source="shirt_name",
                    action_role="ball_carrier",
                ),
            )
        ],
        pack,
    )
    assert resolved[0].actor is not None
    assert resolved[0].actor.resolved_name == "Ángel Di María"


def test_multiword_name_requires_contiguous_tokens_and_number_agreement() -> None:
    pack = safe_pack(
        {
            "away": {
                "starters": [
                    {"name": "Ángel Di María", "number": 11},
                    {"name": "Other Player", "number": 12},
                ]
            }
        }
    )
    resolved = _resolve_observation_actors(
        [
            Observation(
                id="wrong-number",
                at_s=1,
                kind="play",
                description="A player carries the ball.",
                actor=ObservedActor(
                    side="away",
                    shirt_number=12,
                    shirt_name="DI MARIA",
                    identity_source="shirt_name",
                    action_role="ball_carrier",
                ),
            )
        ],
        pack,
    )
    assert resolved[0].actor is None


def test_noncontiguous_and_ambiguous_compound_shirt_names_do_not_resolve() -> None:
    pack = safe_pack(
        {
            "away": {
                "starters": [
                    {"name": "Ángel Di María", "number": 11},
                    {"name": "Lucas Di María", "number": 12},
                ]
            }
        }
    )

    def actor(shirt_name: str) -> ObservedActor:
        return ObservedActor(
            side="away",
            shirt_name=shirt_name,
            identity_source="shirt_name",
            action_role="ball_carrier",
        )

    resolved = _resolve_observation_actors(
        [
            Observation(
                id="noncontiguous",
                at_s=1,
                kind="play",
                description="A player carries the ball.",
                actor=actor("ANGEL MARIA"),
            ),
            Observation(
                id="ambiguous-compound",
                at_s=2,
                kind="play",
                description="A player carries the ball.",
                actor=actor("DI MARIA"),
            ),
        ],
        pack,
    )
    assert [item.actor for item in resolved] == [None, None]


def test_dry_run_discloses_bounded_context_billing() -> None:
    estimate = dry_run_estimate(80)
    assert estimate["observer_context_seconds"] == 2.0
    assert "billed again" in estimate["note"]


@pytest.mark.asyncio
async def test_observer_uses_two_second_context_without_expanding_observation_window() -> None:
    backend = ScriptedBackend()
    backend.always("recorded_observer", ObserverResult(observations=[]))
    frames = [Frame(at_s=index / 2, jpeg=b"frame") for index in range(80)]
    await _observe(
        {
            "frames": frames,
            "config": RecordedDemoConfig(clip=Path("synthetic.mp4"), duration_s=40),
            "audit": {},
        },
        backend=backend,
    )

    calls = backend.calls_tagged("recorded_observer")
    assert len(calls) == 7
    boundary_call = calls[5]  # 30s..36s: shares 28s..29.5s only as context.
    assert '"window":{"start_s":30.0,"end_s_exclusive":36.0}' in boundary_call.text
    assert "context-only frame at_s=29.50" in boundary_call.text
    assert "context-only frame at_s=27.50" not in boundary_call.text
    assert "visibly and continuously establish that same finisher" in boundary_call.system


@pytest.mark.asyncio
async def test_context_only_frame_cannot_be_reported_as_a_new_observation() -> None:
    backend = ScriptedBackend()
    empty = ObserverResult(observations=[])
    backend.queue(
        "recorded_observer",
        [
            empty,
            empty,
            empty,
            empty,
            empty,
            ObserverResult(
                observations=[
                    Observation(
                        id="backdated",
                        at_s=29.5,
                        kind="goal",
                        description="The ball reaches the net.",
                    )
                ]
            ),
        ],
    )
    frames = [Frame(at_s=index / 2, jpeg=b"frame") for index in range(80)]
    with pytest.raises(ValueError, match="outside 30..36s"):
        await _observe(
            {
                "frames": frames,
                "config": RecordedDemoConfig(clip=Path("synthetic.mp4"), duration_s=40),
                "audit": {},
            },
            backend=backend,
        )
