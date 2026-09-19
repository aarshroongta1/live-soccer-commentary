"""Regression contract for deterministic recorded-demo writer timing."""

from __future__ import annotations

import pytest

from commentary.recorded_demo import (
    JointScript,
    Observation,
    ScriptLine,
    _constrain_script_timing,
)


def _observations() -> list[Observation]:
    return [
        Observation(id="start", at_s=1.0, kind="play", description="The move begins."),
        Observation(id="delivery", at_s=5.0, kind="play", description="Ball enters the area."),
        Observation(id="net", at_s=10.0, kind="goal", description="Ball is in the net."),
    ]


def _script(lines: list[ScriptLine]) -> JointScript:
    """Keep focused timing cases valid at the public script-schema boundary."""
    padded = list(lines)
    while len(padded) < 7:
        padded.append(
            ScriptLine(
                at_s=12.0 + len(padded),
                voice="analyst",
                text="Afterward.",
                evidence_ids=["net"],
            )
        )
    return JointScript(lines=padded)


def test_timing_moves_late_only_and_records_the_constraint_provenance() -> None:
    script = _script(
        [
            ScriptLine(
                at_s=0.0,
                voice="caller",
                text="The move begins down the left.",
                evidence_ids=["start"],
            ),
            ScriptLine(
                at_s=2.0,
                voice="analyst",
                text="The defense has little time to reset.",
                evidence_ids=["delivery"],
            ),
        ]
    )

    constrained, adjustments = _constrain_script_timing(script, _observations(), 20.0)

    assert [line.text for line in constrained.lines] == [line.text for line in script.lines]
    assert [line.at_s for line in constrained.lines[:2]] == [1.0, 5.0]
    assert adjustments[:2] == [
        {
            "line": 1,
            "from_at_s": 0.0,
            "to_at_s": 1.0,
            "reasons": ["latest_cited_evidence"],
            "latest_evidence_s": 1.0,
            "previous_line_end_s": None,
        },
        {
            "line": 2,
            "from_at_s": 2.0,
            "to_at_s": 5.0,
            "reasons": ["latest_cited_evidence", "previous_line_end"],
            "latest_evidence_s": 5.0,
            "previous_line_end_s": 3.0,
        },
    ]


def test_goal_reaction_is_not_delayed_by_preceding_speech() -> None:
    script = _script(
        [
            ScriptLine(
                at_s=9.0,
                voice="caller",
                text="One two three four five six.",
                evidence_ids=["delivery"],
            ),
            ScriptLine(
                at_s=10.0,
                voice="caller",
                text="Goal! The ball is in.",
                evidence_ids=["net"],
            ),
        ]
    )

    with pytest.raises(ValueError, match="goal cue line 2 is blocked"):
        _constrain_script_timing(script, _observations(), 20.0)


def test_early_goal_reaction_advances_to_evidence_with_provenance() -> None:
    script = _script(
        [
            ScriptLine(
                at_s=9.0,
                voice="caller",
                text="Goal! The ball is in.",
                evidence_ids=["net"],
            )
        ]
    )

    constrained, adjustments = _constrain_script_timing(script, _observations(), 20.0)

    assert constrained.lines[0].at_s == 10.0
    assert adjustments[0]["reasons"] == ["latest_cited_evidence"]


def test_later_goal_follow_up_is_an_ordinary_later_slot() -> None:
    script = _script(
        [
            ScriptLine(
                at_s=10.0,
                voice="caller",
                text="Goal! The ball is in.",
                evidence_ids=["net"],
            ),
            ScriptLine(
                at_s=10.1,
                voice="caller",
                text="The scorer wheels away in delight.",
                evidence_ids=["net"],
            ),
        ]
    )

    constrained, adjustments = _constrain_script_timing(script, _observations(), 20.0)

    assert constrained.lines[0].at_s == 10.0
    assert constrained.lines[1].at_s > 10.1
    assert adjustments[-1]["reasons"] == ["previous_line_end"]


def test_unfittable_script_is_rejected_without_dropping_a_line() -> None:
    script = _script(
        [
            ScriptLine(
                at_s=9.0,
                voice="caller",
                text="One two three four five six.",
                evidence_ids=["delivery"],
            )
        ]
    )

    with pytest.raises(ValueError, match="cannot fit before clip duration"):
        _constrain_script_timing(script, _observations(), 10.0)
