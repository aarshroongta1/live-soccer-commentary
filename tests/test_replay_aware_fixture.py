"""Offline acceptance fixtures for the Barça equaliser and its replay.

These tests do not claim that the runtime passes the scenario yet.  They lock
the observed failure and provide a deterministic output audit for later stages
of ``docs/PLAN-replay-aware-commentary.md``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from commentary.grading.transcripts import load
from commentary.schemas import CallerLine, Event

FIXTURES = Path(__file__).parent / "fixtures" / "replay_aware"


def json_document(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def json_lines(name: str) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (FIXTURES / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass(frozen=True)
class OutputAudit:
    goal_announcements: int
    urgent_goal_beats: int
    score_announcements: int
    celebration_lines: int

    @property
    def violations(self) -> tuple[str, ...]:
        failures: list[str] = []
        if self.goal_announcements != 1:
            failures.append(f"expected one goal announcement, got {self.goal_announcements}")
        if self.urgent_goal_beats != 1:
            failures.append(f"expected one urgent goal beat, got {self.urgent_goal_beats}")
        if self.score_announcements > 1:
            failures.append(
                f"expected at most one 1-1 announcement, got {self.score_announcements}"
            )
        if self.celebration_lines > 1:
            failures.append(f"expected at most one celebration line, got {self.celebration_lines}")
        return tuple(failures)


def audit_output(rows: list[dict[str, Any]]) -> OutputAudit:
    def score_claim(text: str) -> bool:
        return bool(re.search(r"\b(?:one[- ]one|1[-–:]1)\b", text, flags=re.IGNORECASE))

    return OutputAudit(
        goal_announcements=sum(row["event"] == "goal" for row in rows),
        urgent_goal_beats=sum(row["event"] == "goal" and not row["preemptable"] for row in rows),
        score_announcements=sum(score_claim(row["text"]) for row in rows),
        celebration_lines=sum("celebrat" in row["text"].casefold() for row in rows),
    )


def assert_replay_aware_output(rows: list[dict[str, Any]]) -> None:
    audit = audit_output(rows)
    assert not audit.violations, "; ".join(audit.violations)


def test_reference_transcript_is_aligned_to_the_clip() -> None:
    transcript = load(FIXTURES / "betis_barcelona_transcript.json")

    assert transcript.source == "clips/betis-barcelona-1605-1710.mp4"
    assert transcript.duration_s == pytest.approx(64.86)
    assert "Ferran Torres" in transcript.text
    assert "Betis one, Barça one" in transcript.text
    assert "gets across his man" in transcript.text
    assert "near post" in transcript.text


def test_current_fixture_contains_the_exact_eighteen_caller_observations() -> None:
    rows = json_lines("current_observations.jsonl")
    forms = [
        CallerLine.model_validate({key: value for key, value in row.items() if key != "ts"})
        for row in rows
    ]

    assert len(forms) == 18
    assert [row["ts"] for row in rows] == sorted(row["ts"] for row in rows)
    assert sum(form.event is Event.GOAL for form in forms) == 6
    assert forms[8].line == "Ferran Torres buries it!"
    assert forms[-1].line == "Koundé drives it across the box."


def test_generic_progress_snapshots_do_not_become_five_action_beats() -> None:
    first_five = json_lines("current_observations.jsonl")[:5]
    action_events = {"carry", "pass", "receive", "layoff", "cross", "shot", "save", "finish"}

    assert all(row["event"] == "build_up" for row in first_five)
    assert sum(row["event"] in action_events for row in first_five) == 0


def test_expected_chain_preserves_progression_final_ball_and_finish() -> None:
    chain = json_document("expected_action_chain.json")
    beats = chain["beats"]

    assert [beat["order"] for beat in beats] == list(range(1, len(beats) + 1))
    assert {beat["action"] for beat in beats} >= {"pass", "cross", "movement", "finish"}
    assert {beat["actor"] for beat in beats} >= {
        "Eric García",
        "Jules Koundé",
        "Ferran Torres",
    }
    assert any(
        beat["action"] == "cross"
        and beat["actor"] == "Jules Koundé"
        and beat["target"] == "Ferran Torres"
        for beat in beats
    )
    assert any(
        beat["action"] == "finish"
        and beat["actor"] == "Ferran Torres"
        and beat["destination_zone"] == "near post"
        for beat in beats
    )


def test_every_observation_from_31_seconds_on_is_the_first_goals_aftermath() -> None:
    chain = json_document("expected_action_chain.json")
    low, high = chain["aftermath_and_replay_window_s"]
    aftermath = [
        row for row in json_lines("current_observations.jsonl") if low <= row["ts"] <= high
    ]

    assert aftermath
    assert aftermath[0]["scene"] == "close_up"
    assert any(row["scene"] == "replay" for row in aftermath)
    assert all(row["ts"] > chain["live_goal_window_s"][1] for row in aftermath)
    # These are evidence about the existing incident, even when the old caller
    # mislabeled replay frames as live play or another goal.
    assert chain["goal_incidents"] == [
        {
            "id": chain["incident_id"],
            "live_goal_ts": 21.666666666666668,
            "aftermath_and_replay_window_s": [low, high],
        }
    ]


def test_output_audit_rejects_the_recorded_duplicate_aftermath() -> None:
    audit = audit_output(json_lines("current_output.jsonl"))

    assert audit.goal_announcements == 5
    assert audit.urgent_goal_beats == 4
    assert audit.score_announcements == 2
    assert audit.celebration_lines == 3
    assert len(audit.violations) == 4
    assert any("goal announcement" in failure for failure in audit.violations)
    assert any("urgent goal beat" in failure for failure in audit.violations)
    assert any("1-1 announcement" in failure for failure in audit.violations)
    assert any("celebration line" in failure for failure in audit.violations)
    with pytest.raises(AssertionError, match="one goal announcement"):
        assert_replay_aware_output(json_lines("current_output.jsonl"))


def test_output_audit_accepts_one_goal_one_score_and_one_celebration() -> None:
    desired = [
        {
            "ts": 19.1,
            "event": "cross",
            "preemptable": True,
            "text": "Koundé drives it across the six-yard box.",
        },
        {
            "ts": 21.7,
            "event": "goal",
            "preemptable": False,
            "text": "Ferran turns it in!",
        },
        {
            "ts": 25.0,
            "event": "none",
            "preemptable": True,
            "text": "Ferran Torres, across his man at the near post. Betis one, Barça one.",
        },
        {
            "ts": 32.0,
            "event": "none",
            "preemptable": True,
            "text": "Ferran celebrates with his teammates.",
        },
        {
            "ts": 43.0,
            "event": "none",
            "preemptable": True,
            "text": "The replay shows the first-touch move through Eric García and Koundé.",
        },
    ]

    assert_replay_aware_output(desired)
