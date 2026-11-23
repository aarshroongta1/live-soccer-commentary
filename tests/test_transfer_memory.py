"""Offline checks for the transfer experiment, with no model calls."""

import json
from pathlib import Path

from commentary.lesson_memory import digest
from commentary.scoreboard import confirms_score_increment, hold_unconfirmed_goal

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "examples/lesson-memory/transfer"


def test_goal_gate_requires_one_team_to_increase_without_changing_the_other():
    assert confirms_score_increment(0, 1, 1, 1)
    assert confirms_score_increment(1, 1, 1, 2)
    assert not confirms_score_increment(1, 2, 1, 2)
    assert not confirms_score_increment(1, 2, 2, 3)
    assert not confirms_score_increment(1, 2, 3, 1)
    assert not confirms_score_increment(1, 2, None, 2)
    assert not confirms_score_increment(-1, 2, 0, 2)


def test_unchanged_score_is_not_a_miss_label():
    assert hold_unconfirmed_goal("goal", score_increment=False)
    assert not hold_unconfirmed_goal("goal", score_increment=True)
    for outcome in ("miss", "save", "unclear", "other"):
        assert not hold_unconfirmed_goal(outcome, score_increment=False)


def test_new_incident_uses_frozen_earlier_memory_and_retains_all_pairs():
    report = json.loads((RECORD / "result.json").read_text())
    memory = json.loads((ROOT / report["lesson_source"]).read_text())["memory"]
    assert report["memory"] == memory
    assert digest(memory) == report["memory_sha256"]
    assert report["source_incident"]["start_s"] < report["later_incident"]["start_s"]
    assert [p["id"] for p in report["pairs"]] == ["late-1", "late-2", "earlier-control"]
    for pair in report["pairs"]:
        assert {a["arm"] for a in pair["arms"]} == {"baseline", "memory"}
        assert len({a["system_sha256"] for a in pair["arms"]}) == 1
        for arm in pair["arms"]:
            assert set(arm["payload"]) == {"frame_timestamps", "saved_lessons"}
            assert arm["payload"]["frame_timestamps"] == pair["frame_at_s"]
            assert arm["payload"]["saved_lessons"] == (memory if arm["arm"] == "memory" else [])
            raw = json.loads((RECORD / f"prompt-only/{pair['id']}-{arm['arm']}.json").read_text())
            message = next(o for o in raw["output"] if o["type"] == "message")
            assert arm["output"] == json.loads(message["content"][0]["text"])


def test_gate_results_are_derived_from_unedited_score_readings():
    report = json.loads((RECORD / "result.json").read_text())
    for result in report["checked_workflow"]:
        name = "earlier-control" if result["id"] == "earlier-control" else "late-1"
        raw = json.loads((RECORD / f"scoreboard/scorecheck-{name}.json").read_text())
        message = next(o for o in raw["output"] if o["type"] == "message")
        assert result["score_check"] == json.loads(message["content"][0]["text"])
        check = result["score_check"]
        increment = confirms_score_increment(
            *(check[k] for k in ("first_left", "first_right", "last_left", "last_right"))
        )
        assert result["score_increment_confirmed"] == increment
        assert result["held"] == hold_unconfirmed_goal(
            result["candidate"]["outcome"], score_increment=increment
        )
    assert report["status"] == "reviewed_no_demonstrated_learning_transfer"
    assert report["reported_usd"] < 1
    assert report["calls"] == sum(c["calls"] for c in report["costs"].values())
