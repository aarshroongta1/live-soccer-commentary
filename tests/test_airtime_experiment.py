"""Saved-trace reproduction and counterexamples for the opt-in floor policy."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "airtime_experiment", ROOT / "scripts/airtime_experiment.py"
)
assert spec and spec.loader
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture
def source() -> dict:
    return json.loads((ROOT / "examples/trajectory-outreach/source-replay.json").read_text())


def test_original_is_reproduced_and_newer_analysis_is_preserved(source: dict) -> None:
    baseline = experiment.replay_floor(source, yielding=False)
    candidate = experiment.replay_floor(source, yielding=True)
    recorded = source["speech"][1]["started_s"]
    assert abs(baseline["metrics"]["caller_started_s"] - recorded) < 0.01
    assert candidate["metrics"]["caller_floor_wait_s"] == 0
    assert baseline["metrics"]["caller_floor_wait_s"] > 5
    # Yielding must not win by silencing the caller or all analysis.
    assert candidate["metrics"]["caller_preserved"]
    assert candidate["metrics"]["analyst_spoken_words_estimate"] > 0
    fresh = [
        s
        for s in candidate["speech"]
        if s["line"]["voice"] == "analyst" and s["snapshot_version"] == 2
    ]
    assert len(fresh) == 1 and not fresh[0]["interrupted"]
    assert not any(
        s["snapshot_version"] == 2 and s["line"]["voice"] == "analyst" for s in baseline["speech"]
    )


def test_routine_play_does_not_interrupt(source: dict) -> None:
    for event in source["trace"]:
        if event["event"] == "observation_available" and event["data"]["snapshot_version"] == 2:
            for observation in event["data"]["observations"]:
                observation["action"]["type"] = "pass"
    baseline = experiment.replay_floor(source, yielding=False)
    candidate = experiment.replay_floor(source, yielding=True)
    assert candidate["speech"] == baseline["speech"]


def test_same_event_does_not_interrupt(source: dict) -> None:
    for event in source["trace"]:
        if event["event"] == "agent_action" and event["role"] == "caller":
            line = event["data"]["action"].get("line")
            if line:
                line["evidence_ids"] = ["v1-o3"]
    # Make that old evidence a confirmed major action so chronology is decisive.
    for event in source["trace"]:
        if event["event"] == "observation_available" and event["data"]["snapshot_version"] == 1:
            event["data"]["observations"][-1]["action"].update(
                type="save", certainty="confirmed", cues=["keeper_contact"]
            )
    baseline = experiment.replay_floor(source, yielding=False)
    candidate = experiment.replay_floor(source, yielding=True)
    assert candidate["speech"] == baseline["speech"]


def test_no_ready_caller_means_no_interruption(source: dict) -> None:
    candidate = experiment.replay_floor(source, yielding=True, caller_delay=8)
    first = candidate["speech"][0]
    assert not first["interrupted"]


def test_slower_speech_exposes_loss_of_a_ready_call(source: dict) -> None:
    baseline = experiment.replay_floor(source, yielding=False, words_per_second=2.5)
    candidate = experiment.replay_floor(source, yielding=True, words_per_second=2.5)
    assert not baseline["metrics"]["caller_preserved"]
    assert candidate["metrics"]["caller_preserved"]
    assert candidate["metrics"]["caller_floor_wait_s"] == 0


def test_expired_caller_cannot_cut_off_analysis(source: dict) -> None:
    source["metadata"]["config"]["max_age_s"] = 4
    baseline = experiment.replay_floor(source, yielding=False)
    candidate = experiment.replay_floor(source, yielding=True)
    assert candidate["speech"] == baseline["speech"]
    assert not candidate["metrics"]["caller_preserved"]


def test_all_frozen_timing_variants_remain_causal_without_overlaps(source: dict) -> None:
    for analyst_delay in (0, 0.5, 1):
        for caller_delay in (0, 0.5, 1, 2, 4):
            for yielding in (False, True):
                result = experiment.replay_floor(
                    source,
                    yielding=yielding,
                    analyst_delay=analyst_delay,
                    caller_delay=caller_delay,
                )
                assert result["metrics"]["caller_preserved"]
                for left, right in zip(result["speech"], result["speech"][1:], strict=False):
                    assert left["ended_s"] <= right["started_s"]
                assert result["metrics"]["caller_floor_wait_s"] >= 0
                assert result["metrics"]["analyst_spoken_words_estimate"] > 0
