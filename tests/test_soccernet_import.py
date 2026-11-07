"""Reference conversion and leakage boundaries using fictional source labels."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from commentary.recorded_demo import Observation
from commentary.replay.contracts import ReplayResult
from commentary.replay.evaluation import ReferenceSource, evaluate_replay
from commentary.replay.soccernet import import_case, inspect_labels


def annotation(label: str, position: int, **updates: Any) -> dict[str, Any]:
    return {
        "label": label,
        "position": str(position),
        "gameTime": "2 - 45:10",
        "team": "home",
        "visibility": "visible",
        **updates,
    }


KNOWN_VALID = "england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal"


def labels(*events: dict[str, Any], match_id: str = KNOWN_VALID) -> bytes:
    return json.dumps({"UrlLocal": match_id + "/", "annotations": list(events)}).encode()


def convert(raw: bytes, **updates: Any) -> Any:
    options: dict[str, Any] = dict(
        half=2,
        upstream_split="valid",
        case_id="window",
        duration_s=10,
        video_half_offset_s=3000,
        label_origin="synthetic://test-labels-v2",
        source=ReferenceSource(
            kind="broadcast",
            locator="clips/fictional.mp4",
            start_s=3060,
            provenance="Fictional test source.",
            usage_terms="Project test fixture.",
            annotation_status="reviewed",
        ),
    )
    options.update(updates)
    return import_case(raw, **options)


def test_half_video_position_and_offsets_govern_time_not_match_clock() -> None:
    raw = labels(
        annotation("Goal", 61000, gameTime="1 - 01:01"),  # wrong half
        annotation("Foul", 65000),
        annotation("Yellow card", 60000),  # out of order in source
        annotation("Corner", 70000),  # boundary belongs to the next window
        annotation("Throw-in", 59999),
    )
    case = convert(raw)
    assert [(e.action_type, e.at_s) for e in case.events] == [("yellow_card", 0), ("foul", 5)]
    assert case.match_id == KNOWN_VALID
    assert case.split == "validation"
    assert case.reference_metadata["upstream_split"] == "valid"
    assert case.reference_metadata["split_assignment_verified"]
    assert case.source.annotation_status == "unreviewed"  # alignment is not certified by labels
    assert not case.action_labels_complete
    assert len(case.reference_metadata["window_annotations"]) == 2


def test_unshown_unknown_visibility_and_ambiguous_labels_are_audited_not_scored() -> None:
    case = convert(
        labels(
            annotation("Goal", 61000, visibility="not shown"),
            annotation("Offside", 62000, visibility=None),
            annotation("Penalty", 63000),
            annotation("Ball out of play", 64000),
            annotation("Yellow->red card", 65000),
        )
    )
    assert [e.action_type for e in case.events] == ["red_card"]
    rows = case.reference_metadata["window_annotations"]
    assert [r["excluded_reason"] for r in rows] == [
        "not_shown",
        "visibility_unspecified",
        "unmapped_label",
        "unmapped_label",
        None,
    ]
    assert rows[-1]["label"] == "Yellow->red card"
    assert rows[-1]["team"] == "home"
    assert case.events[0].player is None and case.events[0].team is None


def test_converted_reference_scores_without_exposing_labels_to_runtime() -> None:
    reference = convert(labels(annotation("Shots off target", 61000), annotation("Goal", 68000)))
    result = ReplayResult(
        observations=[Observation(id="e1", at_s=1, kind="play", description="A shot.")],
        metadata={"source_clip": "clips/fictional.mp4", "source_start_s": 3060, "duration_s": 10},
    )
    score = evaluate_replay(result, reference, tolerance_s=0.5)
    assert score["detection"]["matched"] == 1
    assert score["detection"]["missed_required_ids"] == ["sn-h2-a1"]
    assert score["actions"]["labeled_recall"] == 0  # legacy prose gets no typed-action credit
    assert score["reference_metadata"]["window_annotations"][0]["label"] == "Shots off target"
    assert "reference_metadata" not in result.metadata
    assert result.observations[0].action is None


def test_known_match_cannot_be_assigned_to_the_wrong_official_split() -> None:
    raw = labels(annotation("Goal", 61000))
    convert(raw, upstream_split="valid")
    for split in ("train", "test"):
        with pytest.raises(ValueError, match="disagrees with official split 'valid'"):
            convert(raw, upstream_split=split)


def test_unknown_match_is_explicit_in_inspection_and_cannot_be_imported() -> None:
    raw = labels(annotation("Goal", 61000), match_id="league/season/fictional-match")
    summary = inspect_labels(raw)
    assert summary["official_split"] is None
    assert summary["split_status"] == "unknown_match"
    with pytest.raises(ValueError, match="unknown in the pinned"):
        convert(raw)


@pytest.mark.parametrize("position,time", [(-1, "2 - 01:00"), (1, "3 - 01:00"), (1, "2 - 01:60")])
def test_malformed_source_times_fail_instead_of_becoming_shifted_labels(
    position: int,
    time: str,
) -> None:
    with pytest.raises(ValidationError):
        convert(labels(annotation("Goal", position, gameTime=time)))


@pytest.mark.parametrize(
    "updates",
    [
        {"video_half_offset_s": float("nan")},
        {"duration_s": -1},
        {"duration_s": float("inf")},
        {"half": 3},
        {"upstream_split": "unknown"},
    ],
)
def test_invalid_import_window_is_rejected(updates: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        convert(labels(), **updates)


def test_label_inspection_needs_no_video_model_or_network() -> None:
    raw = labels(annotation("Foul", 1000), annotation("Goal", 3000, gameTime="1 - 00:03"))
    summary = inspect_labels(raw)
    assert summary["annotations"] == 2
    assert summary["label_counts"] == {"Foul": 1, "Goal": 1}
    assert summary["half_counts"] == {"1": 1, "2": 1}
