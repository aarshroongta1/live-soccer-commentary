"""Reference-only SoccerNet import. No model, media downloading, or runtime inputs."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from commentary.football import ActionType
from commentary.replay.evaluation import EvaluationCase, ReferenceEvent, ReferenceSource

Split = Literal["train", "valid", "test"]


class Annotation(BaseModel):
    gameTime: str = Field(pattern=r"^[12]\s*-\s*\d{1,3}:[0-5]\d$")
    position: int = Field(ge=0)
    label: str = Field(min_length=1)
    visibility: Literal["visible", "not shown"] | None = None
    team: str | None = None

    @property
    def half(self) -> int:
        return int(self.gameTime.split("-", 1)[0])


class SoccerNetLabels(BaseModel):
    UrlLocal: str = Field(min_length=1)
    annotations: list[Annotation]


# Do not guess penalty award vs kick, or silently turn ball-out into a restart.
EVENTS: dict[str, ActionType] = {
    "Kick-off": "kick_off",
    "Goal": "goal",
    "Substitution": "substitution",
    "Offside": "offside",
    "Shots on target": "shot",
    "Shots off target": "shot",
    "Clearance": "clearance",
    "Throw-in": "throw_in",
    "Foul": "foul",
    "Indirect free-kick": "free_kick",
    "Direct free-kick": "free_kick",
    "Corner": "corner",
    "Yellow card": "yellow_card",
    "Red card": "red_card",
    "Yellow->red card": "red_card",
}


def inspect_labels(raw: bytes) -> dict[str, object]:
    labels = SoccerNetLabels.model_validate_json(raw)
    return {
        "match_id": labels.UrlLocal.rstrip("/"),
        "labels_sha256": hashlib.sha256(raw).hexdigest(),
        "annotations": len(labels.annotations),
        "label_counts": dict(sorted(Counter(a.label for a in labels.annotations).items())),
        "half_counts": dict(Counter(str(a.half) for a in labels.annotations)),
        "visibility_counts": dict(
            Counter(a.visibility or "unspecified" for a in labels.annotations)
        ),
    }


def import_case(
    raw: bytes,
    *,
    half: Literal[1, 2],
    upstream_split: Split,
    case_id: str,
    source: ReferenceSource,
    duration_s: float,
    video_half_offset_s: float,
    label_origin: str,
) -> EvaluationCase:
    """Map half-video milliseconds to selected-clip seconds with an explicit offset.

    video_half_offset_s is the source-video timestamp corresponding to timestamp
    zero of the original dataset's half video. It is not necessarily kickoff time.
    Interval is [start, end), so neighboring windows do not duplicate boundary events.
    """
    labels = SoccerNetLabels.model_validate_json(raw)
    if half not in {1, 2}:
        raise ValueError("unsupported half")
    if upstream_split not in {"train", "valid", "test"}:
        raise ValueError("an official dataset split is required")
    if not math.isfinite(video_half_offset_s):
        raise ValueError("video half offset must be finite")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration must be finite and positive")
    source = ReferenceSource.model_validate(source.model_dump())
    if source.kind != "broadcast":
        raise ValueError("SoccerNet references require a broadcast source")
    # Published labels do not independently verify this user's local video alignment.
    source = source.model_copy(update={"annotation_status": "unreviewed"})
    mapping = EVENTS
    events: list[ReferenceEvent] = []
    audit: list[dict[str, object]] = []
    for index, item in enumerate(labels.annotations):
        at_s = item.position / 1000 + video_half_offset_s - source.start_s
        if item.half != half or not 0 <= at_s < duration_s:
            continue
        action = mapping.get(item.label)
        reason = None
        if item.visibility == "not shown":
            reason = "not_shown"
        elif item.visibility is None:
            reason = "visibility_unspecified"
        elif action is None:
            reason = "unmapped_label"
        row: dict[str, object] = {
            "annotation_index": index,
            "gameTime": item.gameTime,
            "position": item.position,
            "label": item.label,
            "visibility": item.visibility,
            "team": item.team,
            "clip_at_s": at_s,
            "mapped_action": action,
            "excluded_reason": reason,
        }
        audit.append(row)
        if reason is not None or action is None:
            continue
        events.append(
            ReferenceEvent(
                id=f"sn-h{half}-a{index}",
                at_s=at_s,
                kind="goal" if action == "goal" else "play",
                action_type=action,
                annotation_note=(
                    f"Published events-v2 label: {item.label}; local alignment awaits review."
                ),
            )
        )
    events.sort(key=lambda item: item.at_s)
    splits: dict[Split, Literal["development", "validation", "test"]] = {
        "train": "development",
        "valid": "validation",
        "test": "test",
    }
    return EvaluationCase(
        id=case_id,
        match_id=labels.UrlLocal.rstrip("/"),
        split=splits[upstream_split],
        duration_s=duration_s,
        source=source,
        events=events,
        action_labels_complete=False,
        reference_metadata={
            "dataset": "SoccerNet",
            "task": "events-v2",
            "upstream_split": upstream_split,
            "split_assignment_verified": False,
            "half": half,
            "label_origin": label_origin,
            "labels_sha256": hashlib.sha256(raw).hexdigest(),
            "video_half_offset_s": video_half_offset_s,
            "timing": "position milliseconds in original half video, plus explicit video offset",
            "interval": "start inclusive, end exclusive",
            "window_annotations": audit,
            "alignment_reviewed": False,
            "metric_scope": "project diagnostics; not official SoccerNet mAP",
        },
    )
