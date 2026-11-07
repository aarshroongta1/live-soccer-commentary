"""Reference-only diagnostics for replay outputs, never an agent input.

Event matching is a timestamp/kind diagnostic, not a semantic judgment of spoken
words. Reference labels must be reviewed before using scores to promote a prompt.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from commentary.recorded_demo import Observation
from commentary.replay.contracts import ReplayResult, SpeechRecord, TraceEvent


class ReferenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["synthetic", "broadcast"]
    locator: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    usage_terms: str = Field(min_length=1)
    annotation_status: Literal["synthetic", "reviewed", "unreviewed"]

    @model_validator(mode="after")
    def synthetic_is_explicit(self) -> ReferenceSource:
        if (self.kind == "synthetic") != (self.annotation_status == "synthetic"):
            raise ValueError("synthetic sources require synthetic annotation status")
        return self


class ReferenceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    at_s: float = Field(ge=0, allow_inf_nan=False)
    kind: Literal["play", "goal", "celebration", "replay"]
    player: str | None = None
    team: str | None = None
    required: bool = True


class ReferenceCommentary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at_s: float = Field(ge=0, allow_inf_nan=False)
    text: str = Field(min_length=1)
    voice: str | None = None


class EvaluationCase(BaseModel):
    """Labels use clip-relative time, not absolute match or source-file time."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    match_id: str = Field(min_length=1)
    split: Literal["development", "validation", "test"]
    duration_s: float = Field(gt=0, allow_inf_nan=False)
    time_basis: Literal["clip_relative"] = "clip_relative"
    source: ReferenceSource
    events: list[ReferenceEvent] = Field(default_factory=list, max_length=500)
    commentary: list[ReferenceCommentary] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_reference_timeline(self) -> EvaluationCase:
        ids = [item.id for item in self.events]
        if len(ids) != len(set(ids)):
            raise ValueError("reference event IDs must be unique")
        for label, items in (("events", self.events), ("commentary", self.commentary)):
            previous = -1.0
            for item in items:
                if item.at_s < previous or item.at_s > self.duration_s:
                    raise ValueError(f"{label} must be chronological and within clip duration")
                previous = item.at_s
        return self


def validate_manifest(cases: Sequence[EvaluationCase]) -> None:
    """Reject duplicate cases and alternate clips from a match crossing splits."""
    ids: set[str] = set()
    match_splits: dict[str, str] = {}
    for case in cases:
        # Revalidate in case callers used model_copy(update=...) or mutated a model.
        case = EvaluationCase.model_validate(case.model_dump(mode="python"))
        if case.id in ids:
            raise ValueError(f"duplicate evaluation case ID: {case.id}")
        ids.add(case.id)
        if case.match_id in match_splits and match_splits[case.match_id] != case.split:
            raise ValueError(f"match {case.match_id} occurs in multiple splits")
        match_splits[case.match_id] = case.split


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _match_events(
    observations: list[Observation], references: list[ReferenceEvent], tolerance_s: float
) -> dict[int, int]:
    """Maximum-cardinality one-to-one matching, preserving required matches first.

    Candidate edges require kind/time agreement. Iterating required references first
    ensures optional references cannot displace a matched required reference. Among
    candidate edges prefer closer timestamps, then stable observation order. This
    does not claim to minimize total timestamp error among all maximum matchings.
    """
    candidates = {
        i: sorted(
            (
                j
                for j, observation in enumerate(observations)
                if observation.kind == reference.kind
                and abs(observation.at_s - reference.at_s) <= tolerance_s
            ),
            key=lambda j: (abs(observations[j].at_s - reference.at_s), j),
        )
        for i, reference in enumerate(references)
    }
    prediction_to_reference: dict[int, int] = {}

    def augment(reference_index: int, visited: set[int]) -> bool:
        for prediction_index in candidates[reference_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            old_reference = prediction_to_reference.get(prediction_index)
            if old_reference is None or augment(old_reference, visited):
                prediction_to_reference[prediction_index] = reference_index
                return True
        return False

    for index in sorted(range(len(references)), key=lambda i: not references[i].required):
        augment(index, set())
    return prediction_to_reference


def _availability(result: ReplayResult) -> dict[str, float]:
    available: dict[str, float] = {}
    for event in result.trace:
        if event.event != "observation_available":
            continue
        observations = event.data.get("observations", [])
        if not isinstance(observations, list):
            continue
        for observation in observations:
            if isinstance(observation, dict) and isinstance(observation.get("id"), str):
                identifier = observation["id"]
                available[identifier] = min(available.get(identifier, event.at_s), event.at_s)
    return available


def _overlap(speech: list[SpeechRecord]) -> float:
    edges = sorted(
        (point, delta)
        for record in speech
        if record.ended_s > record.started_s
        for point, delta in ((record.started_s, 1), (record.ended_s, -1))
    )
    active = 0
    previous = 0.0
    seconds = 0.0
    for at_s, delta in edges:
        if active > 1:
            seconds += at_s - previous
        active += delta
        previous = at_s
    return seconds


def _research_ids(metadata: dict[str, Any]) -> set[str] | None:
    """Distinguish an explicit empty brief from missing research provenance."""
    if "research" not in metadata:
        return None
    research = metadata["research"]
    if research is None:
        return set()
    if not isinstance(research, dict) or not isinstance(research.get("facts"), list):
        return None
    identifiers: set[str] = set()
    for fact in research["facts"]:
        if not isinstance(fact, dict) or not isinstance(fact.get("id"), str):
            return None
        identifiers.add(fact["id"])
    return identifiers


def _tool_accounting(result: ReplayResult) -> tuple[Counter[str], str, str]:
    """Use the complete role audit so rejected/cancelled turns retain tool costs.

    Successful role traces also occur in runtime.trace; choose a single source,
    rather than summing both. A malformed or demonstrably incomplete role audit
    falls back to runtime traces, which remain visible in trace_counts either way.
    """
    runtime_counts = Counter(item.event for item in result.trace)
    raw_audit = result.metadata.get("role_audit")
    if raw_audit is None:
        return runtime_counts, "runtime.trace", "absent"
    if not isinstance(raw_audit, list):
        return runtime_counts, "runtime.trace", "invalid"
    try:
        audit = [TraceEvent.model_validate(item) for item in raw_audit]
    except (ValueError, TypeError):
        return runtime_counts, "runtime.trace", "invalid"
    audit_counts = Counter(item.event for item in audit)
    if any(audit_counts[name] < runtime_counts[name] for name in ("tool_call", "tool_result")):
        return runtime_counts, "runtime.trace", "incomplete"
    return audit_counts, "metadata.role_audit", "valid"


def evaluate_replay(
    result: ReplayResult, case: EvaluationCase, tolerance_s: float = 2.0
) -> dict[str, Any]:
    """Score a completed output without giving labels to any runtime agent.

    Duplicate detections with different IDs count as false positives. Duplicate IDs,
    non-finite timestamps, and observations outside the reference clip are rejected
    because they make citation/timeline interpretation ambiguous.
    """
    validate_manifest([case])
    case = EvaluationCase.model_validate(case.model_dump(mode="python"))
    if not math.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("tolerance_s must be finite and at least zero")
    ids = [item.id for item in result.observations]
    if len(ids) != len(set(ids)):
        raise ValueError("output observation IDs must be unique")
    if len({item.id for item in result.speech}) != len(result.speech):
        raise ValueError("output speech IDs must be unique")
    for observation in result.observations:
        if not math.isfinite(observation.at_s) or not 0 <= observation.at_s <= case.duration_s:
            raise ValueError("output observation must be within the reference clip duration")
    # Revalidate timing even when a record was produced through model_copy.
    speech = [SpeechRecord.model_validate(item.model_dump(mode="python")) for item in result.speech]
    matches = _match_events(result.observations, case.events, tolerance_s)
    observation_by_id = {item.id: item for item in result.observations}
    reference_by_evidence = {
        result.observations[prediction].id: reference for prediction, reference in matches.items()
    }
    matched_reference = set(matches.values())
    required = {i for i, item in enumerate(case.events) if item.required}
    missed = sorted(required - matched_reference)
    unmatched = [i for i in range(len(result.observations)) if i not in matches]
    availability = _availability(result)
    research_ids = _research_ids(result.metadata)
    coverage: dict[int, float] = {}
    invalid_lines: list[str] = []
    unsupported_citations = 0
    future_citations = 0
    early_reference_citations = 0
    unverified_availability = 0
    unknown_research_citations = 0
    unverified_research_citations = 0
    for record in speech:
        invalid = record.line.voice == "caller" and (
            not record.line.evidence_ids or bool(record.line.research_ids)
        )
        if record.line.voice == "analyst" and not (
            record.line.evidence_ids or record.line.research_ids
        ):
            invalid = True
        references: set[int] = set()
        for identifier in set(record.line.research_ids):
            if research_ids is None:
                unverified_research_citations += 1
            elif identifier not in research_ids:
                unknown_research_citations += 1
                invalid = True
        for identifier in set(record.line.evidence_ids):
            cited_observation = observation_by_id.get(identifier)
            if cited_observation is None:
                unsupported_citations += 1
                invalid = True
                continue
            ready = availability.get(identifier)
            if ready is None:
                unverified_availability += 1
            if cited_observation.at_s > record.started_s or (
                ready is not None and ready > record.started_s
            ):
                future_citations += 1
                invalid = True
            reference_index = reference_by_evidence.get(identifier)
            if reference_index is not None:
                if case.events[reference_index].at_s > record.started_s:
                    early_reference_citations += 1
                    invalid = True
                references.add(reference_index)
        if invalid:
            invalid_lines.append(record.id)
        elif record.ended_s > record.started_s:
            for index in references:
                coverage[index] = min(coverage.get(index, record.started_s), record.started_s)
    latencies = [start - case.events[index].at_s for index, start in coverage.items()]
    required_covered = required & coverage.keys()
    durations = {
        role: sum(item.ended_s - item.started_s for item in speech if item.line.voice == role)
        for role in ("caller", "analyst")
    }
    total_duration = sum(durations.values())
    event_counts = Counter(item.event for item in result.trace)
    tool_counts, tool_accounting_source, role_audit_status = _tool_accounting(result)
    identity_pairs = [
        (result.observations[prediction], case.events[reference])
        for prediction, reference in matches.items()
        if case.events[reference].player is not None
    ]
    identity_correct = sum(
        observation.actor is not None
        and observation.actor.resolved_name is not None
        and observation.actor.resolved_name.casefold() == str(reference.player).casefold()
        for observation, reference in identity_pairs
    )
    return {
        "schema_version": 1,
        "case_id": case.id,
        "match_id": case.match_id,
        "split": case.split,
        "reference_status": case.source.annotation_status,
        "source": case.source.model_dump(mode="json"),
        "scope": {
            "event_match": "one-to-one kind and timestamp; no semantic text scoring",
            "unmatched_detections": "candidate false positives relative to supplied labels",
            "speech_timing": result.metadata.get("speech_timing", "unspecified"),
            "coverage": "cited matched evidence with positive speech duration; words not verified",
            "reference_commentary_used_for_scoring": False,
            "research_claims_verified": False,
            "research_citation_ids_checked": research_ids is not None,
            "requires_human_review": ["spoken factual claims", "analysis", "emotion", "delivery"],
        },
        "detection": {
            "tolerance_s": tolerance_s,
            "predictions": len(result.observations),
            "reference_events": len(case.events),
            "required_events": len(required),
            "matched": len(matches),
            "precision": _ratio(len(matches), len(result.observations)),
            "required_recall": _ratio(len(required & matched_reference), len(required)),
            "missed_required_ids": [case.events[i].id for i in missed],
            "missed_required": len(missed),
            "false_positive_detections": len(unmatched),
            "unmatched_prediction_ids": [result.observations[i].id for i in unmatched],
            "matches": [
                {
                    "evidence_id": result.observations[p].id,
                    "reference_id": case.events[r].id,
                    "timestamp_error_s": result.observations[p].at_s - case.events[r].at_s,
                }
                for p, r in sorted(matches.items())
            ],
        },
        "speech_coverage": {
            "covered_reference_ids": [case.events[i].id for i in sorted(coverage)],
            "required_coverage": _ratio(len(required_covered), len(required)),
            "missed_required_ids": [case.events[i].id for i in sorted(required - coverage.keys())],
        },
        "grounding": {
            "invalid_line_ids": invalid_lines,
            "unknown_evidence_citations": unsupported_citations,
            "future_evidence_citations": future_citations,
            "before_reference_event_citations": early_reference_citations,
            "availability_unverified_citations": unverified_availability,
            "unknown_research_citations": unknown_research_citations,
            "research_unverified_citations": unverified_research_citations,
            "matched_events_with_player_labels": len(identity_pairs),
            "correct_resolved_players": identity_correct,
            "resolved_player_accuracy": _ratio(identity_correct, len(identity_pairs)),
        },
        "latency_s": {
            "samples": len(latencies),
            "p50": _quantile(latencies, 0.5),
            "p95": _quantile(latencies, 0.95),
            "maximum": max(latencies, default=None),
            "basis": "first eligible speech start minus matched reference event time",
        },
        "dialogue": {
            "turns": dict(Counter(item.line.voice for item in speech)),
            "duration_s": durations,
            "analyst_duration_share": _ratio(durations["analyst"], total_duration),
            "overlap_s": _overlap(speech),
            "interruptions": sum(item.interrupted for item in speech),
        },
        "execution": {
            "trace_counts": dict(event_counts),
            "timeouts": event_counts["turn_timeout"] + event_counts["observer_timeout"],
            "stale_results": event_counts["turn_stale"],
            "tool_calls": max(tool_counts["tool_call"], tool_counts["tool_result"]),
            "tool_requests_traced": tool_counts["tool_call"],
            "tool_results_traced": tool_counts["tool_result"],
            "tool_accounting_source": tool_accounting_source,
            "role_audit_status": role_audit_status,
            "rejected_turns": event_counts["turn_rejected"],
            "rejected_observation_batches": event_counts["observation_rejected"],
            "errors": event_counts["turn_error"] + event_counts["observer_error"],
        },
    }
