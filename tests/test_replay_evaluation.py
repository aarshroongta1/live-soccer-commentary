"""Reference isolation, matching, and timing diagnostics for replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import Frame, Observation, ObservedActor
from commentary.replay.agents import RoleAgents
from commentary.replay.contracts import (
    AgentAction,
    ReplayResult,
    SpeechLine,
    SpeechRecord,
    ToolCall,
    TraceEvent,
)
from commentary.replay.evaluation import (
    EvaluationCase,
    ReferenceEvent,
    ReferenceSource,
    evaluate_replay,
    validate_manifest,
)
from commentary.replay.runtime import ReplayRuntime


def case(*events: ReferenceEvent, **updates: object) -> EvaluationCase:
    data: dict[str, object] = {
        "id": "diagnostic",
        "match_id": "fictional-match",
        "split": "development",
        "duration_s": 12,
        "source": ReferenceSource(
            kind="synthetic",
            locator="synthetic://diagnostic",
            provenance="Fictional test labels, not a recorded game.",
            usage_terms="Project-owned test fixture.",
            annotation_status="synthetic",
        ),
        "events": list(events),
    }
    data.update(updates)
    return EvaluationCase.model_validate(data)


def test_budget_limits_are_not_reported_as_provider_errors() -> None:
    result = ReplayResult(
        trace=[
            TraceEvent(event="budget_exhausted", at_s=0, role="caller"),
            TraceEvent(event="turn_error", at_s=1, data={"error": "BudgetExceeded"}),
            TraceEvent(event="observer_error", at_s=2, data={"error": "RuntimeError"}),
        ]
    )
    report = evaluate_replay(result, case())
    assert report["execution"]["budget_stops"] == 2
    assert report["execution"]["errors"] == 1


def event(
    identifier: str, at_s: float, *, kind: str = "goal", required: bool = True
) -> ReferenceEvent:
    return ReferenceEvent.model_validate(
        {"id": identifier, "at_s": at_s, "kind": kind, "required": required}
    )


def observation(identifier: str, at_s: float, *, kind: str = "goal") -> Observation:
    return Observation.model_validate(
        {"id": identifier, "at_s": at_s, "kind": kind, "description": "Visible action."}
    )


def speech(
    identifier: str,
    evidence: list[str],
    start: float,
    end: float,
    *,
    voice: str = "caller",
    interrupted: bool = False,
) -> SpeechRecord:
    return SpeechRecord(
        id=identifier,
        snapshot_version=1,
        line=SpeechLine.model_validate(
            {"voice": voice, "text": "A useful call.", "evidence_ids": evidence}
        ),
        started_s=start,
        ended_s=end,
        interrupted=interrupted,
    )


def test_duplicate_detections_are_false_positives_and_missing_events_reduce_recall() -> None:
    reference = case(event("pass", 2, kind="play"), event("goal", 8))
    result = ReplayResult(
        observations=[observation("goal-a", 8), observation("goal-b", 8.1)],
        speech=[speech("line", ["goal-a"], 8.5, 10)],
        metadata={"speech_timing": "simulated"},
    )

    score = evaluate_replay(result, reference)

    assert score["detection"]["precision"] == 0.5
    assert score["detection"]["required_recall"] == 0.5
    assert score["detection"]["missed_required_ids"] == ["pass"]
    assert score["detection"]["unmatched_prediction_ids"] == ["goal-b"]
    assert score["speech_coverage"]["required_coverage"] == 0.5
    assert score["latency_s"]["p50"] == 0.5
    assert score["scope"]["speech_timing"] == "simulated"
    assert score["scope"]["reference_commentary_used_for_scoring"] is False
    json.dumps(score, allow_nan=False)


def test_matching_uses_augmenting_paths_instead_of_greedy_nearest_only() -> None:
    reference = case(event("first", 2), event("second", 3))
    result = ReplayResult(observations=[observation("shared", 2.5), observation("first-only", 1)])

    score = evaluate_replay(result, reference, tolerance_s=1)

    assert score["detection"]["matched"] == 2
    assert score["detection"]["required_recall"] == 1


def test_optional_reference_cannot_take_the_only_required_match() -> None:
    reference = case(event("optional", 1, required=False), event("required", 2))
    result = ReplayResult(observations=[observation("one", 1)])

    score = evaluate_replay(result, reference, tolerance_s=1)

    assert score["detection"]["matches"][0]["reference_id"] == "required"
    assert score["detection"]["required_recall"] == 1


def test_wrong_event_kind_does_not_match_even_with_identical_timestamp() -> None:
    result = ReplayResult(observations=[observation("replay", 8, kind="replay")])
    score = evaluate_replay(result, case(event("goal", 8)))
    assert score["detection"]["matched"] == 0
    assert score["detection"]["precision"] == 0


def test_future_or_unavailable_evidence_cannot_count_as_spoken_coverage() -> None:
    result = ReplayResult(
        observations=[observation("goal", 8)],
        speech=[speech("too-early", ["goal"], 7, 9), speech("not-ready", ["goal"], 8.5, 9)],
        trace=[
            TraceEvent(
                event="observation_available", at_s=9, data={"observations": [{"id": "goal"}]}
            )
        ],
    )

    score = evaluate_replay(result, case(event("goal", 8)))

    assert score["speech_coverage"]["required_coverage"] == 0
    assert score["grounding"]["future_evidence_citations"] == 2
    assert score["grounding"]["invalid_line_ids"] == ["too-early", "not-ready"]
    assert score["latency_s"]["samples"] == 0


def test_unknown_evidence_and_reference_time_anticipation_are_reported() -> None:
    result = ReplayResult(
        observations=[observation("early-observation", 7)],
        speech=[
            speech("unknown", ["made-up"], 8, 9),
            speech("early", ["early-observation"], 7.5, 8),
        ],
    )
    score = evaluate_replay(result, case(event("goal", 8)))
    assert score["grounding"]["unknown_evidence_citations"] == 1
    assert score["grounding"]["before_reference_event_citations"] == 1
    assert score["speech_coverage"]["required_coverage"] == 0


def test_only_cited_events_count_and_report_earliest_speech_latency() -> None:
    result = ReplayResult(
        observations=[observation("goal", 8), observation("celebration", 10, kind="celebration")],
        speech=[speech("first", ["goal"], 9, 10), speech("later", ["goal"], 11, 12)],
    )
    score = evaluate_replay(
        result, case(event("goal", 8), event("celebration", 10, kind="celebration"))
    )
    assert score["speech_coverage"]["covered_reference_ids"] == ["goal"]
    assert score["latency_s"]["samples"] == 1
    assert score["latency_s"]["maximum"] == 1


def test_zero_duration_speech_does_not_count_as_coverage() -> None:
    result = ReplayResult(
        observations=[observation("goal", 8)], speech=[speech("cancelled", ["goal"], 9, 9)]
    )
    score = evaluate_replay(result, case(event("goal", 8)))
    assert score["speech_coverage"]["required_coverage"] == 0


def test_dialogue_uses_elapsed_durations_and_counts_overlap_once() -> None:
    result = ReplayResult(
        speech=[
            speech("caller", [], 1, 5, interrupted=True),
            speech("analyst", [], 3, 7, voice="analyst"),
            speech("extra", [], 4, 6),
        ],
        trace=[
            TraceEvent(event="turn_timeout", at_s=5),
            TraceEvent(event="observer_timeout", at_s=6),
            TraceEvent(event="turn_stale", at_s=7),
            TraceEvent(event="tool_call", at_s=8),
        ],
    )
    score = evaluate_replay(result, case())
    assert score["dialogue"]["duration_s"] == {"caller": 6, "analyst": 4}
    assert score["dialogue"]["analyst_duration_share"] == 0.4
    assert score["dialogue"]["overlap_s"] == 3
    assert score["dialogue"]["interruptions"] == 1
    assert score["execution"]["timeouts"] == 2
    assert score["execution"]["stale_results"] == 1
    assert score["execution"]["tool_calls"] == 1


def test_resolved_player_accuracy_is_separate_from_event_detection() -> None:
    seen = observation("goal", 8)
    seen.actor = ObservedActor(
        side="home",
        shirt_number=9,
        identity_source="shirt_number",
        action_role="scorer",
        resolved_name="Wrong Player",
    )
    label = event("goal", 8).model_copy(update={"player": "Correct Player"})
    score = evaluate_replay(ReplayResult(observations=[seen]), case(label))
    assert score["detection"]["required_recall"] == 1
    assert score["grounding"]["resolved_player_accuracy"] == 0


def test_rejected_turns_are_separate_from_provider_errors_and_preserve_provenance() -> None:
    reference = case()
    result = ReplayResult(
        trace=[
            TraceEvent(event="turn_rejected", at_s=1),
            TraceEvent(event="turn_rejected", at_s=2),
            TraceEvent(event="turn_error", at_s=3),
            TraceEvent(event="observer_error", at_s=4),
            TraceEvent(event="observation_rejected", at_s=5),
        ]
    )
    score = evaluate_replay(result, reference)
    assert score["execution"]["rejected_turns"] == 2
    assert score["execution"]["errors"] == 2
    assert score["execution"]["rejected_observation_batches"] == 1
    assert score["source"] == reference.source.model_dump(mode="json")


def test_unknown_research_citation_invalidates_coverage_without_claim_semantics() -> None:
    line = speech("invalid-research", ["goal"], 9, 10, voice="analyst")
    line.line.research_ids = ["invented"]
    result = ReplayResult(
        observations=[observation("goal", 8)],
        speech=[line],
        metadata={"research": {"facts": [{"id": "known"}]}},
    )
    score = evaluate_replay(result, case(event("goal", 8)))
    assert score["scope"]["research_claims_verified"] is False
    assert score["scope"]["research_citation_ids_checked"] is True
    assert score["grounding"]["unknown_research_citations"] == 1
    assert score["grounding"]["invalid_line_ids"] == ["invalid-research"]
    assert score["speech_coverage"]["required_coverage"] == 0


def test_missing_research_audit_is_unverified_while_explicit_empty_pack_rejects_ids() -> None:
    line = speech("research", [], 1, 2, voice="analyst")
    line.line.research_ids = ["r1"]
    result = ReplayResult(speech=[line])
    score = evaluate_replay(result, case())
    assert score["scope"]["research_citation_ids_checked"] is False
    assert score["grounding"]["research_unverified_citations"] == 1
    assert score["grounding"]["unknown_research_citations"] == 0
    result.metadata["research"] = None
    score = evaluate_replay(result, case())
    assert score["scope"]["research_citation_ids_checked"] is True
    assert score["grounding"]["unknown_research_citations"] == 1


def test_tool_request_and_completion_traces_are_not_double_counted() -> None:
    result = ReplayResult(
        trace=[
            TraceEvent(event="tool_call", at_s=1),
            TraceEvent(event="tool_result", at_s=2),
        ]
    )
    score = evaluate_replay(result, case())
    assert score["execution"]["tool_calls"] == 1
    assert score["execution"]["tool_requests_traced"] == 1
    assert score["execution"]["tool_results_traced"] == 1


@pytest.mark.asyncio
async def test_failed_agent_turns_keep_executed_tool_counts_through_runtime_and_report() -> None:
    backend = ScriptedBackend()
    for role in ("caller", "analyst"):
        backend.queue(
            f"replay_{role}",
            [
                AgentAction(kind="tool", tool=ToolCall(name="recent_events")),
                AgentAction(
                    kind="speak",
                    line=SpeechLine.model_validate(
                        {"voice": role, "text": "An invalid call.", "evidence_ids": ["unknown"]}
                    ),
                ),
            ],
        )
    roles = RoleAgents(backend, caller_model="scripted", analyst_model="scripted")

    async def observe(_frames: list[Frame], _cutoff: float) -> list[Observation]:
        return [observation("visible", 0, kind="play")]

    # A stationary clock suffices: all work finishes at time zero without speech;
    # runtime wake timers are cancelled by observer/role completion, never slept.
    runtime = ReplayRuntime(observe, roles.run, monotonic=lambda: 0.0)
    result = await runtime.run([Frame(0, b"unused")], duration_s=0)
    assert sum(item.event == "tool_result" for item in roles.audit) == 2
    assert not any(item.event == "tool_result" for item in result.trace)
    result.metadata["role_audit"] = [item.model_dump(mode="json") for item in roles.audit]

    score = evaluate_replay(result, case(event("play", 0, kind="play")))

    assert score["execution"]["rejected_turns"] == 2
    assert score["execution"]["tool_calls"] == 2
    assert score["execution"]["tool_accounting_source"] == "metadata.role_audit"
    assert score["execution"]["role_audit_status"] == "valid"
    assert "tool_result" not in score["execution"]["trace_counts"]


def test_role_audit_does_not_double_count_successful_tools_also_in_runtime_trace() -> None:
    completed = TraceEvent(event="tool_result", at_s=1)
    result = ReplayResult(
        trace=[completed], metadata={"role_audit": [completed.model_dump(mode="json")]}
    )
    score = evaluate_replay(result, case())
    assert score["execution"]["tool_calls"] == 1
    assert score["execution"]["trace_counts"]["tool_result"] == 1


@pytest.mark.parametrize("audit", ["invalid", [{"event": "tool_result", "at_s": "bad"}], []])
def test_invalid_or_incomplete_role_audit_falls_back_to_runtime_trace(audit: object) -> None:
    result = ReplayResult(
        trace=[TraceEvent(event="tool_result", at_s=1)], metadata={"role_audit": audit}
    )
    score = evaluate_replay(result, case())
    assert score["execution"]["tool_calls"] == 1
    assert score["execution"]["tool_accounting_source"] == "runtime.trace"
    assert score["execution"]["role_audit_status"] in {"invalid", "incomplete"}


def test_empty_case_has_undefined_ratios_instead_of_perfect_scores() -> None:
    score = evaluate_replay(ReplayResult(), case())
    assert score["detection"]["precision"] is None
    assert score["detection"]["required_recall"] is None
    assert score["speech_coverage"]["required_coverage"] is None
    assert score["latency_s"]["p95"] is None
    assert score["dialogue"]["analyst_duration_share"] is None


def test_manifest_rejects_match_leakage_but_allows_same_split_clips() -> None:
    first = case()
    second = case(id="other")
    validate_manifest([first, second])
    with pytest.raises(ValueError, match="multiple splits"):
        validate_manifest([first, second.model_copy(update={"split": "test"})])
    with pytest.raises(ValueError, match="duplicate evaluation case"):
        validate_manifest([first, first])


@pytest.mark.parametrize("at_s", [float("nan"), float("inf"), -1, 13])
def test_reference_and_output_timestamps_must_be_clip_relative_and_finite(at_s: float) -> None:
    with pytest.raises(ValueError):
        case(event("bad", at_s))
    with pytest.raises(ValueError, match="reference clip duration"):
        evaluate_replay(ReplayResult(observations=[observation("bad", at_s)]), case())


def test_duplicate_output_ids_are_ambiguous_and_rejected() -> None:
    result = ReplayResult(observations=[observation("same", 1), observation("same", 2)])
    with pytest.raises(ValueError, match="observation IDs"):
        evaluate_replay(result, case())


def test_drain_speech_can_extend_past_clip_end() -> None:
    result = ReplayResult(
        observations=[observation("goal", 12)], speech=[speech("drain", ["goal"], 13, 15)]
    )
    score = evaluate_replay(result, case(event("goal", 12)))
    assert score["latency_s"]["p50"] == 1


def test_reference_schema_rejects_runtime_inputs_and_absolute_time_basis() -> None:
    with pytest.raises(ValidationError):
        case(time_basis="match_absolute")
    with pytest.raises(ValidationError):
        case(frames=[{"at_s": 1, "jpeg": "not a label"}])


def test_synthetic_diagnostic_manifest_is_honestly_labeled() -> None:
    path = Path(__file__).parents[1] / "examples/replay-evaluation/diagnostic.json"
    reference = EvaluationCase.model_validate_json(path.read_text())
    assert reference.source.kind == reference.source.annotation_status == "synthetic"
    assert reference.split == "development"
    validate_manifest([reference])
