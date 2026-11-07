"""Offline behavior checks for independent, tool-using commentary roles."""

from __future__ import annotations

import asyncio
import json
from datetime import date
from typing import Any

import pytest

from commentary.llm.base import Parsed, Usage
from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import Observation, ObservedActor
from commentary.recorded_research import ResearchBrief, ResearchFact
from commentary.replay.agents import RoleAgents
from commentary.replay.contracts import (
    AgentAction,
    ReviewResult,
    RuntimeSnapshot,
    SpeechLine,
    ToolCall,
)


def snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        version=4,
        cutoff_s=8,
        observations=[
            Observation(id=f"e{i}", at_s=i, kind="play", description="A pass moves forward.")
            for i in range(1, 5)
        ],
        research=ResearchBrief(
            fixture="Green v Orange",
            as_of=date(2026, 1, 1),
            facts=[
                ResearchFact(
                    id="r1",
                    subject="Green",
                    text="Green had won two of their previous three league games.",
                    source_url="https://example.org/form",
                    source_date=date(2025, 12, 31),
                )
            ],
        ),
        teams={"home": "Green", "away": "Orange"},
    )


def speak(
    voice: str = "caller", *, evidence: str = "e4", text: str = "A pass forward."
) -> AgentAction:
    return AgentAction(
        kind="speak",
        line=SpeechLine.model_validate({"voice": voice, "text": text, "evidence_ids": [evidence]}),
    )


def agents(backend: ScriptedBackend, **kwargs: object) -> RoleAgents:
    return RoleAgents(backend, "caller-model", "analyst-model", **kwargs)  # type: ignore[arg-type]


async def test_analyst_retains_recent_goal_buildup_during_celebrations() -> None:
    backend = ScriptedBackend()
    events = [
        Observation(
            id=f"e{i}",
            at_s=i,
            kind="goal" if i == 3 else ("play" if i < 3 else "celebration"),
            description="Visible event.",
        )
        for i in range(1, 7)
    ]
    view = RuntimeSnapshot(version=1, cutoff_s=6, observations=events)

    def answer(blocks: Any, _format: Any) -> AgentAction:
        payload = json.loads(blocks[0]["text"])
        assert [o["id"] for o in payload["observations"]] == [f"e{i}" for i in range(1, 7)]
        return speak("analyst", evidence="e2", text="That pass opened the route to goal.")

    backend.register("replay_analyst", answer)
    result = await agents(backend).run("analyst", view)
    assert result.line is not None and result.line.evidence_ids == ["e2"]


async def test_selected_tool_supplies_older_evidence_before_speech() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "replay_caller",
        [AgentAction(kind="tool", tool=ToolCall(name="recent_events")), speak(evidence="e1")],
    )
    result = await agents(backend).run("caller", snapshot())

    assert result.line is not None and result.line.evidence_ids == ["e1"]
    first, second = [json.loads(call.text) for call in backend.calls]
    assert "e1" not in [item["id"] for item in first["observations"]]
    assert second["tool_results"][0]["output"][0]["id"] == "e1"
    assert [item.event for item in result.trace] == ["agent_action", "tool_result", "agent_action"]


async def test_analyst_research_is_returned_with_source_and_used_in_next_action() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "replay_analyst",
        [
            AgentAction(kind="tool", tool=ToolCall(name="research", query="Green")),
            AgentAction(
                kind="speak",
                line=SpeechLine(
                    voice="analyst",
                    text="Green came in with two wins in their previous three league games.",
                    research_ids=["r1"],
                ),
            ),
        ],
    )
    result = await agents(backend).run("analyst", snapshot())
    assert result.line is not None and result.line.research_ids == ["r1"]
    payload = json.loads(backend.calls[-1].text)
    fact = payload["tool_results"][0]["output"][0]
    assert fact["source_url"] == "https://example.org/form"
    assert fact["source_date"] == "2025-12-31"


async def test_caller_never_receives_research_and_cannot_request_it() -> None:
    backend = ScriptedBackend()
    backend.always("replay_caller", AgentAction(kind="tool", tool=ToolCall(name="research")))
    with pytest.raises(ValueError, match="caller cannot use research"):
        await agents(backend, max_steps=1).run("caller", snapshot())
    payload = json.loads(backend.calls[0].text)
    assert payload["allowed_tools"] == ["recent_events", "player_context"]
    assert "previous three league games" not in backend.calls[0].text
    assert "https://example.org/form" not in backend.calls[0].text


async def test_unreadable_player_context_does_not_expose_roster_hints() -> None:
    state = snapshot()
    state.observations[-1].actor = ObservedActor(
        side="home", shirt_number=9, identity_source="shirt_number", action_role="passer"
    )
    backend = ScriptedBackend()
    backend.queue(
        "replay_caller",
        [
            AgentAction(kind="tool", tool=ToolCall(name="player_context")),
            AgentAction(kind="abstain", reason="No readable identity."),
        ],
    )
    result = await agents(backend).run("caller", state)
    assert result.line is None
    first, second = [json.loads(call.text) for call in backend.calls]
    assert "actor" not in first["observations"][-1]
    assert second["tool_results"][0]["output"] == []


async def test_player_tool_links_resolved_identity_to_original_action() -> None:
    state = snapshot()
    state.observations[0].actor = ObservedActor(
        side="home",
        shirt_number=9,
        identity_source="shirt_number",
        action_role="passer",
        resolved_name="Ada Green",
    )
    backend = ScriptedBackend()
    backend.queue(
        "replay_analyst",
        [
            AgentAction(kind="tool", tool=ToolCall(name="player_context", query="Ada")),
            speak("analyst", evidence="e1", text="Ada Green released that pass early."),
        ],
    )
    result = await agents(backend).run("analyst", state)
    assert result.line is not None
    event = json.loads(backend.calls[-1].text)["tool_results"][0]["output"][0]
    assert event["id"] == "e1"
    assert event["actor"] == {"resolved_name": "Ada Green", "action_role": "passer"}


async def test_tool_loop_stops_without_unbounded_model_or_tool_calls() -> None:
    backend = ScriptedBackend()
    backend.always("replay_caller", AgentAction(kind="tool", tool=ToolCall(name="recent_events")))
    result = await agents(backend, max_steps=3).run("caller", snapshot())
    assert result.line is None
    assert result.reason == "action budget exhausted before speech"
    assert len(backend.calls) == 3
    assert sum(item.event == "tool_result" for item in result.trace) == 2


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (speak("analyst"), "wrong voice"),
        (speak(evidence="future"), "evidence not presented"),
        (speak(evidence="e1"), "evidence not presented"),
        (
            AgentAction(kind="speak", line=SpeechLine(voice="analyst", text="An empty claim.")),
            "needs evidence or research",
        ),
    ],
)
async def test_invalid_outputs_are_rejected(action: AgentAction, message: str) -> None:
    backend = ScriptedBackend()
    role = "analyst" if message == "needs evidence or research" else "caller"
    backend.always(f"replay_{role}", action)
    with pytest.raises(ValueError, match=message):
        await agents(backend).run(role, snapshot())


async def test_reviewer_requests_one_revision_and_then_accepts() -> None:
    backend = ScriptedBackend()
    backend.queue("replay_caller", [speak(), speak(text="The pass reaches the forward.")])
    backend.queue(
        "replay_reviewer",
        [
            ReviewResult(accepted=False, feedback="Describe the recipient."),
            ReviewResult(accepted=True),
        ],
    )
    result = await agents(backend, reviewer_model="review-model").run("caller", snapshot())
    assert result.line is not None and result.line.text == "The pass reaches the forward."
    assert json.loads(backend.calls[2].text)["review_feedback"] == "Describe the recipient."
    assert [item.tag for item in backend.calls] == [
        "replay_caller",
        "replay_reviewer",
        "replay_caller",
        "replay_reviewer",
    ]


async def test_reviewer_rejection_discards_draft_after_revision_budget() -> None:
    backend = ScriptedBackend()
    backend.always("replay_caller", speak())
    backend.always("replay_reviewer", ReviewResult(accepted=False))
    result = await agents(backend, reviewer_model="review-model").run("caller", snapshot())
    assert result.line is None
    assert "revision budget exhausted" in result.reason
    assert len(backend.calls) == 4


async def test_concurrent_roles_have_separate_contexts_and_models() -> None:
    backend = ScriptedBackend(latency_s=0.001)
    backend.always("replay_caller", speak())
    backend.always("replay_analyst", AgentAction(kind="abstain", reason="Wait for a lull."))
    roles = agents(backend)
    caller, analyst = await asyncio.gather(
        roles.run("caller", snapshot()), roles.run("analyst", snapshot())
    )
    assert caller.line is not None and analyst.line is None
    assert analyst.reason == "Wait for a lull."
    assert {item.model for item in backend.calls} == {"caller-model", "analyst-model"}
    assert all("reviewer" not in item.tag for item in backend.calls)


async def test_recent_verbatim_repetition_is_rejected() -> None:
    state = snapshot()
    state.recent_lines.append(
        SpeechLine(voice="caller", text="A pass forward.", evidence_ids=["e4"])
    )
    backend = ScriptedBackend()
    backend.always("replay_caller", speak())
    with pytest.raises(ValueError, match="repeated"):
        await agents(backend).run("caller", state)


async def test_same_goal_call_is_allowed_for_a_distinct_goal() -> None:
    state = snapshot()
    state.observations[-1].kind = "goal"
    state.observations[-1].description = "The ball crosses the goal line."
    state.recent_lines.append(
        SpeechLine(voice="caller", text="Goal!", evidence_ids=["earlier-goal"])
    )
    backend = ScriptedBackend()
    backend.always("replay_caller", speak(text="Goal!"))

    result = await agents(backend).run("caller", state)

    assert result.line is not None
    assert result.line.text == "Goal!" and result.line.evidence_ids == ["e4"]


async def test_same_research_fact_cannot_repeat_with_normalized_whitespace() -> None:
    state = snapshot()
    state.recent_lines.append(
        SpeechLine(voice="analyst", text="Green had good form.", research_ids=["r1"])
    )
    backend = ScriptedBackend()
    backend.queue(
        "replay_analyst",
        [
            AgentAction(kind="tool", tool=ToolCall(name="research")),
            AgentAction(
                kind="speak",
                line=SpeechLine(
                    voice="analyst", text=" GREEN  had good form. ", research_ids=["r1"]
                ),
            ),
        ],
    )
    with pytest.raises(ValueError, match="repeated"):
        await agents(backend).run("analyst", state)


async def test_research_citation_requires_tool_retrieval_even_for_analyst() -> None:
    backend = ScriptedBackend()
    backend.always(
        "replay_analyst",
        AgentAction(
            kind="speak",
            line=SpeechLine(voice="analyst", text="Green had good form.", research_ids=["r1"]),
        ),
    )
    with pytest.raises(ValueError, match="research not presented"):
        await agents(backend).run("analyst", snapshot())


async def test_snapshot_is_revalidated_if_nested_evidence_was_mutated_into_future() -> None:
    state = snapshot()
    state.observations[-1].at_s = 100
    backend = ScriptedBackend()
    with pytest.raises(ValueError, match="chronological and visible"):
        await agents(backend).run("caller", state)
    assert backend.calls == []


async def test_rejected_paid_action_keeps_usage_and_response_in_public_audit() -> None:
    class CostBackend(ScriptedBackend):
        async def parse(self, **kwargs: Any) -> Parsed[Any]:
            parsed = await super().parse(**kwargs)
            return Parsed(
                value=parsed.value,
                model=parsed.model,
                usage=Usage(input_tokens=31, output_tokens=12, cost_usd=0.004),
            )

    backend = CostBackend()
    backend.always("replay_caller", speak(evidence="unknown"))
    roles = agents(backend)
    with pytest.raises(ValueError, match="evidence not presented"):
        await roles.run("caller", snapshot())

    assert len(roles.audit) == 1
    record = roles.audit[0]
    assert record.event == "agent_action" and record.role == "caller"
    assert record.at_s == 8
    assert record.data["timestamp_basis"] == "snapshot_cutoff"
    assert record.data["model"] == "caller-model"
    assert record.data["usage"]["cost_usd"] == 0.004
    assert record.data["action"]["line"]["evidence_ids"] == ["unknown"]
