"""Fast, no-network checks for the bounded recorded-clip demo."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from commentary.llm.base import LLMError
from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import (
    WRITER_SYSTEM,
    Frame,
    JointScript,
    Observation,
    ObservedActor,
    ObserverResult,
    PlanLine,
    RecordedDemoConfig,
    ScriptLine,
    _observe,
    _observer_windows,
    _resolve_observation_actors,
    _stamp_frame,
    _validate_joint_script,
    run_recorded_demo,
    safe_pack,
    validate_plan_parts,
)
from commentary.recorded_research import ResearchBrief, ResearchDraft, ResearchFact


class SearchScriptedBackend(ScriptedBackend):
    """A scripted backend with the narrow ``extra_params`` search hook."""

    def __init__(self) -> None:
        super().__init__()
        self.extra_params: dict[str, Any] = {}
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.tools_seen.append(deepcopy(self.extra_params.get("tools")))
        return await super().parse(**kwargs)


def observations() -> list[Observation]:
    return [
        Observation(id="o1", at_s=1, kind="play", description="The move begins."),
        Observation(id="o2", at_s=7, kind="play", description="A pass finds space."),
        Observation(id="o3", at_s=13, kind="goal", description="The ball is in the net."),
        Observation(id="o4", at_s=19, kind="celebration", description="Players celebrate."),
    ]


def test_recorded_defaults_use_the_openai_model_roles() -> None:
    config = RecordedDemoConfig(clip=Path("synthetic.mp4"))
    assert config.observer_model == "gpt-6-astra"
    assert config.writer_model == config.research_model == "gpt-5.6-terra"


def joint_script(*, cite_research: bool = False) -> JointScript:
    return JointScript(
        lines=[
            ScriptLine(
                at_s=1,
                voice="caller",
                text="Moves the ball forward downfield.",
                evidence_ids=["o1"],
            ),
            ScriptLine(
                at_s=3.1,
                voice="analyst",
                text="The first shape stays compact.",
                evidence_ids=["o1"],
            ),
            ScriptLine(
                at_s=7,
                voice="caller",
                text="The attack builds around the area.",
                evidence_ids=["o2"],
            ),
            ScriptLine(
                at_s=9.5,
                voice="analyst",
                text="That challenge briefly slows the move.",
                evidence_ids=["o2"],
            ),
            ScriptLine(at_s=13, voice="caller", text="Goal! The ball is in.", evidence_ids=["o3"]),
            ScriptLine(
                at_s=15.2,
                voice="analyst",
                text="Green had lost one in eleven before this."
                if cite_research
                else "The finish changes the game's rhythm.",
                evidence_ids=[] if cite_research else ["o3"],
                research_ids=["r1"] if cite_research else [],
            ),
            ScriptLine(
                at_s=19,
                voice="caller",
                text="Celebration follows the decisive moment.",
                evidence_ids=["o4"],
            ),
        ]
    )


@pytest.mark.asyncio
async def test_writer_prompt_prioritizes_grounded_names_in_initial_and_repair_calls() -> None:
    backend = ScriptedBackend()
    invalid = joint_script()
    invalid.lines[1] = invalid.lines[1].model_copy(update={"text": "Too short."})
    backend.queue("recorded_writer", [invalid, joint_script()])
    named_observations = observations()
    names = ["Ari Vale", "Bea Cruz", "Cato Reid", "Dani Fox"]
    for number, (observation, name) in enumerate(
        zip(named_observations, names, strict=True), start=7
    ):
        observation.actor = ObservedActor(
            side="home",
            shirt_number=number,
            identity_source="shirt_number",
            action_role="visible_player",
            resolved_name=name,
        )
    named_pack = safe_pack(
        {
            "home": {
                "starters": [
                    {"name": name, "number": number} for number, name in enumerate(names, start=7)
                ]
            }
        }
    )

    await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            reused_observer=ObserverResult(observations=named_observations),
            pack=named_pack,
        ),
        frames=[],
    )

    writer_calls = backend.calls_tagged("recorded_writer")
    assert len(writer_calls) == 2
    assert all(call.system == WRITER_SYSTEM for call in writer_calls)
    normalized_prompts = [" ".join(call.system.split()) for call in writer_calls]
    for phrase in (
        "actor.resolved_name",
        "natural, prominent named moments",
        "Prefer a supported named action to a generic team line",
        "running or celebrating is visible",
        "Integrate distinct observed identities across the clip",
        "meaningful post-goal named reaction",
        "Do not impose a player-name quota",
        "never transfer a celebration identity to an earlier action",
        'Never say "first time" or "first-time"',
        "For both caller and analyst",
        "kit-color or stripe labels",
        '"runner", "player", or "attacker"',
        '"Argentina"',
        '"the French"',
        "action-led sentence that omits the person",
    ):
        assert all(phrase in prompt for prompt in normalized_prompts)
    payloads = [json.loads(call.text) for call in writer_calls]
    assert [item["actor"]["resolved_name"] for item in payloads[0]["observations"]] == names
    assert payloads[1]["observations"] == payloads[0]["observations"]
    assert payloads[1]["invalid_draft"] == invalid.model_dump(mode="json")


def test_observer_windows_keep_global_timestamps_and_isolate_frames() -> None:
    frames = [
        Frame(at_s=0, jpeg=b"0"),
        Frame(at_s=5.9, jpeg=b"1"),
        Frame(at_s=6, jpeg=b"2"),
        Frame(at_s=11.9, jpeg=b"3"),
    ]
    windows = _observer_windows(frames, 40)
    assert [(start, end) for start, end, _frames in windows] == [(0, 6), (6, 12)]
    assert [frame.at_s for frame in windows[0][2]] == [0, 5.9]
    assert [frame.at_s for frame in windows[1][2]] == [6, 11.9]


def test_sampled_frame_timestamp_is_in_top_border() -> None:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    stamped = _stamp_frame(image, 12.5)
    assert stamped.shape[0] > image.shape[0]
    assert stamped.shape[1] == image.shape[1]
    encoded = cv2.imencode(".jpg", stamped)[1]
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape[0] == stamped.shape[0]
    assert int(decoded[:20].mean()) > 0


@pytest.mark.asyncio
async def test_observer_merges_window_ids_and_accumulates_audit(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    backend.queue(
        "recorded_observer",
        [
            ObserverResult(
                observations=[Observation(id="repeat", at_s=1, kind="play", description="A pass.")]
            ),
            ObserverResult(
                observations=[Observation(id="repeat", at_s=13, kind="goal", description="Net.")]
            ),
        ],
    )
    config = RecordedDemoConfig(
        clip=Path("synthetic.mp4"), duration_s=25, audit_dir=tmp_path / "run"
    )
    state = await _observe(
        {
            "frames": [Frame(at_s=1, jpeg=b"1"), Frame(at_s=13, jpeg=b"2")],
            "config": config,
            "audit": {},
        },
        backend=backend,
    )
    assert [item.id for item in state["observations"]] == ["repeat", "w02-o01"]
    assert state["audit"]["observe"]["usage"]["windows"] == 2
    assert (tmp_path / "run" / "observe-window-01.json").is_file()
    assert (tmp_path / "run" / "observe-window-02.json").is_file()


@pytest.mark.asyncio
async def test_observer_rejects_relative_reset_timestamp() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "recorded_observer",
        [
            ObserverResult(
                observations=[Observation(id="first", at_s=1, kind="play", description="A pass.")]
            ),
            ObserverResult(
                observations=[Observation(id="reset", at_s=1, kind="goal", description="Net.")]
            ),
        ],
    )
    with pytest.raises(ValueError, match="outside 6..12s"):
        await _observe(
            {
                "frames": [Frame(at_s=1, jpeg=b"1"), Frame(at_s=7, jpeg=b"2")],
                "config": RecordedDemoConfig(clip=Path("synthetic.mp4"), duration_s=25),
                "audit": {},
            },
            backend=backend,
        )


def research() -> ResearchBrief:
    return ResearchBrief(
        fixture="Green v Orange",
        as_of="2025-12-05",
        facts=[
            ResearchFact(
                id="r1",
                subject="Green",
                text="Green had lost one of their previous eleven matches in all competitions.",
                source_url="https://example.com/preview",
                source_date="2025-12-05",
            )
        ],
    )


@pytest.mark.asyncio
async def test_graph_order_and_audited_offline_plan() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "recorded_observer",
        [
            ObserverResult(observations=observations()[:1]),
            ObserverResult(observations=observations()[1:2]),
            ObserverResult(observations=observations()[2:3]),
            ObserverResult(observations=observations()[3:]),
        ],
    )
    backend.always("recorded_writer", joint_script())
    plan, audit = await run_recorded_demo(
        backend,
        RecordedDemoConfig(clip=Path("synthetic.mp4"), duration_s=25),
        frames=[
            Frame(at_s=1, jpeg=b"test"),
            Frame(at_s=7, jpeg=b"test"),
            Frame(at_s=13, jpeg=b"test"),
            Frame(at_s=19, jpeg=b"test"),
        ],
    )
    assert [call.tag for call in backend.calls] == [
        "recorded_observer",
        "recorded_observer",
        "recorded_observer",
        "recorded_observer",
        "recorded_writer",
    ]
    assert plan.mode == "recorded_clip" and plan.metadata["offline"] is True
    assert list(audit) == ["observe", "script"]
    assert len([line for line in plan.lines if line.voice == "analyst"]) == 3
    assert plan.lines[4].text == "Goal! The ball is in."


@pytest.mark.asyncio
async def test_joint_writer_is_the_only_writing_call() -> None:
    backend = ScriptedBackend()
    backend.always("recorded_writer", joint_script())
    plan, audit = await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            reused_observer=ObserverResult(observations=observations()),
        ),
        frames=[],
    )
    assert [call.tag for call in backend.calls] == ["recorded_writer"]
    assert audit["script"]["result"] == joint_script().model_dump(mode="json")
    assert "retry" not in audit["script"]
    assert {line.voice for line in plan.lines} == {"caller", "analyst"}


@pytest.mark.asyncio
async def test_writer_receives_resolved_actor_but_not_full_roster() -> None:
    backend = ScriptedBackend()
    backend.always("recorded_writer", joint_script())
    pack = safe_pack(
        {
            "home": {
                "name": "Betis",
                "kit": "green",
                "starters": [{"name": "Antony", "number": 7}],
            },
            "away": {
                "name": "Barcelona",
                "kit": "orange",
                "starters": [{"name": "Ferran Torres", "number": 7}],
            },
        }
    )
    named = observations()
    named[0] = named[0].model_copy(
        update={
            "actor": ObservedActor(
                side="away",
                shirt_number=7,
                identity_source="shirt_number",
                action_role="ball_carrier",
            )
        }
    )
    await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            pack=pack,
            reused_observer=ObserverResult(observations=named),
        ),
        frames=[],
    )
    writer_payload = backend.calls_tagged("recorded_writer")[0].text
    assert "Ferran Torres" in writer_payload
    assert "Antony" not in writer_payload


@pytest.mark.asyncio
async def test_reused_research_is_cited_by_analyst_not_leaked_to_caller() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "recorded_observer",
        [
            ObserverResult(observations=observations()[:1]),
            ObserverResult(observations=observations()[1:2]),
            ObserverResult(observations=observations()[2:3]),
            ObserverResult(observations=observations()[3:]),
        ],
    )
    backend.always("recorded_writer", joint_script(cite_research=True))
    plan, audit = await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            research_brief=research(),
            research_source="brief.json",
        ),
        frames=[
            Frame(at_s=1, jpeg=b"test"),
            Frame(at_s=7, jpeg=b"test"),
            Frame(at_s=13, jpeg=b"test"),
            Frame(at_s=19, jpeg=b"test"),
        ],
    )
    writer_text = backend.calls_tagged("recorded_writer")[0].text
    assert "previous eleven" in writer_text
    assert all(not line.research_ids for line in plan.lines if line.voice == "caller")
    assert any(line.research_ids == ["r1"] for line in plan.lines if line.voice == "analyst")
    assert plan.metadata["research"]["facts"][0]["id"] == "r1"
    assert audit["research"]["usage"]["reused"] is True


@pytest.mark.asyncio
async def test_fresh_research_uses_three_searches_and_survives_cached_observer() -> None:
    backend = SearchScriptedBackend()
    backend.always("recorded_research", ResearchDraft(facts=research().facts))
    backend.always("recorded_writer", joint_script())
    plan, audit = await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            reused_observer=ObserverResult(observations=observations()),
            research_fixture="Green v Orange",
            research_as_of="2025-12-05",
        ),
        frames=[],
    )
    assert backend.tools_seen[0] is not None
    assert backend.tools_seen[0][0]["max_uses"] == 3
    assert backend.extra_params == {}
    assert {"research", "observe", "script"} == set(audit)
    assert plan.metadata["research"]["fixture"] == "Green v Orange"
    assert plan.metadata["research"]["as_of"] == "2025-12-05"
    assert audit["research"]["raw_result"]["facts"][0]["id"] == "r1"
    assert audit["research"]["result"]["fixture"] == "Green v Orange"


@pytest.mark.asyncio
async def test_invalid_research_is_archived_and_restores_search_hook(tmp_path: Path) -> None:
    backend = SearchScriptedBackend()
    future = ResearchFact(
        id="future",
        subject="Green",
        text="A later fact.",
        source_url="https://example.com/future",
        source_date="2025-12-06",
    )
    backend.always("recorded_research", ResearchDraft(facts=[future]))
    with pytest.raises(ValueError, match="cannot be after"):
        await run_recorded_demo(
            backend,
            RecordedDemoConfig(
                clip=Path("synthetic.mp4"),
                research_fixture="Green v Orange",
                research_as_of="2025-12-05",
            ),
            frames=[Frame(at_s=0, jpeg=b"test")],
            audit_dir=tmp_path / "run",
        )
    assert backend.extra_params == {}
    archived = (tmp_path / "run" / "research.json").read_text()
    assert "raw_result" in archived and "future" in archived


@pytest.mark.asyncio
async def test_fresh_research_refuses_a_searchless_backend() -> None:
    with pytest.raises(LLMError, match="search-capable"):
        await run_recorded_demo(
            ScriptedBackend(),
            RecordedDemoConfig(
                clip=Path("synthetic.mp4"),
                research_fixture="Green v Orange",
                research_as_of="2025-12-05",
            ),
            frames=[Frame(at_s=0, jpeg=b"test")],
        )


def test_research_rejects_missing_or_future_source_dates() -> None:
    fact = {
        "id": "r1",
        "subject": "Green",
        "text": "A dated fact.",
        "source_url": "https://example.com/source",
    }
    with pytest.raises(ValueError, match="source_date"):
        ResearchFact.model_validate(fact)
    with pytest.raises(ValueError, match="cannot be after"):
        ResearchBrief.model_validate(
            {
                "fixture": "Green v Orange",
                "as_of": "2025-12-05",
                "facts": [{**fact, "source_date": "2025-12-06"}],
            }
        )


def test_pack_filter_keeps_only_identity_mapping() -> None:
    pack = safe_pack(
        {
            "teams": {
                "home": {
                    "name": "Green",
                    "kit_colours": ["green"],
                    "starters": [{"name": "Name", "number": 7, "position": "forward"}],
                    "manager": "Not passed on",
                }
            }
        }
    )
    assert pack == {
        "teams": {
            "home": {
                "name": "Green",
                "kit_colours": ["green"],
                "roster": [{"name": "Name", "number": 7}],
            }
        }
    }


def test_actor_resolution_is_side_specific_and_drops_untrusted_identity() -> None:
    pack = safe_pack(
        {
            "home": {
                "starters": [
                    {"name": "Antony", "number": 7},
                    {"name": "Alex One", "number": 11},
                    {"name": "Alex Two", "number": 12},
                ]
            },
            "away": {"starters": [{"name": "Ferran Torres", "number": 7}]},
        }
    )
    result = _resolve_observation_actors(
        [
            Observation(
                id="home-seven",
                at_s=1,
                kind="play",
                description="A player carries the ball.",
                actor=ObservedActor(
                    side="home",
                    shirt_number=7,
                    identity_source="shirt_number",
                    action_role="ball_carrier",
                    resolved_name="Model guess",
                ),
            ),
            Observation(
                id="away-seven",
                at_s=2,
                kind="play",
                description="A player makes a pass.",
                actor=ObservedActor(
                    side="away",
                    shirt_number=7,
                    identity_source="shirt_number",
                    action_role="passer",
                ),
            ),
            Observation(
                id="contradiction",
                at_s=3,
                kind="play",
                description="A player shoots.",
                actor=ObservedActor(
                    side="home",
                    shirt_number=7,
                    shirt_name="Wrong",
                    identity_source="shirt_number",
                    action_role="shooter",
                ),
            ),
            Observation(
                id="ambiguous",
                at_s=4,
                kind="play",
                description="A player is visible.",
                actor=ObservedActor(
                    side="home",
                    shirt_name="Alex",
                    identity_source="shirt_name",
                    action_role="visible_player",
                ),
            ),
        ],
        pack,
    )
    assert result[0].actor is not None and result[0].actor.resolved_name == "Antony"
    assert result[1].actor is not None and result[1].actor.resolved_name == "Ferran Torres"
    assert result[2].actor is None
    assert result[3].actor is None


@pytest.mark.asyncio
async def test_observer_receives_roster_and_archives_raw_before_identity_resolution() -> None:
    backend = ScriptedBackend()
    raw = Observation(
        id="o1",
        at_s=1,
        kind="goal",
        description="The ball reaches the net.",
        actor=ObservedActor(
            side="away",
            shirt_number=7,
            identity_source="shirt_number",
            action_role="scorer",
            resolved_name="Untrusted model value",
        ),
    )
    backend.always("recorded_observer", ObserverResult(observations=[raw]))
    pack = safe_pack({"away": {"starters": [{"name": "Away Seven", "number": 7}]}})
    state = await _observe(
        {
            "frames": [Frame(at_s=1, jpeg=b"frame")],
            "config": RecordedDemoConfig(clip=Path("synthetic.mp4"), duration_s=25, pack=pack),
            "audit": {},
        },
        backend=backend,
    )
    assert "Away Seven" in backend.calls_tagged("recorded_observer")[0].text
    assert state["observations"][0].actor is not None
    assert state["observations"][0].actor.resolved_name == "Away Seven"
    assert (
        state["audit"]["observe"]["windows"][0]["raw_result"]["observations"][0]["actor"][
            "resolved_name"
        ]
        == "Untrusted model value"
    )


def test_rejects_future_evidence() -> None:
    named_goal = Observation(
        id="named-goal",
        at_s=13,
        kind="goal",
        description="The ball is in the net.",
        actor=ObservedActor(
            side="away",
            shirt_number=7,
            identity_source="shirt_number",
            action_role="scorer",
            resolved_name="Away Seven",
        ),
    )
    lines = [
        PlanLine(
            id="caller-1",
            at_s=2,
            voice="caller",
            text="A shot comes in.",
            evidence_ids=["named-goal"],
        )
    ]
    with pytest.raises(ValueError, match="future evidence"):
        validate_plan_parts([*observations(), named_goal], lines, 25)


def test_joint_script_requires_direct_analyst_play_or_goal_insight() -> None:
    valid = joint_script(cite_research=True)
    _validate_joint_script(valid, observations(), 25, research())

    filler_only = valid.model_copy(deep=True)
    for index in (1, 3):
        filler_only.lines[index] = filler_only.lines[index].model_copy(
            update={"evidence_ids": [], "research_ids": ["r1"]}
        )
    with pytest.raises(ValueError, match="analyst insight on a play or goal"):
        _validate_joint_script(filler_only, observations(), 25, research())


def test_rejects_overlapping_lines() -> None:
    lines = [
        PlanLine(
            id="caller-1",
            at_s=1,
            voice="caller",
            text="One two three four five six.",
            evidence_ids=["o1"],
        ),
        PlanLine(id="caller-2", at_s=2, voice="caller", text="Another line.", evidence_ids=["o1"]),
    ]
    with pytest.raises(ValueError, match="overlaps"):
        validate_plan_parts(observations(), lines, 25)


def test_accepts_rounding_tolerance_but_rejects_estimated_speech_overlap() -> None:
    lines = [
        PlanLine(
            id="caller-1",
            at_s=1,
            voice="caller",
            text="One two three four five six seven eight nine ten eleven.",
            evidence_ids=["o1"],
        ),
        PlanLine(id="caller-2", at_s=5, voice="caller", text="Another line.", evidence_ids=["o1"]),
    ]
    # Eleven words end at 4.667s: 0.333s silence is a tolerated rounding gap.
    validate_plan_parts(observations(), lines, 25)
    overlapping = [*lines]
    overlapping[1] = overlapping[1].model_copy(update={"at_s": 4.65})
    with pytest.raises(ValueError, match="overlaps"):
        validate_plan_parts(observations(), overlapping, 25)


def test_rejects_final_line_that_runs_past_duration() -> None:
    lines = [
        PlanLine(
            id="caller-1",
            at_s=24,
            voice="caller",
            text="One two three four five six.",
            evidence_ids=["o1"],
        )
    ]
    with pytest.raises(ValueError, match="ends after"):
        validate_plan_parts(observations(), lines, 25)


@pytest.mark.asyncio
async def test_refuses_more_than_ninety_supplied_frames() -> None:
    backend = ScriptedBackend()
    frames = [Frame(at_s=0, jpeg=b"x") for _ in range(91)]
    with pytest.raises(ValueError, match="1 to 90"):
        await run_recorded_demo(
            backend, RecordedDemoConfig(clip=Path("synthetic.mp4")), frames=frames
        )


@pytest.mark.asyncio
async def test_timing_adjustment_is_audited_without_a_repair_call(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    invalid = joint_script()
    invalid.lines[0] = invalid.lines[0].model_copy(update={"at_s": 0.0})
    backend.always("recorded_writer", invalid)
    plan, _audit = await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            reused_observer=ObserverResult(observations=observations()),
        ),
        frames=[],
        audit_dir=tmp_path / "run",
    )
    assert (tmp_path / "run" / "observe.json").is_file()
    assert (tmp_path / "run" / "script-initial.json").is_file()
    assert (tmp_path / "run" / "script.json").is_file()
    assert [call.tag for call in backend.calls].count("recorded_writer") == 1
    archived = json.loads((tmp_path / "run" / "script.json").read_text())
    assert plan.lines[0].at_s == 1
    assert archived["initial"]["result"] == invalid.model_dump(mode="json")
    assert archived["timing_adjustments"][0]["reasons"] == ["latest_cited_evidence"]
    assert archived["usage"]["cost_usd"] == backend.total.cost_usd


@pytest.mark.asyncio
async def test_joint_writer_repairs_one_invalid_draft_and_keeps_both_raw_scripts(
    tmp_path: Path,
) -> None:
    backend = ScriptedBackend()
    invalid = joint_script()
    invalid.lines[1] = invalid.lines[1].model_copy(update={"text": "Too short."})
    backend.queue("recorded_writer", [invalid, joint_script()])

    plan, audit = await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            reused_observer=ObserverResult(observations=observations()),
        ),
        frames=[],
        audit_dir=tmp_path / "run",
    )

    assert [call.tag for call in backend.calls] == ["recorded_writer", "recorded_writer"]
    assert plan.lines[0].at_s == 1
    assert audit["script"]["initial"]["result"] == invalid.model_dump(mode="json")
    assert audit["script"]["retry"]["result"] == joint_script().model_dump(mode="json")
    assert (tmp_path / "run" / "script-initial.json").is_file()
    assert (tmp_path / "run" / "script-retry.json").is_file()
    repair_payload = json.loads(backend.calls[-1].text)
    assert repair_payload["total_words"] == 35
    assert repair_payload["max_total_words"] == 57
    assert repair_payload["timing_audit"][-1]["next_at_s"] == 25
    assert repair_payload["timing_audit"][-1]["max_words_before_next"] == 18


@pytest.mark.asyncio
async def test_reused_observer_skips_vision_cost_and_call() -> None:
    backend = ScriptedBackend()
    backend.always("recorded_writer", joint_script())
    plan, _audit = await run_recorded_demo(
        backend,
        RecordedDemoConfig(
            clip=Path("synthetic.mp4"),
            duration_s=25,
            reused_observer=ObserverResult(observations=observations()),
            reuse_source="prior/observe.json",
            prior_observation_cost=0.12,
        ),
        frames=[],
    )
    assert [call.tag for call in backend.calls] == ["recorded_writer"]
    assert plan.metadata["cost"] == 0
    assert plan.metadata["prior_observation_cost"] == 0.12
