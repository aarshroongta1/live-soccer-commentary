"""Comparison controls use scripted backends; never provider calls or TTS."""

import json

import pytest
from pydantic import ValidationError

from commentary.llm.base import LLMError
from commentary.llm.fake import ScriptedBackend
from commentary.recorded_demo import Observation, ScriptLine
from commentary.replay.contracts import AgentAction, SpeechLine
from commentary.replay.text_comparison import (
    AuditedTextBackend,
    ShortJointScript,
    TextComparisonInput,
    export_blind,
    place_lines,
    run_arm,
)


def bundle():
    return TextComparisonInput(
        source_sha256="a" * 64,
        source_start_s=10,
        duration_s=8,
        observations=[
            Observation(
                id="event-1",
                at_s=1,
                kind="play",
                description="The ball moves forward.",
                action={
                    "type": "pass",
                    "certainty": "confirmed",
                    "cues": ["ball_motion"],
                    "support_at_s": [1, 3],
                },
            )
        ],
    )


def adapter(fake, **kwargs):
    return AuditedTextBackend(fake, model="scripted", **kwargs)


@pytest.mark.parametrize("field", ["reference", "commentary", "expected_player", "metadata"])
def test_reference_fields_are_rejected(field):
    with pytest.raises(ValidationError):
        TextComparisonInput.model_validate({**bundle().model_dump(), field: "GOLD SENTINEL"})


@pytest.mark.parametrize("at", [-0.5, 9, float("nan")])
def test_bad_evidence_time_rejected(at):
    data = bundle().model_dump()
    data["observations"][0]["at_s"] = at
    with pytest.raises(ValidationError):
        TextComparisonInput.model_validate(data)


def test_both_arms_share_support_aware_placement():
    draft = ScriptLine(at_s=1, voice="caller", text="Forward they go.", evidence_ids=["event-1"])
    lines, changes = place_lines([draft], bundle())
    assert lines[0].at_s == 3
    assert lines[0].text == draft.text
    assert changes[0]["requested_s"] == 1
    with pytest.raises(ValueError, match="unknown evidence"):
        place_lines([draft.model_copy(update={"evidence_ids": ["gold-id"]})], bundle())


async def test_short_joint_can_output_one_line_without_long_clip_quota():
    fake = ScriptedBackend()
    fake.always(
        "comparison_joint",
        ShortJointScript(
            lines=[
                ScriptLine(
                    at_s=3,
                    voice="caller",
                    text="Forward they go.",
                    evidence_ids=["event-1"],
                )
            ]
        ),
    )
    result = await run_arm(bundle(), adapter(fake), "joint")
    assert result["status"] == "complete"
    assert len(result["lines"]) == 1
    assert "7–12 chronological" not in fake.calls[0].system
    assert "OFFLINE" in fake.calls[0].system


async def test_joint_repairs_once_and_preserves_both_rejected_drafts():
    fake = ScriptedBackend()
    fake.always(
        "comparison_joint",
        ShortJointScript(
            lines=[
                ScriptLine(
                    at_s=0,
                    voice="caller",
                    text="Wrong ID.",
                    evidence_ids=["missing"],
                )
            ]
        ),
    )
    audited = adapter(fake)
    result = await run_arm(bundle(), audited, "joint")
    assert result["status"] == "failed"
    assert result["repairs"] == 1
    assert len(result["validation_failures"]) == len(fake.calls) == 2
    assert audited.calls[0]["result"]["lines"][0]["text"] == "Wrong ID."
    assert "invalid_draft" in fake.calls[1].text


async def test_provider_error_is_not_retried_and_is_audited():
    fake = ScriptedBackend()

    def fail(_blocks, _schema):
        raise LLMError("parse failed")

    fake.register("comparison_joint", fail)
    result = await run_arm(bundle(), adapter(fake), "joint")
    assert result["status"] == "failed"
    assert result["model_calls"] == 1
    assert result["calls"][0]["status"] == "error"
    assert result["usage"]["cost_usd"] == 0


async def test_call_limit_blocks_validation_repair_without_hiding_first_draft():
    fake = ScriptedBackend()
    fake.always(
        "comparison_joint",
        ShortJointScript(
            lines=[
                ScriptLine(
                    at_s=0,
                    voice="caller",
                    text="Wrong ID.",
                    evidence_ids=["missing"],
                )
            ]
        ),
    )
    result = await run_arm(bundle(), adapter(fake, max_calls=1), "joint")
    assert result["error"]["type"] == "BudgetExceeded"
    assert len(fake.calls) == len(result["calls"]) == 1
    assert result["calls"][0]["result"]["lines"]


async def test_agents_receive_full_evidence_pool_without_reference_or_source_names():
    data = bundle()
    data.observations.append(Observation(id="late", at_s=6, kind="play", description="A pass."))
    fake = ScriptedBackend()
    fake.always(
        "replay_caller",
        AgentAction(
            kind="speak",
            line=SpeechLine(
                voice="caller",
                text="Forward they go.",
                evidence_ids=["event-1"],
            ),
        ),
    )
    fake.always("replay_analyst", AgentAction(kind="abstain", reason="No useful insight."))
    result = await run_arm(data, adapter(fake), "agents")
    first = json.loads(fake.calls[0].text)
    assert [item["id"] for item in first["observations"]] == ["event-1", "late"]
    assert first["cutoff_s"] == data.duration_s
    assert first["planned_at_s"] == 0
    assert "reference" not in first
    assert "source_sha256" not in first
    assert "short-clips-v4" in fake.calls[0].system
    assert result["lines"][0]["at_s"] == 3
    assert not any("review" in call.tag for call in fake.calls)


async def test_two_abstentions_end_run_and_remain_visible_as_empty_output():
    fake = ScriptedBackend()
    for role in ("caller", "analyst"):
        fake.always(f"replay_{role}", AgentAction(kind="abstain", reason="Unclear."))
    result = await run_arm(bundle(), adapter(fake), "agents")
    assert result["status"] == "complete"
    assert result["lines"] == []
    assert len(fake.calls) == 2
    assert result["model_calls"] == 2


async def test_same_bundle_hash_and_same_model_for_both_arms():
    fake = ScriptedBackend()
    fake.always("comparison_joint", ShortJointScript(lines=[]))
    for role in ("caller", "analyst"):
        fake.always(f"replay_{role}", AgentAction(kind="abstain"))
    joint = await run_arm(bundle(), adapter(fake), "joint")
    agents = await run_arm(bundle(), adapter(fake), "agents")
    assert joint["input_sha256"] == agents["input_sha256"]
    assert all(call.model == "scripted" for call in fake.calls)


def test_blind_export_hides_architecture_cost_evidence_and_keeps_failed_arms(tmp_path):
    reports = [
        {
            "case_id": "case",
            "arms": [
                {
                    "arm": "joint",
                    "status": "complete",
                    "model": "secret-model",
                    "cost": 99,
                    "lines": [
                        {"at_s": 1, "voice": "caller", "text": "A pass.", "evidence_ids": ["x"]}
                    ],
                },
                {"arm": "agents", "status": "failed", "lines": []},
            ],
        }
    ]
    export_blind(reports, tmp_path / "blind", seed=17)
    text = (tmp_path / "blind/review.json").read_text()
    assert all(
        word not in text for word in ("joint", "agents", "secret-model", "cost", "evidence_ids")
    )
    assert "failed" in text
    assert len(json.loads(text)[0]["candidates"]) == 2
    mapping = json.loads((tmp_path / "blind/mapping.private.json").read_text())
    assert set(mapping["pairs"][0]["mapping"].values()) == {"joint", "agents"}
    with pytest.raises(FileExistsError):
        export_blind(reports, tmp_path / "blind", seed=17)


async def test_output_cap_and_effort_are_identical_despite_callsite_defaults():
    seen = []

    class RecordingBackend(ScriptedBackend):
        async def parse(self, **kwargs):
            seen.append((kwargs["max_tokens"], kwargs["effort"]))
            return await super().parse(**kwargs)

    fake = RecordingBackend()
    fake.always("comparison_joint", ShortJointScript(lines=[]))
    for role in ("caller", "analyst"):
        fake.always(f"replay_{role}", AgentAction(kind="abstain"))
    for arm in ("joint", "agents"):
        await run_arm(bundle(), adapter(fake, max_tokens=512), arm)
    assert seen == [(512, "none")] * 3


async def test_budget_stop_during_agents_preserves_partial_text():
    fake = ScriptedBackend()
    fake.always(
        "replay_caller",
        AgentAction(
            kind="speak",
            line=SpeechLine(
                voice="caller",
                text="Forward they go.",
                evidence_ids=["event-1"],
            ),
        ),
    )
    result = await run_arm(bundle(), adapter(fake, max_calls=1), "agents")
    assert result["status"] == "failed"
    assert result["error"]["type"] == "BudgetExceeded"
    assert len(result["lines"]) == 1
    assert result["calls"][0]["result"]["line"]["text"] == "Forward they go."


def test_cli_dry_run_records_all_missing_cases_without_credentials(tmp_path):
    import os
    import subprocess
    import sys

    inputs = tmp_path / "inputs.json"
    inputs.write_text("[]\n")
    env = {k: v for k, v in os.environ.items() if not k.endswith("API_KEY")}
    result = subprocess.run(
        [
            sys.executable,
            "scripts/compare_text.py",
            "--inputs",
            str(inputs),
            "--out-dir",
            str(tmp_path / "result"),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "0/15 inputs ready" in result.stdout
    report = json.loads((tmp_path / "result/report.json").read_text())
    assert len(report["inventory"]) == 15
    assert all(row["status"] == "missing_input" for row in report["inventory"])
    assert not report["paid"]
    assert report["quality_review"] == "pending"


async def test_cli_executes_both_arms_and_exports_all_cases_with_scripted_provider(
    tmp_path, monkeypatch
):
    import argparse
    import importlib.util
    from pathlib import Path

    import openai

    import commentary.llm.openai_backend as provider_module

    spec = importlib.util.spec_from_file_location("compare_text_cli", "scripts/compare_text.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clients = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0
            self.closed = False
            clients.append(self)

        async def close(self):
            self.closed = True

    def provider(**kwargs):
        assert kwargs["budget"].max_usd == 0.03
        assert kwargs["budget"].max_calls == 8
        fake = ScriptedBackend()
        fake.always("comparison_joint", ShortJointScript(lines=[]))
        for role in ("caller", "analyst"):
            fake.always(f"replay_{role}", AgentAction(kind="abstain"))
        return fake

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(provider_module, "OpenAIBackend", provider)
    inputs = tmp_path / "inputs.json"
    inputs.write_text("[]\n")
    rows = [
        {"case_id": c["id"], "status": "ready" if c["id"] == "penalty" else "missing_input"}
        for c in json.loads(Path("examples/action-suite.json").read_text())["clips"]
    ]
    monkeypatch.setattr(module, "inventory", lambda *_: (rows, {"penalty": bundle()}))
    report = await module.run(
        argparse.Namespace(
            inputs=inputs,
            suite=Path("examples/action-suite.json"),
            spend=True,
            model="scripted",
            max_tokens=512,
            seed=17,
            out_dir=tmp_path / "output",
        )
    )
    assert report["status"] == "completed_with_failures"
    assert len(report["pairs"]) == 15
    penalty = next(pair for pair in report["pairs"] if pair["case_id"] == "penalty")
    assert [arm["status"] for arm in penalty["arms"]] == ["complete", "complete"]
    assert all(client.closed for client in clients)
    assert len(clients) == 2
    assert len(json.loads((tmp_path / "output/blind/review.json").read_text())) == 15
    assert report["reported_usd"] == 0


def test_joint_adapter_removes_conflicting_long_clip_requirements():
    from commentary.replay.text_comparison import SHORT_JOINT_SYSTEM

    assert "three distinct" not in SHORT_JOINT_SYSTEM
    assert "60/40" not in SHORT_JOINT_SYSTEM
    assert "At least one analyst line must cite" not in SHORT_JOINT_SYSTEM
    assert "single short correct caller line" in SHORT_JOINT_SYSTEM


async def test_agent_word_budget_uses_remaining_clip_time():
    fake = ScriptedBackend()
    fake.always(
        "replay_caller",
        AgentAction(
            kind="speak",
            line=SpeechLine(
                voice="caller",
                text="Forward they go.",
                evidence_ids=["event-1"],
            ),
        ),
    )
    fake.always("replay_analyst", AgentAction(kind="abstain"))
    await run_arm(bundle(), adapter(fake), "agents")
    payload = json.loads(fake.calls_tagged("replay_analyst")[0].text)
    assert payload["evidence_ready_at_s"]["event-1"] == 3
    assert payload["speech_budget_words"] == 11  # 8s - (3s + 1s of words + .3s gap)
    assert payload["planned_at_s"] == 4.3


def test_late_evidence_rejection_reports_exact_remaining_words():
    data = bundle()
    data.duration_s = 3.5
    draft = ScriptLine(at_s=0, voice="caller", text="It’s in!", evidence_ids=["event-1"])
    with pytest.raises(ValueError, match="Only 1 words fit at 3.000s"):
        place_lines([draft], data)


async def test_caller_is_not_requested_again_after_analyst_abstains_without_new_evidence():
    fake = ScriptedBackend()
    fake.always(
        "replay_caller",
        AgentAction(
            kind="speak",
            line=SpeechLine(
                voice="caller",
                text="Forward they go.",
                evidence_ids=["event-1"],
            ),
        ),
    )
    fake.always("replay_analyst", AgentAction(kind="abstain", reason="No new point."))
    result = await run_arm(bundle(), adapter(fake), "agents")
    assert result["status"] == "complete"
    assert [c.tag for c in fake.calls] == ["replay_caller", "replay_analyst"]
    assert len(result["lines"]) == 1
