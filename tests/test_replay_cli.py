from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from commentary.replay.contracts import ReplayResult
from commentary.replay.evaluation import EvaluationCase, evaluate_replay

ROOT = Path(__file__).resolve().parents[1]


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "replay_cli", ROOT / "scripts/replay_commentary.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_diagnostic_runs_real_agent_tool_loops_without_a_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = script()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "replay_commentary",
            "--diagnostic",
            "--rate",
            "20",
            "--out",
            str(tmp_path / "result.json"),
        ],
    )
    result = ReplayResult.model_validate(await module.run(module.arguments()))
    assert {record.line.voice for record in result.speech} == {"caller", "analyst"}
    assert any(item.event == "tool_result" for item in result.trace)
    assert result.metadata["provenance"] == "synthetic diagnostic"
    assert result.metadata["speech_simulated"]
    assert not any(item.event in {"turn_rejected", "turn_timeout"} for item in result.trace)
    assert any("goal" in record.line.evidence_ids for record in result.speech)
    reference = EvaluationCase.model_validate_json(
        (ROOT / "examples/replay-evaluation/diagnostic.json").read_text()
    )
    score = evaluate_replay(result, reference)
    assert score["source"]["kind"] == "synthetic"


async def test_dry_run_never_builds_a_paid_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = script()
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_commentary", "--clip", "local.mp4", "--out", str(tmp_path / "result.json")],
    )
    monkeypatch.setattr(module, "validate_inputs", lambda *_args: 28)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("dry run must not construct models or sample frames")

    monkeypatch.setattr(module, "openai_factory", forbidden)
    monkeypatch.setattr(module, "sample_clip", forbidden)
    result = await module.run(module.arguments())
    assert result["mode"] == "dry_run"
    assert not result["paid_calls_enabled"]


async def test_diagnostic_rejects_paid_or_real_context_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = script()
    monkeypatch.setattr(
        sys,
        "argv",
        ["replay_commentary", "--diagnostic", "--spend", "--out", str(tmp_path / "result.json")],
    )
    with pytest.raises(ValueError, match="fictional"):
        await module.run(module.arguments())


def test_failed_provider_run_saves_trace_and_exits_unsuccessfully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = script()
    output = tmp_path / "failed.json"
    monkeypatch.setattr(sys, "argv", ["replay_commentary", "--diagnostic", "--out", str(output)])

    async def failed(_args: object) -> dict[str, object]:
        return {
            "mode": "causal_replay",
            "speech": [],
            "trace": [{"event": "observer_error"}],
            "metadata": {"usage": {"cost_usd": 0}},
        }

    monkeypatch.setattr(module, "run", failed)
    with pytest.raises(SystemExit, match="No speech was produced"):
        module.main()
    assert output.exists()


def test_replay_defaults_are_cheap_and_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    module = script()
    monkeypatch.setattr(sys, "argv", ["replay_commentary", "--clip", "local.mp4"])
    args = module.arguments()
    assert args.observer_model == args.caller_model == args.analyst_model == "gpt-6-luna"
    assert args.max_usd == 0.10
    assert args.max_model_calls == 40
