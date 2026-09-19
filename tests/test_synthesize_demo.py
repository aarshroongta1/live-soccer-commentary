"""Offline contracts for the explicit-spend ElevenLabs helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
_SPEC = importlib.util.spec_from_file_location(
    "synthesize_demo", Path(__file__).parents[1] / "scripts/synthesize_demo.py"
)
assert _SPEC and _SPEC.loader
synthesize_demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = synthesize_demo
_SPEC.loader.exec_module(synthesize_demo)


def plan(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "mode": "recorded_clip",
                "source": {"basename": "clip.mp4", "start_s": 0, "duration_s": 5},
                "observations": [{"id": "o", "at_s": 0, "kind": "play", "description": "x"}],
                "lines": [
                    {
                        "id": "caller-1",
                        "at_s": 0,
                        "voice": "caller",
                        "text": "Exact caller words.",
                        "evidence_ids": ["o"],
                    },
                    {
                        "id": "analyst-1",
                        "at_s": 2,
                        "voice": "analyst",
                        "text": "Exact analyst words.",
                        "evidence_ids": ["o"],
                    },
                ],
                "metadata": {
                    "offline": True,
                    "models": {},
                    "cost": 0,
                    "trace": "offline: saved plan",
                },
            }
        )
    )


def test_dry_run_needs_no_credentials_or_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "plan.json"
    plan(source)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(
        synthesize_demo.dotenv,
        "load_dotenv",
        lambda: (_ for _ in ()).throw(AssertionError("dry run loaded dotenv")),
    )

    output = tmp_path / "audio"
    assert synthesize_demo.main(["--plan", str(source), "--out-dir", str(output)]) == 0
    assert "Dry run: 2 exact plan lines" in capsys.readouterr().out
    assert not output.exists()


def test_spend_requires_credentials_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "plan.json"
    plan(source)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    output = tmp_path / "audio"
    with pytest.raises(synthesize_demo.RenderError, match="API_KEY"):
        synthesize_demo.main(["--plan", str(source), "--out-dir", str(output), "--spend"])
    assert not output.exists()

    output.mkdir()
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    with pytest.raises(synthesize_demo.RenderError, match="overwrite"):
        synthesize_demo.main(["--plan", str(source), "--out-dir", str(output), "--spend"])


@pytest.mark.parametrize("from_dotenv", [False, True])
def test_spend_sends_exact_text_to_distinct_voices_and_writes_incremental_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, from_dotenv: bool
) -> None:
    source = tmp_path / "plan.json"
    output = tmp_path / "audio"
    plan(source)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    for role in ("CALLER", "ANALYST", "LEAD", "SUPPORTING"):
        monkeypatch.delenv(f"ELEVENLABS_{role}_VOICE", raising=False)
    if from_dotenv:

        def fake_dotenv():
            monkeypatch.setenv("ELEVENLABS_CALLER_VOICE", "custom-caller")
            monkeypatch.setenv("ELEVENLABS_ANALYST_VOICE", "custom-analyst")

        monkeypatch.setattr(synthesize_demo.dotenv, "load_dotenv", fake_dotenv)
    requests: list[dict[str, Any]] = []

    class Response:
        content = b"mp3"
        headers = {"character-cost": "12"}

        @staticmethod
        def raise_for_status() -> None:
            return None

    class Client:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def __enter__(self) -> Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, url: str, **kwargs: Any) -> Response:
            if requests:
                assert json.loads((output / "manifest.json").read_text()) == {
                    "caller-1": str((output / "caller-1.mp3").resolve())
                }
                assert (
                    json.loads((output / "synthesis-audit.json").read_text())["lines"][0]["text"]
                    == "Exact caller words."
                )
            requests.append({"url": url, **kwargs})
            return Response()

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(Client=Client))
    assert synthesize_demo.main(["--plan", str(source), "--out-dir", str(output), "--spend"]) == 0

    assert [request["json"]["text"] for request in requests] == [
        "Exact caller words.",
        "Exact analyst words.",
    ]
    caller = "custom-caller" if from_dotenv else synthesize_demo.CALLER_VOICE
    analyst = "custom-analyst" if from_dotenv else synthesize_demo.ANALYST_VOICE
    assert caller in requests[0]["url"]
    assert analyst in requests[1]["url"]
    assert json.loads((output / "manifest.json").read_text()) == {
        "caller-1": str((output / "caller-1.mp3").resolve()),
        "analyst-1": str((output / "analyst-1.mp3").resolve()),
    }
    audit = json.loads((output / "synthesis-audit.json").read_text())
    assert audit["model"] == synthesize_demo.MODEL
    assert audit["lines"][1]["credit_headers"] == {"character-cost": "12"}


def test_synthesis_rejects_identical_voice_ids(tmp_path: Path) -> None:
    source = tmp_path / "plan.json"
    plan(source)
    with pytest.raises(synthesize_demo.RenderError, match="distinct"):
        synthesize_demo.main(
            [
                "--plan",
                str(source),
                "--out-dir",
                str(tmp_path / "audio"),
                "--caller-voice",
                "same",
                "--analyst-voice",
                "same",
            ]
        )
