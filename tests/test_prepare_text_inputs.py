"""Perception preparation keeps gold labels out and never retries failed cases."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from commentary.llm.base import LLMError
from commentary.recorded_demo import Frame


async def test_preparation_uses_one_shared_budget_and_keeps_failed_cases(tmp_path, monkeypatch):
    import openai

    import commentary.llm.openai_backend as provider_module

    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "prepare_text_inputs_cli", "scripts/prepare_text_inputs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"not decoded; frame sampling is scripted")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    clips = []
    for i in range(15):
        ref = tmp_path / f"ref-{i}.json"
        ref.write_text(
            json.dumps(
                {
                    "duration_s": 8,
                    "source": {"locator": str(source), "sha256": source_hash, "start_s": 0},
                    "events": [{"description": "GOLD SENTINEL"}],
                    "commentary": ["TARGET SENTINEL"],
                }
            )
        )
        clips.append({"id": f"case-{i}", "reference": str(ref), "title": "TARGET EVENT"})
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"clips": clips}))
    inputs = tmp_path / "inputs.json"
    inputs.write_text("[]")
    monkeypatch.setattr(module, "inventory", lambda *_: ([{"case_id": c["id"]} for c in clips], {}))
    monkeypatch.setattr(module, "sample_clip", lambda *_a, **_k: [Frame(at_s=0, jpeg=b"frame")])
    budgets = []
    closed = []

    class Client:
        def __init__(self, **kwargs):
            assert kwargs["max_retries"] == 0

        async def close(self):
            closed.append(True)

    class Provider:
        def __init__(self, **kwargs):
            self.budget = kwargs["budget"]
            budgets.append(self.budget)

    attempts = []

    async def observe(backend, frames, **kwargs):
        assert kwargs["research"] is None and kwargs["pack"] is None
        assert frames[0].jpeg == b"frame"
        identifier = await backend.budget.begin()
        await backend.budget.reserve(identifier, 0.001)
        await backend.budget.settle(identifier, 0.001)
        attempts.append(identifier)
        if identifier == 2:
            raise LLMError("scripted malformed output")
        return {"observations": []}

    monkeypatch.setattr(openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(provider_module, "OpenAIBackend", Provider)
    monkeypatch.setattr(module, "observe_recorded", observe)
    report = await module.run(
        argparse.Namespace(
            inputs=inputs,
            suite=suite,
            out_dir=tmp_path / "out",
            spend=True,
            max_usd=0.24,
            model="scripted",
        )
    )
    assert len({id(b) for b in budgets}) == 1
    assert attempts == list(range(1, 16))  # no extra attempt after malformed case 2
    assert len(closed) == 15
    assert report["status"] == "incomplete"
    assert report["cases"][1]["status"] == "failed"
    assert report["budget"]["reported_usd"] == pytest.approx(0.015)
    manifest = json.loads((tmp_path / "out/inputs.json").read_text())
    assert len(manifest) == 14
    for path in (tmp_path / "out/inputs").glob("*.json"):
        text = path.read_text()
        assert "GOLD SENTINEL" not in text and "TARGET SENTINEL" not in text
