"""CLI safety checks must happen before constructing a paid backend."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_agent_demo", ROOT / "scripts/prepare_agent_demo.py"
)
assert SPEC and SPEC.loader
script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(script)


@pytest.fixture
def args(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"offline fixture")
    monkeypatch.setattr(script, "validate_inputs", lambda *args: None)
    monkeypatch.setattr(
        script, "OpenAIBackend", lambda **kwargs: pytest.fail("must not construct paid backend")
    )
    return SimpleNamespace(
        clip=clip,
        start=42,
        duration=20,
        research=ROOT / "examples/recorded-agent-demo/research.json",
        teams=ROOT / "examples/recorded-agent-demo/team-context.json",
        reuse_vision=None,
        max_usd=0.02,
        max_model_calls=2,
        out_dir=tmp_path / "output",
        spend=False,
    )


async def test_dry_run_creates_no_output_and_calls_no_backend(args):
    result = await script.run(args)
    assert result["dry_run"] is True
    assert result["budget"]["attempts"] == 0
    assert not args.out_dir.exists()


async def test_mismatched_cached_source_rejected_before_paid_setup(args, tmp_path):
    args.reuse_vision = tmp_path / "cache.json"
    args.reuse_vision.write_text(
        json.dumps(
            {
                "source": {
                    "path": str(args.clip),
                    "sha256": "wrong",
                    "start_s": 42,
                    "duration_s": 20,
                }
            }
        )
    )
    args.spend = True
    with pytest.raises(ValueError, match="does not match"):
        await script.run(args)
    assert not args.out_dir.exists()
