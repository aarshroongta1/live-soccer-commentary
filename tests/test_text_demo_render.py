"""Silent rendering must preserve words/cues and never synthesize audio."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_text_demo as renderer  # noqa: E402

PLAN = ROOT / "examples/portfolio-demo/penalty-plan.json"


def test_captions_preserve_words_and_stop_before_the_next_cue():
    plan = renderer.read_plan(PLAN)
    ass, rows = renderer.caption_rows(plan, "Automatic dialogue")
    assert [row["text"] for row in rows] == [line.text for line in plan.lines]
    assert [row["at_s"] for row in rows] == [line.at_s for line in plan.lines]
    assert all(row["at_s"] < row["end_s"] <= plan.duration_s for row in rows)
    assert all(left["end_s"] < right["at_s"] for left, right in zip(rows, rows[1:], strict=False))
    assert "Caller:" in ass and "Analyst:" in ass


def test_render_command_removes_source_audio_and_refuses_overwrite(tmp_path, monkeypatch):
    clip = tmp_path / "england-france-penalties.mp4"
    clip.write_bytes(b"offline input")
    monkeypatch.setattr(renderer, "validate_inputs", lambda *_: None)
    monkeypatch.setattr(renderer, "resolve_ffmpeg", lambda *_: "scripted-ffmpeg")
    monkeypatch.setattr(renderer, "require_subtitles_filter", lambda *_: None)
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"scripted silent output")

    monkeypatch.setattr(renderer.subprocess, "run", fake_run)
    output = tmp_path / "render.mp4"
    report = renderer.render(PLAN, clip, output, "Automatic dialogue")
    assert "-an" in commands[0]
    assert report["audio"] is False and report["tts_calls"] == 0
    assert report["words_edited"] is False
    with pytest.raises(ValueError, match="overwrite"):
        renderer.render(PLAN, clip, output, "Automatic dialogue")
    assert len(commands) == 1


def test_portfolio_transcript_matches_automatic_plan_and_trace():
    plan = json.loads(PLAN.read_text())
    data = json.loads((ROOT / "web/data/portfolio-demo.json").read_text())
    trace = json.loads((ROOT / "examples/portfolio-demo/penalty-trace.json").read_text())
    assert data["lines"] == [
        {k: line[k] for k in ("at_s", "voice", "text")} for line in plan["lines"]
    ]
    assert plan["lines"] == trace["dialogue"]["lines"]
    assert data["kind"] == "automatic" and trace["words_edited"] is False
    assert data["generation"]["cost_usd"] == trace["budget"]["reported_usd"]
