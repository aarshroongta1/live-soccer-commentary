"""Plan validation, measured timing and captions for the offline renderer."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from commentary.recorded_demo import DemoPlan as RecordedDemoPlan
from commentary.recorded_demo import Observation as RecordedObservation
from commentary.recorded_demo import PlanLine as RecordedPlanLine
from commentary.recorded_demo import Source as RecordedSource

_SCRIPT = Path(__file__).parents[1] / "scripts" / "render_demo.py"
_SPEC = importlib.util.spec_from_file_location("render_demo", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
render_demo = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = render_demo
_SPEC.loader.exec_module(render_demo)


def test_cli_accepts_only_recorded_plans() -> None:
    parser = render_demo.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--trace", "old.jsonl", "--clip", "clip.mp4"])
    assert parser.parse_args(["--plan", "plan.json", "--clip", "clip.mp4"]).captions is True


def test_complete_manifest_needs_no_say_and_measures_every_line(tmp_path, monkeypatch) -> None:
    source = tmp_path / "plan.json"
    data = _valid_plan()
    source.write_text(json.dumps(data))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    manifest = tmp_path / "manifest.json"
    audio = {line["id"]: tmp_path / f"{line['id']}.mp3" for line in data["lines"]}
    for path in audio.values():
        path.write_bytes(b"saved audio")
    manifest.write_text(json.dumps({key: str(path) for key, path in audio.items()}))
    measured = []

    def probe(path, **kwargs):
        measured.append(path)
        return 60.0 if path == clip else 1.0

    monkeypatch.setattr(render_demo, "probe_duration", probe)
    monkeypatch.setattr(render_demo.shutil, "which", lambda name: None if name == "say" else name)
    monkeypatch.setattr(
        render_demo, "synthesize", lambda *a, **kw: pytest.fail("must reuse saved speech")
    )
    monkeypatch.setattr(render_demo, "render_video", lambda **kw: None)
    assert (
        render_demo.main(
            [
                "--plan",
                str(source),
                "--clip",
                str(clip),
                "--manifest",
                str(manifest),
                "--no-captions",
                "--out",
                str(tmp_path / "preview.mp4"),
            ]
        )
        == 0
    )
    assert measured == [clip, *audio.values()]
    assert (tmp_path / "preview.mp4.report.md").is_file()


@pytest.mark.parametrize("suffix", [".report.md", ".captions.ass"])
def test_existing_sidecars_are_not_overwritten(tmp_path, monkeypatch, suffix) -> None:
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(_valid_plan()))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    sidecar = tmp_path / ("preview.mp4" + suffix)
    sidecar.write_text("keep this")
    monkeypatch.setattr(render_demo, "probe_duration", lambda *a, **kw: 60.0)
    with pytest.raises(render_demo.RenderError, match="overwrite"):
        render_demo.main(
            [
                "--plan",
                str(source),
                "--clip",
                str(clip),
                "--out",
                str(tmp_path / "preview.mp4"),
            ]
        )
    assert sidecar.read_text() == "keep this"


def test_goal_cited_plan_trims_measured_preceding_audio_and_keeps_goal_cue(tmp_path: Path) -> None:
    plan_data = _valid_plan()
    plan_data["observations"] = [
        {"id": "build", "at_s": 0.5, "kind": "play", "description": "The move builds."},
        {"id": "finish", "at_s": 2.0, "kind": "goal", "description": "The ball is in."},
    ]
    plan_data["lines"] = [
        {
            "id": "background",
            "at_s": 1.0,
            "voice": "analyst",
            "text": "The move is developing through the middle.",
            "evidence_ids": ["build"],
        },
        {
            "id": "finish-call",
            "at_s": 2.0,
            "voice": "caller",
            "text": "It is in!",
            "evidence_ids": ["finish"],
        },
    ]
    path = tmp_path / "goal-priority.json"
    path.write_text(json.dumps(plan_data), encoding="utf-8")

    plan = render_demo.read_plan(path)
    timeline = render_demo.plan_timeline(
        plan.lines,
        duration_s=40.0,
        audio_seconds={"background": 3.0, "finish-call": 1.0},
    )

    background, goal = timeline.planned
    assert background.duration_s == 1.0
    assert background.trimmed_for_goal_s == 2.0
    assert goal.start_s == goal.natural_start_s == 2.0
    assert render_demo.audible_role_seconds(timeline, duration_s=40.0) == {
        "caller": 1.0,
        "analyst": 1.0,
    }
    report = render_demo.report_markdown(
        timeline,
        plan=path,
        clip=Path("clip.mp4"),
        source_start_s=0.0,
        duration_s=40.0,
        estimated=False,
    )
    assert "Goal priority trimmed preceding speech: background by 2.00s." in report
    assert "Voice assignment: caller `Daniel`; analyst `Samantha` (distinct)." in report
    assert "Caller (Daniel): 1.00 s (50.0%)" in report
    assert "Analyst (Samantha): 1.00 s (50.0%)" in report


def test_render_trims_audio_input_to_goal_priority_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        render_demo.RenderLine(0, "background", "analyst", "Background words.", 1.0),
        render_demo.RenderLine(1, "goal", "caller", "Goal!", 2.0, goal_evidence=True),
    ]
    timeline = render_demo.plan_timeline(
        lines, duration_s=8.0, audio_seconds={"background": 3.0, "goal": 1.0}
    )
    captured: list[list[str]] = []
    monkeypatch.setattr(
        render_demo.subprocess, "run", lambda command, check: captured.append(command)
    )

    render_demo.render_video(
        ffmpeg="ffmpeg",
        clip=Path("clip.mp4"),
        output=Path("out.mp4"),
        source_start_s=0.0,
        duration_s=8.0,
        planned=timeline.planned,
        audio_by_id={"background": Path("background.aiff"), "goal": Path("goal.aiff")},
    )

    filters = captured[0][captured[0].index("-filter_complex") + 1]
    assert "atrim=duration=1.000" in filters


def test_analyst_goal_citation_does_not_preempt_a_goal_caller(tmp_path: Path) -> None:
    plan_data = _valid_plan()
    lines = plan_data["lines"]
    assert isinstance(lines, list)
    lines[0]["voice"] = "caller"
    lines[0]["at_s"] = 1.0
    lines[0]["text"] = "It is in!"
    lines[1]["voice"] = "analyst"
    lines[1]["at_s"] = 1.1
    lines[1]["text"] = "A post-goal note follows."
    path = tmp_path / "analyst-goal-reference.json"
    path.write_text(json.dumps(plan_data), encoding="utf-8")

    plan = render_demo.read_plan(path)
    timeline = render_demo.plan_timeline(
        plan.lines,
        duration_s=40.0,
        audio_seconds={"l1": 2.0, "l2": 1.0},
    )

    assert [item.start_s for item in timeline.planned] == [1.0, 3.0]
    assert timeline.planned[0].trimmed_for_goal_s == 0.0


def _valid_plan() -> dict[str, object]:
    return {
        "version": 1,
        "mode": "recorded_clip",
        "source": {"basename": "clip.mp4", "start_s": 12.0, "duration_s": 40.0},
        "observations": [{"id": "o1", "at_s": 1.0, "kind": "goal", "description": "A finish"}],
        "lines": [
            {
                "id": "l1",
                "at_s": 1.5,
                "voice": "caller",
                "text": "The finish is found.",
                "evidence_ids": ["o1"],
            },
            {
                "id": "l2",
                "at_s": 4.0,
                "voice": "analyst",
                "text": "That was the opening.",
                "evidence_ids": ["o1"],
            },
        ],
        "metadata": {
            "offline": True,
            "models": {"caller": "claude", "analyst": "claude"},
            "cost": 0.0,
            "trace": "saved.jsonl",
        },
    }


def test_read_plan_validates_schema_and_maps_relative_times(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_valid_plan()), encoding="utf-8")

    plan = render_demo.read_plan(path)

    assert plan.source_basename == "clip.mp4"
    assert plan.source_start_s == 12.0
    assert plan.duration_s == 40.0
    assert [(line.id, line.at_s, line.voice, line.evidence_ids) for line in plan.lines] == [
        ("l1", 1.5, "caller", ("o1",)),
        ("l2", 4.0, "analyst", ("o1",)),
    ]


def test_generated_demo_plan_round_trips_through_renderer_validation(tmp_path: Path) -> None:
    generated = RecordedDemoPlan(
        source=RecordedSource(basename="clip.mp4", start_s=12.0, duration_s=40.0),
        observations=[RecordedObservation(id="o1", at_s=1.0, kind="goal", description="A finish")],
        lines=[
            RecordedPlanLine(
                id="l1",
                at_s=1.5,
                voice="caller",
                text="The finish is found.",
                evidence_ids=["o1"],
            )
        ],
        metadata={
            "offline": True,
            "models": {"caller": "claude"},
            "cost": 0.0,
            "trace": "saved.jsonl",
        },
    )
    path = tmp_path / "generated.json"
    path.write_text(generated.model_dump_json(), encoding="utf-8")

    loaded = render_demo.read_plan(path)

    assert loaded.lines[0].text == "The finish is found."
    assert loaded.metadata["trace"] == "saved.jsonl"


def test_research_is_analyst_only_and_reported_with_citation(tmp_path: Path) -> None:
    plan_data = _valid_plan()
    metadata = plan_data["metadata"]
    assert isinstance(metadata, dict)
    metadata["research"] = {
        "fixture": "Betis v Barcelona",
        "as_of": "2026-09-17",
        "facts": [
            {
                "id": "f1",
                "subject": "Barcelona",
                "text": "Barcelona have won three of their last five away matches.",
                "source_url": "https://example.com/form",
                "source_date": "2026-09-16",
            }
        ],
    }
    lines = plan_data["lines"]
    assert isinstance(lines, list)
    lines[1]["research_ids"] = ["f1"]
    path = tmp_path / "research-plan.json"
    path.write_text(json.dumps(plan_data), encoding="utf-8")

    plan = render_demo.read_plan(path)
    timeline = render_demo.plan_timeline(
        plan.lines,
        duration_s=40.0,
        audio_seconds={"l1": 1.0, "l2": 1.0},
    )
    report = render_demo.report_markdown(
        timeline,
        plan=path,
        clip=Path("clip.mp4"),
        source_start_s=0.0,
        duration_s=40.0,
        estimated=False,
        research=plan.research,
    )

    assert "as of 2026-09-17" in report
    assert "Barcelona have won three" in report
    assert "https://example.com/form" in report
    assert plan.lines[0].research_ids == ()
    assert plan.lines[1].research_ids == ("f1",)


def test_empty_visual_analyst_research_line_is_render_compatible(tmp_path: Path) -> None:
    plan_data = _valid_plan()
    metadata = plan_data["metadata"]
    assert isinstance(metadata, dict)
    metadata["research"] = {
        "fixture": "Fixture",
        "as_of": "2026-09-17",
        "facts": [
            {
                "id": "f1",
                "subject": "Team",
                "text": "A sourced background fact.",
                "source_url": "https://example.com/fact",
                "source_date": "2026-09-17",
            }
        ],
    }
    lines = plan_data["lines"]
    assert isinstance(lines, list)
    lines[1]["evidence_ids"] = []
    lines[1]["research_ids"] = ["f1"]
    path = tmp_path / "empty-visual-plan.json"
    path.write_text(json.dumps(plan_data), encoding="utf-8")

    plan = render_demo.read_plan(path)
    timeline = render_demo.plan_timeline(
        plan.lines,
        duration_s=40.0,
        audio_seconds={"l1": 1.0, "l2": 1.0},
    )

    assert timeline.planned[1].line.evidence_ids == ()
    assert timeline.planned[1].line.research_ids == ("f1",)

    lines[1]["research_ids"] = []
    path.write_text(json.dumps(plan_data), encoding="utf-8")
    with pytest.raises(render_demo.RenderError, match="needs visual evidence or research"):
        render_demo.read_plan(path)


def test_research_ids_cannot_be_caller_or_unknown(tmp_path: Path) -> None:
    plan_data = _valid_plan()
    metadata = plan_data["metadata"]
    assert isinstance(metadata, dict)
    metadata["research"] = {
        "fixture": "Fixture",
        "as_of": "2026-09-17",
        "facts": [
            {
                "id": "f1",
                "subject": "Team",
                "text": "A sourced fact.",
                "source_url": "https://example.com/fact",
                "source_date": "2026-09-17",
            }
        ],
    }
    lines = plan_data["lines"]
    assert isinstance(lines, list)
    lines[0]["research_ids"] = ["f1"]
    path = tmp_path / "caller-research.json"
    path.write_text(json.dumps(plan_data), encoding="utf-8")
    with pytest.raises(render_demo.RenderError, match="caller line l1"):
        render_demo.read_plan(path)

    lines[0].pop("research_ids")
    lines[1]["research_ids"] = ["missing"]
    path.write_text(json.dumps(plan_data), encoding="utf-8")
    with pytest.raises(render_demo.RenderError, match="unknown research"):
        render_demo.read_plan(path)


@pytest.mark.parametrize(
    ("path", "mutate", "message"),
    [
        ("mode.json", lambda plan: plan.update(mode="live"), "mode"),
        (
            "time.json",
            lambda plan: plan["lines"][0].update(at_s=math.inf),
            "finite",
        ),
        (
            "voice.json",
            lambda plan: plan["lines"][0].update(voice="narrator"),
            "voice",
        ),
        (
            "text.json",
            lambda plan: plan["lines"][0].update(text="  "),
            "text",
        ),
    ],
)
def test_read_plan_rejects_invalid_mode_bounds_voice_and_text(
    tmp_path: Path, path: str, mutate, message: str
) -> None:
    plan = _valid_plan()
    mutate(plan)
    target = tmp_path / path
    target.write_text(json.dumps(plan, allow_nan=True), encoding="utf-8")

    with pytest.raises(render_demo.RenderError, match=message):
        render_demo.read_plan(target)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda plan: plan["lines"][0].update(evidence_ids=["missing"]),
            "unknown evidence",
        ),
        (
            lambda plan: plan["lines"][0].update(at_s=0.5),
            "future evidence",
        ),
        (
            lambda plan: plan["lines"].reverse(),
            "chronological",
        ),
    ],
)
def test_read_plan_rejects_unknown_future_or_unsorted_evidence(
    tmp_path: Path, mutate, message: str
) -> None:
    plan = _valid_plan()
    mutate(plan)
    target = tmp_path / "invalid-evidence.json"
    target.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(render_demo.RenderError, match=message):
        render_demo.read_plan(target)


def test_captions_escape_text_and_add_plan_recorded_label() -> None:
    lines = [render_demo.RenderLine(0, "l1", "analyst", "A {literal} line", 1.0)]
    timeline = render_demo.plan_timeline(lines, duration_s=5.0, audio_seconds={"l1": 2.0})

    ass = render_demo.captions_ass(timeline, duration_s=5.0, recorded_label=True)

    assert "RECORDED DEMO • CLAUDE + LANGGRAPH" in ass
    assert "ANALYST: A \\{literal\\} line" in ass
    assert "0:00:01.00" in ass and "0:00:03.00" in ass


def test_editorial_reference_caption_uses_honest_label() -> None:
    timeline = render_demo.plan_timeline(
        [render_demo.RenderLine(0, "l1", "caller", "Reference line.", 1.0)],
        duration_s=5.0,
        audio_seconds={"l1": 1.0},
    )
    ass = render_demo.captions_ass(
        timeline, duration_s=5.0, recorded_label=True, editorial_reference=True
    )
    assert "EDITORIAL REFERENCE - NOT PIPELINE OUTPUT" in ass
    assert "RECORDED DEMO" not in ass


def test_ffmpeg_subtitles_detector_reads_filter_name_column(monkeypatch) -> None:
    monkeypatch.setattr(
        render_demo.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=" T.. subtitles V->V Subtitle burn-in\n T.. scale V->V Scale",
            stderr="",
        ),
    )

    assert render_demo.ffmpeg_has_subtitles("ffmpeg") is True


def test_ffmpeg_subtitles_detector_rejects_missing_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        render_demo.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=" T.. scale V->V Scale\n", stderr=""
        ),
    )

    assert render_demo.ffmpeg_has_subtitles("ffmpeg") is False
    with pytest.raises(render_demo.RenderError, match="no `subtitles` filter"):
        render_demo.require_subtitles_filter("ffmpeg")


def test_ffmpeg_resolver_prefers_bundle_but_honors_explicit_override(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: "/bundled/ffmpeg"),
    )

    assert render_demo.resolve_ffmpeg(None) == "/bundled/ffmpeg"
    assert render_demo.resolve_ffmpeg("/custom/ffmpeg") == "/custom/ffmpeg"


def test_plan_caption_render_fails_before_synthesis_when_filter_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"placeholder")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_valid_plan()), encoding="utf-8")
    monkeypatch.setattr(render_demo, "probe_duration", lambda *args, **kwargs: 60.0)
    monkeypatch.setattr(render_demo.shutil, "which", lambda executable: executable)
    monkeypatch.setattr(render_demo, "ffmpeg_has_subtitles", lambda executable: False)
    monkeypatch.setattr(
        render_demo,
        "synthesize",
        lambda *args, **kwargs: pytest.fail("say must not run before caption validation"),
    )

    with pytest.raises(render_demo.RenderError, match="install a subtitles/libass-capable FFmpeg"):
        render_demo.main(
            [
                "--plan",
                str(plan_path),
                "--clip",
                str(clip),
                "--out",
                str(tmp_path / "demo.mp4"),
            ]
        )
