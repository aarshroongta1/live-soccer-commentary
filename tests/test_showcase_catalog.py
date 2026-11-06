"""The browser transcript must match the retained script used to voice each clip."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from render_demo import read_plan  # noqa: E402

CATALOG = json.loads((ROOT / "web/data/demo-clips.json").read_text())


def test_showcase_has_five_distinct_clips() -> None:
    assert len(CATALOG) >= 5
    for field in ("id", "videoPath", "envKey", "planPath"):
        assert len({clip[field] for clip in CATALOG}) == len(CATALOG)
    assert {clip["id"] for clip in CATALOG} >= {
        "argentina-france",
        "barcelona-bayern",
        "arsenal-norwich",
        "portugal-spain",
        "betis-barcelona",
    }


@pytest.mark.parametrize("clip", CATALOG, ids=lambda clip: clip["id"])
def test_browser_transcript_matches_reviewed_render_plan(clip) -> None:
    plan = read_plan(ROOT / clip["planPath"])
    assert clip["duration_s"] == plan.duration_s
    assert clip["lines"] == [
        {"at_s": line.at_s, "voice": line.voice, "text": line.text} for line in plan.lines
    ]
    assert plan.metadata["review"]["not_pipeline_output"] is True
    assert {line.voice for line in plan.lines} == {"caller", "analyst"}
    observations = json.loads(plan.path.read_text())["observations"]
    first_goal = min(o["at_s"] for o in observations if o["kind"] == "goal")
    analyst_cues = [line.at_s for line in plan.lines if line.voice == "analyst"]
    assert any(cue < first_goal for cue in analyst_cues), "Analyst needs a buildup turn"
    assert any(cue > first_goal for cue in analyst_cues), "Analyst needs a response to the finish"
    assert any(
        observation["kind"] == "goal"
        for observation in json.loads(plan.path.read_text())["observations"]
    )
