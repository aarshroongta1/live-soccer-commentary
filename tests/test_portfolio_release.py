"""Featured media must be traceable to generated words and available evidence."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import render_text_demo as renderer  # noqa: E402

BASE = ROOT / "examples/portfolio-demo/release"


def test_featured_text_has_only_the_disclosed_identity_deletion():
    featured = json.loads((ROOT / "web/data/featured-demos.json").read_text())
    for clip in featured:
        name = clip["id"].removeprefix("generated-")
        raw = json.loads((BASE / f"{name}-raw-plan.json").read_text())
        plan = json.loads((BASE / f"{name}-plan.json").read_text())
        edits = json.loads((BASE / f"{name}-edits.json").read_text())
        original = {line["id"]: line for line in raw["lines"]}
        for line in plan["lines"]:
            expected = original[line["id"]]["text"]
            if name == "argentina":
                expected = expected.replace("Argentina’s number 9", "Argentina")
            assert line["text"] == expected
        assert edits["new_football_claims_added"] is False
        assert clip["lines"] == [
            {key: line[key] for key in ("at_s", "voice", "text")} for line in plan["lines"]
        ]


def test_editorial_cues_preserve_evidence_availability_and_readability():
    for name in ("arsenal", "argentina"):
        plan = renderer.read_plan(BASE / f"{name}-plan.json")
        _, captions = renderer.caption_rows(plan, "Recorded example")
        for line, caption in zip(plan.lines, captions, strict=True):
            assert line.at_s >= line.evidence_ready_s
            assert caption["end_s"] - caption["at_s"] >= len(line.text.split()) / 2.6 - 0.02
        assert {line.voice for line in plan.lines} == {"caller", "analyst"}


def test_release_cost_includes_failed_and_unselected_attempts():
    report = json.loads((BASE / "selection.json").read_text())
    assert len(report["attempts"]) == 4
    assert sum(item["selected"] for item in report["attempts"]) == 2
    assert any(item["error"] for item in report["attempts"])
    total = 0
    for item in report["attempts"]:
        trace = json.loads((ROOT / item["trace"]).read_text())
        assert item["budget"] == trace["budget"]
        total += trace["budget"]["reported_usd"]
    assert abs(total - report["reported_total_usd"]) < 1e-9
    assert total < report["max_total_usd"] <= 1
    assert report["unsettled_reservations_usd"] == 0
