"""Published iteration must preserve raw words and matched evidence."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
BASE = ROOT / "examples/portfolio-demo"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_comparison_is_the_same_clip_evidence_and_research():
    before = json.loads((BASE / "before-trace.json").read_text())
    after = json.loads((BASE / "penalty-trace.json").read_text())
    record = json.loads((BASE / "iteration.json").read_text())
    for field in ("source", "observations", "raw_observations", "research"):
        assert before[field] == after[field]
        assert record["input_hashes"][field] == digest(before[field])
    assert record["word_edits"] is False
    assert record["before"]["words"] == record["after"]["words"] == 33


def test_before_after_transcripts_match_retained_model_output():
    view = json.loads((ROOT / "web/data/iteration-demo.json").read_text())
    before = json.loads((BASE / "before-trace.json").read_text())
    plan = json.loads((BASE / "before-plan.json").read_text())
    assert plan["lines"] == before["dialogue"]["lines"]
    assert view["before"]["clip"]["lines"] == [
        {k: line[k] for k in ("at_s", "voice", "text")} for line in plan["lines"]
    ]
    rejected = [
        e["data"]
        for e in before["role_audit"]
        if e["event"] == "draft_review" and not e["data"]["accepted"]
    ]
    assert len(rejected) == 2
    assert view["before"]["preparation"]["generation"]["calls"] == before["budget"]["attempts"]
