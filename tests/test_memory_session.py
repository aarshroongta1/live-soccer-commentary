"""Verify cumulative memory provenance without making model calls."""

import json
from pathlib import Path

from commentary.lesson_memory import digest

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "examples/lesson-memory/session"


def test_history_is_exact_saved_calls_with_review_provenance():
    report = json.loads((SAVED / "session.json").read_text())
    source = json.loads((ROOT / report["source_path"]).read_text())
    assert digest(source) == report["source_sha256"]
    assert len(report["history"]) == len(report["memory"]) == 3
    for index, item in enumerate(report["history"]):
        original = source["rejected_replays"][item["source_index"]]
        assert item["observations"] == original["observations"]
        assert item["call"] in [s["line"] for s in original["speech"]]
        assert item["review_origin"]
        memory = report["memory"][index]
        assert memory["from_call"] == index + 1
        raw = json.loads((SAVED / f"lesson-{memory['id']}.json").read_text())
        output = next(o for o in raw["output"] if o["type"] == "message")
        assert memory["lesson"] == json.loads(output["content"][0]["text"])


def test_later_calls_receive_all_accumulated_lessons_and_prior_dialogue():
    report = json.loads((SAVED / "session.json").read_text())
    recent = []
    for call in report["later_calls"]:
        assert call["status"] == "generated"
        assert digest(call["input"]) == call["input_sha256"]
        assert call["input"]["saved_lessons"] == report["memory"]
        assert call["input"]["recent_dialogue"] == recent
        assert set(call["output"]["lesson_ids"]) <= {"L1", "L2", "L3"}
        raw = json.loads((SAVED / f"call-{call['id']}.json").read_text())
        output = next(o for o in raw["output"] if o["type"] == "message")
        assert call["output"] == json.loads(output["content"][0]["text"])
        if not call["output"]["abstain"]:
            recent.append({k: call["output"][k] for k in ("voice", "text")})
    assert report["budget"]["reported_usd"] < 0.10
    assert report["budget"]["unsettled_requests"] == 0
    assert (ROOT / "web/public/learning-calls/session.json").read_bytes() == (
        SAVED / "session.json"
    ).read_bytes()
