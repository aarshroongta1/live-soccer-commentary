"""Offline checks for archive boundaries and retrieved-source provenance."""

import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMO = runpy.run_path(str(ROOT / "scripts/graphon_review_demo.py"))


def sample(tmp_path):
    archive = tmp_path / "archive"
    manifest = DEMO["prepare"](archive)
    item = manifest["objects"][0]
    text = (archive / "corpus" / item["key"]).read_text()
    span = {
        "key": item["key"],
        "versionId": "saved-version",
        "unit": "chars",
        "start": 0,
        "end": len(text),
    }
    capture = {
        "runs": [
            {
                "query": "false goal",
                "search": {"timingMs": {"total": 10}, "results": [{"spans": [span]}]},
                "read": {"results": [{"span": copy.deepcopy(span), "text": text}], "next": None},
            }
        ]
    }
    return archive, capture


def test_corpus_is_only_review_reports_and_keeps_original_bytes(tmp_path):
    archive = tmp_path / "archive"
    manifest = DEMO["prepare"](archive)
    assert len(manifest["objects"]) == 8
    for item in manifest["objects"]:
        assert item["source"] in DEMO["SOURCES"]
        assert (archive / "corpus" / item["key"]).read_bytes() == (
            ROOT / item["source"]
        ).read_bytes()
    assert not any(p.suffix == ".json" for p in (archive / "corpus").rglob("*"))
    with pytest.raises(FileExistsError):
        DEMO["prepare"](archive)


def test_report_keeps_exact_text_and_version(tmp_path):
    archive, capture = sample(tmp_path)
    output = DEMO["report"](capture, archive)
    assert "Exact snapshot slice: yes" in output
    assert "saved-version" in output
    assert "> " + capture["runs"][0]["read"]["results"][0]["text"].splitlines()[0] in output


def test_report_rejects_changed_span_or_unknown_key(tmp_path):
    archive, capture = sample(tmp_path)
    capture["runs"][0]["read"]["results"][0]["span"]["versionId"] = "other-version"
    with pytest.raises(ValueError, match="changed span"):
        DEMO["report"](capture, archive)
    capture["runs"][0]["search"]["results"][0]["spans"][0]["key"] = "../secret"
    with pytest.raises(ValueError, match="unexpected source"):
        DEMO["report"](capture, archive)


def test_report_detects_modified_snapshot_and_incomplete_reads(tmp_path):
    archive, capture = sample(tmp_path)
    capture["runs"][0]["read"]["next"] = {"spans": []}
    with pytest.raises(ValueError, match="incomplete read"):
        DEMO["report"](capture, archive)
    capture["runs"][0]["read"]["next"] = None
    key = capture["runs"][0]["read"]["results"][0]["span"]["key"]
    (archive / "corpus" / key).write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        DEMO["report"](capture, archive)


def test_actual_captured_passages_match_original_files_without_service_calls(tmp_path):
    archive = tmp_path / "archive"
    DEMO["prepare"](archive)
    capture = json.loads((ROOT / "examples/graphon-review/capture.json").read_text())
    output = DEMO["report"](capture, archive)
    assert output.count("Exact snapshot slice: yes") == 9
    assert "Exact snapshot slice: no" not in output
    assert "6 ready; 2 still indexing" in output
    assert [run["query"] for run in capture["runs"]] == [
        question["query"] for question in DEMO["QUESTIONS"]
    ]
