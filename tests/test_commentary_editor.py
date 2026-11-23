"""The actual review-to-context loop with isolated storage and a fake Trajectory client."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "editor_worker", ROOT / "scripts/commentary_editor.py"
)
assert spec and spec.loader
editor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(editor)


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(editor, "DIRECTORY", tmp_path)
    monkeypatch.setattr(editor, "STATE", tmp_path / "session.json")
    (tmp_path / "frames").mkdir()
    for at in editor.DATA["transfer"]["frame_times_s"]:
        (tmp_path / "frames" / f"{at:.1f}.jpg").write_bytes(f"fake-frame-{at}".encode())

    class Image:
        @staticmethod
        def from_path(path):
            return SimpleNamespace(
                part=lambda: {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64," + path.read_bytes().hex()},
                }
            )

    class Links:
        def __init__(self, org):
            pass

        def production(self, agent, trajectory_id):
            return "https://example.invalid/" + trajectory_id

    monkeypatch.setitem(sys.modules, "trajectory.lib", SimpleNamespace(ImageInput=Image))
    monkeypatch.setitem(
        sys.modules, "trajectory.lib.platform_links", SimpleNamespace(PlatformLinks=Links)
    )

    class Client:
        def __init__(self):
            self.records, self.calls, self.events = {}, [], []
            self.trajectories = SimpleNamespace(
                create=self.create,
                log_event=lambda *a, **k: self.events.append(k),
                complete=lambda *a, **k: None,
            )
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.generate))

        def create(self, **kwargs):
            tid = f"fake-{len(self.records)}"
            self.records[tid] = {}
            return SimpleNamespace(tid=tid)

        def patch(self, path, *, body, **kwargs):
            self.records[path.rsplit("/", 1)[-1]].update(body)

        def get(self, path, **kwargs):
            return self.records[path.rsplit("/", 1)[-1]]

        def generate(self, **kwargs):
            self.calls.append(copy.deepcopy(kwargs))
            message = "The changed angle opens room for the finish."
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=message))
                ],
                model_dump=lambda **k: {"model": "fake", "text": message},
            )

    return Client()


def review_body():
    return {
        "decision": "rewrite",
        "rewrite": "An old Arsenal player breaks their clean-sheet run.",
        "reason": "Test fixture only: make the sentence shorter.",
        "request_id": "00000000-0000-4000-8000-000000000001",
        "playback_s": 16,
    }


def test_no_implied_or_empty_human_judgment():
    with pytest.raises(ValueError, match="reason"):
        editor.validate_review({"decision": "keep", "reason": " "})
    with pytest.raises(ValueError, match="Change"):
        editor.validate_review(
            {"decision": "rewrite", "rewrite": editor.DATA["proposed"], "reason": "x"}
        )


def test_explicit_edit_roundtrips_and_retry_does_not_duplicate(isolated):
    state = {}
    result = editor.record_review(isolated, state, review_body())
    assert result["review"]["decision"] == "rewrite"
    assert len(isolated.records) == 1 and not isolated.calls
    stored = next(iter(isolated.records.values()))
    assert stored["feedback_score"] == 0
    assert json.loads(stored["feedback_text"])["rewrite"] == review_body()["rewrite"]
    editor.record_review(isolated, state, review_body())
    assert len(isolated.records) == 1


def test_actual_saved_feedback_is_applied_with_identical_images(isolated):
    state = {}
    editor.record_review(isolated, state, review_body())
    result = editor.compare(isolated, state)
    assert len(isolated.calls) == 2
    before, after = isolated.calls
    assert before["messages"][0] == after["messages"][0]
    assert len(before["messages"]) == 1 and len(after["messages"]) == 2
    assert review_body()["reason"] in after["messages"][1]["content"]
    assert "reveal" not in result and len(result["takes"]) == 2
    editor.compare(isolated, state)
    assert len(isolated.calls) == 2  # Reloading/retrying cannot make more model calls.


def test_failed_comparison_cannot_automatically_respend(isolated):
    state = {}
    editor.record_review(isolated, state, review_body())

    def failure(**kwargs):
        raise RuntimeError("fake failure")

    isolated.chat.completions.create = failure
    with pytest.raises(RuntimeError, match="fake failure"):
        editor.compare(isolated, state)
    with pytest.raises(ValueError, match="attempt"):
        editor.compare(isolated, state)


def test_human_can_report_no_improvement_and_then_reveal(isolated):
    state = {}
    editor.record_review(isolated, state, review_body())
    editor.compare(isolated, state)
    result = editor.vote(isolated, state, {"choice": "neither", "reason": "Both are too generic."})
    assert result["vote"]["choice"] == "neither" and "reveal" in result
    tid = state["comparison"]["with_feedback"]["trajectory_id"]
    stored = json.loads(isolated.records[tid]["feedback_text"])
    assert stored["judge"] == "human" and stored["training"] is False
