"""No paid calls: memory isolation and transfer-source controls."""

import importlib.util
import json
from pathlib import Path

from commentary.lesson_memory import Lesson, digest, with_lesson


def test_memory_is_guidance_and_does_not_replace_system():
    lesson = Lesson(
        principle="Keep contact grounded.", apply_when="On shots.", preserve="Real saves."
    )
    assert with_lesson("base permissions", None) == "base permissions"
    augmented = with_lesson("base permissions", lesson)
    assert augmented.startswith("base permissions")
    assert "not evidence" in augmented
    assert "take precedence" in augmented
    assert "Real saves." in augmented


def test_future_inputs_have_no_reference_labels_or_seed_match():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "lesson_experiment", root / "scripts/lesson_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failure, cases = module.inputs()
    assert len(cases) == 3
    assert "keeper touch" in failure["correction"]
    for case in cases:
        assert set(case["payload"]) == {"observations"}
        assert digest(case["payload"]) == case["input_sha256"]
        assert (
            case["source_sha256"]
            != "a9169aa010ecf7d166e341ab506d298d29136ec77e7cf1dc30516e1a1efb5a12"
        )
        assert "Kane" not in str(case["payload"])


def test_published_calls_are_unedited_provider_outputs_with_frozen_memory():
    root = Path(__file__).resolve().parents[1]
    directory = root / "examples/lesson-memory"
    report = json.loads((directory / "result.json").read_text())
    lesson = Lesson.model_validate_json((directory / "lesson.json").read_text())
    assert digest(lesson.model_dump()) == report["lesson_sha256"]
    responses = {}
    for path in directory.glob("response-*.json"):
        response = json.loads(path.read_text())
        output = next(o for o in response["response"]["output"] if o["type"] == "message")
        responses[response["tag"]] = json.loads(output["content"][0]["text"])
    for case in report["cases"]:
        assert case["input_sha256"] == digest(case["payload"])
        for arm in case["arms"]:
            assert arm["lines"] == responses[f"{case['id']}_{arm['arm']}"]["lines"]
            expected = with_lesson(
                report["writer_system"], lesson if arm["arm"] == "memory" else None
            )
            assert digest(expected) == arm["system_sha256"]
            words = sum(len(line["text"].split()) for line in arm["lines"])
            assert words == arm["word_count"]
            assert arm["within_word_target"] == (words <= 35)
    assert report["total_reported_usd"] < 0.20
    assert (root / "web/public/learning-calls/data.json").read_bytes() == (
        directory / "result.json"
    ).read_bytes()
