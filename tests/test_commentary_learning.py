"""Offline checks for private labels and frozen human choice capture."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_private_labels_do_not_enter_actor_request():
    harness = module("examples/commentary-learning/spec/harness.py", "learning_harness")
    case = {
        "id": "test",
        "observation": "A pass goes wide.",
        "options": ["A pass.", "[SILENCE]"],
        "human_choice": 1,
        "reason": "PRIVATE HUMAN REASON",
    }
    request = harness.actor_request(case)
    assert "PRIVATE" not in str(request)
    assert "human_choice" not in str(request)
    assert request["questions"][0]["choices"] == [{"value": "option_0"}, {"value": "option_1"}]


@pytest.mark.parametrize("choice", [None, -1, 2, True, "0"])
def test_only_explicit_valid_choices_are_accepted(choice):
    capture = module("scripts/commentary_learning_reviews.py", "learning_capture")
    with pytest.raises(ValueError):
        capture.validate(
            {"choice": choice, "reason": "my reason"}, {"options": ["line", "[SILENCE]"]}
        )


def test_missing_reason_cannot_be_inferred():
    capture = module("scripts/commentary_learning_reviews.py", "learning_capture_reason")
    with pytest.raises(ValueError):
        capture.validate({"choice": 0, "reason": ""}, {"options": ["line", "[SILENCE]"]})
