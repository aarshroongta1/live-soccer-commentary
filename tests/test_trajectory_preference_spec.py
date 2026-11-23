"""Offline contract/end-reason checks; no Trajectory API or model calls."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "commentary_preference_harness", ROOT / "examples/trajectory-preferences/spec/harness.py"
)
assert spec and spec.loader
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


@pytest.mark.parametrize(
    ("text", "finish", "expected_reward", "ending"),
    [
        ('{"choice":"B","reason":"It adds context without repetition."}', "stop", 1, "ENV_DONE"),
        ('{"choice":"D","reason":"An invalid choice."}', "stop", 0, "ENV_DONE"),
        ('{"choice":"B"', "length", None, "TRUNCATION"),
        ("", "content_filter", None, "ERROR"),
    ],
)
def test_format_grade_is_separate_from_human_taste(
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    finish: str,
    expected_reward: int | None,
    ending: str,
) -> None:
    rewards, endings, events = [], [], []
    client = SimpleNamespace(
        trajectories=SimpleNamespace(
            create=lambda: SimpleNamespace(tid="fake-only"),
            log_reward=lambda *args, **kwargs: rewards.append(kwargs),
            log_event=lambda *args, **kwargs: events.append(kwargs),
            complete=lambda *args, **kwargs: endings.append(kwargs["termination_reason"]),
        ),
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **kwargs: SimpleNamespace(
                    choices=[
                        SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=text))
                    ]
                )
            )
        ),
    )

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            pass

        def __enter__(self) -> SimpleNamespace:
            return client

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setitem(sys.modules, "trajectory", SimpleNamespace(Client=FakeClient))
    monkeypatch.setenv("COMMENTARY_TASK_ID", "penalty-reaction")
    harness.main()
    assert endings == [ending]
    if expected_reward is None:
        assert not rewards and not events
    else:
        assert rewards == [{"name": "response_contract_only", "value": float(expected_reward)}]
        assert events[0]["payload"]["human_preference"] is None


def test_model_error_produces_no_grade(monkeypatch: pytest.MonkeyPatch) -> None:
    endings = []

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self.trajectories = SimpleNamespace(
                create=lambda: SimpleNamespace(tid="fake-only"),
                complete=lambda *args, **kwargs: endings.append(kwargs["termination_reason"]),
            )
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.fail))

        def fail(self, **kwargs: object) -> None:
            raise RuntimeError("simulated provider failure")

        def __enter__(self) -> "FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setitem(sys.modules, "trajectory", SimpleNamespace(Client=FakeClient))
    monkeypatch.setenv("COMMENTARY_TASK_ID", "penalty-reaction")
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        harness.main()
    assert endings == ["ERROR"]
