"""Small, offline contracts for migrating the lead path into a graph."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from commentary.capture.buffer import Frame
from commentary.config import CaptureConfig, DirectorConfig, PhraserConfig, Settings
from commentary.llm.fake import ScriptedBackend
from commentary.runtime import Runtime
from commentary.schemas import CallerLine, KnowledgePack, PhrasedLine, Player, TeamSheet, Trigger

FIXTURES = Path(__file__).parent / "fixtures" / "orchestration"


def _pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Northvale United",
            short="Northvale",
            starters=[
                Player(name="Tomás Peñaló", number=9),
                Player(name="Errol Kimbanda", number=7),
            ],
        ),
        away=TeamSheet(
            name="Carrowmere City",
            short="Carrowmere",
            starters=[Player(name="Ivo Krastanov", number=11)],
        ),
    )


def _runtime(backend: ScriptedBackend) -> Runtime:
    settings = Settings(
        capture=CaptureConfig(
            width=8,
            height=8,
            fps=2,
            delay_s=2.0,
            history_s=2.0,
            present_offset_s=2.0,
        ),
        phraser=PhraserConfig(model="recorded-phraser"),
        director=DirectorConfig(max_beat_age_s=30.0),
    )
    runtime = Runtime(source=object(), backend=backend, pack=_pack(), settings=settings)
    blank = np.zeros((8, 8, 3), dtype=np.uint8)
    for index in range(12):
        runtime.buffer.append(Frame(ts=index / settings.capture.fps, image=blank))
    return runtime


def _load_cases() -> list[dict[str, Any]]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    cases = json.loads((FIXTURES / manifest["lead_turn_fixture"]).read_text())
    assert [case["id"] for case in cases] == manifest["cases"]
    return cases


CASES = _load_cases()


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
@pytest.mark.asyncio
async def test_legacy_lead_turn_contract(case: dict[str, Any]) -> None:
    """Protect semantic event order while implementation details move."""
    backend = ScriptedBackend()
    phrased = case.get("phraser")
    if isinstance(phrased, dict):
        backend.always("phraser", PhrasedLine.model_validate(phrased))
    runtime = _runtime(backend)
    caller_line = CallerLine.model_validate(case["caller"])

    async def call(*_args: Any, **_kwargs: Any) -> CallerLine:
        return caller_line

    runtime.caller.call = call  # type: ignore[method-assign]
    if case.get("setup") == "home_goal_in_state":
        runtime.state.home_score = 1
        runtime._last_goal_ts = runtime.cursor_ts
    elif case.get("setup") == "foul_in_state":
        runtime.state.last_events = [caller_line.event]
    events_before = list(runtime.state.last_events)

    rows: list[dict[str, Any]] = []
    publish = runtime.bus.publish

    def record(*args: Any, **kwargs: Any) -> Any:
        message = publish(*args, **kwargs)
        rows.append(json.loads(message.to_json()))
        return message

    runtime.bus.publish = record  # type: ignore[method-assign]
    await runtime._call([Trigger.SCHEDULED])

    assert [row["topic"] for row in rows] == case["expected_topics"]
    assert [call.tag for call in backend.calls] == case["expected_backend_tags"]
    beats = [row for row in rows if row["topic"] == "beat"]
    gates = [row for row in rows if row["topic"] == "gate"]
    outcome = case["expected_outcome"]
    assert bool(beats) is (outcome == "spoken")
    assert bool(gates and not gates[-1]["passed"]) is (outcome == "refused")

    if case["id"] == "caller_silence":
        assert runtime.state.last_events[-1:] == [caller_line.event]
    if case["id"] == "replay":
        assert runtime.state.last_events == events_before
        assert beats[0]["preemptable"] is True
    if case["id"] == "confirmed_goal":
        assert beats[0]["event"] == "goal"
        assert beats[0]["preemptable"] is False

    if runtime._goal_turn is not None:
        runtime._goal_turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runtime._goal_turn
