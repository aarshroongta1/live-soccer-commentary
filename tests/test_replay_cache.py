from __future__ import annotations

from typing import Any

import pytest

from commentary.recorded_demo import Frame
from commentary.replay.cache import SavedVision


def rows() -> list[dict[str, Any]]:
    return [
        {
            "frames_at_s": [0, 1],
            "cutoff_s": 1,
            "resolved": [{"id": "play", "at_s": 1, "kind": "play", "description": "A pass."}],
            "usage": {"latency_s": 4, "cost_usd": 0.01},
        },
        {
            "frames_at_s": [2, 3],
            "cutoff_s": 3,
            "resolved": [{"id": "goal", "at_s": 3, "kind": "goal", "description": "In the net."}],
            "usage": {"latency_s": 3, "cost_usd": 0.02},
        },
    ]


async def test_cache_exposes_only_matching_available_batch_and_separates_prior_cost() -> None:
    delays = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    saved = SavedVision(rows(), rate=2, sleep=sleep)
    result = await saved([Frame(0, b"a"), Frame(1, b"b")], 1)
    assert [item.id for item in result] == ["play"]
    assert delays == [2]
    assert saved.audit[0]["usage"]["cost_usd"] == 0
    assert saved.audit[0]["prior_usage"]["cost_usd"] == 0.01
    result[0].description = "Changed downstream"
    again = await saved([Frame(0, b"a"), Frame(1, b"b")], 1)
    assert again[0].description == "A pass."


async def test_unknown_batch_never_falls_back_to_paid_vision() -> None:
    saved = SavedVision(rows())
    with pytest.raises(ValueError, match="no paid fallback"):
        await saved([Frame(0, b"a"), Frame(2, b"b")], 2)
    assert saved.audit == []


def test_future_observations_in_a_cache_are_rejected_upfront() -> None:
    data = rows()
    data[0]["resolved"][0]["at_s"] = 2
    with pytest.raises(ValueError, match="inside their batch"):
        SavedVision(data)


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1])
def test_invalid_cached_latency_is_rejected(latency: float) -> None:
    data = rows()
    data[0]["usage"]["latency_s"] = latency
    with pytest.raises(ValueError, match="saved vision latency"):
        SavedVision(data)


def test_partial_inspection_cannot_be_reused_as_a_completed_visual_response() -> None:
    data = rows()
    data[0].update(status="failed", resolved=[])
    with pytest.raises(ValueError, match="completed visual batches"):
        SavedVision(data)
