"""The screen bridge must reject bad frames before making paid requests."""

import base64
import importlib.util
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "stream_worker", Path(__file__).parents[1] / "scripts/stream_worker.py"
)
assert _SPEC and _SPEC.loader
worker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(worker)


@pytest.fixture
def jpeg():
    ok, image = cv2.imencode(".jpg", np.zeros((64, 96, 3), dtype=np.uint8))
    assert ok
    return base64.b64encode(image).decode()


def test_valid_jpeg_is_stamped_without_future_frames(jpeg):
    frames = worker._frames([{"at_s": 1.25, "jpeg": jpeg}], 2, -1)
    assert frames[0].at_s == 1.25
    assert frames[0].jpeg != base64.b64decode(jpeg)


@pytest.mark.parametrize("at_s", [-0.5, True, float("nan"), 61, 2.01])
def test_rejects_bad_or_future_timestamp(jpeg, at_s):
    with pytest.raises(ValueError):
        worker._frames([{"at_s": at_s, "jpeg": jpeg}], 2, -1)


def test_rejects_replayed_frames_and_excess_batch(jpeg):
    frame = {"at_s": 1, "jpeg": jpeg}
    for raw, previous in [([frame], 1), ([frame, frame], -1), ([frame] * 5, -1)]:
        with pytest.raises(ValueError):
            worker._frames(raw, 2, previous)


def test_rejects_bad_and_oversized_jpeg_before_decode(monkeypatch):
    def no_decode(*_args):
        pytest.fail("invalid payload reached image decoder")

    monkeypatch.setattr(worker.cv2, "imdecode", no_decode)
    for jpeg in ["not base64!", base64.b64encode(b"not JPEG").decode(), "a" * 2_000_001]:
        with pytest.raises(ValueError):
            worker._frames([{"at_s": 1, "jpeg": jpeg}], 2, -1)


def test_context_uses_real_pack_path_and_no_generic_files(monkeypatch):
    reads = []

    def read(path, *_args, **_kwargs):
        reads.append(str(path))
        if path.name == "pack-argfra-2022.json":
            return json.dumps({"home": {"name": "Argentina"}, "away": {"name": "France"}})
        return json.dumps(
            {"result": {"fixture": "Argentina v France", "as_of": "2022-12-17", "facts": []}}
        )

    monkeypatch.setattr(Path, "read_text", read)
    assert worker._context("generic") == (None, None)
    assert not reads
    pack, research = worker._context("argfra")
    assert pack["teams"]["home"]["name"] == "Argentina"
    assert research.fixture == "Argentina v France"
    assert reads[0].endswith("clips/pack-argfra-2022.json")
    assert reads[1].endswith("argfra-all-openai-v1-plan.json.run/research.json")
    with pytest.raises(ValueError):
        worker._context("unknown")


async def run_worker(monkeypatch, capsys, requests, *, cost=0.01, fail=False):
    calls = []
    factories = []
    clock = iter([0, 0, *([80] * 100)])
    monkeypatch.setattr(worker.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(worker, "openai_factory", lambda **kw: factories.append(kw))
    monkeypatch.setattr(worker.sys, "stdin", io.StringIO("\n".join(map(json.dumps, requests))))

    class Commentator:
        def __init__(self, _backend, **kwargs):
            assert kwargs["observer_model"] == kwargs["writer_model"] == "gpt-5.6-terra"

        async def process(self, frames, *, playback_s):
            calls.append((frames, playback_s))
            if fail:
                raise ValueError("model failed")
            return {"line": None, "usage": {"cumulative": {"cost_usd": cost}}}

    monkeypatch.setattr(worker, "StreamingCommentator", Commentator)
    await worker.main()
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()], calls, factories


@pytest.mark.asyncio
async def test_spend_is_explicit_and_eof_never_starts_requests(monkeypatch, capsys):
    replies, calls, factories = await run_worker(
        monkeypatch, capsys, [{"action": "start", "context": "generic", "spend": False}]
    )
    assert replies[0]["ok"] is False
    assert not calls and not factories


@pytest.mark.asyncio
async def test_batch_limit_and_stop_prevent_extra_calls(monkeypatch, capsys, jpeg):
    requests = [{"action": "start", "context": "generic", "spend": True}]
    requests.extend(
        {"action": "batch", "playback_s": i, "frames": [{"at_s": i, "jpeg": jpeg}]}
        for i in range(1, 14)
    )
    replies, calls, factories = await run_worker(monkeypatch, capsys, requests)
    assert len(calls) == 12 and len(factories) == 1
    assert replies[-1]["ok"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_cost_or_model_failure_stops_session(monkeypatch, capsys, jpeg, fail):
    batch = {"action": "batch", "playback_s": 1, "frames": [{"at_s": 1, "jpeg": jpeg}]}
    replies, calls, _ = await run_worker(
        monkeypatch,
        capsys,
        [{"action": "start", "context": "generic", "spend": True}, batch, batch],
        cost=0.36,
        fail=fail,
    )
    assert len(calls) == 1
    assert replies[-1]["ok"] is False
    if not fail:
        assert replies[1]["result"]["stopped_reason"]
