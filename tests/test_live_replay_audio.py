"""The public replay must never play broadcast speech beneath the analyst."""

import base64
import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest


def samples(media):
    raw = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(media),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "8000",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    return np.frombuffer(raw, dtype="<f4")


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_replay_contains_only_generated_voice_and_live_footage_is_silent(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "prepare_live_replay", "scripts/prepare_live_search_replay.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "clips/demo-sources/ballard-arsenal.mp4"
    source.parent.mkdir(parents=True)
    public = tmp_path / "public"
    public.mkdir()
    (tmp_path / "runs/outreach").mkdir(parents=True)
    # Distinct synthetic tracks stand in for broadcast speech and generated speech.
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=160x90:r=5:d=22",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=22",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    voice = tmp_path / "voice.mp3"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=880:duration=0.5",
            str(voice),
        ],
        check=True,
    )
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "run_id": "scripted",
                "events": [
                    {
                        "type": "audio_ready",
                        "at_ms": 0,
                        "audio_base64": base64.b64encode(voice.read_bytes()).decode(),
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "PUBLIC", public)
    monkeypatch.setattr(sys, "argv", ["prepare_live_search_replay.py", "--trace", str(trace)])
    module.main()

    live = samples(public / "ballard-original.mp4")
    replay = samples(public / "ballard-live-replay.mp4")
    assert np.max(np.abs(live)) < 1e-5, "Live footage still contains broadcast audio"
    assert np.max(np.abs(replay[8000:24000])) < 1e-5, "Broadcast audio before the analyst"
    assert np.max(np.abs(replay[80000:144000])) < 1e-5, "Broadcast audio after the analyst"
    assert np.sqrt(np.mean(replay[33600:35200] ** 2)) > 0.01, "Generated voice is missing"
