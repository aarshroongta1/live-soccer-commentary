"""Saved speech must survive the mux at its cues, without any paid calls."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from add_demo_narration import add_narration  # noqa: E402


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_real_mux_keeps_late_speech_audible(tmp_path):
    plan = ROOT / "examples/portfolio-demo/penalty-plan.json"
    video, tone, output = (tmp_path / name for name in ["base.mp4", "tone.mp3", "voiced.mp4"])
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=16x16:rate=1",
            "-t",
            "20",
            "-c:v",
            "libx264",
            str(video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=0.3",
            str(tone),
        ],
        check=True,
    )
    lines = json.loads(plan.read_text())["lines"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({line["id"]: str(tone) for line in lines}))
    (tmp_path / "synthesis-audit.json").write_text(
        json.dumps(
            {
                "model": "scripted-tone",
                "lines": [
                    {**line, "voice_id": line["voice"], "credit_headers": {}} for line in lines
                ],
            }
        )
    )
    report = add_narration(plan, video, manifest, output)
    assert len(report["lines"]) == 4
    assert all(line["exported_rms_dbfs"] > -40 for line in report["lines"])
    assert report["lines"][-1]["at_s"] == 15.431
    assert report["tts_calls_during_mux"] == 0
    with pytest.raises(ValueError, match="overwrite"):
        add_narration(plan, video, manifest, output)
