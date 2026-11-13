"""The release gate must reject audible-but-mostly-empty demo exports."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from render_edited_showcase import measure_narration_gaps  # noqa: E402


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is required")
def test_audible_audio_is_not_enough_when_it_leaves_a_long_gap(tmp_path):
    sparse = tmp_path / "sparse.wav"
    full = tmp_path / "full.wav"
    for path, tone, padding in [(sparse, 1, 3), (full, 4, 0)]:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={tone}",
                "-af",
                f"apad=pad_dur={padding}",
                str(path),
            ],
            check=True,
        )
    with pytest.raises(ValueError, match="long silent gap"):
        measure_narration_gaps(sparse, 4)
    assert measure_narration_gaps(full, 4)["speech_occupancy"] > 0.95


def test_release_reports_match_the_edited_scripts_and_have_no_dead_opening():
    catalog = json.loads((ROOT / "web/data/edited-showcase.json").read_text())
    for clip in catalog:
        plan = json.loads((ROOT / clip["planPath"]).read_text())
        report = json.loads(
            (ROOT / "web/public" / clip["videoPath"].lstrip("/"))
            .with_suffix(".narration.json")
            .read_text()
        )
        assert clip["kind"] == "edited-demo"
        assert plan["metadata"]["editorial_reference"] is True
        assert clip["lines"] == [
            {k: line[k] for k in ("at_s", "voice", "text")} for line in plan["lines"]
        ]
        assert [line["text"] for line in report["lines"]] == [
            line["text"] for line in plan["lines"]
        ]
        assert clip["lines"][0]["at_s"] <= 0.25
        assert report["measured_audio"]["longest_gap_s"] <= 1.6
        assert report["measured_audio"]["speech_occupancy"] >= 0.72
