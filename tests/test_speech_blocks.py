"""A continuous performance may be split only at verified phrase boundaries."""

import base64
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import speech_blocks  # noqa: E402
from render_demo import RenderError, RenderLine  # noqa: E402
from speech_blocks import aligned_cuts  # noqa: E402


def alignment(text):
    return {
        "characters": list(text),
        "character_start_times_seconds": [i * 0.1 for i in range(len(text))],
        "character_end_times_seconds": [(i + 1) * 0.1 for i in range(len(text))],
    }


def test_cuts_preserve_phrases_and_use_the_between_phrase_pause():
    result = aligned_cuts(["Kane...", "He's scored!"], alignment("Kane... He's scored!"))
    assert result[0] == pytest.approx((0, 0.75))
    assert result[1] == pytest.approx((0.75, 2.08))


@pytest.mark.parametrize("fault", ["text", "length", "nan", "backwards"])
def test_bad_alignment_cannot_silently_cut_or_rewrite_words(fault):
    data = alignment("A B")
    if fault == "text":
        data["characters"] = list("A C")
    elif fault == "length":
        data["character_end_times_seconds"].pop()
    elif fault == "nan":
        data["character_start_times_seconds"][0] = float("nan")
    else:
        data["character_end_times_seconds"][2] = 0.01
    with pytest.raises(RenderError, match="alignment"):
        aligned_cuts(["A", "B"], data)


def test_one_request_per_voice_block_preserves_each_phrase(tmp_path, monkeypatch):
    calls, cuts = [], []

    def post(url, **kwargs):
        calls.append(kwargs["json"])
        text = kwargs["json"]["text"]
        return SimpleNamespace(
            raise_for_status=lambda: None,
            headers={"character-cost": "10"},
            json=lambda: {
                "audio_base64": base64.b64encode(b"audio").decode(),
                "alignment": alignment(text),
            },
        )

    def run(command, **kwargs):
        cuts.append(command)
        Path(command[-1]).write_bytes(b"segment")

    monkeypatch.setattr(speech_blocks.subprocess, "run", run)
    monkeypatch.setattr(speech_blocks, "resolve_ffmpeg", lambda _: "ffmpeg")
    monkeypatch.setattr(speech_blocks.shutil, "which", lambda _: "ffmpeg")
    plan = SimpleNamespace(
        lines=[
            RenderLine(0, "c1", "caller", "Kane...", 9),
            RenderLine(1, "c2", "caller", "Scores!", 10),
            RenderLine(2, "a1", "analyst", "Low finish.", 14),
        ]
    )
    speech_blocks.synthesize_blocks(
        SimpleNamespace(post=post),
        plan,
        out_dir=tmp_path,
        api_key="test",
        caller="c",
        analyst="a",
        model="test",
    )
    assert [x["text"] for x in calls] == ["Kane... Scores!", "Low finish."]
    assert set(json.loads((tmp_path / "manifest.json").read_text())) == {"c1", "c2", "a1"}
    assert len(cuts) == 3
    assert (tmp_path / "block-1.response.json").is_file()
    audit = json.loads((tmp_path / "synthesis-audit.json").read_text())
    assert [block["voice_id"] for block in audit["blocks"]] == ["c", "a"]
    assert all(len(block["audio_sha256"]) == 64 for block in audit["blocks"])
    assert "no request overrides" in audit["settings_source"]
