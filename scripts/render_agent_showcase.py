#!/usr/bin/env python3
"""Render selected agent words with saved TTS, measured captions and role shares.

No API calls. Existing outputs are protected. Generation attempts and source
excerpt decisions are retained in examples/portfolio-demo/agent-showcase.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from add_demo_narration import add_narration
from render_demo import (
    _ass_text,
    _ass_timestamp,
    _subtitle_filter,
    probe_duration,
    read_plan,
    resolve_ffmpeg,
)
from render_text_demo import caption_rows

ROOT = Path(__file__).resolve().parents[1]


def build(name: str, version: int, *, grounded: bool = False) -> dict:
    release = "grounded-showcase" if grounded else "agent-showcase"
    prefix = "grounded" if grounded else "authentic"
    tag = "grounded-v1" if grounded else "balanced-v1"
    plan_path = ROOT / f"examples/portfolio-demo/{release}/{name}-plan.json"
    plan = read_plan(plan_path)
    work = ROOT / f"runs/demo/{prefix}-v{version}/{name}"
    manifest = work / "audio/manifest.json"
    saved = json.loads(manifest.read_text())
    captions, base = work / "measured.ass", work / "measured.mp4"
    output = ROOT / f"web/public/soccer-commentary/agent-{name}-{tag}.mp4"
    if any(p.exists() for p in (captions, base, output, output.with_suffix(".narration.json"))):
        raise ValueError("refusing to overwrite showcase artifacts")
    header = caption_rows(plan, "")[0].split("Dialogue:", 1)[0]
    events = []
    for i, line in enumerate(plan.lines):
        duration = probe_duration(Path(saved[line.id]), ffprobe="ffprobe")
        end = plan.lines[i + 1].at_s - 0.08 if i + 1 < len(plan.lines) else plan.duration_s - 0.08
        speed = max(1, duration / (end - line.at_s))
        if speed > 1.25:
            raise ValueError(f"speech cannot fit: {name} {line.id}")
        end = line.at_s + duration / speed
        events.append(
            f"Dialogue: 0,{_ass_timestamp(line.at_s)},{_ass_timestamp(end)},Caption,,0,0,0,,"
            f"{line.voice.title()}: {_ass_text(line.text)}"
        )
    captions.write_text(header + "\n".join(events) + "\n")
    source = (
        ROOT
        / "clips"
        / ("action-sources" if name == "penalty" else "demo-sources")
        / plan.source_basename
    )
    subprocess.run(
        [
            resolve_ffmpeg(None),
            "-v",
            "error",
            "-n",
            "-ss",
            str(plan.source_start_s),
            "-t",
            str(plan.duration_s),
            "-i",
            str(source),
            "-an",
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
            + _subtitle_filter(captions),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(base),
        ],
        check=True,
    )
    report = add_narration(plan_path, base, manifest, output)
    seconds = {
        role: sum(row["end_s"] - row["at_s"] for row in report["lines"] if row["voice"] == role)
        for role in ("caller", "analyst")
    }
    total = sum(seconds.values())
    shares = {role: round(value / total, 4) for role, value in seconds.items()}
    report.update(
        speaker_seconds=seconds,
        speaker_shares=shares,
        words_edited=False,
        note=(
            "Selected agent output on reused perception; saved speech; not live latency. "
            "Shares include TTS phrase pauses."
        ),
    )
    output.with_suffix(".narration.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(value < 0.35 or value > 0.65 for value in shares.values()):
        raise ValueError(f"unbalanced rendered speech: {shares}")
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-n",
            "-ss",
            "0.2",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(output.with_suffix(".jpg")),
        ],
        check=True,
    )
    print(name, shares, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounded", action="store_true")
    args = parser.parse_args()
    selected = (
        (("penalty", 1), ("arsenal", 2), ("argentina", 3))
        if args.grounded
        else (("penalty", 3), ("arsenal", 3), ("argentina", 2))
    )
    for name, version in selected:
        build(name, version, grounded=args.grounded)
