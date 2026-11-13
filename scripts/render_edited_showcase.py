#!/usr/bin/env python3
"""Build the three editorial showcase clips from saved speech; never calls an API."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

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
WORK = ROOT / "runs/demo/commentary-edit-v2"
EDITED = ROOT / "examples/portfolio-demo/edited-showcase"
SETTINGS = {
    "penalty": ([0.15, 2.8, 6.25, 10.0, 13.65], [1.0, 1.14, 1.0, 1.0, 0.85]),
    "arsenal": ([0.15, 1.7, 3.9, 6.15, 7.6, 10.0], [1.1, 1.2, 1.25, 1.15, 1.0, 0.95]),
    "argentina": ([0.15, 2.15, 3.8, 7.0, 9.7, 12.95], [1.0, 1.0, 1.0, 1.0, 1.0, 1.05]),
}


def run(command: list[str]) -> None:
    if "-vf" in command:
        command[0] = resolve_ffmpeg(None)
    subprocess.run(command, check=True)


def measure_narration_gaps(video: Path, duration_s: float) -> dict[str, float]:
    """Reject the sparse speech and empty openings in the previous release."""
    probe = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(video),
            "-vn",
            "-af",
            "silencedetect=noise=-40dB:d=0.2",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gaps = [float(t) for t in re.findall(r"silence_duration: ([0-9.]+)", probe.stderr)]
    longest = max(gaps, default=0)
    occupancy = 1 - sum(gaps) / duration_s
    if longest > 1.6:
        raise ValueError(f"long silent gap: {longest:.3f}s")
    if occupancy < 0.72:
        raise ValueError(f"sparse narration: {occupancy:.0%}")
    return {
        "silence_threshold_db": -40,
        "minimum_gap_s": 0.2,
        "longest_gap_s": longest,
        "speech_occupancy": occupancy,
    }


def build(name: str) -> None:
    plan_path = EDITED / f"{name}-plan.json"
    raw = json.loads(plan_path.read_text())
    starts, rates = SETTINGS[name]
    for line, observation, start in zip(raw["lines"], raw["observations"], starts, strict=True):
        line["at_s"] = observation["at_s"] = start
    raw["metadata"]["speech_rates"] = dict(
        zip((item["id"] for item in raw["lines"]), rates, strict=True)
    )
    plan_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n")
    plan = read_plan(plan_path)
    audio_dir = WORK / "audio" / name
    manifest = json.loads((audio_dir / "manifest.json").read_text())
    original_audit = json.loads((audio_dir / "synthesis-audit.json").read_text())
    audit_lines = {}
    for block in original_audit["blocks"]:
        matching = [item for item in raw["lines"] if item["id"] in block["line_ids"]]
        assert " ".join(item["text"] for item in matching) == block["request"]["text"]
        for line in matching:
            audit_lines[line["id"]] = {
                **line,
                "voice_id": block["voice_id"],
                "credit_headers": {},
                "provider_request_id": block["provider_request_id"],
            }
    if name == "penalty":
        previous = ROOT / "runs/demo/showcase-narration-v1/penalty"
        manifest["line-4"] = json.loads((previous / "manifest.json").read_text())["line-3"]
        old = next(
            item
            for item in json.loads((previous / "synthesis-audit.json").read_text())["lines"]
            if item["id"] == "line-3"
        )
        assert old["text"] == raw["lines"][3]["text"]
        audit_lines["line-4"] = {**old, "id": "line-4", "reused": True}
    prepared = WORK / "prepared" / name
    prepared.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    intervals: list[dict[str, Any]] = []
    for line, speed in zip(plan.lines, rates, strict=True):
        target = prepared / f"{line.id}.wav"
        run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-n",
                "-i",
                manifest[line.id],
                "-af",
                f"atempo={speed},asetpts=N/SR/TB",
                "-c:a",
                "pcm_s16le",
                str(target),
            ]
        )
        duration = probe_duration(target, ffprobe="ffprobe")
        saved[line.id] = str(target)
        intervals.append(
            {
                "id": line.id,
                "at_s": line.at_s,
                "end_s": line.at_s + duration,
                "voice": line.voice,
                "text": line.text,
                "speed": speed,
            }
        )
    for i, row in enumerate(intervals):
        end = intervals[i + 1]["at_s"] if i + 1 < len(intervals) else plan.duration_s
        assert row["end_s"] <= end - 0.06, f"speech does not fit: {name} {row}"
    (prepared / "manifest.json").write_text(json.dumps(saved, indent=2) + "\n")
    (prepared / "synthesis-audit.json").write_text(
        json.dumps(
            {
                "model": original_audit["model"],
                "mode": "saved_voice_blocks_with_local_pacing",
                "original_audit": str(audio_dir / "synthesis-audit.json"),
                "lines": list(audit_lines.values()),
            },
            indent=2,
        )
        + "\n"
    )
    header = caption_rows(plan, "")[0].split("Dialogue:", 1)[0]
    events = []
    for row in intervals:
        role = "Caller" if row["voice"] == "caller" else "Analyst"
        events.append(
            f"Dialogue: 0,{_ass_timestamp(row['at_s'])},{_ass_timestamp(row['end_s'])},"
            f"Caption,,0,0,0,,{role}: {_ass_text(row['text'])}"
        )
    captions = prepared / "captions.ass"
    captions.write_text(header + "\n".join(events) + "\n")
    source = (
        ROOT
        / "clips"
        / ("action-sources" if name == "penalty" else "demo-sources")
        / plan.source_basename
    )
    base = prepared / "captioned.mp4"
    run(
        [
            "ffmpeg",
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
        ]
    )
    output = ROOT / f"web/public/soccer-commentary/showcase-{name}-edited-v2.mp4"
    report = add_narration(plan_path, base, prepared / "manifest.json", output)
    report.update(
        {
            "editorial_reference": True,
            "words_edited": True,
            "original_synthesis": original_audit,
            "pacing": intervals,
        }
    )
    measured = measure_narration_gaps(output, plan.duration_s)
    report["measured_audio"] = measured
    longest, occupancy = measured["longest_gap_s"], measured["speech_occupancy"]
    output.with_suffix(".narration.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"{name}: longest pause {longest:.2f}s; speech occupancy {occupancy:.0%}", flush=True)


if __name__ == "__main__":
    for clip_name in SETTINGS:
        build(clip_name)
