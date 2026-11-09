#!/usr/bin/env python3
"""Render a saved commentary plan as a silent captioned clip. No TTS or model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from render_demo import (
    _ass_text,
    _ass_timestamp,
    _subtitle_filter,
    read_plan,
    require_subtitles_filter,
    resolve_ffmpeg,
)

from commentary.recorded_demo import validate_inputs


def caption_rows(plan, label: str) -> tuple[str, list[dict]]:
    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1280",
            "PlayResY: 720",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Caption,Arial,48,&H00FFFFFF,&H00FFFFFF,&H00201C18,&HC0201C18,"
            "0,0,0,0,100,100,0,0,3,10,0,2,70,70,32,1",
            "Style: Label,Arial,17,&H00FFFFFF,&H00FFFFFF,&H00201C18,&H80201C18,"
            "0,0,0,0,100,100,0,0,3,6,0,8,20,20,15,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            "",
        ]
    )
    events = [
        f"Dialogue: 0,0:00:00.00,{_ass_timestamp(plan.duration_s)},Label,,0,0,0,,{_ass_text(label)}"
    ]
    intervals = []
    for index, line in enumerate(plan.lines):
        next_at = plan.lines[index + 1].at_s if index + 1 < len(plan.lines) else plan.duration_s
        end = min(
            plan.duration_s, next_at - 0.08, line.at_s + max(1.5, len(line.text.split()) / 2.6)
        )
        if end <= line.at_s:
            raise ValueError("caption has no readable display interval")
        name = "Caller" if line.voice == "caller" else "Analyst"
        events.append(
            f"Dialogue: 0,{_ass_timestamp(line.at_s)},{_ass_timestamp(end)},Caption,,0,0,0,,"
            f"{name}: {_ass_text(line.text)}"
        )
        intervals.append(
            {"id": line.id, "at_s": line.at_s, "end_s": end, "voice": line.voice, "text": line.text}
        )
    return header + "\n".join(events) + "\n", intervals


def render(plan_path: Path, clip: Path, output: Path, label: str) -> dict:
    plan = read_plan(plan_path)
    if clip.name != plan.source_basename:
        raise ValueError("plan source basename does not match clip")
    validate_inputs(clip, plan.source_start_s, plan.duration_s)
    captions = output.with_suffix(".ass")
    report_path = output.with_suffix(".render.json")
    for path in (output, captions, report_path):
        if path.exists() or path.resolve() in {clip.resolve(), plan_path.resolve()}:
            raise ValueError(f"refusing to overwrite {path}")
    ass, intervals = caption_rows(plan, label)
    ffmpeg = resolve_ffmpeg(None)
    require_subtitles_filter(ffmpeg)
    output.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text(ass)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            str(plan.source_start_s),
            "-t",
            str(plan.duration_s),
            "-i",
            str(clip),
            "-an",
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
            + _subtitle_filter(captions.resolve()),
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
            str(output),
        ],
        check=True,
    )
    report = {
        "source": str(clip),
        "source_start_s": plan.source_start_s,
        "duration_s": plan.duration_s,
        "source_sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
        "plan": str(plan_path),
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "audio": False,
        "tts_calls": 0,
        "words_edited": False,
        "caption_timing": "Source cues with estimated reading time; not speech latency.",
        "lines": intervals,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="Automatic dialogue | saved perception | recorded clip")
    args = parser.parse_args()
    result = render(args.plan, args.clip, args.out, args.label)
    print(f"Rendered {args.out}: {len(result['lines'])} lines, no audio")


if __name__ == "__main__":
    main()
