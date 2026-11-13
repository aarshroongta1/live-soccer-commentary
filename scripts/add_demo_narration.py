#!/usr/bin/env python3
"""Mux saved, audited TTS into a captioned demo. No API calls or speech fallback."""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from render_demo import probe_duration, read_plan


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_narration(
    plan_path: Path, video: Path, manifest_path: Path, output: Path
) -> dict[str, Any]:
    plan = read_plan(plan_path)
    manifest = json.loads(manifest_path.read_text())
    audit_path = manifest_path.with_name("synthesis-audit.json")
    audit = json.loads(audit_path.read_text())
    audited = {line["id"]: line for line in audit["lines"]}
    report_path = output.with_suffix(".narration.json")
    if output.exists() or report_path.exists():
        raise ValueError("refusing to overwrite output or report")
    if abs(probe_duration(video, ffprobe="ffprobe") - plan.duration_s) > 0.1:
        raise ValueError("captioned video duration does not match plan")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video)]
    filters, rows = [], []
    for index, line in enumerate(plan.lines):
        record = audited[line.id]
        if record["text"] != line.text or record["voice"] != line.voice:
            raise ValueError(f"audited speech differs from plan: {line.id}")
        audio = Path(manifest[line.id])
        duration = probe_duration(audio, ffprobe="ffprobe")
        next_cue = (
            plan.lines[index + 1].at_s - 0.08
            if index + 1 < len(plan.lines)
            else plan.duration_s - 0.08
        )
        available = next_cue - line.at_s
        if available <= 0:
            raise ValueError("speech cues must be increasing and inside clip")
        speed = max(1.0, duration / available)
        if not math.isfinite(speed) or speed > 1.25:
            raise ValueError(f"{line.id} cannot fit without more than 1.25x speed")
        command.extend(["-i", str(audio)])
        delay_ms = round(line.at_s * 1000)
        filters.append(
            f"[{index + 1}:a]asetpts=N/SR/TB,atempo={speed:.8f},"
            f"adelay={delay_ms}:all=1,asetpts=N/SR/TB[a{index}]"
        )
        rows.append(
            {
                "id": line.id,
                "voice": line.voice,
                "voice_id": record["voice_id"],
                "text": line.text,
                "at_s": line.at_s,
                "end_s": round(line.at_s + duration / speed, 6),
                "input_duration_s": duration,
                "speed": round(speed, 6),
                "audio_sha256": sha256(audio),
                "provider_character_cost": record["credit_headers"].get("character-cost"),
            }
        )
    inputs = "".join(f"[a{i}]" for i in range(len(rows)))
    filters.append(
        f"{inputs}amix=inputs={len(rows)}:normalize=0,apad,"
        f"atrim=duration={plan.duration_s},loudnorm=I=-16:TP=-1.5:LRA=11,"
        "asetpts=N/SR/TB[out]"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "[out]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-t",
            str(plan.duration_s),
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    subprocess.run(command, check=True)
    decoded = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ]
    )
    samples = array.array("h", decoded)
    if sys.byteorder != "little":
        samples.byteswap()
    if abs(len(samples) / 16000 - plan.duration_s) > 0.1:
        raise ValueError("exported audio duration does not match clip")
    for row in rows:
        segment = samples[round(row["at_s"] * 16000) : round(row["end_s"] * 16000)]
        rms = math.sqrt(sum(float(x) ** 2 for x in segment) / max(1, len(segment))) / 32768
        if rms < 0.001:
            raise ValueError(f"exported speech is silent: {row['id']}")
        row["exported_rms_dbfs"] = round(20 * math.log10(rms), 2)
    report = {
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "captioned_video": str(video),
        "captioned_video_sha256": sha256(video),
        "manifest": str(manifest_path),
        "synthesis_model": audit["model"],
        "output": str(output),
        "output_sha256": sha256(output),
        "duration_s": plan.duration_s,
        "audio": True,
        "tts_calls_during_mux": 0,
        "video_stream": "copied unchanged; existing captions and footage retained",
        "caption_timing": "Original display intervals; speech starts at each caption cue",
        "speech_words_edited": False,
        "lines": rows,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = add_narration(args.plan, args.video, args.manifest, args.out)
    print(f"Added {len(result['lines'])} saved speech lines to {args.out}; no API calls")


if __name__ == "__main__":
    main()
