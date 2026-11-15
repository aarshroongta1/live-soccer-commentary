#!/usr/bin/env python3
"""Publish a real live-run trace and mux its exact generated speech for replay."""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
from pathlib import Path

from commentary.live_research import QUERY

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web/public/soccer-commentary"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_text()
    search_source = None
    if args.trace.suffix == ".ndjson":
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        run_id = events[0]["run_id"]
        search_source = events[0].get("search_source")
    else:
        audit = json.loads(raw)
        events = audit["events"]
        run_id = audit["run_id"]
        search_source = audit.get("search_source")
    # The goal crosses the line just before 5.5s in the source. Trim the
    # run-up so the human-reviewed event follows it at replay second 4.
    source_start = 1.5
    source = ROOT / "clips/demo-sources/ballard-arsenal.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            str(source_start),
            "-i",
            str(source),
            "-t",
            "20",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(PUBLIC / "ballard-original.mp4"),
        ],
        check=True,
    )
    audio_event = next(e for e in events if e["type"] == "audio_ready")
    start = 4 + audio_event["at_ms"] / 1000
    audio = ROOT / "runs/outreach" / f"{run_id}-analyst.mp3"
    audio.write_bytes(base64.b64decode(audio_event["audio_base64"]))
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(audio),
            ],
            text=True,
        )
    )
    natural_duration = float(probe["format"]["duration"])
    available = 19.9 - start
    speed = max(1, natural_duration / available) if available > 0 else 100
    if speed > 1.2:
        raise ValueError(f"The generated speech cannot fit in 20 seconds: start={start:.2f}s")
    end = start + natural_duration / speed
    output = PUBLIC / "ballard-live-replay.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(PUBLIC / "ballard-original.mp4"),
            "-i",
            str(audio),
            "-filter_complex",
            f"[0:a]volume=0.08:enable='between(t,{start:.6f},{end:.6f})'[match];"
            f"[1:a]atempo={speed:.8f},adelay={round(start * 1000)}:all=1[voice];"
            "[match][voice]amix=inputs=2:normalize=0,atrim=duration=20,"
            "loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            "-map",
            "0:v",
            "-map",
            "[out]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            "20",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    published = [
        {
            "run_id": run_id,
            **{k: v for k, v in e.items() if k not in {"audio_base64", "search_source"}},
        }
        for e in events
    ]
    record = {
        "run_id": run_id,
        "query": QUERY,
        "events": published,
        "speech_start_s": start,
        "speech_end_s": end,
        "speech_speed": speed,
        "origin": (
            "Recorded provider responses and timing; newly generated commentary and voice"
            if search_source
            else "Actual parallel provider requests and subsequent model/voice calls"
        ),
        "search_source": search_source,
        "footage": {"source": source.name, "start_s": source_start, "duration_s": 20},
    }
    (PUBLIC / "live-search-run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_id": run_id,
                "speech_start_s": start,
                "speech_end_s": end,
                "speech_speed": speed,
                "output": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
