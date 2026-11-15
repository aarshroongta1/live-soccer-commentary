#!/usr/bin/env python3
"""Render the browser's recording stage plus continuous goal footage to a 20s MP4.

Uses gstack browse for all browser work and FFmpeg for the final composition.
No search, model, or TTS calls. Save screenshots and timing geometry for review.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000/demo/replay?mode=replay")
    parser.add_argument(
        "--browse", type=Path, default=Path.home() / ".agents/skills/gstack/browse/dist/browse"
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "web/public/soccer-commentary/keenable-outreach.mp4"
    )
    parser.add_argument("--reuse-captures", action="store_true")
    args = parser.parse_args()
    work = ROOT / "runs/outreach/render"
    work.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "web/public/soccer-commentary/live-search-run.json").read_text())
    returns = [e for e in data["events"] if e["type"] == "search_returned"]
    race_end = max(e["at_ms"] for e in returns) / 1000
    handoffs = [
        4 + e["at_ms"] / 1000
        for e in data["events"]
        if e["type"] in {"search_started", "fact_selected", "commentary_ready"}
    ]
    times = sorted(
        {
            0,
            4,
            20,
            *(4 + e["at_ms"] / 1000 for e in data["events"] if 4 + e["at_ms"] / 1000 < 20),
            # Let viewers see elapsed wait grow while Keenable has already returned.
            *(round(4 + i / 10, 3) for i in range(1, int(race_end * 10) + 1)),
            *(float(i) for i in range(1, 20)),
            # Capture the 650ms packet transfer after each real handoff.
            *(
                round(at + step / 10, 4)
                for at in handoffs
                for step in range(8)
                if at + step / 10 < 20
            ),
        }
    )

    def browse(*command: str) -> str:
        return subprocess.check_output([str(args.browse), *command], text=True)

    if not args.reuse_captures:
        browse("viewport", "1920x1080")
        separator = "&" if "?" in args.url else "?"
        browse("goto", f"{args.url}{separator}t=0.001")
    segments = []
    for i, (start, end) in enumerate(zip(times, times[1:], strict=False)):
        target = work / f"scene-{i}.png"
        segments.extend([f"file '{target}'", f"duration {end - start:.6f}"])
        if args.reuse_captures:
            if not target.exists():
                raise ValueError("missing saved scene capture")
            continue
        # Seek the existing player. Its event handler updates the inspector;
        # two animation frames allow React to commit before the screenshot.
        browse(
            "js",
            "new Promise(resolve => { const v=document.querySelector('video'); "
            "const done=()=>requestAnimationFrame(()=>requestAnimationFrame(()=>resolve(true))); "
            f"const target={start + 0.001:.5f}; "
            "if(Math.abs(v.currentTime-target)<0.00001 && v.readyState>=2) done(); "
            "else {v.addEventListener('seeked',done,{once:true});v.currentTime=target;} })",
        )
        browse("screenshot", "--selector", ".outreach-stage", str(target))
        print(f"Saved scene {start:.3f}–{end:.3f}s", flush=True)
    segments.append(f"file '{work / f'scene-{len(times) - 2}.png'}'")
    concat = work / "scenes.txt"
    concat.write_text("\n".join(segments) + "\n")
    geometry = (
        json.loads((work / "render.json").read_text())["geometry"]
        if args.reuse_captures
        else json.loads(
            browse(
                "js",
                "JSON.stringify({stage:document.querySelector('.outreach-stage').getBoundingClientRect()"
                ".toJSON(),video:document.querySelector('video').getBoundingClientRect().toJSON(),"
                "native_controls:document.querySelector('video').controls})",
            )
        )
    )
    stage, video = geometry["stage"], geometry["video"]
    source = ROOT / "web/public/soccer-commentary/ballard-live-replay.mp4"
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                str(source),
            ],
            text=True,
        )
    )["streams"][0]
    aspect = stream["width"] / stream["height"]
    # The dev player uses object-fit: contain. Preserve its letterboxing.
    content_width = min(video["width"], video["height"] * aspect)
    content_height = min(video["height"], video["width"] / aspect)
    x = round(video["x"] - stage["x"] + (video["width"] - content_width) / 2)
    y = round(video["y"] - stage["y"] + (video["height"] - content_height) / 2)
    width, height = round(content_width), round(content_height)
    if geometry.get("native_controls"):
        controls_x = round(video["x"] - stage["x"])
        controls_height = min(80, round(video["height"]))
        controls_y = round(video["y"] - stage["y"] + video["height"] - controls_height)
        # Keep the browser's actual control strip above the continuous footage.
        filters = (
            "[0:v]fps=25,format=yuv420p,split[bg][ui];"
            f"[1:v]scale={width}:{height}[goal];"
            f"[bg][goal]overlay={x}:{y}:shortest=1[composed];"
            f"[ui]crop={round(video['width'])}:{controls_height}:"
            f"{controls_x}:{controls_y}[controls];"
            f"[composed][controls]overlay={controls_x}:{controls_y},"
            "scale=1920:1080,setsar=1[out]"
        )
    else:
        filters = (
            f"[0:v]fps=25,format=yuv420p[bg];[1:v]scale={width}:{height}[goal];"
            f"[bg][goal]overlay={x}:{y}:shortest=1,scale=1920:1080,setsar=1[out]"
        )
    # Background stages switch at the recorded arrival times. The actual goal
    # runs continuously, with the audited prepared narration from the media step.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-i",
            str(source),
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-map",
            "1:a:0",
            "-t",
            "20",
            "-r",
            "25",
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
            str(args.out),
        ],
        check=True,
    )
    (work / "render.json").write_text(
        json.dumps(
            {
                "duration_s": 20,
                "size": [1920, 1080],
                "fps": 25,
                "scene_boundaries_s": times,
                "geometry": geometry,
                "narration_origin": (
                    "Exact commentary from the recorded live run; no editorial rewrite"
                ),
                "note": (
                    "Edited scene captures with continuous source video; not a live system run."
                ),
                "output": str(args.out),
            },
            indent=2,
        )
        + "\n"
    )
    print(args.out)


if __name__ == "__main__":
    main()
