#!/usr/bin/env python3
"""Optionally synthesize an offline plan into an ElevenLabs MP3 manifest.

Dry-run is the default. Use --spend to authorize paid requests. Audio is saved
only; this command never plays it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import dotenv
from render_demo import RenderError, read_plan
from speech_blocks import synthesize_blocks

MODEL = "eleven_flash_v2_5"
CALLER_VOICE = "JBFqnCBsd6RMkjVDRZzb"
ANALYST_VOICE = "Xb7hH8MSUJpSbSDYk0k2"


def voice(role: str, default: str) -> str:
    names = [f"ELEVENLABS_{role}_VOICE"]
    names.append("ELEVENLABS_LEAD_VOICE" if role == "CALLER" else "ELEVENLABS_SUPPORTING_VOICE")
    return next((os.environ[name] for name in names if os.environ.get(name)), default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--spend", action="store_true")
    parser.add_argument("--caller-voice")
    parser.add_argument("--analyst-voice")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-characters", type=int, default=5000)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="generate consecutive same-voice phrases in one performance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = read_plan(args.plan)
    if any(Path(line.id).name != line.id for line in plan.lines):
        raise RenderError("line IDs cannot contain a path")
    characters = sum(len(line.text) for line in plan.lines)
    if args.continuous:
        characters += sum(
            a.voice == b.voice for a, b in zip(plan.lines, plan.lines[1:], strict=False)
        )
    if args.max_characters < 1 or characters > args.max_characters:
        raise RenderError("plan exceeds synthesis character limit")
    if args.spend:
        dotenv.load_dotenv()
    caller = args.caller_voice or voice("CALLER", CALLER_VOICE)
    analyst = args.analyst_voice or voice("ANALYST", ANALYST_VOICE)
    if caller == analyst:
        raise RenderError("Choose two distinct ElevenLabs voices")
    if not args.spend:
        print(f"Dry run: {len(plan.lines)} exact plan lines; pass --spend to synthesize.")
        return 0
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        raise RenderError("ELEVENLABS_API_KEY is required with --spend")
    if args.out_dir.exists():
        raise RenderError(f"Refusing to overwrite existing output directory: {args.out_dir}")
    try:
        import httpx
    except ImportError as exc:
        raise RenderError("Install httpx to synthesize") from exc
    args.out_dir.mkdir(parents=True)
    manifest: dict[str, str] = {}
    audit: dict[str, object] = {
        "model": args.model,
        "settings_source": "saved ElevenLabs voice settings; no request overrides",
        "lines": [],
    }
    manifest_path = args.out_dir / "manifest.json"
    audit_path = args.out_dir / "synthesis-audit.json"
    with httpx.Client(timeout=60) as client:
        if args.continuous:
            synthesize_blocks(
                client,
                plan,
                out_dir=args.out_dir,
                api_key=api_key,
                caller=caller,
                analyst=analyst,
                model=args.model,
            )
            return 0
        for index, line in enumerate(plan.lines):
            if Path(line.id).name != line.id:
                raise RenderError(f"line ID cannot contain a path: {line.id!r}")
            speaker = analyst if line.voice == "analyst" else caller
            request = {"text": line.text, "model_id": args.model}
            # Stitch the performance of each voice without feeding the other
            # commentator's words into that voice's acoustic context.
            previous = [item.text for item in plan.lines[:index] if item.voice == line.voice]
            following = [item.text for item in plan.lines[index + 1 :] if item.voice == line.voice]
            if previous:
                request["previous_text"] = " ".join(previous[-2:])
            if following:
                request["next_text"] = " ".join(following[:2])
            response = client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{speaker}?output_format=mp3_44100_128",
                headers={"xi-api-key": api_key, "accept": "audio/mpeg"},
                json=request,
            )
            response.raise_for_status()
            target = args.out_dir / f"{line.id}.mp3"
            target.write_bytes(response.content)
            manifest[line.id] = str(target.resolve())
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            credit_headers = {
                name: value
                for name, value in response.headers.items()
                if "credit" in name.lower() or "character-cost" in name.lower()
            }
            audit["lines"].append(
                {
                    "id": line.id,
                    "voice": line.voice,
                    "voice_id": speaker,
                    "text": line.text,
                    "request": request,
                    "credit_headers": credit_headers,
                }
            )
            audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
            credit_note = f"; provider credit headers: {credit_headers}" if credit_headers else ""
            print(f"saved {target.name}{credit_note}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RenderError as exc:
        print(f"synthesize_demo: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
