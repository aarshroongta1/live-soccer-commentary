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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = read_plan(args.plan)
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
    audit: dict[str, object] = {"model": MODEL, "lines": []}
    manifest_path = args.out_dir / "manifest.json"
    audit_path = args.out_dir / "synthesis-audit.json"
    with httpx.Client(timeout=60) as client:
        for line in plan.lines:
            if Path(line.id).name != line.id:
                raise RenderError(f"line ID cannot contain a path: {line.id!r}")
            speaker = analyst if line.voice == "analyst" else caller
            response = client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{speaker}?output_format=mp3_44100_128",
                headers={"xi-api-key": api_key, "accept": "audio/mpeg"},
                json={"text": line.text, "model_id": MODEL, "voice_settings": {"speed": 1.1}},
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
