#!/usr/bin/env python3
"""Reuse the existing voice synthesis and audited mux for the outreach recording."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from add_demo_narration import add_narration

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spend", action="store_true")
    args = parser.parse_args()
    data = json.loads((ROOT / "web/data/search-comparison.json").read_text())
    work = ROOT / "runs/outreach/media"
    work.mkdir(parents=True, exist_ok=True)
    brief = {
        "fixture": data["fixture"],
        "as_of": "2025-11-08",
        "squads": [],
        "facts": next(p["facts"] for p in data["providers"] if p["provider"] == "keenable"),
    }
    plan = {
        "version": 1,
        "mode": "recorded_clip",
        "source": {"basename": "ballard-arsenal.mp4", "start_s": 0, "duration_s": 20},
        "observations": [
            {
                "id": "demo-intro",
                "at_s": 0,
                "kind": "play",
                "description": "Editorial introduction to the paired search experiment.",
            },
            {
                "id": "goal-1",
                "at_s": 4,
                "kind": "goal",
                "description": "Human-verified Ballard goal in official highlights.",
            },
        ],
        "lines": [
            {
                "id": "intro",
                "at_s": 0.12,
                "voice": "caller",
                "text": "In live commentary, a two-second search can miss the moment. "
                "I tried Keenable and Exa. Same query. Same agent.",
                "evidence_ids": ["demo-intro"],
                "research_ids": [],
            },
            {
                "id": "context",
                "at_s": 10.5,
                "voice": "analyst",
                "text": data["featured_commentary"],
                "evidence_ids": ["goal-1"],
                "research_ids": [fact["id"] for fact in brief["facts"]],
            },
        ],
        "metadata": {
            "offline": True,
            "cost": data["budget"]["reported_usd"],
            "trace": "web/data/search-comparison.json",
            "models": [data["model"]],
            "research": brief,
            "words_origin": "Editorial narration and broadcast rewrite; not exact LLM output.",
            "source_url": data["source_url"],
        },
    }
    plan_path = work / "voice-plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    if not args.spend:
        print(
            f"Dry run: {sum(len(line['text']) for line in plan['lines'])} TTS characters, "
            "two calls, existing ElevenLabs Flash voices. Pass --spend to synthesize."
        )
        return
    audio = work / "audio"
    if not audio.exists():
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/synthesize_demo.py"),
                "--plan",
                str(plan_path),
                "--out-dir",
                str(audio),
                "--max-characters",
                "500",
                "--spend",
            ],
            check=True,
            cwd=ROOT,
        )
    source = ROOT / "clips/demo-sources/ballard-arsenal.mp4"
    silent = work / "goal-silent.mp4"
    if not silent.exists():
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-t",
                "20",
                "-an",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                str(silent),
            ],
            check=True,
        )
    voiced = work / "goal-voiced.mp4"
    if not voiced.exists():
        add_narration(plan_path, silent, audio / "manifest.json", voiced)
    shutil.copy2(voiced, ROOT / "web/public/soccer-commentary/ballard-agent.mp4")
    print("Saved 20-second goal with audited narration. No original broadcast speech remains.")


if __name__ == "__main__":
    main()
