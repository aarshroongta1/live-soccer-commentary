"""Prepare a two-voice replay around the exact recorded research line.

One bounded caller-writing call and two TTS calls on first use; reuse saved
artifacts thereafter. Search responses, timings, and the research line stay intact.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from add_demo_narration import add_narration, sha256
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from commentary.llm.base import text_block
from commentary.llm.budget import RunBudget
from commentary.llm.openai_backend import OpenAIBackend

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "web/public/soccer-commentary"
MODEL = "gpt-5.4-mini"


class CallerDraft(BaseModel):
    buildup: str = Field(min_length=1, max_length=100)
    goal: str = Field(min_length=1, max_length=180)
    closing: str = Field(min_length=1, max_length=70)


def synthesize_exchange(plan_path: Path, out: Path) -> None:
    """One continuous performance per voice, then reuse the existing aligned cuts."""
    plan = json.loads(plan_path.read_text())
    characters = sum(len(line["text"]) for line in plan["lines"]) + len(plan["lines"]) - 2
    if characters > 420:
        raise ValueError("Exchange exceeds the 420-character voice budget")
    audio_root = out / "audio"
    audio_root.mkdir(exist_ok=True)
    manifest, rows, blocks = {}, [], []
    for role in ("caller", "analyst"):
        voice_plan = {**plan, "lines": [line for line in plan["lines"] if line["voice"] == role]}
        voice_path = out / f"{role}-plan.json"
        voice_path.write_text(json.dumps(voice_plan, indent=2) + "\n")
        voice_audio = audio_root / role
        if not (voice_audio / "manifest.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/synthesize_demo.py"),
                    "--plan",
                    str(voice_path),
                    "--out-dir",
                    str(voice_audio),
                    "--max-characters",
                    "420",
                    "--continuous",
                    "--spend",
                    "--caller-voice",
                    os.environ["ELEVENLABS_ANALYST_VOICE"],
                    "--analyst-voice",
                    os.environ["ELEVENLABS_CALLER_VOICE"],
                ],
                check=True,
                cwd=ROOT,
            )
        recorded = json.loads((voice_audio / "synthesis-audit.json").read_text())
        manifest.update(json.loads((voice_audio / "manifest.json").read_text()))
        blocks.extend(recorded["blocks"])
        for line in voice_plan["lines"]:
            block = next(b for b in recorded["blocks"] if line["id"] in b["line_ids"])
            rows.append(
                {
                    "id": line["id"],
                    "voice": role,
                    "voice_id": block["voice_id"],
                    "text": line["text"],
                    "credit_headers": {},
                    "block_sha256": block.get("audio_sha256"),
                }
            )
    if set(manifest) != {line["id"] for line in plan["lines"]}:
        raise ValueError("Continuous synthesis did not produce every planned line")
    (audio_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (audio_root / "synthesis-audit.json").write_text(
        json.dumps(
            {
                "model": "eleven_flash_v2_5",
                "mode": "one_performance_per_voice",
                "lines": rows,
                "blocks": blocks,
            },
            indent=2,
        )
        + "\n"
    )


async def prepare(trace: Path, out: Path, spend: bool) -> None:
    audit = json.loads(trace.read_text())
    events = audit["events"]
    context = next(e["text"] for e in events if e["type"] == "commentary_ready")
    fact = next(e["fact"] for e in events if e["type"] == "fact_selected")
    cue = 4 + next(e["at_ms"] for e in events if e["type"] == "audio_ready") / 1000
    plan_path = out / "plan.json"
    manifest_path = out / "audio/manifest.json"
    if not spend and not manifest_path.exists():
        print(
            "Dry run: one GPT-5.4-mini call capped at $0.03, two Flash TTS calls "
            "capped at 420 characters, zero searches. Pass --spend to prepare."
        )
        return
    load_dotenv(ROOT / ".env")
    out.mkdir(parents=True, exist_ok=True)
    if not plan_path.exists():
        budget = RunBudget(max_usd=0.03, max_calls=1)
        backend = OpenAIBackend(
            timeout_s=30,
            max_retries=0,
            budget=budget,
            response_callback=lambda response, tag: (out / f"{tag}.json").write_text(
                response.model_dump_json(indent=2)
            ),
        )
        prompt = {
            "human_reviewed_cues": {
                "buildup": (
                    "Before 4s: Sunderland send a long ball towards the Arsenal penalty area."
                    " Do not name an individual yet."
                ),
                "goal": (
                    "At 4s: Dan Ballard scores for Sunderland against Arsenal, then "
                    "celebrates. No detailed tactics, scoreline, or footedness is supplied."
                ),
                "closing": (
                    "After the contextual line: acknowledge the moment for Sunderland without"
                    " announcing the goal again or adding another fact."
                ),
            },
            "reserved_other_voice_line": context,
            "words_available_only_to_other_voice": (
                "Academy background and the clean-sheet streak must not appear in your "
                "buildup or goal reaction."
            ),
            "slots": {
                "buildup": "7–9 words, 0.15–3.9s",
                "goal": "12–15 words, 4.05–9.7s",
                "closing": "4–6 words, 18–19.9s",
            },
        }
        (out / "caller-input.json").write_text(json.dumps(prompt, indent=2) + "\n")
        result = await backend.parse(
            model=MODEL,
            system=(
                "Write three connected play-by-play phrases around the reserved second-voice line. "
                "Sound like a live football commentator: anticipation, a lively goal reaction, "
                "then a short closing reaction. Use only each slot's human-reviewed cues. "
                "Let the other voice add the background story. Do not repeat its facts or "
                "announce the same goal twice. No provider names, workflow narration, headings, "
                "or invented tactical explanations. Return exact spoken words in the three fields."
            ),
            blocks=[text_block(json.dumps(prompt))],
            output_format=CallerDraft,
            max_tokens=512,
            tag="caller-draft",
        )
        if any(
            word in (result.value.buildup + " " + result.value.goal).lower()
            for word in ("academy", "clean-sheet", "clean sheet", "eight-game", "keenable", "exa")
        ):
            raise ValueError("Early play-by-play leaked background research")
        rows = [
            ("buildup", 0.15, "caller", result.value.buildup, ["buildup"], []),
            ("goal", 4.05, "caller", result.value.goal, ["goal"], []),
            ("context", cue, "analyst", context, ["goal"], [fact["id"]]),
            ("closing", 18.0, "caller", result.value.closing, ["goal"], []),
        ]
        if sum(len(row[3]) for row in rows) > 420:
            raise ValueError("Exchange exceeds the 420-character voice budget")
        plan = {
            "version": 1,
            "mode": "recorded_clip",
            "source": {"basename": "ballard-arsenal.mp4", "start_s": 1.5, "duration_s": 20},
            "observations": [
                {
                    "id": "buildup",
                    "at_s": 0,
                    "kind": "play",
                    "description": prompt["human_reviewed_cues"]["buildup"],
                },
                {
                    "id": "goal",
                    "at_s": 4,
                    "kind": "goal",
                    "description": prompt["human_reviewed_cues"]["goal"],
                },
            ],
            "lines": [
                {
                    "id": i,
                    "at_s": at,
                    "voice": voice,
                    "text": text,
                    "evidence_ids": evidence,
                    "research_ids": research,
                }
                for i, at, voice, text, evidence, research in rows
            ],
            "metadata": {
                "offline": True,
                "trace": str(trace.relative_to(ROOT)),
                "cost": backend.total.cost_usd,
                "models": {"caller": MODEL, "research_writer": MODEL},
                "research": {
                    "fixture": "Sunderland v Arsenal",
                    "as_of": "2025-11-08",
                    "facts": [
                        {
                            "id": fact["id"],
                            "subject": "Dan Ballard",
                            "text": fact["text"],
                            "source_url": fact["url"],
                            "source_date": "2025-11-08",
                        }
                    ],
                },
                "origin": (
                    "New caller phrases around the unchanged recorded research line; "
                    "human-reviewed visual cues"
                ),
            },
        }
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        (out / "budget.json").write_text(json.dumps(budget.report(), indent=2) + "\n")
    if not manifest_path.exists():
        synthesize_exchange(plan_path, out)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_live_search_replay.py"),
            "--trace",
            str(trace),
        ],
        check=True,
        cwd=ROOT,
    )
    output = out / "exchange.mp4"
    report_path = output.with_suffix(".narration.json")
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else add_narration(plan_path, PUBLIC / "ballard-original.mp4", manifest_path, output)
    )
    if report["plan_sha256"] != sha256(plan_path):
        raise ValueError("Saved audio arrangement does not match the current plan")
    context_row = next(row for row in report["lines"] if row["id"] == "context")
    if context_row["text"] != context:
        raise ValueError("The research line must remain identical to the recorded model output")
    record_path = PUBLIC / "live-search-run.json"
    record = json.loads(record_path.read_text())
    record.update(
        {
            "delivery": report["lines"],
            "speech_start_s": context_row["at_s"],
            "speech_end_s": context_row["end_s"],
            "speech_speed": context_row["speed"],
            "delivery_origin": (
                "Prepared two-voice exchange around the exact recorded research line; "
                "voice casting swapped at user request"
            ),
        }
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    shutil.copy2(output, PUBLIC / "ballard-live-replay.mp4")
    print(
        json.dumps(
            {"delivery": report["lines"], "synthesis_model": report["synthesis_model"]}, indent=2
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spend", action="store_true")
    args = parser.parse_args()
    asyncio.run(prepare(args.trace.resolve(), args.out_dir.resolve(), args.spend))
