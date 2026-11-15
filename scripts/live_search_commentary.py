#!/usr/bin/env python3
"""NDJSON worker: live or recorded searches, two bounded model calls, one voice call."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from dotenv import load_dotenv

from commentary.live_research import research_events
from commentary.llm.budget import RunBudget
from commentary.llm.openai_backend import OpenAIBackend
from commentary.search_providers import SearchRun, search_web

ROOT = Path(__file__).resolve().parents[1]


async def main(spend: bool, replay_searches: Path | None = None) -> None:
    if not spend:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "search_calls": 0 if replay_searches else 2,
                    "model_calls": 2,
                    "model_max_usd": 0.05,
                    "tts_calls": 1,
                    "tts_max_chars": 240,
                }
            )
        )
        return
    search_source = None
    recorded = {}
    if replay_searches:
        previous = json.loads(replay_searches.read_text())
        recorded = {e["provider"]: e for e in previous["events"] if e["type"] == "search_returned"}
        if set(recorded) != {"keenable", "exa"}:
            raise ValueError("Replay requires both recorded provider responses")
        search_source = {"mode": "recorded", "run_id": previous["run_id"]}
    load_dotenv(ROOT / ".env")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:6]
    audit = ROOT / "runs/outreach/live" / run_id
    audit.mkdir(parents=True)
    budget = RunBudget(max_usd=0.05, max_calls=2)

    def capture(response, tag):
        (audit / f"{tag}.json").write_text(response.model_dump_json(indent=2))

    backend = OpenAIBackend(timeout_s=30, max_retries=0, budget=budget, response_callback=capture)

    async def synthesize(text: str) -> dict:
        if len(text) > 240:
            raise ValueError("Voice input exceeded 240 characters")
        voice = os.environ.get("ELEVENLABS_ANALYST_VOICE", "Xb7hH8MSUJpSbSDYk0k2")
        async with httpx.AsyncClient(timeout=25) as client:
            response = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
                headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                json={"text": text, "model_id": "eleven_flash_v2_5"},
            )
        if not response.is_success:
            raise RuntimeError(f"Voice returned HTTP {response.status_code}")
        (audit / "analyst.mp3").write_bytes(response.content)
        return {
            "audio_base64": base64.b64encode(response.content).decode(),
            "character_credits": response.headers.get("character-cost"),
            "text": text,
        }

    packets = []
    try:
        async with asyncio.timeout(65), httpx.AsyncClient(timeout=30) as client:
            replay_started = time.monotonic()

            async def search(provider, query):
                if recorded:
                    packet = recorded[provider]
                    result = SearchRun.model_validate(packet["search"])
                    if result.query != query:
                        raise ValueError("Recorded query differs from the current query")
                    await asyncio.sleep(
                        max(0, packet["at_ms"] / 1000 - (time.monotonic() - replay_started))
                    )
                    return result
                return await search_web(provider, query, client=client)

            async for packet in research_events(backend, search=search, synthesize=synthesize):
                packets.append(packet)
                metadata = {"search_source": search_source} if search_source else {}
                print(json.dumps({"run_id": run_id, **metadata, **packet}), flush=True)
    finally:
        (audit / "trace.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "events": packets,
                    "budget": budget.report(),
                    "search_source": search_source,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spend", action="store_true")
    parser.add_argument(
        "--replay-searches",
        type=Path,
        help="Reuse recorded responses and arrival offsets; make no search API calls",
    )
    args = parser.parse_args()
    asyncio.run(main(args.spend, args.replay_searches))
