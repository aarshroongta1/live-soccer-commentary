#!/usr/bin/env python3
"""JSONL stdin/stdout bridge for one bounded local streaming session."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from commentary.llm import openai_factory
from commentary.recorded_demo import Frame, _stamp_frame, encode_frame, safe_pack
from commentary.recorded_research import ResearchBrief
from commentary.streaming_demo import StreamingCommentator

MAX_BATCHES = 12
MAX_SECONDS = 60.0
MAX_JPEG_BYTES = 1_000_000
MAX_COST_USD = 0.35


def _context(name: str) -> tuple[dict[str, Any] | None, ResearchBrief | None]:
    if name == "generic":
        return None, None
    root = Path(__file__).parents[1]
    if name != "argfra":
        raise ValueError("invalid context")
    pack = json.loads((root / "clips/pack-argfra-2022.json").read_text())
    research = json.loads(
        (root / "runs/demo/argfra-all-openai-v1-plan.json.run/research.json").read_text()
    )
    return safe_pack(pack), ResearchBrief.model_validate(research["result"])


def _frames(raw: object, playback_s: float, previous_s: float) -> list[Frame]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 4:
        raise ValueError("frames must contain 1 to 4 JPEGs")
    frames: list[Frame] = []
    last = previous_s
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("jpeg"), str):
            raise ValueError("frame must contain base64 jpeg")
        at_s = item.get("at_s")
        if (
            not isinstance(at_s, (int, float))
            or not math.isfinite(at_s)
            or isinstance(at_s, bool)
            or not 0 <= at_s <= playback_s <= MAX_SECONDS
            or not last < at_s
        ):
            raise ValueError("frame timestamps must be ordered, current, and within 60 seconds")
        if len(item["jpeg"]) > MAX_JPEG_BYTES * 2:
            raise ValueError("JPEG is too large")
        try:
            jpeg = base64.b64decode(item["jpeg"], validate=True)
        except ValueError as exc:
            raise ValueError("invalid JPEG base64") from exc
        if len(jpeg) > MAX_JPEG_BYTES:
            raise ValueError("JPEG is too large")
        if not jpeg.startswith(b"\xff\xd8"):
            raise ValueError("invalid JPEG")
        image = cv2.imdecode(np.frombuffer(jpeg, dtype="uint8"), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid JPEG")
        height, width = image.shape[:2]
        if width > 1280 or height > 1280 or (width > 720 and height > 720):
            raise ValueError("JPEG exceeds 1280x720 landscape or 1280 portrait")
        stamped = encode_frame(_stamp_frame(image, float(at_s)), max_width=1280)
        frames.append(Frame(float(at_s), stamped))
        last = float(at_s)
    return frames


async def main() -> None:
    load_dotenv()
    commentator: StreamingCommentator | None = None
    batches = 0
    last_s = -1.0
    started = time.monotonic()
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            action = request.get("action")
            if action == "start":
                if commentator is not None or request.get("spend") is not True:
                    raise ValueError("start requires spend:true and no active session")
                pack, research = _context(request.get("context"))
                commentator = StreamingCommentator(
                    openai_factory(timeout_s=40.0, max_retries=0),
                    pack=pack,
                    research=research,
                    observer_model="gpt-5.6-terra",
                    writer_model="gpt-5.6-terra",
                )
                batches, last_s = 0, -1.0
                started = time.monotonic()
                result: dict[str, Any] = {"started": True}
            elif action == "batch" and commentator is not None:
                playback_s = request.get("playback_s")
                if (
                    not isinstance(playback_s, (int, float))
                    or isinstance(playback_s, bool)
                    or not math.isfinite(playback_s)
                    or not 0 <= playback_s <= MAX_SECONDS
                    or playback_s > time.monotonic() - started + 1.0
                ):
                    raise ValueError("playback_s must be finite")
                if batches >= MAX_BATCHES or time.monotonic() - started > 90:
                    raise ValueError("batch limit reached")
                frames = _frames(request.get("frames"), float(playback_s), last_s)
                batches += 1
                result = await commentator.process(frames, playback_s=float(playback_s))
                cost = float(result["usage"]["cumulative"]["cost_usd"])
                if cost >= MAX_COST_USD:
                    commentator = None
                    result["stopped_reason"] = "Estimated token-cost limit reached."
                last_s = frames[-1].at_s
            elif action == "stop" and commentator is not None:
                commentator = None
                result = {"stopped": True}
            else:
                raise ValueError("invalid action or inactive session")
            print(json.dumps({"ok": True, "result": result}), flush=True)
        except Exception as exc:
            commentator = None
            print(json.dumps({"ok": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
