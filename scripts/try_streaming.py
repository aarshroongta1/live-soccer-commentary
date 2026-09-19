#!/usr/bin/env python3
"""Measure online commentary against a recorded clip played at wall-clock speed.

Only frames at or before the current playback position reach the model. One
batch runs at a time; if it is slow, the next batch samples the latest window
instead of replaying a growing queue. Skipped coverage is reported explicitly.
No paid requests without --spend. No speech generation or audio playback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv

from commentary.llm import openai_factory
from commentary.llm.base import encode_frame
from commentary.llm.pricing import PRICES
from commentary.recorded_demo import Frame, _stamp_frame, safe_pack, validate_inputs
from commentary.recorded_research import ResearchBrief
from commentary.streaming_demo import StreamingCommentator


def read_frames(capture: Any, *, source_start: float, begin: float, end: float) -> list[Frame]:
    """Eight samples from the elapsed window, never from beyond its end."""
    frames = []
    for index in range(8):
        at_s = round(begin + (end - begin) * index / 8, 2)
        capture.set(cv2.CAP_PROP_POS_MSEC, (source_start + at_s) * 1000)
        ok, frame = capture.read()
        if ok:
            frames.append(Frame(at_s, encode_frame(_stamp_frame(frame, at_s), max_width=1280)))
    return frames


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--research", type=Path)
    parser.add_argument("--start", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--interval", type=float, default=4.0)
    parser.add_argument("--observer-model", default="gpt-5.6-terra")
    parser.add_argument("--writer-model", default="gpt-5.6-terra")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--spend", action="store_true")
    return parser.parse_args()


async def trial(args: argparse.Namespace) -> None:
    validate_inputs(args.clip, args.start, args.duration)
    if not 0 < args.duration <= 40 or not 4 <= args.interval <= 10:
        raise ValueError("This bounded trial supports at most 40 seconds and 4–10s batches")
    if any(model not in PRICES for model in (args.observer_model, args.writer_model)):
        raise ValueError("Use a model with configured pricing so the trial records estimated cost")
    summary_path = args.out.with_suffix(".summary.json")
    if args.out.exists() or summary_path.exists():
        raise ValueError("Refusing to overwrite an existing trial")
    if not args.spend:
        print(
            json.dumps(
                {
                    "paid_calls": False,
                    "duration_s": args.duration,
                    "interval_s": args.interval,
                    "max_batches": math.ceil(args.duration / args.interval),
                    "observer_model": args.observer_model,
                    "writer_model": args.writer_model,
                },
                indent=2,
            )
        )
        return
    pack = safe_pack(json.loads(args.pack.read_text())) if args.pack else None
    research = None
    if args.research:
        raw = json.loads(args.research.read_text())
        research = ResearchBrief.model_validate(raw.get("result", raw))
    load_dotenv()
    backend = openai_factory(timeout_s=30, max_retries=0)
    commentator = StreamingCommentator(
        backend,
        pack=pack,
        research=research,
        observer_model=args.observer_model,
        writer_model=args.writer_model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.clip))
    started = time.monotonic()
    previous_end = 0.0
    skipped_s = 0.0
    rows: list[dict[str, Any]] = []
    try:
        with args.out.open("x") as log:
            for batch in range(math.ceil(args.duration / args.interval)):
                due = min(args.duration, previous_end + args.interval)
                await asyncio.sleep(max(0.0, due - (time.monotonic() - started)))
                now = time.monotonic() - started
                end = min(now, args.duration)
                begin = max(previous_end, end - args.interval)
                skipped_s += max(0.0, begin - previous_end)
                if end <= begin:
                    break
                frames = read_frames(capture, source_start=args.start, begin=begin, end=end)
                request_started = time.monotonic()
                row: dict[str, Any] = {
                    "batch": batch + 1,
                    "window_start_s": begin,
                    "window_end_s": end,
                    "frame_times": [frame.at_s for frame in frames],
                    "sent_at_s": request_started - started,
                }
                try:
                    row["result"] = await commentator.process(frames, playback_s=end)
                except Exception as exc:
                    row["error"] = str(exc)
                    row["observer_on_failure"] = getattr(commentator, "last_observer", None)
                row["returned_at_s"] = time.monotonic() - started
                row["latency_s"] = time.monotonic() - request_started
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                line = row.get("result", {}).get("line")
                if line:
                    print(
                        f"[{row['returned_at_s']:5.1f}s | footage {end:4.1f}s | "
                        f"{row['latency_s']:.1f}s processing] {line['voice']}: {line['text']}",
                        flush=True,
                    )
                else:
                    print(f"[batch {batch + 1}] {row.get('error', 'No grounded line')}", flush=True)
                previous_end = end
                if end >= args.duration or row.get("error") or backend.total.cost_usd >= 1.0:
                    break
    finally:
        capture.release()
        latencies = [row["latency_s"] for row in rows if not row.get("error")]
        summary = {
            "mode": "wall_clock_online_trial",
            "source": {"clip": str(args.clip), "start_s": args.start, "duration_s": args.duration},
            "models": {"observer": args.observer_model, "writer": args.writer_model},
            "interval_s": args.interval,
            "batches": len(rows),
            "successful_batches": len(latencies),
            "lines": sum(bool(row.get("result", {}).get("line")) for row in rows),
            "errors": [row["error"] for row in rows if row.get("error")],
            "median_processing_s": statistics.median(latencies) if latencies else None,
            "max_processing_s": max(latencies, default=0),
            "batches_within_interval": sum(t <= args.interval for t in latencies),
            "skipped_footage_s": skipped_s,
            "unprocessed_tail_s": max(0.0, args.duration - previous_end),
            "elapsed_s": time.monotonic() - started,
            "usage": asdict(backend.total),
            "note": (
                "No future frames or precomputed commentary. Cost is a configured token estimate."
            ),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(trial(arguments()))
