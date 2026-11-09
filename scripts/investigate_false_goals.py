#!/usr/bin/env python3
"""Bounded observer ablations on existing footage; no labels or target prose.

Default is a free input audit. Paid trials require --spend. Changes are restricted
 to framing, temporal density or explicitly supplied diagnostic magnifications.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from commentary.llm.base import encode_frame, image_block, text_block
from commentary.llm.budget import RunBudget
from commentary.recorded_demo import Frame, sample_clip
from commentary.replay.recorded import observe_recorded


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def crop_blocks(frames: list[Frame], box: list[float], times: list[float]) -> list[dict[str, Any]]:
    left, top, right, bottom = box
    if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
        raise ValueError("crop box must be inside normalized image coordinates")
    selected = {f.at_s: f for f in frames}
    if not set(times) <= selected.keys():
        raise ValueError("crop timestamps must exactly match supplied frames")
    result = []
    for at_s in times:
        image = cv2.imdecode(np.frombuffer(selected[at_s].jpeg, np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        cropped = image[int(top * h) : int(bottom * h), int(left * w) : int(right * w)]
        # Magnification changes attention, not source information. Keep original
        # images as context and identify each crop with its original timestamp.
        cropped = cv2.resize(cropped, (768, round(cropped.shape[0] * 768 / cropped.shape[1])))
        result.extend(
            [
                text_block(f"Diagnostic magnification of existing source frame at_s={at_s}"),
                image_block(encode_frame(cropped, quality=92, max_width=768)),
            ]
        )
    return result


def transformed(system, blocks, condition, frames, box, times):
    if condition == "neutral_word":
        phrase = "This is one recorded highlights sequence, prepared before playback."
        if system.count(phrase) != 1:
            raise ValueError("baseline framing phrase changed; cannot isolate substitution")
        system = system.replace(
            phrase, "This is one recorded football sequence, prepared before playback."
        )
    if condition == "magnified":
        blocks = [*blocks, *crop_blocks(frames, box, times)]
    return system, blocks


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


async def run(args):
    if not 0 < args.max_usd <= 1 or not 1 <= args.repeats <= 3:
        raise ValueError("budget must be at most $1 and repeats 1–3")
    frames = {
        1: sample_clip(args.clip, args.start, args.duration, fps=1),
        2: sample_clip(args.clip, args.start, args.duration, fps=2),
    }
    if len(frames[2]) > 30:
        raise ValueError("recorded observer supports at most 30 frames")
    conditions = ["baseline", "neutral_word", "dense"]
    if args.crop_box:
        crop_blocks(frames[1], args.crop_box, args.crop_times)
        conditions.append("magnified")
    budget = RunBudget(max_usd=args.max_usd, max_calls=len(conditions) * args.repeats)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Controlled observer diagnostics on one existing development window; "
        "not a quality benchmark.",
        "status": "running",
        "paid": args.spend,
        "model": args.model,
        "max_usd": args.max_usd,
        "source_sha256": sha(args.clip.read_bytes()),
        "source_start_s": args.start,
        "duration_s": args.duration,
        "conditions": conditions,
        "repeats": args.repeats,
        "frames": {
            str(fps): [{"at_s": f.at_s, "sha256": sha(f.jpeg)} for f in rows]
            for fps, rows in frames.items()
        },
        "crop_box": args.crop_box,
        "crop_times": args.crop_times,
        "trials": [],
        "references_in_runtime": False,
        "tts": False,
    }
    save(args.out_dir / "report.json", report)
    if not args.spend:
        report.update(status="dry_run", budget=budget.report())
        save(args.out_dir / "report.json", report)
        return report
    import openai
    from dotenv import load_dotenv

    from commentary.llm.openai_backend import OpenAIBackend

    load_dotenv()
    client = openai.AsyncOpenAI(timeout=45, max_retries=0)
    try:
        # Interleave repeats rather than exhaust one condition before the next.
        for repeat in range(args.repeats):
            for condition in conditions:
                selected = frames[2 if condition == "dense" else 1]
                trial_dir = args.out_dir / f"{condition}-{repeat + 1}"
                trial_dir.mkdir()
                row = {"condition": condition, "repeat": repeat + 1, "status": "running"}
                report["trials"].append(row)
                captures = []

                def capture(response, tag, target=trial_dir, log=captures):
                    log.append({"tag": tag, "response": response.model_dump(mode="json")})
                    save(target / "providers.json", log)

                class DiagnosticBackend(OpenAIBackend):
                    async def parse(
                        self,
                        condition_name=condition,
                        selected_frames=selected,
                        target=trial_dir,
                        **kwargs,
                    ):
                        system, blocks = transformed(
                            kwargs["system"],
                            kwargs["blocks"],
                            condition_name,
                            selected_frames,
                            args.crop_box,
                            args.crop_times,
                        )
                        kwargs.update(system=system, blocks=blocks)
                        save(
                            target / "request-audit.json",
                            {
                                "system": system,
                                "system_sha256": sha(system.encode()),
                                "blocks_sha256": sha(json.dumps(blocks, sort_keys=True).encode()),
                                "image_count": sum(b.get("type") == "image" for b in blocks),
                                "max_tokens": kwargs["max_tokens"],
                                "effort": "none",
                            },
                        )
                        return await super().parse(**kwargs)

                backend = DiagnosticBackend(
                    client=client, budget=budget, default_effort="none", response_callback=capture
                )
                try:
                    observed = await observe_recorded(
                        backend,
                        selected,
                        model=args.model,
                        identity_model=args.model,
                        research=None,
                        pack=None,
                    )
                    save(trial_dir / "observations.json", observed)
                    row.update(
                        status="complete",
                        observations=observed["observations"],
                        goal_count=sum(o["kind"] == "goal" for o in observed["observations"]),
                        usage=observed["vision_usage"],
                    )
                except Exception as exc:
                    row.update(
                        status="failed", error={"type": type(exc).__name__, "message": str(exc)}
                    )
                row["budget_after"] = budget.report()
                save(args.out_dir / "report.json", report)
        report["status"] = (
            "completed_with_failures"
            if any(r["status"] == "failed" for r in report["trials"])
            else "complete"
        )
        return report
    finally:
        report["budget"] = budget.report()
        if report["status"] == "running":
            report["status"] = "interrupted"
        save(args.out_dir / "report.json", report)
        await client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clip", type=Path, required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.4-mini")
    p.add_argument("--max-usd", type=float, default=0.25)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--crop-box", type=float, nargs=4)
    p.add_argument("--crop-times", type=float, nargs="*", default=[])
    p.add_argument("--spend", action="store_true")
    result = asyncio.run(run(p.parse_args()))
    print(result["status"], result["budget"])


if __name__ == "__main__":
    main()
