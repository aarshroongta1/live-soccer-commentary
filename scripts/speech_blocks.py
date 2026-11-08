"""Synthesize consecutive same-speaker phrases as one performance, then align cuts."""

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
from itertools import groupby
from pathlib import Path
from typing import Any

from render_demo import OfflinePlan, RenderError, resolve_ffmpeg


def aligned_cuts(texts: list[str], alignment: dict[str, Any]) -> list[tuple[float, float]]:
    """Split only between phrases; refuse ambiguous or invalid provider timing."""
    if not texts or any(not text for text in texts):
        raise RenderError("speech alignment needs nonempty phrases")
    text = " ".join(texts)
    chars = alignment.get("characters", [])
    starts = alignment.get("character_start_times_seconds", [])
    ends = alignment.get("character_end_times_seconds", [])
    if "".join(chars) != text or not len(text) == len(starts) == len(ends):
        raise RenderError("speech alignment does not match the exact requested text")
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if (
            not isinstance(start, int | float)
            or not isinstance(end, int | float)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start <= end
            or (index and start < starts[index - 1])
            or (index and end < ends[index - 1])
        ):
            raise RenderError("invalid speech alignment timestamps")
    boundaries = [0.0]
    offset = 0
    for phrase in texts[:-1]:
        offset += len(phrase)
        before, after = ends[offset - 1], starts[offset + 1]
        if after < before:
            raise RenderError("speech phrases overlap in alignment")
        boundaries.append((before + after) / 2)
        offset += 1
    boundaries.append(ends[-1] + 0.08)
    return list(zip(boundaries, boundaries[1:], strict=False))


def synthesize_blocks(
    client: Any,
    plan: OfflinePlan,
    *,
    out_dir: Path,
    api_key: str,
    caller: str,
    analyst: str,
    model: str,
) -> None:
    """One paid request per consecutive voice block; never regenerate on failure."""
    ffmpeg = resolve_ffmpeg(None)
    if shutil.which(ffmpeg) is None:
        raise RenderError("ffmpeg is required before synthesizing continuous speech")
    manifest: dict[str, str] = {}
    audit: dict[str, Any] = {"model": model, "mode": "continuous_voice_blocks", "blocks": []}
    for number, (role, group) in enumerate(groupby(plan.lines, key=lambda line: line.voice), 1):
        lines = list(group)
        speaker = caller if role == "caller" else analyst
        request = {"text": " ".join(line.text for line in lines), "model_id": model}
        response = client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{speaker}/with-timestamps"
            "?output_format=mp3_44100_128",
            headers={"xi-api-key": api_key, "accept": "application/json"},
            json=request,
        )
        response.raise_for_status()
        data = response.json()
        # Save provider output before validating alignment so paid failures are inspectable.
        (out_dir / f"block-{number}.response.json").write_text(json.dumps(data) + "\n")
        block = out_dir / f"block-{number}.mp3"
        block.write_bytes(base64.b64decode(data["audio_base64"], validate=True))
        row = {
            "voice": role,
            "request": request,
            "line_ids": [x.id for x in lines],
            "credit_headers": {
                k: v
                for k, v in response.headers.items()
                if "credit" in k.lower() or "character-cost" in k.lower()
            },
        }
        audit["blocks"].append(row)
        (out_dir / "synthesis-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        cuts = aligned_cuts([line.text for line in lines], data.get("alignment") or {})
        row["cuts"] = cuts
        for line, (start, end) in zip(lines, cuts, strict=True):
            target = out_dir / f"{line.id}.wav"
            subprocess.run(
                [
                    ffmpeg,
                    "-v",
                    "error",
                    "-i",
                    str(block),
                    "-af",
                    f"atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS",
                    "-c:a",
                    "pcm_s16le",
                    str(target),
                ],
                check=True,
                capture_output=True,
            )
            manifest[line.id] = str(target.resolve())
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (out_dir / "synthesis-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(f"saved voice block {number}: {len(lines)} phrases; {row['credit_headers']}")
