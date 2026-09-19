#!/usr/bin/env python3
"""Render a recorded-clip plan with two voices and captions, without paid calls.

Use --plan and --clip, then --out for an MP4 or --dry-run for a timing report.
Audio comes from macOS say or a supplied line-ID-to-audio --manifest.
The source window is stored in the plan. Rendering never plays audio.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from commentary.recorded_research import ResearchBrief

CALLER_VOICE = "Daniel"
ANALYST_VOICE = "Samantha"
SAY_RATE_WPM = 190
ESTIMATED_WORDS_PER_SECOND = 3.2


class RenderError(RuntimeError):
    """A plan or local rendering tool cannot make a valid demo."""


@dataclass(frozen=True)
class RenderLine:
    """One script line, retaining its identity and evidence for rendering."""

    order: int
    id: str
    voice: str
    text: str
    at_s: float
    evidence_ids: tuple[str, ...] = ()
    research_ids: tuple[str, ...] = ()
    goal_evidence: bool = False


@dataclass(frozen=True)
class OfflinePlan:
    """A precomputed, offline commentary plan for one clip window."""

    path: Path
    source_basename: str
    source_start_s: float
    duration_s: float
    lines: list[RenderLine]
    metadata: Mapping[str, Any]
    research: ResearchBrief | None = None


@dataclass(frozen=True)
class PlannedLine:
    """One utterance positioned on the rendered video timeline."""

    line: RenderLine
    natural_start_s: float
    start_s: float
    end_s: float
    duration_s: float
    delayed_s: float
    truncated_s: float
    trimmed_for_goal_s: float = 0.0


@dataclass(frozen=True)
class Timeline:
    """Planned audio plus rows that cannot be heard in the chosen window."""

    planned: list[PlannedLine]
    omitted_for_goal: list[RenderLine]


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RenderError(f"Plan {label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise RenderError(f"Plan {label} must be a finite number")
    return number


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RenderError(f"Plan {label} must be a non-empty string")
    return value.strip()


def read_plan(path: Path) -> OfflinePlan:
    """Load and validate the versioned offline recorded-clip plan."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RenderError(f"Plan does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"Plan is not valid JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RenderError("Plan must be a JSON object")
    if raw.get("version") != 1:
        raise RenderError("Plan version must be 1")
    if raw.get("mode") != "recorded_clip":
        raise RenderError("Plan mode must be 'recorded_clip'")

    source = raw.get("source")
    if not isinstance(source, dict):
        raise RenderError("Plan source must be an object")
    source_basename = _nonempty_string(source.get("basename"), label="source.basename")
    source_start_s = _finite_number(source.get("start_s"), label="source.start_s")
    duration_s = _finite_number(source.get("duration_s"), label="source.duration_s")
    if source_start_s < 0:
        raise RenderError("Plan source.start_s must be non-negative")
    if duration_s <= 0:
        raise RenderError("Plan source.duration_s must be greater than zero")

    observations = raw.get("observations")
    if not isinstance(observations, list):
        raise RenderError("Plan observations must be an array")
    observation_ids: set[str] = set()
    observation_times: dict[str, float] = {}
    observation_kinds: dict[str, str] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise RenderError(f"Plan observation {index} must be an object")
        observation_id = _nonempty_string(observation.get("id"), label=f"observations[{index}].id")
        if observation_id in observation_ids:
            raise RenderError(f"Plan observation ID is duplicated: {observation_id}")
        observation_ids.add(observation_id)
        at_s = _finite_number(observation.get("at_s"), label=f"observations[{index}].at_s")
        if not 0 <= at_s <= duration_s:
            raise RenderError(f"Plan observations[{index}].at_s is outside the clip window")
        observation_kinds[observation_id] = _nonempty_string(
            observation.get("kind"), label=f"observations[{index}].kind"
        )
        _nonempty_string(observation.get("description"), label=f"observations[{index}].description")
        observation_times[observation_id] = at_s

    metadata = raw.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("offline") is not True:
        raise RenderError("Plan metadata.offline must be true")
    _nonempty_string(metadata.get("trace"), label="metadata.trace")
    cost = _finite_number(metadata.get("cost"), label="metadata.cost")
    if cost < 0:
        raise RenderError("Plan metadata.cost must be non-negative")
    if "models" not in metadata:
        raise RenderError("Plan metadata.models is required")
    research: ResearchBrief | None = None
    if metadata.get("research") is not None:
        try:
            research = ResearchBrief.model_validate(metadata["research"])
        except ValueError as exc:
            raise RenderError(f"Plan metadata.research is invalid: {exc}") from exc
    research_facts = {fact.id: fact for fact in research.facts} if research else {}

    raw_lines = raw.get("lines")
    if not isinstance(raw_lines, list) or not raw_lines:
        raise RenderError("Plan lines must be a non-empty array")
    lines: list[RenderLine] = []
    line_ids: set[str] = set()
    previous_at_s = -math.inf
    for index, line in enumerate(raw_lines):
        if not isinstance(line, dict):
            raise RenderError(f"Plan line {index} must be an object")
        line_id = _nonempty_string(line.get("id"), label=f"lines[{index}].id")
        if line_id in line_ids:
            raise RenderError(f"Plan line ID is duplicated: {line_id}")
        line_ids.add(line_id)
        at_s = _finite_number(line.get("at_s"), label=f"lines[{index}].at_s")
        if not 0 <= at_s < duration_s:
            raise RenderError(f"Plan lines[{index}].at_s is outside the clip window")
        if at_s < previous_at_s:
            raise RenderError("Plan lines must be chronological")
        previous_at_s = at_s
        voice = line.get("voice")
        if voice not in {"caller", "analyst"}:
            raise RenderError(f"Plan lines[{index}].voice must be 'caller' or 'analyst'")
        text = _nonempty_string(line.get("text"), label=f"lines[{index}].text")
        evidence = line.get("evidence_ids")
        if not isinstance(evidence, list) or not all(
            isinstance(item, str) and item.strip() for item in evidence
        ):
            raise RenderError(f"Plan lines[{index}].evidence_ids must be an array of strings")
        for evidence_id in evidence:
            if evidence_id not in observation_ids:
                raise RenderError(f"Plan line {line_id} cites unknown evidence: {evidence_id}")
            if observation_times[evidence_id] > at_s:
                raise RenderError(f"Plan line {line_id} cites future evidence: {evidence_id}")
        raw_research_ids = line.get("research_ids", [])
        if not isinstance(raw_research_ids, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_research_ids
        ):
            raise RenderError(f"Plan lines[{index}].research_ids must be an array of strings")
        research_ids = tuple(item.strip() for item in raw_research_ids)
        if voice == "caller" and research_ids:
            raise RenderError(f"Plan caller line {line_id} cannot cite research")
        for research_id in research_ids:
            if research_id not in research_facts:
                raise RenderError(f"Plan line {line_id} cites unknown research: {research_id}")
        if voice == "caller" and not evidence:
            raise RenderError(f"Plan caller line {line_id} requires visual evidence")
        if not evidence and not research_ids:
            raise RenderError(f"Plan line {line_id} needs visual evidence or research")
        lines.append(
            RenderLine(
                order=index,
                id=line_id,
                voice=voice,
                text=text,
                at_s=at_s,
                evidence_ids=tuple(item.strip() for item in evidence),
                research_ids=research_ids,
                goal_evidence=voice == "caller"
                and any(observation_kinds[item] == "goal" for item in evidence),
            )
        )
    return OfflinePlan(
        path=path,
        source_basename=source_basename,
        source_start_s=source_start_s,
        duration_s=duration_s,
        lines=lines,
        metadata=metadata,
        research=research,
    )


def estimate_duration(text: str) -> float:
    """A conservative dry-run duration; final scheduling probes rendered audio."""
    return max(0.35, len(text.split()) / ESTIMATED_WORDS_PER_SECOND)


def plan_timeline(
    lines: Sequence[RenderLine],
    *,
    duration_s: float,
    audio_seconds: Mapping[str, float] | None = None,
) -> Timeline:
    """Place lines sequentially, reserving their planned cue for offline goal calls."""
    if duration_s <= 0:
        raise RenderError("--duration must be greater than zero")
    audio_seconds = audio_seconds or {}
    planned: list[PlannedLine] = []
    omitted_for_goal: list[RenderLine] = []
    cursor = 0.0
    for line in lines:
        natural = line.at_s
        line_duration = audio_seconds.get(line.id, estimate_duration(line.text))
        if line_duration <= 0:
            raise RenderError(f"Audio for {line.id!r} has no measurable duration")
        actual = max(natural, cursor)
        if line.goal_evidence and actual > natural:
            # Reserve the goal cue even when preceding speech runs long.
            while planned and planned[-1].end_s > natural:
                previous = planned[-1]
                if previous.start_s >= natural:
                    omitted_for_goal.append(previous.line)
                    planned.pop()
                    continue
                trimmed = previous.end_s - natural
                planned[-1] = replace(
                    previous,
                    end_s=natural,
                    duration_s=previous.duration_s - trimmed,
                    truncated_s=previous.truncated_s + trimmed,
                    trimmed_for_goal_s=previous.trimmed_for_goal_s + trimmed,
                )
                break
            actual = natural
        end = actual + line_duration
        planned.append(
            PlannedLine(
                line=line,
                natural_start_s=natural,
                start_s=actual,
                end_s=end,
                duration_s=line_duration,
                delayed_s=actual - natural,
                truncated_s=max(0.0, end - duration_s),
            )
        )
        cursor = end
    return Timeline(
        planned=planned,
        omitted_for_goal=omitted_for_goal,
    )


def audible_role_seconds(timeline: Timeline, *, duration_s: float) -> dict[str, float]:
    """Measured audible speech by role, after goal and clip-end trimming."""
    totals = {"caller": 0.0, "analyst": 0.0}
    for item in timeline.planned:
        audible = max(0.0, min(item.end_s, duration_s) - max(item.start_s, 0.0))
        totals[item.line.voice] += audible
    return totals


def probe_duration(path: Path, *, ffprobe: str) -> float:
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RenderError(f"ffprobe could not read {path}: {result.stderr.strip()}")
    try:
        seconds = float(result.stdout.strip())
    except ValueError as exc:
        raise RenderError(f"ffprobe returned no duration for {path}") from exc
    if seconds <= 0:
        raise RenderError(f"ffprobe returned a non-positive duration for {path}")
    return seconds


def read_manifest(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RenderError(f"Audio manifest does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RenderError(f"Audio manifest is not JSON: {path}: {exc}") from exc
    if not isinstance(raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw.items()
    ):
        raise RenderError(
            "Audio manifest must be a JSON object mapping plan line IDs to audio paths"
        )
    return {
        key: (Path(value) if Path(value).is_absolute() else path.parent / value)
        for key, value in raw.items()
    }


def voice_for(line: RenderLine, *, caller_voice: str, analyst_voice: str) -> str:
    return analyst_voice if line.voice == "analyst" else caller_voice


def synthesize(
    line: RenderLine,
    destination: Path,
    *,
    say: str,
    caller_voice: str,
    analyst_voice: str,
    rate_wpm: int,
) -> None:
    subprocess.run(
        [
            say,
            "-o",
            str(destination),
            "-v",
            voice_for(line, caller_voice=caller_voice, analyst_voice=analyst_voice),
            "-r",
            str(rate_wpm),
            line.text,
        ],
        check=True,
    )


def _ass_timestamp(seconds: float) -> str:
    centiseconds = max(0, int(round(seconds * 100)))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    secs, centis = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_text(text: str) -> str:
    """Escape literal ASS syntax while retaining readable line breaks."""
    return (
        text.replace("\\", "\\\\")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\n", "\\N")
    )


def captions_ass(
    timeline: Timeline,
    *,
    duration_s: float,
    recorded_label: bool = False,
    editorial_reference: bool = False,
) -> str:
    """Create readable bottom captions timed to measured synthesized audio."""
    rows = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,46,&H00FFFFFF,&H00FFFFFF,&H00101010,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,1,2,40,40,48,1",
        "Style: Label,Arial,24,&H00FFFFFF,&H00FFFFFF,&H00101010,&H80000000,"
        "0,0,0,0,100,100,0,0,1,1,1,8,24,24,24,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if recorded_label:
        label = (
            "EDITORIAL REFERENCE - NOT PIPELINE OUTPUT"
            if editorial_reference
            else "RECORDED DEMO • CLAUDE + LANGGRAPH"
        )
        rows.append(
            f"Dialogue: 0,{_ass_timestamp(0.0)},{_ass_timestamp(duration_s)},Label,,0,0,0,," + label
        )
    for item in timeline.planned:
        start = max(0.0, item.start_s)
        end = min(duration_s, item.end_s)
        if start >= end:
            continue
        seat = "CALLER" if item.line.voice == "caller" else "ANALYST"
        rows.append(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},Default,,0,0,0,,"
            f"{seat}: {_ass_text(item.line.text)}"
        )
    return "\n".join(rows) + "\n"


def ffmpeg_has_subtitles(ffmpeg: str) -> bool:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and any(
        len(fields) >= 2 and fields[1] == "subtitles"
        for fields in (line.split() for line in result.stdout.splitlines())
    )


def resolve_ffmpeg(explicit: str | None) -> str:
    """Prefer the optional bundled binary unless the user supplied an override."""
    if explicit:
        return explicit
    try:
        from imageio_ffmpeg import get_ffmpeg_exe  # type: ignore[import-untyped]
    except (ImportError, OSError):
        return "ffmpeg"
    try:
        return cast(str, get_ffmpeg_exe())
    except (OSError, RuntimeError):
        return "ffmpeg"


def require_subtitles_filter(ffmpeg: str) -> None:
    """Fail before synthesis if requested captions cannot be burned in."""
    if not ffmpeg_has_subtitles(ffmpeg):
        raise RenderError(
            "Captions requested, but this FFmpeg has no `subtitles` filter; "
            "install a subtitles/libass-capable FFmpeg or pass --no-captions."
        )


def _subtitle_filter(path: Path) -> str:
    escaped = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return f"subtitles=filename='{escaped}'"


def render_video(
    *,
    ffmpeg: str,
    clip: Path,
    output: Path,
    source_start_s: float,
    duration_s: float,
    planned: Sequence[PlannedLine],
    audio_by_id: Mapping[str, Path],
    captions: Path | None = None,
) -> None:
    if not planned:
        raise RenderError("No planned speech falls inside the selected window")
    command = [
        ffmpeg,
        "-n",
        "-hide_banner",
        "-ss",
        f"{source_start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-i",
        str(clip),
    ]
    for item in planned:
        command.extend(["-i", str(audio_by_id[item.line.id])])
    filters: list[str] = []
    labels: list[str] = []
    for index, item in enumerate(planned, start=1):
        label = f"a{index}"
        filters.append(
            f"[{index}:a]atrim=duration={item.duration_s:.3f},"
            f"adelay={round(item.start_s * 1000)}:all=1[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        f"{''.join(labels)}amix=inputs={len(labels)}:duration=longest:normalize=0[mixed]"
    )
    filters.append(f"[mixed]aresample=async=1,apad=whole_dur={duration_s:.3f}[mix]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "0:v:0",
            "-map",
            "[mix]",
        ]
    )
    if captions is not None:
        command.extend(["-vf", _subtitle_filter(captions)])
    command.extend(
        [
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-t",
            f"{duration_s:.3f}",
            str(output),
        ]
    )
    subprocess.run(command, check=True)


def report_markdown(
    timeline: Timeline,
    *,
    plan: Path,
    clip: Path,
    source_start_s: float,
    duration_s: float,
    estimated: bool,
    captions: Path | None = None,
    research: ResearchBrief | None = None,
    caller_voice: str = CALLER_VOICE,
    analyst_voice: str = ANALYST_VOICE,
    editorial_reference: bool = False,
) -> str:
    lines = [
        "# Demo render report",
        "",
        "- Input mode: precomputed offline plan (not live commentary).",
        f"- Plan: `{plan}`",
        f"- Clip: `{clip}`",
        f"- Window: {duration_s:.3f} seconds; source video starts at {source_start_s:.3f} s.",
        "- Alignment: each line uses its at_s cue relative to the selected clip window.",
        "- Speech order: plan order without overlap; goal calls can trim preceding speech.",
        "- Broadcast audio: muted; the output contains only synthesized commentary.",
        f"- Voice assignment: caller `{caller_voice}`; analyst `{analyst_voice}` (distinct).",
    ]
    if captions is not None:
        lines.append(
            f"- Captions: bottom ASS captions from measured audio timing; sidecar `{captions}`."
        )
    if editorial_reference:
        lines.append(
            "- Editorial reference: not pipeline output; "
            "do not represent this as generated demo text."
        )
    speaking = audible_role_seconds(timeline, duration_s=duration_s)
    total_speaking = sum(speaking.values())
    lines.extend(
        [
            "",
            "## Measured audible speaking balance",
            "",
            "Measured after goal-priority and clip-end trimming; "
            "this reports the selected plan, not a quota.",
            f"- Caller ({caller_voice}): {speaking['caller']:.2f} s"
            + (f" ({speaking['caller'] / total_speaking:.1%})" if total_speaking else " (n/a)"),
            f"- Analyst ({analyst_voice}): {speaking['analyst']:.2f} s"
            + (f" ({speaking['analyst'] / total_speaking:.1%})" if total_speaking else " (n/a)"),
        ]
    )
    cited_research_ids = {
        research_id for item in timeline.planned for research_id in item.line.research_ids
    }
    if research is not None and cited_research_ids:
        lines.extend(
            [
                "",
                f"## Research cited (as of {research.as_of.isoformat()})",
                "",
                "These facts are background only; they did not establish a visual event.",
            ]
        )
        for fact in research.facts:
            if fact.id in cited_research_ids:
                lines.append(
                    f"- `{fact.id}` — {fact.text} "
                    f"([source]({fact.source_url}); published {fact.source_date.isoformat()})"
                )
    if estimated:
        lines.append(
            "- Durations below are estimates for dry-run inspection; "
            "a rendered export probes each audio file."
        )
    lines.extend(
        [
            "",
            "| # | voice | plan time | placed | delay | end | status | text |",
            "|---:|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for number, item in enumerate(timeline.planned, start=1):
        status = (
            "trimmed for goal"
            if item.trimmed_for_goal_s
            else "truncated"
            if item.truncated_s
            else ("late" if item.delayed_s else "on cue")
        )
        safe_text = item.line.text.replace("|", "\\|")
        lines.append(
            f"| {number} | {item.line.voice} | {item.line.at_s:.2f} | {item.start_s:.2f} | "
            f"{item.delayed_s:.2f} | {item.end_s:.2f} | {status} | {safe_text} |"
        )
    if timeline.omitted_for_goal:
        omitted = ", ".join(line.id for line in timeline.omitted_for_goal)
        lines.append(f"\nGoal priority omitted preceding speech: {omitted}.")
    trimmed_for_goal = [item for item in timeline.planned if item.trimmed_for_goal_s]
    if trimmed_for_goal:
        details = ", ".join(
            f"{item.line.id} by {item.trimmed_for_goal_s:.2f}s" for item in trimmed_for_goal
        )
        lines.append(f"\nGoal priority trimmed preceding speech: {details}.")
    truncated = [item for item in timeline.planned if item.truncated_s]
    if truncated:
        lines.append(
            f"\nClip end truncates {len(truncated)} line(s); their audio was generated in full "
            f"but the MP4 ends at {duration_s:.2f} s."
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, type=Path, help="recorded_clip plan JSON")
    parser.add_argument("--clip", required=True, type=Path, help="source broadcast video")
    parser.add_argument("--out", type=Path, help="MP4 to create (required unless --dry-run)")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON mapping existing plan line IDs to replacement audio files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print transcript/timeline; do not synthesize or render",
    )
    captions = parser.add_mutually_exclusive_group()
    captions.add_argument(
        "--captions",
        dest="captions",
        action="store_true",
        help="burn readable bottom captions and export an ASS sidecar",
    )
    captions.add_argument(
        "--no-captions",
        dest="captions",
        action="store_false",
        help="disable captions (plans default to captions)",
    )
    parser.set_defaults(captions=True)
    parser.add_argument("--caller-voice", default=CALLER_VOICE)
    parser.add_argument("--analyst-voice", default=ANALYST_VOICE)
    parser.add_argument("--rate", type=int, default=SAY_RATE_WPM, metavar="WPM")
    parser.add_argument("--say", default="say", help="macOS say executable")
    parser.add_argument(
        "--ffmpeg",
        default=None,
        help="FFmpeg executable (default: bundled imageio-ffmpeg binary when installed)",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = read_plan(args.plan)
    if args.caller_voice == args.analyst_voice:
        raise RenderError("Choose two distinct voices for caller and analyst")
    if args.rate <= 0:
        raise RenderError("--rate must be positive")
    if not args.clip.is_file():
        raise RenderError(f"Clip does not exist: {args.clip}")
    if Path(plan.source_basename).name != args.clip.name:
        raise RenderError("Plan source.basename does not match clip")
    remaining = probe_duration(args.clip, ffprobe=args.ffprobe) - plan.source_start_s
    if plan.duration_s > remaining + 0.05:
        raise RenderError(f"Selected source window is outside the clip ({remaining:.2f} s remain)")

    manifest = read_manifest(args.manifest)
    unknown = sorted(set(manifest) - {line.id for line in plan.lines})
    if unknown:
        raise RenderError(f"Manifest refers to no plan line: {', '.join(unknown)}")
    editorial_reference = (
        isinstance(plan.metadata.get("review"), dict)
        and plan.metadata["review"].get("not_pipeline_output") is True
    )
    report_options = dict(
        plan=plan.path,
        clip=args.clip,
        source_start_s=plan.source_start_s,
        duration_s=plan.duration_s,
        research=plan.research,
        caller_voice=args.caller_voice,
        analyst_voice=args.analyst_voice,
        editorial_reference=editorial_reference,
    )
    if args.dry_run:
        timeline = plan_timeline(plan.lines, duration_s=plan.duration_s)
        print(report_markdown(timeline, estimated=True, **report_options), end="")
        return 0

    if args.out is None:
        raise RenderError("--out is required unless --dry-run is set")
    output = args.out.resolve()
    caption_output = output.with_name(output.name + ".captions.ass")
    report = output.with_name(output.name + ".report.md")
    protected = {
        args.clip.resolve(),
        args.plan.resolve(),
        *[p.resolve() for p in manifest.values()],
    }
    if args.manifest:
        protected.add(args.manifest.resolve())
    for target in [output, report, *([caption_output] if args.captions else [])]:
        if target in protected or target.exists():
            raise RenderError(f"Refusing to overwrite existing output or input: {target}")

    missing_audio = [line for line in plan.lines if line.id not in manifest]
    if missing_audio and shutil.which(args.say) is None:
        raise RenderError(f"Cannot find say executable: {args.say}")
    for line_id, supplied in manifest.items():
        if not supplied.is_file():
            raise RenderError(f"Manifest audio for {line_id!r} does not exist: {supplied}")
    ffmpeg = resolve_ffmpeg(args.ffmpeg)
    if shutil.which(ffmpeg) is None or shutil.which(args.ffprobe) is None:
        raise RenderError("ffmpeg and ffprobe must be available on PATH")
    if args.captions:
        require_subtitles_filter(ffmpeg)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="commentary-demo-", dir=output.parent) as temporary:
        audio_dir = Path(temporary)
        audio_by_id = dict(manifest)
        # Measure every line: estimated scheduling must not decide which actual
        # audio exists, because shorter recordings can change goal-priority cuts.
        for line in missing_audio:
            destination = audio_dir / f"{line.order:05d}.aiff"
            synthesize(
                line,
                destination,
                say=args.say,
                caller_voice=args.caller_voice,
                analyst_voice=args.analyst_voice,
                rate_wpm=args.rate,
            )
            audio_by_id[line.id] = destination
        seconds = {
            line.id: probe_duration(audio_by_id[line.id], ffprobe=args.ffprobe)
            for line in plan.lines
        }
        timeline = plan_timeline(plan.lines, duration_s=plan.duration_s, audio_seconds=seconds)
        caption_path = audio_dir / "captions.ass" if args.captions else None
        if caption_path:
            caption_path.write_text(
                captions_ass(
                    timeline,
                    duration_s=plan.duration_s,
                    recorded_label=True,
                    editorial_reference=editorial_reference,
                ),
                encoding="utf-8",
            )
        render_video(
            ffmpeg=ffmpeg,
            clip=args.clip,
            output=output,
            source_start_s=plan.source_start_s,
            duration_s=plan.duration_s,
            planned=timeline.planned,
            audio_by_id=audio_by_id,
            captions=caption_path,
        )
        if caption_path:
            shutil.copyfile(caption_path, caption_output)

    report.write_text(
        report_markdown(
            timeline,
            estimated=False,
            captions=caption_output if args.captions else None,
            **report_options,
        ),
        encoding="utf-8",
    )
    print(f"Rendered {output} with {len(timeline.planned)} planned line(s). Report: {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RenderError, subprocess.CalledProcessError) as exc:
        print(f"render_demo: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
