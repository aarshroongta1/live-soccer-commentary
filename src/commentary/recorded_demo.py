"""A small, auditable *offline* commentary demo for a recorded video clip.

This is deliberately not the live runtime.  The observer sees sampled frames
from the complete requested clip, then one writer makes a short two-voice
script from that evidence. Every generated sentence is retained verbatim and
strictly validated; code may only move a line later into an audited legal slot.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal, TypedDict

import cv2
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator, model_validator

from commentary.llm.base import LLMBackend, Parsed, encode_frame, image_block, text_block
from commentary.recorded_research import ResearchBrief, research_brief

MAX_FRAMES = 90
OBSERVER_WINDOW_S = 6.0
OBSERVER_CONTEXT_S = 2.0
MAX_OBSERVATIONS_PER_WINDOW = 6
MAX_OBSERVATIONS = 48
WORDS_PER_SECOND = 3.0
LINE_GAP_S = 0.35
# Word-count speech is only an estimate. Preserve a 0.30s minimum silence so
# a 50ms rounding allowance cannot turn into estimated speech overlap.
MIN_LINE_GAP_S = LINE_GAP_S - 0.05


class ObservedActor(BaseModel):
    """A directly readable identity attached only to this observation's action."""

    side: Literal["home", "away"]
    shirt_number: int | None = Field(default=None, ge=1, le=99)
    shirt_name: str | None = Field(default=None, min_length=1, max_length=80)
    identity_source: Literal["shirt_number", "shirt_name", "graphic"]
    action_role: Literal["ball_carrier", "passer", "shooter", "scorer", "visible_player"]
    # The observer may not be trusted with this field. Resolution below overwrites it.
    resolved_name: str | None = None

    @model_validator(mode="after")
    def source_has_readable_identity(self) -> ObservedActor:
        if self.identity_source == "shirt_number" and self.shirt_number is None:
            raise ValueError("shirt_number source requires shirt_number")
        if self.identity_source == "shirt_name" and not self.shirt_name:
            raise ValueError("shirt_name source requires shirt_name")
        if self.shirt_number is None and not self.shirt_name:
            raise ValueError("actor needs a readable shirt number or name")
        return self


class Observation(BaseModel):
    id: str = Field(min_length=1, max_length=40)
    at_s: float
    kind: Literal["play", "goal", "celebration", "replay"]
    description: str = Field(min_length=1, max_length=280)
    actor: ObservedActor | None = None


class ObserverResult(BaseModel):
    observations: list[Observation] = Field(max_length=MAX_OBSERVATIONS)


class PlanLine(BaseModel):
    id: str
    at_s: float
    voice: Literal["caller", "analyst"]
    text: str
    evidence_ids: list[str]
    research_ids: list[str] = Field(default_factory=list)


class ScriptLine(BaseModel):
    at_s: float
    voice: Literal["caller", "analyst"]
    text: str = Field(min_length=1, max_length=220)
    evidence_ids: list[str] = Field(default_factory=list)
    research_ids: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class JointScript(BaseModel):
    """Final two-voice text from the single recorded-demo writer."""

    lines: list[ScriptLine] = Field(min_length=7, max_length=12)


class Source(BaseModel):
    basename: str
    start_s: float
    duration_s: float


class DemoPlan(BaseModel):
    version: Literal[1] = 1
    mode: Literal["recorded_clip"] = "recorded_clip"
    source: Source
    observations: list[Observation]
    lines: list[PlanLine]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Frame:
    """One JPEG frame, where ``at_s`` is relative to the requested clip."""

    at_s: float
    jpeg: bytes


@dataclass(frozen=True)
class RecordedDemoConfig:
    clip: Path
    start_s: float = 0.0
    duration_s: float = 40.0
    pack: dict[str, Any] | None = None
    observer_model: str = "gpt-6-astra"
    writer_model: str = "gpt-5.6-terra"
    audit_dir: Path | None = None
    reused_observer: ObserverResult | None = None
    reuse_source: str | None = None
    prior_observation_cost: float = 0.0
    research_brief: ResearchBrief | None = None
    research_source: str | None = None
    research_fixture: str | None = None
    research_as_of: str | None = None
    research_model: str = "gpt-5.6-terra"


class DemoState(TypedDict, total=False):
    frames: list[Frame]
    config: RecordedDemoConfig
    observations: list[Observation]
    research: ResearchBrief | None
    script: JointScript
    audit: dict[str, dict[str, Any]]


def _words(text: str) -> int:
    return len(text.split())


def _finite_in_range(value: float, duration_s: float, field: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= duration_s:
        raise ValueError(f"{field} must be finite and within 0..{duration_s:g}")


def validate_inputs(clip: Path, start_s: float, duration_s: float) -> float:
    """Check bounds before a model can be called and return source duration."""
    if start_s < 0 or not math.isfinite(start_s):
        raise ValueError("start_s must be a finite value at or above zero")
    if duration_s <= 0 or not math.isfinite(duration_s):
        raise ValueError("duration_s must be a finite positive value")
    if not clip.is_file():
        raise ValueError(f"clip does not exist: {clip}")
    capture = cv2.VideoCapture(str(clip))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError(f"could not read video bounds: {clip}")
    source_duration = frames / fps
    if start_s + duration_s > source_duration + 0.05:
        requested_end = start_s + duration_s
        raise ValueError(
            f"requested {start_s:g}s..{requested_end:g}s exceeds clip duration "
            f"{source_duration:.2f}s"
        )
    return source_duration


def sample_clip(clip: Path, start_s: float, duration_s: float, *, fps: float = 2.0) -> list[Frame]:
    """Sample at most 90 timestamped frames, resized to 1280 pixels wide."""
    validate_inputs(clip, start_s, duration_s)
    count = min(MAX_FRAMES, max(1, math.ceil(duration_s * fps)))
    capture = cv2.VideoCapture(str(clip))
    sampled: list[Frame] = []
    try:
        for index in range(count):
            # At two fps this is ordinary two-fps sampling.  For a longer
            # request the 90-frame ceiling spreads coverage over the whole clip.
            at_s = duration_s * index / count
            capture.set(cv2.CAP_PROP_POS_MSEC, (start_s + at_s) * 1000)
            ok, image = capture.read()
            if not ok:
                break
            sampled.append(
                Frame(
                    at_s=at_s,
                    jpeg=encode_frame(_stamp_frame(image, at_s), max_width=1280),
                )
            )
    finally:
        capture.release()
    if not sampled:
        raise ValueError(f"no frames could be sampled from: {clip}")
    return sampled


def _stamp_frame(image: Any, at_s: float) -> Any:
    """Add a small timestamp strip above the image without covering match content."""
    height, width = image.shape[:2]
    border_height = max(24, round(height * 0.045))
    stamped = cv2.copyMakeBorder(
        image,
        border_height,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(28, 28, 28),
    )
    font_scale = max(0.45, min(0.8, width / 1600))
    cv2.putText(
        stamped,
        f"t={at_s:05.2f}s",
        (12, border_height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return stamped


def safe_pack(value: Any) -> dict[str, Any] | None:
    """Keep only team display details and name/number roster identity mapping."""
    if not isinstance(value, dict):
        return None
    teams = value.get("teams", value)
    safe: dict[str, Any] = {}
    if isinstance(teams, dict):
        for side, team in teams.items():
            if not isinstance(team, dict):
                continue
            name = team.get("name", team.get("team"))
            colours = team.get(
                "kit_colours", team.get("kit_colors", team.get("colours", team.get("kit")))
            )
            roster = _safe_roster(team)
            entry = {
                key: item
                for key, item in {
                    "name": name,
                    "kit_colours": colours,
                    "roster": roster,
                }.items()
                if item
            }
            if entry:
                safe[str(side)] = entry
    return {"teams": safe} if safe else None


def _safe_roster(team: dict[str, Any]) -> list[dict[str, Any]]:
    """Strip a pack's lineup to only roster names and shirt numbers."""
    entries: list[dict[str, Any]] = []
    for field in ("roster", "starters", "bench"):
        people = team.get(field, [])
        if not isinstance(people, list):
            continue
        for person in people:
            if isinstance(person, str):
                raw_name: Any = person
                number: Any = None
            elif isinstance(person, dict):
                raw_name = person.get("name")
                number = person.get("number")
            else:
                continue
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            clean: dict[str, Any] = {"name": raw_name.strip()}
            if isinstance(number, int) and not isinstance(number, bool) and 1 <= number <= 99:
                clean["number"] = number
            if clean not in entries:
                entries.append(clean)
    return entries


def _normalized_name_tokens(value: str) -> tuple[str, ...]:
    """Return accent-insensitive, whole tokens without joining their boundaries."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return tuple(re.findall(r"[^\W_]+", without_marks))


def _actor_matches(actor: ObservedActor, roster: Sequence[dict[str, Any]]) -> str | None:
    """Return one allowlisted name, or none for missing, ambiguous, or contradictory data."""
    candidates = list(roster)
    if actor.shirt_number is not None:
        candidates = [item for item in candidates if item.get("number") == actor.shirt_number]
    if actor.shirt_name:
        readable = _normalized_name_tokens(actor.shirt_name)
        candidates = [
            item
            for item in candidates
            if _shirt_name_matches(readable, _normalized_name_tokens(str(item["name"])))
        ]
    names = {str(item["name"]) for item in candidates}
    return names.pop() if len(names) == 1 else None


def _shirt_name_matches(readable: tuple[str, ...], roster_name: tuple[str, ...]) -> bool:
    """Match a readable shirt name only as an exact contiguous roster token sequence."""
    if not readable or len(readable) > len(roster_name):
        return False
    return any(
        roster_name[index : index + len(readable)] == readable
        for index in range(len(roster_name) - len(readable) + 1)
    )


def _resolve_observation_actors(
    observations: Sequence[Observation], pack: dict[str, Any] | None
) -> list[Observation]:
    """Trust a name only after the raw identity matches this team's safe roster."""
    teams = pack.get("teams", {}) if isinstance(pack, dict) else {}
    resolved: list[Observation] = []
    for observation in observations:
        actor = observation.actor
        team = teams.get(actor.side) if actor is not None else None
        roster = team.get("roster", []) if isinstance(team, dict) else []
        name = (
            _actor_matches(actor, roster)
            if actor is not None and isinstance(roster, list)
            else None
        )
        trusted = (
            actor.model_copy(update={"resolved_name": name}) if actor is not None and name else None
        )
        resolved.append(observation.model_copy(update={"actor": trusted}))
    return resolved


def _writer_pack(pack: dict[str, Any] | None) -> dict[str, Any] | None:
    """The writer needs team labels and kits, never the roster used for resolution."""
    teams = pack.get("teams") if isinstance(pack, dict) else None
    if not isinstance(teams, dict):
        return None
    writer_teams: dict[str, dict[str, Any]] = {}
    for side, team in teams.items():
        if not isinstance(team, dict):
            continue
        details = {key: team[key] for key in ("name", "kit_colours") if team.get(key) is not None}
        if details:
            writer_teams[str(side)] = details
    return {"teams": writer_teams} if writer_teams else None


def _usage(parsed: Parsed[Any]) -> dict[str, Any]:
    return {"model": parsed.model, **asdict(parsed.usage)}


def _persist_stage(audit_dir: Path | None, stage: str, value: dict[str, Any]) -> None:
    """Keep successful paid stages if a later call fails or times out."""
    if audit_dir is not None:
        write_audit(audit_dir, {stage: value})


def _json(value: BaseModel | dict[str, Any] | list[Any]) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, separators=(",", ":"))


def _observer_windows(
    frames: Sequence[Frame], duration_s: float, *, window_s: float = OBSERVER_WINDOW_S
) -> list[tuple[float, float, list[Frame]]]:
    """Partition sampled frames into chronological windows without rebasing timestamps."""
    if not frames:
        return []
    last_at = max(frame.at_s for frame in frames)
    windows: list[tuple[float, float, list[Frame]]] = []
    start = 0.0
    while start <= last_at and start < duration_s:
        end = min(start + window_s, duration_s)
        selected = [
            frame
            for frame in frames
            if start <= frame.at_s < end or (end == duration_s and frame.at_s == end)
        ]
        if selected:
            windows.append((start, end, selected))
        start += window_s
    return windows


def _observer_context_frames(
    frames: Sequence[Frame], window_start: float, *, context_s: float = OBSERVER_CONTEXT_S
) -> list[Frame]:
    """Return a bounded preceding overlap for visual continuity, never new evidence."""
    if window_start <= 0:
        return []
    return [frame for frame in frames if window_start - context_s <= frame.at_s < window_start]


def _observer_blocks(
    frames: Sequence[Frame],
    pack: dict[str, Any] | None,
    *,
    window_start: float = 0.0,
    window_end: float | None = None,
    context_frames: Sequence[Frame] = (),
) -> list[dict[str, Any]]:
    if window_end is None:
        window_end = max((frame.at_s for frame in frames), default=0.0)
    context = {
        "mode": "OFFLINE RECORDED CLIP — you can inspect the whole clip, not live video.",
        "timestamps": (
            "All at_s values are global seconds relative to the requested clip; "
            "do not reset them per window."
        ),
        "window": {"start_s": window_start, "end_s_exclusive": window_end},
        "context_only": {
            "start_s": max(0.0, window_start - OBSERVER_CONTEXT_S),
            "end_s_exclusive": window_start,
        },
        "pack": pack,
    }
    blocks: list[dict[str, Any]] = [text_block(_json(context))]
    for frame in context_frames:
        blocks.append(text_block(f"context-only frame at_s={frame.at_s:.2f}"))
        blocks.append(image_block(frame.jpeg))
    for frame in frames[:MAX_FRAMES]:
        blocks.append(text_block(f"frame at_s={frame.at_s:.2f}"))
        blocks.append(image_block(frame.jpeg))
    return blocks


OBSERVER_SYSTEM = """You are the visual observer for one chronological window of an OFFLINE
recorded football clip. Report at most six concrete observations from this window, in
global timestamp order. Use only the supplied frame timestamps; never reset at_s to zero
for a later window. An observation must fall within the stated window bounds. Prefer the
first clearly visible ball-in-net moment for a goal over later celebration or scoreboard
frames. Track a decisive buildup-to-delivery-to-shot-to-goal transition as distinct
observations when visible; do not collapse an early attack into a goal or repeat the same
goal for later confirmation. Describe visible passes, crosses, shots, goals, celebrations,
or replays. Do not guess player names, score, teams, statistics, or outcomes not visible.
Attach actor only when a readable shirt number/name or action-linked graphic identifies the
player performing this observation's action. Set the actor's side, raw readable identity,
identity_source, and action_role. A celebration identity identifies only that celebration;
never use it to identify an earlier pass, delivery, shot, or goal. Exception: at the current
readable timestamp only, you may attach scorer to a later celebration/follow-up when the
shared frames visibly and continuously establish that same finisher from finish through the
celebration. Do not backdate that identity to the shot or goal, and a mere celebrant, scene
cut, roster position, or proximity is not enough. Leave actor empty for a bystander, an
unclear shirt, or an identity inferred from position. Within this window, use the earliest
readable timestamp and never copy a later-frame identity onto earlier action.
Some preceding frames may be explicitly marked context-only. They can help you visually
follow the same player into this window, but they are not new evidence: do not emit an
observation timestamped in that overlap and do not retroactively identify an earlier action.
You may connect a readable shirt to a subsequent action only when both are visually supported
within the shared supplied frames; never infer a scorer from a celebration alone.
Do not set resolved_name.
If uncertain, say so in description. Track the ball itself: distinguish open play outside
the penalty area from action inside it, and do not infer a precise target spot, body part,
cross, cut-back,
or delivery type unless the frames make it clear. If the ball is obscured, describe the
uncertainty instead of confidently naming its location. Do not manufacture a loose ball
or a goal from a later reaction."""

WRITER_SYSTEM = """Write one final OFFLINE recorded-football two-voice script after the complete
clip has been observed. Return 7–12 chronological lines using caller and analyst voices.
Write spoken broadcast dialogue, not descriptions of footage. The caller follows action;
the analyst responds with why it matters. Plan key goal and delivery cues first, then fit
buildup and analyst turns around them.
Caller lines must cite relevant visual evidence_ids available at or before at_s; analyst
lines use visual evidence and/or research_ids. Caller lines never cite research. Keep all
factual qualifiers grounded in cited IDs. For a typical 40-second clip, use three distinct
substantial analyst turns, distributed as meaningful responses rather than fillers. Aim for
roughly 60/40 caller/analyst word balance without forcing a statistic into an unrelated gap.
Keep total script words at or below clip_duration_s * 2.3. Analyst turns are normally 6–12
words; caller turns stay short outside the final attack. Avoid repetitive empty narration,
caption-like scoreboard narration, and coach-gesture filler. Connect research to current
play rather than standalone trivia, and preserve its all-competitions/competition/time scope
exactly. Use at most two research facts. At least one analyst line must cite a play or goal
directly, without research_ids. Do not use "caption confirms" phrasing.

Tell a coherent action story: buildup, visible delivery into the area, visible shot, then
the first clear ball-in-net goal. Start the goal call at the first clear net evidence and
open it with "Goal!" or "It's in!". If shot-to-goal is under one second, combine them rather
than forcing separate bursts. Use terse 1–4 word bursts only in the final attack. Do not
invent names, body parts, targets, tactical intent, or score details. Before drafting, scan
the observations for actor.resolved_name and plan natural, prominent named moments wherever
the cited evidence and timing permit. Prefer a supported named action to a generic team line
for the same moment. Give a named player whose running or celebrating is visible a meaningful
reaction: that is valid commentary, not inherently filler. Integrate distinct observed
identities across the clip when evidence and timing allow, prioritizing named actions and a
meaningful post-goal named reaction. Do not impose a player-name quota, force every available
name, or turn this into a roll call. Use a player name only when a cited observation has
actor.resolved_name, and only for that same observed action; never take a name from a
description, pack, research fact, or another observation. A later celebration can support
that named player's visible reaction, but cannot identify the earlier scorer or finish; never
transfer a celebration identity to an earlier action. For both caller and analyst, never use
kit-color or stripe labels, or generic "runner", "player", or "attacker" placeholders, as a
spoken identity stand-in. When an actor is unnamed, use a supported team or nationality (for
example, "Argentina" or "the French") when available, or write an action-led sentence that
omits the person. Never say "first time" or "first-time" unless the cited visual description
explicitly establishes that there was no intervening touch. Research facts retain their exact
stated scope; use them only where relevant.
Every line must be a complete sentence except a terse attack burst. Choose evidence and
wording; deterministic code places each line at the earliest legal timestamp at or after
your requested at_s and its cited evidence. Keep each line under 25 words and within the
clip duration. A caller goal cue that opens with "Goal!" or "It's in!" should request the
first cited goal evidence; its requested reaction slot is pinned and cannot be delayed by
earlier script. Research never grants identity attribution. Return the final words exactly;
they will never be rewritten in code."""


async def _research(state: DemoState, *, backend: LLMBackend) -> DemoState:
    config = state["config"]
    if config.research_brief is not None:
        audit = dict(state.get("audit", {}))
        audit["research"] = {
            "result": config.research_brief.model_dump(mode="json"),
            "usage": {"cost_usd": 0.0, "reused": True, "source": config.research_source},
        }
        _persist_stage(config.audit_dir, "research", audit["research"])
        return {"research": config.research_brief, "audit": audit}
    if config.research_fixture is None:
        return {"research": None}
    if config.research_as_of is None:
        raise ValueError("research_as_of is required with research_fixture")
    as_of = date.fromisoformat(config.research_as_of)
    parsed = await research_brief(
        backend,
        fixture=config.research_fixture,
        as_of=as_of,
        model=config.research_model,
    )
    audit = dict(state.get("audit", {}))
    audit["research"] = {
        "raw_result": parsed.value.model_dump(mode="json"),
        "usage": _usage(parsed),
    }
    _persist_stage(config.audit_dir, "research", audit["research"])
    brief = ResearchBrief(fixture=config.research_fixture, as_of=as_of, facts=parsed.value.facts)
    audit["research"]["result"] = brief.model_dump(mode="json")
    _persist_stage(config.audit_dir, "research", audit["research"])
    return {"research": brief, "audit": audit}


async def _observe(state: DemoState, *, backend: LLMBackend) -> DemoState:
    config = state["config"]
    if config.reused_observer is not None:
        usage = {
            "model": config.observer_model,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
            "latency_s": 0.0,
            "reused": True,
            "source": config.reuse_source,
        }
        audit = dict(state.get("audit", {}))
        audit["observe"] = {
            "raw_result": config.reused_observer.model_dump(mode="json"),
            "usage": usage,
        }
        observations = _resolve_observation_actors(config.reused_observer.observations, config.pack)
        audit["observe"]["result"] = ObserverResult(observations=observations).model_dump(
            mode="json"
        )
        _persist_stage(config.audit_dir, "observe", audit["observe"])
        return {"observations": observations, "audit": audit}

    windows = _observer_windows(state["frames"], config.duration_s)
    if not windows:
        raise ValueError("observer needs at least one non-empty frame window")
    merged: list[Observation] = []
    used_ids: set[str] = set()
    window_audits: list[dict[str, Any]] = []
    for window_index, (window_start, window_end, window_frames) in enumerate(windows):
        context_frames = _observer_context_frames(state["frames"], window_start)
        parsed = await backend.parse(
            model=config.observer_model,
            system=OBSERVER_SYSTEM,
            blocks=_observer_blocks(
                window_frames,
                config.pack,
                window_start=window_start,
                window_end=window_end,
                context_frames=context_frames,
            ),
            output_format=ObserverResult,
            max_tokens=3072,
            effort="medium",
            tag="recorded_observer",
        )
        raw_window_audit = {
            "window_index": window_index + 1,
            "start_s": window_start,
            "end_s": window_end,
            "frame_count": len(window_frames),
            "context_frame_count": len(context_frames),
            "raw_result": parsed.value.model_dump(mode="json"),
            "usage": _usage(parsed),
        }
        # Preserve the successful paid response even if deterministic bounds checks
        # below reject it, just as the other stages preserve raw model output.
        _persist_stage(
            config.audit_dir,
            f"observe-window-{window_index + 1:02d}",
            raw_window_audit,
        )
        if len(parsed.value.observations) > MAX_OBSERVATIONS_PER_WINDOW:
            raise ValueError(
                f"observer window {window_index + 1} returned more than "
                f"{MAX_OBSERVATIONS_PER_WINDOW} observations"
            )
        normalized: list[Observation] = []
        for item_index, item in enumerate(parsed.value.observations):
            if not window_start <= item.at_s < window_end:
                raise ValueError(
                    f"observer window {window_index + 1} observation {item.id!r} "
                    f"has timestamp {item.at_s:g}s outside "
                    f"{window_start:g}..{window_end:g}s"
                )
            observation_id = item.id
            if observation_id in used_ids:
                base_id = f"w{window_index + 1:02d}-o{item_index + 1:02d}"
                observation_id = base_id
                suffix = 2
                while observation_id in used_ids:
                    observation_id = f"{base_id}-{suffix}"
                    suffix += 1
            used_ids.add(observation_id)
            normalized.append(item.model_copy(update={"id": observation_id}))
        merged.extend(_resolve_observation_actors(normalized, config.pack))
        window_audit = {
            **raw_window_audit,
            "result": {"observations": [item.model_dump(mode="json") for item in normalized]},
        }
        window_audits.append(window_audit)
        _persist_stage(config.audit_dir, f"observe-window-{window_index + 1:02d}", window_audit)

    merged.sort(key=lambda item: item.at_s)
    result = ObserverResult(observations=merged)
    usage_fields = (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
        "latency_s",
    )
    aggregate_usage: dict[str, Any] = {
        "model": config.observer_model,
        **{
            field: sum(float(window["usage"].get(field, 0.0)) for window in window_audits)
            for field in usage_fields
        },
        "windows": len(window_audits),
    }
    audit = dict(state.get("audit", {}))
    audit["observe"] = {
        "result": result.model_dump(mode="json"),
        "windows": window_audits,
        "usage": aggregate_usage,
    }
    _persist_stage(config.audit_dir, "observe", audit["observe"])
    return {"observations": result.observations, "audit": audit}


async def _write(state: DemoState, *, backend: LLMBackend) -> DemoState:
    """Write both voices, with at most one validation-informed correction."""
    config = state["config"]
    research = state.get("research")
    payload = {
        "mode": "offline_recorded_clip",
        "clip_duration_s": config.duration_s,
        "timing_rule": "next at_s >= current at_s + words(current text)/3 + 0.30",
        "observations": [item.model_dump(mode="json") for item in state["observations"]],
        "research_facts": (
            [item.model_dump(mode="json") for item in research.facts]
            if research is not None
            else []
        ),
        "pack": _writer_pack(config.pack),
    }
    parsed = await backend.parse(
        model=config.writer_model,
        system=WRITER_SYSTEM,
        blocks=[text_block(_json(payload))],
        output_format=JointScript,
        max_tokens=8192,
        effort="low",
        tag="recorded_writer",
    )
    audit = dict(state["audit"])
    initial = {"result": parsed.value.model_dump(mode="json"), "usage": _usage(parsed)}
    # Keep the paid words even if strict deterministic validation rejects them.
    _persist_stage(config.audit_dir, "script-initial", initial)
    audit["script"] = {"initial": initial, "result": initial["result"], "usage": initial["usage"]}
    try:
        constrained, timing_adjustments = _constrain_script_timing(
            parsed.value, state["observations"], config.duration_s
        )
        _validate_joint_script(constrained, state["observations"], config.duration_s, research)
    except ValueError as exc:
        feedback = {
            **payload,
            "invalid_draft": parsed.value.model_dump(mode="json"),
            "timing_audit": _timing_audit(parsed.value, state["observations"], config.duration_s),
            "validation_error": str(exc),
            "total_words": sum(_words(line.text) for line in parsed.value.lines),
            "max_total_words": math.floor(config.duration_s * 2.3),
            "repair_instruction": (
                "Return one complete corrected replacement script. Shorten words to fit each "
                "timing boundary; never move a line before its latest cited evidence, and keep "
                "the goal cue at its evidence timestamp. This is the only repair pass."
            ),
        }
        repaired = await backend.parse(
            model=config.writer_model,
            system=WRITER_SYSTEM,
            blocks=[text_block(_json(feedback))],
            output_format=JointScript,
            max_tokens=8192,
            effort="low",
            tag="recorded_writer",
        )
        retry = {"result": repaired.value.model_dump(mode="json"), "usage": _usage(repaired)}
        _persist_stage(config.audit_dir, "script-retry", retry)
        usage = _combined_usage(initial["usage"], retry["usage"])
        audit["script"] = {
            "initial": initial,
            "retry": retry,
            "result": retry["result"],
            "usage": usage,
            "repair_reason": str(exc),
            "timing_adjustments": [],
        }
        _persist_stage(config.audit_dir, "script", audit["script"])
        constrained, timing_adjustments = _constrain_script_timing(
            repaired.value, state["observations"], config.duration_s
        )
        audit["script"]["result"] = constrained.model_dump(mode="json")
        audit["script"]["timing_adjustments"] = timing_adjustments
        _persist_stage(config.audit_dir, "script", audit["script"])
        _validate_joint_script(constrained, state["observations"], config.duration_s, research)
        return {"script": constrained, "audit": audit}
    audit["script"] = {
        "initial": initial,
        "result": constrained.model_dump(mode="json"),
        "usage": initial["usage"],
        "timing_adjustments": timing_adjustments,
    }
    _persist_stage(config.audit_dir, "script", audit["script"])
    return {"script": constrained, "audit": audit}


def plan_lines(script: JointScript) -> list[PlanLine]:
    """Assign stable IDs without changing any model-authored timing or words."""
    counts = {"caller": 0, "analyst": 0}
    lines: list[PlanLine] = []
    for item in script.lines:
        counts[item.voice] += 1
        lines.append(PlanLine(id=f"{item.voice}-{counts[item.voice]}", **item.model_dump()))
    return lines


def _is_pinned_goal_cue(line: ScriptLine) -> bool:
    """Recognize only the explicit first-reaction openings required by the prompt."""
    opening = line.text.lstrip().casefold()
    return line.voice == "caller" and (
        opening.startswith("goal!") or opening.startswith("it's in!")
    )


def _constrain_script_timing(
    script: JointScript, observations: Sequence[Observation], duration_s: float
) -> tuple[JointScript, list[dict[str, Any]]]:
    """Place every line later into a feasible slot without changing generated words.

    An explicit goal reaction is a deadline. If preceding speech would push it later,
    reject the draft for the existing repair pass instead of making the reaction stale.
    """
    evidence = {item.id: item for item in observations}
    constrained: list[ScriptLine] = []
    adjustments: list[dict[str, Any]] = []
    previous_end = 0.0
    for index, line in enumerate(script.lines, start=1):
        _finite_in_range(line.at_s, duration_s, "line at_s")
        unknown = [item for item in line.evidence_ids if item not in evidence]
        if unknown:
            raise ValueError(f"line {index} has unknown evidence")
        latest_evidence = max((evidence[item].at_s for item in line.evidence_ids), default=0.0)
        previous_minimum = previous_end + MIN_LINE_GAP_S if constrained else 0.0
        if _is_pinned_goal_cue(line):
            pinned_at_s = max(line.at_s, latest_evidence)
            if previous_minimum > pinned_at_s:
                raise ValueError(f"goal cue line {index} is blocked by earlier script")
            scheduled_at_s = pinned_at_s
            reasons = ["latest_cited_evidence"] if latest_evidence > line.at_s else []
        else:
            scheduled_at_s = max(line.at_s, latest_evidence, previous_minimum)
            reasons = []
            if scheduled_at_s > line.at_s:
                if latest_evidence > line.at_s:
                    reasons.append("latest_cited_evidence")
                if previous_minimum > line.at_s:
                    reasons.append("previous_line_end")
        end_s = scheduled_at_s + _words(line.text) / WORDS_PER_SECOND
        if end_s > duration_s:
            raise ValueError(f"line {index} cannot fit before clip duration")
        if scheduled_at_s != line.at_s:
            adjustments.append(
                {
                    "line": index,
                    "from_at_s": line.at_s,
                    "to_at_s": scheduled_at_s,
                    "reasons": reasons,
                    "latest_evidence_s": latest_evidence if line.evidence_ids else None,
                    "previous_line_end_s": previous_end if constrained else None,
                }
            )
        constrained.append(line.model_copy(update={"at_s": scheduled_at_s}))
        previous_end = end_s
    return JointScript(lines=constrained), adjustments


def _validate_joint_script(
    script: JointScript,
    observations: Sequence[Observation],
    duration_s: float,
    research: ResearchBrief | None,
) -> None:
    lines = plan_lines(script)
    validate_plan_parts(observations, lines, duration_s, research=research)
    validate_joint_balance(lines)
    evidence_kinds = {item.id: item.kind for item in observations}
    if not any(
        line.voice == "analyst"
        and not line.research_ids
        and any(evidence_kinds[item] in {"play", "goal"} for item in line.evidence_ids)
        for line in lines
    ):
        raise ValueError("joint script needs one analyst insight on a play or goal")
    if sum(_words(line.text) for line in lines) > duration_s * 2.3:
        raise ValueError("joint script exceeds the total word budget")


def _timing_audit(
    script: JointScript, observations: Sequence[Observation], duration_s: float
) -> list[dict[str, Any]]:
    evidence = {item.id: item.at_s for item in observations}
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(script.lines):
        has_next_line = index + 1 < len(script.lines)
        next_at_s = script.lines[index + 1].at_s if has_next_line else duration_s
        known_times = [evidence[item] for item in line.evidence_ids if item in evidence]
        max_words_before_next = max(
            0,
            math.floor(
                (next_at_s - line.at_s - (MIN_LINE_GAP_S if has_next_line else 0.0))
                * WORDS_PER_SECOND
            ),
        )
        rows.append(
            {
                "line": index + 1,
                "voice": line.voice,
                "at_s": line.at_s,
                "words": _words(line.text),
                "estimated_end_s": round(line.at_s + _words(line.text) / WORDS_PER_SECOND, 3),
                "next_at_s": next_at_s,
                "max_words_before_next": max_words_before_next,
                "latest_evidence_s": max(known_times, default=None),
                "earliest_allowed_start_s": max(known_times, default=None),
                "unknown_evidence_ids": [
                    item for item in line.evidence_ids if item not in evidence
                ],
            }
        )
    return rows


def _combined_usage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    numeric = (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cost_usd",
        "latency_s",
    )
    return {
        "model": str(second.get("model") or first.get("model") or ""),
        **{
            field: float(first.get(field, 0.0)) + float(second.get(field, 0.0)) for field in numeric
        },
    }


def build_recorded_graph(backend: LLMBackend) -> Any:
    """The bounded graph: optional research → observe → one joint script writer."""

    async def researcher(state: DemoState) -> DemoState:
        return await _research(state, backend=backend)

    async def observe(state: DemoState) -> DemoState:
        return await _observe(state, backend=backend)

    async def writer(state: DemoState) -> DemoState:
        return await _write(state, backend=backend)

    graph = StateGraph(DemoState)
    graph.add_node("research", researcher)
    graph.add_node("observe", observe)
    graph.add_node("writer", writer)
    graph.add_edge(START, "research")
    graph.add_edge("research", "observe")
    graph.add_edge("observe", "writer")
    graph.add_edge("writer", END)
    return graph.compile(name="recorded_demo")


def validate_plan_parts(
    observations: Sequence[Observation],
    lines: Sequence[PlanLine],
    duration_s: float,
    *,
    require_caller_count: bool = False,
    research: ResearchBrief | None = None,
) -> None:
    """The non-negotiable, model-independent plan contract."""
    ids = [item.id for item in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("observation IDs must be unique")
    evidence = {item.id: item for item in observations}
    research_ids = {item.id for item in research.facts} if research is not None else set()
    line_ids = [line.id for line in lines]
    if len(line_ids) != len(set(line_ids)):
        raise ValueError("line IDs must be unique")
    previous = -1.0
    caller_count = 0
    for line in lines:
        _finite_in_range(line.at_s, duration_s, "line at_s")
        if line.at_s + _words(line.text) / WORDS_PER_SECOND > duration_s:
            raise ValueError(f"line {line.id} ends after clip duration")
        if line.at_s < previous:
            raise ValueError("lines must be chronological")
        previous = line.at_s
        if line.voice == "caller" and not line.evidence_ids:
            raise ValueError(f"caller line {line.id} has no visual evidence")
        if not line.evidence_ids and not line.research_ids:
            raise ValueError(f"line {line.id} has no evidence or research")
        if any(item not in evidence for item in line.evidence_ids):
            raise ValueError(f"line {line.id} has unknown evidence")
        if line.voice == "caller" and line.research_ids:
            raise ValueError(f"caller line {line.id} cannot cite research")
        if any(item not in research_ids for item in line.research_ids):
            raise ValueError(f"line {line.id} has unknown research")
        if any(evidence[item].at_s > line.at_s for item in line.evidence_ids):
            raise ValueError(f"line {line.id} cites future evidence")
        if _words(line.text) > 25:
            raise ValueError(f"line {line.id} exceeds 25 words")
        if line.voice == "caller":
            caller_count += 1
    for observation in observations:
        _finite_in_range(observation.at_s, duration_s, "observation at_s")
    for left, right in zip(lines, lines[1:], strict=False):
        needed = left.at_s + _words(left.text) / WORDS_PER_SECOND + MIN_LINE_GAP_S
        if right.at_s < needed:
            raise ValueError(f"line {right.id} overlaps {left.id}")
    if require_caller_count and not 4 <= caller_count <= 9:
        raise ValueError("caller must supply 4 to 9 lines")


def validate_joint_balance(lines: Sequence[PlanLine]) -> None:
    """Guard against a nominal second voice that has no meaningful part."""
    analyst = [line for line in lines if line.voice == "analyst"]
    caller = [line for line in lines if line.voice == "caller"]
    if len(caller) < 4:
        raise ValueError("joint script needs at least four caller turns")
    if len(analyst) < 3:
        raise ValueError("joint script needs at least three analyst turns")
    if any(_words(line.text) < 5 for line in analyst):
        raise ValueError("joint script analyst turns must be substantial")
    total_words = sum(_words(line.text) for line in lines)
    analyst_words = sum(_words(line.text) for line in analyst)
    if total_words and not 0.25 <= analyst_words / total_words <= 0.75:
        raise ValueError("joint script analyst share is not meaningful")


async def run_recorded_demo(
    backend: LLMBackend,
    config: RecordedDemoConfig,
    *,
    frames: Sequence[Frame] | None = None,
    audit_dir: Path | None = None,
) -> tuple[DemoPlan, dict[str, dict[str, Any]]]:
    """Run research, observation, and one joint writer, then validate verbatim text."""
    if audit_dir is not None:
        config = replace(config, audit_dir=audit_dir)
    if frames is None and config.reused_observer is None:
        frames = sample_clip(config.clip, config.start_s, config.duration_s)
    if frames is None:
        frames = []
    if (not frames and config.reused_observer is None) or len(frames) > MAX_FRAMES:
        raise ValueError(f"frames must contain 1 to {MAX_FRAMES} items")
    for frame in frames:
        _finite_in_range(frame.at_s, config.duration_s, "frame at_s")
    result = await build_recorded_graph(backend).ainvoke(
        {"frames": list(frames), "config": config, "audit": {}}
    )
    observations = result["observations"]
    research = result.get("research")
    lines = plan_lines(result["script"])
    validate_plan_parts(observations, lines, config.duration_s, research=research)
    validate_joint_balance(lines)
    notes: list[str] = []
    usages = [
        stage["usage"]
        for stage_name, stage in result["audit"].items()
        if not (stage_name == "observe" and config.reused_observer is not None)
    ]
    cost = sum(float(item["cost_usd"]) for item in usages)
    plan = DemoPlan(
        source=Source(
            basename=config.clip.name, start_s=config.start_s, duration_s=config.duration_s
        ),
        observations=observations,
        lines=lines,
        metadata={
            "offline": True,
            "models": {
                "research": (
                    "reused"
                    if config.research_brief is not None
                    else config.research_model
                    if config.research_fixture is not None
                    else None
                ),
                "observer": config.observer_model,
                "writer": config.writer_model,
            },
            "cost": cost,
            "prior_observation_cost": config.prior_observation_cost,
            "reused_observations": config.reuse_source,
            "research": research.model_dump(mode="json") if research is not None else None,
            "research_cost_note": (
                "Research cost is token-only; provider web-search charges are not included. "
                "Source dates are model-declared and schema-validated, not independently verified."
                if research is not None
                else None
            ),
            "trace": "offline: research -> observe -> joint writer",
            "usage": result["audit"],
            "notes": notes,
        },
    )
    return plan, result["audit"]


def write_audit(run_dir: Path, audit: dict[str, dict[str, Any]]) -> None:
    """Write raw structured answers and exact per-stage usage for later review."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for stage, value in audit.items():
        (run_dir / f"{stage}.json").write_text(json.dumps(value, indent=2) + "\n")


def dry_run_estimate(
    frame_count: int, *, fresh_research: bool = False, reused_research: bool = False
) -> dict[str, Any]:
    """An honest ceiling estimate; image-token billing varies by image dimensions."""
    return {
        "offline": True,
        "will_call_models": False,
        "candidate_frames": frame_count,
        "max_frames": MAX_FRAMES,
        "output_token_budgets": {
            **({"research": 4096} if fresh_research else {}),
            "observer_per_window": 3072,
            "joint_writer_initial": 8192,
            "joint_writer_max_repair": 8192,
        },
        "max_joint_writer_calls": 2,
        "observer_window_seconds": OBSERVER_WINDOW_S,
        "observer_context_seconds": OBSERVER_CONTEXT_S,
        "reused_research": reused_research,
        "note": (
            "Dry run validates video bounds and sampling only. A paid run can make "
            "one initial joint-writer call and, only after strict validation fails, "
            "one repair call. Observer windows repeat up to two seconds of preceding "
            "context frames, which are billed again. Use --spend to authorize model calls."
        ),
    }
