"""Prepare source-timed dialogue with the existing observer/caller/analyst agents.

Generation happens before playback. Each speaking turn sees only evidence whose
supporting frames are available by its proposed source-video time. Wall-clock
preparation timings remain a separate trace, never extra video duration.
"""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Self

from pydantic import BaseModel, Field, ModelWrapValidatorHandler, PrivateAttr, model_validator

from commentary.llm.base import LLMBackend, LLMError, Usage, image_block, text_block
from commentary.llm.budget import BudgetExceeded
from commentary.recorded_demo import (
    Frame,
    Observation,
    PlanLine,
    _writer_pack,
    safe_pack,
    validate_plan_parts,
)
from commentary.recorded_research import ResearchBrief
from commentary.replay.contracts import Role, RuntimeSnapshot, SpeechLine, TurnResult
from commentary.replay.identity import identify_players
from commentary.replay.preparation import ACTION_SYSTEM, REPLAY_OBSERVER_SYSTEM, ReplayObservation


class RecordedObservation(ReplayObservation):
    _scene_certainty_downgraded: bool = PrivateAttr(default=False)

    @model_validator(mode="wrap")
    @classmethod
    def uncertain_scene_action(cls, data: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        downgrade = (
            isinstance(data, dict)
            and data.get("kind") in {"celebration", "replay"}
            and isinstance(data.get("action"), dict)
            and data["action"].get("type") == "unknown"
            and data["action"].get("certainty") == "confirmed"
        )
        if downgrade:
            data = {**data, "action": {**data["action"], "certainty": "uncertain"}}
        result = handler(data)
        result._scene_certainty_downgraded = downgrade
        return result


class RecordedObservations(BaseModel):
    observations: list[RecordedObservation] = Field(default_factory=list, max_length=10)


async def observe_recorded(
    backend: LLMBackend,
    frames: list[Frame],
    *,
    model: str,
    identity_model: str,
    research: ResearchBrief | None,
    pack: dict[str, Any] | None,
) -> dict[str, Any]:
    """One full-context vision pass for a short recorded package, then agent lookup."""
    import json

    if (
        not frames
        or len(frames) > 30
        or any(a.at_s >= b.at_s for a, b in zip(frames, frames[1:], strict=False))
    ):
        raise ValueError("recorded observation needs 1–30 chronological frames")
    times = {frame.at_s for frame in frames}
    payload = {
        "playback_s": frames[-1].at_s,
        "available_frame_timestamps": sorted(times),
        "pack": _writer_pack(safe_pack(pack)),
        "mode": "recorded_preparation",
    }
    blocks = [text_block(json.dumps(payload))]
    for frame in frames:
        blocks.extend([text_block(f"Source frame at_s={frame.at_s}"), image_block(frame.jpeg)])
    system = (
        REPLAY_OBSERVER_SYSTEM.replace("Return at\nmost three", "Return up to ten") + ACTION_SYSTEM
    )
    system += (
        "\nThis is one recorded highlights sequence, prepared before playback. Compare "
        "the entire sequence to distinguish live action, repeated replays and reactions. "
        "Do not turn another camera angle of a goal into another goal. Cover the early "
        "on-ball action and any clear decision/setup, not just the final outcome. "
        "Use only necessary supporting frames so late celebrations do not delay evidence. "
        "Actor identity must be readable at that observation, never inferred backward."
    )
    started = time.monotonic()
    parsed = await backend.parse(
        model=model,
        system=system,
        blocks=blocks,
        output_format=RecordedObservations,
        max_tokens=3072,
        tag="recorded_observer",
    )
    previous = -1.0
    for item in parsed.value.observations:
        if (
            item.at_s not in times
            or item.at_s < previous
            or not set(item.action.support_at_s) <= times
        ):
            raise ValueError("recorded observations must cite chronological supplied frames")
        previous = item.at_s
    raw = [
        Observation.model_validate({**item.model_dump(), "id": f"recorded-{i}"})
        for i, item in enumerate(parsed.value.observations, 1)
    ]
    resolved, identity_steps, identity_usage = await resolve_recorded_identities(
        backend, raw, model=identity_model, research=research, pack=safe_pack(pack)
    )
    return {
        "observations": [item.model_dump(mode="json") for item in resolved],
        "raw_observations": [item.model_dump(mode="json") for item in raw],
        "identity_steps": identity_steps,
        "vision_usage": parsed.usage.__dict__,
        "identity_usage": identity_usage.__dict__,
        "generation_duration_s": time.monotonic() - started,
        "normalizations": [
            {
                "id": item.id,
                "field": "action.certainty",
                "from": "confirmed",
                "to": "uncertain",
                "reason": "unknown scene action cannot be confirmed",
            }
            for item in parsed.value.observations
            if item._scene_certainty_downgraded
        ],
        "model": parsed.model,
        "system": system,
    }


async def resolve_recorded_identities(
    backend: LLMBackend,
    observations: Sequence[Observation],
    *,
    model: str,
    research: ResearchBrief | None,
    pack: dict[str, Any] | None,
) -> tuple[list[Observation], list[dict[str, Any]], Usage]:
    """Keep each lookup within the identity agent's three-proposal contract."""
    result = {item.id: item.model_copy(update={"actor": None}) for item in observations}
    candidates = [item for item in observations if item.actor is not None]
    steps: list[dict[str, Any]] = []
    usage = Usage()
    for start in range(0, len(candidates), 3):
        batch = candidates[start : start + 3]
        resolved, trace, spent = await identify_players(
            backend, model=model, observations=batch, research=research, pack=pack
        )
        result.update({item.id: item for item in resolved})
        steps.extend([{**row, "batch_evidence_ids": [item.id for item in batch]} for row in trace])
        usage = usage + spent
    return [result[item.id] for item in observations], steps, usage


def ready_at(observation: Observation) -> float:
    return max([observation.at_s, *(observation.action.support_at_s if observation.action else [])])


async def prepare_dialogue(
    run_role: Callable[[Role, RuntimeSnapshot], Awaitable[TurnResult]],
    observations: Sequence[Observation],
    *,
    duration_s: float,
    research: ResearchBrief | None = None,
    teams: dict[str, str] | None = None,
    max_turns: int = 12,
    max_lines: int = 7,
) -> dict[str, Any]:
    """Bounded turns and timing checks; no model-written words are edited."""
    if not math.isfinite(duration_s) or duration_s <= 0 or max_turns < 1 or max_lines < 1:
        raise ValueError("recorded duration and turn/line limits must be positive")
    ordered = sorted(observations, key=lambda item: item.at_s)
    if len({item.id for item in ordered}) != len(ordered):
        raise ValueError("recorded evidence IDs must be unique")
    if any(not 0 <= item.at_s <= ready_at(item) <= duration_s for item in ordered):
        raise ValueError("recorded evidence must fit the source duration")
    frontiers = sorted({ready_at(item) for item in ordered})
    lines: list[PlanLine] = []
    recent: list[SpeechLine] = []
    turns: list[dict[str, Any]] = []
    requested: set[tuple[str, tuple[str, ...]]] = set()
    cursor = frontiers[0] if frontiers else duration_s
    last_role: Role = "analyst"
    started = time.monotonic()

    def priority(item: Observation) -> bool:
        return item.kind == "goal" or bool(
            item.action
            and item.action.certainty == "confirmed"
            and item.action.type in {"penalty_awarded", "red_card"}
        )

    while len(turns) < max_turns:
        if len(lines) >= max_lines or cursor >= duration_s - 1:
            break
        visible = [item for item in ordered if ready_at(item) <= cursor]
        version = tuple(item.id for item in visible)
        covered = {identifier for line in lines for identifier in line.evidence_ids}
        urgent = any(priority(item) and item.id not in covered for item in visible)
        preferred: Role = "caller" if urgent or last_role == "analyst" else "analyst"
        alternatives: list[Role] = [preferred, "analyst" if preferred == "caller" else "caller"]
        caller_covered = {
            identifier
            for line in lines
            if line.voice == "caller"
            for identifier in line.evidence_ids
        }
        caller_has_new = any(
            item.id not in caller_covered
            and item.kind != "replay"
            and (item.action is None or item.action.certainty == "confirmed")
            for item in visible
        )
        role = next(
            (
                r
                for r in alternatives
                if (r, version) not in requested and (r != "caller" or caller_has_new)
            ),
            None,
        )
        if role is None:
            cursor = next((at for at in frontiers if at > cursor), duration_s)
            continue
        requested.add((role, version))
        # Reserve the next visible goal's slot during recorded preparation. Its
        # contents are never exposed in an earlier speaking agent's snapshot.
        next_goal = min(
            (ready_at(item) for item in ordered if priority(item) and ready_at(item) > cursor),
            default=duration_s,
        )
        available = min(duration_s, next_goal) - cursor - 0.35
        limit = min(18 if role == "caller" else 28, math.floor(available * 2.6))
        if limit < (1 if role == "caller" else 8):
            cursor = next((at for at in frontiers if at > cursor), duration_s)
            continue
        view = RuntimeSnapshot(
            version=len(visible),
            cutoff_s=cursor,
            observations=visible,
            recent_lines=recent,
            research=research,
            teams=teams or {},
            speech_budget_words=limit,
        )
        row: dict[str, Any] = {
            "role": role,
            "source_at_s": cursor,
            "word_budget": limit,
            "visible_evidence": list(version),
        }
        try:
            result = await run_role(role, view)
            row["result"] = result.model_dump(mode="json")
            if (
                result.role != role
                or result.snapshot_version != view.version
                or result.cutoff_s != view.cutoff_s
            ):
                raise ValueError("recorded turn has mismatched origin")
            if result.line is not None:
                if result.line.voice != role or len(result.line.text.split()) > limit:
                    raise ValueError("recorded turn has wrong voice or exceeds word budget")
                if not set(result.line.evidence_ids) <= set(version):
                    raise ValueError("recorded turn cites unavailable evidence")
                role_covered = {
                    identifier
                    for earlier in lines
                    if earlier.voice == role
                    for identifier in earlier.evidence_ids
                }
                if result.line.evidence_ids and set(result.line.evidence_ids) <= role_covered:
                    raise ValueError("speaker already covered this evidence")
                line = PlanLine(
                    id=f"line-{len(lines) + 1}",
                    at_s=round(cursor, 3),
                    voice=role,
                    text=result.line.text,
                    evidence_ids=result.line.evidence_ids,
                    research_ids=result.line.research_ids,
                )
                validate_plan_parts(ordered, [*lines, line], duration_s, research=research)
                lines.append(line)
                recent.append(result.line)
                # Let a major call land before asking the analyst to respond.
                # This is a preparation estimate; the renderer still measures audio.
                pause_s = 1.2 if role == "caller" and urgent else 0.35
                cursor += len(line.text.split()) / 2.6 + pause_s
                last_role = role
        except BudgetExceeded:
            row["rejected"] = "model budget exhausted"
            row["generation_elapsed_s"] = time.monotonic() - started
            turns.append(row)
            break
        except (ValueError, LLMError) as exc:
            row["rejected"] = str(exc)
        row["generation_elapsed_s"] = time.monotonic() - started
        turns.append(row)
    return {
        "lines": [line.model_dump(mode="json") for line in lines],
        "turns": turns,
        "preparation_duration_s": time.monotonic() - started,
        "source_duration_s": duration_s,
        "words": sum(len(line.text.split()) for line in lines),
        "timing": "Source-video seconds; word-count estimates until audio is measured.",
    }
