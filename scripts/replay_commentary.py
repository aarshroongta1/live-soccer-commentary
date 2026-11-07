#!/usr/bin/env python3
"""Run causal, two-agent replay; the default real-video mode is a dry run.

The diagnostic uses fictional observations and a scripted model policy. It tests
control flow, not visual understanding or commentary quality. Speech playback is
simulated in both modes; no ElevenLabs voice or public demo is changed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from commentary.llm import LLMBackend, ScriptedBackend, openai_factory
from commentary.llm.base import Block
from commentary.llm.budget import RunBudget
from commentary.recorded_demo import Frame, Observation, sample_clip, validate_inputs
from commentary.recorded_research import ResearchBrief
from commentary.replay.agents import ANALYST_SYSTEM, CALLER_SYSTEM, RoleAgents
from commentary.replay.cache import SavedVision
from commentary.replay.contracts import AgentAction, ReplayResult, Role, SpeechLine, ToolCall
from commentary.replay.preparation import ReplayObserver, prepare_research, team_names
from commentary.replay.runtime import ReplayConfig, ReplayRuntime


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--diagnostic", action="store_true", help="offline synthetic control-flow test"
    )
    mode.add_argument("--clip", type=Path, help="local video; requires --spend for model calls")
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=28)
    parser.add_argument(
        "--fps", type=float, default=1, help="sampled frames per second (0 < fps <= 2)"
    )
    parser.add_argument("--rate", type=float, default=1, help="replay seconds per wall second")
    parser.add_argument("--batch-seconds", type=float, default=2)
    parser.add_argument("--turn-timeout", type=float, default=8, help="replay seconds")
    parser.add_argument("--observer-timeout", type=float, default=8, help="replay seconds")
    parser.add_argument("--max-age", type=float, default=8, help="replay seconds")
    parser.add_argument("--drain", type=float, default=10, help="replay seconds after final frame")
    parser.add_argument(
        "--no-fast-goals",
        action="store_false",
        dest="fast_goal_calls",
        help="disable the immediate deterministic goal reaction",
    )
    parser.add_argument("--pack", type=Path, help="optional team names and pre-match roster")
    parser.add_argument(
        "--reuse-vision", type=Path, help="saved replay with matching media/settings"
    )
    parser.add_argument("--research", type=Path, help="saved cited pre-match ResearchBrief")
    parser.add_argument(
        "--fixture", help="fixture for fresh web research or saved-brief validation"
    )
    parser.add_argument("--as-of", type=date.fromisoformat, help="pre-match research cutoff")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--observer-model")
    parser.add_argument("--caller-model")
    parser.add_argument("--analyst-model")
    parser.add_argument("--research-model")
    parser.add_argument("--reviewer-model", help="optional bounded reviewer inside role deadline")
    parser.add_argument(
        "--inspect-first",
        action="store_true",
        help="choose a bounded visual crop before classifying each batch; adds a model call",
    )
    parser.add_argument("--spend", action="store_true", help="enable paid model calls")
    parser.add_argument("--max-usd", type=float, default=0.10, help="OpenAI token budget")
    parser.add_argument("--max-model-calls", type=int, default=40, help="OpenAI request limit")
    parser.add_argument("--out", type=Path, default=Path("runs/replay/result.json"))
    args = parser.parse_args()
    default_model = "claude-haiku-4-5" if args.provider == "anthropic" else "gpt-6-luna"
    for field in ("observer_model", "caller_model", "analyst_model", "research_model"):
        if getattr(args, field) is None:
            selected = (
                "gpt-5.4-mini"
                if field == "observer_model" and args.provider == "openai"
                else default_model
            )
            setattr(args, field, selected)
    return args


def diagnostic_backend() -> ScriptedBackend:
    """A deterministic policy that reads only the role's actual supplied context."""
    backend = ScriptedBackend()

    def handler(role: Role) -> Any:
        def choose(blocks: list[Block], _schema: type[BaseModel]) -> AgentAction:
            payload = json.loads(blocks[0]["text"])
            if not payload["tool_results"]:
                return AgentAction(kind="tool", tool=ToolCall(name="recent_events"))
            observations = payload["observations"]
            if not observations:
                return AgentAction(kind="abstain", reason="No visible action yet")
            newest = observations[-1]
            if any(
                line["voice"] == role and newest["id"] in line["evidence_ids"]
                for line in payload["recent_lines"]
            ):
                return AgentAction(kind="abstain", reason="Already covered this event")
            words = {
                "caller": {
                    "play": "Green move the ball forward into space.",
                    "goal": "Green have scored! The ball is in the net!",
                    "celebration": "Green's players come together to celebrate.",
                },
                "analyst": {
                    "play": "That forward pass moves the attack upfield and gives Green room.",
                    "goal": (
                        "The forward pass started that move, "
                        "and Green finish the attack with a goal."
                    ),
                    "celebration": (
                        "Green can enjoy that finish now. Their players are celebrating together."
                    ),
                },
            }
            return AgentAction(
                kind="speak",
                line=SpeechLine(
                    voice=role,
                    text=words[role].get(newest["kind"], "The action continues."),
                    evidence_ids=[newest["id"]],
                ),
            )

        return choose

    backend.register("replay_caller", handler("caller"))
    backend.register("replay_analyst", handler("analyst"))
    return backend


async def diagnostic_observer(frames: list[Frame], cutoff_s: float) -> list[Observation]:
    """Synthetic evidence oracle, explicitly not a vision model."""
    events = [
        Observation(id="pass", at_s=2, kind="play", description="Green pass forward into space."),
        Observation(
            id="goal", at_s=6, kind="goal", description="The ball is in the net for Green."
        ),
        Observation(
            id="celebrate",
            at_s=9,
            kind="celebration",
            description="Green players celebrate together.",
        ),
    ]
    return [event for event in events if frames[0].at_s <= event.at_s <= cutoff_s]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


async def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.out}")
    config = ReplayConfig(
        batch_s=args.batch_seconds,
        rate=args.rate,
        turn_timeout_s=args.turn_timeout,
        observer_timeout_s=args.observer_timeout,
        max_age_s=args.max_age,
        drain_s=args.drain,
        fast_goal_calls=args.fast_goal_calls,
    )
    if not math.isfinite(args.fps) or not 0 < args.fps <= 2:
        raise ValueError("--fps must be finite and greater than zero, at most 2")
    budget = RunBudget(max_usd=args.max_usd, max_calls=args.max_model_calls)
    pack = _load_json(args.pack) if args.pack else None
    saved = ResearchBrief.model_validate(_load_json(args.research)) if args.research else None
    if (args.fixture is None) != (args.as_of is None):
        raise ValueError("research requires --fixture and --as-of together")
    if saved is not None and args.fixture is None:
        raise ValueError("saved research requires --fixture and --as-of for cutoff validation")
    if args.provider == "openai" and args.fixture and saved is None:
        raise ValueError(
            "bounded replay uses saved research; pass --research with a cited pre-match brief"
        )
    if args.diagnostic and any(
        [args.spend, args.pack, args.research, args.fixture, args.reviewer_model, args.reuse_vision]
    ):
        raise ValueError("diagnostic mode uses only fictional evidence and scripted roles")
    source_hash = None
    duration = 12.0 if args.diagnostic else args.duration
    if args.diagnostic:
        frames = [Frame(at_s=float(index), jpeg=b"synthetic") for index in range(13)]
    else:
        validate_inputs(args.clip, args.start, duration)
        if not args.spend:
            return {
                "mode": "dry_run",
                "clip": str(args.clip),
                "duration_s": duration,
                "sample_fps": args.fps,
                "paid_calls_enabled": False,
                "budget": budget.report() if args.provider == "openai" else None,
                "speech_simulated": True,
                "note": "Pass --spend to run the observer and independent role agents.",
            }
        with args.clip.open("rb") as media:
            source_hash = hashlib.file_digest(media, "sha256").hexdigest()
        frames = sample_clip(args.clip, args.start, duration, fps=args.fps)
    backend: LLMBackend
    observer: Any
    research = None
    research_audit: dict[str, Any] = {"source": "disabled"}
    reused: ReplayResult | None = None
    cached_observer: SavedVision | None = None
    if args.reuse_vision:
        reused = ReplayResult.model_validate_json(args.reuse_vision.read_text())
        old = reused.metadata
        expected_pack = hashlib.sha256(args.pack.read_bytes()).hexdigest() if args.pack else None
        if (
            Path(old["source_clip"]).resolve() != args.clip.resolve()
            or (old.get("source_sha256") is not None and old["source_sha256"] != source_hash)
            or old["source_start_s"] != args.start
            or old["duration_s"] != duration
            or old.get("sample_fps") != args.fps
            or old.get("pack_sha256") != expected_pack
            or old["config"]["batch_s"] != config.batch_s
        ):
            raise ValueError(
                "saved vision source, frame rate, batches, duration, or pack do not match"
            )
        cached_observer = SavedVision(
            reused.metadata["observer_audit"],
            rate=args.rate,
            source_rate=float(reused.metadata["rate"]),
        )
    if args.diagnostic:
        backend = diagnostic_backend()
        observer = diagnostic_observer
    else:
        load_dotenv()
        provider_timeout = max(args.turn_timeout, args.observer_timeout) / args.rate
        if args.fixture and saved is None:
            provider_timeout = max(provider_timeout, 90)
        if args.provider == "openai":
            backend = openai_factory(
                timeout_s=provider_timeout, max_retries=0, budget=budget, default_effort="none"
            )
        else:
            from commentary.llm.anthropic_backend import AnthropicBackend

            backend = AnthropicBackend(timeout_s=provider_timeout, max_retries=0)
        if args.fixture:
            async with asyncio.timeout(90):
                prepared = await prepare_research(
                    backend,
                    fixture=args.fixture,
                    as_of=args.as_of,
                    model=args.research_model,
                    saved=saved,
                )
            research, research_audit = prepared.brief, prepared.audit
        observer = cached_observer or ReplayObserver(
            backend, model=args.observer_model, pack=pack, inspection_first=args.inspect_first
        )
    roles = RoleAgents(
        backend,
        caller_model=args.caller_model,
        analyst_model=args.analyst_model,
        reviewer_model=args.reviewer_model,
    )
    runtime = ReplayRuntime(
        observe=observer,
        run_role=roles.run,
        config=config,
        research=research,
        teams={"home": "Green", "away": "Orange"} if args.diagnostic else team_names(pack),
    )
    result = await runtime.run(frames, duration_s=duration)
    result.metadata.update(
        {
            "provenance": "synthetic diagnostic" if args.diagnostic else "model-generated replay",
            "source_clip": None if args.diagnostic else str(args.clip),
            "source_start_s": 0 if args.diagnostic else args.start,
            "source_sha256": reused.metadata.get("source_sha256") if reused else source_hash,
            "sample_fps": args.fps if not args.diagnostic else None,
            "observer_inspection_first": (
                reused.metadata.get("observer_inspection_first") if reused else args.inspect_first
            ),
            "speech_simulated": True,
            "config": asdict(config),
            "provider": "scripted" if args.diagnostic else args.provider,
            "pack_source": str(args.pack) if args.pack else None,
            "pack_sha256": hashlib.sha256(args.pack.read_bytes()).hexdigest()
            if args.pack
            else None,
            "role_prompt_sha256": {
                "caller": hashlib.sha256(CALLER_SYSTEM.encode()).hexdigest(),
                "analyst": hashlib.sha256(ANALYST_SYSTEM.encode()).hexdigest(),
            },
            "models": {
                "observer": "synthetic oracle"
                if args.diagnostic
                else (
                    "saved:" + reused.metadata["models"]["observer"]
                    if reused
                    else args.observer_model
                ),
                "caller": "scripted policy" if args.diagnostic else args.caller_model,
                "analyst": "scripted policy" if args.diagnostic else args.analyst_model,
            },
            "research": research.model_dump(mode="json") if research else None,
            "research_audit": research_audit,
            "observer_audit": observer.audit
            if isinstance(observer, (ReplayObserver, SavedVision))
            else [],
            "vision_reuse": {
                "source": str(args.reuse_vision),
                "sha256": hashlib.sha256(args.reuse_vision.read_bytes()).hexdigest(),
                "media_contents_verified": reused.metadata.get("source_sha256") is not None,
            }
            if args.reuse_vision
            else None,
            "role_audit": [event.model_dump(mode="json") for event in roles.audit],
            "usage": asdict(backend.total),
            "spend_guard": budget.report()
            if not args.diagnostic and args.provider == "openai"
            else None,
        }
    )
    return result.model_dump(mode="json")


def main() -> None:
    args = arguments()
    try:
        output = asyncio.run(run(args))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    if output["mode"] == "dry_run":
        print(json.dumps(output, indent=2))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    speech = output["speech"]
    print(f"Replay saved to {args.out}: {len(speech)} simulated speech turns")
    print(f"Reported model token cost: ${output['metadata']['usage']['cost_usd']:.6f}")
    for record in speech:
        print(f"{record['started_s']:.2f}s {record['line']['voice']}: {record['line']['text']}")
    failures = sum(
        event["event"]
        in {
            "observer_error",
            "turn_error",
            "observer_timeout",
            "turn_timeout",
            "turn_rejected",
            "observation_rejected",
            "budget_exhausted",
        }
        for event in output["trace"]
    )
    if not speech and failures:
        raise SystemExit(
            f"No speech was produced and {failures} stage failures were recorded. "
            f"Inspect the saved trace at {args.out}."
        )


if __name__ == "__main__":
    main()
