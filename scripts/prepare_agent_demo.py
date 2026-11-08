#!/usr/bin/env python3
"""Prepare recorded multi-agent dialogue on video time. Dry-run unless --spend."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from commentary.llm.budget import RunBudget
from commentary.llm.openai_backend import OpenAIBackend
from commentary.recorded_demo import Observation, sample_clip, validate_inputs
from commentary.recorded_research import ResearchBrief
from commentary.replay.agents import RoleAgents
from commentary.replay.preparation import team_names
from commentary.replay.recorded import (
    observe_recorded,
    prepare_dialogue,
    ready_at,
    resolve_recorded_identities,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--teams", type=Path, required=True, help="fixture names and kit colours")
    parser.add_argument("--reuse-vision", type=Path, help="saved preparation with identical source")
    parser.add_argument("--observer-model", default="gpt-5.4-mini")
    parser.add_argument("--speaker-model", default="gpt-5.4-mini")
    parser.add_argument("--max-usd", type=float, default=0.20)
    parser.add_argument("--max-model-calls", type=int, default=24)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spend", action="store_true")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_inputs(args.clip, args.start, args.duration)
    if args.duration > 30:
        raise ValueError("recorded preparation supports at most 30 seconds per pass")
    if args.out_dir.exists():
        raise ValueError("refusing to overwrite an existing preparation")
    research = ResearchBrief.model_validate_json(args.research.read_text())
    teams = json.loads(args.teams.read_text())
    source = dict(
        path=str(args.clip),
        start_s=args.start,
        duration_s=args.duration,
        sha256=hashlib.sha256(args.clip.read_bytes()).hexdigest(),
    )
    cached = json.loads(args.reuse_vision.read_text()) if args.reuse_vision else None
    if cached:
        old = cached["source"]
        if (
            Path(old["path"]).resolve() != args.clip.resolve()
            or old["sha256"] != source["sha256"]
            or old["start_s"] != args.start
            or old["duration_s"] < args.duration
        ):
            raise ValueError("cached vision source/window does not match")
    budget = RunBudget(max_usd=args.max_usd, max_calls=args.max_model_calls)
    report: dict[str, Any] = dict(
        source=source,
        research=research.model_dump(mode="json"),
        budget=budget.report(),
        mode="recorded_preparation",
    )
    if not args.spend:
        return {
            **report,
            "dry_run": True,
            "note": "No model or voice calls. Add --spend to prepare.",
        }
    load_dotenv()
    args.out_dir.mkdir(parents=True)
    responses = 0

    def capture(response: Any, tag: str) -> None:
        nonlocal responses
        responses += 1
        data = response.model_dump(mode="json")
        data = {
            k: data.get(k)
            for k in ["id", "model", "status", "usage", "output", "incomplete_details"]
        }
        data["tag"] = tag
        data["output"] = [item for item in data["output"] or [] if item.get("type") != "reasoning"]
        (args.out_dir / f"response-{responses}.json").write_text(json.dumps(data, indent=2) + "\n")

    backend = OpenAIBackend(
        timeout_s=60, max_retries=0, budget=budget, default_effort="none", response_callback=capture
    )
    started = time.monotonic()
    try:
        if cached:
            raw = [Observation.model_validate(row) for row in cached["raw_observations"]]
            raw = [item for item in raw if ready_at(item) < args.duration]
            resolved, steps, usage = await resolve_recorded_identities(
                backend, raw, model=args.speaker_model, research=research, pack=teams
            )
            observed = dict(
                observations=[item.model_dump(mode="json") for item in resolved],
                raw_observations=[item.model_dump(mode="json") for item in raw],
                identity_steps=steps,
                identity_usage=usage.__dict__,
                vision_reuse=str(args.reuse_vision),
            )
        else:
            observed = await observe_recorded(
                backend,
                sample_clip(args.clip, args.start, args.duration, fps=1),
                model=args.observer_model,
                identity_model=args.speaker_model,
                research=research,
                pack=teams,
            )
        report.update(observed)
        (args.out_dir / "observations.json").write_text(json.dumps(report, indent=2) + "\n")
        agents = RoleAgents(
            backend, args.speaker_model, args.speaker_model, max_steps=1, max_revisions=0
        )
        report["dialogue"] = await prepare_dialogue(
            agents.run,
            [Observation.model_validate(row) for row in observed["observations"]],
            duration_s=args.duration,
            research=research,
            teams=team_names(teams, research),
        )
        report["role_audit"] = [row.model_dump(mode="json") for row in agents.audit]
        plan = dict(
            version=1,
            mode="recorded_clip",
            source=dict(basename=args.clip.name, start_s=args.start, duration_s=args.duration),
            observations=observed["observations"],
            lines=report["dialogue"]["lines"],
            metadata=dict(
                offline=True,
                trace=str(args.out_dir / "result.json"),
                cost=budget.report()["reported_usd"],
                models=dict(
                    observer=args.observer_model,
                    caller=args.speaker_model,
                    analyst=args.speaker_model,
                ),
                research=research.model_dump(mode="json"),
                words_edited=False,
            ),
        )
        (args.out_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    except Exception as exc:
        report["error"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        report.update(budget=budget.report(), generation_elapsed_s=time.monotonic() - started)
        (args.out_dir / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = asyncio.run(run(arguments()))
    print(
        json.dumps(
            result
            if result.get("dry_run")
            else {"lines": len(result["dialogue"]["lines"]), "budget": result["budget"]},
            indent=2,
        )
    )
