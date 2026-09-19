#!/usr/bin/env python3
"""Generate an auditable, offline plan for one recorded football clip.

The default is intentionally a dry run. Passing ``--spend`` explicitly
authorizes the bounded paid model calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from commentary.llm import openai_factory
from commentary.recorded_demo import (
    ObserverResult,
    RecordedDemoConfig,
    dry_run_estimate,
    run_recorded_demo,
    safe_pack,
    sample_clip,
    validate_inputs,
)
from commentary.recorded_research import ResearchBrief


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan an OFFLINE recorded-clip commentary demo")
    parser.add_argument("--clip", required=True, type=Path, help="local source video")
    parser.add_argument(
        "--pack", type=Path, help="optional JSON with teams, kit colours and player identities"
    )
    parser.add_argument("--start", type=float, default=0.0, dest="start_s")
    parser.add_argument("--duration", type=float, default=40.0, dest="duration_s")
    parser.add_argument("--out", type=Path, default=Path("runs/recorded-demo.json"))
    parser.add_argument("--observer-model", default="gpt-6-astra")
    parser.add_argument("--writer-model", default="gpt-5.6-terra")
    parser.add_argument("--research", type=Path, help="validated ResearchBrief JSON to reuse")
    parser.add_argument("--fixture", help="fixture label for one bounded fresh web-research call")
    parser.add_argument("--as-of", dest="research_as_of", help="fresh research cutoff, YYYY-MM-DD")
    parser.add_argument("--research-model", default="gpt-5.6-terra")
    parser.add_argument(
        "--reuse-observations",
        type=Path,
        help="saved observe.json; skips the paid vision call and reuses its evidence",
    )
    parser.add_argument("--spend", action="store_true", help="authorize paid model calls")
    return parser.parse_args()


def _load_pack(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"could not read --pack {path}: {exc}") from exc
    return safe_pack(value)


def _check_outputs(out: Path) -> Path:
    run_dir = out.with_suffix(out.suffix + ".run")
    if out.exists() or run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {out} or {run_dir}")
    if out.resolve() == Path.cwd().resolve():
        raise SystemExit("--out must be a file path")
    return run_dir


def _load_observations(path: Path | None) -> tuple[ObserverResult | None, float]:
    if path is None:
        return None, 0.0
    try:
        stage = json.loads(path.read_text())
        result = ObserverResult.model_validate(stage["result"])
        cost = float(stage.get("usage", {}).get("cost_usd", 0.0))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not validate --reuse-observations {path}: {exc}") from exc
    return result, cost


def _load_research(path: Path | None) -> ResearchBrief | None:
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text())
        return ResearchBrief.model_validate(raw.get("result", raw))
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"could not validate --research {path}: {exc}") from exc


async def _generate(
    args: argparse.Namespace,
    frames: list[Any],
    pack: dict[str, Any] | None,
    reused_observer: ObserverResult | None,
    prior_observation_cost: float,
    research: ResearchBrief | None,
) -> None:
    run_dir = _check_outputs(args.out)
    load_dotenv()
    backend = openai_factory(timeout_s=120.0, max_retries=0)
    config = RecordedDemoConfig(
        clip=args.clip,
        start_s=args.start_s,
        duration_s=args.duration_s,
        pack=pack,
        observer_model=args.observer_model,
        writer_model=args.writer_model,
        reused_observer=reused_observer,
        reuse_source=str(args.reuse_observations) if args.reuse_observations else None,
        prior_observation_cost=prior_observation_cost,
        research_brief=research,
        research_source=str(args.research) if args.research else None,
        research_fixture=args.fixture,
        research_as_of=args.research_as_of,
        research_model=args.research_model,
    )
    try:
        plan, _audit = await run_recorded_demo(backend, config, frames=frames, audit_dir=run_dir)
    except Exception as exc:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "failure.json").write_text(
            json.dumps({"error": str(exc), "usage": asdict(backend.total)}, indent=2) + "\n"
        )
        raise
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(plan.model_dump_json(indent=2) + "\n")
    print(f"OFFLINE recorded-clip plan written to {args.out}")
    print(f"Audit results and usage written to {run_dir}")
    print(f"Model cost: ${plan.metadata['cost']:.4f}")


def main() -> None:
    args = _args()
    validate_inputs(args.clip, args.start_s, args.duration_s)
    reused_observer, prior_observation_cost = _load_observations(args.reuse_observations)
    if args.research and args.fixture:
        raise SystemExit("choose --research to reuse or --fixture/--as-of for fresh research")
    if args.fixture and not args.research_as_of:
        raise SystemExit("--fixture requires --as-of YYYY-MM-DD")
    if args.research_as_of and not args.fixture:
        raise SystemExit("--as-of requires --fixture")
    if args.research_as_of:
        try:
            date.fromisoformat(args.research_as_of)
        except ValueError as exc:
            raise SystemExit(f"--as-of must be YYYY-MM-DD: {exc}") from exc
    research = _load_research(args.research)
    # Sampling itself is local and lets a dry run report the exact candidate count.
    frames = (
        [] if reused_observer is not None else sample_clip(args.clip, args.start_s, args.duration_s)
    )
    pack = _load_pack(args.pack)
    if not args.spend:
        print(
            json.dumps(
                dry_run_estimate(
                    len(frames),
                    fresh_research=args.fixture is not None,
                    reused_research=research is not None,
                ),
                indent=2,
            )
        )
        return
    asyncio.run(_generate(args, frames, pack, reused_observer, prior_observation_cost, research))


if __name__ == "__main__":
    main()
