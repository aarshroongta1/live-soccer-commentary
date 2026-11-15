#!/usr/bin/env python3
"""One paired retrieval run through the existing researcher and analyst.

Dry-run unless --spend. Reuse saved retrieval with --reuse-search. No retries.
This is a retrospective recorded case, not a claim about live search in 2025.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from commentary.football import FootballAction
from commentary.llm.base import text_block
from commentary.llm.budget import RunBudget
from commentary.llm.openai_backend import OpenAIBackend
from commentary.recorded_demo import Observation
from commentary.replay.agents import ANALYST_SYSTEM
from commentary.replay.contracts import SpeechLine
from commentary.replay.preparation import prepare_research
from commentary.search_providers import SearchRun, search_web

QUERY = (
    "Dan Ballard Sunderland Arsenal 8 November 2025 background former club "
    "Arsenal defensive record before match"
)


async def run(args: argparse.Namespace) -> None:
    if not args.spend:
        print(
            json.dumps(
                {
                    "query": QUERY,
                    "providers": ["keenable", "exa"],
                    "results_each": 5,
                    "model": args.model,
                    "model_max_usd": args.max_usd,
                    "max_model_calls": 10,
                    "search_calls": 0 if args.reuse_search else 2,
                    "note": "Pass --spend to run. No voice or automatic retries.",
                },
                indent=2,
            )
        )
        return
    if not 0 < args.max_usd <= 0.35:
        raise ValueError("this development experiment caps model spend at $0.35")
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    args.out.mkdir(parents=True, exist_ok=True)
    budget = RunBudget(max_usd=args.max_usd, max_calls=10)

    def capture(response: object, tag: str) -> None:
        path = args.out / f"response-{len(list(args.out.glob('response-*.json'))):02d}-{tag}.json"
        path.write_text(response.model_dump_json(indent=2))  # type: ignore[attr-defined]

    backend = OpenAIBackend(
        timeout_s=45,
        max_retries=0,
        budget=budget,
        response_callback=capture,
    )
    # A human-reviewed goal trigger from the official clip. No shirt/name identity
    # is fabricated, and search never supplies visual observations to the agent.
    goal = Observation(
        id="goal-1",
        at_s=4,
        kind="goal",
        description="A Sunderland attacker strikes the ball past the goalkeeper into the net.",
        action=FootballAction(
            type="goal",
            certainty="confirmed",
            cues=["ball_in_net"],
            support_at_s=[4],
        ),
    )
    report: dict = {
        "query": QUERY,
        "fixture": "Sunderland v Arsenal",
        "duration_s": 20,
        "mode": "recorded_retrospective",
        "analyst_handoff": (
            "Existing analyst system prompt with an eager research handoff and "
            "human-verified scorer. The live RoleAgents tool loop was not used."
        ),
        "model": args.model,
        "observation_origin": (
            "Human-reviewed goal in official Arsenal highlights; not model vision"
        ),
        "source_url": "https://www.youtube.com/watch?v=SXHoDl2twPg",
        "source_window_s": [0, 20],
        "baseline": "It's in for Sunderland!",
        "baseline_origin": "Editorial goal-only line, shared by both analyst turns",
        "featured_commentary": (
            "Eight clean sheets in all competitions, and it takes one of your own "
            "academy graduates to end the run."
        ),
        "featured_commentary_origin": (
            "Editorial broadcast rewrite grounded in both providers' returned evidence; "
            "exact generated commentary is retained separately for each provider."
        ),
        "providers": [],
    }
    try:
        for provider in ("keenable", "exa"):
            if args.reuse_search:
                search = SearchRun.model_validate_json(
                    (args.reuse_search / f"{provider}.json").read_text()
                )
                if search.query != QUERY or search.provider != provider:
                    raise ValueError("saved search does not match paired query/provider")
            else:
                search = await search_web(provider, QUERY)
            (args.out / f"{provider}-search.json").write_text(search.model_dump_json(indent=2))
            prepared = await prepare_research(
                backend,
                fixture="Sunderland v Arsenal",
                as_of=date(2025, 11, 8),
                model=args.model,
                search_run=search,
            )
            (args.out / f"{provider}-research.json").write_text(
                prepared.brief.model_dump_json(indent=2)
            )
            # Eager research handoff to the existing analyst prompt. Identity is
            # human-reviewed for this case, not an automated vision result.
            turn = await backend.parse(
                model=args.model,
                system=ANALYST_SYSTEM + "\nFor this retrospective recording the fixture "
                "event is HUMAN-VERIFIED, including the scorer; you may use that name. "
                "Research has already been fetched below, so return a SpeechLine now, "
                "not a tool action. Use 18–26 spoken words that connect relevant player "
                "history and team form to the moment. Cite supplied search result IDs "
                "in research_ids and goal-1 in evidence_ids. Preserve competition scope. "
                "Avoid a literal recap of the shot. Use only the supplied sources.",
                blocks=[
                    text_block(
                        json.dumps(
                            {
                                "human_verified_fixture_event": (
                                    "Dan Ballard scores for Sunderland against Arsenal"
                                ),
                                "caller": report["baseline"],
                                "research_brief": prepared.brief.model_dump(mode="json"),
                                "search_evidence": [
                                    {"id": f"result-{hit.rank}", **hit.model_dump()}
                                    for hit in search.results
                                    if hit.published_at and hit.published_at[:10] <= "2025-11-08"
                                ],
                            }
                        )
                    )
                ],
                output_format=SpeechLine,
                max_tokens=512,
                tag="outreach_analyst",
            )
            available = {
                f"result-{hit.rank}"
                for hit in search.results
                if hit.published_at and hit.published_at[:10] <= "2025-11-08"
            }
            available.update(fact.id for fact in prepared.brief.facts)
            if (
                turn.value.voice != "analyst"
                or not turn.value.research_ids
                or not set(turn.value.research_ids) <= available
                or set(turn.value.evidence_ids) != {goal.id}
            ):
                raise ValueError("analyst must cite the supplied goal and search evidence")
            report["providers"].append(
                {
                    "provider": provider,
                    "elapsed_ms": search.elapsed_ms,
                    "request": search.request,
                    "results": [hit.model_dump() for hit in search.results],
                    "facts": [fact.model_dump(mode="json") for fact in prepared.brief.facts],
                    "research_audit": prepared.audit,
                    "commentary": turn.value.text,
                    "research_ids": turn.value.research_ids,
                    "turn": turn.value.model_dump(mode="json"),
                    "analyst_usage": turn.usage.__dict__,
                }
            )
        keen, exa = report["providers"]
        report["latency_ratio"] = round(exa["elapsed_ms"] / keen["elapsed_ms"], 2)
        report["comparison_scope"] = (
            "One query, one request per provider, five results each. Keenable default pro "
            "snippets; Exa auto highlights plus objective. Same 2500-character per-hit input "
            "cap, researcher, analyst, goal observation and caller line. Unblinded, "
            "non-deterministic generation. End-to-end client latency including connection "
            "setup, measured once; not a benchmark. No general quality claim."
        )
        clip = Path("clips/demo-sources/ballard-arsenal.mp4")
        if clip.exists():
            report["source_sha256"] = hashlib.sha256(clip.read_bytes()).hexdigest()
    finally:
        report["budget"] = budget.report()
        (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    if args.web_out:
        args.web_out.parent.mkdir(parents=True, exist_ok=True)
        args.web_out.write_text(json.dumps(report, indent=2) + "\n")
        public_trace = (
            Path(__file__).resolve().parents[1]
            / "web/public/soccer-commentary/search-comparison.json"
        )
        public_trace.parent.mkdir(parents=True, exist_ok=True)
        public_trace.write_text(args.web_out.read_text())
    print(
        json.dumps(
            {
                "providers": [
                    {k: p[k] for k in ("provider", "elapsed_ms", "commentary")}
                    for p in report["providers"]
                ],
                "budget": budget.report(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spend", action="store_true")
    parser.add_argument("--reuse-search", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/outreach/paired"))
    parser.add_argument("--web-out", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-usd", type=float, default=0.35)
    asyncio.run(run(parser.parse_args()))
