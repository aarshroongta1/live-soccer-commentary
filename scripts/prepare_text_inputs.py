#!/usr/bin/env python3
"""Reuse saved comparison evidence and optionally fill missing video perception.

Default is a free inventory. --spend authorizes one vision attempt per missing
case, within a single dollar/call budget. Labels never enter the model prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from compare_text import inventory, write_json

from commentary.llm.budget import RunBudget
from commentary.recorded_demo import sample_clip
from commentary.replay.recorded import observe_recorded
from commentary.replay.text_comparison import TextComparisonInput


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows, bundles = inventory(args.inputs, args.suite)
    if not 0 < args.max_usd <= 1:
        raise ValueError("preparation requires an explicit positive budget at most $1")
    suite = json.loads(args.suite.read_text())
    missing = [row for row in rows if row["case_id"] not in bundles]
    budget = RunBudget(max_usd=args.max_usd, max_calls=max(1, len(missing)))
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "inputs").mkdir()
    report: dict[str, Any] = {
        "status": "running",
        "model": args.model,
        "paid": args.spend,
        "max_usd": args.max_usd,
        "max_calls": len(missing),
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "cases": [],
        "runtime_audio": False,
        "research_calls": False,
        "scope": "One full-context vision pass per missing clip; "
        "no new identities without saved squads.",
    }
    manifest = []
    try:
        for selected in suite["clips"]:
            case_id = selected["id"]
            entry: dict[str, Any] = {
                "case_id": case_id,
                "status": "reused" if case_id in bundles else "missing",
            }
            report["cases"].append(entry)
            if case_id not in bundles and args.spend:
                # This boundary selects only source path/window/hash. Reference events,
                # case titles, manual checks and labels are never given to observe_recorded.
                ref = json.loads(Path(selected["reference"]).read_text())
                source = ref["source"]
                clip = Path(source["locator"])
                source_hash = hashlib.sha256(clip.read_bytes()).hexdigest()
                if source_hash != source["sha256"]:
                    raise ValueError(f"source hash mismatch for {case_id}")
                case_dir = args.out_dir / case_id
                case_dir.mkdir()
                import openai
                from dotenv import load_dotenv

                from commentary.llm.openai_backend import OpenAIBackend

                load_dotenv()
                response_index = 0

                def capture(response: Any, tag: str, target: Path = case_dir) -> None:
                    nonlocal response_index
                    response_index += 1
                    write_json(
                        target / f"provider-{response_index:02}.json",
                        {
                            "tag": tag,
                            "response": response.model_dump(mode="json"),
                        },
                    )

                client = openai.AsyncOpenAI(timeout=45, max_retries=0)
                try:
                    backend = OpenAIBackend(
                        client=client,
                        budget=budget,
                        default_effort="none",
                        response_callback=capture,
                    )
                    observed = await observe_recorded(
                        backend,
                        sample_clip(clip, source["start_s"], ref["duration_s"], fps=1),
                        model=args.model,
                        identity_model=args.model,
                        research=None,
                        pack=None,
                    )
                    write_json(case_dir / "observation-audit.json", observed)
                    bundles[case_id] = TextComparisonInput(
                        source_sha256=source_hash,
                        source_start_s=source["start_s"],
                        duration_s=ref["duration_s"],
                        observations=observed["observations"],
                    ).sanitized()
                    entry.update(status="generated", observations=len(observed["observations"]))
                except Exception as exc:
                    entry.update(
                        status="failed", error={"type": type(exc).__name__, "message": str(exc)}
                    )
                finally:
                    await client.close()
            if case_id in bundles:
                path = args.out_dir / "inputs" / f"{case_id}.json"
                path.write_text(bundles[case_id].model_dump_json(indent=2) + "\n")
                manifest.append({"case_id": case_id, "bundle": f"inputs/{case_id}.json"})
                entry["bundle_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report["budget"] = budget.report()
            write_json(args.out_dir / "inputs.json", manifest)
            write_json(args.out_dir / "preparation.json", report)
        report["status"] = "complete" if len(bundles) == 15 else "incomplete"
        return report
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        report["budget"] = budget.report()
        write_json(args.out_dir / "preparation.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=Path("examples/action-suite.json"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max-usd", type=float, default=0.24)
    parser.add_argument("--spend", action="store_true")
    result = asyncio.run(run(parser.parse_args()))
    print(f"{result['status']}; budget: {result['budget']}")


if __name__ == "__main__":
    main()
