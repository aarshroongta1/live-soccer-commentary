#!/usr/bin/env python3
"""Compare text writers on saved perception. Default: local inventory, zero paid calls.

The input manifest is a list of {"case_id": "...", "bundle": "path/to/bundle.json"}.
Bundle paths resolve relative to that manifest. Bundle schema is TextComparisonInput;
references and expected words belong only to the evaluator, never to the bundle.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from commentary.llm.budget import RunBudget
from commentary.replay.evaluation import EvaluationCase
from commentary.replay.text_comparison import (
    VARIANT,
    AuditedTextBackend,
    TextComparisonInput,
    digest,
    export_blind,
    run_arm,
)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def inventory(inputs: Path, suite_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(inputs.read_text())
    if not isinstance(manifest, list):
        raise ValueError("inputs must be a list of case_id/bundle objects")
    suite = json.loads(suite_path.read_text())
    if len(suite["clips"]) != 15 or len({c["id"] for c in suite["clips"]}) != 15:
        raise ValueError("comparison requires the fixed 15-case suite")
    selected = {c["id"]: c for c in suite["clips"]}
    bundles: dict[str, Any] = {}
    for row in manifest:
        if not isinstance(row, dict) or set(row) != {"case_id", "bundle"}:
            raise ValueError("input entries accept only case_id and bundle")
        case_id = row["case_id"]
        if case_id not in selected or case_id in bundles:
            raise ValueError("unknown or duplicate case ID")
        bundle_path = inputs.parent / row["bundle"]
        bundle = TextComparisonInput.model_validate_json(bundle_path.read_text()).sanitized()
        # Labels are used here for alignment only; run_arm never receives a reference.
        reference = EvaluationCase.model_validate_json(
            Path(selected[case_id]["reference"]).read_text()
        )
        if (
            bundle.source_sha256 != reference.source.sha256
            or bundle.source_start_s != reference.source.start_s
            or bundle.duration_s != reference.duration_s
        ):
            raise ValueError(f"source/window mismatch for {case_id}")
        source = Path(reference.source.locator)
        if hashlib.sha256(source.read_bytes()).hexdigest() != bundle.source_sha256:
            raise ValueError(f"local source hash mismatch for {case_id}")
        bundles[case_id] = bundle
    rows = [
        {
            "case_id": c["id"],
            "status": "ready" if c["id"] in bundles else "missing_input",
            "input_sha256": digest(bundles[c["id"]].model_dump(mode="json"))
            if c["id"] in bundles
            else None,
        }
        for c in suite["clips"]
    ]
    return rows, bundles


async def run(args: argparse.Namespace) -> dict[str, Any]:
    rows, bundles = inventory(args.inputs, args.suite)
    if args.spend and not args.model:
        raise ValueError("paid execution requires an explicit --model")
    per_arm_usd = getattr(args, "per_arm_max_usd", 0.03)
    if not 0 < per_arm_usd <= 0.03:
        raise ValueError("per-arm budget must be positive and at most $0.03")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running",
        "protocol": VARIANT,
        "mode": "offline_full_context",
        "paid": args.spend,
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "inputs_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "model": args.model,
        "effort": "none",
        "max_output_tokens": args.max_tokens,
        "per_arm_max_usd": per_arm_usd,
        "per_arm_max_calls": 8,
        "maximum_usd": len(bundles) * 2 * per_arm_usd,
        "inventory": rows,
        "pairs": [],
        "quality_review": "pending",
        "voice_in_scope": False,
    }
    write_json(args.out_dir / "report.json", report)
    if not args.spend:
        report["status"] = "dry_run"
        write_json(args.out_dir / "report.json", report)
        return report
    # No credentials, model client, research or vision calls are used during dry runs.
    import openai
    from dotenv import load_dotenv

    from commentary.llm.openai_backend import OpenAIBackend

    load_dotenv()
    try:
        for row in rows:
            case_id = row["case_id"]
            pair: dict[str, Any] = {"case_id": case_id, "arms": []}
            report["pairs"].append(pair)
            for arm in ("joint", "agents"):
                if case_id not in bundles:
                    pair["arms"].append({"arm": arm, "status": "missing_input", "lines": []})
                    continue
                arm_dir = args.out_dir / case_id / arm
                arm_dir.mkdir(parents=True)
                budget = RunBudget(max_usd=per_arm_usd, max_calls=8)
                raw_count = 0

                def capture(response: Any, tag: str, target: Path = arm_dir) -> None:
                    nonlocal raw_count
                    raw_count += 1
                    write_json(
                        target / f"provider-{raw_count:02}.json",
                        {
                            "tag": tag,
                            "response": response.model_dump(mode="json"),
                        },
                    )

                client = openai.AsyncOpenAI(timeout=45, max_retries=0)
                provider = OpenAIBackend(
                    client=client,
                    timeout_s=45,
                    max_retries=0,
                    budget=budget,
                    default_effort="none",
                    response_callback=capture,
                )
                adapter = AuditedTextBackend(
                    provider,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    emit=lambda event, target=arm_dir: write_json(
                        target / "latest-call.json", event
                    ),
                )
                try:
                    outcome = await run_arm(bundles[case_id], adapter, arm)
                    outcome["budget"] = budget.report()
                    write_json(arm_dir / "result.json", outcome)
                    pair["arms"].append(outcome)
                    write_json(args.out_dir / "report.json", report)
                finally:
                    await client.close()
        report["status"] = (
            "completed_with_failures"
            if any(arm["status"] != "complete" for pair in report["pairs"] for arm in pair["arms"])
            else "generated_pending_review"
        )
        report["reported_usd"] = sum(
            arm.get("budget", {}).get("reported_usd", 0)
            for pair in report["pairs"]
            for arm in pair["arms"]
        )
        export_blind(report["pairs"], args.out_dir / "blind", seed=args.seed)
        return report
    finally:
        if report["status"] == "running":
            report["status"] = "interrupted"
        report["unsettled_reservations_usd"] = sum(
            arm.get("budget", {}).get("unsettled_reservations_usd", 0)
            for pair in report["pairs"]
            for arm in pair["arms"]
        )
        write_json(args.out_dir / "report.json", report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--suite", type=Path, default=Path("examples/action-suite.json"))
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", help="explicit existing model ID; required with --spend")
    parser.add_argument(
        "--max-tokens", type=int, default=1024, help="128–2048; same cap per call in both arms"
    )
    parser.add_argument("--per-arm-max-usd", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--spend", action="store_true")
    args = parser.parse_args()
    if not 128 <= args.max_tokens <= 2048:
        parser.error("--max-tokens must be between 128 and 2048")
    result = asyncio.run(run(args))
    ready = sum(row["status"] == "ready" for row in result["inventory"])
    print(f"{result['status']}: {ready}/15 inputs ready")


if __name__ == "__main__":
    main()
