"""Record real model conversations for native human review in Trajectory.

Default is offline preview. --spend permits at most one 256-token call per case,
three cases total, no retries, tools, sandbox, grader model, or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "examples/trajectory-preferences"
RECEIPT = DIRECTORY / "native-runs.json"
MODEL = "openai/gpt-6-luna"
MAX_OUTPUT = 256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spend", action="store_true")
    parser.add_argument("--revision", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--case",
        choices=(
            "penalty-reaction",
            "late-save-call",
            "after-goal-reaction",
        ),
    )
    args = parser.parse_args()
    tasks = json.loads((DIRECTORY / "spec/tasks.json").read_text())
    tasks = [t for t in tasks if args.case is None or t["id"] == args.case]
    # A deliberately generous byte-based upper estimate for these short text prompts.
    max_input = [len(json.dumps(t["messages"]).encode()) + 2048 for t in tasks]
    if any(size > 16000 for size in max_input):
        raise ValueError("Prompt exceeds this small text-only experiment's budget")
    estimated_max_usd = sum(n * 0.13 + MAX_OUTPUT * 0.65 for n in max_input) / 1_000_000
    if estimated_max_usd > 0.05:
        raise ValueError("Planned model calls exceed the $0.05 experiment allowance")
    print(
        json.dumps(
            {
                "model": MODEL,
                "maximum_calls": len(tasks),
                "max_output_tokens_per_call": MAX_OUTPUT,
                "conservative_token_cost_estimate_usd": estimated_max_usd,
                "live": args.spend,
            }
        ),
        flush=True,
    )
    if not args.spend:
        return

    from trajectory import Client
    from trajectory.lib.platform_links import PlatformLinks

    state = json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}
    reviews = state.setdefault(
        "production_reviews" if args.revision == "v1" else "production_reviews_v2", {}
    )
    workspace = json.loads((DIRECTORY / "workspace.json").read_text())
    with Client(max_retries=0, timeout=45) as client:
        org = client.organizations.retrieve_current()
        if org["organization_key"] != workspace["organization_key"]:
            raise ValueError("Authenticated organization differs from the saved workspace")
        links = PlatformLinks(org["organization_key"])
        available = {model.id for model in client.inference.list_models().data}
        if MODEL not in available:
            raise ValueError("Selected model is not available for inference")
        for task in tasks:
            digest = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
            entry = reviews.get(task["id"])
            if entry and entry["prompt_sha256"] != digest:
                raise ValueError(
                    "Prompt changed; preserve the existing review and version the task"
                )
            if entry is None:
                tid = client.trajectories.create(
                    body={"agent_id": workspace["agent_id"]},
                    idempotency_key=(
                        f"commentary-live-review-{args.revision}-{task['id']}-{digest[:16]}"
                    ),
                ).tid
                entry = {
                    "trajectory_id": tid,
                    "prompt_sha256": digest,
                    "status": "created",
                    "agent_id": workspace["agent_id"],
                }
                reviews[task["id"]] = entry
                RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
            tid = entry["trajectory_id"]
            print(
                task["id"], links.production(workspace["agent_id"], trajectory_id=tid), flush=True
            )
            if entry["status"] == "created":
                entry["status"] = "call_submitted"
                RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
                try:
                    reply = client.chat.completions.create(
                        model=MODEL,
                        messages=task["messages"],
                        x_trajectory_id=tid,
                        idempotency_key=(
                            f"commentary-live-call-{args.revision}-{task['id']}-{digest[:16]}"
                        ),
                        max_tokens=MAX_OUTPUT,
                        temperature=0.5,
                        disable_thinking=True,
                        timeout=45,
                    )
                    entry["response"] = reply.model_dump(mode="json")
                    entry["status"] = "response_received"
                    RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
                except Exception:
                    entry["status"] = "call_failed_or_uncertain"
                    RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
                    raise
            if entry["status"] == "response_received":
                reason = entry["response"]["choices"][0]["finish_reason"]
                ending = (
                    "ENV_DONE"
                    if reason == "stop"
                    else ("TRUNCATION" if reason == "length" else "ERROR")
                )
                client.trajectories.complete(tid, termination_reason=ending)
                entry.update(status="completed", termination_reason=ending)
                RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
            if entry["status"] != "completed":
                raise RuntimeError("An earlier call is uncertain; inspect it before making another")
            trace = client.get(
                f"/api/v1/agents/{workspace['agent_id']}/traces/{tid}",
                cast_to=dict,
            )
            steps = client.trajectories.steps.list(tid).items
            entry["verified_steps"] = len(steps)
            entry["task_preview"] = trace.get("task_preview")
            RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "steps": len(steps),
                        "task_preview": (trace.get("task_preview") or "")[:100],
                        "human_feedback_present": trace.get("feedback_text") is not None,
                        "termination_reason": entry["termination_reason"],
                    }
                ),
                flush=True,
            )
            if not steps or not trace.get("task_preview"):
                raise RuntimeError("Conversation is not yet visible; inspect the saved trajectory")


if __name__ == "__main__":
    main()
