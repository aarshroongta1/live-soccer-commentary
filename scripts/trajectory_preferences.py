"""Store and read commentary review tasks in the actual Trajectory product.

No chat, evaluation, grading-model, training, or deployment API is called.
Run with the Python belonging to `uv tool install trajectory-sdk` or another
environment with trajectory-sdk installed. Credentials come from `tj login`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "examples/trajectory-preferences"
RECEIPT = DIRECTORY / "workspace.json"
NAME = "soccer-commentary-editor"


def cases() -> list[dict]:
    return [
        json.loads((DIRECTORY / f"{name}.json").read_text())
        for name in (
            "penalty-reaction",
            "late-save-call",
            "after-goal-reaction",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--upload", action="store_true", help="Store review tasks; no model calls")
    actions.add_argument("--read-feedback", action="store_true", help="Read actual native reviews")
    parser.add_argument("--case", choices=[c["id"] for c in cases()])
    args = parser.parse_args()
    selected = [c for c in cases() if args.case is None or c["id"] == args.case]
    if not (args.upload or args.read_feedback):
        print(
            json.dumps({"mode": "offline_preview", "model_calls": 0, "tasks": selected}, indent=2)
        )
        return

    from trajectory import Client
    from trajectory.lib.platform_links import PlatformLinks

    with Client(max_retries=0, timeout=20) as client:
        org = client.organizations.retrieve_current()
        organization_key = org["organization_key"]
        links = PlatformLinks(organization_key)
        state = (
            json.loads(RECEIPT.read_text())
            if RECEIPT.exists()
            else {
            "organization_key": organization_key,
                "cases": {},
            }
        )
        if state["organization_key"] != organization_key:
            raise ValueError("Saved workspace belongs to a different organization")
        if args.upload:
            agent = next((a for a in client.agents.list().items if a.name == NAME), None)
            if agent is None:
                agent = client.agents.create(
                    name=NAME,
                    description="Human commentary preferences; saved outputs, no training.",
                )
            state["agent_id"] = agent.agent_id
            RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
            print("Agent:", links.agent(agent.agent_id), flush=True)
            for case in selected:
                entry = state["cases"].get(case["id"])
                digest = hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
                if entry and entry.get("content_sha256") not in (None, digest):
                    raise ValueError(
                        "Review task changed; preserve its existing labels and version it"
                    )
                if entry is None:
                    created = client.trajectories.create(
                        body={"agent_id": agent.agent_id},
                        idempotency_key=f"commentary-pref-{case['id']}-{digest[:24]}",
                    )
                    entry = {"trajectory_id": created.tid, "status": "created"}
                    state["cases"][case["id"]] = entry
                    RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
                tid = entry["trajectory_id"]
                print(
                    case["title"], links.production(agent.agent_id, trajectory_id=tid), flush=True
                )
                if entry["status"] == "created":
                    client.trajectories.log_event(
                        tid,
                        event_id="preference-review-task-v1",
                        name="commentary.preference_task",
                        payload=case,
                    )
                    client.trajectories.complete(tid, termination_reason="ENV_DONE")
                    entry["status"] = "complete_unreviewed"
                stored = list(client.trajectories.list_events(tid).items)
                if not any(event.event.get("payload") == case for event in stored):
                    raise RuntimeError("Stored task event did not match the prepared review")
                entry["content_sha256"] = digest
                RECEIPT.write_text(json.dumps(state, indent=2) + "\n")
                print("Verified stored event. No preference supplied.", flush=True)
        else:
            agent_id = state.get("agent_id")
            if not agent_id:
                raise ValueError("No saved workspace; upload first")
            output = []
            for case in selected:
                entry = state["cases"].get(case["id"])
                if entry is None:
                    continue
                tid = entry["trajectory_id"]
                trace = client.get(f"/api/v1/agents/{agent_id}/traces/{tid}", cast_to=dict)
                output.append(
                    {
                        "case_id": case["id"],
                        "trajectory_id": tid,
                        "url": links.production(agent_id, trajectory_id=tid),
                        "feedback_score": trace.get("feedback_score"),
                        "feedback_text": trace.get("feedback_text"),
                        "has_reviewer": bool(trace.get("reviewer_email")),
                    }
                )
            (DIRECTORY / "feedback-export.json").write_text(json.dumps(output, indent=2) + "\n")
            print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
