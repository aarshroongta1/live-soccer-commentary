"""Local human label capture. No model calls or training; writes actual native feedback."""

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/commentary-learning"
AGENT = "agt_bc2248eb42e44f629e3175ea9330e705"


def persist(value):
    tmp = RUNS / "reviews.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(RUNS / "reviews.json")


def validate(body, case):
    index = body.get("choice")
    if type(index) is not int or not 0 <= index < len(case["options"]):
        raise ValueError("Choose the line you would air, including silence if appropriate.")
    reason = str(body.get("reason", "")).strip()
    if not reason or len(reason) > 1200:
        raise ValueError("Add a short reason for your choice.")
    return {"choice": index, "reason": reason, "request_id": str(UUID(body.get("request_id", "")))}


def main():
    from trajectory import Client
    from trajectory.lib.platform_links import PlatformLinks

    body = json.load(sys.stdin)
    state_path = RUNS / "reviews.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if body.get("action") == "status":
        print(json.dumps({"reviews": state, "connected": True}))
        return
    if body.get("action") != "review":
        raise ValueError("Invalid action")
    catalog = json.loads((RUNS / "catalog.json").read_text())
    choices = json.loads((RUNS / "choices.json").read_text())
    case = next(c for c in catalog["cases"] if c["id"] == body.get("case_id"))
    frozen = choices[case["id"]]
    review = validate(body, frozen)
    if case["id"] in state:
        existing = state[case["id"]]
        if any(existing[k] != review[k] for k in ["choice", "reason", "request_id"]):
            raise ValueError("This review is frozen for the experiment.")
        print(json.dumps({"reviews": state, "connected": True}))
        return
    review.update(
        case_id=case["id"],
        split=case["split"],
        selected_line=frozen["options"][review["choice"]],
        options_sha256=hashlib.sha256(json.dumps(frozen["options"]).encode()).hexdigest(),
        reviewed_at=datetime.now(UTC).isoformat(),
        origin="explicit human selection in local player",
    )
    with Client(max_retries=0, timeout=25) as client:
        if client.organizations.retrieve_current()["organization_key"] != "aarsh":
            raise ValueError("Wrong organization")
        tid = client.trajectories.create(
            body={"agent_id": AGENT}, idempotency_key=f"learning-review-{review['request_id']}"
        ).tid
        client.trajectories.log_event(
            tid,
            event_id="choice-set",
            name="commentary.choice_set",
            payload={
                "case_id": case["id"],
                "split": case["split"],
                "options": frozen["options"],
                "decision_at_s": case["decision_at_s"],
                "source_generation_trajectory_id": frozen["trajectory_id"],
                "task_prompt": "Choose the commentary you would air at second 15.",
            },
        )
        client.trajectories.complete(tid, termination_reason="ENV_DONE")
        feedback = json.dumps(review, ensure_ascii=False)
        client.patch(
            f"/api/v1/agents/{AGENT}/traces/{tid}",
            cast_to=dict,
            body={"feedback_score": None, "feedback_text": feedback},
        )
        actual = client.get(f"/api/v1/agents/{AGENT}/traces/{tid}", cast_to=dict)
        if actual.get("feedback_text") != feedback:
            raise ValueError("Native save could not be verified; retry the same save ID.")
        review.update(
            trajectory_id=tid,
            trajectory_url=PlatformLinks("aarsh").production(AGENT, trajectory_id=tid),
        )
        state[case["id"]] = review
        persist(state)
    print(json.dumps({"reviews": state, "connected": True}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, StopIteration) as error:
        print(json.dumps({"error": str(error) or "Unknown clip."}))
    except Exception:
        print(
            json.dumps(
                {
                    "error": "Trajectory save could not be confirmed. Retry the same "
                    "save; no model calls were made."
                }
            )
        )
