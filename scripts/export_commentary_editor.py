"""Publish a verified, completed human-review comparison as a read-only replay.

No model calls. Requires a real native review; a missing comparison vote remains null.
Reviewer email, credentials, request IDs, and inline images are not exported.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from trajectory import Client

    state = json.loads((ROOT / "runs/commentary-editor/session.json").read_text())
    if state.get("comparison_status") != "complete":
        raise ValueError("A completed comparison is required")
    agent = "agt_bc2248eb42e44f629e3175ea9330e705"
    with Client(max_retries=0, timeout=20) as client:
        review = state["review"]
        native = client.get(
            f"/api/v1/agents/{agent}/traces/{review['trajectory_id']}", cast_to=dict
        )
        actual_review = json.loads(native["feedback_text"])
        for key in ("decision", "reason", "rewrite", "original_line"):
            if actual_review[key] != review[key]:
                raise ValueError("The recorded review no longer matches the native feedback")
        if state.get("vote"):
            tid = state["comparison"]["with_feedback"]["trajectory_id"]
            judged = client.get(f"/api/v1/agents/{agent}/traces/{tid}", cast_to=dict)
            if json.loads(judged["feedback_text"]) != state["vote"]:
                raise ValueError("The judgment no longer matches native feedback")
        policy = state["policy_revision"]
        steps = client.trajectories.steps.list(policy["policy_trajectory_id"]).items
        if not any(
            m.role == "assistant" and m.content == policy["policy"]
            for step in steps
            for m in step.messages
        ):
            raise ValueError("Policy must match the native model response")
        for arm in state["comparison"].values():
            steps = client.trajectories.steps.list(arm["trajectory_id"]).items
            messages = steps[-1].messages if steps else []
            actual = [m.content for m in messages if m.role == "assistant"]
            if not actual or actual[-1] != arm["text"]:
                raise ValueError("Exported model words must match the actual native response")
    data = json.loads((ROOT / "web/data/editor-review.json").read_text())
    hashes = [
        hashlib.sha256(
            (ROOT / "runs/commentary-editor/frames" / f"{at:.1f}.jpg").read_bytes()
        ).hexdigest()
        for at in data["transfer"]["frame_times_s"]
    ]
    if hashes != state["frame_hashes"]:
        raise ValueError("The retained frame inputs changed")
    public_review = {
        key: review[key]
        for key in (
            "decision",
            "reason",
            "rewrite",
            "original_line",
            "source_run_id",
            "reviewed_at",
        )
    }
    public_review["trajectory_url"] = ""
    public = {
        "recorded": True,
        "review": public_review,
        "comparison_status": "complete",
        "takes": [
            {"label": label, "text": state["comparison"][arm]["text"], "trajectory_url": ""}
            for label, arm in state["order"].items()
        ],
        "recorded_order": state["order"],
        "reveal": state["order"],
        "policy": policy["policy"],
        "owner_vote": state["vote"],
    }
    evidence = {
        "review": public_review,
        "order": state["order"],
        "human_judgment": state["vote"],
        "policy": {
            k: policy[k]
            for k in ("policy", "mechanism", "policy_trajectory_id", "shared_input_sha256")
        },
        "earlier_attempt": {
            arm: {"text": run["text"], "trajectory_id": run["trajectory_id"]}
            for arm, run in state["earlier_attempt"]["comparison"].items()
        },
        "frames": data["transfer"]["frame_times_s"],
        "frame_sha256": hashes,
        "shared_context": state["common_context"],
        "runs": {
            arm: {
                "trajectory_id": value["trajectory_id"],
                "model": value["model"],
                "text": value["text"],
                "termination_reason": value["termination_reason"],
                "usage": value["response"].get("usage"),
                "input_messages_sha256": value["input_messages_sha256"],
            }
            for arm, value in state["comparison"].items()
        },
        "limits": [
            "One development comparison; no general improvement or training claim.",
            "Both runs receive identical six frames, identity context, and research facts.",
            (
                "The revised run receives a model-distilled preference plus an "
                "application-authored "
                "instruction to prioritize one relevant background fact."
            ),
            "The second task's background pool was prepared after the first review.",
            (
                "Raw-feedback transfer did not add background context. "
                "One explicit policy revision "
                "followed; both attempts are retained. No human comparison judgment is recorded."
            ),
            "First clip uses retained TTS; take previews use a browser voice.",
            "New vision and inference ran via Trajectory; no model weights were updated.",
        ],
    }
    out = ROOT / "web/public/editor-review"
    (out / "session.json").write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(
        "Exported native-verified review, policy and outputs; missing human judgment remains null."
    )


if __name__ == "__main__":
    main()
