"""Publish a verified, completed human-review comparison as a read-only replay.

No model calls. Requires a real native review and a real native comparison vote.
Reviewer email, credentials, request IDs, and inline images are not exported.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from trajectory import Client

    state = json.loads((ROOT / "runs/commentary-editor/session.json").read_text())
    if state.get("comparison_status") != "complete" or not state.get("vote"):
        raise ValueError("A completed comparison and the user's actual judgment are required")
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
        tid = state["comparison"]["with_feedback"]["trajectory_id"]
        judged = client.get(f"/api/v1/agents/{agent}/traces/{tid}", cast_to=dict)
        if json.loads(judged["feedback_text"]) != state["vote"]:
            raise ValueError("The recorded judgment no longer matches the native feedback")
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
        "owner_vote": state["vote"],
    }
    evidence = {
        "review": public_review,
        "order": state["order"],
        "human_judgment": state["vote"],
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
            "Only one receives the prior human review, read back from Trajectory.",
            "The second task's background pool was prepared after the first review.",
            "Model output and judge outcome were retained without retries or cherry-picking.",
            "First clip uses retained TTS; take previews use a browser voice.",
            "New vision and inference ran via Trajectory; no model weights were updated.",
        ],
    }
    out = ROOT / "web/public/editor-review"
    (out / "session.json").write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
    (out / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print("Exported native-verified review, model outputs, and actual human judgment.")


if __name__ == "__main__":
    main()
