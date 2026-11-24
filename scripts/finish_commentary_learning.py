"""One declared revision: distill the real review into policy, then apply it once.

The initial raw-feedback failure is retained. This is inference-time policy memory,
not training or Trajectory Continual Grading. Two calls maximum, no retries.
"""

import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import commentary_editor as editor  # noqa: E402


def main() -> None:
    from trajectory import Client
    from trajectory.lib.platform_links import PlatformLinks

    state = json.loads(editor.STATE.read_text())
    if state.get("policy_revision", {}).get("status") == "complete":
        print("The declared revision is already complete; no new calls.")
        return
    if state.get("policy_revision"):
        raise ValueError("An existing revision must be inspected, not automatically retried")
    if state.get("comparison_status") != "complete":
        raise ValueError("The original comparison must remain available")
    with Client(max_retries=0, timeout=45) as client:
        if client.organizations.retrieve_current()["organization_key"] != "aarsh":
            raise ValueError("Wrong organization")
        review = state["review"]
        native = client.get(
            f"/api/v1/agents/{editor.AGENT}/traces/{review['trajectory_id']}", cast_to=dict
        )
        feedback = json.loads(native["feedback_text"])
        content, hashes, context = editor.visual_input()
        baseline_messages = [{"role": "user", "content": content}]
        digest = hashlib.sha256(json.dumps(baseline_messages, sort_keys=True).encode()).hexdigest()
        if digest != state["comparison"]["baseline"]["input_messages_sha256"]:
            raise ValueError("The shared input differs from the original baseline")
        if hashes != state["frame_hashes"]:
            raise ValueError("Visual inputs changed")
        state["earlier_attempt"] = {
            "comparison": copy.deepcopy(state["comparison"]),
            "order": state["order"],
            "vote": state.get("vote"),
            "finding": (
                "Raw positive feedback did not make the output use supplied background facts."
            ),
        }
        revision = state["policy_revision"] = {
            "status": "distilling",
            "source_review": review["trajectory_id"],
        }
        editor.persist(state)
        tid = client.trajectories.create(
            body={"agent_id": editor.AGENT},
            idempotency_key=f"editor-policy-{review['request_id']}-v1",
        ).tid
        revision["policy_trajectory_id"] = tid
        editor.persist(state)
        print(
            "Policy:",
            PlatformLinks("aarsh").production(editor.AGENT, trajectory_id=tid),
            flush=True,
        )
        response = client.chat.completions.create(
            model=editor.MODEL,
            x_trajectory_id=tid,
            max_tokens=128,
            temperature=0,
            disable_thinking=True,
            timeout=45,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Turn the human's explicit review into one reusable editorial preference, "
                        "at most 35 words. Return only the preference. Preserve what the person "
                        "liked or disliked, without inventing additional tastes. Do not include "
                        "player names, teams, or match statistics from the example. Do not turn "
                        "one positive example into a rule that applies regardless of context."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {k: feedback[k] for k in ("decision", "reason", "original_line", "rewrite")}
                    ),
                },
            ],
            idempotency_key=f"editor-policy-call-{review['request_id']}-v1",
        )
        policy = response.choices[0].message.content or ""
        if response.choices[0].finish_reason != "stop" or not policy.strip():
            raise ValueError("Policy generation did not complete; no retry")
        client.trajectories.complete(tid, termination_reason="ENV_DONE")
        revision.update(
            policy=policy, policy_response=response.model_dump(mode="json"), status="applying"
        )
        editor.persist(state)
        messages = [
            {
                "role": "system",
                "content": (
                    "Apply this persistent editorial preference "
                    "derived from the human's saved review:\n"
                    + policy
                    + "\nWhen it fits this analyst turn, prioritize one relevant supplied "
                    "background fact over simply restating visible action. Use at most one fact, "
                    "keep the 24-word limit, and never invent facts. The earlier match's names and "
                    "statistics are not evidence for this match."
                ),
            },
            *baseline_messages,
        ]
        tid = client.trajectories.create(
            body={"agent_id": editor.AGENT},
            idempotency_key=f"editor-policy-transfer-{review['request_id']}-v1",
        ).tid
        print(
            "Revised commentary:",
            PlatformLinks("aarsh").production(editor.AGENT, trajectory_id=tid),
            flush=True,
        )
        revision["commentary_trajectory_id"] = tid
        editor.persist(state)
        response = client.chat.completions.create(
            model=editor.MODEL,
            messages=messages,
            x_trajectory_id=tid,
            max_tokens=256,
            temperature=0,
            disable_thinking=True,
            timeout=45,
            idempotency_key=f"editor-policy-transfer-call-{review['request_id']}-v1",
        )
        text = response.choices[0].message.content or ""
        if response.choices[0].finish_reason != "stop" or not text.strip():
            raise ValueError("Revised commentary did not complete; no retry")
        client.trajectories.complete(tid, termination_reason="ENV_DONE")
        state["comparison"]["with_feedback"] = {
            "trajectory_id": tid,
            "url": PlatformLinks("aarsh").production(editor.AGENT, trajectory_id=tid),
            "status": "complete",
            "model": editor.MODEL,
            "response": response.model_dump(mode="json"),
            "text": text,
            "termination_reason": "ENV_DONE",
            "input_messages_sha256": hashlib.sha256(
                json.dumps(messages, sort_keys=True).encode()
            ).hexdigest(),
        }
        revision.update(
            status="complete",
            shared_input_sha256=digest,
            mechanism="Model-distilled preference applied as an explicit editorial instruction",
        )
        state.update(order={"A": "baseline", "B": "with_feedback"}, vote=None)
        editor.persist(state)
        print(
            json.dumps(
                {
                    "policy": policy,
                    "baseline": state["comparison"]["baseline"]["text"],
                    "revised": text,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
