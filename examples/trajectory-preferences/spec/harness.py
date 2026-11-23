"""One editorial recommendation; human taste remains a separate native review.

The automatic score checks the response contract only. It is not a quality,
factuality, or preference score. No preferred option is supplied to this harness.
"""

import json
import os
from pathlib import Path


def contract_valid(text: str) -> bool:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    return (
        isinstance(value, dict)
        and set(value) == {"choice", "reason"}
        and value["choice"] in ("A", "B", "C")
        and isinstance(value["reason"], str)
        and 1 <= len(value["reason"].split()) <= 60
    )


def main() -> None:
    from trajectory import Client

    tasks = json.loads(Path(__file__).with_name("tasks.json").read_text())
    task = next(t for t in tasks if t["id"] == os.environ["COMMENTARY_TASK_ID"])
    with Client(max_retries=0, timeout=45) as client:
        tid = client.trajectories.create().tid
        try:
            reply = client.chat.completions.create(
                model="policy",
                x_trajectory_id=tid,
                messages=task["messages"],
            )
            choice = reply.choices[0]
            if choice.finish_reason == "length":
                client.trajectories.complete(tid, termination_reason="TRUNCATION")
                return
            if choice.finish_reason != "stop":
                client.trajectories.complete(tid, termination_reason="ERROR")
                return
            text = choice.message.content or ""
            valid = contract_valid(text)
            client.trajectories.log_event(
                tid,
                event_id="contract-check-v1",
                name="commentary.response_contract",
                payload={
                    "contract_valid": valid,
                    "human_preference": None,
                    "note": "Format only. Human editorial judgment has not been collected.",
                },
            )
            client.trajectories.log_reward(
                tid,
                name="response_contract_only",
                value=float(valid),
            )
            client.trajectories.complete(tid, termination_reason="ENV_DONE")
        except Exception:
            client.trajectories.complete(tid, termination_reason="ERROR")
            raise


if __name__ == "__main__":
    main()
