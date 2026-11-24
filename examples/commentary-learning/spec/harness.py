"""Frozen choice input; human labels are private to the scorer, never sent to actor."""

import json
import os


def actor_request(case):
    options = case["options"]
    return {
        "input": json.dumps(
            {
                "observation": case["observation"],
                "decision_at_s": 15,
                "seconds_left_in_excerpt": 5,
                "commentary_options": dict(enumerate(options)),
            },
            ensure_ascii=False,
        ),
        "questions": [
            {
                "type": "choice",
                "name": "commentary",
                "instructions": (
                    "Choose the commentary to air now. Select one option, "
                    "including silence if appropriate."
                ),
                "choices": [{"value": f"option_{i}"} for i in range(len(options))],
            }
        ],
    }


def main():
    from trajectory import Client

    case = json.loads(os.environ["CASE_INPUT"])
    expected = int(os.environ["HUMAN_CHOICE"])
    if not 0 <= expected < len(case["options"]):
        raise ValueError("Invalid human label")
    with Client() as client:
        tid = client.trajectories.create().tid
        response = client.decisions.create(
            model="policy",
            **actor_request(case),
            x_trajectory_id=tid,
            idempotency_key=f"{tid}:choice",
        )
        if response.result is None:
            raise RuntimeError("Missing decision")
        for index in range(len(case["options"])):
            client.trajectories.log_reward(
                tid,
                name=f"decision.outcome.{index}",
                value=float(index == expected),
                min_value=0,
                max_value=1,
                weight=0,
                extra_body={"reward_id": f"{tid}:label:{index}"},
            )
        client.trajectories.log_reward(
            tid,
            name="agreement_with_human",
            value=float(response.result.choice == f"option_{expected}"),
            weight=1,
            extra_body={"reward_id": f"{tid}:agreement"},
        )
        client.trajectories.log_event(
            tid,
            event_id="selector-output",
            name="commentary.selection",
            payload={
                "case_id": case["id"],
                "choice": response.result.choice,
                "probabilities": response.result.probabilities,
                "options": case["options"],
            },
        )
        client.trajectories.complete(tid, termination_reason="ENV_DONE")


if __name__ == "__main__":
    main()
