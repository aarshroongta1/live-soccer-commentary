"""Prepare or upload the commentary preference spec; never start a model run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "examples/trajectory-preferences"
RUNTIME = DIRECTORY / "spec"


def prepare() -> list[dict]:
    tasks = []
    for name in ("penalty-reaction", "late-save-call", "after-goal-reaction"):
        case = json.loads((DIRECTORY / f"{name}.json").read_text())
        observation_lines = []
        for observation in case["observations"]:
            actor = observation.get("actor") or {}
            identity = (
                f" Resolved actor: {actor['resolved_name']}; action role: {actor['action_role']}."
                if actor.get("resolved_name")
                else ""
            )
            observation_lines.append(
                f"- {observation['id']} at {observation['at_s']}s: "
                f"{observation['description']}{identity} "
                f"[observer certainty: "
                f"{observation.get('action', {}).get('certainty', 'unspecified')}]"
            )
        observations = "\n".join(observation_lines)
        options = "\n".join(f"{o['id']}. {o['text']}" for o in case["options"])
        context = case.get("recent_spoken_lines", [])
        if case.get("currently_speaking"):
            context.append("Currently speaking: " + case["currently_speaking"]["text"])
        prompt = (
            case["title"]
            + "\n\n"
            + case["task"]
            + "\n\nAvailable observations (saved model reports):\n"
            + observations
            + "\n\nRecent dialogue:\n"
            + ("\n".join(context) or "None.")
            + "\n\nCommentary options:\n"
            + options
            + "\n\nRecommend A, B, or C and briefly explain. Consider grounding, "
            "what has already been said, timing, and whether silence is better. "
            "There is no supplied human preference or reference answer. "
            'Return only JSON: {"choice":"A|B|C","reason":"your explanation"}. '
            "Use at most 60 words for the reason."
        )
        if "replay_clock_s" in case:
            prompt += (
                f"\nThe caller-ready clock is {case['replay_clock_s']:.3f}s; "
                f"the latest cited action occurred at {case['latest_cited_action_at_s']}s. "
                "This clock includes processing/drain after an eight-second clip."
            )
        tasks.append({"id": name, "messages": [{"role": "user", "content": prompt}]})
    (RUNTIME / "tasks.json").write_text(json.dumps(tasks, indent=2, ensure_ascii=False) + "\n")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--revision", choices=("v1", "v2"), default="v1")
    args = parser.parse_args()
    receipt_path = DIRECTORY / (
        "spec-upload.json" if args.revision == "v1" else "spec-upload-v2.json"
    )
    tasks = prepare()
    from trajectory import BenchmarkSpec, Client, TaskSpec
    from trajectory.lib import DockerfileBuild, start_push
    from trajectory.lib.platform_links import PlatformLinks

    manifest = BenchmarkSpec(
        name="Commentary editorial preferences v1",
        description=(
            "Three saved-context editorial tasks. Recommend A/B/C; human reviewer supplies "
            "their preferred option and why. Automatic reward checks format only, not taste. "
            "Evaluation-only: all tasks are test; no training authorized."
        ),
        visibility="private",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=task["id"],
                split="test",
                input_messages=task["messages"],
                run_command="python /app/harness.py",
                env_vars={"COMMENTARY_TASK_ID": task["id"]},
                env_resources={"cpus": 1, "memory_mb": 512},
            )
            for task in tasks
        ],
    )
    if not args.upload:
        print(manifest.model_dump_json(indent=2))
        return
    if receipt_path.exists():
        print("Existing upload receipt; resume it instead of submitting again:")
        print(receipt_path.read_text())
        return
    workspace = json.loads((DIRECTORY / "workspace.json").read_text())
    with Client(max_retries=0, timeout=30) as client:
        org = client.organizations.retrieve_current()
        if org["organization_key"] != workspace["organization_key"]:
            raise ValueError("Authenticated organization does not match the saved agent")
        operation = start_push(
            client,
            manifest,
            agent_id=workspace["agent_id"],
            root=RUNTIME,
            build_images=True,
            idempotency_key=f"commentary-editorial-preferences-{args.revision}",
            upload_max_retries=0,
        )
        receipt = {
            "operation_id": operation.id,
            "agent_id": workspace["agent_id"],
            "organization_key": workspace["organization_key"],
        }
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        print("Agent:", PlatformLinks(workspace["organization_key"]).agent(workspace["agent_id"]))
        print("Upload started. No diagnostic, evaluation, training, or inference submitted.")


if __name__ == "__main__":
    main()
