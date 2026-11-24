"""Explicit native stages: baseline, train, final eval, status. No automatic reruns."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/commentary-learning"
MODEL = "Qwen/Qwen3.5-4B"
COMMON = {
    "max_turns_per_trajectory": 1,
    "max_output_tokens_per_turn": 128,
    "max_output_tokens_per_trajectory": 128,
    "reasoning_effort": "off",
    "temperature": 0,
    "trajectory_timeout_seconds": 120,
}


def persist(state):
    tmp = RUNS / "native-runs.tmp"
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(RUNS / "native-runs.json")


def main():
    from trajectory import Client
    from trajectory.lib.platform_links import PlatformLinks

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["baseline", "train", "after", "status"])
    args = parser.parse_args()
    upload = json.loads((RUNS / "upload.json").read_text())
    for name in ["catalog", "choices"]:
        if (
            hashlib.sha256((RUNS / f"{name}.json").read_bytes()).hexdigest()
            != upload[f"{name}_sha256"]
        ):
            raise ValueError("Frozen inputs changed after upload")
    state_path = RUNS / "native-runs.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if args.stage != "status" and state.get(args.stage):
        print(json.dumps(state[args.stage], indent=2))
        return
    with Client(max_retries=0, timeout=45) as client:
        if client.organizations.retrieve_current()["organization_key"] != "aarsh":
            raise ValueError("Wrong organization")
        links = PlatformLinks("aarsh")
        if args.stage == "status":
            for key in ["baseline", "after"]:
                if state.get(key):
                    result = client.evals.runs.retrieve(state[key]["id"])
                    print(key, state[key]["url"], result.model_dump_json())
            if state.get("train"):
                print(
                    "train",
                    state["train"]["url"],
                    client.training.runs.progress(state["train"]["id"]).model_dump_json(),
                )
            return
        if args.stage == "train":
            before = client.evals.runs.retrieve(state["baseline"]["id"])
            if before.status != "completed" or before.reward_mean is None:
                raise ValueError("Inspect completed baseline native grades before training")
            options = client.training.list_options(
                bench_id=upload["bench_id"], base_model_slug=MODEL
            )
            supported = options.models[0].options["algorithm"].choices
            if "rlcd" not in supported:
                raise ValueError(
                    "Native decision training is not advertised for this spec/model; inspect access"
                )
            settings = {
                **COMMON,
                "algorithm": "rlcd",
                "num_steps": 10,
                "batch_size": 2,
                "group_size": 1,
                "min_trajectories_per_group": 1,
                "train_on_all_fail": True,
                "train_on_all_pass": True,
                "train_on_zero_advantage": True,
                "val_enabled": False,
                "learning_rate": 0.0001,
                "lr_warmup_steps": 0,
                "max_concurrent_trajectories": 2,
            }
            run = client.training.create(
                bench_id=upload["bench_id"],
                base_model_slug=MODEL,
                options=settings,
                display_name="Commentary taste — 10 bounded updates",
                idempotency_key=f"{upload['bench_id']}:training-v1",
            )
            state["train"] = {
                "id": run.training_run_id,
                "url": links.training(run.training_run_id),
                "options": settings,
            }
        else:
            kwargs = {}
            if args.stage == "after":
                training_id = state["train"]["id"]
                progress = client.training.runs.progress(training_id)
                if progress.status != "completed" or progress.completed_steps < 1:
                    raise ValueError("Completed optimizer updates required before final eval")
                checkpoints = client.training.checkpoints.list(training_id).items
                if not checkpoints:
                    raise ValueError("No checkpoint")
                # Final checkpoint only: never select against the held-out labels.
                final = max(checkpoints, key=lambda c: c.step_index)
                kwargs["parent_checkpoint_id"] = final.checkpoint_id
            settings = {
                **COMMON,
                "val_num_tasks": 3,
                "val_trajectories_per_task": 1,
                "val_max_concurrent_trajectories": 1,
            }
            run = client.evals.create(
                bench_id=upload["bench_id"],
                base_model_slug=MODEL,
                options=settings,
                task_split="test",
                display_name=f"Commentary choice — {args.stage}",
                idempotency_key=f"{upload['bench_id']}:{args.stage}-v1",
                **kwargs,
            )
            state[args.stage] = {
                "id": run.eval_run_id,
                "url": links.evaluation(run.eval_run_id),
                "options": settings,
                **kwargs,
            }
        persist(state)
        print(json.dumps(state[args.stage], indent=2), flush=True)


if __name__ == "__main__":
    main()
