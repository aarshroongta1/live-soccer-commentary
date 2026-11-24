"""Verify actual saved labels and upload a native decision spec. No training starts."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/commentary-learning"
AGENT = "agt_bc2248eb42e44f629e3175ea9330e705"


def main():
    from trajectory import BenchmarkSpec, Client, TaskSpec
    from trajectory.lib import DockerfileBuild, push
    from trajectory.lib.platform_links import PlatformLinks

    catalog = json.loads((RUNS / "catalog.json").read_text())
    if catalog.get("phase") == "pilot":
        raise ValueError("Pilot quality review only: freeze the full train/test set before upload")
    choices = json.loads((RUNS / "choices.json").read_text())
    reviews = json.loads((RUNS / "reviews.json").read_text())
    if set(reviews) != {c["id"] for c in catalog["cases"]}:
        raise ValueError("All ten training and three test clips need explicit human choices first.")
    receipt = RUNS / "upload.json"
    if receipt.exists():
        print(receipt.read_text())
        return
    with Client(max_retries=0, timeout=60) as client:
        if client.organizations.retrieve_current()["organization_key"] != "aarsh":
            raise ValueError("Wrong organization")
        tasks = []
        for case in catalog["cases"]:
            review = reviews[case["id"]]
            frozen = choices[case["id"]]
            actual = client.get(
                f"/api/v1/agents/{AGENT}/traces/{review['trajectory_id']}", cast_to=dict
            )
            native = json.loads(actual["feedback_text"])
            for key in ["case_id", "split", "choice", "reason", "request_id", "options_sha256"]:
                if native[key] != review[key]:
                    raise ValueError("Native review changed")
            expected_hash = hashlib.sha256(json.dumps(frozen["options"]).encode()).hexdigest()
            if expected_hash != native["options_sha256"]:
                raise ValueError("Reviewed options changed")
            actor = {
                "id": case["id"],
                "observation": frozen["observation"],
                "options": frozen["options"],
            }
            tasks.append(
                TaskSpec(
                    name=case["id"],
                    split=case["split"],
                    run_command="python /app/harness.py",
                    env_vars={
                        "CASE_INPUT": json.dumps(actor),
                        "HUMAN_CHOICE": str(review["choice"]),
                    },
                )
            )
        spec = BenchmarkSpec(
            name="Commentary choice learning v1",
            benchmark_type="decision",
            runtime=DockerfileBuild("Dockerfile"),
            tasks=tasks,
        )
        result = push(
            client,
            spec,
            agent_id=AGENT,
            root=ROOT / "examples/commentary-learning/spec",
            build_images=True,
        )
        links = PlatformLinks("aarsh")
        record = {
            "bench_id": result.bench_id,
            "agent_id": AGENT,
            "agent_url": links.agent(AGENT),
            "train_count": 10,
            "test_count": 3,
            "catalog_sha256": hashlib.sha256((RUNS / "catalog.json").read_bytes()).hexdigest(),
            "choices_sha256": hashlib.sha256((RUNS / "choices.json").read_bytes()).hexdigest(),
        }
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
