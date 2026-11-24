"""Generate and freeze review choices once. At most 13 bounded calls, no retries."""

import hashlib
import json
import random
from pathlib import Path

from trajectory import Client
from trajectory.lib import ImageInput
from trajectory.lib.platform_links import PlatformLinks

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/commentary-learning"
AGENT = "agt_bc2248eb42e44f629e3175ea9330e705"
MODEL = "openai/gpt-5.4-mini"


def main():
    catalog = json.loads((RUNS / "catalog.json").read_text())
    output = RUNS / "choices.json"
    saved = json.loads(output.read_text()) if output.exists() else {}
    with Client(max_retries=0, timeout=60) as client:
        if client.organizations.retrieve_current()["organization_key"] != "aarsh":
            raise ValueError("Wrong organization")
        links = PlatformLinks("aarsh")
        for case in catalog["cases"]:
            identity = case["id"]
            if identity in saved:
                if saved[identity].get("status") != "complete":
                    raise ValueError(
                        f"{identity} incomplete: inspect native trace, do not resubmit"
                    )
                continue
            tid = client.trajectories.create(
                body={"agent_id": AGENT},
                idempotency_key=f"commentary-learning-v1-choices-{identity}",
            ).tid
            saved[identity] = {"status": "started", "trajectory_id": tid}
            output.write_text(json.dumps(saved, indent=2))
            print(f"{identity}: {links.production(AGENT, trajectory_id=tid)}", flush=True)
            content = [
                {
                    "type": "text",
                    "text": (
                        "These chronological soccer frames span seconds 3 to 15 of a "
                        "20-second excerpt. "
                        "You are producing choices for the next analyst/commentator turn at "
                        "second 15. "
                        "Return JSON only with keys observation (a neutral description of "
                        "visible action, "
                        "maximum 60 words), and lines (exactly three distinct plausible "
                        "commentary lines). "
                        "Line 1: your normal generated commentary, at most 18 words. "
                        "Line 2: an alternative adding a tactical observation grounded in "
                        "these frames, "
                        "at most 18 words. Line 3: a brief alternative, at most 8 words. "
                        "Do not invent player names, statistics, score, or outcomes not visible. "
                        "Treat these as real alternatives, not a quiz with one deliberately "
                        "incorrect answer. "
                        "No narration of your generation process. No original audio or "
                        "future frames are provided."
                    ),
                }
            ]
            for frame in case["frames"]:
                path = ROOT / frame["path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
                    raise ValueError("Input frame changed")
                content.append(ImageInput.from_path(path).part())
            response = client.chat.completions.create(
                model=MODEL,
                x_trajectory_id=tid,
                messages=[{"role": "user", "content": content}],
                max_tokens=500,
                temperature=0,
                disable_thinking=True,
                idempotency_key=f"{tid}:choices",
                timeout=60,
            )
            text = response.choices[0].message.content or ""
            saved[identity]["response"] = response.model_dump(mode="json")
            output.write_text(json.dumps(saved, indent=2))
            if response.choices[0].finish_reason != "stop":
                raise ValueError("Generation capped; inspect before proceeding")
            parsed = json.loads(
                text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            )
            if not isinstance(parsed.get("observation"), str) or not isinstance(
                parsed.get("lines"), list
            ):
                raise ValueError("Invalid choice response")
            lines = parsed["lines"]
            if len(lines) != 3 or any(
                not isinstance(t, str) or not t.strip() or len(t.split()) > 22 for t in lines
            ):
                raise ValueError("Expected three short lines")
            options = lines + ["[SILENCE]"]
            random.Random(identity).shuffle(options)
            saved[identity].update(
                status="complete",
                observation=parsed["observation"],
                options=options,
                normal_line=lines[0],
                model=MODEL,
            )
            client.trajectories.complete(tid, termination_reason="ENV_DONE")
            output.write_text(json.dumps(saved, indent=2) + "\n")
    public = {
        **catalog,
        "cases": [
            {
                **{k: v for k, v in case.items() if k not in ["frames", "source_sha256", "source"]},
                "options": saved[case["id"]]["options"],
                "normal_line": saved[case["id"]]["normal_line"],
            }
            for case in catalog["cases"]
        ],
    }
    (ROOT / "web/public/commentary-learning/catalog.json").write_text(
        json.dumps(public, indent=2) + "\n"
    )
    print("Frozen 13 sets of options. No human labels or training yet.", flush=True)


if __name__ == "__main__":
    main()
