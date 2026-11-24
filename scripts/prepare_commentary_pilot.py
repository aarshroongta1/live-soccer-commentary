"""Three short, moment-specific editorial choices. No model calls or human labels."""

import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/commentary-learning"
OUT = ROOT / "web/public/commentary-learning"


def main():
    # Preserve the failed 20s batch and any real reviews; never reuse its labels.
    archive = RUNS / "rejected-20s-batch"
    if not archive.exists():
        archive.mkdir()
        for name in ["catalog.json", "choices.json", "reviews.json"]:
            path = RUNS / name
            if path.exists():
                shutil.copy2(path, archive / name)
        if (RUNS / "reviews.json").exists():
            (RUNS / "reviews.json").unlink()
    elif (RUNS / "reviews.json").exists():
        raise ValueError("Pilot already has reviews: do not replace its frozen choices")
    original = json.loads((ROOT / "web/public/soccer-commentary/live-search-run.json").read_text())
    ballard = next(line["text"] for line in original["delivery"] if line["id"] == "context")
    definitions = [
        {
            "id": "pilot-ballard",
            "title": "After Ballard’s goal",
            "source": "clips/demo-sources/ballard-arsenal.mp4",
            "start": 10,
            "cue": 1.4,
            "role": "Analyst",
            "observation": "Dan Ballard has scored for Sunderland against Arsenal. "
            "The caller has already announced the goal. Ballard was an Arsenal academy player; "
            "the goal ends Arsenal’s eight-game clean-sheet run. The "
            "analyst now has a gap to speak.",
            "normal": ballard,
            "edits": [
                "Arsenal’s eight-game clean-sheet run is over. And it’s "
                "one of their own who ends it.",
                "Ballard never made a first-team appearance for Arsenal. "
                "He’s made his mark against them now.",
            ],
            "context": "The goal has already been called. Choose the next analyst line.",
            "source_record": "web/public/soccer-commentary/live-search-run.json",
            "research_provenance": "Retained Keenable source excerpts in "
            "live-search-run.json: Guardian and Athletic. "
            "Both support his Arsenal academy background; Athletic "
            "states no first-team appearance.",
        },
        {
            "id": "pilot-arsenal",
            "title": "Arsenal’s finish",
            "source": "clips/demo-sources/arsenal-team-goals.mp4",
            "start": 94,
            "cue": 3,
            "role": "Caller",
            "observation": "Arsenal number 7 dribbles past Norwich defenders and "
            "scores low beyond the diving goalkeeper. The ball is visibly in the net. "
            "This is the immediate goal call; no earlier goal call has been spoken.",
            "normal": "And Arsenal’s number 7 finishes it!",
            "edits": [
                "He’s beaten his man—and beaten the goalkeeper! Arsenal score.",
                "Arsenal score! He gets away from his marker and rolls it beyond the keeper.",
            ],
            "context": "Choose the immediate goal call, as the ball goes in.",
            "source_record": "web/public/soccer-commentary/generated-arsenal.trace.json",
        },
        {
            "id": "pilot-kane",
            "title": "Kane’s penalty",
            "source": "clips/action-sources/england-france-penalties.mp4",
            "start": 49,
            "cue": 3,
            "role": "Caller",
            "observation": "Harry Kane scores England’s penalty low to the left against "
            "France. The pre-kick score shown is England 0 France 1. The ball has crossed the goal "
            "line and Kane turns away to celebrate. This is the immediate goal call.",
            "normal": "Kane scores the penalty!",
            "edits": [
                "Harry Kane! England are level!",
                "Kane scores! Low into the corner, and England have their equaliser.",
            ],
            "context": "Choose the immediate reaction to the penalty going in.",
            "source_record": "web/public/soccer-commentary/prepared-penalty.trace.json",
        },
    ]
    cases, choices = [], {}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for case in definitions:
        source = ROOT / case["source"]
        video = OUT / f"{case['id']}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-y",
                "-ss",
                str(case["start"]),
                "-i",
                str(source),
                "-t",
                "8",
                "-an",
                "-vf",
                "scale=640:-2",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
        options = [case["normal"], *case["edits"], "[SILENCE]"]
        random.Random(case["id"]).shuffle(options)
        frames = []
        folder = RUNS / case["id"]
        folder.mkdir(exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        for at in [0, case["cue"] / 2, case["cue"]]:
            capture.set(cv2.CAP_PROP_POS_MSEC, (case["start"] + at) * 1000)
            ok, frame = capture.read()
            if not ok:
                raise ValueError("Cannot decode pilot frame")
            path = folder / f"{at:.1f}.jpg"
            cv2.imwrite(str(path), cv2.resize(frame, (512, 288)))
            frames.append(
                {
                    "at_s": at,
                    "path": str(path.relative_to(ROOT)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        capture.release()
        cases.append(
            {
                "id": case["id"],
                "title": case["title"],
                "split": "pilot",
                "video_url": f"/commentary-learning/{case['id']}.mp4",
                "duration_s": 8,
                "decision_at_s": case["cue"],
                "context": case["context"],
                "role": case["role"],
                "source": case["source"],
                "source_start_s": case["start"],
                "frames": frames,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
        choices[case["id"]] = {
            "status": "complete",
            "options": options,
            "normal_line": case["normal"],
            "observation": case["observation"],
            "source_record": case["source_record"],
            "provenance": "Original retained generated line plus two "
            "assistant-authored editorial alternatives. "
            "No preferred answer is assigned. No new model calls.",
            "research_provenance": case.get("research_provenance"),
        }
    catalog = {
        "version": 2,
        "phase": "pilot",
        "training_footage_s": 0,
        "test_footage_s": 0,
        "target_id": None,
        "cases": cases,
    }
    (RUNS / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (RUNS / "choices.json").write_text(json.dumps(choices, indent=2) + "\n")
    public = {
        **catalog,
        "cases": [
            {
                **{k: v for k, v in c.items() if k not in ["frames", "source", "source_sha256"]},
                "options": choices[c["id"]]["options"],
                "normal_line": choices[c["id"]]["normal_line"],
            }
            for c in cases
        ],
    }
    (OUT / "catalog.json").write_text(json.dumps(public, indent=2) + "\n")
    print(
        "Prepared 3 × 8-second pilot moments; original generated "
        "lines plus authored alternatives. No training."
    )


if __name__ == "__main__":
    main()
