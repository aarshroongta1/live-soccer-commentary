"""Prepare 3m20s of review footage and three disjoint 20s test clips, offline."""

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/public/commentary-learning"
RUNS = ROOT / "runs/commentary-learning"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    # Frozen before human labeling or selector training. No overlap between splits.
    train = [
        ("liga-f-routine.mp4", 0, "Possession"),
        ("liga-f-routine.mp4", 20, "Possession"),
        ("liga-f-routine.mp4", 40, "Possession"),
        ("chelsea-burnley-2015.webm", 0, "Chelsea vs Burnley"),
        ("chelsea-burnley-2015.webm", 20, "Chelsea vs Burnley"),
        ("chelsea-burnley-2015.webm", 40, "Chelsea vs Burnley"),
        ("everton-var-offside.webm", 0, "Everton"),
        ("everton-var-offside.webm", 20, "Everton"),
        ("france-jamaica.webm", 0, "France vs Jamaica"),
        ("france-jamaica.webm", 20, "France vs Jamaica"),
    ]
    tests = [
        ("argentina-france-dimaria.mp4", 0, "Argentina vs France"),
        ("portugal-spain-live.mp4", 0, "Portugal vs Spain"),
        ("betis-barcelona-ferran.mp4", 0, "Betis vs Barcelona"),
    ]
    cases = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for split, rows in [("train", train), ("test", tests)]:
        for index, (filename, start, title) in enumerate(rows):
            source = (
                ROOT
                / "clips"
                / ("action-sources" if split == "train" else "demo-sources")
                / filename
            )
            identity = f"{split}-{index + 1:02d}"
            video = OUT / f"{identity}.mp4"
            if not video.exists():
                subprocess.run(
                    [
                        ffmpeg,
                        "-v",
                        "error",
                        "-y",
                        "-ss",
                        str(start),
                        "-i",
                        str(source),
                        "-t",
                        "20",
                        "-an",
                        "-vf",
                        "scale=640:-2",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "25",
                        "-movflags",
                        "+faststart",
                        str(video),
                    ],
                    check=True,
                )
            folder = RUNS / identity
            folder.mkdir(exist_ok=True)
            capture = cv2.VideoCapture(str(source))
            frames = []
            # Proposals occur at 15s; the final 5s are reviewer-only playback.
            for at in [3.0, 6.0, 9.0, 12.0, 15.0]:
                capture.set(cv2.CAP_PROP_POS_MSEC, (start + at) * 1000)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Cannot decode {identity} at {at}")
                path = folder / f"{at:.0f}.jpg"
                cv2.imwrite(
                    str(path), cv2.resize(frame, (512, 288)), [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
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
                    "id": identity,
                    "split": split,
                    "title": title,
                    "source": str(source.relative_to(ROOT)),
                    "source_start_s": start,
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "video_url": f"/commentary-learning/{identity}.mp4",
                    "duration_s": 20,
                    "decision_at_s": 15,
                    "frames": frames,
                }
            )
    catalog = {
        "version": 1,
        "training_footage_s": 200,
        "test_footage_s": 60,
        "target_id": "test-01",
        "cases": cases,
    }
    path = RUNS / "catalog.json"
    if path.exists() and json.loads(path.read_text()) != catalog:
        raise ValueError("Frozen catalog changed; inspect before replacing")
    path.write_text(json.dumps(catalog, indent=2) + "\n")
    print(f"Prepared {len(cases)} clips; 200s training, 60s test; no model calls.")


if __name__ == "__main__":
    main()
