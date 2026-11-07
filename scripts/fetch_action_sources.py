#!/usr/bin/env python3
"""Verify the small action source pool, optionally fetching missing excerpts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--id", action="append", help="Limit to selected catalog IDs; repeatable.")
    args = parser.parse_args()
    catalog = json.loads((ROOT / "examples/action-sources.json").read_text())
    selected = set(args.id or [row["id"] for row in catalog["sources"]])
    known = {row["id"] for row in catalog["sources"]}
    if selected - known:
        parser.error(f"unknown source IDs: {sorted(selected - known)}")
    failures: list[str] = []
    for row in catalog["sources"]:
        if row["id"] not in selected:
            continue
        path = ROOT / row["local_path"]
        if not path.exists() and args.download_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--no-overwrites",
                    "--download-sections",
                    f"*{row['download_start_s']}-{row['download_end_s']}",
                    "-f",
                    row["format_id"],
                    "--socket-timeout",
                    "15",
                    "--retries",
                    "1",
                    "--fragment-retries",
                    "1",
                    "-o",
                    str(path),
                    row["url"],
                ],
                check=True,
            )
        if not path.exists():
            failures.append(f"{row['id']}: missing; pass --download-missing to fetch")
            continue
        with path.open("rb") as media:
            digest = hashlib.file_digest(media, "sha256").hexdigest()
        if digest != row["sha256"]:
            failures.append(
                f"{row['id']}: hash differs; review media and labels before updating catalog"
            )
        else:
            print(f"{row['id']}: verified ({path.stat().st_size / 1_000_000:.2f} MB)")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
