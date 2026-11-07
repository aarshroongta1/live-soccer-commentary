#!/usr/bin/env python3
"""Inspect public SoccerNet JSON or import a matched local window for evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from commentary.recorded_demo import validate_inputs
from commentary.replay.evaluation import ReferenceSource
from commentary.replay.soccernet import import_case, inspect_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--inspect", action="store_true", help="Inspect labels without video or models."
    )
    parser.add_argument("--half", type=int, choices=[1, 2])
    parser.add_argument("--upstream-split", choices=["train", "valid", "test"])
    parser.add_argument("--clip", type=Path)
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--video-half-offset", type=float)
    parser.add_argument("--case-id")
    parser.add_argument("--label-origin", help="Publisher URL or versioned annotation locator.")
    parser.add_argument(
        "--usage-terms", help="Actual source-video usage terms; no assumed license."
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    raw = args.labels.read_bytes()
    if args.inspect:
        print(json.dumps(inspect_labels(raw), indent=2))
        return
    required = [
        "half",
        "upstream_split",
        "clip",
        "start",
        "duration",
        "video_half_offset",
        "case_id",
        "label_origin",
        "usage_terms",
        "out",
    ]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("import requires: " + ", ".join("--" + n.replace("_", "-") for n in missing))
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    validate_inputs(args.clip, args.start, args.duration)
    with args.clip.open("rb") as media:
        digest = hashlib.file_digest(media, "sha256").hexdigest()
    case = import_case(
        raw,
        half=args.half,
        upstream_split=args.upstream_split,
        case_id=args.case_id,
        duration_s=args.duration,
        video_half_offset_s=args.video_half_offset,
        label_origin=args.label_origin,
        source=ReferenceSource(
            kind="broadcast",
            locator=str(args.clip),
            start_s=args.start,
            sha256=digest,
            provenance=f"Published SoccerNet events-v2 annotations from {args.label_origin}.",
            usage_terms=args.usage_terms,
            annotation_status="unreviewed",
        ),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also protects against a file appearing during video validation.
    with args.out.open("x") as destination:
        destination.write(case.model_dump_json(indent=2) + "\n")
    print(
        f"Imported {len(case.events)} mapped events to {args.out}; "
        "verify video alignment before scoring."
    )


if __name__ == "__main__":
    main()
