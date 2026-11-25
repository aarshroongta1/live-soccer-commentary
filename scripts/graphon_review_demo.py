#!/usr/bin/env python3
"""Prepare a small review archive, or replay captured Graphon search passages.

Both commands are offline. Upload/query with Graphon MCP or the official CLI;
this script never reads credentials or supplies review material to the caller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "docs/FALSE_GOAL_INVESTIGATION.md",
    "docs/TEXT_EVALUATION_TRIAL.md",
    "docs/RECOGNITION_PASS.md",
    "examples/lesson-memory/README.md",
    "examples/lesson-memory/session/README.md",
    "examples/lesson-memory/transfer/README.md",
    "examples/portfolio-demo/README.md",
    "examples/keenable-outreach/README.md",
)
# Chosen before querying. Expected files are navigation checks, not answer labels.
QUESTIONS = (
    {
        "id": "false-goal",
        "query": "Why can a goal call have valid evidence citations and still be false?",
        "expected_source": "docs/FALSE_GOAL_INVESTIGATION.md",
    },
    {
        "id": "memory-transfer",
        "query": "Did saved lessons improve recognition on a different later penalty?",
        "expected_source": "examples/lesson-memory/transfer/README.md",
    },
    {
        "id": "reviewer",
        "query": "Why was the optional reviewer turned off for the penalty commentary?",
        "expected_source": "examples/portfolio-demo/README.md",
    },
)


def prepare(out: Path) -> dict:
    """Copy only the explicit report allowlist, preserving original UTF-8 bytes."""
    out.mkdir(parents=True, exist_ok=False)
    objects = []
    for source in SOURCES:
        data = (ROOT / source).read_bytes()
        # Plain text retains original punctuation/markup for auditable char spans.
        key = source + ".txt"
        target = out / "corpus" / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        objects.append(
            {
                "key": key,
                "source": source,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "purpose": "offline review only",
        "objects": objects,
        "total_bytes": sum(item["bytes"] for item in objects),
        "questions": QUESTIONS,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def report(capture: dict, archive: Path) -> str:
    """Display exact returned text and verify it against the uploaded snapshot."""
    manifest = json.loads((archive / "manifest.json").read_text())
    known = {item["key"]: item for item in manifest["objects"]}
    lines = [
        "# Graphon review archive: captured retrieval",
        "",
        "Saved service responses, not generated answers or live playback.",
        "",
    ]
    if "indexAtQuery" in capture:
        counts = capture["indexAtQuery"]["counts"]
        lines += [
            f"Index at query: {counts['ready']} ready; "
            f"{counts['indexing']} still indexing; {counts['failed']} failed; "
            f"{counts['unsupported']} unsupported.",
            "",
        ]
    for run in capture["runs"]:
        lines += [f"## {run['query']}", ""]
        search = run["search"]
        lines += [f"Search time reported by Graphon: {search['timingMs']['total']} ms.", ""]
        for warning in search.get("warnings", []):
            lines += [f"Warning: {warning['code']}: {warning['message']}", ""]
        requested = [span for hit in search["results"] for span in hit["spans"]]
        read = run["read"]
        if read.get("next"):
            raise ValueError("incomplete read: capture the remaining spans first")
        if len(requested) != len(read["results"]):
            raise ValueError("search/read span count mismatch")
        for expected, result in zip(requested, read["results"], strict=True):
            span = result["span"]
            if span != expected or span["key"] not in known:
                raise ValueError("unexpected source or changed span")
            item = known[span["key"]]
            data = (archive / "corpus" / span["key"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("archive source hash changed")
            if result.get("error"):
                lines += [f"Read failed for {item['source']}: {result['error']}", ""]
                continue
            text = result.get("text", "")
            exact = span["unit"] == "chars" and data.decode()[span["start"] : span["end"]] == text
            lines += [
                f"### {item['source']}",
                "",
                f"Version: `{span['versionId']}`; {span['unit']} "
                f"{span.get('start')}–{span.get('end')}. "
                f"Exact snapshot slice: {'yes' if exact else 'no'}.",
                "",
            ]
            lines += ["> " + line for line in text.splitlines()]
            lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "runs/graphon-review/demo-v1")
    parser.add_argument("--capture", type=Path, help="Replay saved search/read responses")
    args = parser.parse_args()
    if args.capture:
        print(report(json.loads(args.capture.read_text()), args.archive))
    else:
        print(json.dumps(prepare(args.archive), indent=2))
