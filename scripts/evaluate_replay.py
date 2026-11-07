#!/usr/bin/env python3
"""Score a saved replay against external references; never feeds labels to agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commentary.replay.contracts import ReplayResult
from commentary.replay.evaluation import EvaluationCase, evaluate_replay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.out}")
    result = ReplayResult.model_validate_json(args.run.read_text())
    reference = EvaluationCase.model_validate_json(args.reference.read_text())
    scorecard = evaluate_replay(result, reference, tolerance_s=args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scorecard, indent=2) + "\n")
    print(
        f"Evaluation saved to {args.out}; check provenance and limitations before comparing runs."
    )


if __name__ == "__main__":
    main()
