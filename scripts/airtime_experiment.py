"""Replay saved model outputs through the real speech-floor code, without API calls.

This isolates scheduling. It does NOT rerun perception, generation, or a complete
causal agent loop: model outputs and availability remain fixed even when altered
speech history would have changed subsequent model prompts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from commentary.recorded_demo import Observation
from commentary.replay.airtime import YieldingReplayRuntime
from commentary.replay.contracts import ReplayResult, RuntimeSnapshot, SpeechLine, TurnResult
from commentary.replay.runtime import ReplayConfig, ReplayRuntime, _Candidate

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/trajectory-outreach"


async def unused(*args: object) -> None:
    raise AssertionError("This experiment must not call agents or vision")


def replay_floor(
    source: dict,
    *,
    yielding: bool,
    analyst_delay: float = 0,
    caller_delay: float = 0,
    words_per_second: float = 3,
) -> dict:
    now = [0.0]
    cls = YieldingReplayRuntime if yielding else ReplayRuntime
    config = dict(source["metadata"]["config"])
    config["speech_words_per_second"] = words_per_second
    runtime = cls(unused, unused, config=ReplayConfig(**config), monotonic=lambda: now[0])
    # Deliberately exercise the same private floor implementation as production.
    # The adapter owns no agent tasks and does not use run().
    runtime._result = ReplayResult()
    snapshots = {}
    pending = {}
    actions = {}
    events = []
    for event in source["trace"]:
        kind, data, role = event["event"], event["data"], event["role"]
        if kind == "observation_available":
            pending[data["snapshot_version"]] = data
            events.append((event["at_s"], "observe", copy.deepcopy(data)))
        elif kind == "turn_started":
            version = data["snapshot_version"]
            observations = [
                o for v, batch in pending.items() if v <= version for o in batch["observations"]
            ]
            snapshots[(role, version)] = RuntimeSnapshot(
                version=version,
                cutoff_s=data["cutoff_s"],
                observations=[Observation.model_validate(o) for o in observations],
            )
        elif kind == "agent_action":
            actions[role] = data["action"]
        elif kind == "speech_queued":
            snapshot = snapshots[(role, data["snapshot_version"])].model_copy(deep=True)
            action = actions[role]
            assert action["kind"] == "speak"
            line = SpeechLine.model_validate(action["line"])
            candidate = _Candidate(
                TurnResult(
                    role=role,
                    snapshot_version=snapshot.version,
                    cutoff_s=snapshot.cutoff_s,
                    line=line,
                ),
                snapshot,
            )
            delay = caller_delay if role == "caller" else analyst_delay
            events.append((event["at_s"] + delay, "ready", candidate))
    events.sort(key=lambda event: event[0])
    while events or runtime._active is not None or runtime._queue:
        wake = events[0][0] if events else float("inf")
        if runtime._active is not None:
            wake = min(wake, runtime._active.ended_s)
        elif runtime._queue:
            wake = min(wake, max(now[0], runtime._next_speech_s))
        now[0] = wake
        while events and events[0][0] <= wake:
            _, kind, data = events.pop(0)
            if kind == "observe":
                runtime._result.observations.extend(
                    Observation.model_validate(o) for o in data["observations"]
                )
            else:
                runtime._validate_turn(data.result, data.result.role, data.snapshot)
                runtime._queue[data.result.role] = data
        runtime._speech()
    result = runtime._result.model_dump(mode="json")
    caller = next((s for s in result["speech"] if s["line"]["voice"] == "caller"), None)
    ready = next(
        at
        for at, kind, _ in sorted(
            [
                (e["at_s"] + caller_delay, e["event"], e)
                for e in source["trace"]
                if e["event"] == "speech_queued" and e["role"] == "caller"
            ]
        )
        if kind == "speech_queued"
    )
    result["metrics"] = {
        "caller_ready_s": ready,
        "caller_started_s": caller["started_s"] if caller else None,
        "caller_floor_wait_s": caller["started_s"] - ready if caller else None,
        "caller_event_age_s": caller["started_s"] - caller["event_at_s"] if caller else None,
        "caller_preserved": caller is not None,
        "analyst_spoken_words_estimate": sum(
            (s["ended_s"] - s["started_s"]) * words_per_second
            for s in result["speech"]
            if s["line"]["voice"] == "analyst"
        ),
    }
    return result


def experiment(source_path: Path) -> dict:
    source = json.loads(source_path.read_text())
    baseline = replay_floor(source, yielding=False)
    candidate = replay_floor(source, yielding=True)
    recorded = next(s for s in source["speech"] if s["line"]["voice"] == "caller")
    fidelity = abs(recorded["started_s"] - baseline["metrics"]["caller_started_s"])
    assert fidelity < 0.01, "Baseline replay must reproduce the original caller within 10 ms"
    sweep = []
    # Fixed, exhaustive grid; these are 15 timing variants of ONE saved clip.
    for analyst_delay in (0, 0.5, 1):
        for caller_delay in (0, 0.5, 1, 2, 4):
            a = replay_floor(
                source, yielding=False, analyst_delay=analyst_delay, caller_delay=caller_delay
            )
            b = replay_floor(
                source, yielding=True, analyst_delay=analyst_delay, caller_delay=caller_delay
            )
            sweep.append(
                {
                    "analyst_delay_s": analyst_delay,
                    "caller_delay_s": caller_delay,
                    "baseline": a["metrics"],
                    "yielding": b["metrics"],
                    "baseline_speech": a["speech"],
                    "yielding_speech": b["speech"],
                }
            )
    sensitivity = []
    for wps in (2.5, 3, 3.5):
        a = replay_floor(source, yielding=False, words_per_second=wps)
        b = replay_floor(source, yielding=True, words_per_second=wps)
        sensitivity.append(
            {"words_per_second": wps, "baseline": a["metrics"], "yielding": b["metrics"]}
        )
    return {
        "title": "A correct line can still miss its moment",
        "run_date": "2026-11-22",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_original": "runs/replay/recognition-pass-20261107/save.json",
        "source_revision": "08c0497",
        "new_api_calls": 0,
        "new_cost_usd": 0,
        "timing": "Simulated speech, word count / 3 words per second; no new audio",
        "baseline_fidelity_error_s": fidelity,
        "recorded_caller_started_s": recorded["started_s"],
        "baseline": baseline,
        "yielding": candidate,
        "sweep": sweep,
        "sensitivity": sensitivity,
        "observations": [e for e in source["trace"] if e["event"] == "observation_available"],
        "limitations": [
            "One development clip; 15 correlated timing variants, not independent trials.",
            "Fixed saved VLM observations and model text; no new perception or generation.",
            "Evidence is model-reported. Earlier analyst text overstates uncertain observations.",
            "Only the speech floor ran. Changed history could change later generated words.",
            "Mid-sentence interruption is a cost; no word-boundary audio or listener study ran.",
            "No Trajectory API, Continual Grading, training, RL, or model improvement was tested.",
            "Policy and initial preference are agent-authored; no human expert labels collected.",
            "Eight-second clip; clock includes processing and drain after footage ends.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=EXAMPLE / "source-replay.json")
    parser.add_argument("--out", type=Path, default=EXAMPLE / "result.json")
    args = parser.parse_args()
    result = experiment(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "baseline": result["baseline"]["metrics"],
                "yielding": result["yielding"]["metrics"],
                "baseline_fidelity_error_s": result["baseline_fidelity_error_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
