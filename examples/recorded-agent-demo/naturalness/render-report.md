# Demo render report

- Input mode: precomputed offline plan (not live commentary).
- Plan: `examples/recorded-agent-demo/naturalness/candidate-plan.json`
- Clip: `clips/action-sources/england-france-penalties.mp4`
- Window: 20.000 seconds; source video starts at 42.000 s.
- Alignment: each line uses its at_s cue relative to the selected clip window.
- Speech order: plan order without overlap; goal calls can trim preceding speech.
- Broadcast audio: muted; the output contains only synthesized commentary.
- Voice assignment: caller `Custom caller`; analyst `Custom analyst` (distinct).
- Captions: bottom ASS captions from measured audio timing; local sidecar under `runs/recorded-agents/penalty-natural-quality-v2/`.

## Measured audible speaking balance

Measured after goal-priority and clip-end trimming; this reports the selected plan, not a quota.
- Caller (Custom caller): 6.55 s (66.8%)
- Analyst (Custom analyst): 3.25 s (33.2%)

| # | voice | plan time | placed | delay | end | status | text |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | caller | 1.00 | 1.00 | 0.00 | 2.21 | on cue | England drive on. |
| 2 | caller | 3.00 | 3.00 | 0.00 | 5.83 | on cue | Penalty to England! They've got the big decision. |
| 3 | caller | 10.00 | 10.00 | 0.00 | 12.51 | on cue | Harry Kane scores for England! Cool as you like. |
| 4 | analyst | 14.66 | 14.66 | 0.00 | 17.91 | on cue | He sends the keeper the wrong way, keeping it low and precise. |
