# Demo render report

- Input mode: precomputed offline plan (not live commentary).
- Plan: `examples/recorded-agent-demo/penalty-plan.json`
- Clip: `clips/action-sources/england-france-penalties.mp4`
- Window: 20.000 seconds; source video starts at 42.000 s.
- Alignment: each line uses its at_s cue relative to the selected clip window.
- Speech order: plan order without overlap; goal calls can trim preceding speech.
- Broadcast audio: muted; the output contains only synthesized commentary.
- Voice assignment: caller `Custom caller`; analyst `Custom analyst` (distinct).
- Captions: bottom ASS captions from measured audio timing; local sidecar retained under `runs/recorded-agents/penalty-full-20261108-delivery/`.

## Measured audible speaking balance

Measured after goal-priority and clip-end trimming; this reports the selected plan, not a quota.
- Caller (Custom caller): 3.44 s (34.3%)
- Analyst (Custom analyst): 6.59 s (65.7%)

| # | voice | plan time | placed | delay | end | status | text |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | caller | 1.00 | 1.00 | 0.00 | 2.02 | on cue | England driving in! |
| 2 | caller | 3.00 | 3.00 | 0.00 | 4.07 | on cue | Penalty England! |
| 3 | analyst | 4.12 | 4.12 | 0.00 | 7.37 | on cue | The challenge drew the referee’s penalty signal in the box. |
| 4 | caller | 10.00 | 10.00 | 0.00 | 11.35 | on cue | Kane scores the penalty! |
| 5 | analyst | 11.89 | 11.89 | 0.00 | 15.23 | on cue | Kane’s low penalty beats the diving keeper into the left side. |
