# Demo render report

- Input mode: precomputed offline plan (not live commentary).
- Plan: `examples/recorded-agent-demo/continuity/candidate-plan.json`
- Clip: `clips/action-sources/england-france-penalties.mp4`
- Window: 20.000 seconds; source video starts at 42.000 s.
- Alignment: caller cues preserved; analyst follows measured caller audio by 0.25 s, never before cited evidence is available.
- Speech order: plan order without overlap; goal calls can trim preceding speech.
- Broadcast audio: muted; the output contains only synthesized commentary.
- Voice assignment: caller `Custom caller`; analyst `Custom analyst` (distinct).
- Captions: bottom ASS captions from measured audio timing; sidecar retained in the local run directory.

## Measured audible speaking balance

Measured after goal-priority and clip-end trimming; this reports the selected plan, not a quota.
- Caller (Custom caller): 4.92 s (60.2%)
- Analyst (Custom analyst): 3.25 s (39.8%)

| # | voice | plan time | placed | delay | end | status | text |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | caller | 1.00 | 1.00 | 0.00 | 2.01 | on cue | England drive on. |
| 2 | caller | 3.00 | 3.00 | 0.00 | 5.19 | on cue | Penalty to England! They've got the big decision. |
| 3 | caller | 9.00 | 9.00 | 0.00 | 9.54 | on cue | Kane... |
| 4 | caller | 10.00 | 10.00 | 0.00 | 11.18 | on cue | Kane scores! |
| 5 | analyst | 14.66 | 11.43 | 0.00 | 14.68 | on cue | He sends the keeper the wrong way, keeping it low and precise. |
