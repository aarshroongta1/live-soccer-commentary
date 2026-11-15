# Record the engineering demo

Open **http://127.0.0.1:3000/demo/replay** (also linked from `/demo`).
This is a debugging view over a saved run: footage/audio, the original dialogue
turns, supporting frames, identity tool records, and raw JSON. It is a viewer
added for inspecting existing artifacts, not a recording of the original tooling.

## Run

From the repository root:

```sh
node scripts/prepare_replay.mjs
cd web
npm install # only if dependencies are missing
npm run build
npm run start -- --port 3000
```

For changes, use `npm run dev -- --port 3000` instead. Stop the other server first.
Preparation needs `ffmpeg`, `ffprobe`, the original
`clips/action-sources/england-france-penalties.mp4`, and the voiced render
`web/public/soccer-commentary/prepared-penalty-agents.mp4`. All are already
available in this workspace. Media and extracted frames are ignored by Git;
a fresh clone needs the same source assets restored. No API calls are made.

## What to talk through (about 90 seconds)

Use a desktop browser at 1440×900 or larger, 100% zoom. Capture system/tab audio.

1. Play the clip. Explain what you built: a vision observer, an identity stage
   that can request roster context, and separate caller and analyst agents.
   This example is prepared commentary, not live inference.
2. Pause and click the **3.000 / caller / speak** row. Point to `recorded-2`,
   the actual supporting frames, and the `referee_signal` cue. The caller's
   payload includes the evidence ID, reason, tokens, model and request latency.
   Explain that the referee signal supports the penalty decision.
3. Click **identity_steps [4]**. The right panel shows the saved
   `research_roster` action and returned squad. Explain why the observer reports
   a shirt number and a separate identity step maps it to a researched name.
   The original lookup batch spans multiple observations; its placement at 9s
   is an inspection shortcut, not an execution timestamp.
4. Click **10.000 / caller / speak**. Show `recorded-4`, ball-in-net / goal-line
   cues and Harry Kane as the resolved actor. Resume to hear the original line.
   Compare it with the analyst turn at 11.888s: separate role, same goal evidence,
   different output. Source cues and model latency are separate columns.
5. Click **16.469 / analyst / rejected**. The original duplicate draft is
   still in the JSON; the validation error is `speaker already covered this
   evidence`. Contrast it with the caller's **abstain** at the same source time.
   Both rows are independently selectable. At 17s, the caller abstains on replay.
6. Use **run config** to discuss the observer prompt, model, saved source hash,
   budget guard and reused vision. Use **observations** for the structured
   evidence. The full trace download preserves every record.

Checking “follow playback” resumes automatic selection. Clicking any turn pauses
and pins that exact row; pressing play follows the video again. The filter matches
role, action and text. This lets you navigate failures while talking about the
implementation rather than following a presentation script.

## Artifact chain and limits

- Trace: `web/public/soccer-commentary/prepared-penalty.trace.json`.
- Plan: `examples/recorded-agent-demo/penalty-plan.json` (five original lines).
- Speech placement: `examples/recorded-agent-demo/render-report.md`.
- `web/data/replay-delivery.json` is derived from that measured report.
- Still frames come from the source at `42 + support_at_s`, with no drawn or
  generated detections. The descriptions and cues are saved model claims.
- Full-sequence vision used 24 seconds of context; dialogue snapshots filter
  evidence by source-time cutoff. This does not establish causal live inference.
- Identity records have no playback timestamp. They are displayed by relevant
  evidence, not reconstructed execution time.
- Final dialogue cost: $0.0099225, excluding reused preparation. No paid calls
  were made to implement or verify this viewer. Audio is the original synthesized
  delivery, not the later edited showcase wording.

## Verify

`node scripts/prepare_replay.mjs` validates source hash, exact text, evidence
cutoffs, non-overlapping measured delivery and 20-second audio/video streams.
`npm run build --prefix web` checks production compilation and TypeScript.
Browser verification covers playback, exact turn selection (including simultaneous
turns), identity records, evidence frames, raw payloads, filtering, follow mode,
and desktop/mobile overflow. Stream presence and sound-enabled playback are
checked; voice quality has not received an independent listening evaluation.
