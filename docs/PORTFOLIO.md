# Portfolio demo

The public demo is https://www.aarshroongta.com/projects/soccer-commentary.
It presents three clips with edited demo commentary, AI caller/analyst voices
and captions: England–France (19s), Arsenal–Norwich (18s), Argentina–France (20s).

The current scripts were rewritten after the user rejected the previous narration
for sparse coverage and weak commentary. Each begins at 0.15s, follows the actual
build-up, calls the goal, and uses replay/celebration for analysis. The penalty
retains the successful original Kane goal-call recording. Its final second of
repeated replay is trimmed. No automatic-quality gain is claimed for these edits.

The native personal-website showcase lives at the URL above. Videos use the
soccer-commentary-demo.vercel.app media origin. The “Edited demo commentary” label
makes the presentation provenance visible. “How I built it” links to the GitHub
README architecture section. The separate saved-run inspector retains the original
agent words, evidence and comparison artifacts.

The current catalog is web/data/edited-showcase.json, mirrored into the personal
website. Reproduction and pacing checks are in
[the edited showcase record](../examples/portfolio-demo/edited-showcase/README.md).
Earlier sections below describe archived automatic/silent/first-voice variants.

## Review locally

```bash
npm run build --prefix web
npm run start --prefix web -- --port 3000
```

Stop an existing production preview before building over `.next`. Restart after
adding public media. Browsing clips and traces makes no model calls.

The public showcase lives in the `personal-website` repository, with the same
three featured clips and the website's theme/navigation. Its video assets use
`https://soccer-commentary-demo.vercel.app` as their media origin. The standalone
Next.js deployment also retains the optional saved-run inspector.

[The demo walkthrough](DEMO_WALKTHROUGH.md) gives a two-minute presentation, a
small code-reading route, interview questions and a resume bullet.
[The fresh-run record](../examples/portfolio-demo/release/README.md) documents all
four attempts, the $0.27827 reported spend, and the editorial changes.

## Automatic example

Durable artifacts are in [examples/portfolio-demo](../examples/portfolio-demo/):
`penalty-plan.json`, `penalty-trace.json`, `before-plan.json`, `before-trace.json`,
`iteration.json` and `selection.json`.
The source is seconds 42–62 of the existing England–France footage. The model
receives saved visual readings and pre-match context, never evaluation labels or
an earlier commentary script. Roster lookups are rerun; the observer proposes
Harry Kane for readable number 9. Caller and analyst have separate LangGraph turns.

Two bounded development attempts were made, with a combined $0.24 maximum.
Reported token cost was **$0.02666745**, including identity and dialogue calls,
but excluding the prior saved vision and research. The selected second attempt
used nine calls and $0.0087087. Its optional reviewer was disabled after attempt
one rejected even a valid short goal reaction. Both attempts are retained; the
selection is not a quality benchmark or a controlled reviewer ablation.

An early oversized line in attempt two was rejected. The retained output contains
four unedited lines: penalty award, run-up, goal, then analysis. It was checked
against the existing source window for those claims. No flawless general action
recognition or continuous real-time speech is claimed.

The [portfolio case study](PORTFOLIO_CASE_STUDY.md) provides resume wording,
architecture decisions, evaluation findings and interview talking points.

## Media and reproduction

Media remains local and git-ignored. Current player files are:

- `web/public/soccer-commentary/showcase-penalty-edited-v2.mp4`
- `web/public/soccer-commentary/showcase-arsenal-edited-v2.mp4`
- `web/public/soccer-commentary/showcase-argentina-edited-v2.mp4`

Recruiter variants omit the burned-in preparation labels using `--label ''`.
Their words, source cuts and caption cues match the annotated engineering versions.
The original silent base clip is rendered without TTS or model calls:

```bash
uv run python scripts/render_text_demo.py \
  --plan examples/portfolio-demo/penalty-plan.json \
  --clip clips/action-sources/england-france-penalties.mp4 \
  --out runs/demo/portfolio-penalty.mp4
```

The script refuses to overwrite outputs. Captions preserve source cues and exact
words; display intervals use estimated reading time and are not speech latency.
The new Arsenal and Argentina clips use the same renderer with their edited plans;
reproduction commands are in the release record. Original curated media remains archived.

`web/data/portfolio-demo.json` supplies the automatic transcript and run summary;
`web/data/featured-demos.json` supplies the new lightly edited examples;
`web/data/demo-clips.json` retains all five archived showcase transcripts. Optional
remote sources can be configured with `PORTFOLIO_PENALTY_VIDEO_URL`,
`PORTFOLIO_ARSENAL_VIDEO_URL`, `PORTFOLIO_ARGENTINA_VIDEO_URL` and
`PORTFOLIO_BEFORE_VIDEO_URL`. A fresh clone
without media displays its transcripts and missing-media guidance.

## Narrated showcase

The public player starts unmuted when the viewer presses Play. The three
`showcase-*-voiced.mp4` variants copy the existing captioned video stream and mix
saved ElevenLabs caller/analyst speech at the plan cues. No original broadcast
audio is mixed in. Exact words are retained. The short penalty run-up phrase is
sped up 1.2255x to finish before the goal call; the other nine lines use their
original generated speed. Existing captions retain their reading intervals.

Reproduce from saved speech (no API calls):

```bash
uv run python scripts/add_demo_narration.py \
  --plan examples/portfolio-demo/penalty-plan.json \
  --video web/public/soccer-commentary/showcase-penalty.mp4 \
  --manifest runs/demo/showcase-narration-v1/penalty/manifest.json \
  --out web/public/soccer-commentary/showcase-penalty-voiced.mp4
```

The command refuses existing outputs. Per-clip `.narration.json` reports record
input/output hashes, exact words, measured durations, voice IDs and placement.
Speech generation used `eleven_flash_v2_5`, 10 requests, 497 characters and 249
reported provider character-cost credits; no retries or new commentary calls.
The [advertised Flash API rate](https://elevenlabs.io/pricing/api) gives an
estimated $0.01988 before tax; actual
account billing was not read. No subjective voice-quality benchmark is claimed.
