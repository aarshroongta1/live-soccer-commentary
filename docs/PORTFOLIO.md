# Portfolio demo

The local `/` and `/demo` pages lead with three selected, silent captioned clips:

| Clip | Length | Provenance |
| --- | --- | --- |
| England vs France penalty | 20s | New automatic caller/analyst dialogue on saved vision and cited squad research. Exact generated words. |
| Arsenal vs Norwich team goal | 28s | Existing manually reviewed and edited showcase. |
| Argentina vs France counterattack | 28s | Existing manually reviewed and edited showcase. |

The automatic example includes transcript seeking, source-time turns, rejected
turns, visual observations and a downloadable preparation trace. The other two
illustrate the intended presentation and are labelled **Edited showcase**.
The page explains the agent roles, evaluation findings and recorded/live limits.
Saved causal replay runs remain in an expandable section.

## Review locally

```bash
npm run build --prefix web
npm run start --prefix web -- --port 3000
```

Stop an existing production preview before building over `.next`. Restart after
adding public media. Browsing clips and traces makes no model calls.

The separate integrated portfolio page at
`https://www.aarshroongta.com/projects/soccer-commentary` lives in the
`personal-website` repository. These local changes do not update or deploy that
page. The five older curated plans and videos remain available; the focused
player selects two of them alongside the automatic example.

## Automatic example

Durable artifacts are in [examples/portfolio-demo](../examples/portfolio-demo/):
`penalty-plan.json`, `penalty-trace.json` and `selection.json`.
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

Media remains local and git-ignored. Expected player files are:

- `web/public/soccer-commentary/portfolio-penalty.mp4`
- `web/public/soccer-commentary/portfolio-arsenal-norwich.mp4`
- `web/public/soccer-commentary/portfolio-argentina-france.mp4`

The automatic clip is rendered without audio, TTS or model calls:

```bash
uv run python scripts/render_text_demo.py \
  --plan examples/portfolio-demo/penalty-plan.json \
  --clip clips/action-sources/england-france-penalties.mp4 \
  --out runs/demo/portfolio-penalty.mp4
```

The script refuses to overwrite outputs. Captions preserve source cues and exact
words; display intervals use estimated reading time and are not speech latency.
The curated player copies remove audio from the existing captioned videos with
FFmpeg `-map 0:v:0 -c:v copy -an`. The original voiced media remains unchanged.

`web/data/portfolio-demo.json` supplies the automatic transcript and run summary;
`web/data/demo-clips.json` retains all five edited showcase transcripts. Optional
remote sources can be configured with `PORTFOLIO_PENALTY_VIDEO_URL`,
`PORTFOLIO_ARSENAL_VIDEO_URL` and `PORTFOLIO_ARGENTINA_VIDEO_URL`. A fresh clone
without media displays its transcripts and missing-media guidance.
