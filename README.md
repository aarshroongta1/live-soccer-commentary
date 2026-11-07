# Soccer Commentary

[Watch the demo](https://www.aarshroongta.com/projects/soccer-commentary)

AI commentary for short soccer clips, with a play-by-play caller and an analyst.
The recorded workflow turns match footage into a script, then adds two voices and
captions to the video. A browser experiment also generates commentary text while
a local video or shared screen plays.

## How it works

A LangGraph workflow gathers cited pre-match research, reads sampled video frames,
and writes both speaking roles together. Player names are resolved against a
supplied roster. Each line references visual observations or research; Python
checks those references and the script's timing before rendering.

Observations are saved, so the script can be revised without repeating vision
calls. FFmpeg combines the video, speech, and captions. Speech uses macOS voices
by default, with optional ElevenLabs synthesis.

Built with **Python, LangGraph, Pydantic, OpenAI, OpenCV, FFmpeg, and
Next.js/TypeScript**.

[View the architecture flowchart](diagrams/commentary-pipeline.svg), or open
the [editable diagram](diagrams/commentary-pipeline.excalidraw) in Excalidraw.

## Explore the demo

Requires Node.js/npm. The saved demo makes no model calls and needs no API key:

```bash
cd web
npm ci
npm run dev
```

Open http://127.0.0.1:3000 for five recorded examples: Di María vs France,
Messi vs Bayern, Arsenal vs Norwich, Ronaldo vs Spain, and Ferran Torres vs Betis.
The showcase scripts are reviewed and edited before voicing; see their
[plans and review notes](examples/demo-showcase/README.md). Each has a seekable
transcript. The matching MP4s live in
`web/public/soccer-commentary/`. Source
footage and generated media are not committed. Without the videos, the saved
transcripts still work. See the [portfolio guide](docs/PORTFOLIO.md) for media setup
and deployment.

## Generate your own commentary

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and FFmpeg. Copy
`.env.example` to `.env` and set `OPENAI_API_KEY` for paid generation:

```bash
uv sync --extra demo --extra voice --dev
```

Follow the [recorded demo guide](docs/DEMO.md) to generate, review, and render a
script. The [checked-in example](examples/recorded-demo/) includes reviewed output
for inspection without an API key.

The [screen commentary experiment](docs/LIVE.md) runs at `/live` on localhost.
Choose a local clip or shared screen, confirm the paid-call notice, and press
**Start commentary**. Public deployments show local setup instructions.

Run offline tests with `uv run pytest -q`. Check the website with
`cd web && npm run lint && npm run build`.

## Replay agents

An experimental replay path separates the researcher, visual observer, caller,
and analyst. Caller and analyst each run a bounded LangGraph tool loop; an asyncio
coordinator advances the video clock, rejects stale results, and prioritizes goals.
The existing recorded workflow and public showcase remain available.

Try the synthetic diagnostic without credentials:

```bash
uv run python scripts/replay_commentary.py --diagnostic --rate 10 \
  --out runs/replay/diagnostic.json
uv run python scripts/evaluate_replay.py --run runs/replay/diagnostic.json \
  --reference examples/replay-evaluation/diagnostic.json \
  --out runs/replay/diagnostic-score.json
```

This exercises agent-selected tools and speech scheduling using fictional evidence
and scripted model responses. Speech timing is simulated. See the
[replay guide](docs/REPLAY.md) to run model agents over a local video, inspect traces,
and understand what the initial evaluation does and does not measure.

## Current limits

Screen commentary is text-only and trails the action by several seconds. Player
identification and descriptions can be wrong; valid evidence references do not
guarantee factual accuracy. Recorded scripts need review before voicing. The
checked-in example includes documented manual corrections.
