# Soccer Commentary

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

## Run locally

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and Node.js/npm.
Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to generate commentary.

```bash
uv sync --extra demo --extra voice --dev
cd web
npm ci
npm run dev
```

Open [the live page](http://127.0.0.1:3000/live), choose a local video or share a
match window, and select **Generic match** for your own footage. Model calls begin
when you confirm the notice and press **Start commentary**.

The saved player at `/demo` needs a rendered MP4 in `web/public/`. Source footage
and generated media are not included in the repository. See the
[recorded demo guide](docs/DEMO.md) for rendering instructions, the
[screen commentary guide](docs/LIVE.md) for capture details, or the
[example script](examples/recorded-demo/) to inspect output without an API key.

Run the offline tests from the repository root with `uv run pytest -q`.

## Current limits

Screen commentary is text-only and trails the action by several seconds. Player
identification and descriptions can be wrong; valid evidence references do not
guarantee factual accuracy. Recorded scripts need review before voicing. The
checked-in example includes documented manual corrections.
