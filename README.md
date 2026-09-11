# Live Soccer Commentary

Two AI voices call a live soccer broadcast in real time, from a screen capture,
using only what a human commentator has: the picture, the sound, and notes
prepared before kickoff.

**No live data feed reaches any agent at runtime.** The score comes from reading
the scoreboard on screen. Names come from the roster, shirt numbers, and
on-screen graphics. Events come from looking at the pitch. A play-by-play feed
is used only afterwards, as ground truth for the eval.

Status: day 1 of a two-week sprint. See [`PLAN.md`](PLAN.md).

## How it works

Frames and audio land in a ring buffer. Perception runs at the live edge, but
the narration cursor trails it by a few seconds, so the caller can peek at what
happens next before it commits to a line — the shot it is describing has already
gone in, or has not. The viewer watches the video from the cursor too, which is
what makes the delay invisible.

```
screen capture ─► delay buffer ─┬─► board reader (score bug)
                                ├─► caller (vision, structured output)
                                └─► analyst (lulls, MCP tools)
                                        │
                        match state ─► fact gate ─► director ─► two voices
```

Full architecture, agent roster, and evaluation plan: [`PLAN.md`](PLAN.md).
Research behind the design choices: [`docs/research/`](docs/research).

## Quickstart

```bash
uv sync --all-extras --dev
cp .env.example .env          # add ANTHROPIC_API_KEY
bash scripts/list_devices.sh  # pick "<screen>:<audio>" for AVFOUNDATION_DEVICE
uv run python -m commentary   # 10 s of capture stats — the day 1 gate
```

Play any match video full screen and the frames stream into Python.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
uv run mypy          # types
uv run pre-commit install
```

## Rights

Broadcast footage is processed locally and never committed or hosted. Only short
clips are published for demos.
