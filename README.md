# Live Soccer Commentary

Two AI voices call a live soccer broadcast in real time, from a screen capture,
using only what a human commentator has: the picture, the sound, and notes
prepared before kickoff.

**No live data feed reaches any agent at runtime.** The score comes from reading
the scoreboard on screen. Names come from the roster, shirt numbers, and
on-screen graphics. Events come from looking at the pitch. A play-by-play feed
is used only afterwards, as ground truth for grading.

## How it works

Frames and audio land in a ring buffer. Perception runs at the live edge, but
the narration cursor trails it by a few seconds, so the caller can peek at what
happens next before it commits to a line — the shot it is describing has already
gone in, or has not. The viewer watches the video from the cursor too, which is
what makes the delay invisible.

```
screen capture ─► delay buffer ─┬─► board reader (score bug)
                                ├─► caller (vision, structured output)
                                └─► analyst (lulls, match-state tools)
                                        │
                        match state ─► fact gate ─► director ─► two voices
```

One subtlety governs the whole runtime. The board is read at the **live edge**,
because that is how the system learns a goal went in before the cursor reaches
it. But a board change is not applied to match state until the cursor passes the
moment it happened. The evidence arrives early; the belief arrives on time, so
the commentary can never announce something the viewer has not been shown.

Full architecture, agent roster, and evaluation plan: [`PLAN.md`](PLAN.md).
Research behind the design choices: [`docs/research/`](docs/research).

## Quickstart

```bash
uv sync --all-extras --dev
uv run python -m commentary run --source sim --seconds 30
```

That calls a match end to end with **no API key and no footage** — see below.

To call something real:

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
bash scripts/list_devices.sh    # pick "<screen>:<audio>" for AVFOUNDATION_DEVICE
uv run python -m commentary run --source screen --backend anthropic --serve
```

Then open <http://127.0.0.1:8000> and play a match full screen.

## The simulator

There is a synthetic broadcast in [`src/commentary/sim/`](src/commentary/sim):
a rendered pitch, twenty-two dots, a ball, a camera that follows the play, a
score bug in the same crop a real broadcaster uses, replay segments with the bug
removed, and lower-third name graphics. It carries a machine-readable timestamp
burned into the frame, and it ships a written record of everything that happened.

It exists for two reasons. It makes the whole pipeline runnable before any
footage is in hand, and — more usefully — it gives the grader a ground truth that
a real broadcast cannot. `SimOracle` stands in for the model and can be told to
lie at a set rate, injecting the three errors the fact gate exists to stop: a
name on no roster, a scoreline that contradicts the board, and a goal that never
happened. That turns "the fact gate seems to work" into a number.

```bash
uv run python -m commentary sim --out /tmp/match.mp4     # watch it
uv run python -m commentary run --source sim --error-rate 0.35 --seconds 35
```

The oracle models two different failures, because they behave differently.
Hallucination — an invented name, a scoreline that contradicts the board, a goal
that never happened — happens at a fixed rate whatever the buffer depth. Getting
an outcome wrong is different: when the lookahead does not reach far enough to
show how a move ended, the oracle has to guess, and guesses wrong. That second
one is the failure the delay buffer exists to prevent, and it is the only reason
the delay curve below means anything.

## Ablations

```bash
uv run python -m commentary.grading.baselines --duration 600 --seconds 150 \
    --error-rate 0.3 --speed 1
```

Every variant against the same seeded match, with the stand-in model
hallucinating on 30% of calls and guessing wrong on 45% of outcomes it cannot
see resolve:

| run | lines | factual err | recall | gate rej | lag p50/p95 s | silence |
|---|---:|---:|---:|---:|---:|---:|
| worldcupvoice | 24 | 87.5% | 95% | 0.0% | 0.0 / 0.0 | 64% |
| **full** | 31 | **6.5%** | 100% | 54.8% | 8.0 / 8.0 | 57% |
| no-gate | 39 | 71.8% | 100% | 0.0% | 8.0 / 8.0 | 39% |
| no-delay | 30 | 3.3% | 100% | 65.4% | 0.0 / 0.0 | 61% |
| single-voice | 33 | 0.0% | 100% | 43.1% | 8.0 / 8.0 | 61% |

The fact gate is the result. Switching it off takes factual error rate from 6.5%
to 71.8% — the caller proposes plenty of nonsense either way, and the gate is
the only thing standing between that and a voice. The reproduced worldcupvoice
loop sits at 87.5%, which is what an ungated caller with no state and no roster
does: 11 invented names, 7 goals that never happened, 3 wrong scorelines.

### The delay

| delay | 0 s | 2 s | 4 s | 8 s |
|---|---:|---:|---:|---:|
| factual error | 6.2% | 5.7% | 3.1% | 6.1% |
| phantom goals reaching air | 0 | 0 | 0 | 0 |
| `unconfirmed_goal` rejections | 25 | 21 | 24 | 21 |

**No delay effect is visible in the error rate.** At ~32 lines per variant a
single error moves the number three points, so that top row is noise around one
value.

The other two rows are there because they used to say something alarming. The
gate originally accepted a board change anywhere in `[cursor - 2, cursor +
delay_s]`, so the confirmation window *widened with the buffer*: choosing to
wait longer also made the gate accept a scoreboard change further from the
moment being called. Phantom goals reaching air ran 1, 1, 4, 6 as the buffer
deepened, and `unconfirmed_goal` rejections fell 33, 32, 25, 14. The delay was
buying the caller information and paying for it by loosening the check. With
the window fixed to a constant — how long a score bug lags a goal is a fact
about television, not about our buffer — both rows go flat and no phantom goal
survives at any depth.

That is the honest state of it: the delay does not show a measurable benefit
here, but the thing that was actively hiding one is gone. Whether a benefit
exists at all is a question about how a real vision model fails, and the
simulator's answer to that is an assumption, not a measurement.

Read all of it for what it is: a stand-in model on generated video. What these
establish is that the pipeline, the gate and the measurement work, and that the
ablations are wired correctly enough to run against the real thing.

## Commands

| Command | What it does |
|---|---|
| `run --source sim\|screen\|file` | Call a match. `--serve` adds the watch page, `--voice elevenlabs` adds sound and needs a key. |
| `sim` | Describe the synthetic match, or `--out x.mp4` to render it. |
| `capture [seconds]` | Prove frames reach Python. The day-one gate. |
| `grade runs/*.jsonl` | Metrics for saved runs. |
| `python -m commentary.grading.baselines` | The ablation suite and results table. |
| `python -m commentary.mcp_server` | The match-state tools over MCP. |

## Why there is no agent framework in here

LangGraph, CrewAI, AutoGen and the rest are built on the assumption that an
agent finishes its turn. The most important thing this system does is stop one
halfway through a word: the instant a goal goes in, whatever the analyst was
saying is wrong to keep saying, and a human producer would cut the mic. The
director is forty lines of `asyncio` because that is what the problem is.

## Grading

Nothing in [`src/commentary/grading/`](src/commentary/grading) is imported by
the runtime. The play-by-play feed and the human commentator's transcript live
on that side of the wall and only on that side, which makes the project's
central claim structural rather than promised.

Measured per run: factual error rate, event recall, fact-gate rejection rate by
reason, lag p50/p95, silence ratio, repetition, and cost. The ablations —
worldcupvoice's loop reproduced, no delay, no fact gate, single voice — run from
one command, and the headline chart is factual error rate against delay depth.

Factuality is judged twice, and the two are reported separately because they
catch different things. A deterministic checker finds the three failures that
are checkable outright: a name on no roster, a stated score that contradicts the
board, a goal with no goal behind it. A model judge (`grading/judge.py`) finds
what that cannot — a line that is fluent, on-roster, correctly-scored, and
describes something that did not happen. There is also a blind pairwise
comparison against the human commentator's own words for the same ten seconds,
run in both orderings, counting a win only where the two agree: a judge that
flips when you swap the order is reporting its own noise, not a preference.

The feed adapter (`grading/feed.py`) reads a saved play-by-play file in this
project's own shape — never a live fetch, and never somebody else's export
without a visible conversion first — and aligns its match clock to video time
using the board reader's own clock readings, reporting the residual so a bad
alignment is visible instead of quietly making every recall number wrong. Transcripts come from local Whisper
with the roster as the prompt, so it spells the players right.

## Development

```bash
uv run pytest        # 333 tests, no network, no key
uv run ruff check .  # lint
uv run mypy          # strict
uv run pre-commit install
```

The whole suite runs offline. No test makes a model call.

## Status

Days 1–12 of the two-week sprint in [`PLAN.md`](PLAN.md) are built and tested
against the simulator. What has **not** happened yet: a run against real
broadcast footage, and a run against the real Anthropic API — there was no key
on this machine when it was built, so every model call has been exercised
through the scripted and oracle backends. Both are a matter of dropping a key
into `.env` and pointing `--source` at a clip; nothing else should need to
change, but nothing else has been proven either.

## Rights

Broadcast footage is processed locally and never committed or hosted. Only short
clips are published for demos.
