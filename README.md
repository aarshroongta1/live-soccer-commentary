# Live Soccer Commentary

Two AI voices call a live soccer broadcast in real time, from a screen capture,
using only what a human commentator has: the picture and notes prepared
before kickoff.

**No live data feed reaches any agent at runtime.** The score comes from reading
the scoreboard on screen. Names come from the roster, shirt numbers, and
on-screen graphics. Events come from looking at the pitch. A play-by-play feed
is used only afterwards, as ground truth for grading — and as one off-by-default
row of the ablation table, to show what one would have bought.

## How it works

Frames land in a ring buffer. Perception runs at the live edge, but
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

## Names

Naming players is the hardest thing on that list, and the answer is smaller
than it was. The first attempt was a local chain — detect, cluster the kits
with SigLIP, read the shirt with PARSeq — and on three minutes of real
broadcast it confirmed **no shirt number at all**, while Claude read nine
correct number-and-name pairs off the very same frames. The second attempt
kept a detector and a tracker (RF-DETR, ByteTrack, an HSV kit split, a SigLIP
gallery for re-identification) purely to draw a letter tag over each body, so
that a number Claude read could ride that body through the frames where it
turned away.

That was measured on real footage and it lost. Six 45-second clips from three
matches, two runs each, tags on against tags off, Haiku 4.5 calling both arms:

| | tags on | tags off |
|---|---:|---:|
| spoken lines | 40 | 45 |
| lines naming a roster player | 26 | 43 |
| distinct players named | 8 | 20 |
| sightings bound to the roster | 70 of 235 | 115 of 120 |
| wrong or off-roster names | 0 | 0 |

The tags cost the caller names rather than buying them: it named Ronaldo,
Messi and Henderson — the close-ups — and little else, and reported letter
tags as sightings that bound to nobody. The median track lived about 2.5 s,
which is no longer than the caller's own frame window. So the vision extra is
gone, torch with it, and there is no detector, tracker or OCR anywhere in the
runtime.

What holds a name now is the registry and a carry rule: a number the caller
reads is checked against the team sheet and believed for that player, and a
name spoken on the ball stays on it for eight seconds in the same phase of
play, so the next line may keep it without re-reading a shirt that has turned
away. Names arrive when the camera is close enough to read one — the shooter,
the scorer, the fouled and the fouler, the taker — and it is role and kit
until then. The `named` and `name prec` columns of the results table are what
that claim is worth against StatsBomb; the per-clip evidence is in
[`docs/CLIPS.md`](docs/CLIPS.md).

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

## Playground

The watch page is a Next.js app in [`web/`](web). It shows the delayed picture
beside everything the agents decided: the trigger that fired, the form the
caller filled in, which shirt numbers it read and which of them the team sheet
turned into a player, every line the fact gate blocked and why, and the spend.

Three commands. Each Python one serves the runtime on `:8000`; the page proxies
to it, so the page is the only address to open.

```bash
cd web && npm install && npm run dev                       # the page, on :3000

uv run python -m commentary run --source sim --serve \
    --port 8000 --seconds 90                               # a live session, no key

uv run python -m commentary replay \
    --trace runs/screen/dimaria/screen-20260913-222906.jsonl \
    --path clips/argfra-dimaria.mp4 --start 20 \
    --serve --port 8000                                    # a past run, again
```

`replay` is the one to reach for. A run costs real money and happens once;
the trace it leaves is exactly what the bus published, so the clip and the
trace together reproduce the session frame for frame, calling nothing and
needing no key. `--start` is how far into the clip that run began — a
`--source file` run began at 0, and the screen runs played a clip in a video
player and started part-way in. The timestamps on the page are the trace's
own, so the goal called at 0:39 in the trace is called at 0:39 here.
Add `--loop` and the replay keeps playing the trace from the start once it
ends, so the page still has something running for whoever opens it late.

`?mock=1` on the page replays a built-in fixture with no runtime behind it at
all, which is how the UI is developed.

### Hearing a change without paying for it twice

`--voice` on `replay` is the same idea applied to the audio. A trace is a list
of lines a model was already paid for, so the cheap way to try a voice, a
sink, or a change to the director is to hand those lines to a real director
and listen.

```bash
uv run python -m commentary replay \
    --trace runs/voice/dimaria-goal/file-20260914-015533.jsonl \
    --path clips/dimaria-goal.mp4 \
    --voice elevenlabs --out runs/voice/replay-pcm
```

`--voice say` does the same through macOS's built-in speech for nothing at
all; `log`, the default, stays silent and republishes the trace as before.

With a voice on, the queueing, the ageing-out and the mid-word cut on a goal
all happen again, now — so the trace's own `spoken` and `preempted` rows are
dropped and the director publishes its own. That is the point: what the page
shows is the audio in the room, not a recital of the file. A trace of the
playback is written under `--out` (defaulting to `runs/`), carrying `seconds`
and `first_audio_s` for every line that came out.

Sound needs the audio extra, which is PortAudio:

```bash
uv sync --extra voice --extra audio
```

### What the voice costs in time

Two numbers per line, both in the trace. `seconds` is how long the director
held the channel; `first_audio_s` is how much of that the listener spent
waiting for any sound at all. They want reading together — five seconds for a
six-word line is the model writing too much if the sound started at once, and
the network or the output format if it did not.

Audio plays through PortAudio in this process
([`voice/playback.py`](src/commentary/voice/playback.py)), asking ElevenLabs
for raw `pcm_22050` rather than an mp3. The sink then knows exactly how much
speech it is holding, so finishing a line is arithmetic on bytes handed over
instead of waiting on a player to notice its input has ended — which ffplay,
given a pipe, does not do: four of the ten lines in the Di María trace sat out
the full five-second drain timeout and were killed. Measured on that trace,
replayed: those four fell from 5.10 / 5.22 / 4.92 / 5.50 s to 4.48 / 4.10 /
4.53 s and one cut, and the wait for first sound is 0.22–0.39 s once PortAudio
is warm. Opening its first stream costs a second on top, which the first line
of a match pays.

Raw PCM is four times the bytes of `mp3_22050_32`, so on a connection that
cannot deliver 44 KB/s the download, not the speech, becomes what `seconds`
measures. Pass `output_format="mp3_22050_32"` to `ElevenLabsSpeaker` and the
ffplay sink takes over, timeout and all.

One thing to know before match day: `--voice say` cuts macOS's `say` off
mid-utterance on every preemption, and doing that repeatedly can wedge the
Mac's audio stack until `sudo killall coreaudiod`. Nothing then plays — not
`say`, not PortAudio, not the browser. The sink gives up on a line after five
seconds of a device taking no audio and reports it cut, so the match keeps
calling rather than going silent for good, but the sound will not come back
on its own. Use `--voice say` to check the plumbing, not to rehearse.

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

| run | lines | factual err | recall | gate rej | lag p50/p95 s | silence | repeat | named | name prec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| worldcupvoice | 15 | 66.7% | 88% | 0.0% | 0.0 / 0.0 | 70% | 13.3% | 13% | 100% |
| **full** | 25 | **16.0%** | 100% | 37.8% | 8.0 / 8.0 | 54% | 12.0% | 16% | 75% |
| no-delay | 24 | 8.3% | 100% | 60.4% | 0.0 / 0.0 | 56% | 8.3% | 8% | 100% |
| no-gate | 30 | 70.0% | 100% | 0.0% | 8.0 / 8.0 | 40% | 10.0% | 7% | 100% |
| single-voice | 23 | 8.7% | 100% | 48.9% | 8.0 / 8.0 | 64% | 8.7% | 13% | 100% |
| wire-10s | 23 | 8.7% | 100% | 43.2% | 8.0 / 8.0 | 58% | 8.7% | 13% | 100% |

Variants called slightly different stretches — 142 to 150 s of match — so recall
and silence are fractions of each one's own window.

**The fact gate is the result.** Switching it off takes factual error rate from
16.0% to 70.0%: the caller proposes plenty of nonsense either way, and the gate
is the only thing between that and a voice. The reproduced worldcupvoice loop
sits at 66.7%, which is what an ungated caller with no state and no roster does
— 5 invented names, 4 goals that never happened, 1 wrong scoreline in fifteen
lines.

**Two columns here cannot show what they are named after, and it is worth
being plain about which.** The stand-in oracle writes its lines from the
simulator's script; it decodes a timestamp out of the frame and never reads
the picture. So the `named` and `name prec` columns are near-meaningless here: the simulator's ground truth names a player at a
goal, a save, a foul, a card and a substitution and nowhere else — about thirty
moments in ten minutes, with no passes at all — so a correct name said during
ordinary play has nothing to be marked right against. Those two columns need
real footage and StatsBomb, which names every touch.

### The wire

The last row is not a claim this project makes. It is the ceiling: the full
system plus a statistician's play-by-play feed, so the writeup can put a number
next to what vision achieves on its own.

It is off by default and nothing loads one unless asked (`run --wire
events.json --lineups lineups.json --wire-latency 10`). The state fields a feed
fills — who is on the ball, who fouled whom, the last few named events — are
the same fields vision fills, so the caller's prompt is one prompt whichever
source is on, and the rule text does not change when neither is.

The interesting part is the latency, because a feed is always late. Two clocks
govern it: an event is **known** when the live edge passes `video_ts +
latency_s`, and **applied** when the narration cursor passes `video_ts`. So a
correction reaches the commentary at

```
cursor = video_ts + max(0, latency_s - delay_s)
```

With the buffer deeper than the feed's lag, the buffer absorbs the lag entirely
and the statistician is telling the caller who has the ball in the frame it is
looking at. At `delay_s = 0` the same feed names whoever had the ball ten
seconds ago. A live feed's latency is therefore a floor under the delay: a
system that wants to use one and also wants to be right has to wait at least as
long as the feed does. That is an argument for the delay that does not depend on
the lookahead at all.

The fact gate is the result. Switching it off takes factual error rate from 6.5%
to 71.8% — the caller proposes plenty of nonsense either way, and the gate is
the only thing standing between that and a voice. The reproduced worldcupvoice
loop sits at 87.5%, which is what an ungated caller with no state and no roster
does: 11 invented names, 7 goals that never happened, 3 wrong scorelines.

### The delay

| delay | 0 s | 2 s | 4 s | 8 s |
|---|---:|---:|---:|---:|
| lines | 23 | 27 | 29 | 23 |
| factual error | 13.0% | 3.7% | 10.3% | 8.7% |
| errors reaching air | 3 | 1 | 3 | 2 |
| phantom goals reaching air | 1 | 1 | 1 | 1 |
| `unconfirmed_goal` rejections | 12 | 11 | 9 | 6 |

**No delay effect is visible in the error rate.** At ~25 lines per variant a
single error moves the number four points, and the whole column spans three
errors, so that row is noise around one value.

The other two rows are there because they used to say something alarming. The
gate originally accepted a board change anywhere in `[cursor - 2, cursor +
delay_s]`, so the confirmation window *widened with the buffer*: choosing to
wait longer also made the gate accept a scoreboard change further from the
moment being called. Phantom goals reaching air ran 1, 1, 4, 6 as the buffer
deepened, and `unconfirmed_goal` rejections fell 33, 32, 25, 14. The delay was
buying the caller information and paying for it by loosening the check. With
the window fixed to a constant — how long a score bug lags a goal is a fact
about television, not about our buffer — the phantom-goal row goes flat: one
at every depth, which is one line the gate lets through on the same goal
whatever the buffer is doing.

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
| `run --source sim\|screen\|file` | Call a match. `--serve` adds the watch page, `--voice say` adds free sound via macOS's built-in speech (testing only), `--voice elevenlabs` adds the real two-voice sound and needs a key. |
| `replay --trace t.jsonl --path clip.mp4` | Watch a finished run again: its clip through the delay buffer, its trace back onto the bus. `--start` is the clip offset the run began at, `--serve` adds the watch page. Calls nothing and costs nothing. |
| `replay --trace t.jsonl --path clip.mp4 --voice elevenlabs` | The same, but said out loud through a real director, so a voice or a sink can be heard on lines the model was already paid for. `--voice say` is free; `--out` says where to write the trace of what actually came out. No model is called either way. |
| `sim` | Describe the synthetic match, or `--out x.mp4` to render it. |
| `capture [seconds]` | Prove frames reach Python. The day-one gate. |
| `crop --path m.mp4 --at 300` | One frame with the score-bug box drawn on it and the bug beside it, so the crop is checked by eye before a run spends money. |
| `captions x.en.json3` | yt-dlp's auto-captions to a transcript: the human commentary, for free. |
| `feed events.json --home X --away Y` | StatsBomb's event data to the feed shape the grader reads. |
| `grade runs/*.jsonl` | Metrics for saved runs. |
| `grade run.jsonl --pack p.json --statsbomb e.json --lineups l.json` | The whole thing: StatsBomb aligned onto video time from the trace's own board readings, the results table, and the brief's twelve-item definition of done with the evidence for each. Refuses to grade an alignment it cannot trust. |
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
reason, lag p50/p95, silence ratio, repetition, name rate, name precision, and
cost. The ablations — worldcupvoice's loop reproduced, no delay, no fact gate,
single voice, and the play-by-play wire as a ceiling —
run from one command, and the headline chart is factual error rate against
delay depth.

Factuality is judged twice, and the two are reported separately because they
catch different things. A deterministic checker finds the three failures that
are checkable outright: a name on no roster, a stated score that contradicts the
board, a goal with no goal behind it. A model judge (`grading/judge.py`) finds
what that cannot — a line that is fluent, on-roster, correctly-scored, and
describes something that did not happen. There is also a blind pairwise
comparison against the human commentator's own words for the same ten seconds,
run in both orderings, counting a win only where the two agree: a judge that
flips when you swap the order is reporting its own noise, not a preference.

Ground truth for facts comes from StatsBomb's open event data
(`grading/statsbomb.py` converts a saved events file into this project's feed
shape) and ground truth for timing and naturalness from the broadcast's own
captions (`grading/captions.py` reads the `.en.json3` yt-dlp writes beside the
video). Captions are never consulted about facts, which is why auto-caption name
errors cost nothing here.

The feed adapter (`grading/feed.py`) reads a saved play-by-play file in this
project's own shape — never a live fetch, and never somebody else's export
without a visible conversion first — and aligns its match clock to video time
using the board reader's own clock readings, reporting the residual so a bad
alignment is visible instead of quietly making every recall number wrong. Transcripts come from local Whisper
with the roster as the prompt, so it spells the players right.

## Development

```bash
uv run pytest        # 448 tests, no network, no key, no model weights
uv run ruff check .  # lint
uv run mypy          # strict
uv run pre-commit install
```

The whole suite runs offline. No test makes a model call.

## Status

Days 1–12 of the two-week sprint in [`PLAN.md`](PLAN.md) are built and tested
against the simulator, and the pipeline has now been run end to end against the
real Anthropic API — Haiku 4.5 reading the score bug, Opus 5 calling, through a
video file, through the fact gate, and with a wire. Since then
it has been run on eighteen clips of real broadcast from six matches and four
broadcasters, graded against StatsBomb; that evidence is in
[`docs/CLIPS.md`](docs/CLIPS.md) and the current state of the project is in
[`docs/HANDOFF.md`](docs/HANDOFF.md). What has **not** happened yet is a live
source: everything so far is recorded clips.

## Rights

Broadcast footage is processed locally and never committed or hosted. Only short
clips are published for demos.
