# Live Soccer Commentary — Sprint Plan v5

A system that watches a live soccer broadcast on your screen and calls the
game with two voices, in real time, using only what a human commentator has:
the picture, the sound, and notes prepared before kickoff.

**The rule.** No live data from outside the broadcast reaches any agent at
runtime. Score comes from reading the scoreboard on screen. Names come from
the roster, jersey numbers, and on-screen graphics. Events come from looking
at the pitch. The ESPN play-by-play feed is used only after the fact, to
grade the system.

**The lineage.** This is worldcupvoice's loop (frames in, one sentence out)
with the four things it lacks: a delay buffer so the model knows outcomes, a
match state it can read off the screen, a fact gate, and a second voice with
a director. worldcupvoice is baseline one in the eval; every change is
measured against it.

Research: `docs/research/`.

---

## 1. What the system sees and hears

| Input | How | What it yields |
|---|---|---|
| Broadcast video | ffmpeg screen capture, 720p, 15 fps | Frames for the caller, the board reader, and the analyst |
| Broadcast audio | Same capture | Whistle (play stopped) and crowd roar (something happened), under 1 s |
| Pre-match notes | Researcher agent before kickoff: rosters with numbers, form, storylines, from web search | The knowledge pack every voice can cite |

Nothing else.

---

## 2. Architecture

```
  Screen capture (video + audio)
        │
        ├──► Audio triggers: whistle, roar ──────────────────────────┐
        │                                                            │
        ▼                                                            │
  ┌──────────────────────────────┐                                   │
  │ Delay buffer, 8 s            │  frames + audio, ring buffer      │
  │ cursor trails the live edge  │  agents read at cursor, peek ahead│
  └───────┬───────────────┬──────┘                                   │
          │               │                                          │
          ▼               ▼                                          ▼
  ┌───────────────┐ ┌────────────────────────────┐   ┌──────────────────────┐
  │ Board reader  │ │ Caller (vision)            │   │ Speak predictor      │
  │ Haiku 4.5,    │ │ Sonnet 5 or Haiku 4.5      │◄──│ triggers + salience  │
  │ score-bug crop│ │ 4 frames at cursor + 2 at  │   │ + silence pressure   │
  │ every 2 s     │ │ live edge, state, notes,   │   │ + rate cap           │
  │ -> score,     │ │ names drawn on the frames  │   └──────────────────────┘
  │ clock, replay │ │ -> structured: scene, event│
  │               │ │ names seen, line, confidence│
  └───────┬───────┘ └─────────────┬──────────────┘
          │                       │
  ┌───────┴───────┐               │
  │ Player tracker│               │  RF-DETR + ByteTrack + SigLIP kit
  │ local, open   │───────────────┤  clustering + PARSeq shirt numbers;
  │ every 2nd     │               │  (team, number) -> name, drawn on a
  │ frame, 640 px │               │  COPY of the caller's frames only
  └───────┬───────┘               │
          ▼                       ▼
  ┌──────────────────────────────────────────────┐
  │ Match state + entity registry                │  score, clock, possession
  │ (from board reader and caller output only)   │  streaks, storylines
  └──────────────────────┬───────────────────────┘
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Fact gate (deterministic)                    │  names on roster; score
  │                                              │  matches board; goal needs
  │                                              │  board change or lookahead
  └──────────────────────┬───────────────────────┘  celebration
                         ▼
  ┌──────────────────────────────────────────────┐
  │ Director (asyncio)                           │  picks caller / analyst,
  │                                              │  min gaps, cancels on goal
  └───────┬──────────────────────────┬───────────┘
          ▼                          ▼
     Caller line             Analyst (Opus 5, low effort, vision + MCP
                             tools over state and notes; lulls only)
          └──────────────┬───────────┘
                         ▼
          Bus: SSE, ElevenLabs two voices, cancel-aware queue
                         ▼
          Next.js: delayed video, feed, agent panel, state
```

### Why each piece exists

- **Delay buffer with lookahead.** GetStream measured frame narration wrong
  more than half the time because the model reasons over a few frames with
  no idea what happens next. Give it the next 6 seconds and the shot it
  describes has already gone in or not.
- **Board reader.** A human glances at the scoreboard. So does this. A crop
  of the score bug to a cheap vision call every 2 seconds, confirmed over
  three reads. No OCR library. Score bug absent means replay, so the caller
  is told not to narrate replays as live.
- **Structured caller output.** The caller does not just emit a sentence.
  It fills a form: scene type, event, team, jersey numbers or names it can
  actually read, confidence, and the line. The form feeds state and the
  fact gate; the line feeds the voice. worldcupvoice's 40-token cap and
  repetition gate are kept.
- **Fact gate.** Names must be in the roster or read from a graphic. Score
  mentions must match the board. A goal needs a board change or a
  celebration visible in the lookahead frames. Rejections are logged with
  reasons and reported in the eval.
- **Speak predictor.** worldcupvoice speaks on a 4 s timer. Here the
  triggers are whistle, roar, camera cut, and salience of the last call,
  with silence pressure and a hard cap of one line per 4 to 5 s.
- **Two voices.** The analyst fires in lulls and after big moments, sees a
  wider frame window at lower rate, and has the notes and match state over
  MCP tools. The director hard-preempts it on a goal.

---

## 3. Agents

| Agent | Model | Sees | Rate |
|---|---|---|---|
| Caller | `claude-sonnet-5` (try `claude-haiku-4-5` for speed) | 4 frames at cursor, 2 at live edge, state, last 5 lines, notes | On trigger, max 1 per 4 s |
| Board reader | `claude-haiku-4-5` | Score-bug crop | Every 2 s |
| Analyst | `claude-opus-5`, effort low | 6 frames over 20 s, state, MCP tools | Lulls, after big moments |
| Researcher | `claude-opus-5`, web search | Team names typed in at kickoff | Once pre-match, again on subs seen on screen |
| Fact gate | none | Caller form, state, roster | Every line |
| Director | asyncio, Haiku for angle choice | Beat queue, agent states | Every tick |

Cost per match, rough: caller ~1,300 calls × 6 images ≈ $20 on Sonnet, ~$7 on
Haiku; board reader ~$1; analyst ~$2. Measure and pick.

---

## 4. Stack

| Layer | Choice |
|---|---|
| Capture | ffmpeg avfoundation, OpenCV, numpy ring buffer |
| Audio | librosa or YAMNet for whistle and roar; `mlx-whisper` for eval transcripts |
| Core | Python 3.12, `uv`, Pydantic v2, asyncio, pytest |
| LLM | `anthropic` SDK: vision, structured outputs, streaming, prompt caching, web search tool, Batch for eval |
| Tools | MCP server (FastMCP) over match state and notes |
| Observability | Langfuse |
| API / UI | FastAPI, SSE, MJPEG delayed video; Next.js, Tailwind |
| Voice | ElevenLabs streaming, two voices; LiveKit Agents as stretch |
| Hygiene | ruff, mypy, pre-commit, GitHub Actions, Docker Compose |

No detector, tracker, OCR library, or agent framework. README explains the
last one: LangGraph, CrewAI, AutoGen assume an agent finishes its turn;
commentary needs it cut off mid-sentence.

---

## 5. Schedule

| Day | Build | Gate |
|---|---|---|
| 1 | Repo, CI, schemas, capture with audio, ring buffer | Any on-screen video streams into Python |
| 2 | Board reader, replay detection, match state | Score and clock track a recording within 3 s of every goal |
| 3 | Caller with structured output, lookahead, repetition gate, SSE, minimal page | A recording gets called in the browser |
| 4 | Audio triggers, speak predictor, fact gate | Silence feels right; no unconfirmed goal is ever spoken |
| 5 | Researcher, knowledge pack, entity registry, MCP server | Names appear only when on roster or read from a graphic |
| 6 | Analyst, director, cancellation, agent panel | A goal preempts the analyst mid-chunk, trace shows it |
| 7 | Live run on a weekend match; tune prompts, triggers, rate | 15 unedited minutes are watchable |
| 8 | Eval: feed adapter for grading, Whisper transcripts, alignment by clock, metrics | Factual error rate, lag, silence ratio are numbers |
| 9 | Baselines: worldcupvoice reproduced; no-delay; no-fact-gate; single voice | Results table |
| 10 | ElevenLabs, two voices, cancel-aware queue | Demo has sound |
| 11 | Hardening: stream stalls, refusal handling, cost cap, 90 min run | Survives a full match |
| 12 | README, diagram, demo video | Three-command quickstart |
| 13–14 | Buffer, optional items | |
| 13/14 Oct | UCL matchday 2 live recording | Showcase |

Optional if time remains, in order: Echoes style retrieval for the caller;
a detector overlay (boxes burned into frames, as GetStream did) to test
whether it helps the caller; a trained speak predictor.

---

## 6. Evaluation

Ground truth for facts is the ESPN feed for the same match, aligned to video
time via the board reader's clock. Ground truth for timing and naturalness
is the real commentators on the recording: transcribe the captured broadcast
audio once with Whisper locally (`mlx-whisper`, large-v3-turbo, roster passed
as the hint prompt). Both are used only here, never at runtime.

- **Factual error rate** of spoken lines: wrong team, wrong scorer, wrong
  score, event that did not happen. Judged by `claude-opus-5` against the
  feed, spot-checked by hand.
- **Event recall**: fraction of feed goals, shots, corners, cards that got a
  line within 10 s of video time.
- **Fact gate rejection rate** and reasons.
- **Lag**: buffer depth plus generation time, p50 and p95.
- **Silence ratio**, **repetition**, **cost per match**.
- **Name rate** and **name precision**: what fraction of spoken lines name a
  player, and how many of those names are the player the feed says was
  involved within three seconds. This is the number the vision naming chain
  is answerable for, and the `no-marks` ablation is what it is compared
  against.
- **Pairwise judge** win rate, blind, both orderings, Batch API: your line
  versus the human commentator's line for the same 10 s window, and your
  line versus each baseline's.

Baselines, all on the same clips: (1) worldcupvoice's loop reproduced with
the same model, fixed 4 s cadence, no delay, no state; (2) full system at
delay 0; (3) full system without the fact gate; (4) single voice; (5) the
full system with no names drawn on the caller's frames. The headline chart is
factual error rate versus delay depth at 0, 2, 4, 8 s.

There is a sixth row, and it is a ceiling rather than a baseline: the full
system plus a statistician's play-by-play feed at 10 s of modelled latency.
The feed stays behind the grading wall in every other respect — the default
runtime never loads one, and the project's claim is unchanged by it — but a
writeup that says vision can name players should say what a feed would have
bought instead, and at what. The row also makes the delay argument twice
over: a feed is late by construction, so an event is known when the live
edge passes `video_ts + latency_s` and applied when the cursor passes
`video_ts`, and a correction therefore lands at `video_ts + max(0,
latency_s - delay_s)`. A system that wants a live feed and wants to be right
has to wait at least as long as the feed does.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| Vision caller still wrong on fast play | Lookahead, structured confidence, fact gate; measured, and the delay chart shows the tradeoff |
| Board reader misreads | Three-read confirmation; per-broadcaster crop presets |
| Names rarely legible | Say the team when unsure; names from graphics on subs and goals; registry with decay |
| Cost | Haiku for caller if quality holds; caching; hard per-match cap |
| Stream stalls | Buffer drains, director goes quiet, reconnect loop |
| Rights | Process locally, post short clips, never host broadcast video |

---

## 8. Resume lines

- Built a real-time two-voice AI commentator that watches live broadcast
  soccer from a screen capture with no external data: vision-model caller
  over a lookahead delay buffer, on-screen scoreboard reading, audio event
  triggers, a deterministic fact gate, and an asyncio director with
  mid-sentence cancellation across six Claude agents.
- Reproduced the most popular open-source approach, measured its failure
  modes against Opta ground truth, and cut factual errors from X% to Y%;
  published the error-versus-delay curve.
- Stack: Python asyncio, Anthropic SDK, MCP, Langfuse, ElevenLabs, FastAPI,
  Next.js, ffmpeg, Docker.
