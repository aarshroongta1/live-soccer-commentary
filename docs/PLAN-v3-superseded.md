# Live Soccer Commentary from a Screen-Captured Stream — 2-Week Sprint Plan (v3)

Version 3 replaces the 14-week plan, now archived at `docs/PLAN-v2-full.md`.
The research behind both is in `docs/research/` (four surveys, ~90 repos and
papers). Read v2 §2 "What the research says" once; it is still the argument
for every design choice here. This document is the executable version.

**What changed since v2:** the input is a screen capture of a live broadcast
stream you are watching, the build is a two-week sprint with Claude Code doing
most of the typing, the stack leans on tools recruiters recognize wherever
that does not cost correctness, and the Champions League showcase is a
scheduled recording session after the sprint, not the end of it.

---

## Read this first: the final plan in stages

No NDA, no training, pretrained weights only. One bullet per stage, each
with tools and a done-when goal. Later sections are the detailed reference.

1. **Capture.** ffmpeg screen grab at 720p / 15 fps into an OpenCV loop.
   Done when any video playing on screen streams into Python in real time.
2. **Detect and track.** `roboflow/sports` pretrained weights, `supervision`,
   `roboflow/trackers` ByteTrack, team colors frozen at kickoff, on MPS.
   Done when players and ball have stable boxes and teams at 10+ fps.
3. **Read the broadcast.** Score-bug OCR (PaddleOCR) at 1 Hz for score,
   clock, teams; bug missing = replay. Whistle detector on the broadcast
   audio (small spectral classifier) = play stopped. Cut detector.
   Done when every goal in a recorded match is caught within 3 s and every
   stoppage is flagged.
4. **Rules.** Pure Python: possession with hysteresis, pass, turnover, out of
   play. Pydantic events with confidence and evidence frames.
   Done when event precision and recall are in a table.
5. **Delay buffer.** 8 s ring buffer of frames, audio, events, state.
   Narration cursor trails the live edge with lookahead.
   Done when the frontend plays video from the cursor.
6. **Scout.** Sonnet 5 vision, structured output, 3 frames from behind the
   cursor, fired on whistle, camera cut, and score-bug appear/disappear,
   with a 10 s fallback. Reports cards, VAR, corners, subs, celebrations,
   injuries, and any name on a graphic, each with a confidence.
   Done when a card and a corner in the recording are logged as events.
7. **Fact source.** API-Football (or similar) live events polled every 15 s
   for official events with player names. Merged into the event log as
   confirmations, never as triggers. Researcher agent (Opus 5 + web search)
   builds the knowledge pack at kickoff from the score-bug team names.
   Done when a card gets a player name within 90 s without the scout guessing.
8. **Agents.** Caller (Haiku 4.5, one streamed sentence). Analyst (Opus 5
   low effort, tools over an MCP server exposing state, events, knowledge).
   Director (asyncio priority queue, cancels in-flight calls on a goal).
   Fact gate (deterministic; confidence decides how specific a line may be).
   Speak predictor (salience + silence pressure, max one line per 4 to 5 s).
   Langfuse on every call.
   Done when a goal preempts the analyst mid-chunk and the trace shows it.
9. **Output.** FastAPI + SSE, MJPEG delayed video with overlay, Next.js page
   with feed, agent activity, match state. ElevenLabs two voices with a
   cancel-aware audio queue.
   Done when a stranger watches a 15-minute segment with sound.
10. **Eval.** Ground truth from Whisper on the real commentators' audio of
    the recorded match (eval only, never a live input). Metrics: factual
    error rate, lag, silence ratio, repetition, cost, pairwise judge via
    Batch API. Baselines: frame narration every 4 s (worldcupvoice),
    delay 0, scout off.
    Done when the results table and the accuracy-vs-delay chart exist.
11. **Ship.** README with diagram, results, three-command quickstart, and a
    "why no agent framework" section. 90 s demo video. Public repo.
    Then the UCL matchday 2 live recording on 13/14 Oct.

Optional upgrades, in order: Echoes style retrieval, SoccerNet spotter on
the buffer window, calibration and radar, trained speak predictor.

---

## 0. Footage and timing

Testing "live" needs nothing special: play any highlight or full-match video
on screen and the capture loop feeds it to the pipeline in real time, exactly
like a stream. Use a recorded UCL match for development and eval. The final
showcase is a live UCL match at matchday 2, 13/14 Oct 2026 (matchday 1 ended
10 Sep). Rights: process whatever you like locally, post a short demo clip,
do not host broadcast video on a public page.

---

## 1. Goals

The project must prove five things, each with an artifact:

1. **Video in, commentary out, live, on a broadcast you did not prepare for.**
   Artifact: the UCL matchday 2 recording.
2. **Real-time systems engineering.** Perception at the live edge, narration
   behind a delay buffer, async cancellation of in-flight LLM calls, measured
   per-stage latency. Artifact: the agent activity panel in the demo and a
   latency table in the README.
3. **Multi-agent design that earns its complexity.** Six agents with different
   latency budgets, models, and tools. Artifact: a director trace where a goal
   preempts the analyst mid-sentence.
4. **Applied AI rigor.** Typed events instead of pixels on the play-by-play
   path, a deterministic fact gate, and an evaluation with baselines.
   Artifact: a results table and one chart, accuracy versus delay depth.
5. **A codebase someone can read in twenty minutes.** Artifact: README with
   architecture diagram, results, and a three-command quickstart.

### Non-goals for the sprint

- No training from scratch. Pretrained detectors and a pretrained SoccerNet
  action spotter. Fine-tuning the spotter head is a week 2 stretch if video
  access clears.
- No named-player identification from jersey numbers. Names come from the
  broadcast score bug, the knowledge pack, and the scout's reading of on-screen
  graphics (lower-third name cards on subs and goals), never from guessing.
- No SoccerNet. The NDA will not clear in time and the pipeline does not need
  it. Ground truth is three clips you label yourself.
- No hosted live GPU demo. Recorded demo plus a public repo is what an
  internship reviewer actually looks at.

---

## 2. Architecture

```
  Screen capture (ffmpeg avfoundation, 720p @ 15 fps)
         │
         ▼
  ┌──────────────────────┐
  │  Perception (MPS)    │  Pretrained RF-DETR or roboflow/sports weights: players, ball, refs
  │  live edge           │  ByteTrack via roboflow/trackers; team clusters frozen at kickoff
  │                      │  Pitch keypoints at 2-5 Hz -> homography, rejected when unstable
  │                      │  Shot classifier: cut detection + green ratio + score-bug present?
  │                      │  Score bug OCR: score, clock, team abbreviations (1 Hz)
  └──────────┬───────────┘
             │  Tracks + calibration + shot type + HUD per frame
             ▼
  ┌──────────────────────┐
  │ Symbolic Event Layer │  Pure Python rules with hysteresis, gated to main-camera shots:
  │                      │  possession, pass, turnover, shot, out of play, restart,
  │                      │  GOAL = score-bug change confirmed over N frames
  └──────────┬───────────┘  Every event: confidence, evidence frame ids
             ▼
  ┌──────────────────────┐
  │ Match State +        │  Score, clock, possession %, territory, momentum, storylines
  │ Entity Registry      │  Team names from HUD; player names only from knowledge pack + scout
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │   Delay Buffer       │  Ring buffer of frames + state, 8 s deep. Narration cursor
  │                      │  trails the live edge; agents may peek ahead to the edge.
  └──────────┬───────────┘  Frontend plays video FROM THE CURSOR.
             ▼
  ┌──────────────────────┐        ┌──────────────────────┐
  │  Speak Predictor     │        │  Scout (Sonnet 5,    │  Every ~8 s, off the hot path:
  │  salience + silence  │◄───────│  vision, structured) │  3 delayed frames -> scene type,
  │  pressure -> P(speak)│        └──────────────────────┘  celebration, card, sub, name card
  └──────────┬───────────┘
             ▼
  ┌──────────────────────┐
  │     Director         │  asyncio. Priority queue of beats. Picks speaker + angle.
  │                      │  One Task per utterance; Task.cancel() on preemption.
  └──┬────────┬──────────┘  Each agent sees its peer's last line.
     │        │
  ┌──▼────┐ ┌─▼──────┐  ┌────────────┐
  │Caller │ │Analyst │  │ Researcher │  Caller: Haiku 4.5, 1 sentence, streamed.
  │       │ │        │  │            │  Analyst: Opus 5 low effort, lulls only, MCP tools.
  └──┬────┘ └─┬──────┘  └─────┬──────┘  Researcher: Opus 5 + web search, pre-kickoff + subs.
     └────────┴───────────────┘
                     ▼
           ┌───────────────────┐
           │    Fact Gate      │  Deterministic. Names in registry, score/clock match state
           └─────────┬─────────┘  at the cursor, event claims match the event log.
                     ▼
           ┌───────────────────┐
           │  Broadcast Bus    │  SQLite/Postgres log, SSE to client, TTS queue (day 11)
           └─────────┬─────────┘
                     ▼
           ┌───────────────────┐
           │  Next.js client   │  Delayed MJPEG video with overlay, feed, agent panel, state
           └───────────────────┘
```

### Design rules (unchanged from v2, each backed by a measured result)

1. Narrate behind the live edge. 8 s to start. Peek ahead for outcomes.
2. The LLM never sees pixels on the play-by-play path. The scout is the one
   exception, and it feeds evidence into the event stream with a confidence,
   it does not narrate.
3. Perception, events, speak decision, and LLM calls run at different rates.
4. Speaking is a decision stage. Cap at one line per 4 to 5 seconds.
5. Identity is resolved before the prompt, never in it.
6. Utterances are short because cancellation is lossy.
7. Every line traces to evidence frame ids.

### What is new in v3 and why

**Broadcast input, not a fixed camera.** The stream cuts between the main
camera, close-ups, replays, and crowd shots. Two cheap signals handle this.
Broadcasters hide the score bug during replays, so "score bug absent" is a
near-free replay detector. Green-pixel ratio plus a cut detector separates
the main camera from close-ups. The event layer only runs on main-camera
shots; the registry decays track identity across every cut.

**The score bug is your ground truth for the two facts that matter most.**
OCR the score and clock at 1 Hz. A goal is a score change that holds for N
consecutive reads. This is the same trick the EA FC repos use on the HUD, and
it makes goal announcement essentially unfakeable. Nobody in the survey does
this on live broadcast.

**The scout agent.** Perception built in two weeks will be thin, so a vision
call every ~8 seconds on three frames from behind the cursor classifies the
scene (open play, set piece, goal celebration, VAR check, injury, card,
substitution, replay) and reads on-screen name cards. Its output is typed,
carries a confidence, and goes through the fact gate like any event. The
eval runs with the scout on and off so the README can say what it adds.

---

## 3. Agents

| Agent | Model | Fires when | Sees | Produces | Budget |
|---|---|---|---|---|---|
| Caller | `claude-haiku-4-5` | Speak predictor fires for play-by-play | Last 5 events, state, analyst's last line | One sentence, present tense, streamed | < 1.5 s to first token |
| Analyst | `claude-opus-5`, effort low | Lulls, after big moments | State, storylines, MCP tools (stats, knowledge pack) | 2 to 3 short chunks | < 4 s, cancellable |
| Researcher | `claude-opus-5`, web search tool | Pre-kickoff, on substitution, on a streak | Team names from HUD, knowledge pack | Nuggets to a shared scratchpad | Background, no deadline |
| Scout | `claude-sonnet-5`, vision, structured output | Every ~8 s | 3 frames from behind the cursor | Scene type, names on graphics, confidence | Off hot path |
| Fact Gate | none, deterministic | Every candidate line | Registry, state at cursor, event log | Accept, rewrite, or reject with reason | < 5 ms |
| Director | asyncio + `claude-haiku-4-5` for angle choice only | Every tick | Beat queue, agent states | Speaker, angle, cancel signals | < 50 ms per tick |

Prompt caching on every agent: frozen system prompt and tool list first,
volatile state last. Judge and eval runs go through the Batch API at half
price. Rough cost per 90-minute match, dominated by the scout's images:

| Item | Calls | Approx cost |
|---|---|---|
| Caller | ~1,000 | $2 |
| Scout | ~650 | $6 |
| Analyst + Researcher | ~120 | $2 |
| Total | | ~$10 |

---

## 4. Stack, and why each piece is there

| Layer | Choice | Why this and not the alternative |
|---|---|---|
| Capture | `ffmpeg` avfoundation screen device, or OBS virtual camera into OpenCV | Zero setup, works on any stream in any browser |
| Detection | `rf-detr` pretrained, or `roboflow/sports` weights, via `supervision` | Apache 2.0, soccer example already exists, MPS runs 10 to 15 fps at 640 px |
| Action spotting | Pretrained SoccerNet spotter via `opensportslib`, run on the delay buffer window | 17 event classes rules cannot derive; the buffer supplies the future frames these models need |
| Tracking | `roboflow/trackers` ByteTrack | Clean, benchmarked on SoccerNet |
| Calibration | `roboflow/sports` pitch keypoint model, decimated to 2 to 5 Hz | Pretrained, good enough for zones and shot direction |
| OCR | PaddleOCR or EasyOCR on a cropped score bug | Score and clock are the two facts the fact gate needs most |
| Events, state, buffer | Pure Python, Pydantic v2, pytest + hypothesis | The part that must be right and is cheap to test |
| LLM | `anthropic` Python SDK: streaming, structured outputs, vision, prompt caching, Batch, server-side web search | One SDK covers every agent |
| Tools | **MCP** server (FastMCP) exposing match state, event log, knowledge pack | Analyst and researcher use it over MCP; recruiters know the acronym; it also makes the tools reusable from Claude Desktop for debugging |
| Orchestration | Hand-rolled asyncio director | Every framework surveyed is turn-completion based; commentary is turn-abandonment based. LangGraph's `interrupt()` is a durable pause measured in hours. State this in the README; it is a stronger signal than using a framework |
| Observability | **Langfuse** | Every LLM call traced with model, latency, tokens, cost, cache hit; buffer depth and cursor lag as custom metrics |
| API | FastAPI + SSE, MJPEG endpoint for the delayed video | Simple, no WebRTC needed |
| Storage | SQLite for the sprint, Postgres + pgvector only if the knowledge pack outgrows it | Do not spend a day on infra |
| Frontend | Next.js, Tailwind, canvas overlay | |
| Audio | ElevenLabs streaming, two voices, browser-side queue that honors director cancels | Day 11. **LiveKit Agents** is the stretch upgrade: its `SpeechHandle.interrupt()` is the cleanest barge-in surface surveyed |
| Hygiene | `uv`, `ruff`, `mypy`, `pre-commit`, GitHub Actions, Docker Compose | Day 1, not day 13 |

Deliberately absent: LangGraph, CrewAI, AutoGen (wrong timescale, explained in
README), Modal (perception runs locally at the fps the delay buffer needs),
SoccerNet (NDA), any training.

---

## 5. Evaluation

Three 5-minute clips from tonight's UCL recording, chosen to include at least
one goal, one replay sequence, and one dull stretch. Label them with a
30-line keyboard tool: press a key at each pass, shot, turnover, out, goal.
That is your ground truth.

| Metric | How |
|---|---|
| Event precision / recall | Per class, 1 s tolerance, against your labels. Rule events and spotter events reported separately. Main camera only |
| Factual error rate | Each emitted line judged against the label log by `claude-opus-5` with the event log as reference, spot-checked by you on 50 lines |
| Fact gate rejection rate and reasons | Logged |
| Lag | Buffer depth plus cursor-to-first-token, p50 and p95 |
| Silence ratio | Fraction of time speaking. Target: near real broadcast (roughly 60 to 70 percent) |
| Repetition | Jaccard over the last 5 lines, the worldcupvoice gate |
| Pairwise judge win rate | Blind, both orderings, Batch API, against each baseline |
| Cost per match | From Langfuse |

**Baselines, all on the same three clips**

1. **Raw frames to a vision model every 4 s, 40-token cap.** The worldcupvoice
   architecture, reproduced with Sonnet 5. Expect it to be wrong often and to
   announce goals that did not happen.
2. **Full system at delay 0.** Same pipeline, no buffer. This gives the
   accuracy-versus-delay curve at 0, 2, 4, 8 s: the headline chart.
3. **Full system, scout off.** Shows what the vision agent adds.
4. **Full system, spotter off.** Shows what the learned event model adds
   over rules alone.

---

## 6. Sprint schedule

Assumes roughly full days with Claude Code writing most of the code. Every
day ends with a runnable state and a commit. Each day has a done-when gate;
if the gate slips by more than half a day, cut scope from that day's stretch
items, never from the next day.

### Day 0 (done)
- Get a recorded UCL match on disk for dev and eval.

### Day 1, Fri 11 Sep: skeleton and schemas
- Repo layout (§7), `uv`, ruff, mypy, pytest, pre-commit, CI, Docker Compose.
- Pydantic models: `Frame`, `Track`, `Calibration`, `HUD`, `MatchEvent`,
  `MatchState`, `Beat`, `Line`.
- Delay buffer with cursor and lookahead, match state engine, both tested.
- Label feed: replay SoccerNet's free action-spotting labels for one real
  match at wall-clock pace, with seek and speed, and the SoccerNet-Echoes
  commentator transcript for the same match alongside. No NDA needed for
  labels or transcripts. This is the development feed until perception
  exists, and the human lines are the comparison in the README.
- **Done when:** replaying the labeled match prints correct score and
  events and the buffer exposes a trailing cursor.

### Day 2, Sat 12 Sep: narration on real events
- Speak predictor v1 (salience, silence pressure, rate cap, repetition gate).
- Style retrieval: index Echoes lines by event type; the caller's prompt gets
  three real commentator lines for the current event type.
- Caller agent, streamed, with prompt caching. Fact gate v1. Broadcast bus,
  SSE endpoint, Langfuse tracing.
- Minimal Next.js page: commentary feed and agent state.
- Evening: watch a live match and take notes on when real commentators speak.
- **Done when:** the labeled match gets called in the browser, the caller
  never announces a goal before the buffer confirms it.

### Day 3, Sun 13 Sep: capture and detection
- `ffmpeg` screen capture into an OpenCV loop at 15 fps.
- RF-DETR or sports weights on MPS. ByteTrack. Team clustering frozen at
  kickoff. Per-stage timing.
- Run on tonight's recording and on a live domestic match in the afternoon.
- Spotter check: run a pretrained SoccerNet action-spotting checkpoint
  (opensportslib or a challenge repo) on 10 minutes of the UCL recording.
  If it finds most fouls, corners, and saves, it becomes the day 7 track.
- Apply for SoccerNet video access. Free to apply; only needed if you
  fine-tune the spotter head in week 2.
- **Done when:** boxes and team colors are stable on the main camera at 10+
  fps and the timing table is printed.

### Day 4, Mon 14 Sep: shot type, score bug, calibration
- Cut detector, green-ratio shot classifier, score-bug localization and OCR
  at 1 Hz with N-read confirmation.
- Pitch keypoints decimated to 2 to 5 Hz, homography with rejection and
  stale fallback. Radar output.
- **Done when:** on the UCL recording, every goal is detected from the score
  bug within 3 s, replays are flagged, and the radar looks right on main
  camera shots.

### Day 5, Tue 15 Sep: symbolic events
- Possession with hysteresis, pass, turnover, shot, out of play, restart,
  goal from HUD. Camera-motion compensation via median player displacement.
  Confidence and evidence ids on every event.
- Labeling tool. Label the three eval clips.
- **Done when:** event precision and recall per class is in a table, even if
  the numbers are ugly.

### Day 6, Wed 16 Sep: video in, commentary out
- Swap the synthetic feed for the perception feed. Delayed MJPEG endpoint
  with overlay. Frontend plays from the cursor.
- **Done when:** the UCL recording is commentated end to end in the browser
  with the video landing on the commentary.

### Day 7, Thu 17 Sep: learned events, scout, registry
- Action spotter on the delay buffer window: the pretrained 17-class
  SoccerNet model runs on the 8 s behind the cursor, where the future frames
  it needs already exist. Adds fouls, cards, offsides, corners, throw-ins,
  saves, headers, clearances that rules cannot derive. Each spotted event
  enters the event log with the model's confidence. Report per-class
  precision and recall on your labeled clips.
- Scout agent with structured output, on three delayed frames every 8 s,
  narrowed to what the spotter misses: celebrations, VAR checks, injuries,
  name cards. Entity registry fed by HUD team names and scout name cards.
- If SoccerNet video access has cleared: fine-tune the spotter head on a few
  games (stretch, one day).
- Knowledge pack: researcher builds it at kickoff from HUD team names via the
  web search tool, stores rosters, form, storylines.
- **Done when:** a goal celebration and a substitution are called with the
  right team, and a name from a lower-third graphic appears in a line only
  after the fact gate resolved it.

### Day 8, Fri 18 Sep: second voice and director
- MCP server exposing state, events, knowledge pack. Analyst with tools.
- Director: priority queue, preemption, minimum gaps, peer context,
  generate-ahead with stale discard. Agent activity panel with buffer depth.
- **Done when:** a goal preempts the analyst mid-chunk, the UI shows the
  cancellation, and the Langfuse trace shows the cancelled call.

### Day 9, Sat 19 Sep: live run two
- Run the full system on a live domestic match. Fix what breaks. Tune
  salience weights and silence pressure against your notes from day 2.
- **Done when:** a 15-minute unedited live segment is watchable.

### Day 10, Sun 20 Sep: evaluation
- Harness, three baselines, judge via Batch, delay curve at 0/2/4/8 s.
- **Done when:** the results table and the delay chart exist.

### Day 11, Mon 21 Sep: audio
- ElevenLabs streaming, two voices, browser audio queue honoring cancels.
- **Done when:** the demo has sound and a goal cuts the analyst off audibly.

### Day 12, Tue 22 Sep: hardening
- Reconnect logic for capture drops, stream stalls, OCR misreads, LLM
  timeouts. Refusal-stop handling and server-side fallbacks on Opus calls.
  Cost cap per match.
- **Done when:** the system survives a 90-minute run without a restart.

### Day 13, Wed 23 Sep: write-up
- README: architecture diagram, results, quickstart, "why no framework"
  section, honest limitations. Demo video from the day 9 run, 90 seconds.
- **Done when:** a stranger can run it on a clip in three commands.

### Day 14, Thu 24 Sep: buffer day
- Whatever slipped. If nothing slipped: LiveKit Agents for audio, or a
  learned speak predictor from your own labeled clips.

### Showcase, Tue 13 / Wed 14 Oct: UCL matchday 2
- Pick a match with a strong storyline. Run live. Screen-record the browser
  with audio. Post a 2-minute cut and the full-half recording.

---

## 7. Repo layout

```
commentary/
  capture/        screen.py (ffmpeg), file.py, base.py
  perception/     detect.py, track.py, teams.py, shots.py, hud.py, calibrate.py, motion.py, loop.py
  events/         rules.py, confidence.py, models.py
  state/          engine.py, storylines.py, registry.py
  buffer/         delay.py
  speak/          predictor.py, router.py, repetition.py
  agents/         caller.py, analyst.py, researcher.py, scout.py, director.py, factgate.py, prompts/
  llm/            client.py (SDK wrapper, caching, tracing, fallbacks)
  mcp/            server.py (FastMCP: state, events, knowledge)
  knowledge/      pack.py, store.py
  api/            app.py, sse.py, mjpeg.py
  eval/           label.py, harness.py, baselines/, judge.py, report.py
  frontend/       Next.js
  tests/
  docs/           research/, PLAN-v2-full.md, ARCHITECTURE.md, RESULTS.md
  PLAN.md  README.md  docker-compose.yml  pyproject.toml
```

---

## 8. How it reads on a resume

- Built a system that commentates live broadcast soccer from a screen
  capture: RF-DETR perception at 12 fps on Apple Silicon, score-bug OCR for
  ground-truth goals, a rule-based event layer, an 8-second delay buffer for
  outcome-aware narration, and six cooperating Claude agents producing
  two-voice commentary with audio.
- Designed an asyncio director that schedules agents by event salience,
  cancels in-flight streaming calls on goals, and coordinates voices through
  shared context; exposed match state to agents over MCP; traced every call
  in Langfuse.
- Cut factual errors from N% (frame-narration baseline) to M% via the delay
  buffer, a deterministic fact gate, and roster grounding; published the
  accuracy-versus-delay curve and pairwise LLM-judge win rates.
- Stack: PyTorch, RF-DETR, ByteTrack, OpenCV, PaddleOCR, Python asyncio,
  FastAPI, Anthropic SDK, MCP, Langfuse, ElevenLabs, Next.js, Docker,
  GitHub Actions.

---

## 9. Risks specific to the sprint

| Risk | Mitigation |
|---|---|
| Pretrained detector is poor on your stream's resolution or broadcaster graphics | Test on tonight's recording on day 3; fall back to `roboflow/sports` weights; 960 px input if MPS allows |
| Score-bug layout differs per broadcaster | Localize by finding the stable high-contrast region in the top corners over 5 s; keep per-broadcaster crop presets |
| Calibration flaky on broadcast pans | Events degrade gracefully: possession and turnover need no homography; zones and shots do |
| Ball lost most of the time | Possession from player trajectories; goals from HUD, not ball |
| Mac too slow with detection plus scout plus TTS | Detection at 640 px, calibration decimated, scout is async; profile on day 3 |
| Stream stalls or capture drops | Reconnect loop, buffer drains gracefully, director goes quiet rather than hallucinating |
| Cost runaway on a long session | Hard per-match cap, Haiku on the hot path, caching, Batch for eval |
| Two weeks is not enough | Day gates and a buffer day; the day 6 state alone is a legitimate portfolio piece |

---

## 10. Tonight

1. Start recording a UCL match at 21:00 CEST (15:00 ET). Full match, 720p.
2. Clone and read `davidtkunz/soccer-caster` (delay buffer, director,
   possession hysteresis). It is small. Then `pncnmnp/xpong` for turn-taking.
3. `git clone roboflow/sports`, run the soccer radar example on its sample
   clip with `--device mps`, note the fps.
4. Get keys: Anthropic, ElevenLabs, Langfuse.
