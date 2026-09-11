# Live Soccer Commentary from Video — Multi-Agent Project Plan (v2.1)

A system that watches soccer video and produces two-voice broadcast commentary
in near real time. A perception layer turns frames into tracks, a symbolic
event layer turns tracks into typed events with confidence scores, a match
state layer turns events into storylines, and a team of cooperating agents
decides what to say, who says it, and when to stay quiet. Narration runs a few
seconds behind the live edge on purpose, so it never announces a goal that did
not happen.

This is a rewrite after surveying roughly 90 repos and papers. The four
surveys are in `docs/research/` and are worth reading in full. Section 2 is
the short version.

---

## 1. Goals

### What the project must prove

1. **Video in, commentary out, live.** The system sees the game. Input is a
   video stream, not a data feed. Only a handful of research demos have done
   this for soccer, and none of them is a usable system.
2. **Real-time systems engineering.** Perception at high frame rate feeding
   narration at low frame rate through a delay buffer, async cancellation of
   in-flight LLM calls, bounded latency, backpressure. Measured, not claimed.
3. **Multi-agent design with a real reason to exist.** A fast caller, a slow
   analyst, a background researcher, a fact gate, and a director. Different
   latency budgets, models, and tools. The first video-first soccer system
   with two voices and turn-taking.
4. **Applied AI rigor.** A symbolic event layer so the model narrates facts
   rather than guesses. Entity grounding to a roster. A speak-or-stay-silent
   stage. An evaluation harness with timing-aware metrics, a reproduced-failure
   baseline, and pairwise LLM-judge scores.
5. **A public demo on footage you have the right to show**, plus a recorded
   demo on broadcast footage.

### Non-goals

- No training of a video-LLM from scratch. LiveCC, Proact-VL, and VLM-TSI
  already did that with hundreds of GPU-hours. You can plug one in later as a
  color source or a baseline.
- No named player identification from broadcast in v1. The best offline
  systems score about 64 out of 100 on this. Use team plus shirt number when
  legible, and full names only when roster and a legible number agree. The EA
  FC demo lane sidesteps this entirely via the on-screen HUD.
- No production audio broadcast quality. Text-first; TTS in the final phase.

---

## 2. What the research says

Full reports in `docs/research/`: `research-commentary.md`, `research-cv.md`,
`research-realtime.md`, `research-fifa-recent.md`.

**Raw frames into a realtime LLM does not work for play-by-play.** GetStream
built exactly that: RF-DETR detections burned into frames, OpenAI Realtime or
Gemini Live narrating. Time to first audio was 0.39 s on OpenAI, but the
commentary was wrong more than half the time, and raising the frame rate made
both latency and accuracy worse. Google's own docs say the Gemini Live API is
unsuitable for high-speed sports play-by-play. This is your baseline to
reproduce and beat, not your architecture.

**The best-engineered systems put a symbolic layer in the middle.** MARIO, a
RoboCup sportscaster, runs a detector, maps to field coordinates, derives
pass, shot, and goal events with velocity and distance rules, and only then
prompts the LLM. The authors say prompting over raw coordinates invites
hallucination. AURA FC, a solo project, reached the same shape independently:
detector, state machine, LLM sees only high-signal events.

**Narrate on a delay.** soccer-caster, a tested but unfinished repo from
August 2026, makes the sharpest argument in the field: a live commentator
must talk about a play whose outcome it does not know, and the worst failure
is announcing a goal that did not happen. So hold a rolling buffer of frames
and state, run perception at the live edge and narration several seconds
behind. Broadcasts already run on delay. Nobody notices, and the whole class
of hallucinated-outcome bugs disappears. The TGLG benchmark names the same
problem formally as "contingency awareness" and "perceptual updating."

**Speak-or-stay-silent belongs as its own stage.** An ACL 2025 demo
(soccer-bg-commentary) makes "is this a moment to speak" the first decision,
using a silence distribution measured from real broadcasts, then routes to
play-by-play or background information, and resolves player names in the
tracking layer before generation. A League of Legends commentator arrived at
the same order independently. Proact-VL learns a per-second speaking
probability. AURA FC found empirically that about one callout every 4 to 5
seconds is the ceiling before it reads as spam.

**Two voices exist, but not on video.** xPong does turn-taking on Pong with a
priority-ranked event queue. MatchCaster does play-by-play plus analyst under
a tick-driven director over StatsBomb replays. Proact-VL coordinates
co-commentators by feeding each its peer's last utterance. Nobody has done
two voices on soccer video.

**Perception is solved in parts.** Player detection on the main camera is
reliable. Ball detection is the weak link; SAHI slicing roughly doubles ball
recall. Pitch calibration runs at 4 to 7 Hz. Every strong action-spotting
model looks at future frames, so none runs live as-is, which is another
reason the delay buffer is the right call. Identity resets at every camera
cut. Start from the Roboflow sports repo, use RF-DETR to avoid the AGPL on
Ultralytics YOLO, and derive v1 events from geometry rules. UniSoccer's
MatchVision is the best soccer-pretrained visual encoder if you need one.
opensportslib is the maintained successor to the pile of SoccerNet challenge
repos.

**No multi-agent framework handles hot-path cancellation.** LangGraph's
interrupt is a durable pause measured in hours. AutoGen's speaker selection is
the right idea but blocking. People hand-roll asyncio and use LiveKit or
Pipecat only as an audio sink. Cancellation is lossy at the TTS boundary.

**Evaluation has three tiers.** METEOR on the SoccerNet captioning benchmark
tops out at 0.26 and cannot tell good commentary from fluent nonsense. LLM
judges over free-form text are better but blind to timing. TRACE, from the
TGLG benchmark, scores content and timing jointly, and TGLG ships a
SoccerNet-narration checkpoint you can benchmark against. SoccerNet-Echoes,
real broadcast commentary transcripts, is CC BY 4.0 and gives ground truth
for when commentators speak.

**Rights.** Every broadcast-footage dataset is research-only. SoccerNet needs
an NDA and cannot be redistributed. Two clean demo lanes exist: footage you
film yourself, or EA FC gameplay, where the HUD gives score and ball-carrier
name for free via OCR.

### Where this project beats the existing ceiling

| Gap in the field | What you build |
|---|---|
| Frame-narration is wrong half the time | Perception to typed events to LLM, narrated behind a delay buffer |
| Systems announce outcomes they have not seen | Delay buffer plus goal confirmation; measured with TRACE |
| Nobody grounds player names in a live system except one zero-star demo | Roster plus team cluster plus jersey number or HUD, resolved in tracking, passed as names |
| Two voices exist only on Pong and stats feeds | Two-voice director with speak predictor on soccer video |
| Evaluation is METEOR or nothing | Perception accuracy vs labels, factual accuracy, TRACE, silence ratio vs real commentators, pairwise judge, cost |
| No soccer commentary system has a public demo | Owned footage or EA FC lane |

---

## 3. Features

### V1

- **Video in.** Local file or RTSP/WebRTC stream. Fixed-camera footage you
  filmed, EA FC captures, and broadcast footage locally for evaluation.
- **Perception overlay.** Player and ball boxes, team colors, tracking IDs,
  and a top-down radar of the pitch beside the video.
- **Delayed two-voice commentary feed.** Caller and analyst lines with speaker
  label, timestamp, beat type, and the event that triggered them. The video
  the viewer sees is the delayed stream, so commentary lands on the play.
- **Agent activity panel.** Each agent's live state: idle, thinking, speaking,
  cancelled, rejected by fact gate. Plus the delay buffer's fill level.
- **Match state panel.** Score, clock, possession, shots, territory,
  momentum, and a running storyline list.
- **Replay controls.** Speed, seek, and a toggle to run narration on ground
  truth labels instead of live perception, to show the perception error
  budget.

### V2

- **TTS with two voices** and an audio queue that respects director cancels.
- **VLM color.** A vision-capable Claude call every 10 to 15 seconds, off the
  critical path, for scene color: celebrations, crowd, injuries.
- **Learned speak predictor** trained on SoccerNet-Echoes timestamps.
- **Pacing.** Speak faster and shorter when the game speeds up, slower in
  lulls, per the "Don't Pause" token pacer idea.
- **Fan Q&A agent** that answers viewer questions without disrupting the feed.
- **Eval dashboard** comparing configurations.

---

## 4. Data

### Development and evaluation: SoccerNet and friends

- **SoccerNet**: 550 broadcast games at 25 fps with labels for action spotting
  (17 classes), ball actions (12 classes), tracking, calibration, jersey
  numbers, captions. NDA for video. Do not redistribute. Use for training and
  evaluating perception, replaying labels as a perfect-perception stub, and
  measuring event precision and recall.
- **SoccerNet-Echoes** (CC BY 4.0): timestamped ASR of real broadcast
  commentary. Use for learning when commentators speak, style reference, and
  the silence-ratio metric.
- **TGLG annotations and checkpoints** (Hugging Face `kpyu/tglg`,
  `kpyu/soccernet-videollm-online`): the TRACE metric and a soccer narration
  baseline model.
- **SoccerReplay-1988** (UniSoccer, NDA): 1,988 matches if you want more
  training data than SoccerNet.
- **AgentPitch**: a simulator where every player is an agent and the event
  stream is exact ground truth. Cheap synthetic matches for testing the
  director and fact gate before perception is ready.
- Optional: DFL Bundesliga Kaggle clips, LFC dataset (INLG 2025) for
  commentary text.

### Demo footage: two lanes, pick one for v1

**Lane A: film your own match.** A Duke club or intramural game from an
elevated fixed position. A fixed camera removes camera cuts and per-shot
calibration, you calibrate the homography once, tracking IDs persist for
minutes, and you have unambiguous rights. Get rosters with shirt numbers.

**Lane B: EA FC gameplay capture.** Perfect visibility, consistent camera,
and the HUD shows score, clock, and the name of the player on the ball.
OCR the HUD and you get entity grounding for free, which is the hardest
unsolved problem on broadcast. Six small repos have tried this; none is
mature, so the lane is open. Caveats: the detector needs its own fine-tune
because jersey colors and camera angles differ from broadcast, and gameplay
capture is EA's copyright, which is broadly tolerated for non-commercial
streaming but is not the same as owning it.

Recommendation: Lane B for fast iteration on the event layer and agents
because ground truth is on screen, Lane A for the public demo because the
rights are clean and it is a better story. Both feeds use the same interface.

### Grounding corpus

Per-match knowledge pack: roster with numbers and positions, team facts,
Wikipedia player intros, and for broadcast matches, player aggregates from
StatsBomb open data where the match overlaps. Postgres with pgvector.

---

## 5. Architecture

```
  Video (file / RTSP / WebRTC / screen capture)
         │
         ▼
  ┌─────────────────────┐
  │  Perception Layer   │  RF-DETR players+ball at 15-30 fps, SAHI tiles for ball
  │  (GPU, live edge)   │  ByteTrack IDs, camera-motion compensation
  │                     │  SigLIP team clusters frozen at kickoff
  │                     │  Pitch keypoints at ~5 Hz, smoothed homography
  │                     │  HUD OCR (EA FC) / jersey voting (broadcast, v2)
  └──────────┬──────────┘
             │  Tracks: (id, team, pitch x/y, velocity, conf) per frame
             ▼
  ┌─────────────────────┐
  │ Symbolic Event Layer│  Geometry rules with hysteresis:
  │  (CPU, per frame)   │  possession, pass, turnover, shot, save, out of play,
  │                     │  set piece, goal. Each event: confidence + evidence.
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │  Match State Engine │  Score, clock, possession %, territory, momentum,
  │  + Entity Registry  │  storylines, track id -> team -> number -> name
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │   Delay Buffer      │  Ring buffer of frames + state, ~6-8 s deep.
  │                     │  Narration cursor trails the live edge and can
  │                     │  peek ahead to the edge to learn outcomes.
  └──────────┬──────────┘
             ▼
  ┌─────────────────────┐
  │  Speak Predictor    │  Per-tick salience + silence pressure -> P(speak),
  │                     │  then route: play-by-play / analysis / background.
  └──────────┬──────────┘  Heuristic in v1, learned from Echoes in v2.
             ▼
  ┌─────────────────────┐
  │     Director        │  Priority queue of candidate beats. Picks speaker
  │  (rules + small LLM)│  and angle. Cancels in-flight tasks on preemption.
  └──┬──────┬──────┬────┘  Feeds each agent its peer's last line.
     │      │      │
  ┌──▼──┐ ┌─▼────┐ ┌▼─────────┐
  │Call-│ │Analy-│ │Researcher│  Caller: fast model, 1 sentence, streamed.
  │ er  │ │ st   │ │ (tools)  │  Analyst: Opus low effort, lulls only.
  └──┬──┘ └─┬────┘ └──────────┘  Researcher: SQL + vector tools, proactive.
     └──────┴────────┐
                     ▼
           ┌───────────────────┐
           │   Fact Gate       │  Names in registry, score matches state,
           └─────────┬─────────┘  claims match events. Reject or rewrite.
                     ▼
           ┌───────────────────┐
           │  Broadcast Bus    │  Persist, SSE to client, TTS queue.
           └─────────┬─────────┘
                     ▼
           ┌───────────────────┐
           │ Next.js frontend  │  Delayed video + overlay + radar + feed.
           └───────────────────┘
```

### Design rules, each traceable to a measured result

1. **Narrate behind the live edge.** Perception runs live; the narration
   cursor trails by a fixed delay, 6 to 8 seconds to start. Agents may peek
   to the live edge for outcomes before committing a line. The viewer watches
   the delayed video, so commentary lands on the play. Latency is a budget
   you choose, not a number you chase. Tune the delay down later and measure
   the accuracy cost.
2. **The LLM never sees pixels or raw coordinates on the play-by-play path.**
   It sees typed events with confidence and names. Vision calls are for color
   only, off the critical path.
3. **Perception and narration run at different rates.** Detection at 15 to
   30 fps, calibration at 5 Hz, event layer every frame, speak decision every
   500 ms, LLM calls only when the predictor fires.
4. **Speaking is a decision stage, not a prompt instruction.** A salience
   score and a silence-pressure term produce a probability each tick, then a
   router picks play-by-play, analysis, or background. Real commentators are
   silent much of the time. Start at roughly one line per 4 to 5 seconds at
   most and measure the silence ratio against Echoes.
5. **Resolve identity in the tracking layer, never in the prompt.** The
   registry maps track id to team to number to name with confidence. Names
   pass downstream only above a bar. On EA FC the HUD makes this exact; on a
   fixed camera, seed identities by kickoff position and maintain them with
   re-identification.
6. **Cancellation is lossy, so utterances are short.** The caller produces
   one sentence. The analyst produces two or three chunks. A goal preempts
   everything; a preempted chunk is dropped, not resumed mid-sentence.
7. **Coordination is mostly decentralized.** Each agent gets the other's last
   line and is told not to repeat it. The director hard-preempts only on
   high-salience events and enforces minimum gaps.
8. **Generate ahead, then discard stale.** While one line plays, pre-generate
   the next candidate from the delayed cursor. Drop it if the game outruns it.
9. **Every claim traces to evidence.** Each event carries the frame ids it
   came from, so the fact gate and the eval can trace any line back to pixels.

### Components

**Perception.** Fork `roboflow/sports` (MIT). Swap YOLOv8 for RF-DETR
(Apache 2.0) fine-tuned on the Roboflow football-players dataset, plus a
separate EA FC fine-tune if using Lane B. Use `roboflow/trackers` for
ByteTrack. Wrap the per-frame scripts in a streaming loop with a ring
buffer. Freeze team clustering at kickoff. Decimate pitch keypoints to every
5th frame and smooth the homography, with rejection and stale-transform
fallback. Ball: SAHI slicing on the region around the last known position;
when the ball is lost, infer possession from player trajectories. Subtract
median player displacement per frame to cancel camera pans before computing
velocities. HUD OCR module for Lane B. Apple Silicon MPS gets 10 to 20 fps
at 640 px for development. Rent a GPU (Modal, RunPod) for full-rate runs.

**Symbolic Event Layer.** Pure Python, unit tested, no models. Possession is
nearest player to ball with hysteresis and temporal smoothing. Pass is a
possession change within a team with ball travel above a threshold. Shot is
ball velocity aimed at the goal mouth from inside a zone. Out of play is
ball outside the pitch polygon for N frames. Goal is ball crossing the line
plus sustained possession reset, plus HUD or scoreboard OCR change where
available. Because narration runs behind the buffer, the goal event is
confirmed before anyone speaks about the shot.

**Match State Engine and Entity Registry.** Score, clock, possession,
territory, momentum, streak storylines. Registry links track ids to team,
number, and roster name with per-link confidence and decay across cuts.

**Delay Buffer.** Ring buffer keyed by frame time holding frames, tracks,
events, and state snapshots. Exposes a narration cursor and a lookahead
window to the live edge. Configurable depth. The frontend plays video from
the cursor, not the edge.

**Speak Predictor.** V1: salience of recent events, decayed, plus silence
pressure rising with quiet time, minus a repetition penalty, with a hard
cap on rate. Routes to play-by-play, analysis, or background. V2: a small
classifier trained on Echoes, where the label is "a commentator spoke within
the next two seconds" and features are your event stream.

**Director.** asyncio. One task per in-flight utterance. Priority queue of
candidate beats. Hard rules for preemption and minimum gaps. A structured
output call on a small model picks the angle when there is real choice.
Tracks each agent's state for the UI. Read xPong, MatchCaster, and
soccer-caster's Director before writing it.

**Caller.** Haiku 4.5 or Sonnet 5, measured. One sentence, present tense,
streamed. Context: last 5 events, state, the analyst's last line.

**Analyst.** Opus 5 at low effort. Fires in lulls and after big moments.
Tools: state query, player stats, context search. Two or three short chunks.

**Researcher.** Sonnet 5 with tools. Pre-computes nuggets at kickoff, reacts
to substitutions and streaks, writes to a shared scratchpad. Background
lane per soccer-bg-commentary: Wikipedia RAG on named players.

**Fact Gate.** Deterministic. Proper nouns must resolve in the registry.
Score and clock mentions must match state at the cursor. Event claims must
match an event in the last N seconds. Rejections logged with reasons.

**Observability.** Langfuse or OpenTelemetry. Every LLM call carries model,
latency, tokens, cost, cache hit, outcome. Every perception frame carries
per-stage timing. Buffer depth and cursor lag are first-class metrics.

---

## 6. Technology

### Perception (Python, GPU)

| Tool | Why |
|---|---|
| `roboflow/sports`, `supervision`, `roboflow/trackers` | Soccer-tuned detection, pitch keypoints, ByteTrack, team clustering, radar. MIT and Apache |
| `rf-detr` | Apache 2.0 real-time detector. Avoids Ultralytics AGPL |
| SAHI | Sliced inference for ball recall |
| PnLCalib weights | Strongest open pitch registration; run decimated |
| `opensportslib` | Maintained SoccerNet-community framework if you need spotting or captioning models later |
| UniSoccer MatchVision | Optional soccer-pretrained encoder for a learned event model in v2 |
| EasyOCR or PaddleOCR | HUD and scoreboard reading |
| PyTorch with MPS or CUDA; Modal or RunPod | Local dev, rented GPU for full rate |

### Narration (Python)

| Tool | Why |
|---|---|
| asyncio, FastAPI, SSE | Hand-rolled director, streaming to client |
| Pydantic v2 | Event and state schemas, structured outputs |
| `anthropic` SDK | Streaming, structured outputs, tool use, prompt caching, vision for color |
| Postgres + pgvector | Events, lines, knowledge pack, embeddings |
| Redis Streams (optional) | Decouple perception process from narration process |
| Langfuse | LLM tracing and cost |
| pytest, hypothesis | Event layer and buffer tests, property tests on geometry rules |
| ruff, mypy, pre-commit, GitHub Actions | Hygiene and CI |

### Frontend (TypeScript)

Next.js, video element driven from the delay cursor, canvas overlay for
boxes and tracks, SVG radar, Recharts for momentum, Tailwind.

### Audio (phase 6)

Kokoro (open, local) or ElevenLabs. LiveKit Agents only if you want WebRTC
delivery; its speech handle has clean interrupt semantics. A learned
importance-to-prosody mapping is a documented stretch idea.

### Read these before writing code, in this order

1. `davidtkunz/soccer-caster`: delay buffer, director importance gate and
   barge-in, possession hysteresis. Small, tested, rationale written down.
2. `pncnmnp/xpong` and `hippograndet/MatchCaster`: two-voice scheduling.
3. `zaemon1251-hesty/soccer-bg-commentary` and its ACL 2025 paper:
   speak-or-not as a stage, utterance routing, names resolved in tracking.
4. `yukw777/tglg` and arXiv 2505.11326: TRACE metric, contingency awareness.
5. `roboflow/sports` soccer example and `shubhamgoel27/soccer-co-commentator`
   build log: your perception starting point and two cheap tricks.
6. GetStream's football commentator post: inherit their measurements.
7. MARIO (arXiv 2607.14809) and Proact-VL (arXiv 2603.03447): symbolic layer,
   speak prediction, co-commentary.

---

## 7. Evaluation

### Perception (against SoccerNet labels, local only)

- Detection mAP for players and ball on held-out clips, with and without SAHI.
- Event precision and recall per class with a 1 s tolerance window against
  action-spotting labels. Report main camera and post-cut separately.
- Calibration error against SoccerNet calibration labels.
- Per-stage latency and frame-to-event latency at the live edge.

### Commentary

- **Factual accuracy**: lines checked against ground-truth labels, not just
  your own events. Report the fact gate's rejection rate too.
- **TRACE**: content plus timing alignment, from the TGLG benchmark.
- **Lag**: chosen buffer depth plus generation time from cursor to first
  token. Report both, and the accuracy curve as depth shrinks.
- **Coverage**: fraction of high-salience labeled events that got a line.
- **Silence ratio**: fraction of time speaking versus real commentators on
  the same match via Echoes.
- **Repetition**: n-gram overlap against the previous 20 lines.
- **Pairwise judge**: an LLM judge compares two systems' commentary on the
  same window with the event log as reference, blind, both orderings. Report
  win rate and judge agreement. Batch API.
- **Cost** per match.

### Baselines

1. **Raw frames to a vision model** every second, the GetStream architecture,
   reproduced with a Claude vision call. Expect it to be wrong often.
2. **TGLG's soccernet-videollm-online checkpoint**, a published real-time
   narration model, run on the same clips.
3. **Single agent on your events**, no director, no fact gate, no delay.
4. **Your full system on ground-truth labels** instead of live perception.
   The gap to the live run is your perception error budget.

---

## 8. Build plan (14 weeks at roughly 10 hours per week)

### Phase 0: Foundations (week 1)
- Repo, tooling, CI, Docker Compose, SoccerNet NDA application.
- `MatchEvent`, `Track`, `MatchState` schemas.
- Label-feed adapter: replays SoccerNet action-spotting labels or AgentPitch
  synthetic matches as events at wall-clock pace with seek and speed.
- Match State Engine and Delay Buffer with tests.
- **Done when:** replaying a labeled match prints correct score and events,
  and the buffer exposes a trailing cursor with lookahead.

### Phase 1: Narration on ground truth (weeks 2 to 3)
- Speak predictor v1, caller agent, SSE endpoint, minimal Next.js feed with
  delayed playback.
- **Done when:** a labeled match gets called in the browser with believable
  silence, and the caller never announces an unconfirmed goal.

### Phase 2: Perception v1 (weeks 4 to 6)
- Fork roboflow/sports, streaming loop, RF-DETR swap, trackers, SAHI for
  ball, camera-motion compensation, frozen team clustering, decimated
  calibration, radar output. HUD OCR if Lane B.
- Detection eval against SoccerNet.
- Film your own match or capture EA FC sessions. Calibrate once.
- **Done when:** a top-down radar of your footage looks right at 10+ fps on
  your Mac, and detection mAP is reported.

### Phase 3: Symbolic events (weeks 7 to 8)
- Geometry-rule event layer with confidence and evidence.
- Event eval against SoccerNet labels. Tune thresholds.
- Swap the label feed for the perception feed. Narration now runs on video.
- **Done when:** the system commentates your footage end to end, and the
  event precision and recall table is in the README.

### Phase 4: Director and the second voice (weeks 9 to 10)
- Analyst, researcher, knowledge pack, fact gate, entity registry.
- Director with priority queue, cancellation, peer context, pacing rules.
- Agent activity panel with buffer depth.
- **Done when:** a goal preempts the analyst mid-chunk and the UI shows it.

### Phase 5: Evaluation (week 11)
- Harness, the four baselines, TRACE, the pairwise judge, cost and lag
  report, accuracy-versus-delay curve.
- **Done when:** the README has a results table beating baselines 1 and 2 on
  accuracy and baseline 3 on judge win rate.

### Phase 6: Ship (weeks 12 to 14)
- TTS with two voices. VLM color calls. Deploy.
- Public demo on your footage. Recorded demo on broadcast footage.
- Architecture diagram, write-up, demo video.
- Stretch: learned speak predictor from Echoes, pacing, fan Q&A.
- **Done when:** a stranger opens a URL and watches your match get called.

### Demo deployment note

Full-rate perception needs a GPU. For the public demo, precompute perception
tracks on your footage once, then run the buffer, event layer, director, and
agents live on every visit. Say so on the page. Show the fully live GPU run
in the demo video. That is an honest and impressive demo.

---

## 9. Repo layout

```
commentary/
  perception/
    commentary_vision/
      detect.py        RF-DETR wrapper, SAHI tiling for ball
      track.py         trackers + team clustering + registry hooks
      motion.py        camera-motion compensation
      calibrate.py     keypoints -> homography, smoothing, fallback
      hud.py           EA FC / scoreboard OCR
      stream.py        ring buffer, per-stage fps, Track emitter
      jersey.py        v2
    eval/              detection + calibration eval vs SoccerNet
  backend/
    commentary/
      feeds/           label_feed.py, agentpitch_feed.py, perception_feed.py, base.py
      events/          rules.py, confidence.py, models.py
      state/           engine.py, storylines.py, registry.py
      buffer/          delay.py (ring buffer, cursor, lookahead)
      speak/           predictor.py, router.py, features.py, train.py (v2)
      agents/          caller.py, analyst.py, researcher.py, director.py, factgate.py
      llm/             client.py, prompts/
      knowledge/       pack.py, store.py, tools.py
      api/             app.py, sse.py
      eval/            harness.py, baselines/, trace.py, judge.py, report.py
    tests/
  frontend/            Next.js
  infra/               docker-compose.yml, modal_app.py
  docs/
    research/          the four surveys
    ARCHITECTURE.md, EVAL_RESULTS.md, diagram.svg
  PLAN.md
  README.md
```

---

## 10. How it reads on a resume

- Built a system that commentates soccer video in near real time: RF-DETR
  perception at N fps, a rule-based symbolic event layer, a delay buffer for
  outcome-aware narration, and five cooperating LLM agents producing
  two-voice commentary. First video-first soccer system with two voices.
- Designed an asyncio director that schedules agents by event salience,
  cancels in-flight streaming calls on goals, and coordinates voices through
  shared context.
- Reduced factual errors from N% (frame-narration baseline) to M% with a
  delay buffer, a deterministic fact gate, and a roster-grounded entity
  registry, measured against SoccerNet labels.
- Built an evaluation harness with perception accuracy, factual accuracy,
  TRACE timing alignment, silence ratio against real broadcast commentary,
  and pairwise LLM-judge win rate; published results against four baselines
  including a published real-time narration model.
- Stack: PyTorch, RF-DETR, ByteTrack, OpenCV, Python asyncio, FastAPI,
  Anthropic API, Postgres and pgvector, Next.js, Modal, Docker, GitHub Actions.

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Ball detection drops out during the moments that matter | SAHI near last position; trajectory-based possession when lost; outcomes confirmed via buffer lookahead |
| Camera cuts on broadcast reset identity | Evaluate on broadcast, demo on fixed camera or EA FC; registry with confidence decay |
| Perception too slow on a Mac | Decimate calibration, 640 px inference, rent GPU for full-rate runs |
| Commentary lags or babbles | Speak stage with silence pressure and rate cap; short utterances; generate-ahead buffer |
| Hallucinated names or events | Delay buffer; fact gate; names only via registry; events only from the event layer |
| SoccerNet NDA delay | Apply week 1; use AgentPitch, Roboflow public clips, and your own footage meanwhile |
| EA FC footage rights for a public demo | Use Lane B for development, Lane A for the public page |
| Scope creep | Phases have done-when gates; TTS and VLM color only in phase 6 |
| Cost | Haiku or Sonnet for the caller, caching, Batch API for judge, per-run cost tracking |

---

## 12. First week checklist

1. Apply for SoccerNet access at soccer-net.org.
2. Read soccer-caster end to end. Then xPong. Both are small.
3. Clone `roboflow/sports`, run the soccer radar example on the sample clip
   with `--device mps`. Note the fps.
4. Read the GetStream football commentator post.
5. Set up the repo layout above, CI, and the event, track, state, and buffer
   schemas.
6. Decide Lane A or Lane B for v1. If A, find a match to film and ask for
   rosters. If B, record two EA FC halves and check the HUD OCR on ten frames.
