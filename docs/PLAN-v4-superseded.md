# Live Soccer Commentary — Sprint Plan v4

Two-week build of a system that watches a live soccer broadcast on your
screen and produces two-voice commentary with audio. Video leads, the data
feed follows. Six cooperating agents. Measured against Opta ground truth.

Research: `docs/research/` (four surveys, ~90 repos and papers). Earlier
plans: `docs/PLAN-v2-full.md` (14-week CV-heavy), `docs/PLAN-v3-superseded.md`.

**The decision (2026-09-10).** Perception from the video calls the play
first. The free ESPN play-by-play feed arrives seconds later and confirms,
names, and grades it. The delay buffer holds narration a few seconds behind
the live edge so outcomes are known before lines commit.

**The fallback, decided up front.** `lead_source` is a config flag with two
values, `video` and `feed`. If by the end of stage 6 the video path is too
slow or too wrong (criteria in §7), flip it to `feed`: the feed triggers
events, held until the on-screen clock reaches them, and video supplies only
timing and texture. Nothing downstream changes.

---

## 1. Stages

One bullet per stage. Tools, then done-when.

1. **Capture.** ffmpeg avfoundation screen grab, 720p, 15 fps, plus the
   broadcast audio, into an OpenCV loop. Done when any video on screen
   streams into Python in real time with audio.
2. **Feed adapter.** ESPN scoreboard and summary endpoints (no key), polled
   every 5 s. Parses key events, play-by-play text, rosters, clock, score,
   with wall-clock timestamps. One file, swappable for Sportmonks. Done when
   a live match's events print as they happen and Saturday's latency test
   (§8) is logged.
3. **Detect and track.** `roboflow/sports` pretrained weights, `supervision`,
   `roboflow/trackers` ByteTrack, team colors frozen at kickoff, MPS. Cut
   detector and green-ratio shot classifier so rules only run on the main
   camera. Done when players and ball have stable boxes at 10+ fps.
4. **Audio triggers.** Whistle and crowd-roar detector on the broadcast audio
   (small spectral classifier or YAMNet). Done when stoppages and big
   moments fire under 1 s.
5. **Rules.** Pure Python: possession with hysteresis, pass, turnover, shot
   toward goal, out of play. Pydantic events with confidence and evidence
   frame ids. Done when precision and recall against the feed are in a table.
6. **Scout.** Sonnet 5 vision, structured output, three frames from behind
   the cursor, fired on whistle, cut, crowd spike, and a 10 s fallback.
   Returns phase of play, attacking team, replay, celebration, VAR, injury,
   card, sub, name on graphic, clock, score. Done when a card and a corner
   in the recording appear as events with confidence. **Stage 6 gate: apply
   the §7 criteria and set `lead_source`.**
7. **Reconciler and delay buffer.** Video events and feed events merge into
   one log. A feed event within the window confirms or corrects the matching
   video event and attaches names. Delay buffer of 8 s holds frames, audio,
   events, state; narration cursor trails with lookahead. Done when a video
   shot becomes a named, confirmed shot before the caller speaks.
8. **State and knowledge.** Match state engine (score, clock, possession,
   momentum, storylines). Entity registry from feed rosters. Researcher
   agent (Opus 5 + web search) builds the knowledge pack at kickoff. MCP
   server (FastMCP) exposing state, events, knowledge. Done when the
   analyst can answer "who has the most fouls" over MCP.
9. **Agents.** Caller (Haiku 4.5, one streamed sentence). Analyst (Opus 5
   low effort, MCP tools, lulls only). Director (asyncio priority queue,
   Task.cancel on preemption, min gaps, peer context). Fact gate
   (deterministic; confidence sets how specific a line may be; names only
   from the registry). Speak predictor (salience + silence pressure, max one
   line per 4 to 5 s, repetition gate). Langfuse on every call. Done when a
   goal preempts the analyst mid-chunk and the trace shows the cancel.
10. **Output.** FastAPI + SSE, delayed MJPEG video with overlay, Next.js page
    with feed, agent activity, source-of-each-line badge, match state.
    ElevenLabs two voices with a cancel-aware audio queue. Done when a
    stranger watches 15 minutes with sound.
11. **Eval.** Ground truth is the feed. Metrics: video event precision and
    recall per class, factual error rate of lines, lag, silence ratio,
    repetition, cost, pairwise judge via Batch API. Baselines: frame
    narration every 4 s (worldcupvoice), delay 0, feed-led. Done when the
    results table and the accuracy-vs-delay chart exist.
12. **Ship.** README with diagram, results, three-command quickstart, "why
    no agent framework" section, honest limitations. 90 s demo video.
    Public repo. Then the UCL matchday 2 live recording on 13/14 Oct.

Optional, in order, if time remains: Echoes style retrieval, SoccerNet
spotter on the buffer window, calibration and radar, trained speak predictor.

---

## 2. Architecture

```
  Screen capture (ffmpeg, video + audio)          ESPN feed (poll 5 s)
        │                                                │
        ▼                                                │
  ┌──────────────────────┐   ┌──────────────────┐        │
  │ Detector + tracker   │   │ Audio triggers   │        │
  │ cut / shot classifier│   │ whistle, roar    │        │
  └──────────┬───────────┘   └────────┬─────────┘        │
             ▼                        │                  │
  ┌──────────────────────┐            │   ┌──────────────▼───────┐
  │ Rules: possession,   │◄───────────┘   │ Feed events + text   │
  │ pass, turnover, shot,│                │ names, clock, score  │
  │ out of play          │                └──────────────┬───────┘
  └──────────┬───────────┘                               │
             │      ┌──────────────────┐                 │
             │      │ Scout (Sonnet 5) │ on whistle/cut/ │
             │      │ scene, card, VAR │ roar/10 s       │
             │      └────────┬─────────┘                 │
             ▼               ▼                           ▼
  ┌──────────────────────────────────────────────────────────┐
  │ Reconciler: one event log. Video calls first; feed       │
  │ confirms, corrects, names. lead_source = video | feed    │
  └──────────────────────────┬───────────────────────────────┘
                             ▼
  ┌──────────────────────┐   Match state, registry, storylines
  │ Delay buffer, 8 s    │   Narration cursor trails live edge
  └──────────┬───────────┘
             ▼
  Speak predictor -> Director (asyncio, cancel) -> Caller / Analyst / Researcher
             -> Fact gate -> Bus (SSE, TTS queue) -> Next.js + ElevenLabs
```

Design rules: narrate behind the live edge; the caller sees typed events,
never pixels; speaking is a decision stage; names resolve in the registry
before the prompt; utterances are short because cancellation is lossy; every
line carries its source and evidence.

---

## 3. Agents

| Agent | Model | Fires | Produces |
|---|---|---|---|
| Caller | `claude-haiku-4-5` | Speak predictor, play-by-play | One sentence, streamed, < 1.5 s |
| Analyst | `claude-opus-5`, effort low | Lulls, after big moments | 2 to 3 chunks, MCP tools, cancellable |
| Researcher | `claude-opus-5`, web search | Kickoff, subs, streaks | Knowledge pack, storylines |
| Scout | `claude-sonnet-5`, vision | Whistle, cut, roar, 10 s | Typed scene evidence with confidence |
| Fact gate | none | Every line | Accept, rewrite, reject with reason |
| Director | asyncio (+ Haiku for angle) | Every tick | Speaker, angle, cancels |

Prompt caching on all agents. Batch API for the judge. Rough cost per match:
about $10, mostly the scout.

---

## 4. Stack

| Layer | Choice |
|---|---|
| Capture | ffmpeg avfoundation, OpenCV |
| Feed | ESPN public endpoints (adapter), Sportmonks Starter €29/mo as fallback |
| Perception | `roboflow/sports`, `supervision`, `roboflow/trackers`, `rf-detr`, PyTorch MPS |
| Audio | librosa or YAMNet for whistle and crowd |
| Core | Python 3.12, `uv`, Pydantic v2, asyncio, pytest + hypothesis |
| LLM | `anthropic` SDK: streaming, structured outputs, vision, caching, web search tool, Batch |
| Tools | MCP server via FastMCP |
| Observability | Langfuse |
| API / UI | FastAPI, SSE, MJPEG; Next.js, Tailwind |
| Audio out | ElevenLabs streaming; LiveKit Agents as stretch |
| Hygiene | ruff, mypy, pre-commit, GitHub Actions, Docker Compose |

Not used, and why, in the README: LangGraph, CrewAI, AutoGen (turn-completion
frameworks; commentary needs turn abandonment). LangGraph is acceptable for
the researcher's kickoff workflow if you want the keyword; say why it lives
there and not in the director.

---

## 5. Schedule

| Day | Stages | Gate |
|---|---|---|
| 1 | Repo, CI, schemas, feed adapter (2) | Feed events print live |
| 2 | Capture (1), detector (3) | Boxes at 10+ fps on a recording |
| 3 | Audio triggers (4), rules (5) | Precision/recall table vs feed |
| 4 | Scout (6) | Card and corner detected |
| 5 | Reconciler, delay buffer (7). **Stage 6 gate, set lead_source** | Named confirmed shot before speech |
| 6 | State, registry, researcher, MCP (8) | Analyst answers over MCP |
| 7 | Caller, speak predictor, fact gate, SSE, minimal page | Match called in browser |
| 8 | Analyst, director, cancellation, agent panel | Goal preempts analyst |
| 9 | Live run on a weekend match; tune | 15 min watchable |
| 10 | Eval harness, baselines, judge, delay chart (11) | Results table |
| 11 | ElevenLabs, audio queue (10) | Demo has sound |
| 12 | Hardening: reconnects, feed gaps, refusals, cost cap | 90 min without restart |
| 13 | README, diagram, demo video (12) | Three-command quickstart works |
| 14 | Buffer | |
| 13/14 Oct | UCL matchday 2 live recording | Showcase |

---

## 6. Evaluation

Ground truth is the feed, aligned to video time via the on-screen clock read
by the scout. Report video-only precision and recall per class, then the
reconciled system's factual error rate per line, lag (buffer + generation,
p50 and p95), silence ratio, repetition, cost. Pairwise judge, blind, both
orderings, Batch API. Baselines: worldcupvoice-style frame narration, delay
0, feed-led. Headline chart: accuracy versus delay depth at 0, 2, 4, 8 s.

---

## 7. Stage 6 gate: video or feed leads

Measured on 30 minutes of a recorded match against the feed:

- Video precision on goals, shots, corners, and cards below 60 percent, or
- Frame-to-event latency p95 above the delay window (8 s), or
- Detector below 8 fps on your Mac with the scout running

Any one of these flips `lead_source` to `feed`. Document the measurement
either way; it is a README result regardless of outcome.

---

## 8. Open checks, in order

1. **Saturday: ESPN publish latency.** Poll every 5 s during a live Premier
   League match. Log wall-clock of first appearance for each event versus
   when it shows on your stream. Need: feed leads the stream, or lags by
   less than the buffer. If it fails, Sportmonks Starter.
2. Detector fps and ball recall on the UCL recording at 640 px.
3. ElevenLabs voice latency and interrupt behavior.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| ESPN endpoint changes or throttles | Adapter file; Sportmonks fallback; cache last good state |
| Feed slower than stream | Video leads anyway; feed becomes late confirmation; measured in §8 |
| Ball lost most of the time | Possession from players; shots and goals confirmed by feed |
| Perception too slow or wrong | §7 gate flips to feed-led; project still ships |
| Scout misses a 3 s card | Event-triggered on cuts; feed catches it within seconds |
| Cost runaway | Per-match cap, Haiku on the hot path, caching, Batch for eval |
| Rights | Process locally, post short clips, never host broadcast video |

---

## 10. Resume lines

- Built a real-time system that commentates live broadcast soccer from a
  screen capture: RF-DETR perception at 12 fps, audio event triggers, a
  vision scout, a rule-based event layer reconciled against a live data feed,
  an 8 s delay buffer, and six cooperating Claude agents with two voices.
- Designed an asyncio director that cancels in-flight streaming calls on
  goals, exposed match state to agents over MCP, traced every call in
  Langfuse.
- Measured video event detection against Opta ground truth on N live
  matches; cut factual errors from X% (frame-narration baseline) to Y%;
  published the accuracy-versus-delay curve.
