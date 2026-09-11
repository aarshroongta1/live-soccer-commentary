# Real-Time Multimodal Agents & Live Commentary: Reference Survey

Research date: 2026-09-10. Scope: open-source projects and reference architectures for real-time multimodal AI agents that watch a video/screen stream and narrate it continuously, plus multi-agent orchestration patterns for latency-sensitive settings. Compiled as reference for a video-first, multi-agent live soccer commentary system (perception layer -> agents that take turns speaking -> streamed text/TTS).

---

## 1. Projects Surveyed

### GetStream Vision-Agents
- **URL:** https://github.com/GetStream/Vision-Agents
- **Stars / activity:** ~8.1k stars, 1,162 commits, very active. MIT licensed.
- **Input modality:** Live video + audio over WebRTC.
- **Models:** Adapters for Gemini Live and OpenAI Realtime; vision processors for Ultralytics YOLO, Roboflow, Moondream; STT/TTS via Deepgram, AssemblyAI, ElevenLabs, Cartesia, Inworld; HeyGen for avatars.
- **Low-latency approach:** Pluggable processor pipeline where each stage gets its own frame rate. Detection can run at 30 FPS while the reasoning model sees 10 FPS, expressed as `gemini.Realtime(fps=10)`. Media latency under 30ms and ~500ms join time on Stream's edge network. Processors run before/after the LLM call.
- **Turn-taking / interruption:** Ships turn detection with VAD, diarization, and "smart turn-taking". Has a text back-channel feature for injecting silent instructions during an active conversation.
- **Biggest weakness:** Turn-taking is still flagged as a work-in-progress plugin area. The framework is shaped around one agent per call, not several agents contending for the mic. Vendor-adjacent (built by Stream, though it works with any video edge network).
- **Relevance:** Closest existing thing to the target architecture. Start here.

### Google gemini-live-api-examples (official)
- **URL:** https://github.com/google-gemini/gemini-live-api-examples
- **Docs:** https://ai.google.dev/gemini-api/docs/live-api and https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs
- **Input modality:** Audio, video frames (JPEG), screen share, text.
- **Hard constraints that matter more than the code:**
  - Video input capped at **1 FPS**, recommended native 768x768.
  - Audio-only sessions limited to 15 minutes; **audio + video sessions limited to 2 minutes**.
  - Connection lifetime ~10 minutes without context window compression.
  - Built-in barge-in: "Users can interrupt the model at any time."
- **Google's own disqualifier (direct quote):** "This specification makes the Live API unsuitable for use cases that require analyzing fast-changing video, such as play-by-play in high-speed sports."
- **Biggest weakness:** Ruled out for this exact use case by its own documentation.

### GetStream "Lessons from Building a Real-Time Football Commentator with Video AI"
- **URL:** https://getstream.io/blog/ai-football-commentator-lessons/
- **The single most useful document for this project.** A team with better models and more resources tried exactly the student's approach, measured it, and published the failure.
- **Architecture:** Two-stage. Roboflow RF-DETR detects players and ball at **5 FPS** -> bounding boxes burned into frames -> Gemini Live or OpenAI Realtime narrates the annotated stream at **1-3 FPS** -> commentary audio out.
- **Measured latency (time-to-first-audio):**

| Provider | Input FPS | Mean TTFA | Range |
|---|---|---|---|
| OpenAI Realtime | 1 | 0.39s | 0.31 - 0.72s |
| OpenAI Realtime | 2 | 0.47s | 0.32 - 1.20s |
| Gemini Live | 1 | 3.06s | 1.52 - 5.05s |
| Gemini Live | 2 | 4.08s | 2.75 - 6.85s |

- **Findings:** OpenAI was roughly **8x faster to start responding** than Gemini Live. Both models were **wrong more than half the time** because they "chose to reply based on a couple of frames rather than the overall context," when sports analysis requires 5-10 seconds of context. Raising frame rate from 1 to 2 FPS degraded **both** latency and accuracy: the bottleneck was model comprehension, not input volume. Fast action is where the models fall apart, since detection confidence drops and the model works with incomplete visual info. Meta's SAM3 for better segmentation was too slow (~1 call per 2 seconds via cloud API).
- **Their conclusion:** "Real-time models aren't ready for high-motion video." Recommended reverting to custom models that recognize game events, with structured prompts and TTS for narration.

### pncnmnp/xPong
- **URL:** https://github.com/pncnmnp/xpong
- **Stars / activity:** ~150 stars; README content dated around May 2025.
- **Input modality:** Game state events, not pixels.
- **Models:** OpenAI `gpt-4o-mini-tts` for "cost-effective, near-realtime" generation plus speech.
- **Architecture (the important part):** Event-driven pipeline. Actions are logged as events -> periodically parsed into metrics -> **metrics ranked by priority to decide which deserve mention next** -> resulting text fed through TTS. Three commentary layers: opening with scorecard, in-game ball-by-ball, closing. Color commentary comes from nearest-neighbor search across **15 simulated years** of tournament data to find comparable past matches.
- **Turn-taking:** **Two commentators alternate**, talking to each other and to the audience. Commentary pauses during critical in-game events and resumes afterward.
- **Biggest weakness:** The game is synthetic and trivially instrumented, so it sidesteps the perception problem entirely. No published latency numbers.
- **Relevance:** The only repo in this survey that actually implements multi-commentator turn-taking driven by a priority-ranked event queue. This is the "director" layer, working, in a small readable codebase.

### showlab/videollm-online
- **URL:** https://github.com/showlab/videollm-online (CVPR 2024, NUS + Meta Reality Labs)
- **Input modality:** Streaming video frames.
- **Core primitive:** Predicts an **`[EOS]` token at each frame** to decide whether to stay silent or generate a response. This is proactive, real-time interaction expressed as a per-frame binary decision.
- **Low latency:** The LIVE (Learning-In-Video-Stream) framework parallelizes video encoding, LLM forwarding for frames, and LLM response generation asynchronously. **5-10 FPS on an NVIDIA 3090; 10-15 FPS on an A100** over 10-minute videos. Supports streaming dialogue in a 5-minute clip at over 10 FPS on A100.
- **Biggest weakness:** Dated backbone; no domain-specific sport evaluation.

### showlab/LiveCC
- **URL:** https://github.com/showlab/livecc | https://showlab.github.io/livecc/ (CVPR 2025)
- **Stars:** ~476 stars, 57 forks, 61 commits.
- **Approach:** Trains a video LLM by **densely interleaving ASR words and video frames by timestamp** - effectively learning to commentate from YouTube commentary tracks. Predicts `[EOS]` to stay silent or emits commentary tokens frame-by-frame, enabling real-time play-by-play narration.
- **Performance:** LiveCC-7B-Instruct beats 72B models (Qwen2.5-VL-72B, LLaVA-Video-72B) on commentary quality **while running in real-time mode**. SOTA at 7B/8B scale on VideoMME and OVOBench. Context config: max 480 frames per video (480/60/2 = 4 min), min 100 visual frame tokens to LLM, max 24k video tokens (leaving 8k for language).
- **Biggest weakness:** The repo explicitly warns that the CVPR camera-ready paper version is wrong and to follow the arXiv version instead. No FPS or latency numbers in the README.

### Proact-VL
- **URL:** https://arxiv.org/abs/2603.03447 | https://proact-vl.github.io/ | PyPI `proact-vl` (ICML 2026)
- **The most directly relevant paper in this survey.** Attacks the "when-to-speak" problem head-on for live gaming commentary.
- **Mechanism:** A **FLAG-token response head**. A lightweight gated MLP over the hidden state of a special `<|FLAG|>` token produces a speaking probability `p_t = sigmoid(MLP(h_t))`, thresholded each second to decide speak vs. stay silent. Threshold 0.5 gave the most practical pattern: ~20s initial silence, then sustained commentary, then alternating speech and silence.
- **Streaming design:** Chunk-wise. Each 1-second chunk carries current visual content, an optional user query, and environmental context from prior commentary. A persistent KV cache accumulates tokens across time steps.
- **Numbers:** Chunk size **1 second**; video sampled at **2 FPS** during inference; handles 10-15 FPS video streams. Cache update **0.08-0.10s**, text generation **0.24-0.32s**, per-token decoding **0.043-0.045s** (roughly constant). Backbones: Qwen series and LiveCC-7B. Training cost ~200 H100 GPU-hours. Peak memory 16-17 GB.
- **Co-commentary (critical for multi-agent):** The Live Gaming Benchmark includes **solo commentary, co-commentary, and user guidance** scenarios. Multi-speaker coordination is achieved by injecting **"other assistants' last-second commentary" into each model's history context**, letting it observe recent peer utterances and adapt its speaking behavior, reducing redundant overlap while maintaining narrative continuity. This is a *decentralized* alternative to a central director agent.

### mit-han-lab/streaming-vlm
- **URL:** https://github.com/mit-han-lab/streaming-vlm | https://arxiv.org/abs/2510.09608 | https://hanlab.mit.edu/projects/streamingvlm
- **Approach:** Real-time understanding of effectively infinite video streams using a compact KV cache, instilled via supervised fine-tuning with full attention on short overlapped chunks (mimicking the inference-time attention pattern without training on prohibitively long contexts).
- **Performance:** Stable, real-time at **up to 8 FPS on a single NVIDIA H100**. 66.18% win rate vs GPT-4o mini on Inf-Streams-Eval. +4.30 on LongVideoBench, +5.96 on OVOBench Realtime without VQA-specific fine-tuning.
- **Biggest weakness:** H100-class hardware for 8 FPS is out of reach for a student project.

### Event-VStream
- **URL:** https://arxiv.org/abs/2601.15655 (CVPR 2026 Findings)
- **Approach:** The "narrate on events, not on a timer" idea made into a model. Represents continuous video as a sequence of discrete, semantically coherent events. Detects meaningful state transitions by integrating motion, semantic, and predictive cues, and **triggers language generation only at those boundaries**. Explicitly argues that fixed-interval decoding produces repetitive output and cache pruning discards temporal information.
- **Performance:** +10.4 points on OVOBench-Realtime over a VideoLLM-Online-8B baseline; close to Flash-VStream-7B with only a general-purpose LLaMA-3-8B text backbone; ~70% GPT-5 win rate on 2-hour Ego4D streams.

### jyrao/MatchTime (MatchVoice)
- **URL:** https://github.com/jyrao/MatchTime | https://arxiv.org/abs/2406.18530 (EMNLP 2024 Oral)
- **Soccer-specific.** Three contributions: hand-annotated timestamps for 49 matches creating the SN-Caption-test-align benchmark; a multimodal temporal alignment pipeline that automatically corrects and filters existing SoccerNet caption data at scale; and MatchVoice, a trained commentary generation model achieving SOTA.
- **Biggest weakness:** Offline generation over clips, not a streaming system. Gives the student a dataset, an aligned corpus, and an evaluation target rather than a live architecture.
- **Related:** https://github.com/chidaksh/SoccerCommentary. GameSight is a two-stage knowledge-enhanced visual reasoning approach to soccer commentary with accurate entity (player/team) references.

### NVIDIA-AI-IOT/live-vlm-webui
- **URL:** https://github.com/NVIDIA-AI-IOT/live-vlm-webui
- **Stars / activity:** ~447 stars, 156 commits, Apache 2.0.
- **Architecture:** WebRTC via Python `aiortc`. **Async processing so video keeps streaming while the VLM processes a frame in the background** - a good pattern to copy. Configurable "Frame Interval" from 1 to 3600 frames (5-30 = heavy GPU use; 60-300 = power saving).
- **Backends:** Any OpenAI-compatible API - Ollama (14+ vision models), vLLM, SGLang, NVIDIA NIM (Cosmos-Reason1-7B), NVIDIA API Catalog, OpenAI. Runs on Jetson Orin/Thor (JetPack 6.x/7.0+), Apple Silicon, x86 Linux, WSL2.
- **Instrumentation:** Real-time display of inference latency (last, average, total count).
- **Biggest weakness:** No published benchmarks, no multi-session support, no turn-taking of any kind, model download "coming soon."

### ngxson/smolvlm-realtime-webcam
- **URL:** https://github.com/ngxson/smolvlm-realtime-webcam
- **Setup:** `llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF` (add `-ngl 99` for GPU), then open `index.html`.
- **Value:** The cheapest possible proof that local sub-second frame captioning works, with no cloud dependency. Entire client is one HTML file.
- **Biggest weakness:** Stateless single-frame captions, no memory, no event model, no turn-taking. A toy, but a useful floor.

### alessioborgi/RealTime-VLM
- **URL:** https://github.com/alessioborgi/RealTime-VLM
- Browser continuously captures webcam frames, posts image+text to any OpenAI-compatible API, displays responses with **sub-second latency**. Works with local or hosted VLMs. Useful as a minimal client pattern.

### Moondream + Photon
- **URL:** https://moondream.ai/blog/photon-real-time-vision-ai-is-finally-here | https://moondream.ai/models
- Not a commentary system, but the perception layer worth pricing out. **Over 60 inferences per second on an H100**, batching up to ~70 requests/sec. Purpose-built CUDA kernels, automatic batching, prefix caching. Runs from H100 servers down to Jetson Orin Nano. Moondream 3 Preview uses 2B active parameters. Anecdotally 8-20 frames/minute on a MacBook Pro M3 with the older 1.6B model.
- Already has a first-class plugin in GetStream Vision-Agents.

### CerebriumAI/realtime-ai-commentator
- **URL:** https://github.com/CerebriumAI/realtime-ai-commentator
- ~5 stars, 47 commits. Basketball commentary. React/TS/Vite frontend, **LiveKit** for video streaming and sync, **Cerebrium** for inference, **Cartesia** for TTS, CV-based player and ball tracking.
- **Biggest weakness:** Essentially undocumented. No benchmarks, no stated latency, no error handling notes.

### Bundesliga on AWS (the structured-event counterexample at production scale)
- **URL:** https://aws.amazon.com/blogs/media/revolutionizing-fan-engagementcer-bundesliga-generative-ai-powered-live-commentary/
- **Architecture:** ~1,600 data points per match (shots, corners, passes) flow from the Bundesliga Datahub into an ECS Fargate container that extracts event attributes -> Lambda builds a prompt instructing the LLM to "generate a captivating football ticker describing what's happening on the pitch," with a selectable style (Sports Journalist, Casual, Gen Z) -> Amazon Bedrock generates text -> DynamoDB persistence -> AppSync GraphQL subscriptions push to the UI.
- **Latency:** **7 to 12 seconds average**, measured from the moment the action happened on the pitch to when the ticker entry is visible in the UI. AWS notes this sits within the broadcasting delay and syncs with live streaming distribution.
- **Notable gap:** **No TTS.** Text only. Audio generation is described as aspirational future work.

### Smaller event-driven commentary bots worth a skim
- `briancaffey/RocketLeagueBotChat` - https://github.com/briancaffey/RocketLeagueBotChat - BakkesMod plugin wiring in-game events to a local TensorRT-LLM service. Event-driven, fully local inference.
- `aradfir/chess-analysis-llm` - https://github.com/aradfir/chess-analysis-llm - live commentary via Ollama + Stockfish, Flask/jQuery.
- `rohansadaphule/lichess-commentary-app` - Stockfish analysis -> Ollama LLM -> pyttsx TTS.
- `chrisbutner/ChessCoach` - neural chess engine with natural language commentary.
- `ably-labs/football-data-live-ai-commentary` - small football data + realtime demo.
- `xISSAx/Alpha-Co-Vision` - frame capture -> caption -> conversational LLM response.
- `menzHSE/vlm_live_demo` - live captioning with OpenCV + MLX + LLaVA/Ollama on macOS.
- `NijatZeynalov/AI-Powered-Sports-Commentary-Generator` - live game stats + NLG + voice synthesis.

### Related streaming-video-LLM literature (fast-moving; see the awesome lists)
- Curated lists: https://github.com/sotayang/Awesome-Streaming-Video-Understanding and https://github.com/ydyhello/Awesome-VLM-Streaming-Video
- **LiveStar** (2025) - perplexity-based verification to trigger responses. https://github.com/sotayang/LiveStar
- **ROMA** (arXiv 2601.10323) - unifies proactive and reactive streaming audio-video interaction with a **lightweight "speak head" parallel to the LM head**, decoupling response timing from content generation. Targets event alerts, real-time narration, reactive QA.
- **Streamo** (2025) - explicit response state tokens (Silence / Standby / Response) in a unified autoregressive sequence.
- **StreamChat**, **VideoChat-Online** (pyramid memory bank), **Flash-VStream**.
- Benchmarks: **OVO-Bench** (backward tracing, real-time understanding, forward active responding), **StreamingBench**, **Inf-Streams-Eval**, **VSAS-Bench**, **StreamingEval**.

---

## 2. Orchestration & Interruption Frameworks

### Pipecat
- **Docs:** https://docs.pipecat.ai/server/utilities/interruption-strategies | https://docs.pipecat.ai/guides/learn/pipeline
- **Mechanism:** Frame-based, and therefore the most transparent about cancellation. Interruption is a control frame pushed through the pipeline (`StartInterruptionFrame`, **now deprecated in favor of `InterruptionFrame`**) that flushes queues and resets accumulators.
- **Configurable barge-in:** `PipelineParams(interruption_strategies=[...])` replaces immediate-interrupt behavior with conditional criteria, e.g. minimum word count or audio volume, to suppress backchannel ("yeah", "mm-hmm"). First strategy evaluating true triggers the interruption.
- **Known leaks (important - a commentary director will cancel constantly):**
  - Issue #4466: `MediaSender.handle_interruptions` calls `_cancel_clock_task()` which discards every `TTSTextFrame` still in `_clock_queue`, including frames already spoken (pts <= clock time).
  - Issue #950: ElevenLabs and PlayHT TTS services fail to interrupt during long LLM completions. Already-sent text survives inside the provider's websocket; `_receive_task_handler` keeps receiving and pushing synthesized audio downstream.
  - `_handle_interruption` in `tts_service.py` (~lines 1031-1040) unconditionally sets `self._streamed_text = ""` with no flush.
  - Issue #3949: unexpected STT-triggered interruption while the LLM is being called can leave the bot permanently mute.
  - Issue #2791: context not updated on user interruptions.
  - PR #719: had to force an ElevenLabs websocket data flush for in-progress requests on `StartInterruptionFrame`.
- **Verdict:** Best documented cancellation semantics, but genuinely leaky at the provider boundary. Read the issue tracker before committing.

### LiveKit Agents
- **Docs:** https://docs.livekit.io/agents/build/audio/ | https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/
- **Cleanest imperative control surface.** `session.say()` (plays a predefined message, can skip TTS with pre-synthesized audio) and `session.generate_reply()` (LLM-driven) both return a **`SpeechHandle`** with an `interrupted` property, `add_done_callback()`, and awaitable completion. `session.current_speech` exposes the active handle. `allow_interruptions` is a per-utterance flag, default True.
- **Adaptive interruption model:** Analyzes streaming audio chunks during overlapping speech to classify true barge-in vs. conversational backchanneling **acoustically**, rather than via fixed timing or volume thresholds.
- **Why it matters here:** For a director that must preempt a mid-sentence agent when a goal is scored, `session.current_speech.interrupt()` followed by a new `say()` is close to a one-liner.
- **Architecture note:** LiveKit bundles media infrastructure with agent logic (built on its own WebRTC layer); Pipecat does not and lets you pick your own transport.

### Vocode
- **URL:** https://github.com/vocodedev/vocode-core
- ~3,773 stars, but **last commit to `vocode-core` was November 2024**, only two open issues, and the project is asking for community maintainers. Attention has moved to the hosted product. **Skip it.**

### General multi-agent frameworks (all the wrong shape, for different reasons)
- **LangGraph:** `interrupt()` (stable since v1.0) pauses execution at a node, persists full state to a checkpoint store, and resumes after human approval - hours or days later. This is **durable human-in-the-loop pause, not sub-second preemption of a token stream**. Great for approval workflows, wrong timescale for commentary.
- **OpenAI Agents SDK:** Sessions preserve conversation history so an agent can pick up, but there is **no mechanism guaranteeing a partially completed multi-step call resumes where it left off**. An interruption means re-running the current turn.
- **AutoGen:** `GroupChat` with a custom `speaker_selection_method` callable **is genuinely the director pattern in library form** - the function receives the last speaker and the GroupChat object and returns the next Agent (or one of 'auto'/'manual'/'random'/'round_robin'). Docs: https://microsoft.github.io/autogen/0.2/docs/topics/groupchat/customized_speaker_selection/ . **But** the loop is strictly turn-based and blocking, and 'auto' mode runs a nested chat over the full message history to pick a speaker, which is far too slow and blows up context length. Known bug: custom `speaker_selection_method` + function calling fails to find registered tools (issue #2472).
- **CrewAI:** Sequential/hierarchical process orchestration, no streaming-preemption story.
- **Claude Agent SDK:** The one general framework with **real in-flight cancellation**. `ClaudeSDKClient.interrupt()` in streaming-input mode stops the current task. Caveat: messages already produced by the interrupted task, including its `ResultMessage`, **remain in the stream and must be drained with `receive_response()`** before reading the response to a new query. Streaming input mode is designed for a long-lived process that handles interruptions and session management. Docs: https://platform.claude.com/docs/en/agent-sdk/streaming-vs-single-mode

### The honest answer on frameworks
Nobody found doing this well uses a multi-agent framework on the hot path. They hand-roll asyncio tasks with `asyncio.Task.cancel()` and a shared event bus, and use Pipecat or LiveKit only as the audio transport underneath. Multi-agent frameworks are built around turn-completion semantics; live commentary is built around turn-*abandonment* semantics.

---

## 3. Synthesis

### 3.1 Recurring architecture patterns for real-time narration

Four patterns show up across nearly every serious system:

1. **Split perception from narration, at different frame rates.** A fast cheap detector at high FPS feeds a slow expensive language model at low FPS. GetStream: RF-DETR at 5 FPS -> realtime model at 1-3 FPS. Vision-Agents: `fps=30` for detection, `fps=10` for LLM reasoning. Proact-VL: 2 FPS sampling into 1s chunks. NVIDIA live-vlm-webui: async background VLM inference while video streams uninterrupted.

2. **Make speaking a *prediction*, not a schedule.** This is the single most important design idea in the recent literature. VideoLLM-online predicts `[EOS]` per frame. LiveCC predicts `[EOS]` to stay silent or emits commentary tokens. Proact-VL uses a `<|FLAG|>` token with a sigmoid speaking probability per second. ROMA uses a speak head parallel to the LM head. Streamo uses explicit Silence/Standby/Response state tokens. Event-VStream triggers generation only at detected semantic boundaries. Fixed-interval decoding produces repetitive, badly-timed output.

3. **Event log -> metric extraction -> priority queue -> text.** xPong's structure: log actions as events, periodically parse into metrics, **rank metrics by priority to decide what deserves mention next**, then generate. This decouples "what is happening" from "what is worth saying," which is exactly the separation a multi-agent commentary system needs. The AWS/Bundesliga system is the same shape at production scale.

4. **Generate ahead, then cancel.** Pre-generate candidate utterances into a buffer while the current one is still being spoken; discard candidates the game has outrun. The "Low-Latency Real-Time Audio Game Commentary System via LLM-Based Parallel Text Generation" paper (arXiv 2606.13322) formalizes this: initiate generation as soon as a new video segment is available while the current utterance is still being spoken, hold multiple candidates in a buffer, rank by relevance to current state, and cancel scheduled utterances that have gone stale.

### 3.2 Realistic latency numbers

| Approach | Measured | Source |
|---|---|---|
| OpenAI Realtime, annotated frames @ 1 FPS | 0.39s mean TTFA (0.31-0.72s) | GetStream |
| OpenAI Realtime, annotated frames @ 2 FPS | 0.47s mean (0.32-1.20s) | GetStream |
| Gemini Live, annotated frames @ 1 FPS | 3.06s mean (1.52-5.05s) | GetStream |
| Gemini Live, annotated frames @ 2 FPS | 4.08s mean (2.75-6.85s) | GetStream |
| Proact-VL, per 1s chunk | 0.08-0.10s cache update + 0.24-0.32s generation | Proact-VL (ICML 2026) |
| Proact-VL per-token decode | 0.043-0.045s | Proact-VL |
| VideoLLM-online end-to-end | 5-10 FPS on RTX 3090; 10-15 FPS on A100 | VideoLLM-online |
| StreamingVLM sustained | up to 8 FPS on one H100 | MIT Han Lab |
| Moondream Photon | 60+ inferences/sec on H100; ~70 req/s batched | Moondream |
| Gemini Live API ceiling | 1 FPS video input, 2-min A/V session cap | Google docs |
| Bundesliga structured events (text only) | 7-12s pitch-to-screen | AWS |
| Broadcast tolerance | <1s preferred; 2-4s too slow | AWS |
| Opta Vision AI match reports | published within 60s of final whistle | Stats Perform |

**Two conclusions fall out of this table.**

First, **OpenAI Realtime is roughly 8x faster to first token than Gemini Live on identical input**, and Gemini Live's 1 FPS / 2-minute-session ceiling makes it structurally unsuitable regardless.

Second, and counterintuitively, **structured-event narration was dramatically slower end-to-end (7-12s) than VLM-on-frames (0.39s TTFA)** in the two published production-scale measurements. That inverts the usual assumption. But the comparison is not apples to apples: the AWS figure includes a full cloud round-trip through Datahub, Fargate, Lambda, Bedrock, DynamoDB and AppSync, while the GetStream figure is time-to-*first-audio*, not time-to-*correct*-audio. On accuracy the comparison is not close at all: the frame-narration models were **wrong more than half the time**. A locally-run structured-event pipeline would not carry the AWS plumbing tax and should land well under 1s.

### 3.3 How interruption and cancellation are actually implemented

**In frameworks, cancellation is a control frame that flushes downstream queues, and it leaks.** Pipecat's issue tracker is the best available documentation of the failure modes, precisely because Pipecat is honest about them: already-committed text survives inside a TTS provider's websocket and inside the output transport's clock queue after the interrupt fires. Expect 200-400ms of audio that cannot be recalled.

**Ranked by ease for this use case:**

- **Easy - LiveKit Agents.** `SpeechHandle` + `current_speech.interrupt()` + `allow_interruptions` per utterance. Adaptive acoustic barge-in classification. Designed for exactly this control pattern.
- **Easy - Claude Agent SDK.** `ClaudeSDKClient.interrupt()` genuinely cancels in-flight generation; just remember to drain the stream.
- **Medium - Pipecat.** Powerful and configurable via `interruption_strategies`, but you will hit the known TTS-flush leaks and need to work around them.
- **Hard - LangGraph.** `interrupt()` solves a different problem (durable checkpointed pause/resume). Wrong timescale by orders of magnitude.
- **Hard - AutoGen, CrewAI, OpenAI Agents SDK.** Turn-based and blocking. AutoGen's `speaker_selection_method` is conceptually right but operationally too slow.
- **What people actually do.** Own the cancellation logic in asyncio: one `asyncio.Task` per in-flight agent utterance, a shared event bus, `Task.cancel()` on preemption, and treat the voice framework as a dumb audio sink.

**Design implication:** build the director assuming cancellation is *lossy*. Do not architect around clean mid-sentence preemption. Two viable mitigations from the literature: (a) generate in short 1-second chunks so the cancellation granularity is small (Proact-VL), and (b) coordinate *decentrally* by feeding each agent its peers' last-second utterances so they self-deconflict rather than requiring a central interrupt (Proact-VL co-commentary).

### 3.4 The three repos most worth reading code from

1. **GetStream/Vision-Agents** - https://github.com/GetStream/Vision-Agents
   The processor pipeline and the per-stage FPS split are the architecture the student wants. Working Gemini Live and OpenAI Realtime adapters, YOLO/Roboflow/Moondream plugins, VAD + diarization turn detection, and a text back-channel for silent mid-conversation instructions. 8.1k stars, MIT, actively developed. Read `Processor`, the golf-coaching example (`ultralytics.YOLOPoseProcessor`), and the cricket DRS example.

2. **pncnmnp/xPong** - https://github.com/pncnmnp/xpong
   Small enough to read in an afternoon and the **only repo here that actually implements two commentators taking turns**, with a priority-ranked event queue deciding what gets said next and pause/resume around critical moments. That is the director layer, working. Ignore the Pong-ness; steal the scheduler.

3. **showlab/videollm-online** - https://github.com/showlab/videollm-online
   Read it for the per-frame speak-or-stay-silent `[EOS]` decision and for the async parallelization of encode / forward / decode, which is the concurrency structure that makes any of this real-time. Pair it with the **Proact-VL** paper (https://arxiv.org/abs/2603.03447) for the modern version of the same idea plus explicit co-commentary handling.

*Runner-up:* **LiveKit Agents'** speech handling (`SpeechHandle`, adaptive interruption docs) if the student goes streamed-TTS rather than text-only.

### 3.5 Strongest single recommendation

The GetStream team tried exactly the video-first approach with better models and more resources than a student project will have, measured it rigorously, and concluded that **real-time models are not ready for high-motion video**. Their models were wrong more than half the time, and increasing frame rate made things worse, not better, because the bottleneck was comprehension rather than input volume.

The design the evidence supports is a **hybrid**:
- A purpose-trained detector (RF-DETR, YOLO, or a fine-tuned SoccerNet event model) produces **structured soccer events** at high FPS. MatchTime/SoccerNet gives you the labeled data for this.
- Agents narrate those **structured events**, not raw frames, using an xPong-style event-log -> metrics -> priority-queue scheduler.
- A VLM (Moondream via Photon, or SmolVLM locally) is used sparingly for **scene color and context**, not play-by-play, at a low frame rate off the critical path.
- Speaking decisions are **predicted** per tick (FLAG-token / EOS-style logic, or a simple learned/heuristic salience threshold), not scheduled on a timer.
- Multi-agent coordination happens by feeding each agent its peers' last-second output (Proact-VL style), with a lightweight director only for hard preemption on high-salience events like goals.
- Cancellation is hand-rolled asyncio; LiveKit or Pipecat is used only as the audio sink.

---

## Sources

- https://github.com/GetStream/Vision-Agents
- https://getstream.io/blog/ai-football-commentator-lessons/
- https://getstream.io/blog/vision-agents-v0-2/
- https://github.com/google-gemini/gemini-live-api-examples
- https://ai.google.dev/gemini-api/docs/live-api
- https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs
- https://github.com/pncnmnp/xpong
- https://github.com/showlab/videollm-online
- https://github.com/showlab/livecc
- https://showlab.github.io/livecc/
- https://arxiv.org/abs/2603.03447 (Proact-VL, ICML 2026)
- https://proact-vl.github.io/
- https://github.com/mit-han-lab/streaming-vlm
- https://arxiv.org/abs/2510.09608 (StreamingVLM)
- https://arxiv.org/abs/2601.15655 (Event-VStream, CVPR 2026)
- https://github.com/jyrao/MatchTime
- https://arxiv.org/abs/2406.18530 (MatchTime, EMNLP 2024)
- https://github.com/NVIDIA-AI-IOT/live-vlm-webui
- https://github.com/ngxson/smolvlm-realtime-webcam
- https://github.com/alessioborgi/RealTime-VLM
- https://moondream.ai/blog/photon-real-time-vision-ai-is-finally-here
- https://github.com/CerebriumAI/realtime-ai-commentator
- https://aws.amazon.com/blogs/media/revolutionizing-fan-engagementcer-bundesliga-generative-ai-powered-live-commentary/
- https://arxiv.org/pdf/2606.13322 (Low-Latency Real-Time Audio Game Commentary via Parallel Text Generation)
- https://docs.pipecat.ai/server/utilities/interruption-strategies
- https://github.com/pipecat-ai/pipecat/issues/4466
- https://github.com/pipecat-ai/pipecat/issues/950
- https://github.com/pipecat-ai/pipecat/issues/3949
- https://docs.livekit.io/agents/build/audio/
- https://docs.livekit.io/agents/logic/turns/adaptive-interruption-handling/
- https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection
- https://github.com/vocodedev/vocode-core
- https://microsoft.github.io/autogen/0.2/docs/topics/groupchat/customized_speaker_selection/
- https://platform.claude.com/docs/en/agent-sdk/streaming-vs-single-mode
- https://github.com/sotayang/Awesome-Streaming-Video-Understanding
- https://github.com/ydyhello/Awesome-VLM-Streaming-Video
- https://github.com/briancaffey/RocketLeagueBotChat
- https://github.com/aradfir/chess-analysis-llm
- https://github.com/ably-labs/football-data-live-ai-commentary
