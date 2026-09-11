# LLM Sports Commentary: Existing Projects Survey

Reference research for a video-first, multi-agent live soccer commentary system.
Survey date: 2026-09-10. Star counts approximate at time of check.

---

## 1. Video-native commentary models

### LiveCC (ShowLab) — the closest thing to a solved video-first real-time commentator
- **URL:** https://github.com/showlab/livecc
- **Paper:** CVPR 2025, https://arxiv.org/pdf/2504.16030
- **Project page:** https://showlab.github.io/livecc/
- **Stars / activity:** actively maintained, released models + datasets + benchmark
- **Input modality:** raw video frames, streaming
- **Detection:** none as a separate stage — this is the point
- **Generation:** 7B video LLM trained by densely interleaving ASR words with video frames at their timestamps. The model learns to emit either a silence token or the next commentary token, frame by frame.
- **Latency:** genuine real-time mode
- **Evaluation:** LiveSports-3K, a free-form commentary benchmark judged by an LLM
- **Data released:** Live-CC-5M (pretraining), Live-WhisperX-526K (SFT)
- **Result:** LiveCC-7B-Instruct beats 72B general video models (Qwen2.5-VL-72B, LLaVA-Video-72B) on commentary quality while running in real time.
- **Weakness:** single narrator, no persona separation, no factual grounding to roster or score, and it inherits YouTube ASR style so it sounds like a streamer rather than a broadcast booth.

### StreamingVLM (MIT Han Lab) — solves the infinite-stream problem
- **URL:** https://github.com/mit-han-lab/streaming-vlm
- **Paper:** https://arxiv.org/html/2510.09608v1
- **Input modality:** unbounded video stream
- **Architecture:** compact KV cache built from attention sinks, a 16-second vision window, and a 512-token text window. Contiguous RoPE keeps positions numerically contiguous with the last retained token so positions never drift as older tokens are evicted. Training uses overlapped 24-second chunks with 12-second overlap to match inference conditions without prohibitively long training sequences.
- **Latency:** up to 8 FPS on a single NVIDIA H100, stable low per-token latency across infinite streams. Full attention OOMs; overlapping sliding windows are redundant.
- **Evaluation:** Inf-Streams-Eval, 20 complete games averaging 2.12 hours each with per-second frame-text alignment. 66.18% win rate vs GPT-4o mini. +4.30 on LongVideoBench without VQA-specific training.
- **Weakness:** a general streaming perceiver, not a commentator. Style, timing, and excitement modeling are absent.

### MatchTime / MatchVoice — the most-cited soccer-specific work
- **URL:** https://github.com/jyrao/MatchTime
- **Paper:** EMNLP 2024 Oral, https://arxiv.org/abs/2406.18530
- **Stars:** ~105, 27 commits
- **Real contribution is data hygiene:** existing SoccerNet commentary is misaligned with video. Authors hand-annotated timestamps for 49 matches to build SN-Caption-test-align, then built an automatic multimodal temporal alignment pipeline using WhisperX ASR plus an LLM (LLaMA3 as alignment agent) to correct and filter the rest at scale.
- **MatchVoice architecture:** frozen pretrained visual encoder (options: ResNet_PCA_512, C3D_PCA_512, CLIP ViT-B/32, InternVideo), Perceiver-style temporal aggregator, LLM-based decoder. Foundational code derived from Video-LLaMA.
- **Latency:** offline. Batch inference over test sets, plus single-video inference designed for 30-second clips.
- **Evaluation:** pycocoevalcap standard captioning metrics plus GPT-based scoring via OpenAI API.
- **Deps:** Python 3.8+, PyTorch 2.0+, Transformers 4.42.3+, WhisperX, SoccerNet lib.
- **Weakness:** strictly clip-level and offline. The released checkpoint narrates events without knowing who is on the pitch.

### SoccerNet-Caption — the benchmark everyone reports against
- **URL:** https://github.com/SoccerNet/sn-caption
- **Task page:** https://www.soccer-net.org/tasks/dense-video-captioning
- **Paper:** CVPRW 2023, https://arxiv.org/pdf/2304.04565
- **Stars:** ~39, 21 commits
- **Data:** 37k timestamped commentaries over 715.9 hours across 471 broadcast games. Two resolutions (720p, 224p). Features pre-extracted at 2 fps. **Video access requires signing an NDA.**
- **Baseline:** Temporally Aware Pooling, in Benchmarks/TemporallyAwarePooling. 2023 baselines scored METEOR 21.25 and 15.24.
- **Task shape:** localize the moment a caption belongs (like action spotting), then generate the sentence.
- **Primary metric:** METEOR. Also BLEU 1-4, ROUGE-L, CIDEr, Recall, Precision.

### DeLTA Lab — SoccerNet 2024 dense video captioning winner
- **URL:** https://github.com/gladuz/soccernet-caption-deltalab
- **Stars:** ~12, 7 commits
- **Architecture:** BLIP-2-inspired. 4-layer transformer decoder, 512 dim, 8 trainable query tokens, over pre-extracted visual features with a 30-frame window, feeding GPT-2 base / GPT-2 medium.
- **Key changes vs baseline:** end-to-end training with the LLM and vision encoder unfrozen rather than frozen; spotting confidence converted to multi-class softmax classification; low-confidence actions filtered; outputs merged across multiple models with duplicate removal.
- **Results:** METEOR 0.259, BLEU-1 0.396, BLEU-4 0.250, CIDEr 0.495.
- **Note:** 0.259 METEOR is roughly the ceiling of the whole benchmark. That is low enough that METEOR is arguably not measuring anything useful about commentary quality.

### GOAL (THU-KEG) — the knowledge-grounded angle
- **URL:** https://github.com/THU-KEG/goal
- **Paper:** CIKM 2023, https://arxiv.org/abs/2303.14655
- **Stars:** ~15, 3 commits
- **Data:** 8.9k soccer clips, 22k sentences, 42k knowledge triples. Built from 80 English-narrated full games in SoccerNet-v2, filtered down to 20 games as annotation candidates. Human annotation covered commentary proofreading, video-text alignment, and knowledge annotation of entity mentions (players, teams, soccer terms).
- **Baseline:** KGVC model based on ALPRO plus a transformer decoder.
- **Why it matters:** the only dataset that treats naming entities as a retrieval problem rather than a generation problem.
- **Weakness:** effectively abandoned.

### TimeSoccer — end-to-end single-pass over full halves
- **URL:** https://github.com/vpx-ecnu/TimeSoccer
- **Paper:** ACM Multimedia 2025, https://arxiv.org/abs/2504.17365
- **Project page:** https://vpx-ecnu.github.io/TimeSoccer-Website/
- **Stars:** ~2, 1 commit. Treat as a paper artifact.
- **Task:** Single-anchor Dense Video Captioning (SDVC). Jointly predicts timestamps and generates captions in one pass, enabling global context modeling across 45-minute matches, instead of the usual two-step spot-then-caption paradigm.
- **Architecture:** built on TimeChat-7B (LLaMA-2 7B Chat backbone, LoRA fine-tuning). Vision: EVA ViT-G plus InstructBLIP Q-Former. 192 frames sampled per video, configurable max_frame_pos around 192-288.
- **Evaluation:** DVC (COCO-format), TVG (temporal video grounding), VHD (video highlight detection) evaluators in-repo.
- **License:** BSD-3-Clause.

### Commentary Generation for Soccer Highlights
- **Paper:** https://arxiv.org/abs/2508.07543
- Ports MatchVoice to the GOAL highlights dataset, experimenting with training configs, hardware setups, and window sizes for zero-shot performance. Reports promising generalization but is mainly a reproduction and analysis paper. Code released.

---

## 2. Engineered pipelines and real-time attempts

### Vision-Agents (GetStream) — the most valuable negative result in the space
- **Framework URL:** https://github.com/GetStream/Vision-Agents
- **Football example:** https://github.com/GetStream/Vision-Agents/tree/main/examples/03_football_commentator_example
- **Lessons writeup:** https://getstream.io/blog/ai-football-commentator-lessons/
- **Docs:** https://visionagents.ai/examples/football-commentator
- **Stars:** ~8.1k, 682 forks, 1162 commits. Best-engineered codebase in this survey.
- **Architecture:** two-model pipeline. Roboflow RF-DETR for player and ball detection, bounding boxes drawn onto frames, then OpenAI Realtime or Gemini Live narrates the annotated video. They also prototyped Meta SAM3 for semantic detection but cloud API limits capped it at one call every 2 seconds.

**Measured findings:**

| Finding | Value |
|---|---|
| Detection frame rate | 5 fps |
| Frames sent to realtime model | 1 to 3 fps |
| OpenAI Realtime time-to-first-audio | 0.39s mean, range 0.31-0.72s, WebRTC |
| Gemini Live time-to-first-audio | 3.06s mean, range 1.52-5.05s, WebSocket/TCP |
| Effect of raising LLM input 1 fps to 2 fps | latency worsened 0.39s to 0.47s, quality unchanged |
| Commentary factual accuracy | wrong more than half the time |

- **Key insight:** the bottleneck is the realtime model's reasoning over a handful of frames, not the frame supply. Models reason over a few frames rather than sustained sequences and cannot synthesize multi-frame narratives into coherent description. Even a better object detector did not rescue them.
- **Hard constraint:** realtime models require audio or text to trigger a response. Video alone will not prompt output.
- **Framework features worth reusing:** VAD, diarization, smart turn-taking, TTS across ElevenLabs, Cartesia, Deepgram, AWS Polly. Targets sub-30ms A/V latency and 500ms connection setup over Stream's edge network. The golf example shows Gemini Live consuming pose data from YOLO11n-pose at 10 fps via a processor stage.
- **Conclusion they reached:** current realtime APIs cannot narrate high-motion sport from frames alone.

### MARIO — LLM sportscast for RoboCup, the best architectural template found
- **Paper:** https://arxiv.org/abs/2607.14809 (HTML: https://arxiv.org/html/2607.14809)
- **Validation:** three clips streamed live on YouTube, RoboCup German Open, March 14 2026
- **Perception:** YOLOv12 for robot detection, ResNet-18 for team classification, OCR for scoreboard. Camera calibration maps pixels to field space via homography with radial distortion correction. All outputs land in a unified metric field frame.
- **Symbolic event layer (the important part):** the LLM never sees raw coordinates. The system computes hints from kinematics — possession as nearest robot to ball, directional intent from trajectory vectors — then fires discrete pass / shot / goal events through rule-based logic gated on minimum velocity and distance thresholds and confirmed against scoreboard changes. Authors state plainly that interpreting raw coordinate streams through prompting alone is unreliable and invites hallucination.
- **Sportscast policy, dual autonomy:**
  - *Event-reactive lane:* high-priority events bypass queues, priority ordering goals > shots > passes, with temporal guardrails suppressing duplicates and stale calls.
  - *Periodic lane:* during quiet windows, aggregated context (possession trends, ball zone, recent events, history buffer) generates continuity sentences.
- **Causal processing:** no future-frame lookahead.
- **Evaluation:** robot detection RMSE against GameController self-reported logs. Mean error 0.637-0.930m on a 14x9m field, 75% of frames under 1.3m. Commentary coherence assessed qualitatively by manual review only.
- **Weaknesses:** no TTS at all, text only. Manual camera calibration per broadcast angle. Depends on jersey color detection. False positives on ball-tracking edge cases including human referee ball handling. Ground truth sparse at 1-2 Hz. No multilingual support. Code release promised via supplemental GitHub link but availability unconfirmed.

### soccer-ai-commentator (AntoineBohin) — fullest open soccer pipeline
- **URL:** https://github.com/AntoineBohin/soccer-ai-commentator
- **Stars:** ~11, 65 commits, recent activity
- **Five stages:**
  1. Action spotting: EfficientNetV2-B0 CNN backbone with 3D temporal modeling, derived from the 2023 SoccerNet Action Spotting Challenge baseline
  2. Tracking: YOLOv8 plus ByteTrack
  3. Action description: Qwen2.5-VL-7B, chosen after benchmarking six VLMs
  4. Commentary styling: Gemma 3-4B Instruct, chosen for creativity vs reliability balance
  5. TTS: Zonos (Coqui-TTS framework) with voice cloning from reference samples
- **Latency:** offline batch, sequential stages, no streaming
- **Multi-agent:** none, modular pipeline only
- **Evaluation:** none. No metrics, no user study, no baseline comparison.
- **Stated limits:** 24GB VRAM minimum. VLM sees only 10-second clips downsampled to 4 fps at roughly 220p. Complex instructions reduced VLM output reliability. SoccerNet v2 NDA required.

### worldcupvoice — cleanest streaming-transport reference
- **URL:** https://github.com/zicojiao/worldcupvoice
- **Stars:** ~103, 17 forks, 9 commits
- **Architecture:** RTMP ingest through Agora Media Gateway into an RTC channel. The AI commentator backend runs as a server-side participant using the Agora Python SDK, sampling frames from the same channel viewers watch, and publishing synthesized audio back into that channel with synchronized transcripts.
- **TTS:** swappable across OpenAI TTS (default), ElevenLabs (recommended for demo quality), Fish Audio (Chinese voices).
- **Vision/LLM model:** not documented in the README. Code-level read in `research-fifa-recent.md` section 1: it is `gpt-5.4-mini` over the ordinary Responses endpoint, not a realtime API, fed 4 JPEGs per 4-second timer tick with no perception stage. Ignore the 'thin realtime wrapper' classification below for this repo.
- **Cost controls worth copying:** explicit session start/stop, viewer heartbeats to prevent unbilled token burn, hard max runtime TTL per session.
- **Weakness:** roadmap lists lowering frame-to-commentary latency as the top open item. Single-commentator only, limited language and voice variety.

### CerebriumAI/realtime-ai-commentator
- **URL:** https://github.com/CerebriumAI/realtime-ai-commentator
- **Demo:** https://realtime-ai-commentator.vercel.app
- **Stars:** ~5, 47 commits
- Basketball. LiveKit for video streaming and sync, Cartesia for player and ball tracking, Cerebrium for commentary generation, React/TypeScript/Vite/Tailwind frontend synchronizing playback with commentary. No latency metrics documented. Useful only as a wiring example.

---

## 3. Event-data and game-state driven commentary

### LLM-Commentator — the strongest non-video work
- **Paper:** Knowledge-Based Systems, https://www.sciencedirect.com/science/article/pii/S0950705124008530
- **Open access PDF:** https://orca.cardiff.ac.uk/id/eprint/171222/1/1-s2.0-S0950705124008530-main.pdf
- **Code:** https://github.com/Iron-Chef/MascotAI/tree/main/data_processing
- **Data:** web-scraped BBC football commentary paired with Opta event data
- **Approach:** three distinct fine-tuning strategies for open-source LLMs on consumer-grade hardware, targeting near-real-time commentary from raw match event data.

### devoxx-ai-sports-commentary (mneedham) — the streaming-infrastructure version
- **URL:** https://github.com/mneedham/devoxx-ai-sports-commentary
- **Stars:** ~2, 20 commits. Devoxx UK 2024 talk.
- **Architecture:** match events stream into Redpanda. ClickHouse stores live events via materialized view alongside historical player and tournament stats. Flink windows event batches, which go to an OpenAI model together with contextual stats queries. Streamlit app shows recent events and offers an AI-generated message button.
- **Notable stance:** deliberately a human-in-the-loop copilot, inspired by BBC live text coverage, where an editor uses, modifies, or rejects each suggestion.

### Other event-driven repos
- https://github.com/lornamariak/NBA_Live_Commentary_Generation — NBA Stats API live play-by-play plus player stats into an LLM. Duke OpenAI Hackathon project.
- https://github.com/NijatZeynalov/AI-Powered-Sports-Commentary-Generator — live game statistics into NLG plus voice synthesis.
- https://github.com/aditya-018/LLM-Football — StatsBomb ingestion, xG modeling, tactical clustering, Streamlit UI, optional LLM reports.
- https://github.com/cemrtkn/LLM-football-simulation-engine — Mistral-7B-v0.3 QLoRA fine-tune generating realistic match event sequences.
- https://github.com/statsbomb/open-data — free StatsBomb event data.

---

## 4. Multi-agent turn-taking

### xPong — the only project that gets the hard part right
- **URL:** https://github.com/pncnmnp/xpong
- **Stars:** ~150, 6 forks, 72 commits. MIT license.
- **Domain:** Pong, with a simulated 15-year tournament history driven by Elo ratings, culminating in a championship match between the top two players.
- **Turn-taking:** two AI commentators take turns talking to each other and the audience, across three commentary layers — opening commentary with player statistics, ball-by-ball in-game commentary, and post-match closing commentary.
- **Interruption:** commentary is interrupted when an important in-game event occurs, then resumes from where it left off afterward. This is the single most transferable behavior in the survey.
- **Event pipeline:** actions logged as events, periodically parsed into metrics, metrics ranked by priority to decide which deserves mention next, resulting text fed through TTS.
- **Context enrichment:** nearest-neighbor search over past games to surface comparable stat callbacks.
- **Models:** OpenAI gpt-4o-mini-tts. Python 3 plus Eel, requires a Chromium-based browser.

### ESOPN — two personas over screen capture
- **URL:** https://github.com/thefirebanks/esopn
- **Stars:** ~1, 6 commits
- **Domain:** live play-by-play commentary on AI coding sessions, sports-broadcast style.
- **Capture:** MSS screenshots at a configurable interval, default 3 seconds, gated by smart change detection that only triggers commentary when screen difference exceeds 5%.
- **Vision:** Gemini Vision analyzes captured screenshots for code changes, terminal activity, user actions.
- **Personas:** Alex (S1) does high-energy play-by-play describing specific actions. Morgan (S2) does color analysis explaining technical significance and patterns. The commentary LLM orchestrates turn-taking between the two voices.
- **TTS:** Gemini TTS native two-speaker synthesis with distinct voices, free tier.
- **Worth copying despite 1 star:** the change-detection gate and native two-speaker TTS.

### Low-Latency Real-Time Audio Game Commentary via LLM-Based Parallel Text Generation
- **Paper:** https://arxiv.org/pdf/2606.13322
- **Domain:** esports and gaming video
- **Technique:** generates multiple commentary continuations in parallel rather than sequentially, then selects the most appropriate, so the system never waits on a single decode path.
- **Pipeline:** video input capture, parallel LLM generation, TTS synthesis, audio output.
- **Evaluation:** ROUGE plus crowdsourced human annotation plus real-time latency testing on game footage.

### Other turn-taking / commentary repos
- https://github.com/aradfir/chess-analysis-llm — live chess commentary via Ollama plus Stockfish, Flask/jQuery.
- https://github.com/jayanth9844/Ai-Commentator — chess commentary, modular architecture.
- https://github.com/dylanhogg/llmbanter and https://github.com/famiu/llm_conversation — generic two-LLM dialogue, useful only as pattern reference.
- https://github.com/ayushpai/Sports-Buddy — GPT-4 Vision plus Whisper plus OpenAI TTS conversational sports assistant.

---

## 5. Adjacent and supporting work

- **SoccerAgent** — multi-agent soccer understanding. https://arxiv.org/abs/2505.03735, project page https://jyrao.github.io/SoccerAgent/. Built on SoccerWiki, the first large-scale multimodal soccer knowledge base covering players, teams, referees, venues. DeepSeek-v3 drives both planning and execution modules over a tool library. Handles action classification, commentary generation, replay grounding, jersey color recognition, multi-view foul recognition. Evaluated on SoccerBench, ~10K multimodal multi-choice QA pairs across 13 tasks. It is QA, not live narration, but **SoccerWiki is the entity-grounding asset nobody else has.**
- **MSUE: Multi-Modal Soccer Understanding Expert** — https://arxiv.org/abs/2606.12106. 2026 SoccerNet VQA Challenge solution. LLM dynamically dispatches questions to text, image, and video experts.
- **COACH: A Multi-Agent Framework for Sports Video Analysis** — https://arxiv.org/pdf/2512.01853
- **SportR: benchmark for MLLM reasoning in sports** — https://arxiv.org/html/2511.06499v3
- **BoxComm: category-aware commentary and narration rhythm in boxing** — https://arxiv.org/pdf/2604.04419. Narration rhythm is a rare and directly relevant framing.
- **Knowledge Guided Entity-aware Video Captioning, basketball benchmark** — https://arxiv.org/pdf/2401.13888
- **AI-Generated Game Commentary: A Survey and Datasheet Repository** — https://arxiv.org/html/2506.17294v1
- **Curated lists:** https://github.com/lg-li/awesome-streaming-agents and https://github.com/sotayang/Awesome-Streaming-Video-Understanding are both current and on-point. Also https://github.com/wywyWang/Awesome-Sports-Analytics and https://github.com/AtomScott/awesome-sports-analytics.
- **AutoCam-AI** — https://github.com/chele-s/AutoCam-AI. RF-DETR plus Extended Kalman Filter virtual camera with low-latency RTMP/YouTube streaming and PID smoothing. Not commentary, but a good reference for production-grade low-latency sports video plumbing.

---

## 6. Synthesis

### Three architectures recur

1. **The academic clip captioner.** Pre-extracted features at 2 fps over a fixed 30-second window, feeding a small decoder, scored by METEOR. MatchTime, sn-caption, DeLTA Lab, GOAL.
2. **The cascade.** Detector, then tracker, then VLM description, then LLM restyling, then TTS. Modular and debuggable, but it accumulates latency and error at every hop. soccer-ai-commentator, MARIO, Vision-Agents.
3. **The thin realtime wrapper.** Sampled frames thrown at a hosted realtime API. worldcupvoice, CerebriumAI, most hackathon demos.

Only LiveCC and StreamingVLM collapse the stack into a single streaming model. Only MARIO puts a deliberate symbolic layer between perception and the LLM, and it is explicit about why: raw coordinates in a prompt produce hallucination.

### What nobody does well

- **Real-time and accurate simultaneously.** Unsolved for open-field sport. The Vision-Agents measurements are the hard evidence: sub-0.4s time-to-first-audio is achievable, but the content is wrong more than half the time because the model reasons over a handful of frames.
- **Entity grounding.** Almost every system says "the player" because jersey-number-to-roster resolution is skipped. GOAL and SoccerWiki are the only serious attempts, and neither is wired into a live commentator.
- **Knowing when not to speak.** Silence is half of commentary. Only LiveCC models it explicitly, via a silence token. MARIO approximates it with a periodic-vs-reactive split.
- **Real turn-taking in sport.** xPong and ESOPN prove the pattern works, but on Pong and on screencasts. No soccer system has two voices.
- **Evaluation.** METEOR at 0.259 cannot distinguish good commentary from fluent nonsense. The only credible alternatives are LiveCC's LiveSports-3K LLM-as-judge and StreamingVLM's pairwise win rates.
- **Narrative state across a match.** Nothing tracks that this is the third foul by the same defender, or that a team has been pinned in their own half for ten minutes. xPong's nearest-neighbor stat callbacks are the closest thing, and that is on a toy domain.

### Read the code of these three

1. **https://github.com/showlab/livecc** — the streaming video-to-commentary core, plus a judged evaluation harness you can reuse instead of METEOR.
2. **https://github.com/pncnmnp/xpong** — event priority ranking, interruption with resume, and two-commentator turn-taking. The single best fit for a multi-agent design, despite the toy domain.
3. **https://github.com/GetStream/Vision-Agents** — production transport, turn detection, TTS wiring. Read alongside https://getstream.io/blog/ai-football-commentator-lessons/ so you inherit their measurements instead of repeating the experiment.

Then read the MARIO paper at https://arxiv.org/abs/2607.14809 for the symbolic event layer even though its code release is uncertain, and https://github.com/jyrao/MatchTime if you need aligned training data, since its alignment pipeline is more valuable than its model.

### Implications for a video-first multi-agent design

- Do not send raw frames to a realtime API and expect play-by-play. That path is measured and it fails.
- Put a symbolic event layer between perception and generation, MARIO-style, and let agents reason over typed events with confidence scores rather than pixels.
- Separate the reactive lane from the periodic lane. Goals interrupt; filler fills.
- Resolve entities once, in a tracking layer, and pass names downstream. Never ask the language model to read a jersey.
- Budget evaluation from day one as LLM-as-judge over free-form commentary, following LiveSports-3K. Captioning metrics will mislead you.
