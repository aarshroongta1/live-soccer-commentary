# FIFA, EA FC, and Recent Soccer Repos: Follow-Up Survey

Targeted second pass over `research-commentary.md`, `research-cv.md`, and `research-realtime.md`.
Section 1 is a code-level read of a repo those surveys already list. Everything from Section 2
onward is **new** — none of it appears in those three files.

Survey date: 2026-09-10. Method: a source read of one cloned repository, plus authenticated
GitHub REST search (60 distinct queries across `search/repositories`, sorted by recency), Hugging
Face model and space APIs, and web search for arXiv and hackathon clusters. Star counts, commit
counts, and dates pulled live from the API.

Verdict up front: **the surveys missed three genuinely important academic projects** (UniSoccer,
TGLG/VLM-TSI, and the ACL 2025 background-information system), **one outstanding independent
engineering project** (soccer-caster), and **a large 2026 World Cup hackathon wave** that is mostly
noise but contains a few real ideas. The EA FC / FIFA game-footage angle the user remembered is
real but genuinely thin — about six repos, none of them mature. Separately, reading worldcupvoice's
source corrects how the earlier survey classified it: there is no realtime API in it, and its
entity grounding is a prompt paragraph rather than a perception stage.

---

## 1. Code-level read: worldcupvoice

Requested follow-up. `research-commentary.md` covers this repo at README level and notes the
vision and LLM model are undocumented. They are not documented, but they are in the code.

- **URL:** https://github.com/zicojiao/worldcupvoice
- **Read at commit:** shallow clone of `main`, 2026-09-10. 103 stars, 17 forks.
- **Shape:** Next.js frontend plus a FastAPI Python backend under `server/`. The whole commentator
  is one file, `server/app/backend_commentator.py`, 1,837 lines, with 734 lines of tests beside it.
  Defaults live in `server/app/config.py`; personas in `server/app/commentator_profiles.py`.

**Correction to the existing survey entry.** The survey files worldcupvoice under "the thin
realtime wrapper — sampled frames thrown at a hosted realtime API." The first half is right, the
second is wrong, and the difference matters. There is no realtime API here. It calls the plain
**OpenAI Responses endpoint** (`https://api.openai.com/v1/responses`) once every four seconds and
waits for the whole answer. That sidesteps the Vision-Agents constraint that realtime models need
audio or text to trigger a response, at the cost of giving up streamed audio for most providers.

### Which model watches, and at what rate

| Parameter | Value | Where |
|---|---|---|
| Vision model | **`gpt-5.4-mini`** (env `OPENAI_VISION_MODEL`) | `config.py` |
| API | OpenAI Responses, `POST /v1/responses`, HTTP timeout 35s | `backend_commentator.py:35` |
| Frame sample cadence | **0.55 s**, so ~1.8 fps pulled off the stream | `commentary_frame_sample_seconds` |
| Frames per request | **4**, oldest first | `commentary_context_frames` |
| Visual window per call | ~2.2 s of history | 4 x 0.55 s |
| Commentary cadence | **one call every 4.0 s**, max ~15/min | `commentary_interval_seconds` |
| Frame encoding | Agora I420 to JPEG, max width **960 px**, quality **72**, base64 data URI | `_agora_frame_to_jpeg_base64` |
| Generation limit | **`max_output_tokens: 40`**, `temperature: 0.55` | `_describe_frames` |
| TTS | OpenAI `gpt-4o-mini-tts` (default), ElevenLabs `eleven_flash_v2_5`, Fish Audio `s2-pro` | `config.py` |

So the model sees roughly **1.8 fps of visual evidence but is only asked to speak at 0.25 Hz**, and
each answer is capped at 40 tokens. The 40-token cap is the single most consequential number in the
repo: it makes a wrong call cheap and short, and it forces the one-sentence broadcast cadence that
the prompt also asks for.

### How frames are batched

A thread-side `_AgoraRemoteFrameObserver` receives every decoded frame from the Agora channel,
filters by `remote_uid` so it only watches the match feed and not other participants, and applies
a wall-clock gate: if less than 0.55 s has passed since the last kept frame, drop it. Kept frames
are converted to JPEG on the observer thread and handed to the asyncio loop with
`call_soon_threadsafe`.

The queue is bounded at `context_frames * 3` = 12 and **drops the oldest frame when full**, so the
system degrades toward freshness rather than backing up. A separate `deque` holds the buffer that
`_latest_frames()` reads. There is no motion gating, no scene-change detection, and no keyframe
selection — the sampler is a pure fixed-rate decimator. Compare ESOPN, which only triggers when
the screen differs by more than 5%; worldcupvoice pays for every window whether or not anything
moved.

### The prompt

`_build_visual_prompt` assembles a fresh prompt on every call. It is not a translated template:
there are three fully separate hand-written prompts for Chinese, French, and English, each written
in the target language. Structure, in order:

1. **Role framing, stated as a negation.** "You are a live football play-by-play commentator,
   **not an image captioner**." That single line is doing a lot of work against the default
   behaviour of a vision model handed four JPEGs.
2. **Persona.** Profile label plus a `style_prompt` from `commentator_profiles.py`.
3. **Match context.** Teams, competition, venue, date, storyline, and jersey colours.
4. **Roster map.** Both squads as `#23 Martinez (Emiliano Martinez) [starter/GK]`, with optional
   per-player `notes`, plus free-text `playerIdentificationNotes` and `broadcastNotes`.
5. **Video clock.** `Current video clock in the source: {t}s`, from the frame's `render_time_ms`.
6. **Ordering instruction.** "a short burst of frames, oldest first and newest last."
7. **What to call**, as an explicit list: ball movement, dribble, pass, cross, shot, save,
   clearance, press, counterattack, defensive line, celebration, crowd surge, players organizing.
8. **Length and cadence.** "short when the action is fast, longer when the play is developing,
   usually 4 to 16 words, one sentence max. It is okay to sound clipped, urgent, or mid-play."
9. **The NO_CALL rule** (below).
10. **The naming procedure** (below).
11. **A hallucination blocklist** (below).
12. **The last 5 calls**, verbatim, under "Recent calls to avoid repeating."

### How it decides when to speak

Three gates, and only the middle one is interesting.

1. **Fixed cadence.** The loop runs every 4 s regardless. There is no event trigger.
2. **The model votes.** The prompt instructs: *"Default to a grounded call when a live game,
   players, pitch, or ball-side action is visible. Return exactly `NO_CALL` only when the newest
   frame is not readable, no football action is visible, or the scene is clearly a static
   timeout/replay/crowd-only shot with no new visible change."* `_is_no_call()` normalizes and
   matches `no_call`/`nocall`/`no call`, and the turn is dropped silently.
3. **Audio backpressure.** After publishing, `_wait_for_audio_drain()` blocks until the outbound
   audio buffer falls to **250 ms** (`commentary_audio_drain_target_ms`), with an **8 s** timeout.
   The commentator physically cannot start a new call while the previous one is still being spoken.

That third gate is the closest thing to a turn-taking mechanism, and it is a side effect of the
audio pacer rather than a design intent. Note the asymmetry: the silence decision is delegated
entirely to the vision model's judgement in prose, with a default that *biases toward speaking*
("Default to a grounded call"). There is no learned silence model, no importance threshold, and no
priority ordering of events. Compare the ACL 2025 system (Section 2), which makes speak/don't-speak
its own trained stage before generation.

### How it handles repetition

This is the most carefully engineered part of the file, and it is the part most worth stealing.
`_is_repetitive_commentary()` checks each new call against the **last 4** accepted calls:

- Exact match after normalization (lowercase, strip everything but `[a-z0-9 ]`), reject.
- Otherwise compute content words: split, drop tokens of length ≤ 2, drop a 16-word stoplist
  (`a an and are as at for from in is of on the to with`).
- If either side has fewer than 4 content words, skip the comparison — too short to judge.
- Otherwise **Jaccard similarity** over content-word sets; if `|A ∩ B| / |A ∪ B| >= 0.72`, reject.

Rejected calls are logged and dropped, and crucially **are not appended to history**, so a
near-duplicate does not poison the comparison window. Accepted calls append to a list truncated to
the **last 8**; the **last 5** are injected into the next prompt as negative examples. So there are
two overlapping defences: prompt-level avoidance over 5 calls, and a hard mechanical reject over 4.

The threshold of 0.72 is unexplained but tuned-looking. The whole mechanism is ~35 lines and has no
dependencies.

**There is a confirmed bug in it, and it breaks the default persona.** The normalization regex is
`[^a-z0-9 ]+`, which is ASCII-only, so every CJK character is stripped and any pure-Chinese call
normalizes to the empty string. `_is_repetitive_commentary` opens with `if not normalized: return
True`. I extracted the two functions and ran them on sample Chinese, French, and English calls:

```
normalize("梅西中路接球，直塞打穿防线！") -> ''
  zh call 1: rejected=False   (history empty, so the early return fires first)
  zh call 2: rejected=True
  zh call 3: rejected=True
  zh accepted total: 1

normalize("Mbappé accélère sur l'aile droite, quel arrêt !") -> 'mbapp acclre sur laile droite quel arrt'
  fr call 1: rejected=False
  fr call 2: rejected=False
```

The first Chinese call is accepted only because history is still empty. From the second call
onward every line is rejected, and because rejects are not appended, history is frozen at length 1
and the condition never clears. **A Chinese-language session emits exactly one line of commentary
and is then silent for its entire 15-minute lifetime.** The default profile is
`zh-cn-fish-meme`, and two of the four shipped personas are Chinese.

French degrades but survives: accented characters are stripped mid-word (`arrêt` becomes `arrt`,
`accélère` becomes `acclre`), which is consistent across calls, so Jaccard still compares
meaningfully. English is unaffected.

The repo's tests use Chinese text, but only in `test_visual_prompt_can_use_chinese_commentator_profile`,
which checks prompt assembly. `test_repetitive_commentary_detection_blocks_same_action` is
English-only, so nothing covers this path.

### Entity grounding

Better than the survey's summary implies, and worth reading even though it is prompt-only. The
naming procedure is a stated sequence, not a hope:

> "Before writing, inspect visible shirt numbers on the ball carrier, passer, crosser, shooter,
> goalkeeper, and nearest defender. Naming priority: if a shirt number is readable and the team kit
> matches the roster map, use that player's short name instead of a generic role. If the number is
> not readable, fall back to a generic role."

Backed by a hand-authored per-match JSON (`data/matches/argentina-france-2022-final.json`) carrying
full rosters with numbers, short names, positions, per-player notes, and prose jersey descriptions
("white and sky-blue striped Argentina shirts with white shorts").

**This is exactly what the survey's implications section says not to do:** "Resolve entities once,
in a tracking layer, and pass names downstream. Never ask the language model to read a jersey."
worldcupvoice asks the language model to read the jersey, on every call, from a 960 px JPEG. It
gives the model the best possible chance — a roster keyed by number, kit colours, and an explicit
procedure — but there is no detector, no tracker, and no re-identification anywhere in the repo.

The anti-hallucination blocklist is similarly prompt-only but unusually specific: do not say
kick-off, penalty, goal, or equaliser unless the newest frame visibly supports it; do not invent
player names, fouls, sounds, scores, or off-screen events. And a nice touch — the match file
contains `finalScore`, and the prompt explicitly quarantines it: *"Treat the known final score as
private match metadata, not the live score to announce."* The system is replaying a finished 2022
match and has to be told not to spoil it.

### Latency

**No measured or claimed latency figures exist anywhere in the repo.** The README's roadmap still
lists "Lower latency between the video frame and the AI commentary output" as open.

But the code is fully instrumented for it. Every turn emits one structured log line, `AI_AUDIO_PIPELINE`,
carrying `describe_ms`, `transcript_ms`, `tts_ms`, `audio_publish_ms`, `audio_drain_ms`, `total_ms`,
`pcm_bytes`, `sent_bytes`, `pcm_ms`, and the text itself. Anyone running this gets a per-stage
latency breakdown for free. If you want the numbers the survey wanted, run it and read the logs.

Structural floor, reasoning from the constants: a call describes a window ending at the most recent
sampled frame, up to 0.55 s stale, then waits on a non-streaming vision request, then TTS. The
transcript is published **before** audio synthesis starts, deliberately, so text appears while the
clip is still being prepared. Only ElevenLabs has a true streaming path
(`_stream_publish_speech_elevenlabs`, publishing PCM as chunks arrive); OpenAI and Fish Audio
synthesize the whole clip first. So the ElevenLabs path should be materially faster to first audio,
and the config's `eleven_flash_v2_5` with `speed: 1.12` reflects that.

### Session lifecycle and heartbeat

`server/app/session_manager.py` is the cost-control layer, and it is more deliberate than the
README's "explicit session start/stop, viewer heartbeats, hard max runtime" summary suggests. The
whole thing is organized around one idea: **a vision model running on a live stream burns money
whether or not anyone is watching.**

- **Exactly one session exists server-wide.** `start()` clears the session dict and stops every
  existing session with reason `replaced by newer live session` before creating a new one. There
  is no multi-tenancy; a second viewer starting a session evicts the first.
- **Session id** is `{channel_name}:{media_uid}:{unix_ts}`; the agent id is a parallel
  `backend-commentator-{channel}-{ts}`. Both are accepted as lookup keys on every endpoint.
- **A monitor task per session** polls on an interval of `max(0.05, min(5.0, guard/3))` where
  `guard` is the smaller of the two timeouts. With shipped defaults that is **every 5 seconds**.
- **Two independent auto-stop conditions**, checked under the lock:
  - session age ≥ `live_session_max_seconds` (**900 s**, 15 minutes), reason `max session age reached`
  - heartbeat age ≥ `viewer_heartbeat_timeout_seconds` (**45 s**), reason `viewer heartbeat timeout reached`
- **The frontend must actively keep the agent alive.** `POST /api/session-heartbeat` refreshes
  `last_viewer_heartbeat_at`. Close the tab and the commentator is torn down within 45 to 50
  seconds. This is the inverse of the usual design, where an agent runs until told to stop.
- **`ai_spending_state` is a first-class field on the status response**, distinct from `state`,
  with four values: `active` (frames have been sampled, tokens are being spent), `idle_no_video`
  (session up, Agora joined, but no frames have arrived, so nothing is being billed), `stopped`,
  and `missing`. Naming the spend state separately from the run state is a small thing that makes
  a demo safe to leave open.
- **Lifecycle events** accumulate in a `deque(maxlen=40)`, with the last 12 returned by status:
  `session_started`, `viewer_heartbeat`, `auto_stop_requested`, `stop_requested`,
  `session_stopped`, `stop_failed`. Message text is written for humans, e.g. "Backend AI session
  stopped. AI spend is now off."
- **Stopped sessions are retained** in a separate `_records` dict capped at 100, so status still
  resolves after teardown and a client can learn *why* its agent went away.
- **Startup rollback:** if `commentator.start()` raises, the partially-built commentator is wrapped
  in a synthetic `LiveSession` and pushed through the normal stop path before the exception
  propagates, so a failed start cannot leak an Agora connection.

The `StartSessionResponse` labels the pipeline `source_mode: "agora-gateway"` and
`vision_mode: "backend-openai-vision-rtc"`, which is the closest thing in the repo to a written
statement of the architecture.

### What it does not do

- **No perception stage at all.** No detection, no tracking, no homography, no pose, no OCR.
  Pillow and a JPEG encoder are the entire vision stack. The dependency list is 8 packages.
- **No symbolic event layer.** Nothing computes possession, distance, velocity, or shot/pass
  events. This is precisely the layer MARIO argues is mandatory. Frames go straight to the LLM.
- **No score or game state.** The system is forbidden from announcing the score because it has no
  way to know it. No scoreboard OCR, despite the scoreboard being visible in every frame.
- **No turn-taking.** A single `agent_uid` joins the channel. One voice, one persona per session,
  chosen up front. The four profiles are alternatives, not colleagues.
- **No memory or narrative state.** History is a list of at most 8 raw commentary strings used only
  for de-duplication. Nothing tracks that this is the third foul by the same defender.
- **No lookahead.** Strictly causal, describing the newest frame. It is therefore exposed to the
  exact failure mode `soccer-caster` (Section 5) eliminates with an 8-second delay buffer.
- **No evaluation.** No metrics, no benchmark, no user study, no baseline comparison.
- **Does not generalize to an unseen match.** The roster, jersey colours, storyline, and
  identification notes are a hand-written JSON file per match. Point it at an arbitrary live
  stream and the entity grounding degrades to generic roles.

### What to take from it

1. **The repetition gate.** Stopword-filtered Jaccard at 0.72 over the last 4 calls, with rejects
   excluded from history, plus the last 5 injected into the prompt as negative examples. Roughly
   35 lines, no dependencies, and it addresses a failure mode every fixed-cadence commentator has.
   Widen the normalization regex to Unicode word characters before reusing it, or it silences
   every non-Latin script after the first line.
2. **`max_output_tokens: 40`.** A hard cap is a better cadence control than asking politely for one
   sentence, and it bounds the cost of a wrong call.
3. **Audio-drain backpressure as an implicit floor on speech rate.** Not letting the next turn start
   until the previous clip has drained to 250 ms is a cheap way to stop a commentator talking over
   itself, and it generalizes directly to a two-voice booth.
4. **Publish the transcript before the audio.** Perceived latency drops even when real latency does not.
5. **Quarantining known-future metadata in the prompt.** If you replay finished matches for
   development, you will need this exact guard.
6. **The per-stage `AI_AUDIO_PIPELINE` log line.** Copy the shape. It is the difference between
   having latency numbers and having a roadmap item.

And the negative lesson, which is the same one the survey already drew from Vision-Agents: this is
a well-built pipeline with no perception in it, and the grounding it does have is a paragraph of
English asking a general vision model to read shirt numbers off a 960-pixel-wide JPEG.

---

## 2. The three biggest misses

These are not FIFA-specific, but they are the most consequential omissions found, and two of them
are directly about the video-first real-time problem.

### UniSoccer / MatchVision — the successor to MatchTime, by the same author

- **URL:** https://github.com/jyrao/UniSoccer
- **Paper:** CVPR 2025, https://arxiv.org/abs/2412.01820
- **Project page:** https://jyrao.github.io/UniSoccer/
- **Stars:** 232, 21 forks, 10 commits, 84 MB. Created 2024-11-26, last push 2025-09-08.
- **Language:** Python. Topics: computer-vision, multimodal-learning, sports-analytics.

The survey covers `jyrao/MatchTime` (105 stars) and `SoccerAgent` but not this, which is the same
lab's follow-up and is more than twice as popular. It supersedes MatchVoice.

- **MatchVision encoder:** a soccer-specific spatiotemporal visual encoder, pretrained two ways —
  supervised event classification, and *contrastive commentary retrieval*. The contrastive
  commentary objective is the interesting one: it aligns video clips to commentary text without
  requiring generation, which is a cheaper and better-behaved pretraining signal than captioning loss.
- **SoccerReplay-1988:** annotations and video references for **1,988 complete matches** with an
  automated annotation pipeline. This is roughly 4x SoccerNet-Caption's 471 games and is the
  largest multimodal soccer dataset released. On Hugging Face at `Homie0609/SoccerReplay-1988`,
  gated behind a non-commercial NDA form.
- **Checkpoints:** https://huggingface.co/Homie0609/UniSoccer
- **Downstream tasks:** event classification, commentary generation, multi-view foul recognition.
  Reports state of the art on all three.
- **Input modality:** 30-second clips (15s either side of a timestamp), either raw mp4 or
  pre-extracted `.npy` features. Two training paths, and the `.npy` path locks the visual encoder.
- **Runs live:** no. Offline clip-level, same shape as MatchTime.
- **Relevance:** this is the best available soccer-pretrained visual encoder. For a video-first
  system, MatchVision is the encoder to put underneath a symbolic event layer rather than a generic
  CLIP or InternVideo backbone. The contrastive-commentary pretraining is also the cleanest way to
  get a video embedding that is already commentary-aware.

### TGLG and VLM-TSI — real-time narration with a metric for *when*, not just *what*

- **URL:** https://github.com/yukw777/tglg
- **Paper:** "Temporally-Grounded Language Generation: A Benchmark for Real-Time Vision-Language
  Models", https://arxiv.org/abs/2505.11326
- **Stars:** 5, 1 fork, **206 commits**. Created 2025-02-17, last push 2026-05-12. Python.
- **Models on Hugging Face:**
  - `kpyu/soccernet-videollm-online` — VideoLLM-Online 8B fine-tuned on curated SoccerNet
  - `kpyu/soccernet-vlm-tsi` — VLM-TSI fine-tuned on SoccerNet
  - `kpyu/ego4d-goalstep-vlm-tsi` — same model on egocentric data
  - `kpyu/tglg` — the benchmark annotations
- **Star count is misleading.** 206 commits, released models, released annotations, released
  benchmark. This is a serious lab artifact that nobody has found yet.

Why it matters more than its star count suggests:

- **The task definition is exactly the problem the survey identified as unsolved.** TGLG requires
  a model to generate utterances whose *content and timing* both align with streaming video. The
  paper names the two capabilities as **perceptual updating** (revising what you were about to say
  as the picture changes) and **contingency awareness** (knowing that what you say depends on an
  outcome you have not seen yet). That is a precise formulation of the hallucinated-goal failure
  mode measured in the Vision-Agents work.
- **TRACE metric:** jointly scores semantic similarity and temporal alignment. The survey's
  synthesis section says METEOR at 0.259 cannot distinguish good commentary from fluent nonsense
  and proposes LLM-as-judge as the only credible alternative. TRACE is a third option, and it is
  the only one that scores *timing*. For a system where silence is half the product, a
  content-only judge is not enough.
- **VLM-TSI architecture:** interleaves visual and linguistic tokens in a time-synchronized manner,
  explicitly dropping the turn-based assumption. This is the same family as LiveCC and
  VideoLLM-online but built around the timing question rather than around throughput.
- **Sports broadcasting is one of the two evaluation domains** (the other is HoloAssist
  egocentric). The soccer half is built on SoccerNet, including a play-by-play-vs-background
  classifier trained on manually labeled transcripts.
- **Relevance:** read this alongside LiveCC. It is the missing evaluation harness. Also note the
  released `soccernet-videollm-online` checkpoint is a directly usable soccer-narration baseline,
  and the survey has VideoLLM-online listed but not this soccer fine-tune of it.

### soccer-bg-commentary — the only system that models silence, entity grounding, and background retrieval together

- **URL:** https://github.com/zaemon1251-hesty/soccer-bg-commentary
- **Paper:** ACL 2025 System Demonstrations, https://aclanthology.org/2025.acl-demo.38/
  ("Live Football Commentary System Providing Background Information", Mori, Tanaka, Maekawa et al.)
- **Stars:** 0, 0 forks, **77 commits**, 209 MB. Created 2024-07-17, last push 2025-07-30. Python.
- README is in Japanese; the system is bilingual (EN/JA play-by-play).

This has zero stars and is the most architecturally relevant repo in this entire report. It does
three things the survey explicitly lists as "what nobody does well", in one system:

1. **Knowing when not to speak.** It ships a `silence_distribution.csv` derived from real
   broadcasts, and the pipeline's first decision is *whether this is an appropriate moment to
   speak at all* — before deciding what to say.
2. **Utterance-type routing.** Once it decides to speak, it chooses between play-by-play and
   background information. Background goes through RAG over Wikipedia player articles
   (`addinfo_retrieval/`, 3,249 entries). The paper's example: "Fiorentina acquired him for
   7 million euros." This is the periodic/reactive split from MARIO, but with an actual knowledge
   source behind the periodic lane.
3. **Entity grounding done properly.** `players_in_frames_sn_gamestate.csv` is tracklab output
   post-processed to attach *player names and ball coordinates* to frames. Names are resolved in
   the tracking layer and passed downstream, which is precisely the recommendation in the survey's
   implications section — and this is the only system found that actually implements it.
- **Also ships:** labeled commentary data (`commentary/scbi-v2.csv`), action-spotting label CSVs,
  action-and-rates statistics, and a Gradio web interface. Outputs SRT.
- **Runs live:** the demo generates commentary causally over video and renders demo videos; it is
  a research demo, not a streaming service.
- **Related dataset from the same community:** **LFC (Live Football Commentary)**, INLG 2025,
  https://aclanthology.org/2025.inlg-main.13/ — a large-scale dataset for football commentary
  generation models. Worth pulling if you need training text beyond SoccerNet.

---

## 3. EA FC / FIFA video game commentary

This is the angle the user remembered. It exists, but it is small and immature. No repo here has
more than 11 stars. The advantage the user identified — game footage sidesteps rights issues and
has a clean HUD with perfect visibility — is real, and **the HUD OCR shortcut is the recurring
trick across all of these**: you read the score and the ball-carrier's name straight off the
screen instead of solving jersey-number recognition.

### gabe-mc/EAFC-ML-Remaster — the clearest statement of the EA FC thesis

- **URL:** https://github.com/gabe-mc/EAFC-ML-Remaster
- **Stars:** 0, 11 commits, 235 MB. Created 2025-03-10, last push 2025-10-09. Python.
- **Explicit design constraint:** enhance EA FC "without directly interfacing with the game's
  internal data. Instead, all functionality will be derived exclusively from **video footage**."
  That is the same constraint a broadcast system operates under, which makes EA FC a legitimate
  proxy environment rather than a shortcut.
- **Planned pipeline:** YOLO player detection with kickoff-time positional name assignment (plus
  image embeddings for re-identification), a separate high-frequency ball bounding-box model, an
  LLM fine-tuned on real transcripts from Conor McNamara and Jim Beglin, and **Fish Speech** for
  voice synthesis.
- **Also scoped:** Sky Sports / FuboTV-style broadcast graphics overlays.
- **Status:** marked WORK IN PROGRESS, and it reads more like a design document than a system.
  11 commits.
- **Relevance:** the kickoff-time name assignment trick is worth stealing. At kickoff you know the
  formation and therefore roughly who is standing where, so you can seed identities positionally
  and then only maintain them with re-ID rather than solving recognition from scratch.

### AngeloMeridiani/Multimodal-Soccer-Commentary — event-aware prosody, learned rather than hand-tuned

- **URL:** https://github.com/AngeloMeridiani/Multimodal-Soccer-Commentary
- **Stars:** 0, **58 commits**, 515 MB. Created 2026-06-22, last push 2026-07-08. Python.
  README in Italian.
- **Input modality:** FIFA / EA FC gameplay video. Phase 1 is **OCR of the HUD** to read the score
  and the player currently on the ball, producing a structured event log
  `{t, type, player, importance}`.
- **The research contribution is the part worth reading.** Standard TTS reads text at a flat,
  constant tone; real commentary lives in prosody. This project learns the mapping from *event
  importance* to *prosodic parameters* (rate, pitch, energy) **from real broadcast commentary**,
  rather than writing the rules by hand, then applies it to synthesis. Validated with a listener
  study scored by Mean Opinion Score.
- **Explicit ethics stance:** the voice is a controllable generic persona ("dramatic hero voice"),
  deliberately not a clone of a real commentator, citing copyright and personality rights.
- **Relevance:** this is the only project in either survey that treats *how loud and how fast* as
  a learned function of game state. For a multi-agent booth, that is the difference between two
  voices and two personalities. The five-stage relay architecture is otherwise conventional.

### noahbass/highlight-bot — the oldest and most production-shaped of the game-footage projects

- **URL:** https://github.com/noahbass/highlight-bot
- **Stars:** 11, 4 forks, 1.4 MB. Created 2020-07-10, last push 2022-12-08. Python + Kotlin.
  Topics: ai, fifa20, kafka, kotlin, python, tesseract-ocr, twitch.
- **What it does:** monitors a **live Twitch stream** of FIFA 20 FUT gameplay through the Twitch
  API, detects goals via OCR on the scoreboard region, clips the ~20 seconds of build-up, and
  broadcasts the clip to pluggable downstream "hooks" over Kafka.
- **Architecture:** independent microservices arranged as a streaming data pipeline, Kafka as the
  real-time broker, fully stateless and in-memory apart from the final clip write.
- **Runs live:** yes, on a live Twitch stream. It is the only game-footage project here that
  actually consumes a live stream rather than a saved file.
- **Relevance:** dated (2022, FIFA 20) but the *transport* design is the right one — OCR the HUD
  for ground truth, put events on a durable bus, and let any number of consumers subscribe. That
  bus is exactly where a commentary agent would sit.

### Others in this category

| Repo | Stars | Created / pushed | Notes |
|---|---|---|---|
| [asif256000/realistic_football_commentary_generator](https://github.com/asif256000/realistic_football_commentary_generator) | 0 | 2024-02 / 2024-05 | Virginia Tech Spring 2024 Creative AI capstone. Generates commentary from FIFA gameplay video captures. Has a recorded demo video in the README. Summary-plus-static-images output, not live; the TODO list asks for faster audio and event-specific imagery. |
| [anoophundal/ai-tactical-analysis](https://github.com/anoophundal/ai-tactical-analysis) | 0 | 2026-07 / 2026-08 | 7 commits, 28 KB. YOLOv8 + ByteTrack + KMeans jersey-colour team split + homography radar, explicitly supporting **both real broadcast and FIFA / EA FC capture**. Notably documents that FIFA capture needs its own fine-tune because jersey colours and camera angles differ substantially from broadcast. That warning is the useful content. |
| [bhatticoder/Automated-Fifa-Commentary](https://github.com/bhatticoder/Automated-Fifa-Commentary) | 0 | 2026-07 / 2026-08 | 16 MB, heavily badged README (YOLOv8 + NLP + TTS, "real-time"). Conventional cascade, no evaluation, no metrics. |
| [sultann-ai/Fifa-commentary-generation](https://github.com/sultann-ai/Fifa-commentary-generation) | 0 | 2026-02 | 22 KB, empty README. Skip. |
| [Omar59687/football-ai-commentator](https://github.com/Omar59687/football-ai-commentator) | 0 | 2026-07 / 2026-08 | 90 MB, empty README. Unverifiable. |
| [abdulgafarabdurrasheed/football-tournament-manager](https://github.com/abdulgafarabdurrasheed/football-tournament-manager) | 0 | 2026-01 / 2026-02 | eFootball and EA FC community tournament tracker with an "AI pundit that roasts". Community tooling, not video. |

**Mods and tooling adjacent to EA FC**, if you ever need to inject audio back into the game rather
than overlay it: [FC-Nexus-Mod-Hub](https://github.com/Openoolead99/FC-Nexus-Mod-Hub) (57),
[FC-Mod-Symphony](https://github.com/BodyLieutenantLock/FC-Mod-Symphony) (45),
[AttoConjurer/FC-FIFA-Mod-Manager](https://github.com/AttoConjurer/FC-FIFA-Mod-Manager) (44),
[Undeferential-andrew98/FIFA-Audio-Editor-Tool](https://github.com/Undeferential-andrew98/FIFA-Audio-Editor-Tool)
(imports and exports commentary, chants, and sound effects for EA Sports FC and FIFA).
[TheNaeem/FifaSharp](https://github.com/TheNaeem/FifaSharp) (23) wraps the EA FC web app API.
None of these are AI; they are the path to shipping a mod rather than a screen overlay.

**Context on EA itself:** EA confirmed in early 2026 that EA Sports FC 26 uses AI to generate some
commentator voiceover, including replicating player names in Guy Mowbray's voice with his
permission, while keeping human-recorded main commentary. There is a well-upvoted EA Forums thread
requesting a full real-time AI commentator. So the demand signal is public, and the incumbent has
already crossed the synthetic-voice line for names.

---

## 4. The 2026 World Cup wave

The tournament ran June 11 to July 19, 2026, and produced a very large volume of repos. Almost all
are prediction, fixtures, or fan-companion apps rather than commentary, and almost all are
single-weekend hackathon output. Filtered to what has substance or a transferable idea.

### wzk1015/WorldCupArena — the best-engineered World Cup repo found

- **URL:** https://github.com/wzk1015/WorldCupArena
- **Tech report:** https://arxiv.org/pdf/2607.18084
- **Stars:** 24, 3 forks, **509 commits**, 38 MB. Created 2026-04-17, last push 2026-08-09. Python.
- Benchmarks LLMs and deep-research agents on real football prediction across five layers, each
  separately scored: core result (Brier, RPS), player level (Jaccard, F1+nDCG), event level
  (Hungarian-matched MAE, event-F1), tactics and stats (sMAPE), tournament macro (Kendall tau,
  bracket score). Composite 0-100.
- Three leaderboards, and the third is the clever one: **Research Uplift**, the score gain from
  giving a model tools and letting it search versus injecting a fixed context pack. It isolates
  how much retrieval is actually worth.
- Also scores **Above-Market**: composite gain versus Pinnacle closing odds.
- **Relevance:** not commentary. But if you ever want a color-analyst agent to make a prediction on
  air, this is the harness that tells you whether its predictions are worth anything, and the
  Hungarian-matched event-timing metric is reusable for scoring event detection.

### xiaoyangyyy/Generative-Football-Society

- **URL:** https://github.com/xiaoyangyyy/Generative-Football-Society
- **Stars:** 5, 179 MB. Created 2026-05-09, last push 2026-09-09. Python. Frozen baseline v6.0.0.
- A World Cup-scale multi-agent social simulation: team status and historical memory, agent
  psychology, beliefs, media pressure, tactics, macro xG dynamics, optional 6-second micro match
  physics, optional LLM cognition over an OpenAI-compatible API, optional latent world model
  planning. Ships sealed acceptance metrics and artifact fingerprints per release.
- **Relevance:** the *memory and narrative state* layer is the interesting part. The survey notes
  that nothing tracks match-level narrative (third foul by the same defender, ten minutes pinned
  in their own half). This project's historical-memory and media-pressure fields are an attempt at
  exactly that, on a simulator where it is cheap to test.

### gangtao/AgentPitch

- **URL:** https://github.com/gangtao/AgentPitch
- **Stars:** 21, 2 forks, **261 commits**, 211 MB. Created 2026-04-19, last push 2026-08-01.
  Topics: agentic-ai, llm, soccer-analytics. CI, releases, Docker image.
- Every player on the field is an AI agent running a `decide(game_state, player_state, history)`
  callback in Python, JavaScript, or Rust inside a sandbox. An LLM code-generation pipeline writes
  each player's strategy before the match; a post-match evolution pipeline rewrites it based on
  performance. Arena, cup, and league modes with a live match view and event feed.
- **Relevance:** not a commentator, but a well-built free source of synthetic matches with a
  fully known ground-truth event stream. That is a much better test harness for a commentary
  system than real footage, because you can check the commentary against the actual game state.
  This is the same role Pong plays for xPong, but for soccer.

### The TxODDS World Cup Hackathon cluster

A cohort of repos built on a live event feed called TxLINE, all created late June to mid July 2026.
Uniformly small and unmaintained, but the cluster tells you what a room full of people converged on
when handed a live World Cup feed. All have 0 stars.

| Repo | Idea |
|---|---|
| [Tuborrr-Dev/pitchline](https://github.com/Tuborrr-Dev/pitchline) | Real-time AI football commentary rendered "in the language of financial markets". Next.js + .NET 9 + FastAPI + Gemini 2.5 Flash + SSE. Very long README. |
| [AditiChaudharyy14/worldcup-pundit](https://github.com/AditiChaudharyy14/worldcup-pundit) | Real-time TxLINE data, event detection, multilingual commentary (English, Nepali, Hindi). |
| [smohamedjavid/the-pit](https://github.com/smohamedjavid/the-pit) | Rival AI pundits make picks sealed on-chain before kickoff, graded against the feed, presented like a fight-night broadcast. The *adversarial two-pundit* framing is the transferable bit. |
| [Bsh54/horus](https://github.com/Bsh54/horus) | Telegram pundit bot with live notifications, market context, TTS voice notes, multilingual. |

### Other World Cup repos worth a line

- [jordanlyall/wc26-mcp](https://github.com/jordanlyall/wc26-mcp) — 34 stars, 9 forks, on npm.
  18-tool MCP server covering matches, teams, venues, head-to-head, injuries, odds, standings,
  bracket. **All data ships with the package**, so zero API keys and zero external calls. Works
  with Claude Desktop, Claude Code, Cursor, and Telegram. The most polished soccer MCP server found.
- [arturogarrido/claudinho](https://github.com/arturogarrido/claudinho) — 28 stars. World Cup live
  scores as an MCP server, CLI, and **Claude Code statusline**. No API keys.
- [lacausecrypto/mcp-sports-hub](https://github.com/lacausecrypto/mcp-sports-hub) — 22 stars,
  9 forks. 41 sports API providers, 396 tools, 70+ sports in one process; 19 work with no API key.
  If you need a live data plane behind a color-analyst agent, start here.
- [Azzaraell/consumer-and-fan-experiences](https://github.com/Azzaraell/consumer-and-fan-experiences)
  ("PitchPulse") — 0 stars but 125 tests and a clean design: **one SSE stream feeds both a
  Claude-narrated match timeline and a self-resolving prediction game**, so the two surfaces cannot
  drift out of sync. Claude Haiku 4.5. Demo replays a canned match with no API keys needed. The
  single-stream-two-surfaces discipline is the reusable idea.
- [upstash/agents-worldcup](https://github.com/upstash/agents-worldcup) — 11 stars. Claude, Gemini,
  and OpenAI agents competing to predict match outcomes.
- [26worldcup/26worldcup.github.io](https://github.com/26worldcup/26worldcup.github.io) — 67 stars.
  Companion PWA in 23 languages with win probabilities.
- [baekkyoungjung/worldcup-live-cli](https://github.com/baekkyoungjung/worldcup-live-cli) — live
  World Cup commentary streamed into a Claude Code session as logger output. A joke, but a
  nicely-shaped one.
- [AhmedHazem02/fifa-world-cup-2026-prediction-agent](https://github.com/AhmedHazem02/fifa-world-cup-2026-prediction-agent)
  — 131 stars, the most-starred World Cup AI repo. Pure prediction (Elo + Poisson + form ensemble),
  no commentary.

---

## 5. Recent soccer video-to-commentary repos not in the surveys

### davidtkunz/soccer-caster — the strongest independent engineering find in this report

- **URL:** https://github.com/davidtkunz/soccer-caster
- **Stars:** 0, 9 commits, 101 KB. Created and pushed 2026-08-16. Python.
- Built on `roboflow/supervision` and `roboflow/sports`.

**The core idea is the single best answer found to the survey's central unsolved problem.** Quoting
the README's framing: a live AI commentator has to talk about a play whose outcome it does not yet
know, and the worst failure mode is confidently announcing a goal that did not happen. So:

> Don't run live. Hold a rolling **8-second buffer** of frames and derived state. Perception runs
> at the live edge; narration runs eight seconds behind it. By the time the caster describes a shot
> being struck, the system has already seen whether it went in. Broadcasts already run on delay.
> Nobody notices the eight seconds, and the whole class of hallucinated-event bugs disappears.

This directly dissolves the Vision-Agents result (commentary wrong more than half the time) and the
TGLG "contingency awareness" problem, by trading latency nobody perceives for correctness everyone
does. It is also the reason the sub-0.4s time-to-first-audio benchmark may be optimizing the wrong
variable.

- **Implemented, per the README status table:** shot/replay segmentation with cut detection and
  scoreboard localisation; game state with **possession hysteresis** and temporal smoothing;
  event detection for passes, turnovers, shots, goals, saves, restarts; per-frame homography with
  rejection and stale-transform fallback; post-game summary with possession, player stats,
  heatmaps, timeline, recap; the **delay buffer** (ring buffer, narration cursor, lookahead); and a
  **Director** with an importance gate, silence handling, and barge-in.
- **Not implemented:** narration and voice are thin skeleton wrappers over external APIs, and the
  eval set is blocked on needing real footage.
- **187 tests passing** against synthetic data, including an integration test that runs the stack
  end to end. For a 9-commit repo, that is unusual discipline.
- Detection models are injected, so it expects you to bring your own weights.
- **Relevance:** read this repo. The delay buffer, the Director's importance gate plus barge-in,
  and possession hysteresis are all directly transferable, and the design rationale is written down
  clearly enough to argue with.

### shubhamgoel27/soccer-co-commentator ("AURA FC")

- **URL:** https://github.com/shubhamgoel27/soccer-co-commentator
- **Build log:** https://shubham.gg/blog/building-aura-fc
- **Stars:** 0, 1 commit, 288 KB. 2026-06-11. Python.
- Three layers: YOLOv8 detection and tracking **with SAHI slicing** (reported to roughly double
  ball recall on broadcast footage), then a state machine naming passes, turnovers, sprints and
  shots, then an LLM that only sees high-signal events.
- **Two measured details worth keeping:**
  - **Camera-motion compensation by subtracting median player displacement**, so a broadcast pan
    does not turn every player into a sprinter. Cheap and obviously correct once stated.
  - **Restraint is tuned deliberately: roughly one callout every 4 to 5 seconds.** More reads as
    spam. That is a concrete number for the silence problem, arrived at empirically.
- Known weakness: vertical clips fail because the ball spends most of its time cropped out of frame.

### OpenSportsLab/opensportslib — the new SoccerNet-community library

- **URL:** https://github.com/OpenSportsLab/opensportslib
- **Docs:** https://opensportslab.github.io/opensportslib/ · PyPI: `opensportslib`
- **HF org:** https://huggingface.co/OpenSportsLab
- **Stars:** 18, 5 forks, **494 commits**. Created 2025-11-03, **last push 2026-09-10** (today).
  Python 3.12+, CUDA 12.6/12.8/13.0.
- Unified train/eval/inference framework for **action classification, action spotting/localization,
  VQA, action retrieval, and action description/captioning** — config-driven, with an OSL JSON
  data format. VQA backends for both X-VARS and Qwen (`Qwen2.5-7B-Instruct`, `Qwen3.5-9B-Base`).
  Optional SpoTTA test-time adaptation for E2ESpot inference.
- **Relevance:** this is the actively-maintained successor to the pile of one-off SoccerNet
  challenge repos in `research-cv.md`. If you need spotting and captioning under one API with
  reproducible configs, this replaces four repos. Its recency (pushed today, 494 commits) makes it
  the safest CV dependency in either survey.

### Penny-03/personalised-sports-commentary — screen capture plus live API ground truth

- **URL:** https://github.com/Penny-03/personalised-sports-commentary
- **Stars:** 0, 38 commits. Created 2026-05-15, last push 2026-05-30.
- Captures frames from a live broadcast **by screen capture**, sends each to Gemini 2.5 Flash for
  on-pitch action and visual context, pulls scoreline, minute, teams, goal events and status from
  API-Football, then **fuses vision output with API output into a single structured game-state
  object** before generation. Generates a personalised audio track per user persona.
- **Relevance:** the fusion step is the point. Vision supplies *what it looks like*; the API
  supplies *what is definitionally true* (score, minute, who scored). Letting the API override the
  model on facts it can never get right from pixels is cheaper and more reliable than any amount of
  prompt engineering. Screen capture also means it works on any stream you can legally watch.

### WhiteInfinite/AI-Based-Football-Match-Event-Recognition — multimodal event fusion

- **URL:** https://github.com/WhiteInfinite/AI-Based-Football-Match-Event-Recognition
- **Stars:** 1, 25 KB. 2026-07-18. Jupyter Notebook.
- Fuses four signals for event detection: **Whisper on the existing commentary audio**, CLIP visual
  similarity, EasyOCR on the scoreboard, and Gemini acting as a "VAR referee" vision-language
  verifier, combined by confidence fusion.
- **Relevance:** the insight is that a broadcast already contains a human commentator telling you
  what happened, and ASR over that track is a nearly free high-precision event detector. For a
  system that intends to *replace* the commentator this is only useful for building ground truth,
  but for that purpose it is very cheap. Note the survey already has `SoccerNet/sn-echoes` for ASR
  data; this is the live version of the same idea, plus OCR cross-checking.

### hippograndet/MatchCaster — two-voice booth over StatsBomb replay

- **URL:** https://github.com/hippograndet/MatchCaster
- **Stars:** 0, 24 commits, 2.9 MB. Created 2026-04-12, last push 2026-06-23.
  Topics: fastapi, football, llm, statsbomb, tts.
- Replays real StatsBomb open-data matches second by second, runs momentum/xG/pattern analysis,
  and a **tick-driven Director** orchestrates **two LLM voices: play-by-play plus analyst**, with
  TTS and an interactive pitch visualizer. Prepares commentary slightly ahead of time. Groq by
  default, Ollama for fully local.
- **Relevance:** the survey says "no soccer system has two voices" and points to xPong (Pong) and
  ESOPN (screencasts) as the only turn-taking examples. **This is a soccer system with two voices.**
  It is event-data driven rather than video-first, and it is small, but it closes that specific gap
  and the Director abstraction matches soccer-caster's independently.

### Others, briefly

| Repo | Stars / commits | Notes |
|---|---|---|
| [martinjolif/football-game-tracking](https://github.com/martinjolif/football-game-tracking) | 2 / **215** | Detection and tracking of ball and players, pitch keypoints, homography to pitch coordinates, then LLM commentary from the derived insights. MLflow experiment tracking, Docker image, reproducible training instructions. Well-run for a solo project; created 2025-10, last push 2026-03. |
| [allanchan339/VLM_Soccer_Commentator_THG](https://github.com/allanchan339/VLM_Soccer_Commentator_THG) | 0 / — | **Cantonese** soccer commentary with a synchronized **talking-head video** (lip-sync), EdgeTTS and SoVITS variants, multiple demo videos in the README. The only non-English, avatar-based commentary system found. Pushed 2025-12. |
| [Ayuyo/transfusion-soccer](https://github.com/Ayuyo/transfusion-soccer) | 0 / — | Soccer broadcast frame to commentary using a unified vision-language transformer (Transfusion, Meta AI 2024). 35 KB, empty README, 2026-06. Idea only. |
| [WWandP/SoccerEye](https://github.com/WWandP/SoccerEye) | 14 / — | Monocular football video system: detection, player grouping, bird's-eye trajectory visualization, real-time speed display, advertising maps. Pretrained models and test video provided. Created 2024-03, last push 2025-03. A cleaner starting point than several repos in `research-cv.md`. |
| [mradovic38/football-analysis](https://github.com/mradovic38/football-analysis) | **177** / — | Popular CV analysis pipeline, 397 MB, created 2024-07, last push 2025-01. Comparable to `abdullahtarek/football_analysis` which the CV survey already covers. Included for completeness since it is more starred than most of that survey. |
| [roboflow/trackers](https://github.com/roboflow/trackers) | **3,762** / — | Not commentary. Clean Apache-2.0 re-implementations of SORT, ByteTrack, OC-SORT, BoT-SORT with TrackEval, benchmarked on SoccerNet and SportsMOT. Created 2025-04, pushed today. The CV survey covers `roboflow/supervision`, `sports`, and `rf-detr` but not this, and this is the tracking piece of that stack. |
| [kacossio/soccernet-gsr](https://github.com/kacossio/soccernet-gsr), [Do-sensei/sn-nvs-2026](https://github.com/Do-sensei/sn-nvs-2026), [cvail-research/soccernet-nvs-2026](https://github.com/cvail-research/soccernet-nvs-2026) | 0 | SoccerNet 2026 challenge entries. `sn-nvs-2026` took 2nd in Novel View Synthesis with antialiased 3D Gaussian Splatting at 28.94 dB PSNR. Confirms SoccerNet 2026 added an NVS track. Marginal for commentary. |
| [letmebesai/football-multimodal-rag-analytics](https://github.com/letmebesai/football-multimodal-rag-analytics) | 0 | Multimodal RAG over spatial match data plus broadcast video, pgvector on Neon, FastAPI, Streamlit. Tactical analysis, not narration. 2026-08. |

---

## 6. Cross-sport commentary systems worth reading

Not soccer, but each solves a piece of the problem and each is new.

- **[Kun-AzureLotus/LOL-AI-Commentary](https://github.com/Kun-AzureLotus/LOL-AI-Commentary)** —
  1 star, Rust, Windows desktop, 2026-08. Reads the Riot Live Client API plus optional OBS capture.
  The pipeline order is stated as: **decide *whether* to speak, then generate**, and the README is
  explicit that it stays silent when an event is not worth mentioning. Same architecture as the
  ACL 2025 soccer system, arrived at independently, which is a decent signal that
  speak/don't-speak belongs as its own stage rather than as a prompt instruction. Also draws a
  clear product boundary: commentator, not coach, not cheat, comments only on legally visible
  information.
- **[HullyMully/cs2-ai-commentator](https://github.com/HullyMully/cs2-ai-commentator)** — 2 stars,
  Python, FastAPI, 2026-06 to 2026-08. Counter-Strike 2 commentary from computer vision plus event
  detection plus LLM plus TTS. Explicitly a prototype, shipped without weights or datasets.
- **[LyraV / "Don't Pause"](https://arxiv.org/abs/2606.06991)** — arXiv 2026-06, Yang, Zhang, Qian,
  Dong, Xu. Not sport-specific, and worth flagging for the realtime survey rather than this one.
  Two components: a **Frame-Driven Transition Controller**, a training-free verification-based
  finite state machine that decides *when* to respond, and a **Streaming Token Pacer** that adapts
  generation speed to the pace of the visual content. Per-frame incremental sub-budget decoding so
  perception is never interrupted. Reports **98.29% synchrony with video playback at 3.89 FPS**.
  The pacing idea — talk faster when the game speeds up — is not present in any commentary system
  in either survey.
- **[ThiagoMonica/NBA-Event2Text](https://github.com/ThiagoMonica/NBA-Event2Text)**,
  [whodeanie/live-game-state-tracker-ts](https://github.com/whodeanie/live-game-state-tracker-ts)
  (NBA game state over WebSocket with per-play Claude commentary),
  [prakashiOrbit/f1-apex-race-control](https://github.com/prakashiOrbit/f1-apex-race-control)
  (F1 commentary suggestions with OBS overlays) — all small, all 2026, all event-data driven.
- **Fine-tuned commentary LoRAs on Hugging Face**, if you need a cheap styled generator:
  `jsantillana/*-f1-commentary-lora` (Mistral-7B, Llama-3.2-1B, Phi-4-mini, Qwen2.5-3B — a clean
  cross-model comparison on one commentary task), `kattymandy/cricket-commentary-qwen2.5-1.5b-lora`,
  `may-ur08/qwen2.5-vl-cricket-commentary` (vision-language), `Mook21/gpt2-medium-football-commentary`.

---

## 7. What this changes

**Five things the surveys concluded should be revised.**

0. **worldcupvoice is not a realtime-API wrapper.** The survey's taxonomy files it under "sampled
   frames thrown at a hosted realtime API." Its source calls the ordinary OpenAI Responses endpoint
   with four JPEGs on a fixed 4-second timer, using `gpt-5.4-mini` and a 40-token output cap. It is
   a polling loop, not a realtime session, and that is why it does not hit the "realtime models
   need audio or text to trigger a response" constraint the survey records from Vision-Agents.
   Full read in Section 1.

1. **"No soccer system has two voices" is no longer true.** `hippograndet/MatchCaster` runs a
   play-by-play voice and an analyst voice under a tick-driven Director over StatsBomb replay. It
   is small, but it exists, and it is soccer.

2. **"Real-time and accurate simultaneously is unsolved" understates the situation — the framing
   may be wrong.** `davidtkunz/soccer-caster` argues that you should not try. Run perception at the
   live edge and narration eight seconds behind, since broadcasts are already delayed and viewers
   cannot tell. That converts an unsolved accuracy problem into a latency budget you can spend.
   TGLG names the same issue formally as *contingency awareness*.

3. **Evaluation has a third option.** The survey offers METEOR (bad) or LLM-as-judge (better).
   **TRACE**, from the TGLG benchmark, scores semantic similarity and temporal alignment jointly.
   Since timing is half of what a commentary system does, and LLM-as-judge over free-form text does
   not see timing at all, TRACE is the metric that matches the product.

4. **Entity grounding has been done.** The survey says GOAL and SoccerWiki are the only serious
   attempts and neither is wired into a live commentator.
   `zaemon1251-hesty/soccer-bg-commentary` wires tracklab tracking output to player names and ball
   coordinates per frame, then retrieves Wikipedia background about those named players at
   generation time. It is an ACL 2025 demo with 77 commits and zero stars.

**Read these four, in this order:**

1. https://github.com/davidtkunz/soccer-caster — the delay buffer, the Director's importance gate
   and barge-in, possession hysteresis. Short, tested, and the reasoning is written down.
2. https://github.com/yukw777/tglg + https://arxiv.org/abs/2505.11326 — the TRACE metric, the
   perceptual-updating / contingency-awareness framing, and a released SoccerNet-narration
   checkpoint to benchmark against.
3. https://github.com/zaemon1251-hesty/soccer-bg-commentary +
   https://aclanthology.org/2025.acl-demo.38/ — speak/don't-speak as a first-class stage, utterance
   type routing, and name resolution in the tracking layer.
4. https://github.com/jyrao/UniSoccer — MatchVision as the soccer-pretrained encoder, and
   SoccerReplay-1988 (1,988 matches) as training data if the NDA is acceptable.

**Then, for the EA FC path specifically:** the thesis is sound and the field is nearly empty. Six
repos, all under 11 stars, none mature. The recurring technique across all of them is **OCR the
HUD** for score and ball-carrier identity, which removes the jersey-number problem that blocks
every broadcast system. `gabe-mc/EAFC-ML-Remaster` states the video-only constraint most clearly,
`AngeloMeridiani/Multimodal-Soccer-Commentary` has the only learned event-to-prosody mapping found
anywhere, and `noahbass/highlight-bot` is the only one that has actually run against a live stream.
If the goal is a demo with clean perception and no rights exposure, this is an open lane.
