# Agent frameworks and related commentary systems

Researched November 6, 2026. This is a design recommendation, not an integration
report. Sources below are official documentation, author repositories, and papers.
External systems were inspected through their documentation and selected paper
sections; their code, benchmark results, and deployment claims have not been
independently reproduced. No new dependencies or services were installed.

## Recommended stack

Keep **LangGraph + asyncio + Pydantic** as the core. Add **LangSmith** as an optional
trace and experiment interface, and try **standalone GEPA** for offline prompt
optimization after the evaluation harness exists. Retain local JSON traces and
scorecards so replay and regression checks work without a hosted account.

Consider **LiveKit** when we connect a live source and stream audio to viewers.
Start replay with our own clock and speech coordinator. Keep media transport behind
an adapter so it does not own our evidence model or agent-routing policy.

The important engineering work remains ours: event-time visibility, grounded player
identification, per-role tool permissions, stale-result rejection, bounded revisions,
speech preemption, and reproducible evaluation. Framework adoption does not prove
these properties.

## Framework comparison

| Candidate | Verified capabilities | Recommendation for this project |
| --- | --- | --- |
| [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) | Shared-state graphs mixing deterministic code and model steps, streaming, persistence, and explicit control flow. Already a direct dependency here. | Keep as the orchestrator. Use bounded agent turns and conditional handoffs; use asyncio for the continuously advancing replay clock and cancellable speech. Checkpointing must never cause already-played speech to be emitted again. |
| [Pydantic AI](https://ai.pydantic.dev/multi-agent-applications/) | Typed agents, tool-based delegation, programmatic handoffs, graph patterns, shared usage limits, and OpenTelemetry instrumentation. | Strong alternative for a fresh Python implementation. Our existing Pydantic usage does not mean we already use Pydantic AI. No demonstrated need to introduce a second agent runtime or migrate now. |
| [CrewAI Flows](https://docs.crewai.com/en/concepts/flows) | Event-driven flow methods, conditional routing, structured state, persistence, and agents inside flows. | Viable alternative, especially for research/report workflows. Our preference for explicit deadline and evidence handling favors retaining the graph we already have; this is a project-fit judgment, not a latency benchmark. |
| [Microsoft Agent Framework](https://github.com/microsoft/agent-framework) | Multi-provider agents, graph workflows, handoffs, checkpointing, streaming, and OpenTelemetry support. | Viable alternative if Microsoft/.NET integration becomes a requirement. Migration currently adds work without resolving a measured gap. The [AutoGen repository](https://github.com/microsoft/autogen) now states it is archived and directs new users to this successor; do not start on AutoGen based on older tutorials. |
| [LiveKit Agents](https://docs.livekit.io/agents/logic/sessions/) | Audio/video session orchestration, tools, agent handoffs, video sampling, and controllable speech playback. | Leading candidate for the later live media adapter. Its conversational defaults need adjustment for a match-driven broadcast; its default Python video sampler varies with user speech, which is not our sampling signal. |
| [Pipecat](https://docs.pipecat.ai/pipecat/get-started/introduction) | Python pipelines for audio/text/video frames, provider and transport integrations, and worker coordination. | Alternative to LiveKit for the media layer. Test both only when transport needs are concrete; do not run two media frameworks in the first version. |
| [GEPA](https://github.com/gepa-ai/gepa) / [DSPy GEPA](https://dspy.ai/api/optimizers/GEPA/) | Feedback-guided search over textual candidates; standalone custom-system integration and DSPy pipeline optimization. | Prefer a small standalone adapter around our versioned role prompts and evaluator. Converting the runtime to DSPy is not required. Keep scheduling code and evidence rules outside the optimization surface initially. |
| [LangSmith](https://docs.langchain.com/langsmith/evaluation) | Traces, reference datasets, code/human/model evaluators, pairwise experiments, and regression comparisons. | First optional observability/evaluation interface because we already use LangGraph. Instrument our custom SDK calls, tools, and speech events explicitly; graph traces alone do not measure event-to-audio delay. |
| [Arize Phoenix](https://arize.com/docs/phoenix) | OpenTelemetry/OpenInference traces, datasets, evaluations, and experiment comparisons. | Good alternative if we prioritize running the observability backend ourselves. Pick one interface; keep exported results portable. |

### Live speech constraints

LiveKit supports [predefined and pre-synthesized speech playback](https://docs.livekit.io/agents/multimodality/audio/).
That lets our caller/analyst agents own the text while the media layer handles
playback. A human-interruption detector is not a goal-preemption policy: our
coordinator must decide when a goal overrides the analyst and reject stale speech.

The [ElevenLabs plugin](https://docs.livekit.io/agents/models/tts/plugins/elevenlabs/)
still supports a direct ElevenLabs account; the docs state that ElevenLabs was
retired from LiveKit Inference on August 31, 2026. Keep our existing account and
custom voice IDs. Verify that any adapter preserves saved voice settings and does
not inject new defaults. Measure quality using full phrases before changing
streaming chunk sizes or TTS models.

No framework reviewed here establishes a hard real-time guarantee for our system.
Compare event-to-speech latency, quality, interruption behavior, and cost using the
same replay inputs rather than adopting vendor latency numbers as our results.

## Related research and implementations

### StreamSoccer: event memory for causal streaming

[Paper and method](https://arxiv.org/html/2608.19723v1)

Maintains bounded active event memory, retains completed events locally, and builds
retrievable historical records. A scheduler selects current-event, recent-window,
historical-memory commentary, or silence. Its generator is unified; this is not
evidence of independent caller and analyst agents. The learned visual memory
architecture is also not a drop-in library for our application.

Borrow the distinction between an evolving event, recent completed events, and
earlier match history. Implement a simpler structured event ledger first, with
timestamps, evidence IDs, completion state, and bounded recent context. Retrieve
only records available by the replay cutoff. Evaluate current action, recent trends,
historical recall, and silence separately. Cite the inspiration without claiming
equivalence to their trained memory encoder or published performance.

### MatchTime / MatchVoice: alignment before learning

[Project](https://haoningwu3639.github.io/MatchTime/) ·
[Code](https://github.com/jyrao/MatchTime/) ·
[Paper](https://arxiv.org/abs/2406.18530)

Studies video/commentary misalignment, provides a manually aligned test benchmark,
and uses an ASR plus temporal-alignment pipeline to curate training data. MatchVoice
is a learned commentary model, not an agent orchestration framework.

Use it to guide reference-data preparation and investigate its aligned data as a
benchmark candidate. Its offline video windows are not automatically valid for
causal replay: recheck every input window against the emission cutoff. A poor
timestamp target should not become the reward for an otherwise timely call.

### GameSight: player grounding and knowledge refinement

[Project](https://gamesight2025.github.io/gamesight2025) ·
[Paper and training setup](https://arxiv.org/html/2604.00057v1)

Separates entity alignment using visual/contextual cues from enrichment using
historical statistics and changing match state. Reports SFT and GRPO for the
player-alignment task with a binary reward, followed by training-free knowledge
refinement. This directly supports investigating a narrowly grounded RL task rather
than optimizing vague prose similarity first.

Borrow separate identity checks and cited, time-constrained background retrieval.
For a later training experiment, compare player/team identification accuracy before
and after SFT, then GRPO, using a model we can train. Include abstention for unclear
images. The paper's segment-based setup and use of surrounding shots must be
adapted to causal replay; subsequent close-ups cannot justify an earlier live call.
Its reported gains do not establish performance for our system.

### SoccerNet-Echoes: real broadcast transcripts

[Paper](https://arxiv.org/abs/2405.07354)

Adds automatically generated broadcast commentary transcripts using Whisper and
translation. This is more directly relevant to spoken commentary references than
live text feeds alone. ASR names, translation, and timing still need review; do not
treat automatic transcripts as verified event truth. Inspect access terms, language,
alignment quality, and match coverage before choosing a data subset.

### AI-Powered Football Commentary Platform: a practical video pipeline

[Author repository](https://github.com/Mohamed3alaa/AI-powered-football-commentary)

The README describes action spotting, clip extraction, tracking, visual description,
fine-tuned commentary, and English/Arabic TTS with a React/FastAPI interface. Useful
as a reference for staged outputs and training-data preparation. It describes a
sequential pipeline; the README does not establish autonomous agent coordination
or a causal streaming system. Treat training numbers and quality scores as author
reports until reproduced. We did not inspect its full implementation or run it.

### Ably live commentary: event-driven UI and two personalities

[Author repository](https://github.com/ably-labs/football-data-live-ai-commentary)

The README describes a simulated match with user-triggered events, preloaded stats,
two commentator personalities, streamed responses, and history/state synchronization.
Borrow visible progress and reconnect behavior. Its structured simulated events
bypass the hard problem of grounding a real video feed; the described four-second
event batching also needs evaluation before use for urgent calls. Marketing scale
and latency claims are not evidence of our throughput or end-to-end performance.

### Early GPT-2 football commentator: simulation-based reference

[Author repository](https://github.com/ChintanTrivedi/football_ai_commentary)

Trains GPT-2 and runs commentary over Google's Football Environment with pyttsx3.
Useful historical context for event-to-language generation, but simulator access
does not solve broadcast-video perception. Do not adopt its old environment as our
runtime foundation.

## Consequences for our build

1. Build the previously agreed replay core: pre-match researcher, observer, caller,
   analyst, deterministic coordinator, and review outside the urgent speech path.
2. Keep separate pre-match knowledge, evolving visual evidence, and completed-event
   memory. Tools expose bounded, timestamp-filtered views rather than entire histories.
3. Add local role/tool/speech traces and evaluation first, then optionally mirror
   experiments into LangSmith. Observe both intermediate decisions and the final audio.
4. Curate aligned reference commentary using the lessons and candidate data above.
   Keep original audio and reference transcripts out of the runtime agent inputs.
5. Try bounded GEPA prompt search against a frozen baseline and validation split.
   Consider grounded player identification as the first later SFT/RL experiment.
6. Add one media framework only when testing a live source and viewer playback.
   Preserve the current curated demo until generated output earns replacement.

The interview story should be a measured, inspectable system: show which agent
used which evidence/tool, how deadlines changed routing, where it abstained, and
whether a versioned change improved held-out quality at an acceptable cost. These
are intended deliverables, not capabilities already demonstrated by this research.
