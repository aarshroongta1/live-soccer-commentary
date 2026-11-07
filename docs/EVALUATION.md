# Commentary evaluation and improvement

Status: the initial structural replay evaluator and synthetic diagnostic are
implemented; see [the replay guide](REPLAY.md). Broadcast reference curation,
semantic/audio evaluation, baseline experiments, and training/optimization remain
proposed next stages.

Build the replayable real-time multi-agent core first. Keep the pre-match
researcher, observer, caller, and analyst separate, with timestamped evidence,
bounded tool access, and a deterministic speech coordinator. Evaluate the complete
system as well as each role. Preserve the curated five-clip demo and the saved
ElevenLabs voice settings while developing this path.

## Reference data

Use actual broadcast commentary as a reference for event selection, correct player
mentions, analysis, and delivery. A broadcaster's sentence is one valid description,
not the only acceptable output. Verify factual claims against video, event labels,
and cited pre-match sources; do not treat every broadcast assertion as ground truth.

Store clip and match IDs, source and usage terms, video/audio offsets, speaker turns,
timestamped reference transcripts, verified event/entity annotations, and annotation
confidence. Review automatic transcriptions for names and numbers. Keep reference
audio, transcripts, and future event labels outside the agents' runtime inputs.
Remove intelligible original commentary from runtime audio, or start video-only, so
the system cannot simply hear its evaluation target. Crowd sound can also reveal
events and must respect the same replay clock.

The researcher receives only information available before the fixture. For historic
matches, an article's publication date alone does not prove that its current contents
were available then; use archived or verified snapshots, otherwise omit the claim.

SoccerNet-Caption is a candidate source of timestamped textual commentary and team
lineups, not a substitute for a verified broadcast audio collection. Its published
captioning protocol uses a 30-second matching window and METEOR, which is too coarse
to establish low-latency live commentary quality. Check dataset access and usage
terms before acquiring it.

Split development, validation, and final test data by match, keeping overlapping
clips and alternate uploads of the same event in one split. Include routine play,
missed chances, ambiguous identities, replays, and quiet stretches alongside goals.
The existing five famous goals are showcase examples, not an unseen test set. Famous
events also carry model-memorization risk; evaluate unfamiliar matches and changed
context as well. Never feed future frames to a replay agent, including through
precomputed summaries or tool responses.

## Scorecard

Report dimensions separately before considering any combined objective:

| Dimension | Evidence and measurement |
| --- | --- |
| Event agreement | Match predicted events to verified reference events within explicit time tolerances; report precision, recall, misses, and invented events. |
| Factual grounding | Check player/team identities, score, sequence, and statistical claims against available evidence and source dates. Track unsupported assertions and appropriate abstentions. |
| Real-time behavior | Measure event-to-speech-start delay and percentiles, deadline misses, stale outputs, interruptions, and overlapping speech using actual playback timestamps. |
| Commentary quality | Blind pairwise human judgments of relevance, useful analysis, repetition, natural dialogue, and appropriate silence. Use a calibrated model judge as an aid. |
| Audio delivery | Blind listening comparisons for pronunciation, emotion, pacing, and caller/analyst handoffs, with voice configuration held fixed when comparing scripts. |
| Agent behavior | Tool validity, evidence citations, failed handoffs, budget compliance, and whether each role contributes useful work. |
| Efficiency | Per-role and complete-run latency, tokens, tool calls, TTS usage, and cost. |

Use aligned semantic comparisons as supporting diagnostics, not exact-word targets.
An accurate paraphrase should pass; an exciting sentence with the wrong scorer
should fail. Align broadcast audio to visible events before judging timing because
the source itself can have an offset. Preserve both event time and spoken time.

An in-run reviewer helps the agents, but is not an independent evaluation. Use
separate held-out annotations and a distinct evaluator context. Validate any model
judge against human labels, randomize pairwise ordering, and track disagreements.
Use match-level uncertainty estimates when there are enough independent matches;
do not count many windows from one match as independent evidence.

## Improvement sequence

1. **Establish a baseline.** Freeze prompts, model IDs, configuration, dataset split,
   and budgets. Run the current automatic joint-writer path and the new multi-agent
   path on identical eligible clips. Manually edited showcase scripts are not raw
   baseline outputs. Compare agents under similar budgets, then report any useful
   quality-versus-cost tradeoff.
2. **Optimize prompts and examples.** Start with bounded failure-driven prompt
   optimization, with GEPA as the first candidate. Give it development traces and
   feedback, let it propose prompt revisions, and select candidates on validation
   data under a fixed evaluation budget. This optimizes prompts, not model weights.
   Keep test data untouched and version every candidate. A bounded critic revision
   during a run is a separate mechanism from this across-run improvement loop.
3. **Learn from preferences if justified.** Collect preferred/rejected generations
   for the same input, including our manual corrections. Consider supervised
   fine-tuning for good demonstrations, then DPO for pairwise preferences on a model
   we can actually fine-tune. Neither method by itself trains the ElevenLabs voices.
4. **Try RL only after the reward is credible.** A later GRPO experiment could train
   a small caller or analyst policy using multiple sampled outputs and grounded
   quality feedback. It requires a trainable model, rollout budget, and independently
   checked rewards. Start with one role; whole-system multi-agent credit assignment
   adds complexity. Do not equate reward-model score with demonstrated improvement.

Further domain research suggests **player grounding** as an alternative first RL
target: [GameSight](https://arxiv.org/html/2604.00057v1) reports SFT and GRPO for
player alignment with a binary reward. This offers a verifiable subtask before
attempting subjective commentary-style rewards. Its segment-based visual context
must be adapted to our causal replay constraints. See [the framework and related
work review](RESEARCH.md) for this and reference-data candidates including MatchTime
and SoccerNet-Echoes.

Correctness, evidence availability, and temporal leakage checks gate candidate
promotion. Among passing candidates, compare quality, delay, and cost. Do not let a
weighted average compensate for a wrong scorer or a future-information leak.
Targeted adversarial tests should include silence that games hallucination metrics,
verbose stat dumping, copied reference wording, and a judge-favored but false call.

Keep improvement offline. No prompt self-editing during a match and no automatic
production promotion. Promote versioned candidates only after regression checks;
publish measured results, including regressions, rather than claiming RL gains
before training and evaluation have occurred.

## First deliverables

- A versioned reference manifest and a small manually reviewed diagnostic set,
  with development/validation/test provenance and no runtime access to references.
- A replay evaluation command producing per-clip JSON and an aggregate scorecard,
  plus saved role/tool/speech traces and blind listening comparisons.
- A baseline-versus-agent experiment and a bounded prompt optimization experiment,
  with data hashes, prompt versions, budgets, failures, and repeatable run settings.
- A decision based on those results about whether fine-tuning or RL is worthwhile.

## Research consulted

- [SoccerNet dense video captioning](https://www.soccer-net.org/tasks/dense-video-captioning):
  timestamped soccer commentary data, lineups, and the published captioning protocol.
- [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457):
  reflection on trajectories to propose and evaluate prompt updates. Results from
  its tasks motivate an experiment here; they do not establish gains on commentary.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290): preference
  fine-tuning using chosen/rejected responses without a PPO training loop.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300): introduces GRPO in mathematical
  reasoning. Its results are not evidence that GRPO will improve soccer commentary.
