# Prepared agent conversation

The `/demo` page includes a 20-second England–France penalty sequence, taken
from source seconds 42–62. Five unedited model-written lines cover the attack,
penalty decision, goal and analysis: three caller turns and two analyst turns.
Both saved custom ElevenLabs voices are used. The measured render has no delayed,
overlapping or truncated lines; see [render-report.md](render-report.md).
Subjective voice delivery has not been rated by listening.

This is prepared recorded commentary. A full-sequence vision pass compares the
action and replays, an identity agent requests the researched roster, and separate
caller/analyst agents take bounded turns. Each speaking snapshot includes only
observations whose supporting frames are available by its source-time cue.
Word budgets reserve space for upcoming important events. These checks do not
make the full-context observer a live causal system. Processing time is shown in
the trace, separately from the 20 seconds of video playback.

The accepted vision pass saw 24 seconds. Its final four seconds cut to a different
attack, so the accepted output uses the first 20 seconds and excludes later
evidence. No benchmark labels or reference commentary were given to the agents.
No models were trained or fine-tuned.

- [Plan](penalty-plan.json): unchanged generated words and evidence citations.
- [Preparation trace](../../web/public/soccer-commentary/prepared-penalty.trace.json):
  observations, identity lookup decisions, accepted/rejected turns and source times.
- [Research](research.json): reused cited pre-match squad brief.
- [Evaluation record](evaluation.json): failed iterations and their costs.
- [Voice summary](synthesis-summary.json): five requests, 91 credits, saved voice
  settings without request overrides. Caller spoke 3.44 seconds, analyst 6.59.

Known model usage across the iterations was $0.2035728, plus approximately $0.051
for one failed vision response whose exact usage was not retained (about $0.26
total). Mini handled development and speaking; two deliberate full-context vision
attempts used GPT-5.4. The final trace's budget and the plan's `metadata.cost`
cover the final dialogue pass only, not reused preparation or experimentation.
Inherited usage fields in reused artifacts describe their originating phase.
This is one reviewed example, not evidence of improvement over all 15 clips.

## Reproduce

From the repository root, inspect a free dry run:

```bash
uv run python scripts/prepare_agent_demo.py \
  --clip clips/action-sources/england-france-penalties.mp4 \
  --start 42 --duration 20 \
  --research examples/recorded-agent-demo/research.json \
  --teams examples/recorded-agent-demo/team-context.json \
  --reuse-vision web/public/soccer-commentary/prepared-penalty.trace.json \
  --max-usd 0.05 --max-model-calls 18 \
  --out-dir runs/prepared-demo
```

`--spend` enables model calls, rerunning roster resolution and dialogue from the
saved observations. Omitting `--reuse-vision` also runs fresh vision; the default
observer and speakers use Mini. The accepted stronger observer was explicitly
selected with `--observer-model gpt-5.4-2026-03-05`. Source SHA-256 and window
checks reject incompatible cached observations. Outputs and provider responses
are saved under the new output directory; existing directories cannot be overwritten.
Model failures consume budget, are recorded, and do not trigger automatic retries.
TTS and fresh web research are separate from this command.

To render this exact accepted plan, follow the [speech and rendering commands](../../docs/DEMO.md#render)
with `--plan examples/recorded-agent-demo/penalty-plan.json` and the matching source
clip. Generation can vary; a fresh model run is not promised to reproduce these
words. Media files are ignored by Git. Locally the prepared MP4 is at
`web/public/soccer-commentary/prepared-penalty-agents.mp4`; a hosted copy can be
configured with `PREPARED_DEMO_VIDEO_URL`. A fresh clone can inspect the transcript,
research and trace without video or API calls.
