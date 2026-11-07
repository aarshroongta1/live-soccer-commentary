# Replay agents

This is an experimental core alongside the recorded demo and existing `/live`
browser experiment. It does not replace the public five-clip showcase. The first
implementation supports causal video replay and independent role/tool decisions;
it does not synthesize or stream live audio yet.

## Responsibilities

- **Researcher:** before replay, searches for up to six cited, dated player/team
  facts, or loads a saved brief bound to the fixture and cutoff. Research is optional;
  source dates/URLs are structurally validated, not independently verified.
- **Observer:** receives only available sampled frames. Readable action-linked
  shirt details are resolved against a separately supplied roster. The speakers do
  not receive that roster as a way to guess identities. It can select one bounded
  `inspect_frame` crop of a supplied frame, then return its final observations.
  A failed or cancelled inspection preserves its request in the observer audit.
- **Caller:** independent prompt, model selection, and turn state for play-by-play.
  May choose `recent_events` or `player_context`; cannot access research.
- **Analyst:** independent prompt, model selection, and turn state for explanation.
  May choose the same visual tools or a cited pre-match `research` lookup.
- **Coordinator:** deterministic asyncio code owns the clock, input visibility,
  deadlines, speaking turns, and goal priority. It is not an LLM supervisor.
- **Optional draft reviewer:** a bounded review/revise branch in a role turn.
  Disabled by default, since it adds latency. It is not independent evaluation or
  an asynchronous post-match critic.

Each role's LangGraph chooses a typed `tool`, `speak`, or `abstain` action. Python
executes only permitted tools and routes results back to that role. This is a
model-selected application action protocol over our existing structured-output
backend, not provider-native function calling. Default budget: three actions and,
if enabled, at most one reviewer-requested revision within that budget.

The clock continues while model calls run. There is one observer job and at most
one job per speaking role. When vision falls behind, pending batches merge into at
most eight frames spanning the unseen window instead of replacing the entire
earlier batch. If this exceeds the frame limit, evenly spaced sampling can still
miss an event; dropped frames and merged batches are traced. Each role receives a
deep-copied evidence snapshot. Invalid
citations, mismatched result origins, stale responses, and future evidence are
rejected. New goal evidence interrupts stale speech from either voice and protects
a queued goal call from being replaced by later routine play.
Routine play-by-play is also discarded when newer visible celebration or replay
evidence makes it obsolete, even if a debugging run uses a relaxed age limit.

These are cooperative asyncio deadlines, not hard real-time guarantees. A backend
must propagate cancellation. No framework can make an unresponsive provider finish
within a match deadline.

## Offline control-flow diagnostic

Run from the repository root after `uv sync --dev`:

```bash
uv run python scripts/replay_commentary.py --diagnostic --rate 10 \
  --out runs/replay/diagnostic.json
uv run python scripts/evaluate_replay.py \
  --run runs/replay/diagnostic.json \
  --reference examples/replay-evaluation/diagnostic.json \
  --out runs/replay/diagnostic-score.json
```

Output paths must be new; commands refuse to overwrite existing files. The
diagnostic models a fictional Green–Orange match. It uses scripted role responses
and an evidence oracle, with real tool execution, LangGraph routing, independent
role turns, and replay scheduling. Its reference labels are loaded only by the
separate evaluator. This is a reproducible software diagnostic, not an unseen
commentary benchmark or a demonstration of visual understanding.

## Replay a local video with model agents

Without `--spend`, this validates the request and prints a dry-run summary:

```bash
uv run python scripts/replay_commentary.py \
  --clip clips/demo-sources/argentina-france-dimaria.mp4 \
  --start 12 --duration 28 --out runs/replay/dimaria.json
```

Add `--spend` to enable paid observation and role calls using `OPENAI_API_KEY` in
the local `.env`. OpenAI development defaults use GPT-6 Luna, reasoning `none`,
and no retries. Each role can be selected independently using
`--observer-model`, `--caller-model`, and `--analyst-model`. For Anthropic, pass
`--provider anthropic`; the default becomes `claude-haiku-4-5` and requires
`ANTHROPIC_API_KEY`. An optional `--pack` file uses the recorded workflow's team
and roster JSON format.

Pre-match research requires both `--fixture "Argentina v France"` and
`--as-of 2022-12-17`. For budgeted OpenAI replay, also supply
`--research path/to/brief.json` to reuse a `ResearchBrief`; its fixture
must match exactly and its date cannot exceed the requested cutoff. For historical
matches, verify archived source contents; a source page may have changed since its
publication date. Research cannot establish who performed an action in the footage.

OpenAI paid replay defaults to `--max-usd 0.10 --max-model-calls 40`. Each request
preflights image/schema input tokens and reserves maximum output cost with a 25%
margin. Concurrent calls share the ledger. Failed or cancelled calls retain their
reservations when final usage is unknown. Hosted tools and long-context requests
are refused under this guard, so fresh web research cannot add untracked fees.
Saved research is reusable without a search call. These limits apply to the OpenAI
replay path, not to Anthropic or the older generation scripts, and are application
guards rather than a provider-account billing cap. Completed usage and outstanding
reservations are saved in `metadata.spend_guard`.

GPT-6 Luna standard pricing was checked on November 6, 2026: $0.10 input and $0.50
output per million tokens, with $0.01 cached input. See the
[official model page](https://developers.openai.com/api/docs/models/gpt-6-luna).
Use scripted diagnostics and tests for regressions; paid runs are short smoke tests.
For deliberate vision quality checks, `--observer-model gpt-5.4-mini` keeps the
speaking roles on Luna. Its checked standard rates are $0.75 input and $4.50 output
per million tokens. One saved Di María finish-window call recognized the goal for
$0.00841 in reported token usage; this is a single development case, not a benchmark.
Luna remained uncertain on the same footage. Tool availability alone did not make
Luna request a crop in these trials, so its inspection policy still needs evaluation.
The observer output cap is 1,024 tokens; an optional crop adds at most one model call
inside the existing deadline and shared spend budget.

Use `--reviewer-model` only for a deliberate draft-quality experiment. It consumes
the same turn deadline. The default live-oriented path leaves it off.

Frame JPEGs are sampled locally before replay, then revealed only as the clock
reaches their timestamps. The model does not receive the original commentary audio,
reference transcripts, curated scripts, or future video. This is bounded to short
clips using the existing sampler's 90-frame ceiling, not a full-match capture system.

`--rate` controls replay seconds per wall second. Deadlines and stale-age limits are
expressed in replay seconds, so accelerating replay shortens real provider budgets.
Use rate 1 when evaluating model latency; accelerate the scripted diagnostic for
convenience. `--batch-seconds`, `--observer-timeout`, `--turn-timeout`, `--max-age`,
and `--drain` control the bounded run. Drain permits pending work and simulated
speech to finish after the final video frame, so some turns can start after clip end.

## Inspecting a run

The saved JSON contains visual observations, speech records, and chronological
coordinator traces. Metadata records exact runtime configuration, model choices,
role prompt hashes, roster source/hash, research content/provenance, observer audits,
and aggregate provider usage. Treat local roster and source files as run inputs;
their hashes alone do not recover the files.

Role audits preserve selected actions, tool results, reviewer feedback, and usage,
including rejected model responses. Their timestamps denote snapshot cutoffs,
explicitly marked by `timestamp_basis`; coordinator traces record actual replay
arrival/start/cancellation times. A failed or cancelled turn can appear in the role
audit without a final speech record. A cancelled provider response may still incur
charges that the provider did not return in a completed usage object.

Speech duration is currently an estimate from word counts. Interruption records
shorten estimated occupancy, but do not establish which words were actually audible.
The configured ElevenLabs voices and saved settings are untouched. Actual TTS
streaming, cancellation, audio capture, and listening tests are the next integration.

## Initial evaluation

`scripts/evaluate_replay.py` consumes a completed result and a separate
`EvaluationCase`. `validate_manifest` prevents duplicate cases or the same match
crossing development/validation/test splits. The checked-in case is synthetic.

The scorecard measures one-to-one event-kind/timestamp matches, missed and unmatched
events, citation-based speech coverage, evidence availability, resolved-player
agreement where annotated, latency percentiles, role duration/share, overlap,
interruptions, errors, stale results, and tool calls. Invalid research citation IDs
are distinguishable from unverified factual claims.

Matching an event and citing its ID does not prove that spoken words describe it
correctly. Reference transcripts are stored separately but semantic commentary
comparison is not implemented. These diagnostics also do not assess tactical
insight, emotion, pronunciation, or vocal quality. Do not present synthetic scores
or simulated timing as broadcast quality or measured audio latency.

## Next experiments

1. Run the model path on actual footage and inspect failures before voicing it.
2. Curate aligned broadcast reference transcripts and diverse, match-separated
   evaluation clips, including missed chances, ambiguity, and quiet play.
3. Integrate the existing ElevenLabs voices behind a cancellable speech adapter;
   measure actual event-to-audio timing and listen to both roles.
4. Compare generated outputs to the original automatic pipeline under fixed budgets.
5. Add bounded GEPA prompt experiments. SFT/GRPO and automatic prompt promotion are
   not implemented; see [the evaluation plan](EVALUATION.md) and
   [research review](RESEARCH.md).
