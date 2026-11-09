# Text commentary acceptance and comparison

Version 1, 9 November 2026. Scope: the 15 windows in
[action-suite.json](../examples/action-suite.json), 112.5 seconds. Voice, TTS,
pronunciation and listening quality are excluded. This document defines a protocol;
it does not itself report a completed comparison or an accepted quality level.

The [first paid trial](TEXT_EVALUATION_TRIAL.md) is now recorded. It found major
timing and perception failures and establishes no multi-agent quality advantage.
The current comparison adapter is v4; earlier v1–v3 trial outputs are preserved.

## What the saved evidence establishes

The reference audit finds 15 development cases, all with unreviewed annotations,
sparse action labels and empty `commentary` arrays. There are no aligned broadcaster
transcripts in those references. The existing manual checks are useful starting
points, not completed semantic evaluations. No new footage or transcripts are
required to begin reviewing visible claims.

| Saved evidence | Finding | Consequence for the next comparison |
| --- | --- | --- |
| [Action smoke](../examples/evaluation-results/action-smoke-20261107.json) | Some windows stopped on budgets; configurations changed between attempts. | Count budget stops as execution failures; do not interpret them as isolated recognition failures. |
| [Recognition pass](../examples/evaluation-results/recognition-pass-20261107.json) | Save detected while shot/header sequence remained wrong; tackle remained unresolved. | Score sequence meaning and outcomes, not only individual action tags. |
| [Vision comparison](../examples/evaluation-results/vision-model-comparison-20261107.json) | Controlled perception-only comparison on two cases; tackle used a shorter window. | Reuse for diagnosis, not a full-suite or dialogue improvement claim. |
| [Naming iteration](../examples/evaluation-results/player-name-iteration-20261107.json) | Only penalty had valid end-to-end commentary; its negative control did not pass. | Preserve identity abstention and negative controls as explicit checks. |
| [Researched handoff](RESEARCH_HANDOFF.md) | One penalty case exposed an invented keeper touch and an unsupported lead claim before revisions. | Check every asserted contact, identity and score-state claim independently of citation validity. |
| [Session handoff](SESSION_HANDOFF.md) | Text/voice candidates were not accepted as coherent ongoing commentary. | In-run reviewer acceptance and structural passes do not establish natural dialogue. |

The historical joint writer is in `src/commentary/recorded_demo.py`; its automatic
entry point is `scripts/generate_demo.py`. It sees the full clip's observations.
`scripts/prepare_agent_demo.py` prepares recorded agent dialogue, also with
full-context perception. `scripts/replay_commentary.py` is the causal replay path.
Comparing the joint writer directly with causal replay would change information
access and timing as well as architecture.

## Review unit and evidence

Review a complete clip and its timestamped text, then annotate atomic claims in
each line: action, outcome, actor, team, ball path, sequence, tactical explanation,
score state, or research fact. Record each claim as supported, contradicted,
unverifiable, or explicitly uncertain. Attach source time/frame or dated research
evidence and a short reason. A valid evidence ID alone cannot make a claim supported.

Review the video independently before seeing candidate text. Preserve uncertainty
when contact, identity or timing is unclear. Record when each fact first becomes
visible, not just when the underlying action begins. Keep model output out of
reference drafting. A later celebration identity does not identify an earlier
scorer unless action-linked evidence establishes that connection.

The evaluator-only [worksheet](../examples/evaluation-results/text-review-v1.json)
contains all 15 case IDs, reference hashes, draft events and review checks. A coding
agent inspected 12 evenly spaced frames per clip and recorded provisional must-call
meanings and unresolved details. This is sampled reference triage, not full-motion
verification. Candidate ratings remain null: **not scored**, never zero or pass.
It is not a runtime input or a completed result. Record reviewer identity and whether review was by a human or
coding agent; never silently promote agent review to human-approved truth.

## Rubric

Score each applicable dimension 0–3 with a concrete timestamped example. A missing
artifact or unresolved reference is `not_assessable`, not a passing score. Scores
below are proposed development acceptance rules, to be frozen before generation.

| Dimension | 0 | 1 | 2 — acceptable | 3 — strong |
| --- | --- | --- | --- | --- |
| Event and sequence recognition | Wrong central event/outcome or none recognized | Partial recognition; material sequence omitted/confused | Main event and outcome correct; minor detail missed | Relevant sequence, outcome and replay distinction all correct |
| Player naming | Wrong player or wrong action attribution | Clear action-linked identity missed, or uncertainty overstated | Correct supported names; abstains where unreadable | Uses available names naturally with correct roles across the sequence |
| Grounded meaning | Material contradiction or invented fact | Unsupported noncritical detail or overconfident inference | All asserted facts supported; uncertainty explicit | Precise claims, well-calibrated uncertainty and useful evidence selection |
| Text timing | Outcome stated before support, or central event never usefully called | Main call is late, stale or displaced by filler | Main call follows support within 2 source seconds; minor awkward placement | Prompt, ordered reaction and explanation at useful moments |
| Useful analysis | False explanation or invented tactical pattern | Generic praise, repetition or irrelevant statistics | One specific supported insight when the clip permits it | Concise explanation connects visible actions and adds understanding |
| Dialogue coherence | Contradictory or disconnected turns | Repetition, awkward restart or forced handoff | Connected progression without needless repetition | Caller and analyst build a compact, natural exchange with appropriate restraint |

Naming is `not_applicable` when independent review finds no readable action-linked
identity. Track those abstentions separately from missed readable identities.
Analysis is `not_applicable` when the short window offers no supported insight or
reasonable space for a response. An accurate caller-only clip can pass; forced
analyst participation cannot earn credit. An absent analyst on an eligible clip
scores 0 for useful analysis. Coherence applies to the text sequence even with one
speaker; it does not measure vocal delivery.

The 2-second timing target is a proposed source-placement criterion, not measured
model latency. Report timestamp uncertainty (currently roughly 1 second in drafts)
and mark boundary cases unresolved until inspected. Report both event-to-text-
availability delay and event-to-planned-line delay when available. Prepared output
has offline generation time and planned source placement; simulated speech lengths
and execution clocks cannot establish actual live delivery latency.

## Gates and silence

A clip fails acceptance if it contains a wrong scorer/team, invented goal/save,
wrong official decision, unsupported score state, replay announced as a new event,
or an asserted research fact contradicted by its source. Also fail any verified
future-information use in a run declared causal. Full-context recorded runs must
be labelled offline and are ineligible for causal claims even when lines are
placed after event timestamps. Invalid citations, unknown provenance, budget
violations and incomplete execution block acceptance pending resolution.

For each clip, independently mark the must-call events, acceptable uncertainty,
and any quiet intervals. Existing `required` labels include replay/celebration
diagnostics; they are not automatically a requirement to narrate each one. Freeze
the must-call list before looking at outputs. Measure semantically correct must-
call coverage separately from the existing citation-coverage metric.

Silence during routine circulation or unreadable identity is reasonable. Silence
through a clear goal, miss or final decision is a coverage failure. An empty output
has zero must-call coverage and cannot win by having no hallucinations. Report
claims, supported claims, contradictions, unverifiable claims, must-call coverage,
missed readable names and correct abstentions as counts with denominators. No
claims means an undefined supported-claim rate. Do not optimize words per second
or require every frame to be narrated.

A clip passes only with resolved references, no gate failure, complete must-call
coverage and every applicable dimension at least 2. Whole-suite acceptance requires
15/15 clip passes; otherwise publish the individual failures. This strict
development target is not evidence of general accuracy. Do not combine dimensions
into a weighted score that can hide factual failures.

## Controlled comparison

1. **Review references first.** Inspect the existing local windows; refine event
   times, first-visible support, outcomes, identity evidence, must-call and analysis
   eligibility in the worksheet. No expected script is required. If broadcaster
   transcripts become available later, verify names/offsets and use them as an
   additional paraphrase-compatible reference, outside runtime inputs.
2. **Freeze a recorded text comparison.** A is an automatic joint writer;
   B is separate caller/analyst without the optional reviewer. Both get the same
   immutable full-context observations, accepted identities, fixture context and
   saved research facts. Neither gets references, target prose, labels or worksheet
   notes. Use opaque runtime clip IDs, not descriptive case/file names. This tests
   writing/coordination on shared perception, not end-to-end recognition or live
   superiority. Score shared recognition once and carry its failures into both
   outputs' grounding review; it cannot explain a writing advantage.
3. **Hold resources fixed.** Pin the same existing inexpensive text model and
   reasoning setting in both arms, prompt/config hashes, source SHA/window,
   observation/research hashes, output allowance and seeds if supported. Reserve
   at most $0.03 and 8 model calls per clip per arm: 30 runs, $0.90 and 240 calls
   maximum for the text-only suite. Count invalid/rejected responses and repairs.
   Actual model IDs and reservation feasibility must be frozen before execution;
   insufficient allowance is reported, never silently increased. Shared perception
   preparation is separately costed and requires a separate explicit budget if
   missing. No fresh research, TTS, provider retries or automatic reruns. These are
   proposed caps. The user authorizes experiments at $1 or less and requires
   approval above $1; include every phase in the total before starting.
4. **Control corrections.** Disable provider retries; allow at most one explicitly
   logged validation repair in each arm, inside its cap. Disable optional agent
   reviewer, deterministic fast-goal shortcuts and manual wording edits in the
   primary comparison. Keep normal deterministic validation/timing adjustments
   visible in audits. A reviewer-enabled variant is a later separately labelled
   ablation, not the primary B arm.
5. **Review blind.** Shuffle A/B position per clip using a saved seed and hidden
   mapping. Present identical formatting and timestamps. Evaluate each arm against
   video before choosing A/B/tie/neither with a reason. Keep architecture, model
   and cost hidden during quality review; reveal them afterward. The in-run reviewer
   cannot serve as the independent judge. Start with manual review; calibrate any
   later model judge against it.
6. **Report all attempts.** Per clip: gate failures, six dimension scores, coverage,
   claim counts, preference, actual cost/calls/tokens, failures, generation time and
   planned timing. Include all 15 cases in the completion table; failed/missing
   arms do not disappear from averages. Show valid paired preferences separately
   from technical failures. Claim a preliminary development advantage only if B
   wins more valid pairs than A, adds no critical failures, and does not reduce
   must-call coverage. Otherwise report mixed or inconclusive results. No
   significance claim from one generation per arm.

All 15 windows have already been used for development. Do not relabel them held
out. Resolve match groups before future prompt optimization: England penalties,
miss and award stay together; France–Jamaica header/card stay together. A source
compilation can contain different matches (the two saves windows have different
reference match IDs), and alternate uploads can contain the same match. Split by
verified fixture, not file or case ID. Future unseen evidence is a separate task;
do not acquire more data for this comparison.

## Comparison implementation

`scripts/compare_text.py` and `src/commentary/replay/text_comparison.py` implement
the shared-input runner and blind JSON review export. Default mode inventories
all 15 cases and validates supplied bundles, source hashes/windows and schemas
without creating a model client. Explicit `--spend --model MODEL_ID` opts into
generation, with hard $0.03/eight-call limits per arm by default
(`--per-arm-max-usd` may lower the dollar cap), no provider retries,
1,024 output tokens per call by default, and reasoning effort `none`. Every
response, rejected draft, repair, partial output and failure remains in the audit.
The paid branch is also exercised end to end with a scripted provider in tests.

The runner's version is **short-clips-v4**, an explicitly adapted baseline rather
than unchanged historical output. The old joint schema requires 7–12 lines and
three substantial analyst turns, which cannot reasonably fit five-second clips.
The comparison permits 0–7 lines, optional analysis and one validation repair in
each arm. It preserves historical grounding instructions and uses the existing
role/tool loops with the reviewer disabled. The agent adapter presents the same
complete observation pool at each turn, with an explicit offline planning prompt;
it does not use the causal replay or source-filtered recorded scheduler. Both
arms use the same support-aware placement, spacing and total-word constraints.
No goal shortcut or hand-edited output participates.

The original scripts remain individual entry points:
`generate_demo.py` has no dollar/call CLI guard and the joint writer allows an
8,192-token response plus one validation repair. `prepare_agent_demo.py` is
budgeted but enables the reviewer. Do not run those defaults and assume they
satisfy this protocol.

Use scripted backends to verify budget stops, rejected output accounting, reference
isolation, shared inputs and one-repair limits before paid execution. This protocol
does not require a new framework, service, model judge or prompt optimizer.

The existing `scripts/evaluate_replay.py` remains useful for saved `ReplayResult`
structural checks with its explicit default 2-second event-match tolerance. It does
not accept a joint-writer `DemoPlan` directly, verify sentence meaning, or turn
sparse-label unmatched detections into proven false positives. Preserve original
outputs and use a clearly identified adapter if adding a common scorecard.

Seven historical perception bundles and their provenance are saved in
[text-comparison](../examples/text-comparison/README.md). The later paid preparation
added five usable bundles; `prepared-inputs.json` contains 12 inputs, while corner,
foul-decision and offside remain explicit failures. This is not a newly
generated or quality-approved perception baseline. Recognition prompt changes
and GEPA follow measured failures; neither is required to run this comparison.
