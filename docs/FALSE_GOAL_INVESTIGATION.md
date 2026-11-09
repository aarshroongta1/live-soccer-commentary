# False-goal investigation — 9 November 2026

The confirmed failure is **overconfident visual classification followed by a
self-reported evidence check**. The observer invents a goal and the supporting
`ball_in_net` / `goal_line_crossing` tags together. The contract checks that these
fields are consistent; it does not independently verify the pixels. The caller
and joint writer then receive the same false confirmed event and narrate it.

This explains how the error reaches commentary. It does not fully explain the
model's internal visual mistake or establish a reliable runtime fix.

## Evidence before experiments

The original raw provider responses already contained the false goal, so Python
normalization, identity resolution and speaker phrasing did not create it.
Reconstructed JPEGs matched all ten saved larger-model input hashes exactly.
The underlying clip is 640×360 at approximately 29.97fps; the recorded observer
received one sampled frame per second. No stale input or source-window shift was
found. The disputed ball/goal configuration is small and visually ambiguous.

The relevant code path is `sample_clip` in `src/commentary/recorded_demo.py`,
`observe_recorded` in `src/commentary/replay/recorded.py`, and `supported_claim`
in `src/commentary/football.py`. The last function checks membership in a set
of cue names and requires support timestamps. It has no images to inspect.
This is a structural contract, not an independent visual verifier.

## Controlled checks

The [complete record](../examples/evaluation-results/false-goal-investigation-20261109.json)
contains source/frame hashes, actual prompt/block hashes, outputs, errors and
provider usage. The existing small model, output schema, output allowance and
reasoning setting stayed fixed. Each condition had two planned samples, with
no provider retries or schema-repair retries.

| Condition | Structurally accepted responses | Raw responses claiming a goal |
| --- | --- | --- |
| Original framing, 1fps | 1/2 | 2/2 |
| Replace only “highlights” with “football” | 1/2 | 1/2 |
| Original framing, 2fps | 2/2 | 2/2 |
| Original framing plus goalmouth magnifications | 2/2 | 2/2 |

The two rejected responses still explicitly asserted a goal; their error was a
scene/action schema inconsistency. They must not count as correct abstentions.
The neutral-word response without a goal instead asserted a save; that alternative
was not independently verified and is not a recognition-quality pass.

The magnifications were manually specified diagnostic regions of frames 5, 6 and
7 seconds, taken from the old observer's claimed support times. Original full
frames remained available. No expected outcome, labels, transcript or previous
model claim was supplied. This tested attention to the disputed region, not an
autonomous crop-selection improvement. Upscaling adds no original source detail.

Neither denser sampling nor magnification removed this failure in these samples.
The wording result shows output sensitivity, but two samples cannot establish that
the word “highlights” is the cause. There is no supported one-parameter fix here.

## A more revealing inconsistency

A final neutral task asked the model only to locate the match ball in those same
three frames, without requesting an event or providing the earlier goal claim.
It marked the ball **ambiguous in all three frames**. Its coordinates were also
inconsistent with the earlier claim that the ball was clearly inside the left
side of the net. The localization output is another model prediction, not ground
truth or a trace of the model's internal reasoning.

This supports a diagnosis of poorly calibrated visual interpretation: event
narration commits to a familiar scoring story even when an object-level task
cannot establish clear ball positions. Raw responses then describe the keeper's
later recovery as happening “after the goal,” rather than reconsidering the
initial interpretation. Narrative completion is an inference from these outputs;
we cannot prove a particular internal neural mechanism.

## Fix direction and limits

Adding another cue tag or asking the same goal question again does not make the
evidence independent. A correction needs a more reliable ball/goal-plane check or
conservative abstention when that evidence cannot be established. It must also
retain real-goal detection: blocking every goal would game the test.

No runtime change was made or advertised as a fix in this investigation. A future
guard needs false-goal and real-goal controls before promotion. The diagnostic
harness is `scripts/investigate_false_goals.py`; its default is a free audit and
paid conditions require `--spend`. Regression tests use synthetic images only.

The investigation used nine call attempts, with a combined maximum budget of
$0.27. Reported token cost was **$0.0348588**, with no unsettled reservations.
No larger-model call, audio generation, new footage, training or deployment was
part of this investigation.
