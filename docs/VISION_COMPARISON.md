# Full model versus Mini

November 7, 2026. Four completed perception calls cost **$0.0566778** in reported
model tokens, within a shared $0.10 budget. Zero outstanding reservations. No
training, additional media, caller/analyst generation, or TTS.

| Existing clip | GPT-5.4 Mini | Full GPT-5.4 |
| --- | --- | --- |
| Shot/rebound/save | Calls the buildup an aerial cross, then identifies keeper contact. | Recognizes the initial shot and subsequent save. Still omits the woodwork and rebound header. |
| Defensive challenge | Suggests an uncertain shot/save after the carry. | Describes a sliding defender apparently poking the ball away, labeled an **uncertain tackle**. It avoids the invented shot/save story. |

The full model helped on these inputs, but neither clip is finished showcase output.
It did not recover the complete rebound sequence or confidently establish the
challenge outcome. A model change alone has not solved perception. No percentage
accuracy claim is justified by two development examples and one generation each.

## What was held fixed

For each case, both models received identical image bytes, frame stamps, crop
requests, prompt text, schema, `reasoning.effort=none`, and a 768-token output cap.
The eight original frames and three fixed contact crops came from the prior
inspection runs. This bypassed inspection planning for both models, isolating
classification rather than comparing different tool choices. Labels, reference
transcripts, case names, and expected outcomes were not model inputs.

The save packet spans its selected eight-second source window; the tackle packet
uses the earlier focused 3.5-second contact window. This is recorded perception
diagnosis, not causal replay timing or real-time performance. Source/frame/crop,
input-block, prompt, and schema hashes are in the result record. The per-case hashes
are equal across the two models. All returned timestamps were checked against the
supplied frame set. All four responses completed within the output limit and none
was retried or manually repaired.

The candidate was the dated `gpt-5.4-2026-03-05` snapshot; the baseline was the
`gpt-5.4-mini` alias. [Official model documentation](https://developers.openai.com/api/docs/models/gpt-5.4)
was checked for image input, structured output, reasoning `none`, and standard
$2.50/$15 per-million input/output pricing. These verified rates were registered
only in the comparison process; production pricing/defaults were not changed.
Initial token preflights reserved at most $0.09969375 across the four calls, including
maximum output and the existing 25% margin. No hosted tools or research were used.

Observed totals were $0.0092628 for Mini and $0.047415 for the full model. The Mini
tackle call used cached input, so this ratio is not a general operating-cost estimate.
Backend call durations were about 4.2–4.5s for Mini and 5.5–5.9s for the full model;
these include per-call token preflight and are not event-to-speech delays.

## Decision

Keep the cheap default while investigating the remaining contact/sequence evidence.
Full GPT-5.4 is a useful candidate for difficult observations, with a visible quality
improvement here, but these two outputs do not justify automatically switching all
15 clips to a more expensive model. The next targeted change should address frame
selection and temporal event coverage around the woodwork/header and tackle contact,
using this same footage. Do not compensate for wrong perception by polishing a
confident commentary sentence.

The [complete comparison record](../examples/evaluation-results/vision-model-comparison-20261107.json)
contains all outputs, usage, ledger, source hashes, fixed crop details, and review
limits. Review was performed by the coding agent; it is not human-approved ground
truth. Raw local artifacts and the dated comparison script are under
`runs/replay/vision-model-comparison-20261107/`. No production code or default changed.
