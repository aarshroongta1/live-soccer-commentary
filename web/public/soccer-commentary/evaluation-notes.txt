# Text evaluation trial — 9 November 2026

The pilot found execution and perception failures. **It does not demonstrate that
the multi-agent writer improves commentary.** No candidate was promoted.

The [saved record](../examples/evaluation-results/text-comparison-20261109.json)
contains per-case outputs, rejected drafts, errors, hashes, call summaries, budgets
and limited agent review. The combined experiment had a maximum budget of **$1**.
Reported token cost was **$0.17030795**, with no unsettled reservations: 77 call
attempts using `gpt-5.4-mini` and one deliberate perception-only check using
`gpt-5.4-2026-03-05`. No TTS, fresh research, training or deployment occurred.
Regression tests used scripted backends.

## Completion is different from quality

The suite pilot used identical saved perception within each pair, a $0.02 cap per
writer arm, reasoning effort `none`, 1,024 output tokens per call, zero provider
retries, and at most one validation repair. The optional agent reviewer was off.
Perception preparation had a separate shared $0.24/eight-call cap. It reused seven
historical bundles and attempted the other eight once each.

| Stage | Result |
| --- | --- |
| Perception inputs | 12/15 usable: seven reused, five generated. |
| Perception failures | Corner: inconsistent goal/scene output; foul-decision: invalid frame timestamps; offside: confirmed action without required supporting cue. |
| Writer execution | 5/24 attempted arms completed structural validation; 19 failed. Six additional arms were missing because their perception failed. |
| Complete pairs | Cross/header and penalty. Card had one complete arm. |
| Source review | Both cross/header candidates falsely announce a goal. The penalty candidates correctly call the named low finish; this is one short development example. |
| Whole-suite acceptance | None. References remain unreviewed and major failures remain. |

The original suite pilot, `short-clips-v2`, also exposed a validation asymmetry:
the role loop rejected caller references to uncertain/replay actions, whereas the
joint adapter did not apply that same gate. This was corrected in v3. **Do not use
the v2 completion counts to rank architectures.** They are integration diagnostics.
All original results remain recorded rather than replaced with later successes.

## What changed after the failures

The first $0.0083 penalty smoke showed that the historical writer's long-clip turn
quotas conflicted with short windows. The adapter removed those quotas and supplied
explicit timing information. The subsequent suite still exposed drafts that could
not fit after their latest supporting frame. Repair feedback that merely said a
line ended too late was ineffective.

Targeted development checks on dribble and goal then added per-evidence remaining
word allowances. The final `short-clips-v4` adapter reports the exact number of
words that fit in validation feedback, applies the same uncertainty/replay gate
to both writers, and stops asking the caller again after an analyst abstention
when no newer eligible evidence exists. The original demo/live paths are unchanged.

In the final paid dribble check, both arms completed with **“Goal!” at source 6.0s**.
That establishes a feasible call under the supplied evidence timing; it omits the
dribble and shot sequence and supplies no analysis. It is a timing fix, not a
commentary-quality pass. No complete suite was rerun after these targeted changes.

## The important perception failure

The new cross/header observation invented a confirmed goal, complete with
`goal_line_crossing` and `ball_in_net` tags. Both writers followed that false
evidence. Structurally valid tags and citations therefore failed to protect
spoken factual accuracy.

Source frames at clip-relative 6, 7, 8 and 9 seconds show the goalkeeper continuing
play rather than a scored-goal outcome. The evaluator reference describes a headed
chance, not a goal. This was coding-agent source review, not human approval of the
reference or an independent blinded study.

A deliberate one-call check with the larger existing model used the identical
system prompt, source hash/window, sampler and output cap. **It also invented a
goal.** The check cost $0.016085 and did not justify changing production defaults.
The original baseline frame hashes were not independently saved; identical frame
bytes are inferred from the same source and sampler, rather than directly proven.

## Next work

Prioritize the observer's handling of this false goal using the existing bounded
inspection tools and explicit ball/goalmouth sequence evidence. Do not add a cue
tag in Python or feed the expected outcome to the observer to make the case pass.
Review the minimal necessary supporting frames too: unnecessarily late support
can leave a writer room for only one word even when an event was visible earlier.

Then resolve the three failed perception cases and repeat targeted text checks
under the corrected common contract. A controlled quality comparison needs usable
outputs, reviewed must-call meanings and consistent permissions before any
architecture or prompt-improvement claim. Keep all 15 development cases and every
failed attempt in the record; no broader dataset or new framework is needed.
