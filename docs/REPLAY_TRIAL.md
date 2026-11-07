# First recorded replay investigation

Development trial, November 6–7, 2026. One twelve-second window from the local
Di María goal source, starting at source-file second 26. These are diagnostic
results from a familiar match, not a held-out benchmark. Speech is simulated.

## What failed

The first Luna run replaced the pending finish frames while its initial vision
request took 5.3 seconds. It saw the attack and then the aftermath. A routine caller
line later started at second 17.36 of the replay, after the action had moved on.

Preserving the frames exposed a separate perception limitation: Luna remained
uncertain about the goal. One GPT-5.4 Mini vision call recognized the ball inside
the net for $0.008410 of reported token usage. The caller and analyst could then
use that saved evidence with Luna for $0.000234 combined.

The clocked mixed-model run revealed more scheduling problems. A later celebration
view was classified as another goal, restarting goal priority and cancelling the
analyst. The analyst's initial last-three-observation context also lost the buildup.

## Changes and verification

- Pending visual input now merges into a bounded sample spanning the unseen window.
  It still has at most eight frames; large gaps can still lose individual frames.
- Routine calls become stale when newer celebration/replay evidence supersedes them.
- Goal reports consolidate until visible play resumes. This prevents another view
  of the same event from restarting the speaking sequence.
- The CLI immediately reacts to an accepted goal with a deterministic, evidence-cited
  caller line. It supplies no unverified scorer or scoreline. `fast_goal_call` traces
  identify this path; the line is not attributed to a language model.
- The analyst retains the recent goal and two preceding observations alongside the
  latest view, with at most six observations initially exposed.
- Saved vision replay preserves exact batch boundaries and recorded observation
  delays, while keeping old usage separate from fresh model charges. Cache misses,
  partial records, and mismatched settings fail without paid vision fallback.
- The observer can request one bounded frame inspection. Unit tests verify crop
  selection and cutoff enforcement. Neither model selected that tool in these paid
  checks; availability alone did not solve Luna's uncertainty.

All 215 offline tests, Ruff, and strict mypy passed. The suite includes regressions
for lost finish frames, stale routine calls, duplicate-goal cancellation, analyst
context, cache isolation, and budget accounting. A separate source review found no
remaining actionable correctness issue in the documented short-clip scope.

## Same saved vision, revised speaking path

The final comparison replays the mixed-model run's visual responses and delays.
Only the speaking-agent requests are fresh. This isolates orchestration changes
from another vision request, although speaking-model output remains stochastic.

| Diagnostic | Previous clocked run | Saved vision with revised runtime |
| --- | --- | --- |
| Goal observations admitted | 2 | 1 |
| First goal reaction, replay seconds | 14.34 | 9.47 |
| Speaking roles appearing | Caller | Caller and analyst |
| Fresh reported model token cost | $0.016891, including vision | $0.000534, speaking only |

The goal reaction starts approximately 4.87 seconds sooner. This is simulated
speech scheduling, not measured audio playback. The revised timeline is:

| Start, replay seconds | Role | Behavior |
| --- | --- | --- |
| 8.83 | Analyst | Sets up the attack; interrupted when goal evidence arrives. |
| 9.47 | Caller | Immediate deterministic goal reaction. |
| 11.72 | Analyst | Explains the preceding move using the retained observations. |

The observer still confirms the goal several seconds behind the video. Final
analysis extends into the configured drain period beyond this twelve-second clip.
This is not yet a synchronized, voiced showcase.

The analyst also repeats an inaccurate "lofted ball" interpretation from the
observer. The footage shows a ground-level attacking sequence. Valid citations
do not prove visual or tactical accuracy; independent factual evaluation remains
necessary before replacing the curated public demos.

## Cost and local evidence

The seven paid checks in this investigation reported $0.031523 of model token usage
in total, about 3.2 cents. Two cancelled requests in the earlier mixed-model run
retain approximately $0.002135 of conservative reservations because their final
usage was unavailable. These figures are application accounting, not an invoice.
No TTS was generated during these checks. Regression tests make no API calls.

Local artifacts are ignored by Git:

- `runs/replay/luna-coverage-fix.json`: preserved finish frames, uncertain Luna output.
- `runs/replay/mini-vision-focused.json`: Mini's isolated goal observation.
- `runs/replay/luna-saved-goal-roles.json`: both speaking agents using saved evidence.
- `runs/replay/mixed-model-clocked.json`: the original mixed-model clocked run.
- `runs/replay/cached-vision-fast-goal.json`: revised orchestration and fresh Luna text.

The [replay guide](REPLAY.md) documents spending limits, source reuse, and controls.
Next quality work needs independently checked event descriptions and aligned
broadcast references. Audio playback, prompt optimization, and RL are not evaluated
by this diagnostic.
