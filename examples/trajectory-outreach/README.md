# When should a commentary agent yield?

Public demo: https://www.aarshroongta.com/demo/trajectory

## Finding

In one retained Villa–Arsenal development replay, a caller line about a save and
clearance was ready at **10.618s**, but older analysis held the microphone until
**15.927s**. Replaying the unchanged speech-floor implementation reproduced that
start at **15.925s** (1.79ms difference). An opt-in rule lets the ready caller
interrupt older analysis when it cites a newer confirmed major action. It started
the identical call at **10.618s**, removing **5.307s** of simulated floor waiting.

The newer analyst explanation then received a complete turn rather than expiring
in the queue. This is not simply an improvement from suppressing all analysis:
estimated analyst airtime increased from 23 to 31.83 words across the older
partial sentence and newer explanation. The original source’s speech times were
already word-count estimates, not measured TTS.

At 2.5 words/second, the baseline caller expires before admission; the candidate
preserves it. At 3 and 3.5 words/second both retain the call. These are controlled
speech-speed assumptions, not measurements of real speakers.

## What ran

- Python's existing `ReplayRuntime._speech`, once unchanged and once through
  `YieldingReplayRuntime`, with a deterministic event clock.
- The exact saved observations, caller/analyst drafts, observation availability,
  candidate-ready times, citations, and originating snapshots from
  `runs/replay/recognition-pass-20261107/save.json` at source revision `08c0497`.
  An unchanged copy is committed as `source-replay.json`; its SHA-256 is in results.
- Fifteen fixed response-delay variants: analyst +0/0.5/1 seconds × caller
  +0/0.5/1/2/4 seconds. These are correlated variants of **one** selected clip.
- Speech-speed sensitivity at 2.5/3/3.5 words per second. Source config: 12s maximum
  snapshot age, 0.25s line gap. Library default maximum age is 8s.
- Regression controls for routine play, same-event evidence, no ready caller,
  stale callers, caller retention, useful analysis retention, and no overlapping speech.

No new model, search, TTS, Trajectory API, training, or RL calls ran. New spend: $0.
The local floor adapter is deliberately coupled to private scheduler objects;
baseline fidelity and regression tests guard against drift. This is an experiment,
not a new stable public API or enabled production behavior.

The adapter does not rerun the entire agent loop: altered dialogue history could
produce different subsequent model outputs. The source is an eight-second clip;
the clock includes processing and drain time after footage ends. No later match
events are supplied. Source observer failure at 11.639s remains in the raw trace;
the experiment does not invent a replacement response. No broadcast media is
republished for this experiment.

The saved earlier analyst line overstates uncertain observations. The demo does
not certify it as true. Even a perfectly accurate version would take the same
airtime. This isolates a distinct editorial signal: correct/supportable words can
still be badly scheduled. The caller's save/clearance is supported by the retained
VLM output, not an independent human fact-check.

The new policy cuts the old sentence midstream. Real playback should test safe
word-boundary interruption, comprehension, and listener preference. There are no
human expert labels yet; policy and initial preference are agent-authored.

## Reproduce

From the commentary repository:

```sh
.venv/bin/python scripts/airtime_experiment.py
.venv/bin/python -m pytest tests/test_airtime_experiment.py tests/test_replay_runtime.py -q
```

`result.json` contains both schedules, metrics, full variant schedules, speech-speed
sensitivity, limits, and provenance. To refresh the existing website:

```sh
cp examples/trajectory-outreach/result.json ../personal-website/public/trajectory/data.json
cp examples/trajectory-outreach/source-replay.json ../personal-website/public/trajectory/source-replay.json
```

The website follows the existing `/demo/auctor` static-page pattern. Its source is
in `../personal-website/public/trajectory/`; `/demo/trajectory` is a Vercel rewrite.
The original Keenable, Auctor, and main soccer showcase routes are preserved.

## Feedback record

Viewers can prefer either schedule and record a reason. The page stores this in
their browser, with JSONL download. A record binds the preference to the source
hash, current timing variant, both schedules and metrics, available observations,
and interruption cost. Nothing uploads automatically. `local-airtime-preference-v1`
is a local schema, **not a verified Trajectory import format**. A QA-generated
browser judgment must not be represented as independent expert evidence.

Use separate signals for factual support, time to call, useful explanation retained,
and interruption quality. Optimizing only latency could reward silence or abrupt
cuts; this demo preserves caller coverage and makes the interruption cost visible.

## Trajectory research and access

Checked November 22, 2026, using public primary sources:

- [Platform introduction](https://www.trajectory.ai/field-notes/introducing-the-trajectory-platform):
  self-serve post-training from tasks and rewards, evaluation, and deployment;
  Continual Grading and Group Size 1 are research previews.
- [Continual Grading](https://www.trajectory.ai/field-notes/continual-grading):
  expert judgments inform a procedure that creates task-specific graders. This
  experiment supplies a potential judgment unit, not a metagrader implementation.
- [Introduction](https://docs.trajectory.ai/introduction) and
  [production traces](https://docs.trajectory.ai/guides/production-traces):
  benchmark/spec tasks and production traces are distinct; production learning
  is documented as private beta. The docs expose explicit reward logging and
  completion on a trajectory.
- [Authentication](https://docs.trajectory.ai/authentication): terminal browser
  login or organization API credentials. The platform opened to sign-in; no
  environment key, project `.env` entry, or saved config directory was present.
  No account was created and no authenticated API was tested. This does not imply
  that all of Trajectory is invite-only or unavailable self-serve.

The strongest connection is defining what good looks like **at the interaction
level**, including when a grounded line gets the floor. Any eventual training or
Continual Grading experiment would need expert-reviewed preferences and separate
evaluation cases. This repository currently prohibits model training and RL.

## Outreach wording

I built live soccer commentary agents and found that a grounded line could still
be the wrong thing to air. In one saved replay, a save call was ready but waited
another 5.31 seconds behind older analysis; a small scheduling change removed that
wait and let the newer explanation air too. I made a replay where you can inspect
both traces and label the handoff you prefer: https://www.aarshroongta.com/demo/trajectory.
I thought this was a useful small example of the interaction-level judgment behind
a value function, and I’d love to work on problems like this as a summer engineering intern.

This wording describes a local scheduling experiment, not use of Trajectory or a
learned improvement. No outreach message has been sent.
