# A growing memory across calls

Public replay: https://www.aarshroongta.com/demo/noeri

The UI shows six calls, the same penalty footage, and three lessons accumulating
from earlier corrections. Later calls receive cumulative memory plus their own
preceding dialogue. Green highlights show the lesson IDs returned by the model.
This replaces the clip comparison as the main experience; the earlier three-clip
experiment, including its regression, is retained in the parent directory.

## What is real

- Earlier calls are exact retained lines from three takes of the England–France
  penalty in `examples/evaluation-results/researched-handoff-20261107.json`.
- The contact and score corrections come from that retained investigation.
  The repetition correction is a new coding-agent editorial review, not human
  feedback. The three takes are arranged in review order, not original run order.
- A model distilled each correction separately, then the script persisted the
  growing list after each lesson. The three principles are unedited model output.
- Three new calls actually received all three saved lessons, the same saved
  observations and preceding new dialogue. The final call chose to abstain.
- Raw outputs, exact prompts/inputs, source hashes and costs are retained here.
  Model-selected lesson references are inspectable but not proof that a lesson
  caused a particular phrase. All three lessons were supplied; this is not a
  retrieval-system benchmark. The score lesson was not referenced by the model.

This is a reconstructed review history and a later pass over a **familiar event**.
It demonstrates memory persistence and reuse, not generalization to new events,
autonomous discovery of errors, live model timing, or flawless improvement. The
later analyst still partly repeats the goal. No new visual recognition ran.

An input-recording bug kept a shared dialogue list, so later dialogue appeared
in earlier records when the final report was serialized. The actual requests
had already been serialized before later calls existed. The saved inputs were
restored from the preceding outputs and verified against their original pre-call
hashes. `original-capture.json` preserves the uncorrected record, and the recorder
now snapshots the list. No model calls or generated words were replaced.

The same source window is repeated so viewers can follow earlier takes and later
calls without switching between clips. It is not one causal run through a match.
Playback pacing is authored for readability; it does not represent API timings.
No speech was synthesized. The seven-second source window starts at 50.5 seconds
in `clips/action-sources/england-france-penalties.mp4`. The source-file hash is in
`session.json`. The footage is muted and has no added commentary captions.

## Cost and checks

Six GPT-5.4 Mini calls: three lesson extractions and three subsequent role calls.
Reported token cost: **$0.00531675**, under a $0.10 hard cap, with no unsettled
reservations. No retries, larger models, search, fresh vision, TTS, training, RL,
or Noeri API calls were used. Including the earlier experiment, this session's
total reported model-token spend is $0.0211545.

Free audit and regression checks:

```sh
.venv/bin/python scripts/memory_session.py
.venv/bin/python -m pytest tests/test_memory_session.py tests/test_lesson_memory.py -q
```

To regenerate, choose a new output directory and explicitly enable spending:

```sh
.venv/bin/python scripts/memory_session.py --spend --out runs/lesson-memory/new-session
```

New output needs review before publication. Nothing automatically overwrites the
saved demonstration. Each call input receives observations, not reference labels
or expected commentary. Corrections are supplied only in lesson extraction.

Render the footage locally without model calls:

```sh
ffmpeg -ss 50.5 -i clips/action-sources/england-france-penalties.mp4 \
  -t 7 -an -vf 'scale=640:-2' -c:v libx264 -preset fast -crf 23 \
  -movflags +faststart web/public/learning-calls/penalty.mp4
```

The commentary repo ignores video assets. The public portfolio serves this small
derived clip alongside the buildless UI. No outreach has been sent.
