# Learning from the last call

Public demo: https://www.aarshroongta.com/demo/noeri

One reviewed past failure becomes a model-written, persistent instruction for
three different clips. This is a small application-level memory experiment using
existing models, inspired by Noeri's work on agents that improve from experience.
No Noeri API, weight updates, fine-tuning, RL, search, fresh vision, or TTS ran.

## Source and protocol

The lesson comes from the rejected England–France analyst call in
`examples/evaluation-results/researched-handoff-20261107.json`. The analyst
claimed a keeper touch from a weak audit tag. A retained coding-agent correction
identifies that unsupported detail. This is not independent human feedback.

GPT-5.4 Mini distilled `lesson.json` from that failure and correction. It contains
no match names or transfer answers. The file was saved before transfer calls.
The same lesson, unchanged after the initial pilot, was used for all later calls.

The three transfer cases were preselected as `goal`, `dribble`, and `save`, from
the existing 15-clip development suite. They use different source files from the
lesson source. The writer receives only their saved observations; reference
labels, expected commentary, research and earlier calls are excluded. These are
full-context offline text reactions, not causal live playback or a vision test.

The comparison uses a compact experimental writer, not the current production
role prompts. The same model, base instructions, evidence, output schema, and
output allowance are used in both arms. Only the memory instructions differ.
There is one sample per arm, counterbalanced call order, and no automatic retries.
Every planned case and output is retained, including the regression.

## Results

| Case | Without memory | With memory | Review |
| --- | --- | --- | --- |
| Save | Adds “at the near post,” absent from the supplied observations; 38 words | Retains the save, removes the unsupported location, qualifies the follow-up; 31 words | Favorable single example |
| Crowded goalmouth | Narrates uncertain delivery/contact as established; 32 words | Qualifies the final touch, but still cites uncertainty and exceeds the word target; 37 words | Mixed |
| Dribble and finish | Compact supported outcome; 23 words | Adds unnecessary uncertainty and repetitive analysis; 50 words | Regression |

Review was performed by a coding agent against the supplied observations, not
independent pixels. Saved perception can itself be false. Existing citation IDs
are not proof of factual support. The save example is not a repeatable accuracy
claim; the broad lesson also causes harm. Two memory scripts meet the 35-word
target; one does not. Words are counted with Python `str.split()`, including any
inline citation strings actually emitted by the model. Text is not rewritten.

The first pilot reused a legacy long-script prompt with conflicting short-output
instructions. Three of six writer responses failed parsing. `pilot.json` and
`pilot-responses/` retain that attempt. After fixing the base prompt for both
arms, the comparison reused the original lesson and made six fresh writer calls.
This was a protocol correction, not a reroll of one unfavorable arm.

Total reported model token cost: **$0.01583775 across 13 calls**, including lesson
extraction and the failed pilot. The pilot's $0.01025775 settled spend plus the
second run's $0.18 hard cap remained below the overall $0.20 ceiling. No unsettled
reservations remain. All calls used GPT-5.4 Mini, reasoning none, no retries,
with at most 400 lesson tokens or 600 writer output tokens.

## Reproduce

Free audit:

```sh
.venv/bin/python scripts/lesson_experiment.py
.venv/bin/python -m pytest tests/test_lesson_memory.py -q
```

A new paid comparison with the frozen lesson (use a new output directory):

```sh
.venv/bin/python scripts/lesson_experiment.py --spend --max-usd 0.10 \
  --lesson "$PWD/examples/lesson-memory/lesson.json" \
  --out runs/lesson-memory/new-comparison
```

Omit `--lesson` to distill a fresh lesson first. New results require review; the
script never overwrites or automatically promotes them to the published demo.
`result.json` contains the reviewed comparison, source/input/prompt/lesson hashes,
actual outputs, and usage. `response-*.json` retains the six raw provider results.

## Website

The canonical buildless UI lives in `web/public/learning-calls/`. Mirror that
directory into `../personal-website/public/learning-calls/`; `/demo/noeri` is a
Vercel rewrite. The static page shows saved results without API calls. Its
main view shows the earlier mistake, the exact saved lesson, and the next caller
line with and without it. Full caller/analyst outputs and the other two cases
are under one disclosure. The experiment persisted the lesson to disk and
actually supplied it to the model before generating calls.

No new broadcast media is published. All generated words, the full lesson, and
the failed pilot remain inspectable. No outreach message has been sent.
