# Did an earlier correction help at a later point in the game?

**No successful transfer of commentary recognition was established.** The earlier
same-incident demonstration was insufficient, and this experiment does not
justify presenting a learning-success demo. The public UI was not redesigned
around an unsupported claim.

## Actual later-event test

All footage comes from the existing action suite and one source file:
`clips/action-sources/england-france-penalties.mp4`.

- Earlier: source 50.5–57.5s, on-screen clock about 53:25. Kane's scored penalty.
- Later: source 103–108s, clock about 83:31. A different penalty, missed.
- The three saved lessons were frozen from earlier calls on the first incident
  before any of these later-event calls. No later outcome or reference label was
  supplied as a runtime input.

The primary test used the same GPT-5.4 Mini model, prompt, schema and exact frame
hashes in each pair. Only the saved memory changed. Two paired samples were
preselected on the later incident, with counterbalanced order, plus one paired
earlier goal control. These are repeated samples of one development incident,
not independent test cases or a representative benchmark.

| Condition | Later penalty | Earlier goal control |
| --- | --- | --- |
| Baseline | False goal in both samples | Goal |
| Frozen lesson memory | False goal in both samples | Goal |

Memory shortened/rephrased the commentary but did not fix perception. Both
conditions misread the player reaction and invented ball-in-net evidence. This
is a transfer failure, not a small successful improvement.

## Workflow hypothesis: check the scoreboard

The earlier score-state lesson motivated a separate, **authored** workflow change:
read the scoreboard literally before publishing a new goal claim. A new goal
requires one team's score to rise by one while the other's remains unchanged.
An unchanged or unreadable scoreboard leaves the claimed goal unconfirmed; it
does not prove a miss or save. Score updates can lag, so this is an opt-in
development safeguard, not an enabled production policy.

The tool read 1–2 at both ends of the later clip and 0–1 to 1–1 in the earlier
control. The deterministic gate therefore holds both late memory drafts and
releases the earlier control. The same late tool reading is reused for both
samples, so these are not two independent verification successes.

The tool also hallucinated ball position, and the additional Mini commentary
calls still said goal. Only its literal numeric reading plus the application
gate prevented publication. **This does not yet produce the correct miss call**
or demonstrate a full commentary-quality pass. Suppressing every outcome would
not count as success; the earlier goal control only checks one positive case.

The earlier control tool response exceeded a 220-character auxiliary gesture
field. Its raw response was retained. After relaxing that field limit, its exact
numeric scores can be parsed without another call or editing model words. The
reviewed gate result is derived offline from those captured fields. The original
`scoreboard/verification.json` was captured before the exception and still says
running; it is not evidence of a completed provider loop. `result.json` records
the recovery and limited interpretation.

## Optional larger verifier

A deliberate quality evaluation tried existing GPT-5.4 as an outcome verifier
on these same frames and score readouts. No training or fine-tuning ran.

The initial later response exhausted its 1,100-token output allowance; that is
an execution failure, not a recognition judgment. Both cases were then run with
a fixed 2,400-token allowance. The later incident was still called a goal. Both
earlier control responses called the genuine goal. All attempts are retained;
there were no automatic retries or further attempts to obtain a winning answer.

These verifier calls changed model and workflow. Even if they had succeeded,
they would not establish that memory alone improved the writer.

## What the result supports

- Earlier lessons can be persisted and supplied to later incidents, but that
  mechanism alone did not demonstrate improved recognition here.
- A narrowly checked application policy can stop a false assertion while
  preserving one positive control. It has latency/coverage costs that need
  evaluation, particularly when score overlays update slowly.
- Structural consistency and multiple agreeing models remain insufficient
  evidence that the observed event is true.
- No new learning-success page was published from these results. The next
  necessary improvement is reliable outcome verification, not presentation.

Review was by a coding agent inspecting source frames, not independent human
ratings. Full clip context was available; these are offline calls, not measured
causal live latency. Original action-suite labels remain unreviewed. No target
commentary, reference JSON, hosted search, audio or research result was passed
into the runtime calls.

## Costs and reproduction

| Model | Attempts | Reported token cost |
| --- | ---: | ---: |
| GPT-5.4 Mini | 10 | $0.04164435 |
| GPT-5.4 (2026-03-05) | 4 | $0.091008 |
| Total, including failures | 14 | **$0.13265235** |

Per-phase hard caps were $0.20, $0.15, $0.15 and $0.15, below the $1 authorization.
No unsettled reservations remain. Including the earlier memory experiments,
reported model-token cost in this conversation is $0.15380685. No TTS, training,
new source acquisition, search API, or Noeri API calls were used.

Free audits and tests:

```sh
.venv/bin/python scripts/transfer_memory.py
.venv/bin/python scripts/verify_transfer.py \
  --source examples/lesson-memory/transfer/prompt-only/transfer.json
.venv/bin/python scripts/escalate_transfer.py
.venv/bin/python -m pytest tests/test_transfer_memory.py -q
```

Paid execution requires `--spend` and a new `--out` directory. Nothing
automatically overwrites saved artifacts or promotes a candidate to the public
demo. Each directory retains the raw provider responses and its run report.
`result.json` is the reviewed synthesis, including all phases and qualifications.
