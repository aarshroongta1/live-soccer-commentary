# Watch, correct, test on another clip

Connected owner workflow: http://127.0.0.1:3010/demo/editor

The user watches the existing Ballard footage with its generated two-voice audio,
then explicitly keeps, rewrites, or rejects the analyst line and gives a reason.
The local worker imports that saved proposal with its provenance, writes the
actual decision into Trajectory's native feedback fields, and reads it back before
showing a saved confirmation. It does not invent a human judgment or reclassify
the original saved generation as a new Trajectory model call.

After a real review is saved, the user may generate two analyst takes on different
footage (Messi vs Bayern). Both use GPT-5.4 Mini, temperature 0, one 256-token output
cap, and the identical six image parts, caller context, reviewed identity, and a shared
three-fact UEFA background pool. Only one receives the
actual human feedback read back from Trajectory. Frames are sampled at
7.5/9/10.5/12/13.5/15 seconds from the uncaptioned source video. No target
commentary, reference labels, or future frames enter either request.

Both unchanged outputs appear as randomly ordered takes A/B. The human can prefer
either, call them equal, or reject both, with a reason. This judgment is written
and verified in Trajectory before the mapping is revealed. A non-improvement is
retained. This is a small within-user comparison, not an independent blind study
or a generalization benchmark.

## What learning means here

The agent retains explicit human feedback in its context for a later task.
No weights are updated, no RL or model training runs, and no native Continual
Grading feature is claimed. Trajectory supplies actual trace/feedback storage and
the inference endpoint. The surrounding application owns the feedback-to-context
connection. Capturing feedback alone is not evidence of improvement; the later
comparison needs the user's actual judgment.

The owner saved an actual Keep decision, explaining that career and team-performance
facts enriched the commentary. This was verified in Trajectory. Both subsequent
visual model calls completed, unchanged and without retries; the second human
judgment is pending. No owner feedback was fabricated. Tests use an isolated fake
client and temporary files.

The background pool was prepared after that first review to provide an opportunity
to use the stated preference. It is identical for both runs. This is an adaptive
development example, not a preregistered held-out evaluation. The UEFA page was
published before the match but marked updated after it; selected facts concern
pre-match context, and a historical archived copy was not checked.

Actual visual-run usage: baseline 1,552 input / 26 output tokens; with feedback
1,668 input / 29 output tokens. At the discovered uncached list rates, their
combined token-cost estimate is $0.00346125. That is an estimate, not a settled
account invoice.

## Run

The existing `tj login` must authenticate the `aarsh` organization. The local
Next route invokes the SDK tool's Python; it never sends credentials to a browser.

```sh
# Offline asset preparation; reuses local source footage and the saved voice run.
.venv/bin/python scripts/prepare_commentary_editor.py

# Run from web/; the connected writes are enabled only in development on loopback.
npm run dev -- --port 3010

# From repo root: offline regression tests, no API or model calls.
.venv/bin/python -m pytest tests/test_commentary_editor.py tests/test_trajectory_preference_spec.py -q
```

Session and exact model responses are retained in the gitignored
`runs/commentary-editor/session.json`. All frames are retained in that directory.
The source and input hashes remain inspectable. Browser-voice previews of edits
and comparison takes make no TTS API calls; the first proposal uses the actual
existing synthesized audio in the recorded video.

Only an explicit comparison triggers new paid work: two bounded calls, six
512×288 JPEGs per call, no tools and no provider retries. A failed or uncertain
comparison is recorded and cannot automatically spend again. Reopening an
existing completed comparison reuses the saved results. The planned token budget
is comfortably below the repository's $1 experiment cap; this is an application
bound on workload, not a provider billing limit.

The public deployment is an interactive preview with browser-local decisions.
Its API cannot use the owner's credentials or start model calls. It does not
pretend those preview decisions were sent to Trajectory. The actual connected
recording should use the loopback URL above. Public sharing of a completed
learning result should follow the user's actual review and comparison, not a
prepared fictional review.

## Share the recorded result

After the user saves the A/B judgment, run `scripts/export_commentary_editor.py`
with the installed SDK Python. It verifies the original human review, the
comparison judgment, and both model responses against Trajectory, and checks
retained frame hashes before exporting a public replay and evidence JSON. The
public app replays that owner example; visitor judgments stay browser-local and
never modify the owner’s Trajectory records. It cannot export an unjudged pair.
