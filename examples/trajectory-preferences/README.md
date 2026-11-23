# Commentary preferences in the actual Trajectory product

This follows the user's corrected experiment: show commentary alternatives for a
match moment, let **the user** choose what they like, and retain their reason.
It is separate from the earlier local scheduling demo.

## Verified on November 22, 2026

- Installed `trajectory-sdk==0.10.25` using `uv tool install trajectory-sdk`.
- Authenticated with `tj login --no-browser`; verified organization key `aarsh`.
- Created the `soccer-commentary-editor` agent in that account.
- Confirmed the production-trace list API was available to this organization.
- Created one completed trajectory and recorded a `commentary.preference_task`
  event containing the penalty context, options A/B/C, and their provenance.
- Read the event back and checked equality with the local task.
- Read the native review fields: `feedback_score` and `feedback_text` were null;
  no reviewer was recorded. No human preference was fabricated.
- Billing's cached run-cost list was empty after the writes.

Agent: https://platform.trajectory.ai/aarsh/agents/agt_bc2248eb42e44f629e3175ea9330e705

First review:
https://platform.trajectory.ai/aarsh/agents/agt_bc2248eb42e44f629e3175ea9330e705/production-trajectories/traj_d0fdf09c3a1a43ef9878fbaf23072960

These links require access to the user's account; they are not public demo links.
The first task has no model steps: its options are recorded in an application
event, not fabricated Trajectory model calls. The user's November 23 screenshot
confirmed that this displays raw event JSON under **No task prompt**, together
with a native Pass/Fail verdict, written feedback field, and Submit review button.
This proves the event and native review form exist. It is **not a polished
commentary-options demo**: no native prompt or model response was recorded.

## Proper task spec uploaded November 23

The same screenshot advertised $100 in credits for uploading a first spec. A real
three-task, evaluation-only spec was therefore prepared and uploaded to the
existing agent, without starting a model call, diagnostic, evaluation, or training.

Spec:
https://platform.trajectory.ai/aarsh/agents/agt_bc2248eb42e44f629e3175ea9330e705/benchmarks/bm_7c04347797194a0e97d59a837f353b18

Upload operation: `iop_8f491e8d484d998ddf582ba90dce4655`.
The operation finished `succeeded`: three registered tasks, one ready runtime,
and no failures. `retrieve_readiness` returned `ready: true` with no missing
secrets or pending images. No model run has been started.

`spec/tasks.json` contains readable input messages with each match context,
alternatives A/B/C, and the task-specific editorial question. The harness requests
a recommendation and reason through the actual Trajectory model endpoint **only
when a separate run is explicitly started**. Its automatic `response_contract_only`
reward measures valid output format, not preference alignment, factual correctness,
or commentary quality. Human taste is never synthesized as a gold answer.

All three tasks are `test` tasks. The dedicated Docker build context includes only
the harness and prepared task prompts; it excludes credentials, workspace receipts,
human feedback, the project checkout, video, and evaluation reference labels.
Five offline tests verify valid/invalid choices, truncated output, unsupported
endings, and model errors. No managed execution has yet verified this harness.

```sh
SDK_PYTHON="$(uv tool dir)/trajectory-sdk/bin/python"

# Prepare and validate the manifest locally, without submitting work.
"$SDK_PYTHON" scripts/upload_commentary_spec.py

# Resume status of the upload already submitted; do not create another spec.
tj operations get iop_8f491e8d484d998ddf582ba90dce4655 --json
tj benchmarks readiness bm_7c04347797194a0e97d59a837f353b18 --json

# Offline checks only; no model or API calls in tests.
.venv/bin/python -m pytest tests/test_trajectory_preference_spec.py -q
```

Uploading is distinct from running. The banner's credit grant has not been verified
against an account balance. Once the build finishes, check the advertised credit
grant in the account before any model run. A native model call is needed to display
an actual prompt/response trace; logging another JSON event will not fix the old
prototype's missing conversation.

## Tasks prepared

1. `penalty-reaction.json`: immediate goal reaction; two exact retained lines
   differing in brevity/context, plus silence. Uploaded and verified.
2. `late-save-call.json`: short caller reaction vs longer explanation vs leaving
   the current analyst uninterrupted. Saved model timings, with explicit limits.
3. `after-goal-reaction.json`: repeat the already-called goal, add explanation,
   or stay silent. An explicitly constructed review checkpoint.

The latter two were not uploaded as production events; all three are now included
as tasks in the separate spec above. No preferred answers or reasons have been prefilled. The alternatives
mix exact retained model text with explicitly authored role/abstention options.
They were not generated by Trajectory. Observations are saved VLM reports, not
newly human-verified facts. No source video or reference-label file was uploaded.

## Run

Use an environment containing the SDK. With the installed standalone tool:

```sh
SDK_PYTHON="$(uv tool dir)/trajectory-sdk/bin/python"

# Offline preview; never authenticates or writes remotely.
"$SDK_PYTHON" scripts/trajectory_preferences.py

# Reconnect to the one recorded item and verify its contents.
"$SDK_PYTHON" scripts/trajectory_preferences.py --upload --case penalty-reaction

# Read the user's actual feedback. This does not insert or change any label.
"$SDK_PYTHON" scripts/trajectory_preferences.py --read-feedback

# Once the native review UI has been confirmed, upload the other prepared items.
"$SDK_PYTHON" scripts/trajectory_preferences.py --upload
```

`workspace.json` stores resumable IDs. `feedback-export.json` stores actual native
feedback. Both are gitignored. The script never prints credentials or reviewer
email addresses. It uses the SDK's documented API routes for production review,
whose delayed-feedback endpoints are described in the bundled OpenAPI contract.

Repeated uploads reuse saved trajectories and deduplicated event IDs. Changed
task content must be versioned; it cannot overwrite an existing reviewed task.

## Recording sequence, once the UI is confirmed

Open a review item in Trajectory and inspect the context and commentary options
in its stored task event. Choose an option in the written feedback, with a reason
in your own words; rewrite or reject all options if you prefer. Do not use a
pre-written preference as if it were your own judgment. Save and revisit the item
to confirm the feedback persisted. Then run `--read-feedback` to verify the
saved native feedback against what you entered.

This would demonstrate a real task-specific human feedback record inside
Trajectory. It would **not** demonstrate that a grader learned from it, that a
model improved, or that training ran. If the interface does not display the task
event and editable feedback, this event-only route is not a usable recording demo;
report that limitation rather than substituting a custom UI and claiming it is
Trajectory's labeling product.

No chat completion, managed diagnostic, evaluation, model grader, sandbox,
training, or deployment was started. Those would need a separate supported
workflow and credits. The repository also prohibits model training and RL.

## Primary documentation

- [Production traces](https://docs.trajectory.ai/guides/production-traces)
- [Events and telemetry](https://docs.trajectory.ai/guides/telemetry)
- [Pricing](https://docs.trajectory.ai/pricing): first 10 GB of trace storage free;
  model-based grading and model runs are separately billed.
- [Continual Grading](https://www.trajectory.ai/field-notes/continual-grading):
  relevant motivation, not a feature this experiment has executed.
