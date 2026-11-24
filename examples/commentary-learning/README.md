# Native commentary choice learning

Current status: the 20-second review batch was rejected by the user for generic
options and overly long footage per line. It is archived in
`runs/commentary-learning/rejected-20s-batch/` and will not train anything. The
local page now shows three **8-second pilot moments**, using retained normal
generated commentary plus two explicitly assistant-authored editorial alternatives.
No preferred labels are assigned. Reasons are optional. `prepare_commentary_pilot.py`
builds this set; the uploader explicitly refuses a pilot catalog. The full
train/test set must be rebuilt around short single-line moments after quality
feedback. The original experiment plan below is superseded where it refers to
20-second review chunks.


Owner review: http://127.0.0.1:3010/demo/trajectory-training

This experiment was explicitly authorized by Aarsh after the original project
no-training rule: ten 20-second training clips (3m20s) and three disjoint 20-second
test clips. The first test, Argentina vs France, is frozen as the recording target.
The test choices are scored but never used for training or checkpoint selection.
These tiny samples test the workflow, not general improvement.

GPT-5.4 Mini generated three commentary options and a scene observation from five
causal frames per clip. Silence is a fourth option. Candidate order is shuffled
once and frozen before reviews. Source audio is removed, and frames after the
15s decision point are not passed to the model. The reviewer may watch the full
20s; their taste therefore includes hindsight that the selector does not see.
Generated observations and alternatives can contain errors. Picking silence is
valid if none of the lines should air; do not invent a preference to fill labels.

The human chooses a line and supplies a reason. Each actual choice is saved to
Trajectory's native feedback, read back and verified, then retained locally.
The choice is the supervised label; the free-text reason is an audit record,
not automatically translated into a reward. No labels are supplied to the actor.
The original generated line is retained independently for the requested normal
commentary comparison.

The next step packages those real labels as a Trajectory decision spec. Qwen 3.5
4B is the intended trainable selector. A managed baseline checks the harness,
then one small RLCD run updates its choice probabilities using the native Brier
loss workflow. A final-checkpoint eval uses exactly the baseline's three test
cases and input options. Compare both native selector accuracy and the target
clip's selected commentary. Also retain the original generated commentary.
There is no requirement that training improve the result; report unchanged or
worse results. Do not search checkpoints or rerun to obtain a favorable clip.

The generator and scene observations remain fixed. This trains a text-based
editorial selector, not the visual perception model, commentary generator, or
TTS. It supports a clear recording across the player, the Trajectory training
run, and native eval results. It does not claim Continual Grading access.

## Stages

1. Offline prep: `.venv/bin/python scripts/prepare_commentary_learning.py`.
2. Bounded, once-only candidates: run `scripts/generate_commentary_choices.py`
   using the installed Trajectory SDK Python. Existing completed cases are reused;
   incomplete cases require inspection, not automatic resubmission.
3. Review all 13 cases in the connected page. No model calls are made on save.
4. Run `scripts/upload_commentary_learning.py` with SDK Python. It requires all
   actual labels and verifies native feedback before packaging the train/test split.
5. Check decision-spec support and model options, run a three-task baseline, then
   launch one bounded training run and compare its final checkpoint. No run starts
   merely because this page is opened or reviews are saved.

## Budget and planned limits

Existing account credits are available. The project authorizes at most $1 per
experiment; no higher spend is authorized. Initial choice generation: 13 calls,
max 500 output tokens each, no retries. Intended native run: 10 optimizer steps,
batch size 2, group size 1, 20 training decisions; no intermediate validation;
three baseline plus three final-checkpoint test decisions. Disable reasoning,
limit turns to one and generated tokens to 128; use small sandboxes. Validate
those options with the actual decision spec before launch. Stop and inspect any
runtime or cost anomaly; do not resubmit an uncertain run.

Published rates for Qwen 3.5 4B are $0.3168/M input, $0.9696/M sampled and
$1.2864/M trained tokens. With short text choice inputs, planned token spend is
well below $1 including the candidate-generation calls; sandbox costs are extra
($0.20/CPU-hour plus $0.04/GiB-hour). These are planning estimates, not a provider
hard spending cap or settled invoice. Source: https://docs.trajectory.ai/pricing .

Raw source catalog, images, exact responses, and real reviews live under the
ignored `runs/commentary-learning/`. Public reviewer data contains no credentials
or private human labels. Only loopback development requests may write native
feedback. Never test saves by impersonating a human reviewer.
