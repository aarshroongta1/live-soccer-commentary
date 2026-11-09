# Commit preferences

- Commit completed, working changes as you go.
- Use short, natural commit messages.
- Do not add co-author credits or `Co-Authored-By` trailers.

# Model spend

- Paid experiments with a total cap of $1 or less are authorized; ask before any
  experiment above $1. Do not split a larger experiment to bypass this limit.
- Use scripted backends for regression tests; do not make paid calls in tests.
- Use cheaper models for development smoke tests, with short inputs and explicit
  dollar/call limits. Keep retries, fresh web research, and TTS out of routine tests.
- Reserve larger models for deliberate quality evaluations and report model usage.

# Existing models only

- Do not train models, fine-tune weights, or run RL. Use existing models.
- Public datasets are for evaluation and checking prompt/workflow changes. Keep
  reference labels and target commentary outside runtime agent inputs.

# Evaluation scope

- Keep evaluation to 12–15 short clips covering different actions/events; the
  selected set is examples/action-suite.json (15 clips).
- Prioritize recognition and commentary quality on those clips over more dataset
  infrastructure or broad benchmark acquisition. The selected footage is available;
  add sources only to replace an actual coverage gap.
