# Commit preferences

- Commit completed, working changes as you go.
- Use short, natural commit messages.
- Do not add co-author credits or `Co-Authored-By` trailers.

# Model spend

- Use scripted backends for regression tests; do not make paid calls in tests.
- Use cheaper models for development smoke tests, with short inputs and explicit
  dollar/call limits. Keep retries, fresh web research, and TTS out of routine tests.
- Reserve larger models for deliberate quality evaluations and report model usage.

# Existing models only

- Do not train models, fine-tune weights, or run RL. Use existing models.
- Public datasets are for evaluation and checking prompt/workflow changes. Keep
  reference labels and target commentary outside runtime agent inputs.
