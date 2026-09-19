# Review-assisted example

`named-plan.json` is the retained 40-second Betis–Barcelona example.
It includes the generated script, evidence, research and explicit review metadata.

Inputs:

- `identity-reviewed-observations.json`: reviewed visual observations and identities.
- `research-context.json`: reviewed pre-match facts with citations.

This is not untouched end-to-end output. The review linked Koundé to visible
involvement and Ferran Torres to the shot, and removed the unsupported words
“first time” from one generated sentence. Other wording and timestamps were
preserved.

The matching copyrighted source clip is not committed. With your local copy:

```bash
uv run python scripts/render_demo.py \
  --plan examples/recorded-demo/named-plan.json \
  --clip clips/betis-barcelona-1605-1710.mp4 \
  --out runs/demo/example-preview.mp4
```

See [the demo guide](../../docs/DEMO.md) to generate and render your own script.
