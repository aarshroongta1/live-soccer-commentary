# Light-edit release

Two fresh 28-second perception/dialogue runs supply the additional featured clips.
Inputs contain only source frames, team/kit metadata and an empty research brief.
No old commentary, human target script or evaluation labels enter the agents.
The optional reviewer is disabled. There is no voice or fresh web research.

Four attempts were retained, including the failed stronger Argentina observation.
Combined reported token cost: **$0.27827**, under the **$0.80** experiment cap.
All reservations settled. Speakers use `gpt-5.4-mini`; initial observers use the
same model, follow-up observers use `gpt-5.4-2026-03-05`. The stronger observer was
not consistently better. This selection is not a model comparison benchmark.

| Featured example | Selected attempt | Editorial work |
| --- | --- | --- |
| Arsenal, 18s | `arsenal-2` | Select the first incident, omitting the later compilation segment. Keep all three retained lines verbatim. Move the analyst cue from 17.000s to 11.508s, after its evidence is available. |
| Argentina, 20s | `argentina-1` | Select the last 20 seconds of the original 28s. Remove “’s number 9” from one caller line. Other retained wording and source-relative cues are unchanged. |

Full-context vision originally saw 28 seconds for each. The source cuts and the
retiming are editorial, not runtime improvements. The earlier analyst line in
Argentina and the later goal call in Arsenal remain in their raw outputs. No
football action, identity or explanation was added. The new Arsenal example is a
carry and finish, not the manually described Wilshere team-goal sequence.

`selection.json` retains attempt costs, outputs and failure status. `*-raw-plan`
files preserve generated text, `*-plan` files define the featured render, and
`*-edits` records make every change explicit. Source review was performed by the
coding assistant against frames; this is not an independent human rubric rating.

Re-render without paid calls (choose a new output path; existing files are protected):

```bash
uv run python scripts/render_text_demo.py \
  --plan examples/portfolio-demo/release/arsenal-plan.json \
  --clip clips/demo-sources/arsenal-team-goals.mp4 \
  --out runs/demo/generated-arsenal.mp4 \
  --label 'AI dialogue | source cut + caption retiming | recorded clip'

uv run python scripts/render_text_demo.py \
  --plan examples/portfolio-demo/release/argentina-plan.json \
  --clip clips/demo-sources/argentina-france-dimaria.mp4 \
  --out runs/demo/generated-argentina.mp4 \
  --label 'AI dialogue | source cut + identity phrase removed | recorded clip'
```

See `docs/DEMO_WALKTHROUGH.md` for the short explanation and demonstration order.
