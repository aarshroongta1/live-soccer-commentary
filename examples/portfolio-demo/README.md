# Selected automatic portfolio demo

`penalty-plan.json` contains exact generated words and source cues for the
20-second England–France sequence. `penalty-trace.json` retains source metadata,
research, identity lookup decisions, role actions, rejected turns and budget.
The web summary is `web/data/portfolio-demo.json`.

`selection.json` preserves both bounded attempts and the choice of attempt two.
The first optional reviewer rejected a valid short goal call; the second attempt
ran without that reviewer. An early oversized line was rejected in the second
attempt, leaving a caller penalty/run-up/goal sequence and an analyst response.
No model-written words were manually edited. These are selected development
outputs on reused perception, not an automatic baseline quality benchmark.

Saved vision came from the prior continuity candidate, with its script excluded
from runtime input. The original full-context vision preparation saw a longer
window; only saved observations inside the selected 20 seconds are used here. Resolved names were cleared and re-proposed using the saved,
cited England squad. No new vision, research or voice generation was performed.
Total reported token cost for both attempts: $0.02666745, under a $0.24 cap.

Render a silent captioned clip with scripts/render_text_demo.py. Local MP4s are
ignored; plans, transcripts, traces and render tooling are retained. See
[the portfolio guide](../../docs/PORTFOLIO.md) and
[the case study](../../docs/PORTFOLIO_CASE_STUDY.md).

## Matched before/after iteration

`before-plan.json` and `before-trace.json` retain attempt one. `iteration.json`
records matched-input hashes, the reviewer intervention, observed role/cost changes
and limitations. The local player can switch both source-timed captioned versions.
This is a focused development case following the 15-clip pilot, not the earliest
system baseline or a claim that every output dimension improved. No additional
model calls were made to package the comparison.

## Fresh vision and light edits

The additional featured clips are in [release](release/README.md), with fresh
perception, raw outputs, every attempted run, exact edit records and no added
football facts. These replace the heavily rewritten showcases in the player.
