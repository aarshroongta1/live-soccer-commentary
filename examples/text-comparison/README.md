# Shared text comparison inputs

Seven historical model-perception bundles are frozen here for the text-only
[comparison protocol](../../docs/TEXT_EVALUATION.md). They preserve known mistakes;
they are not gold references or corrected vision. No reference event labels,
expected commentary, manual checks or contact-sheet review notes enter the bundles.

`inputs.json` is the runner manifest. `input-provenance.json` records original
artifact paths, hashes, JSON pointers and transformations. Some original runs are
local ignored files; their selected perception is retained here for reproducibility.
Legacy deterministic identity objects are removed from the free-kick bundle.
The penalty uses saved observer-owned identity proposals and researched context.
No new model, research or voice calls were used to prepare these inputs.

Run a free inventory from the repository root, using a fresh output directory:

```bash
uv run python scripts/compare_text.py \
  --inputs examples/text-comparison/inputs.json \
  --out-dir runs/text-comparison-v1/inventory
```

It reports **7/15 ready**. Buildup, interception, cross-header, miss, corner,
throw-in, foul-decision and offside remain missing. A failed earlier attempt at the
miss is not a usable saved perception result. No input is constructed from labels
to fill these gaps. Historical perception differs across clips; only within-pair
writer comparisons share identical inputs. This cannot measure end-to-end
recognition improvement or causal live behavior.

Explicit paid execution additionally requires `--spend --model MODEL_ID`. It has
a maximum $0.42 token reservation budget for the current seven pairs, eight calls
per arm and no provider retries. Shared-perception preparation is excluded and
requires its own cap if performed. Actual usage and unsettled reservations appear
in each arm's budget report. No paid result is represented by this input manifest.

Paid execution saves `report.json`, per-arm provider responses and drafts, and
`blind/review.json`. Keep `blind/mapping.private.json` hidden while rating text.
Review candidates against source video with the rubric; the runner does not assign
semantic quality or declare an empty transcript acceptable. All 15 cases remain
in the report and blind export, including missing or failed arms.
