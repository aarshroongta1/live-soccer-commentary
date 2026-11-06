# Edited demo showcase

Five short, recorded examples span the World Cup, Champions League, Premier
League, and La Liga. These scripts are **manually reviewed and edited showcase
commentary**, not untouched automatic system output. The player and burned-in
captions disclose this.

| Plan | Moment | Length |
| --- | --- | --- |
| `argentina-france.json` | Di María's counterattack in the 2022 final | 28 s |
| `barcelona-bayern.json` | Messi past Boateng and over Neuer, 2015 | 34 s |
| `arsenal-norwich.json` | Wilshere's one-touch team goal, 2013 | 28 s |
| `portugal-spain.json` | Ronaldo's late free-kick equalizer, 2018 | 32 s |
| `betis-barcelona.json` | Ferran Torres's equalizer, 2025 | 32 s |

Each plan records its source URL, downloaded window, selected source window,
reviewed observations, exact script, and editorial changes. Observation and
line timestamps are relative to the selected window. Live calls cover the
action; longer explanation follows the finish or accompanies a replay.
The source URLs and timestamps identify the footage, rather than claiming
that player identification was performed automatically.

All five use the same ElevenLabs caller and analyst, preserving each voice's
saved settings without a forced speed. The caller uses fuller phrases for
the big moments. The analyst has a turn before the decisive action and a
substantial response afterward, with additional exchanges during replays.

In the current render, the analyst accounts for 43–53% of audible speech:
45% for Di María, 53% for Messi, 53% for Arsenal, 46% for Ronaldo, and 43%
for Betis. These are measured durations, rather than line-count estimates.

Audio durations were
measured before rendering to check for delayed, overlapping, or truncated
lines. The browser catalog at `web/data/demo-clips.json` contains these exact
words and cues. A test checks that the two artifacts stay aligned.

## Reproduce a render

Source footage and generated media are local and git-ignored. Put the matching
source file under `clips/demo-sources/`; its basename must match the plan.
The plan's review metadata identifies the download window. For example, the
Di María source is seconds 2188–2228 of its upload; the plan selects its final
28 seconds.

Generate paid speech only after reviewing the plan:

```bash
uv run python scripts/synthesize_demo.py \
  --plan examples/demo-showcase/argentina-france.json \
  --out-dir runs/demo/new-argentina-audio --spend
```

Then render using the saved speech:

```bash
uv run python scripts/render_demo.py \
  --plan examples/demo-showcase/argentina-france.json \
  --clip clips/demo-sources/argentina-france-dimaria.mp4 \
  --manifest runs/demo/new-argentina-audio/manifest.json \
  --caller-voice "ElevenLabs caller" --analyst-voice "ElevenLabs analyst" \
  --out runs/demo/new-argentina-preview.mp4
```

Without `--manifest`, the renderer uses local macOS voices for a draft. Its
measured report and ASS captions are saved beside the MP4. Check the report
before copying a render to the catalog's `web/public/soccer-commentary/` path.
Keep the portfolio catalog synchronized when publishing a change.
