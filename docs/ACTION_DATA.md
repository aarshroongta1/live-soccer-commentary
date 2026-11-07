# Action development footage

Six video-only excerpts were downloaded on November 7, 2026, totaling 24.5 MB.
The media stays locally in `clips/action-sources/`, outside Git. Sources, publisher
channel IDs, requested upstream sections, formats, decoded durations, sizes, and
SHA-256 hashes are recorded in [the source catalog](../examples/action-sources.json).
This is a development pool, not a held-out benchmark. No model calls were made to
acquire or inspect it, and nothing has been added to the public demo.

| Local source | Publisher / original | Material inspected |
| --- | --- | --- |
| `pl-tackles.mp4` | [Premier League](https://www.youtube.com/watch?v=j4rbHl-3aL0) | Recovery runs, last-ditch tackles, and repeated replay angles |
| `pl-saves.mp4` | [Premier League](https://www.youtube.com/watch?v=7UgMDj43lrY) | Shots, headed follow-ups, saves, corners, and replays |
| `pl-interceptions.webm` | [Premier League](https://www.youtube.com/watch?v=_KCzxPGQGAs) | Candidate interceptions and clearances; rating graphics must be excluded |
| `france-jamaica.webm` | [FOX Soccer](https://www.youtube.com/watch?v=Zs7UXviIH90) | Scoreless women's match: set pieces, goalkeeping, and missed chances |
| `france-jamaica-late.webm` | [FOX Soccer](https://www.youtube.com/watch?v=Zs7UXviIH90) | Late missed chances, challenge replays, and a clearly visible red card |
| `liga-f-routine.mp4` | [DAZN Women's Football](https://www.youtube.com/watch?v=6A6NR4pIPVY) | Continuous Atlético–Real Madrid play: ordinary possession and restarts |

The catalog records original rights holders; no permissive license was supplied
in the downloaded metadata. Public viewing availability does not establish
redistribution or training rights. These files have not been published or used
for training.

## Draft references

Four sparse development cases are in
[`examples/replay-evaluation/actions/`](../examples/replay-evaluation/actions/):

| Case | Local window | Draft labels |
| --- | --- | --- |
| Norwich–Tottenham recovery | `pl-tackles.mp4`, 5–12s | Carry, tackle |
| Villa–Arsenal follow-up | `pl-saves.mp4`, 10–18s | Shot, header, save |
| Liga F buildup | `liga-f-routine.mp4`, 24–39s | Ground pass |
| France–Jamaica card | `france-jamaica-late.webm`, 37.5–42.5s | Red card |

These annotations were drafted from direct local frame inspection, independently
of generated commentary or observer predictions. They are explicitly **unreviewed**,
with approximate timestamps and `action_labels_complete=false`. They are not human
approved ground truth. Do not promote a prompt or claim recognition accuracy from
these drafts. Check actions, event times, and omissions at normal playback and
frame-by-frame before changing their status. Player labels are deliberately absent
where identity has not been independently verified.

Case event timestamps are relative to the case window. `source.start_s` locates
that window within the downloaded local file. It is separate from the catalog's
upstream download start: section downloads can include keyframe preroll. The
scorecard rejects source path/window mismatches and, when both records carry a
hash, different source contents. Legacy outputs without a hash remain explicitly
unverified for media contents. Labels are read only by the separate evaluator.

One Liga F frame around local 15s showed a feed glitch; the selected 24–39s window
avoids it. The interception compilation mixes match footage with rating graphics;
its title alone must not establish an interception label. No interception reference
has been accepted yet. The card window establishes visible red only, not the
booking history, identity, or reason for dismissal.

## Public benchmark priority

Use the [published datasets](PUBLIC_DATA.md) as the main evaluation path. SoccerNet
already supplies action/event annotations; the local excerpts above are supplementary
smoke cases. Matching protected videos require the official access process.

## Reuse and next pass

Verify existing local files without networking:

```bash
uv run python scripts/fetch_action_sources.py
```

To fetch missing files, install `yt-dlp` and FFmpeg, then add `--download-missing`.
Use `--id pl-tackles` to restrict the operation. The script never overwrites an
existing file. A changed upstream encode or mux can change the hash; inspect the
new file and recheck timing before updating the catalog/reference hashes.

A replay request for the save case can be prepared without spending:

```bash
uv run python scripts/replay_commentary.py \
  --clip clips/action-sources/pl-saves.mp4 --start 10 --duration 8 \
  --out runs/replay/villa-save.json
```

After annotation review, run a small, explicitly budgeted baseline and evaluate
saved outputs against these separate references. Add independent matches for
validation/test before prompt optimization. Still missing are accepted interception
labels, verified offsides/yellow cards/penalty decisions, varied blocks, exhaustive
routine-play labels, and enough independent matches for meaningful scores.

[SoccerNet ball-action spotting](https://www.soccer-net.org/tasks/ball-action-spotting)
is a longer-term labeled-data candidate: its documented classes include passes,
drives, headers, high passes, blocks, successful tackles, and free kicks. Protected
videos require account access and NDA acceptance through the
[official data process](https://www.soccer-net.org/data); none were downloaded here.
