# Public datasets for recognition and commentary

Checked November 7, 2026. Established datasets should be the primary evaluation
source. The manually drafted cases in `examples/replay-evaluation/actions/` remain
small development smoke cases, not a replacement for published annotations.

| Dataset | Relevant contents | Access checked |
| --- | --- | --- |
| [SoccerNet Ball Action Spotting](https://www.soccer-net.org/tasks/ball-action-spotting) | Seven annotated games, 12 action classes: pass, drive, header, high pass, out, cross, throw-in, shot, player block, successful tackle, free kick, goal | [SN-BAS-2024](https://huggingface.co/datasets/SoccerNet/SN-BAS-2024) currently requires a Hugging Face access request, NDA acceptance, and manual approval. Its card reports 19.3 GB. The 2025 package is also gated. |
| [SoccerNet Action Spotting](https://www.soccer-net.org/tasks/action-spotting) | 500 games, 17 event classes, including shots on/off target, fouls, cards, offsides, corners, substitutions, and goals | [SN-Labels](https://huggingface.co/datasets/SoccerNet/SN-Labels/tree/main) exposes annotations anonymously; matching broadcast videos are protected. |
| [SoccerNet-Echoes](https://huggingface.co/datasets/SoccerNet/SN-echoes) | Timestamped broadcast-commentary transcripts and English translations, indexed by game and half | Public text dataset, CC BY 4.0 according to its card. Whisper/translation output, with transcription errors and hallucinations; not human-verified factual ground truth. |

The [official development kit](https://github.com/SoccerNet/sn-spotting) includes
benchmarks and evaluation code. Its older password-based download instructions
predate the [current Hugging Face access process](https://www.soccer-net.org/data).
Do not equate public dataset documentation with anonymous access to all videos.

## Verified public annotation sample

Downloaded the official Chelsea–Burnley 2015 `Labels-v2.json` anonymously:
37,609 bytes containing 190 event annotations across 15 labels in this match.
Examples of the available classes include fouls, offsides, clearances, throws,
shots, and a red card. This is the 17-class action-spotting annotation format,
not the dense 12-class ball-action format; it does not label routine passes.

The raw file is local under `runs/datasets/soccernet/chelsea-burnley-2015/` and
ignored by Git. Its URL, hash, class counts, and inspection status are recorded in
[the sample manifest](../examples/soccernet-label-sample.json). No matching video
has been downloaded. No recognition accuracy can be scored from labels alone.

The Hugging Face combined dataset viewer currently reports a schema-cast error
for SN-Labels; this does not prevent individual JSON files from downloading.

## Integration order

1. Use Ball Action Spotting for passes, carries/drives, headers, crosses, tackles,
   and blocks. Use Action Spotting for decisions and match events. These cover
   different targets; do not assume either contains every action in our vocabulary.
2. Write import adapters preserving dataset release, match, split, half, original
   labels, timestamps, visibility, and mapping decisions. In `Labels-v2.json`,
   `position` is milliseconds within a half; half identity comes from `gameTime`.
   Apply the actual source-video and selected-clip offsets, rather than copying
   match-clock timestamps into a short highlight clip.
3. Preserve official splits. Select a small development subset of matched videos
   after authorized access; do not download the entire collection on this disk.
   Labels stay exclusively in the evaluator, with no future events in agent inputs.
4. Evaluate using the published metrics as well as commentary latency and safety
   diagnostics. Action Spotting has tight temporal mAP; Ball Action Spotting uses
   mAP at a one-second tolerance. Our existing kind/time diagnostic is not a
   substitute for those benchmark protocols.
5. Add Echoes as a noisy commentary reference after aligning the same game/half.
   Review errors and judge semantic agreement, rather than exact word matching.
   Do not use the target broadcast transcript as an observer input.

The first `Labels-v2.json` importer is implemented. The dense ball-action adapter,
Echoes import, official mAP integration, and benchmark runs remain pending. The
immediate prerequisite for visual evaluation is access to matching videos, not more
hand-labeling of unrelated highlight compilations. No NDA was accepted or account
request submitted on the user's behalf.

## Use the event-label importer

Inspect the downloaded public sample with no video, network, or model calls:

```bash
uv run python scripts/import_soccernet.py \
  --labels runs/datasets/soccernet/chelsea-burnley-2015/Labels-v2.json --inspect
```

For an authorized, matching half video, import a short evaluation window:

```bash
uv run python scripts/import_soccernet.py \
  --labels /path/to/Labels-v2.json --half 1 --upstream-split valid \
  --clip /path/to/1_224p.mkv --start 600 --duration 20 \
  --video-half-offset 0 --case-id match-half1-window600 \
  --label-origin 'publisher URL or versioned label-file locator' \
  --usage-terms 'applicable terms for this source video' \
  --out runs/references/match-half1-window600.json
```

Supply the match's actual published split, not an arbitrary partition. The importer
verifies that assignment offline against the official SoccerNet package lists pinned
at [commit 95d76e2](https://github.com/SoccerNet/SoccerNet/tree/95d76e2b46b43738dbf18f2d7ab3d7283d5aeff5/SoccerNet/data).
The package's [list-selection code](https://github.com/SoccerNet/SoccerNet/blob/main/SoccerNet/utils.py)
selects `SoccerNetGamesTrain/Valid/Test/Challenge.json` for event spotting; the
separate ball-action lists are not interchangeable with these partitions.

The bundled snapshot contains 300 development/train, 100 validation, 100 test, and
50 challenge match identities. The downloaded Chelsea–Burnley 2015 sample belongs
to `train`/development. `--inspect` reports the official assignment. A contradictory
claim, a challenge match, or an unknown identity fails before the CLI reads the
video. Unknown matches require checking the dataset/release and updating the pinned
official manifest; there is no unverified-assignment override.

Successful imports record `split_assignment_verified=true`, the official repository,
revision, and bundled-manifest hash. Original-list URLs and byte hashes are retained
in `src/commentary/replay/data/soccernet-events-v2-splits.json`. This verifies split
membership, not video alignment, annotation accuracy, or a user's declared label
origin. Imported windows therefore remain unreviewed for alignment.

The mapping is `train` to development, `valid` to validation, and `test` to test;
these are dataset partition names, not instructions to train a model. Evaluation
manifest validation also rejects a match crossing splits.

`--video-half-offset` is the timestamp in the supplied source video corresponding
to timestamp zero of the original SoccerNet half video. Use zero for the unchanged
half file. It is **not necessarily kickoff time**. The conversion is:

`clip_time = position_ms / 1000 + video_half_offset - selected_clip_start`

The importer selects only the requested half and the interval from start inclusive
to end exclusive. It preserves the raw label, half, position, visibility, team,
annotation index, origin, and label hash in evaluator-only `reference_metadata`.
Source video bytes are hashed and the output refuses to overwrite an existing file.

Events marked not shown, events with unspecified visibility, and unmapped labels
are excluded with an audit reason. Penalty labels remain unmapped because award
versus kick is ambiguous; ball-out is not silently converted to a restart. Shot
outcomes and second-yellow detail remain in the raw annotation even when mapped
to the broader project vocabulary. These event labels do not annotate all passes,
saves, or other actions, so action precision remains unscored for incomplete labels.

Imported windows remain unreviewed until local video alignment has been checked.
The existing replay evaluator can read the exported JSON, but its output remains
project diagnostics rather than official SoccerNet mAP. No labels are passed to
observer or commentary agents, and no weights are updated.
