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

The import adapters and benchmark runs are not implemented yet. The immediate
prerequisite for visual evaluation is access to the matching videos, not more
hand-labeling of unrelated highlight compilations. No NDA was accepted or account
request submitted on the user's behalf.
