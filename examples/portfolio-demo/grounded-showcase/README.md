# Connected commentary with broad visual detail

Published following rejection of both the polished editorial scripts and
the terse six-word-turn preview. This is saved-video playback, not a live demo
or a measured improvement to perception.

The selected output uses connected phrases, ordinary football reactions, and
one broad point from the analyst. Recorded balanced mode now permits nine words
for a caller turn and fourteen for an analyst response instead of six for both.
Goal priority, source-time evidence cutoffs, abstention and no consecutive
analyst turns remain. Prompts discourage touch-by-touch descriptions, exact
shot geometry, body mechanics, inferred tactics and guessed identities.

Unresolved jersey numbers embedded in observer prose are replaced with “player”
in speaker payloads. The original observations remain unchanged in the trace.
This is a runtime input change, not an edit to generated speech. A scripted test
checks that it does not mutate the source observation.

| Clip | Selection | Caller / analyst placed speech |
| --- | --- | --- |
| England–France | Attempt 1 | 50% / 50% |
| Arsenal–Norwich | Attempt 2 | 41% / 59% |
| Argentina–France | Attempt 3 | 42% / 58% |

Selected words are unchanged. Penalty and Arsenal retain their original 20- and
18-second windows. Argentina selects source seconds 23–40 from the 28-second
generation window; the opening eleven seconds contain no generated speech.
All retained cues keep their original position relative to the source footage.
Caption display intervals use measured TTS phrase duration.

Six development attempts are retained with budgets, role outputs and rejected
turns. Arsenal attempt 1 and Argentina attempt 1 repeated unverified shirt
numbers. Argentina attempt 2 used fresh perception but incorrectly called a
save and placed its goal call late; it was rejected. The selected attempts use
saved perception. The fresh-vision failure remains visible and this selection
must not be interpreted as an automatic baseline or recognition benchmark.

This deliberate quality evaluation used `gpt-5.4-2026-03-05` for dialogue and
identity; the rejected fresh observer used `gpt-5.4-mini`. Reported model cost
across all six attempts is $0.1773995. Thirteen `eleven_flash_v2_5` voice requests
used 528 characters, conservatively estimated at $0.132 at $0.25/1K. Combined
estimate: $0.3093995, below the $1 total experiment cap. No retries, external
research, training or fine-tuning. Actual account billing was not queried.

Role shares measure placed TTS phrase duration, including phrase pauses. The
renderer verifies every line has signal, that audio fits the source cues, and
that each voice gets 35–65% of the placed speech. These checks do not measure
subjective commentary quality. No listening review is claimed.

Reproduce with saved source footage and audio present, using new output paths
or moving existing generated media aside first:

```bash
uv run python scripts/render_agent_showcase.py --grounded
```

Rendering makes no API calls. Plans, complete attempts and selection costs are
tracked here. Provider audio stays under `runs/demo/grounded-v{1,2,3}`. Export
hashes, voices, phrase durations and signal checks are in the public
`agent-*-grounded-v1.narration.json` files.
