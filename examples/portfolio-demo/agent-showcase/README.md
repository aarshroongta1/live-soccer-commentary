# Generated dialogue showcase

The public demo replaces the editorial rewrites with selected role-agent output.
No generated words were rewritten, deleted, reassigned to another speaker or
patched after watching the footage. Nine attempts remain here, including failures
to give the analyst a turn. Selection is development work, not a benchmark.

| Clip | Selected attempt | Caller / analyst speech | Footage |
| --- | --- | --- | --- |
| England–France | 3 | 39% / 61% | Original 20-second window |
| Arsenal–Norwich | 3 | 59% / 41% | Original 18-second window |
| Argentina–France | 2 | 47% / 53% | Last 20 seconds of the 28-second window |

Raw plans and all attempt traces preserve exact outputs, rejected turns, model
usage and evidence. Argentina's opening 8 seconds contain no dialogue; removing
them changes only the source window and offsets. No retained speech cue is moved
relative to its source footage. The full-context observer originally saw longer
windows for Arsenal and Argentina. This is saved-perception dialogue generation,
not a new end-to-end perception run or a causal recognition evaluation.

The saved observations still have recognition weaknesses. In particular, the
Argentina caller says “number 9” from the saved observer's reading. That claim is
not independently verified; it is intentionally preserved in the generated text.
The short commentary can be awkward or superficial. The UI describes selection
and recording rather than implying a live or flawless system.

Attempt 1 used the existing scheduler and often lost analyst responses to word
budget rejection. Attempt 2 limited caller turns to six words and analyst turns
to twelve. Attempt 3 uses six-word turns for either role and prevents consecutive
analyst turns. Confirmed goals retain caller priority. Evidence and timing checks
still apply, and abstention is permitted. The opt-in `--balanced-speakers` mode is
covered by a scripted regression test. It provides opportunities, not a guarantee
of balanced audio; exported phrase durations are checked separately.

All nine runs used `gpt-5.4-mini` for dialogue/identity with reused vision, no
reviewer, no fresh research and no retries. Reported model cost: $0.079437.
Fourteen TTS requests used `eleven_flash_v2_5`, the configured two voices, and
366 characters: conservative estimate $0.0915 at $0.25/1K. Combined estimate
$0.170937, within the total $0.915 reservation cap. Provider character-cost
headers are saved; account billing was not queried.

The render keeps original pauses, aligns captions with measured TTS phrases,
and only speeds speech when needed to fit its next source cue (maximum 1.25x).
Role shares count placed TTS phrase durations, including phrase pauses; they are
not word counts or a diarization measurement. No subjective listening review is
claimed. Per-line exported signal and clip duration checks pass.

Reproduce locally with source footage and saved voice manifests present:

```bash
uv run python scripts/render_agent_showcase.py
```

This makes no paid calls and refuses to overwrite outputs. Selected plans are
tracked here, saved audio remains under `runs/demo/authentic-v{2,3}`, and public
`agent-*-balanced-v1.narration.json` reports retain input/output hashes, voice IDs,
phrase durations and role shares.
