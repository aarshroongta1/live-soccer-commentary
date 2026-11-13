# Edited demo commentary

The user rejected the original narrated showcase for long blank stretches and
weak commentary, then explicitly authorized rewriting the commentary. These
three scripts are editorial rewrites by the coding assistant against the local
footage, with AI caller/analyst voices. They are **not unedited agent output or a
measured improvement to the agent system**. Original plans, traces and evaluation
results are retained unchanged in the parent directories.

| Clip | Duration | Change | Longest measured pause | Speech occupancy |
| --- | --- | --- | --- | --- |
| England–France | 19s | Covers the foul, award and run-up; reuses the original Kane goal recording; new analysis; final replay second trimmed. | 0.61s | 91% |
| Arsenal–Norwich | 18s | Calls the quick passes and continuing run, then explains the move over the replay. | 1.15s | 88% |
| Argentina–France | 20s | Calls the counterattack from the opening, wide runner, final pass and goal, then explains the space and finish. | 1.18s | 84% |

The original voice exports had about 10.3/20, 11.0/18 and 6.6/20 seconds of placed
speech respectively; Argentina's first speech cue was 7.581s. The old scripts
were sparse captions, and the release checks verified sound presence rather than
useful coverage. New checks measure actual exported silence at -40dB, counting
pauses of at least 0.2s, and reject pauses over 1.6s or occupancy below 72%.
Occupancy is a pacing check, not proof of commentary or voice quality. Source
frames were reviewed at half-second intervals. This environment cannot listen
to audio; no subjective listening review is claimed.

Scripts and source cues are in the three `*-plan.json` files. New observations
are editorial source-review notes, not model perception results. Names in the
edited script can use visible player close-ups from this recorded sequence;
this is not evidence of causal player identification. No reference scripts or
labels were fed to runtime agents, and no fresh observer or dialogue run was made.

Speech uses the configured ElevenLabs caller/analyst IDs and `eleven_flash_v2_5`.
Consecutive caller phrases were generated as one performance and split using
provider character alignment. Six requests submitted 834 characters and reported
414 character-cost credits, with no retries. The estimate at the conservative
$0.25 per 1K rate was $0.2085, within the $0.75 experiment cap. Actual account
billing was not read. Local pace adjustments range from 0.85x to 1.25x; no speech
is cut off to fit. The successful original Kane goal performance is reused intact.

Reproduce from the saved, ignored speech and source footage, with new output
paths or after moving previous generated outputs aside:

```bash
uv run python scripts/render_edited_showcase.py
```

Saved provider audio/alignment/audits live in `runs/demo/commentary-edit-v2/audio`.
The script rebuilds clean captioned base videos, mixes saved speech, checks that
every line has audible signal, and enforces the pause/coverage gate. It makes no
API calls and refuses to overwrite media. A subtitles-capable bundled FFmpeg is
used for caption rendering. Public `.narration.json` sidecars record wording,
voice IDs, input/output hashes, local pacing, original synthesis audit and measured
coverage. Media is git-ignored; the scripts and release evidence are tracked.
