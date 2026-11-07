# Focused recognition pass

November 7, 2026. Existing card, save, and tackle footage from the fixed
[15-clip set](ACTION_SUITE.md), with no training, extra downloads, research, or TTS.
Eight short development runs cost **$0.08738345** in reported model tokens against
a $0.10 task ceiling, with **zero outstanding reservations**. Ordinary tests were
fully offline. Observer: `gpt-5.4-mini`; caller/analyst: `gpt-6-luna`; retries: zero.

| Case | Earlier behavior | Latest useful result | Still wrong |
| --- | --- | --- | --- |
| Red card | Yellow card; scene labeled celebration | Red card, scene play. Caller: “A red card is shown, and protests erupt around the referee.” | Analyst abstained; reason for dismissal is unavailable. |
| Save | Important baseline batch budget-stopped; first new pass misattributed an earlier save/clearance | Inspection detected keeper contact at clip 5.5s. Caller and analyst both spoke. | Calls the buildup a low cross; misses the initial shot/woodwork and rebound header. Commentary is therefore not ready to showcase. |
| Tackle | Important baseline batch budget-stopped; first new pass confidently invented a shot | Latest focused window describes a challenge/poked-loose ball as uncertain and no longer asserts a shot/save. | Still does not recognize the tackle confidently. The analyst also says “confirmed carry,” which sounds unnatural. |

The save caller said: “A low ball flashes across goal, and the goalkeeper dives
to turn it away!” This shows recognition of the save while preserving the incorrect
buildup, rather than hiding it behind an action-label score. The final tackle analyst
said: “The confirmed carry brought the blue attacker diagonally to the left edge of
the box, where two defenders converged; the loose-ball outcome remains unclear.”

## Changes and evidence

Replay observation now prioritizes decisive contacts and referee decisions, separates
protest from celebration, and distinguishes keeper contact from woodwork, challenges
from shots, and a raised card from the referee's kit. This affects replay agents;
the original showcase and ElevenLabs configuration are unchanged.

The bounded `inspect_frame` operation can compare the same region across a focus
frame and two other supplied frames. Optional `--inspect-first` requires the model
to choose that region before classifying. Every selected stamp must exactly match
an available frame within the current cutoff. Default optional inspection remains
one call when no inspection is requested. Inspection-first adds a model call and
can still pick the wrong detail: the first tackle inspection selected the aftermath.

A targeted 3fps check exposed rounded timestamp labels: the prompt printed three
decimals while validation required the original floating point stamp. Labels now
round-trip exactly and the payload lists available stamps. Membership and future
frame checks remain strict. The failed attempt is included in spend and results.

Analyst inputs now retain a recent confirmed save, tackle, interception, block, or
decision and its preceding evidence, using the same bounded window as goals.
Prompts explicitly treat uncertain descriptions as hypotheses and allow useful
analysis of defensive actions without a goal. This is guidance, not proof that
every generated sentence follows uncertainty correctly.

An earlier experiment supplying frames from the previous batch was discarded:
the model repeatedly cited context-only stamps for current observations. Those
outputs were rejected rather than weakening the boundary. The first-pass traces
retain those failures. No ambiguous concatenated JSON was salvaged.

## Limits and reproducibility

These are development checks, not a controlled A/B test or validated accuracy
estimate. References remain drafts. The original save/tackle baseline had critical
budget stops, so their misses do not isolate model capability. New checks finished
without budget stops; this still does not make their labels correct.

Inspection-first used one batch per short recorded clip (9s batching, 8 selected
frames), and speech started roughly 12–20s after playback began. The final tackle
check narrowed the existing source to 6.5–10s at 3fps. Those are recorded quality
diagnostics, not evidence of real-time performance or a clean comparison with the
earlier streaming configuration. Speech timing is simulated, with no audio rendered.

The [result record](../examples/evaluation-results/recognition-pass-20261107.json)
contains all eight attempts, transcripts, failures, source/run hashes, prompt hashes,
configurations and separate budget ledgers. Raw runs remain local under
`runs/replay/recognition-pass-20261107*`. Source revision hashes distinguish the
intermediate configurations. Code commits: `afd4e94` and `f435bdd`.

Validation: 280 offline tests passed, Ruff passed, and mypy passed for 23 source
files. Regressions cover bounded motion crops, unavailable-frame rejection,
inspection-first execution, fractional timestamps, and analyst evidence retention.
The next quality work should focus on the save's rebound/header sequence and the
tackle contact, before checking the remaining selected clips. Do not present the
current tackle or save commentary as finished demo output.
