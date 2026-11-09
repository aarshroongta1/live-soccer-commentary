# Fixed action clips

Selected November 7, 2026: **15 short windows, 112.5 seconds total**. The selection is frozen for the next recognition/commentary pass. No training, broad benchmark expansion, or gated-video access is needed.

The machine-readable selection is [action-suite.json](../examples/action-suite.json). Each entry links to a separate evaluator reference and records its source window, action targets, and manual commentary checks. Labels never go to the runtime agents.

The next pass uses the [text-only rubric and comparison protocol](TEXT_EVALUATION.md)
and its [15-case review worksheet](../examples/evaluation-results/text-review-v1.json).
Worksheet scores are pending; voice is excluded.

| # | Case | Local source window | Main checks |
| --- | --- | --- | --- |
| 1 | Routine buildup | `liga-f-routine.mp4`, 24–39s | pass, carry |
| 2 | Dribble and finish | `barcelona-bayern-messi.mp4`, 7.5–14s | dribble, shot, goal |
| 3 | Recovery tackle | `pl-tackles.mp4`, 5–12s | carry, tackle |
| 4 | Cut out a pass | `pl-interceptions.webm`, 37–43.5s | interception |
| 5 | Cross and headed chance | `france-jamaica-late.webm`, 3–13s | cross, header |
| 6 | Shot, rebound header, save | `pl-saves.mp4`, 10–18s | shot, header, save |
| 7 | Penalty missed over the bar | `england-france-penalties.mp4`, 103–108s | penalty_kick |
| 8 | Goal and reaction | `chelsea-burnley-2015.webm`, 12–18.5s | goal |
| 9 | Corner delivery into a crowded box | `pl-saves.mp4`, 38.5–46s | corner, save |
| 10 | Direct free kick | `portugal-spain-live.mp4`, 19.5–25.5s | free_kick, goal |
| 11 | Penalty scored, then replay | `england-france-penalties.mp4`, 50.5–57.5s | penalty_kick, goal |
| 12 | Long throw into the area | `delap-match-throw.webm`, 23–29.5s | throw_in |
| 13 | Foul followed by a penalty award | `england-france-penalties.mp4`, 43–49s | penalty_awarded |
| 14 | Red-card decision | `france-jamaica-late.webm`, 37.5–42.5s | red_card |
| 15 | VAR decision: no goal, offside | `everton-var-offside.webm`, 92–102s | offside |

The set includes routine play, famous finishes, club and international matches, women’s football, and a spectator-angle throw-in. It includes explicit replay checks; repeat angles must not become new events. Scored and missed penalties are different windows of the same match. Those windows are not independent statistical samples.

## What is ready

- Every selected local file exists and its hash matches the catalog/reference.
- Every window fits its source, and every reference timeline passes schema validation.
- All fifteen cases belong to development. This is a fixed regression set, not held-out evidence of general accuracy.
- Three new usable public source files filled the penalty, offside and throw-in gaps. Existing showcase footage supplied the dribble and free kick. No source video was duplicated to create the windows.

## What still needs care

The selection is frozen; the labels are **drafts** with approximately one-second timing uncertainty and sparse coverage. Most labels were drafted from direct source inspection, not copied from model predictions. The Chelsea goal uses a published label with an approximate on-screen-clock alignment. None is represented as human-approved ground truth.

The foul case uses a referee-confirmed penalty award, rather than inferring a decision from a collision. It does not separately cover all ordinary midfield fouls. The offside case requires the final NO GOAL OFFSIDE decision, not just a possible-offside review banner. The miss case checks outcome through the manual notes; the existing JSON scorer does not semantically verify those notes.

A captioned throw-in explainer was rejected because it would supply the answer in narration text. The selected throw uses an uncaptioned public match recording. A previously suggested throw in the Liga F footage was not confirmed and is no longer claimed as throw-in coverage.

Next work is improving existing-model perception, useful caller/analyst output, and handoffs on this set. Begin with the already observed card/keeper/challenge failures, and ensure the budget allows each selected window to finish before interpreting misses. Do not run repeated paid full-suite tests for routine code changes.
