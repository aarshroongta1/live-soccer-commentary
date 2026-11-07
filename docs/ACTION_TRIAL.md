# First action checks

November 7, 2026. This was a small development investigation using existing models,
not model training or an official SoccerNet benchmark. Eleven short runs across
four clip windows cost **$0.0631551** in reported model tokens, with no outstanding
reservations, against a $0.09 ceiling. No TTS was generated.

| Clip | Latest result | Limitation |
| --- | --- | --- |
| Chelsea–Burnley goal | Detected the goal and produced caller plus analyst text | One published SoccerNet label, aligned to an independent public highlight by its on-screen clock; approximate, not a benchmark. Buildup analysis still needs review. |
| France–Jamaica red card | Called it a yellow card; analyst repeated that mistake | A self-reported visual cue is not proof. Scene was also incorrectly labeled celebration. |
| Norwich–Tottenham tackle | Described carry/pass movement; missed the tackle | A later vision batch hit the budget guard, so this does not isolate model recognition ability. |
| Villa–Arsenal save | Described a cross/recovery; missed the header and save | A vision batch covering important action hit the budget guard. |

The first runs produced no commentary because strict application validation rejected
whole observations. Regressions now cover: aerial paths on shots, later available
frames confirming earlier actions, cutoff/supplied-frame checks, all eight supporting
frames, and audited trajectory abstention without discarding a valid goal. Goal
confirmation still requires direct net/line evidence. These are reliability fixes,
not evidence that the model became accurate. One role response still contained
multiple JSON objects and was rejected.

The successful goal call was the coordinator's deterministic reaction to accepted
visual evidence. The analyst text was model generated. Speech timing is a word-count
simulation, not real audio delivery or playback latency.

Source pool: [catalog](../examples/action-sources.json). The Chelsea clip is the
public [FOX Sports Italia highlight](https://www.youtube.com/watch?v=2MW6hl9d1vk),
not a protected SoccerNet download. Its [reference](../examples/replay-evaluation/public/chelsea-burnley-public-goal.json)
records the exact local window, video/label hashes, official match partition, clock
anchor, and uncertainty. The other references remain agent-drafted and unreviewed.

All attempts, failures, budget stops, transcripts, and costs are preserved in the
[trial record](../examples/evaluation-results/action-smoke-20261107.json). Full raw
runs and per-run scorecards are local under `runs/replay/action-baseline-20261107/`.
Different revisions/batches were tried; this was not a controlled A/B experiment.

## Next scope

Keep roughly six short, fixed clips covering a goal, a save/header or missed chance,
a tackle/carry, a card decision, passing/routine buildup, and a set piece. Existing
footage is sufficient. Check labels and timing, then improve existing-model
recognition, caller/analyst content, and handoffs on those clips. Use small explicit
budgets that let the chosen window finish; a budget-stopped trial is incomplete.
No more broad dataset acquisition, full benchmark infrastructure, model training,
or automated prompt-optimization project is needed for the current demo milestone.
