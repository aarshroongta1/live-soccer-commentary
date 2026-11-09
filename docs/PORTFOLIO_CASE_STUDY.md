# Football commentary agents: portfolio case study

## A short project explanation

I built a football commentary prototype that combines visual observations,
pre-match research and separate caller/analyst agents. The agents can request
bounded context tools, speak or abstain. A deterministic coordinator controls
clock visibility, deadlines, turn-taking and stale results. Saved traces make
agent decisions and failures inspectable.

The demo contains one automatic conversation and two labelled edited showcases.
The automatic conversation is prepared on saved perception before playback;
causal replay is a separate prototype with simulated speech occupancy. Voice work
is outside the current finishing scope.

## What the agents actually decide

| Component | Responsibility and decision |
| --- | --- |
| Researcher | Prepares cited, dated squad/team context; unreliable or late sources can be excluded. Research does not prove an on-pitch event. |
| Observer | Proposes typed actions and action-linked identity; can inspect a crop and request the researched roster. Code validates proposals rather than selecting a player from a number. |
| Caller | Chooses an immediate call, permitted context request or abstention. Visual evidence is required. |
| Analyst | Chooses a supported explanation, relevant research request or abstention. It cannot manufacture tactics to fill silence. |
| Coordinator | Deterministic asyncio scheduling, evidence visibility, deadlines and stale-result handling. It is not an LLM supervisor. |

The core stack is Python, LangGraph, Pydantic, asyncio and OpenCV. Next.js presents
saved clips and traces; FFmpeg produces captioned media. Tool use is an
application-executed structured action protocol, not provider-native function
calling. No model weights were trained or fine-tuned.

## A concrete successful path

In the selected England–France sequence, saved vision sees the penalty decision,
readable number 9, the kick and goal. The observer requests the dated England squad
and proposes Harry Kane for the relevant actions. The caller announces the penalty,
anticipates the run-up and calls the goal. The analyst comments on the keeper going
the wrong way. The words are retained exactly in the downloadable trace.

The final pass also produced an early oversized line that the contract rejected.
That is an observable control working, rather than a claim that every generated
line is correct. A first attempt's reviewer rejected a valid short goal call;
the second selected attempt disabled that optional component.

## Evaluation-led iteration: recovering the caller’s goal reaction

The automatic example now has a before/after switch. These are the first and
second attempts in the bounded demo polish, not the earliest system output.
[The retained comparison](../examples/portfolio-demo/iteration.json) verifies
identical source window, observations, resolved identities and research. Both
runs used gpt-5.4-mini and retained their generated words without manual edits.

The first run's optional reviewer rejected even the supported draft **Kane
scores!**. The caller’s rewrite allowance was exhausted, and the analyst announced
the finish instead. Disabling that optional reviewer restored the caller’s goal
reaction followed by an analyst response. Deterministic evidence, role-permission
and timing checks remained enabled.

| Observed measure | Before | After |
| --- | --- | --- |
| Caller announces goal | No | Yes |
| Analyst follows caller’s goal | No | Yes |
| New model calls | 17 | 9 |
| Reported token cost | $0.01795875 | $0.0087087 |
| Retained words | 33 | 33 |
| First call | 1s | 3s |

This failure was diagnosed during focused demo review after the 15-clip pilot.
The pilot did not measure this reviewer change. A scripted counterfactual test
holds the draft and evidence fixed to reproduce how a rejecting reviewer can
suppress a valid goal call while the direct path preserves it.

One fresh generation per configuration does not establish general quality or cost
improvement. The later run also loses the early call; model-wording variation is
uncontrolled. The defensible result is restored role separation on this matched
case, with lower observed calls and cost. Independent naturalness and factual
accuracy gains across the suite remain unproven.

## What evaluation taught me

I used a fixed development selection of 15 clips, totalling 112.5 seconds, spanning
routine play, dribbling, saves, goals and referee decisions. Labels remain sparse
and unreviewed; this is not held-out evidence of general accuracy.

The text pilot had 12 usable perception inputs. Only 5 of 24 attempted writer
arms passed structural checks, and a caller-validation asymmetry made its counts
unsuitable for ranking architectures. Later targeted checks aligned permissions
and improved timing feedback; no full-suite quality improvement is claimed.

The most useful failure was a false goal that reached both writers through shared
perception. The observer supplied its own `ball_in_net` tags, so valid citations
and schemas did not establish factual truth. A larger model repeated the error.
Denser frames and magnification also failed in a focused diagnostic. Neutral ball
localization marked the disputed frames ambiguous while event narration claimed a
goal, illustrating poor calibration and shared upstream error propagation.

I distinguish event recognition, player identity, grounded meaning, timing,
analysis and dialogue coherence. I retain rejected attempts and reported costs,
use scripted backends for regression tests, and put hard dollar/call caps around
paid development checks. The in-run reviewer is not an independent evaluator.

## Interview discussion

- **Why separate agents?** Distinct evidence permissions, role objectives and
  tool access make behavior inspectable and independently configurable. Added
  agents also add latency, cost and failure paths; I have not demonstrated a
  quality advantage over a single writer.
- **Why deterministic coordination?** Clocks, deadlines, cancellation and evidence
  availability require predictable enforcement. Language models choose content
  and tools within those constraints.
- **What was the hardest failure?** A plausible observation was structurally valid
  but visually false. Downstream agreement amplified shared bad evidence.
- **How did I measure improvement?** I preserved the original outputs and froze
  footage/evidence for a before/after reviewer configuration check. The caller
  recovered the goal announcement and the analyst responded afterward. I also
  reported the delayed first call and avoided claiming a suite-wide quality gain.
- **What would I do next?** Improve visual verification and calibration against
  real-goal and false-goal controls before expanding live integration. For this
  portfolio release, those limitations are documented rather than hidden.

## Resume wording

- Built a LangGraph multi-agent football commentary prototype combining visual
  observation, cited pre-match research and independent caller/analyst tool loops,
  with deterministic coordination and inspectable execution traces.
- Developed a 15-clip evaluation workflow covering recognition, identity, evidence
  grounding and timing, with reproducible inputs, budgeted model experiments and
  documented failure analysis.

Use the first bullet if space is tight. Avoid claims of production readiness,
real-time spoken latency, benchmark accuracy or a proven multi-agent quality gain.
