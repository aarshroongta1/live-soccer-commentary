# Researched identity handoff

The default automatic demo now uses a researcher-generated England squad rather
than the supplied roster fallback. The observer requests that context, proposes
Harry Kane for readable number 9, and the caller/analyst produce separate calls.
No words were manually rewritten, and no speech was synthesized.

The [complete evaluation record](../examples/evaluation-results/researched-handoff-20261107.json)
retains the final replay, research, source check, iteration outcomes and spend.
This is one existing seven-second penalty clip, not a benchmark accuracy claim.

## Research and source review

A squad-focused research call found the official [England squad-number article](https://www.englandfootball.com/articles/2022/Nov/14/england-2022-world-cup-squad-numbers-20221114),
published 14 November 2022. All 26 number/name entries matched the page when
checked with the browser. The pre-match cutoff is 9 December 2022. The accepted
England entries are unchanged model output. The same call returned France
context whose source returned HTTP 403; that side was excluded before use.
Source review was performed by the coding agent, not an autonomous source verifier.

The first broad research attempt returned empty squads and later-dated sources;
it was rejected. An England-only follow-up cited an undated general team page
and was not used. All three calls are included in the spend and failure record.
`--research-focus squads` now directs searches toward dated tournament squad
numbers; `--research-side home` can scope a check to the relevant side. Unknown
sides and out-of-scope returned squads are rejected. The normal context focus
still supports optional pre-match facts.

## Observer and commentary improvements

The observer's first identity response now uses a lookup-only schema: request
`research_roster` or abstain. After the tool returns, it can identify or abstain.
The previous combined action schema sometimes produced concatenated JSON, losing
readable identities even when vision had found number 9. Stage-specific response
contracts address that failure without a new agent or framework. Code still
validates proposals rather than choosing names.

An early analyst invented a goalkeeper touch from a weak visual cue. The observer
now requires actual visible ball/keeper contact for that cue; the analyst must
not turn audit tags into extra events. A later caller said England were ahead,
although the goal levelled the match. The caller now omits lead/equaliser/winner
claims unless explicit score evidence establishes them.

The final unedited lines are:

- Caller: “Harry Kane scores for England!”
- Analyst: “The finish was low and driven across goal, forcing the keeper into a full dive before it went in.”

The final replay reuses visual readings, identity decisions and the analyst line
from the preceding same-evaluation run, preserving the recorded delays. Only the
caller was regenerated for the score-grounding revision. Earlier fresh-vision
and commentary attempts remain in the report. An experimental short temporal
identity link was reverted after tracing the immediate problem to response
parsing; the strict same-action identity rule remains.

## Validation and limits

All 321 offline tests pass with network connections blocked; Ruff, mypy, web lint
and production build pass. No training, fine-tuning, RL, new footage or TTS was used.
The architecture remains researcher → observer/tool lookup → shared evidence →
caller/analyst → coordinator and trace.

The total estimate is **$0.1128842 across 21 generation calls**, including **five
hosted search actions ($0.05)**. All model calls used GPT-5.4 Mini. Each research
attempt allowed one generation, at most three hosted tool calls and 4096 output
tokens, without automatic retries. Search-content tokens mean those research calls
were call/output limited rather than hard dollar capped. Replay phases retained
the existing hard token-budget guard. Tool fees use the checked [official pricing](https://developers.openai.com/api/docs/pricing#built-in-tools);
the provider invoice may differ from this usage estimate.

This does not establish general player recognition, real-time latency, a correct
missed-penalty control, or robust replay/camera-cut classification. Those remain
limitations. Source/event timing and identity attribution were reviewed by the
coding agent; the references are not human-approved ground truth.
