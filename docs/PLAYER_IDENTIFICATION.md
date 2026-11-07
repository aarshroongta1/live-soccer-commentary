# Player identification check

Current replay identity decisions are agent-owned. The researcher prepares cited
squads; the observer requests `research_roster` and proposes the player. Code accepts
or rejects its proposal without selecting a substitute. Earlier checks below retain
the previous deterministic mapping for comparison.

The next quality target is accurate names during play. A roster maps a readable
shirt identity to a player; pre-match statistics cannot identify an on-pitch actor.
This work uses existing models and two clips from the fixed 15-clip development
set. It adds no footage, training, research calls, synthesis or demo media.

## Evaluation changes

Replay scorecards now have an `identity` section. It includes all player-labeled
events, including missed detections, rather than scoring only matched events.
Per-event statuses separate missed events, unresolved or wrong names, mismatched
action roles, absence of eligible speech, and resolved names omitted from speech.
A named visible player does not satisfy an explicitly labeled shooter/scorer.

Spoken-name checks require cited evidence available when speech starts, positive
speech duration, and a line that passes the existing grounding checks. Matching
uses accent-insensitive whole tokens and explicit evaluation-only aliases. This
is a lexical diagnostic, not a judgment of semantic attribution or pronunciation.
No player labels means a null recall, not success. The older matched-only metric
remains for compatibility.

The existing dribble and free-kick references now include player labels, explicit
action roles and optional readable celebration identities. These are coding-agent
annotations with human review pending. Reference labels and aliases are never
runtime inputs. A named celebration cannot retroactively identify earlier play.

## Bounded model check

[Saved outputs, evidence hashes and usage](../examples/evaluation-results/player-name-check-20261107.json)
compare the previous observer prompt with the first identity-focused prompt on
identical eight-frame packets and partial rosters. GPT-5.4 Mini made four vision
calls, then two static caller/analyst snapshot calls. The combined budget was
$0.08 and eight calls; actual reported usage was **$0.04129275 for six calls**,
with no unsettled requests or retries. No reviewer or hosted tools were used.

- Messi: neither prompt resolved an actor, including the late celebration.
- Ronaldo: the candidate resolved a named visible player in celebration. It did
  not establish a named scorer. The baseline failed with `LLMError`; the harness
  retained only the exception class, so the exact cause is unknown. This is not
  a clean paired improvement. The earlier progress description of invalid
  structured output was more specific than the retained evidence supports.
- Given the candidate observations, the caller abstained. The analyst used
  “Ronaldo” in describing the celebration. The full wording still needs work;
  this was a static snapshot check, not measured live commentary or synthesized
  audio. Both roles received the same snapshot without preceding dialogue.

The candidate's raw identity expanded the shirt text to “Cristiano Ronaldo”.
The visible shirt says “RONALDO”. A subsequent observer instruction explicitly
requires literal shirt text and forbids deriving numbers from the roster. This
last clarification has offline regression validation but was not paid-tested.
The speaker prompt now prefers verified names where natural, while retaining
same-action evidence and action-role restrictions. No automatic goal call was
replaced and none of these results was promoted to the voiced showcase.

## What remains

The main bottleneck is reading identity during the action, rather than the
speaker's ability to pronounce a supplied name. Next, inspect potentially
readable action-participant shirts using the existing bounded crop tool. If
identity remains unreadable, retain a generic call. Do not manufacture a named
scorer from a later celebration, roster familiarity or a famous clip title.

Offline regression checks cover missed events, wrong identities, omitted names,
alias token boundaries, future/uncited evidence, late celebration identities,
and visible-player versus scorer roles. All 291 project tests pass, with network
connections blocked; Ruff and mypy also pass.

## Iteration: a named call and a useful handoff

The [next bounded iteration](../examples/evaluation-results/player-name-iteration-20261107.json)
kept the same source library and used four existing cases: dribble, free kick,
penalty and miss. Vision now receives team/kit details without the roster name
lookup table. Python resolves the literal observed number/name afterward.

`--inspection-focus identity` makes the observer select one participant crop,
then read the supplied original frames and bounded crop sequence. It reuses
`inspect_frame`; there is no new agent, tracker or trained model. The normal
action focus remains the default because crop localization is still imperfect.
A repeated/out-of-order crop context is canonicalized locally; nonfinite,
negative, unsupplied and future frame requests remain invalid.

The penalty case produced an unedited caller line:

> Harry Kane buries the penalty! England level at 1-1!

The first analyst repeated the score. After a general prompt change to add one
supported explanation, the revised analyst produced:

> The penalty was driven low and out of the goalkeeper’s reach, with the dive going the wrong way.

This last replay reused the exact model-produced visual evidence and caller line
with their recorded delays; only the analyst was regenerated. Number 9 is
visible during the kick in the original frames and resolves to Harry Kane. The
selected crop itself missed the intended player: do not claim the crop caused
this success. The two independent turns were coordinated by the real replay
runtime. Neither line was manually rewritten. Both speech records completed
without interruption. The seven-second source clip takes about 24 seconds to
finish this diagnostic replay; the caller begins around 15.5 seconds. This is
not real-time performance, and speech timing remains simulated.

The result is an inspectable named-call example, not a broad accuracy win. The
lexical identity diagnostic covers one of two player-labeled events; the earlier
penalty-setup observation is still unresolved. Messi remained unresolved and
Ronaldo remained identified only in celebration. The miss control failed twice:
first on crop-context formatting, then on inconsistent scene/action output. The
validator prevented accepting that inconsistent goal; **the negative control did
not pass**, and no correct-miss claim is made. All failures remain in the report.

This iteration used **12 GPT-5.4 Mini calls for $0.0664908**, below its $0.20 cap,
with no unsettled requests. This is additional to the earlier six-call check.
There were no model calls in the 301 passing regression tests. Ruff and mypy
also pass. The reproducible local scripts and raw outputs are retained under
`runs/replay/player-name-iteration-20261107/`.

## Agent-owned research and identity decisions

The current replay moves name selection out of Python. The researcher can prepare
up to two cited squads (team side, kit context, canonical names, numbers, URL and
date) alongside its six optional facts. The observer first reads available visual
evidence, then has at most two short turns: request `research_roster`, and propose
an identity or abstain. It cites the observation and returned squad ID. No separate
identity agent, tracking framework or training was added.

Code only accepts/rejects proposals. It checks that the cited observation exists,
the squad was returned by the tool, side and literal reading agree, and the named
player is in that squad. A wrong proposal is not replaced by a matching name.
The observer can abstain even on a unique number. A celebration still cannot fill
an earlier missing identity. Failures leave visual events intact and unnamed.

The [saved model check](../examples/evaluation-results/agent-owned-identity-20261107.json)
shows an observer requesting the roster and proposing Harry Kane for the readable
number 9 observations. The first call failed JSON parsing; after clarifying one
action per turn, two new calls succeeded. Total new reported spend was $0.00328725
for three GPT-5.4 Mini calls. The source was the previously supplied roster, not
fresh web research; researcher squad generation and the handoff were verified
with offline scripted tests. No source-verification claim is made for this check.

The demo replays the saved agent identity decisions with the existing unedited
visual readings/commentary and recorded delays. Its provenance identifies that
reuse and the supplied roster. Previous deterministic reports remain historical
comparisons. Old identity caches cannot be reused as current agent-generated
identities; changed research squads also invalidate cached vision.
