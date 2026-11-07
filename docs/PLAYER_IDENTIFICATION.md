# Player identification check

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
