# Action-dense, replay-aware commentary plan

## Goal

Make the system explain the football in front of it. On the Barça–Betis clip
that means:

1. identify players whenever the pictures support a name;
2. distinguish new actions—carry, pass, receiver, cross, shot, finish—instead
   of repeatedly saying that Barcelona are advancing;
3. connect those actions into one move with continuity;
4. describe how the goal happened, not merely that it happened;
5. call the decisive moment briefly and give the verified score once;
6. use later replay pictures to confirm or enrich the move without announcing
   it as a second goal.

The target is not the largest possible number of lines. It is the largest
amount of new, correctly ordered information that can physically reach air.

## Reference clip

Clip: `clips/betis-barcelona-1605-1710.mp4`

Window: 16:05–17:10 of the source video.

The English audio track was transcribed locally with
`mlx-community/whisper-large-v3-turbo`. A few words in the live goal call are
uncertain because of crowd noise, but the broadcast structure is clear:

> Watch out here because Barça… in search of the equaliser—yes, it is! Just
> like that, they’re level.

> And it’s Ferran Torres again who scores. Betis one, Barça one. We’re right
> back in this now.

> Easy play. That was brilliant. That was such a simple play—just first-touch
> passing.

> I think it was between Eric García and Jules Koundé out on that right side.

> It was Lamine, Bardghji, Koundé.

> I think that’s on. That’s tight for offside, I’ll tell you what, but I think
> it’s just on.

> A brilliant team goal from Barça, and Ferran, like every good striker should,
> gets across his man and makes the finish at the near post.

The pictures after the equaliser are replays of the same goal. They are not a
second Barcelona attack and goal.

## What the current system gets wrong

The concurrent observer fixed a real scheduling bug: model latency no longer
blocks the system from looking at the match. On this clip it increased caller
observations from 8 to 18 and complete aired lines from 6 to 13.

That exposed the next layer of problems:

- five opening lines say little beyond “Barcelona advance,” with no player
  chain and almost no new football information;
- the model does not reliably extract the ball carrier, receiver, passer, or
  crosser even when several frames show the progression;
- overlapping observations produce generic versions of the same build-up;
- the pass and cross are sometimes present in forms but do not reach air;
- the goal call names Ferran but does not describe the cross, movement, touch,
  placement, or finish;
- close-ups and replays retain `event=goal`, so the director treats every one
  as another urgent goal;
- three celebration descriptions say the same thing;
- the replay sequence is mistaken for a new live attack;
- the same goal is announced again at the end;
- the old 1–1 score is attached to that false second announcement;
- replay frames are not used to improve the explanation of the move.

The primary failure is semantic extraction and move continuity. Replay
handling is a real correctness issue, but it is secondary.

## Design

### 1. Make the observation action-shaped

The vision model should first report structured football, not immediately
write a sentence. Each meaningful beat needs:

- frame or cursor anchor;
- acting player, with shirt number and side when visible;
- action: carry, pass, receive, layoff, cross, shot, save, or finish;
- target player when supported;
- origin and destination zones;
- direction, delivery, body part, and outcome when visible;
- confidence for the action and each identity.

One observation window may return several chronological beats. “Barcelona
advance down the right” is not an acceptable substitute when the frames show
Koundé receiving, playing inside, and delivering across goal.

Unknown identities must remain unknown. The model may say “the right-back” or
omit the name, but it may not replace a missing player observation with a team
name and pretend that the line gained detail.

### 2. Ground and carry player identity

Use the pack, shirt sightings, side, field position, and adjacent observations
to resolve names. Carry an identity across a short continuous possession only
when there is no cut, turnover, or contradictory sighting.

The prompt should show the likely on-pitch candidates relevant to the visible
side and position before showing the full squad. The output must keep the
evidence that justified the name so the gate can reject only the identity,
without discarding an otherwise useful action.

The desired fallback order is:

1. verified player name;
2. verified number or role;
3. pronoun only when its referent is unambiguous;
4. team name as a last resort, not the default subject of every line.

### 3. Keep a short move buffer

Retain the last 10–15 seconds of useful live-play forms before a goal. The
buffer should contain structured facts, not finished prose:

- side in possession;
- named or numbered player sightings;
- pass, carry, cross, shot, and goal events;
- direction and location;
- short action detail;
- cursor timestamp.

Adjacent forms that describe the same action should merge rather than append.
Pass → receiver → cross → finish must remain separate.

The buffer is the continuity layer. It should answer “what changed since the
last line?” and “how did this chance develop?” without relying on several
in-flight prompts having seen one another's prose.

When a goal arrives, freeze the buffer onto the incident. The goal description
can then cite the final ball, Ferran's movement, and the placement rather than
producing an isolated “buries it.”

### 4. Plan commentary from deltas, not snapshots

Before a line reaches the director, compare its structured beats with the move
buffer and recently aired facts.

A line is eligible only when it adds at least one of:

- a newly identified player;
- a new action or receiver;
- a meaningful change of zone or direction;
- a chance outcome;
- a goal detail;
- a tactical or factual explanation appropriate to a stoppage.

This should remove sequences such as:

> Barcelona carry it forward down the right.

> Barcelona work it down the right.

> Barcelona probe around the edge of the box.

without using a broad similarity threshold that would also erase genuine
pass → cross → shot progression.

### 5. Describe the goal from the move

The live call should be short enough to beat the director queue:

> Ferran turns it in!

The next verified line may add scorer and score:

> Ferran Torres, across his man at the near post. Betis one, Barça one.

The live goal call and its first follow-up should draw from the frozen move:

- who supplied the final ball;
- delivery type and direction;
- scorer's movement;
- touch or body part;
- placement and distance;
- goalkeeper or defender involvement when clear.

Do not require every detail to fit in the shout. The first follow-up is where
the system can explain what the shout could not.

### 6. Treat replay as enrichment, not the main solution

Replay matters after action extraction works. It may confirm a player, offside
line, touch, or finish detail that was unclear live. It must not be the only
place the system can understand the move.

Replay analysis may be longer because the ball is dead:

> Eric García into Koundé, worked through Lamine and Bardghji.

> Ferran gets across his man and finishes at the near post.

Do not ask every overlapping observation to produce another sentence. The
model may choose silence when it adds no new action, name, decision, or detail.

Give each goal a small lifecycle—`live`, `celebration`, `replay`, `finished`—so
the same incident is not announced twice. Reuse the current goal-follow-up and
replay machinery rather than building another service.

### 7. Make director priority action-aware

Only the initial goal call may preempt speech as an urgent goal.

For later beats:

- one celebration line maximum;
- replay analysis is preemptable;
- a replay line cannot delete or overtake an earlier sibling from the same
  incident;
- stale generic build-up is dropped before it occupies the voice channel;
- a new live event always outranks replay analysis.

If the pass or cross cannot finish before the live goal call, do not speak it
late as though play is still unfolding. Preserve it in the move buffer and use
it in the replay reconstruction.

### 8. Tie score language to the incident

The score may be spoken once per goal incident. It requires evidence associated
with that goal, not merely the fact that some earlier goal is present in match
state.

Rules:

- strip model-written score claims;
- append the verified score on the first eligible line only;
- never append it again during celebration or replay;
- do not use the previous incident's score evidence for a later goal claim;
- if the score is not yet verified, call the finish without a number.

## Implementation stages

### Stage 1: lock the failure into fixtures

- Save the aligned reference transcript as a test fixture.
- Add a fixture representing the current 18 caller observations.
- Add the expected action chain for the goal: identified buildup participants,
  final-ball action, Ferran's movement, and near-post finish.
- Assert that five generic Barcelona-progress lines do not count as five
  meaningful beats.
- Assert that timestamps 31–65 seconds belong to the first goal's aftermath
  and replay, not a second goal.
- Add an output-level regression that fails on repeated celebrations, repeated
  goals, or two 1–1 announcements.

No model calls are required for this stage.

### Stage 2: introduce action beats

- Add the minimal structured action-beat schema.
- Let one frame window return multiple chronological beats.
- Keep identity evidence separate from action evidence so an uncertain name
  does not erase a clear pass or cross.
- Map frame anchors to exact cursor timestamps.

### Stage 3: resolve player continuity

- Rank likely players from the pack, number, side, role, and position.
- Carry verified identities only across continuous possession.
- Test cuts, turnovers, and contradictory shirt sightings.
- Prefer a role or unnamed action over repetitive team-only prose.

### Stage 4: build and freeze the move buffer

- Collect meaningful live forms before the goal.
- Merge overlapping descriptions of the same action while preserving distinct
  passes, receptions, crosses, shots, and finishes.
- Freeze the sequence when the goal is called.
- Make the frozen move available to the goal call and follow-up.

### Stage 5: generate from new information

- Select only beats that add a player, action, transition, or outcome.
- Write short live fragments from selected beats.
- Build the goal follow-up from the final-ball and finish beats.
- Ensure important pass/cross information is spoken before the goal when there
  is channel time, and retained for the follow-up when there is not.

### Stage 6: add the goal lifecycle and replay enrichment

- Add the small goal-incident state.
- Transition it from live goal to celebration/replay and finally to finished.
- Reclassify close-up and replay forms before beat construction.
- Ensure only the initial live goal becomes an urgent director beat.
- Update the caller prompt with the current incident phase and move summary.
- Ask replay turns for one missing fact: identity, sequence, offside, technique,
  or finish.
- Allow only one line for each fact category.
- Keep replay wording in retrospective tense.

### Stage 7: adjust director sequencing

- Preserve the urgent live goal call.
- Do not let aftermath forms masquerade as urgent goals.
- Prevent generic queued lines from airing after the moment has passed.
- Confirm that replay analysis yields immediately to resumed live play.

### Stage 8: evaluate cheaply

Run deterministic tests first. Then run the 65-second clip once with Luna.

Only compare Terra after the Luna run passes the structural criteria. Do not
spend API credits tuning prompts around a broken incident lifecycle.

## Acceptance criteria

The clip passes when all of these are true:

- at least three distinct actions from the scoring move are represented;
- at least two buildup participants are correctly identified when visible;
- Koundé's involvement or the final-ball provider is described if supported by
  the observation evidence;
- the commentary distinguishes progression, final ball, and finish rather than
  repeating Barcelona's field position;
- the goal description includes at least one concrete finish detail such as
  movement across the defender or near-post placement;
- no two consecutive aired lines merely paraphrase the same action or field
  position;
- exactly one live goal incident is created;
- exactly one urgent goal beat is submitted;
- Ferran Torres is identified as the scorer;
- 1–1 is spoken no more than once;
- the replay sequence is never described as a new live attack or second goal;
- no more than one celebration line airs;
- replay analysis adds information rather than being required to rescue an
  empty live description;
- every line arrives in chronological incident order;
- all existing unit and integration tests pass;
- the run stays below $0.05 with Luna.

Line count is reported but is not itself a pass condition.

## Explicit non-goals

- Fine-tuning a model before the runtime structure works.
- Reintroducing OCR as a mandatory dependency for every frame.
- Building production-grade retries, security layers, or distributed queues.
- Narrating every touch.
- Preserving old architecture solely for compatibility.

## Expected result

The system should sound closer to this shape:

> Koundé receives on the right.

> Driven across the six-yard box—Ferran turns it in!

> Ferran Torres gets across his man at the near post. Betis one, Barça one.

> A simple first-touch move down the right, worked through Eric García and
> Koundé.

> The replay confirms Ferran's movement and the tight onside decision.

That is fewer lines than the noisy 13-line run, but substantially more actual
commentary.
