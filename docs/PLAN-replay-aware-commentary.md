# Replay-aware commentary plan

## Goal

Make the system follow a football incident the way the real Barça–Betis
broadcast did:

1. call the decisive moment briefly and immediately;
2. name the scorer and give the verified score once;
3. recognize that the following pictures are celebrations and replays of the
   same incident;
4. use those replays to reconstruct the move, discuss offside, and explain the
   finish;
5. resume live-play commentary only when the match has actually restarted.

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

- overlapping observations produce several generic versions of the same
  build-up;
- the pass and cross are observed but do not reach air before the goal;
- close-ups and replays retain `event=goal`, so the director treats every one
  as another urgent goal;
- three celebration descriptions say the same thing;
- the replay sequence is mistaken for a new live attack;
- the same goal is announced again at the end;
- the old 1–1 score is attached to that false second announcement;
- replay frames are not used to improve the explanation of the move.

The failure is now incident continuity, not observation frequency.

## Design

### 1. Give each goal one lifecycle

Represent the current goal as one small runtime object rather than inferring
its identity independently on every caller turn.

It needs only:

- a stable incident ID;
- the cursor time of the live goal;
- scoring side and scorer when known;
- phase: `live`, `celebration`, `replay`, or `finished`;
- whether the live call has aired;
- whether the score has aired;
- whether a celebration line has aired;
- whether the move has been reconstructed;
- the useful observations from the build-up.

Reuse the existing goal-follow-up and replay machinery where possible. Do not
introduce a service, database, or durable queue for this local demo.

### 2. Separate an incident from the picture currently on screen

`event=goal` means the ball has crossed the line in the moment being shown. It
must not also mean “these players are celebrating a goal from thirty seconds
ago.”

The caller form should distinguish:

- a live goal;
- a celebration or close-up related to the current goal;
- a replay of the current goal;
- resumed live play.

Only the first is an urgent `Event.GOAL` beat. Celebration and replay output is
aftermath: lower priority, preemptable, and tied to the existing incident ID.

Runtime evidence should help this classification:

- a goal was just called;
- the score bug disappeared or the broadcast entered a replay sequence;
- the pictures show the same finish from another angle;
- the score has not changed again;
- no credible restart has been observed.

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

When the goal arrives, freeze that buffer onto the goal incident. Replay turns
then receive both the replay frames and this move summary, allowing them to
confirm names and describe the sequence retrospectively.

### 4. Use different language for live play and replay analysis

The live call should be short enough to beat the director queue:

> Ferran turns it in!

The next verified line may add scorer and score:

> Ferran Torres again. Betis one, Barça one.

Replay analysis may be longer because the ball is dead:

> Eric García into Koundé, worked through Lamine and Bardghji.

> Ferran gets across his man and finishes at the near post.

Do not ask every overlapping observation to produce another sentence. The
model may choose silence when it adds no new action, name, decision, or detail.

### 5. Make director priority incident-aware

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

### 6. Tie score language to the incident

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
- Assert that timestamps 31–65 seconds belong to the first goal's aftermath
  and replay, not a second goal.
- Add an output-level regression that fails on repeated celebrations, repeated
  goals, or two 1–1 announcements.

No model calls are required for this stage.

### Stage 2: implement the goal lifecycle

- Add the small goal-incident state.
- Transition it from live goal to celebration/replay and finally to finished.
- Reclassify close-up and replay forms before beat construction.
- Ensure only the initial live goal becomes an urgent director beat.

### Stage 3: build and freeze the move buffer

- Collect meaningful live forms before the goal.
- Merge overlapping duplicates.
- Freeze the sequence when the goal is called.
- Feed the sequence to replay commentary.

### Stage 4: make replay output analytical

- Update the caller prompt with the current incident phase and move summary.
- Ask replay turns for one new fact: sequence, offside, technique, or finish.
- Allow only one line for each fact category.
- Keep replay wording in retrospective tense.

### Stage 5: adjust director sequencing

- Preserve the urgent live goal call.
- Do not let aftermath forms masquerade as urgent goals.
- Prevent generic queued lines from airing after the moment has passed.
- Confirm that replay analysis yields immediately to resumed live play.

### Stage 6: evaluate cheaply

Run deterministic tests first. Then run the 65-second clip once with Luna.

Only compare Terra after the Luna run passes the structural criteria. Do not
spend API credits tuning prompts around a broken incident lifecycle.

## Acceptance criteria

The clip passes when all of these are true:

- exactly one live goal incident is created;
- exactly one urgent goal beat is submitted;
- Ferran Torres is identified as the scorer;
- 1–1 is spoken no more than once;
- the replay sequence is never described as a new live attack or second goal;
- no more than one celebration line airs;
- replay analysis mentions at least two verified participants from the move;
- replay analysis describes at least one meaningful detail such as the
  first-touch passing, offside decision, movement across the defender, or
  near-post finish;
- no two consecutive aired lines merely paraphrase the same field position;
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

> Barça threaten—Ferran turns it in!

> Ferran Torres again. Betis one, Barça one.

> A simple first-touch move down the right, Koundé involved in the final ball.

> Ferran gets across his defender and finishes at the near post.

That is fewer lines than the noisy 13-line run, but substantially more actual
commentary.
