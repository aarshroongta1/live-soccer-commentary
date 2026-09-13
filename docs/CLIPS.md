# Clips: what happens on footage the system was never tuned on

Everything before this was one three-minute clip — Argentina v France 2022,
video 36:00 to 39:00, Di María's goal — run and re-run until it produced good
commentary. Every rule in the gate, every number in `perception`, and the
whole definition of done were fitted to those three minutes. What none of it
had been asked is whether any of it is about football or only about that clip.

So: five more clips off the same broadcast, each carrying an event type the
system has never seen — a penalty, two substitutions, a card, a second-half
comeback, a shootout — run once each with the same command, and graded by the
same machine rather than by eye:

```
uv run python -m commentary run --source file --path clips/<clip>.mp4 \
    --backend anthropic --pack clips/pack-argfra-2022.json \
    --seconds 195 --delay 8 --marks --out runs/<clip>
uv run python -m commentary grade runs/<clip>/*.jsonl \
    --pack clips/pack-argfra-2022.json \
    --statsbomb clips/statsbomb-events-3869685.json \
    --lineups clips/statsbomb-lineups-3869685.json
```

## The video clock

The clips come out of one FIFA upload
(`https://www.youtube.com/watch?v=RgqKdplLIk4`, 2:37:00 long), and the offset
between the match clock and that upload's clock is different in every period,
because the upload cuts the intervals down. Each one was measured by pulling a
three-second probe and reading the score bug off a frame:

| period | probe | bug read | offset (video − match) |
|---|---|---|---|
| 1 first half | 36:00 | 34:24 | **+96 s** |
| 2 second half | 1:00:00 | 50:01 | **+599 s** |
| 2 (check) | 1:30:00 | 80:01 | +599 s |
| 3 extra time 1 | 2:05:00 | 104:54 | **+1206 s** |
| 4 extra time 2 | 2:22:00 | 120:00 +0:04 | **+1320 s** |
| 5 shootout | 2:29:30 | *no clock* — tally 1-1 | **≈ +1645 s** |

The shootout has no clock to measure against: StatsBomb stamps the kicks
120:13 to 125:57 and the broadcast shows a tally with no time on it at all.
The offset above is fitted from one frame (the tally at 1-1 with Coman already
saved, so between 121:42 and 122:27) and is good to about twenty seconds.

## The clips

Six, all pulled with `yt-dlp --download-sections` at 720p, all checked frame
by frame before a run: the first frame's score bug says the match clock the
offset predicted, and the default crop `0,0,0.42,0.16` holds the whole bug.

| clip | match | video | length | first frame reads | new event types |
|---|---|---|---|---|---|
| penalty1 | 20:30-24:00 | 22:06-25:36 | 210 s | 20:35 ✓ | penalty won, penalty scored |
| subs | 39:45-42:45 | 41:21-44:21 | 180 s | 39:50 ✓ | two substitutions |
| card | 53:00-56:00 | 1:02:59-1:05:59 | 180 s | 53:05 ✓ | yellow card, injury stoppage, second half |
| mbappe | 78:00-81:30 | 1:27:59-1:31:29 | 210 s | 78:05 ✓ | second penalty, two goals in 95 s, a comeback |
| shootout | shootout | 2:30:30-2:33:30 | 180 s | **no bug at all** | a shootout: no clock, a tally, a miss, a winner |
| offside | 28:30-31:30 | 30:06-33:06 | 180 s | 29:30 at +60 s ✓ | offside |

**The shootout clip has no score bug in the crop, and none anywhere.** The
broadcaster replaces it with a bottom-centre bar — `ARGENTINA 2 ◆ 1 FRANCE`
with a row of green and red diamonds for the kicks taken — and the only thing
in the top-left corner is the stadium's own screen behind the crowd. So the
board reader will read an absent bug for three minutes, `BUG_GONE_S` will fire
at 30 s, and the summary will say the score and clock may be stale for the
rest of the run. That is the A6 fix doing exactly what it was built for, and
the run will show whether a caller told the board is stale can still call a
shootout. The tally itself is unreadable by anything in the system: nothing
looks at the bottom of the frame.

## Budget

$10, hard, across every run on this page.

| run | cost | running total |
|---|---:|---:|
| penalty1 | $0.72 | $0.72 |

**The runs stopped here.** At cursor 131 s of the first clip every model call
started coming back `400 invalid_request_error: Your credit balance is too low
to access the Anthropic API` — 175 of them (95 caller, 44 analyst, 36 board)
from there to the end of the run. The budget was not what stopped it: $9.28 of
the $10 is unspent. Clips b to f are downloaded, their offsets are measured and
their crops are checked, and they run the moment the key has credit.

## Per-clip reports

### a. penalty1 — match 20:30-24:00, video 22:06-25:36

New event types: **a penalty won and converted** (Dembélé fouls Di María 20:53,
Messi scores from the spot 22:24), and with it the first long dead-ball
passage the system has seen — ninety-one seconds between the award and the
kick with nothing happening on the ball.

Cost $0.72. **Partial: the API credit ran out at cursor 131 s of 202.** The
kick and the celebration are inside what ran; the last seventy seconds are
error rows. Everything below is measured on the 131 s that had a model behind
it, and item 12 fails on those errors rather than on anything the system did.

**7 of 12.**

| item | | |
|---|---|---|
| 1 goal | FAIL | The goal is called on time and **the scorer is not named**: "He steps up, sends the keeper the wrong way, and buries it — Argentina have their goal!" Thirty seconds earlier it had said "Messi steps away from the spot", and it had bound `10 Messi` five times in the preceding minute. |
| 2 recall | FAIL | 3/5 non-goal events within 10 s (foul 1/1, penalty 1/1, tackle 1/1, clearance 0/1, kickoff 0/1). The two misses are after the credit ran out. |
| 3 phantom | PASS | 9 event claims, all in the feed. Only after the grader was fixed — see below. |
| 4 unconfirmed_goal | PASS | None. A penalty goal went through the gate with no argument. |
| 5 precision | PASS | No invented name in 17 lines. 2 of 9 names have the feed within 3 s, and the other seven are a dead-ball passage where StatsBomb has nothing at all — Dembélé trudging back, Messi over the ball, Lloris on his line. Two checked against the frame by eye — MacAllister at 117 s is the number 20 running in on Messi, Dembélé at 41 s is the France shirt in the replay of his own foul. |
| 6 name_rate | FAIL | 3 of 9 caller lines in live play name somebody (33%), against 60%. |
| 7 open play | PASS | 3 distinct (Romero, Dembélé, MacAllister); Dembélé carried on mark CM. |
| 8 sightings | FAIL | 16 of 34 bound (47%), 6 on a live mark. The 18 drops are one cause — below. |
| 9 silence | PASS | Longest live-play gap 10 s. |
| 10 analyst | PASS | 1 line, 26 words, nothing false. |
| 11 scoreline | PASS | None in 17 lines, repetition 0%. |
| 12 health | FAIL | $0.72, tracker 6.7 passes/s, median track life 2.57 s — and **175 error rows**, all of them the credit failure. |

Every name said, checked against StatsBomb and the lineups: Romero, Dembélé
(×3), Di María, Scaloni, Messi (×2), Lloris, MacAllister. All correct, all
really on the pitch at that moment, and Scaloni is Argentina's manager on the
touchline, which is where the camera was.

**What the new event type exposed.**

- **The caller called the penalty as a penalty, and it took the graphic to do
  it.** The referee points to the spot at 20:53; the caller said "the referee
  points for the foul — Argentina have a free kick" and then wrote `free_kick`
  on six consecutive lines. It switched to `penalty` at 21:43, fifty seconds
  later, and said outright what changed its mind: "Messi steps away from the
  spot, **the graphic tells us the story**". Seven `penalty` lines after that,
  all correct. So the vocabulary is there and the picture of a referee's arm is
  not enough; a broadcast graphic is.
- **The gate let the penalty goal through with no argument.** Zero rejections
  in twenty judged lines, and no `unconfirmed_goal` — the board moved 0-0 to
  1-0 and the goal-confirmation work from A8/A16 handled a penalty exactly as
  it handles open play. Nothing here needed changing.
- **The system will describe a penalty for ninety seconds and never say who is
  taking it.** Six of the seven build-up lines are about the ball, the wall,
  the keeper and the crowd. It named Messi twice in that stretch and then, at
  the moment of the kick, said "He".
- **A number both squads wear names nobody, and that is most of the drops.**
  18 of 34 sightings failed to bind and nearly every one is a number-only read
  of a number two players wear: 5, 7, 9, 10, 11, 13, 14, 18, 20, 22. The rule
  is deliberate (run 3 bound France's Thuram to Argentina's Molina off a bare
  26), but on this clip it is throwing away half of what the caller reads.
- **The grader was wrong about the build-up and is fixed.** Five lines over the
  ball on the spot were scored as phantom penalties, because StatsBomb stamps
  the award as an instant and the broadcast spends a minute and a half on it.
  Same shape as the A19 goal-talk bug. Each claimable event now has the tail
  its coverage really runs to.

## Needs a decision

*(design changes a clip exposed, which are not this session's to make)*

1. **A number alone, when both squads wear it, names nobody — and that is half
   of every read.** penalty1 made 34 sightings and dropped 18, all but a couple
   of them a bare number that two players in this match wear. Binding one needs
   a side, the only source of a side in the picture is the kit split, and C11
   took the kit split out of the naming path on purpose after it put France's
   26 on Argentina's Molina. Three ways out, all of them rule changes: let the
   kit split vote on a side when its distance to a centroid is small enough;
   ask the caller for the side it read the number off, as a field beside the
   number; or accept the side the possession state already believes. The first
   is what C11 removed, the second is a schema change to `Sighting`, the third
   is an inference rather than a read.
2. **A player the system named thirty seconds ago is "he" at the moment that
   matters.** The caller had `10 Messi` bound five times and said his name
   twice in the build-up, then called the penalty goal itself with no name at
   all. The rules tell it to name a player it can *read*, and at the moment of
   a kick the camera is behind the taker with no number showing. Whether a name
   already established in this passage of play may be carried into a line where
   nothing is legible is a rule decision — it is the difference between naming
   from the picture and naming from memory, which is the thesis.
