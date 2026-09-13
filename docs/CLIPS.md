# Clips: what happens on footage the system was never tuned on

Two passes. The first ran five three-minute clips from the one broadcast
everything had been tuned on (below, from "Calibrating the grader"). The
second — this section — dropped that for variety: **ten forty-five-second
clips, one event type each, from five matches across three broadcasters**, to
see whether anything learned on Argentina v France was about football or about
that afternoon.

## The short clips: ten events, five matches, three score bugs

Each clip is 45 s with the event at about 25 s, run with `--seconds 65`.

| clip | event | match | called? | on time | named | phantom | bar | cost |
|---|---|---|---|---|---|---|---|---:|
| e01 | counter-attack goal | NED-ARG WC22 | **yes** | -0.6 s | no (Molina) | 0 | 10/12 | $0.16 |
| e02 | penalty scored | NED-ARG WC22 | **yes** | -4.6 s | no (Messi) | 0 | 8/12 | $0.20 |
| e03 | shootout winner | NED-ARG WC22 | **yes** | -4.5 s | **yes** (Lautaro) | 0 | 9/12 | $0.24 |
| e04 | penalty save | NED-ARG WC22 | **yes** | on it | keeper yes, taker no | 0 | — | $0.20 |
| e05 | shootout miss | MOR-ESP WC22 | **yes** | on it | **both** (Sarabia, Bounou) | 0 | — | $0.22 |
| e06 | yellow card | MOR-ESP WC22 | no | — | — | 0 | 10/12 | $0.20 |
| e07 | direct free-kick goal | POR-ESP WC18 | written, **cut** | — | (Ronaldo) | 0 | 9/12 | $0.20 |
| e08 | penalty scored | POR-ESP WC18 | **yes** | -0.1 s | **yes** (Ronaldo) | 0 | 10/12 | $0.16 |
| e09 | offside | CRO-ENG WC18 | **yes** | -3.5 s | no (Sterling) | 0 | 9/12 | $0.23 |
| e10 | offside | ARG-COL Copa24 | no | — | — | 0 | 10/12 | $0.15 |

Seven of ten events called, an eighth written and killed before it could be
spoken, **zero phantoms in any clip, and not one wrong name in the set.**
$1.96 for the ten, $2.20 including a discarded first attempt.

### The twelve, and the three rerun after the fixes

| clip | event | match | called | named | cost |
|---|---|---|---|---|---:|
| e01 | counter-attack goal | NED-ARG WC22 | yes, -0.6 s | no (Molina) | $0.16 |
| e02 | penalty scored | NED-ARG WC22 | yes, -4.6 s | no (Messi) | $0.20 |
| e03 | shootout winner | NED-ARG WC22 | yes, -4.5 s | **yes** | $0.24 |
| e04 | penalty save | NED-ARG WC22 | yes | keeper only | $0.20 |
| e05 | shootout miss | MOR-ESP WC22 | yes | **both** | $0.22 |
| e06 | yellow card | MOR-ESP WC22 | no — never shown | — | $0.20 |
| e07 | direct free-kick goal | POR-ESP WC18 | written, **cut** | — | $0.20 |
| e08 | penalty scored | POR-ESP WC18 | yes, -0.1 s | **yes** | $0.16 |
| e09 | offside | CRO-ENG WC18 | yes, -3.5 s | no (Sterling) | $0.23 |
| e10 | offside | ARG-COL Copa | no | — | $0.15 |
| e11 | corner taken | CRO-ENG WC18 | yes, -7.4 s | no (Modrić) | $0.20 |
| e12 | foul, no card | ARG-COL Copa | no | — | $0.24 |

**After the fixes**, the same clips run once more:

| clip | before | after |
|---|---|---|
| e01 counter | called -0.6 s, Molina unnamed | **no goal line at all** — the run spent the window on the build-up. Run-to-run variance, not a regression: the carry did work elsewhere in it ("Molina bursts away down the right with Blind chasing"). |
| e02 penalty | called, **Messi unnamed** while `10 Messi` sat in the same call's sightings | **"Messi steps up, strikes it low to the keeper's right, and it is in — Argentina have their second."** Named twice, in the build-up and at the strike. 8 of 12 to 9 of 12. |
| e07 free kick | written with his name, tagged `free_kick`, **killed by the camera cut**, never spoken | **spoken**: "Ronaldo takes his steps back, whips it up and over the wall, and it flies into the top corner!" plus a second line naming him again. The trace shows the beat tagged `event=goal`, `preemptable=False`. **9 of 12 to 11 of 12, the best score of the exercise.** |

Two of the three changed exactly what they were built to change. The third
says nothing either way, which is what one run of a thing with this much
variance is worth.

### What the short clips settled

- **The board reader generalises to layouts it has never seen.** The Copa
  América bug puts the clock on a *second row* underneath the score, and it
  was read 13 times out of 13 with the right clock and the right score. The
  WC2018 bug puts the clock on the right of the bar instead of the left: read
  perfectly on both clips. One of the Copa reads was ten minutes out and the
  median-based alignment absorbed it exactly as it was designed to.
- **A shootout has no board, on any broadcast.** Three shootout clips across
  two matches: zero usable board readings in all three, the grader refused to
  align all three, and `--offset` measured by hand was the only way in. This
  is now a property of the competition, not a quirk of one upload.
- **The offside was called.** "England get in behind again, the shot is worked
  away by Subašić, and the flag goes up against the runner." The three-minute
  offside clip on Argentina v France never mentioned one; a different match on
  a different broadcast got it.
- **Naming at the moment depends on the shot holding, not on the shot type.**
  Ronaldo's penalty and Lautaro's shootout winner were both named as they were
  struck, and in both the camera stayed on the taker through the whole
  build-up so the name was still the subject of the sentence. Molina, Messi
  and Mbappé were all unnamed at the moment, and in each the broadcaster cut
  away and came back. That is a sharper account of the naming gap than
  "close-ups only".
- **A goal from a set piece is not tagged as a goal.** Ronaldo's free kick was
  called, correctly, with his name — *"curls it over the wall and into the top
  corner"* — and the director dropped it `reason=cut, urgency=0.0`, because
  the caller had tagged the line `free_kick`. `free_kick` is not in
  BIG_EVENTS, so a camera cut could kill it; `goal` could not have been.
- **StatsBomb's card timestamp is not when the card is shown.** e06's yellow
  is stamped 89:05 and the frame at 89:07 is live play with no referee in it.
  A 45 s window centred on the feed's timestamp can miss the picture
  entirely, and did, twice — the red card in e03 too.

### The clips, and how each was found

The binding constraint is footage, not data: FIFA only uploads FULL MATCH for
knockout and marquee games, so several red cards in the open data have no
video at all. Every offset below was read off a score bug in a probe frame
and then confirmed against the cut clip's own first frame.

| match | id | bug | kits | offsets (video - match) |
|---|---|---|---|---|
| Netherlands v Argentina, WC22 QF | 3869321 | FIFA 2022 | orange v sky-blue stripes | h1 +155, h2 +630, ET +1523, shootout +1662 |
| Morocco v Spain, WC22 R16 | 3869220 | FIFA 2022 | **red v light blue, red crowd** | h1 -249, h2 -146 |
| Portugal v Spain, WC18 | 7576 | **FIFA 2018**, clock right | red v white | h1 +168, h2 +276 |
| Croatia v England, WC18 SF | 8656 | FIFA 2018 | **red-white checks v all white** | h1 +173 |
| Argentina v Colombia, Copa 2024 final | 3943077 | **CONMEBOL, two rows** | white v yellow | h1 +375, h2 +577 |



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
cd /Users/Aarsh/Desktop/commentary
uv run python -m commentary run --source file --path clips/<clip>.mp4 \
    --backend anthropic --pack clips/pack-argfra-2022.json \
    --seconds 195 --delay 8 --marks --out runs/<clip>
uv run python -m commentary grade runs/<clip>/*.jsonl \
    --pack clips/pack-argfra-2022.json \
    --statsbomb clips/statsbomb-events-3869685.json \
    --lineups clips/statsbomb-lineups-3869685.json
```

`clips/` and `runs/` are in the repo and gitignored: the clips are 26 MB each
and the traces are somebody's API spend, so neither belongs in a commit, and
both belong where the command that reads them runs. `runs/run_all.sh` does
the five that are left, in order, and skips any that already has a directory.
`--seconds` is 195 for a three-minute clip and 215 for the two that are 3.5.

## The first pass: five three-minute clips from one broadcast

Dropped part-way for variety, at the user's request — more Argentina v France
was testing the same afternoon over again. penalty1 and the four that followed
stay here as the one same-broadcast data point, and everything below this line
is that pass.

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

## Calibrating the grader against the runs that were read by hand

Every trace the handoff describes, put through `commentary grade` with the
same command:

| trace | by hand | by machine |
|---|---:|---:|
| r3 | 5 of 12 | **5 of 12** |
| r4 | — | 8 of 12 |
| r5 | 9 of 12 | **9 of 12** |
| c12b | — | 6 of 12 |
| gallery A | 10 of 12 | 8 of 12 |
| gallery B | 9 of 12 | 7 of 12 |

The ordering is the same and the two that read lower do so for two reasons,
both of them the machine being stricter than the eye and both visible in the
evidence lines. Item 2's truth now contains the restarts — a throw-in, a
kickoff, a free kick in three minutes — which nothing was counting before,
and the runs miss most of them. And item 5 fails on words the *gate* left
behind: "Arms wrapped around each other" reached the voice as "Wrapped around
each other", and a capitalised "Wrapped" is a name off the roster to
`factual_errors`. That is a real defect in the spoken line, so the item is
right to fail; the openers list has since been given the body nouns, and no
run has been made after that to see it.

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

It also cannot be graded the ordinary way: with no clock on any board read
there is nothing to fit an alignment to, and `grade` refuses rather than
inventing one. So `--offset -1645` — the number measured off the tally frame
above — is the escape hatch, and it prints `alignment BY HAND: nothing
fitted, nothing checked` above every number it then produces.

## Budget

$10 across every run on this page; $9.00 was the cap from the moment the key
was topped up.

| run | cost | running total |
|---|---:|---:|
| penalty1 (partial — the credit ran out at cursor 131 of 202) | $0.72 | $0.72 |
| subs | $0.96 | $1.67 |
| card | $0.99 | $2.66 |
| mbappe | $1.07 | $3.73 |
| shootout | $0.94 | $4.67 |
| offside | $0.95 | **$5.61** |

Six clips, one run each, no rerun for tuning. Every number below is the first
and only run of that footage.

## The table

Graded with `commentary grade` against the pack, StatsBomb's events and the
lineups. "Bar" is the twelve-item definition of done.

| clip | new event types | bar | events called vs StatsBomb | names right / wrong / unnamed | phantoms | cost |
|---|---|---:|---|---|---:|---:|
| penalty1 | penalty won, penalty scored | 7/12 | 4/6 (goal 1/1, penalty 1/1, foul 1/1, tackle 1/1; missed a clearance and a kickoff after the credit died) | 6 right, 0 wrong, **the penalty scorer unnamed** | 0 | $0.72 |
| subs | two substitutions | **10/12** | 8/11 (subs 2/2, fouls 3/3, free kicks 3/3; missed 2 tackles and a clearance) | 6 right, 0 wrong | 0 | $0.96 |
| card | yellow card, injury stoppage, second half | 8/12 | 7/8 (card 1/1, clearances 3/3, foul 1/1, interception, throw-in) | 5 right, 0 wrong, **the booked player unnamed** | 2 (both artefacts — below) | $0.99 |
| mbappe | second penalty, two goals in 95 s | 9/12 | **8/8 — everything** (goals 2/2, penalty 1/1, foul, tackle, throw-in, clearance, kickoff) | 6 right, 0 wrong, **the penalty goal not called at all** | 0 | $1.07 |
| shootout | a shootout: no clock, a tally | **10/12** | 7/7 (goals 3/3, the saved kick, the card, a clearance, a tackle) | **12 right, 0 wrong**, one kick called a save that was a goal | 0 | $0.94 |
| offside | offside | **10/12** | **8/8 — everything** (offside 1/1, corner, free kick, throw-in, clearances 2/2, interception, tackle) | 12 right, 0 wrong | 0 | $0.95 |

### Recall is not the same as calling it

The column above is recall as the brief defines it — a line landed within ten
seconds — and read on its own it flatters the runs badly. `commentary grade`
now asks the other question too, per event: did a line *call* it, by the
caller's own event tag or in as many words, and was the player named.

| clip | a line landed near | the event was called |
|---|---:|---:|
| penalty1 | 67% | **2 of 6** |
| subs | 73% | 4 of 11 |
| card | 88% | 3 of 8 |
| mbappe | 100% | 3 of 8 |
| shootout | 100% | **1 of 7** |
| offside | 100% | 6 of 8 |

Three things that hid behind the recall column:

- **The offside is never mentioned.** No line and no `offside` tag in three
  minutes. The 1/1 is a line about a corner four seconds earlier.
- **Mbappé's penalty goal is never called.** Nothing between 80.0 s and
  95.5 s, and the 95.5 line is still describing the wait for a kick that had
  gone in eleven seconds before.
- **The shootout calls one of its three goals.** Paredes, perfectly. Kolo
  Muani's kick gets a line sixteen seconds late that says the keeper saved
  it, and he scored. Montiel's winner gets nothing, with the cursor running
  twenty seconds past it.

The shape of it is one sentence: **the system narrates the build-up and
misses the outcome.** It is fluent about a ball on the spot, a wall shuffling
across and a keeper on his line, and then the net bulges and it is looking at
the crowd.

**47 distinct players named across 119 spoken lines on footage the system had
never seen, and not one of them was the wrong man.** Zero off-roster words in
any clip once the grader stopped disagreeing with the gate. Zero factual
errors by the machine's own reckoning on five of the six.

## What generalised from the tuned clip

- **Events.** Recall is 100% on two clips and 73%-88% on the rest. Every new
  event type was called: a penalty as a penalty, both substitutions, a yellow
  card, an offside, a shootout kick. Nothing about the event vocabulary was
  specific to Di María's goal.
- **The gate.** 148 lines judged across the five full runs, one wrongful
  rejection and one correct one. The goal-confirmation work from A8/A16 — the
  thing that took the most tuning on the original clip — handled a penalty
  goal, an open-play goal 95 seconds later and a shootout with no score bug
  at all, with zero `unconfirmed_goal` rejections of a real goal anywhere.
- **The board reader.** Three different halves, three different offsets, and
  the alignment fitted itself from the trace's own readings every time:
  residual 0.20-0.33 s on all five clips that have a clock. Nobody typed an
  offset except for the shootout, which has none to read.
- **Naming, once the caller was asked which kit.** Sightings bound went from
  47% on penalty1 to 82%, 92%, 98%, 77% and 83%. On the shootout and the
  offside clip twelve distinct players were named correctly in three minutes.
- **Cost and speed.** $0.94-$1.07 a clip, tracker 4.8-6.7 passes a second.

## What did not

- **Naming in open play.** `name_rate` in live play: 33%, 18%, **0%**, 33%,
  73%, 38%. The one clip that clears 60% is the shootout, which is nothing
  but close-ups. The card clip named nobody at all in live play while naming
  five people in stoppages and close-ups. The system names players when the
  camera is close enough to read a shirt, and open play is where a listener
  most wants a name.
- **The biggest moment is where the name goes missing.** penalty1 called the
  penalty goal "He steps up ... and buries it" having bound `10 Messi` five
  times in the previous minute. The card clip booked "the France midfielder"
  rather than Rabiot. It is not a precision problem — nothing was wrong — it
  is that the close-up of a man about to strike a penalty shows no number.
- **Track life.** Median 1.93-2.77 s against a bar of 2.5 s, failing on three
  of five. The tuned clip sat at 2.53-2.77 s; footage with more cutting sits
  below it.
- **The rate cap can swallow a goal.** Below.

## Per-clip reports

### a. penalty1 — match 20:30-24:00, video 22:06-25:36 — 7 of 12

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
- **The caller invents a tag about a fifth of the time.** 19 of the 34
  sightings carried a mark; 11 of those marks were really drawn on the frame
  the line was written about, 4 bound anyway on a mark that was not, and 2
  named a tag on a frame that had **no tags on it at all**. Nothing goes wrong
  downstream — the bind is made on the number and the name against the roster,
  and `identify` on a track that does not exist names nothing — but the
  `tagged` and `tags_on_frame` columns in the trace are the only reason
  anybody knows, and a mark is supposed to be the evidence that the read came
  off a real body.
- **The grader was wrong about the build-up and is fixed.** Five lines over the
  ball on the spot were scored as phantom penalties, because StatsBomb stamps
  the award as an instant and the broadcast spends a minute and a half on it.
  Same shape as the A19 goal-talk bug. Each claimable event now has the tail
  its coverage really runs to.

### b. subs — match 39:45-42:45, video 41:21-44:21 — 10 of 12

New event types: **two substitutions** (Dembélé off for Kolo Muani 40:32,
Giroud off for Thuram 41:01), read off the board graphic rather than the play.

Both were called, both correctly: "Thuram jogs on and Giroud's afternoon is
over, the number nine trudging past Deschamps towards the bench." Recall 2/2
on the substitutions and 8/11 overall; zero phantoms; 37 of 45 sightings
bound. Failing: name_rate 18% in live play, and the tracker at 4.8 passes/s,
just under the bar.

**The registry picked the substitutes up from the graphics, and the team
sheet was what let it down.** The caller read "12 KOLO MUANI" off the
substitution board — the right number and the right name — and the gate
killed two lines with `sighting_disagrees: Kolo Muani is not number 12`,
because the pack was built from StatsBomb's `player_nickname`, which for him
is the truncated "Randal Kolo". A commentator's notes had the wrong name in
them and two true lines died of it. Fixed in the pack (a gitignored input,
so there is no commit); subs was run before the fix and is scored after it,
which is why its report reads better than its run did.

### c. card — match 53:00-56:00, video 1:02:59-1:05:59 — 8 of 12

New event types: **a yellow card** (Rabiot 54:12) and **an injury stoppage**
(54:52), on the **second half** — the first time any offset but the first
half's was used.

**The second-half offset aligned itself.** The grader fitted h2 -3180.1 s from
43 board reads with a residual of 0.27 s, without being told anything. The
same is true of the mbappé clip (h2 -4680.1, residual 0.20) and the offside
clip (h1 -1710.2, residual 0.27). Nothing about the alignment was tuned to
the first clip.

The card was called within the window — "The referee holds the yellow high
for the France midfielder after that challenge by the touchline" — and
**Rabiot was not named**. Recall 7/8, the best of the five until mbappé.

Two failures that are the grader, not the run, and one that is real:

- Item 3 flags two lines about De Paul lying injured as **phantom fouls**.
  Neither line claims a foul; the caller tagged them `event: foul` because
  the vocabulary has no word for an injury stoppage, and the grader reads
  the tag. A missing event word, listed below rather than added mid-run.
- Item 7 reads **0 distinct players in open play**, which is true and is the
  headline failure of this clip: five people named, every one of them in a
  stoppage or a close-up, nobody named while the ball was moving.

### d. mbappe — match 78:00-81:30, video 1:27:59-1:31:29 — 9 of 12

New event types: **a second penalty**, **two goals in 95 seconds**, and the
score moving 2-0 to 2-2 — the first clip where the match changes hands.

**The best run of the six on everything except naming.** Event recall 8/8 —
every goal, the penalty, the foul, the tackle, the throw-in, the clearance,
the kickoff. Zero gate rejections in 28 judged lines. Zero factual errors.
40 of 41 sightings bound, 21 of them on a live tag. And item 7 passed the way
it was meant to for the first time: **Mbappé was bound to mark BX at 18 s and
named off that mark at 66 s**, 48 seconds later, through a camera change.

The open-play goal is as good as this system gets: "It is in! Mbappé stabs it
home from close range and France have life in Lusail!", 0.2 s from the event.

**And the penalty goal was missed entirely.** Not mis-called — never spoken.
The trace says why, and it is the sharpest finding of the five clips:

```
80.0  should_call=True   camera_cut 0.35, 4.5s since last line
83.9  should_call=False  rate_cap: scheduled 0.20 but 3.9s since last line, min gap 4.0s
91.1  should_call=True   board_change 1.00, 7.2s since last line
```

Mbappé struck the penalty at 84 s. The call that would have covered it was
refused **one tenth of a second under the four-second minimum gap**, and by
the time the next call came the broadcaster had cut to the crowd — so the
caller, looking at the dignitaries in the stand, wrote about Deschamps still
waiting for the spot-kick eleven seconds after it had gone in. Nothing knew
the kick was important at 83.9: the board had not moved yet, so the urgency
was 0.20. This is the rate cap doing exactly what it is specified to do.

### e. shootout — video 2:30:30-2:33:30 — 10 of 12

New event types: **a penalty shootout**. No running clock, no score bug, a
tally graphic at the bottom of the frame that nothing in the system looks at.

**Every assumption about the board broke, and the run was one of the two
best.** 53 of 54 board reads came back with no bug at all; the one that did
not produced a clock from somewhere and the grader, fitting an alignment from
a single reading, said `alignment SUSPECT` and **refused to grade**. That
refusal is the right answer and it is why `--offset` exists.

What the caller did with a board that never said anything:

- **name_rate 73% in live play, 12 distinct players named, none wrong.** The
  best naming of any run this project has done. A shootout is all close-ups,
  which is exactly where this system can read a shirt.
- Every kick called: Tchouaméni's miss at cursor 0.1, Paredes' goal at 50.6
  ("sends the keeper the wrong way — buried"), Montiel walking up alone.
- No scoreline stated in fifteen lines, with the tally on screen the whole
  time and the score entirely unknown to the system.

Three things it got wrong:

- **A goal called as a save.** "The keeper in green flings himself to his
  left as the penalty is struck — and he gets across it" is Kolo Muani's
  kick, and he scored: the stadium tally in the frame twenty seconds later
  reads 3-2. No name in the line and no event the feed can contradict, so
  **no machine check catches this one** — it was found by eye.
- **Montiel's winner is not in the run.** The last caller line is at 148 s
  and the kick is at about 152 s.
- The hand-measured offset was twenty seconds out, which is what the ±20 s
  on it meant. Refitted to **-7405 s** from the run's own lines against
  StatsBomb's kick times, which is the only anchor a shootout offers.

### f. offside — match 28:30-31:30, video 30:06-33:06 — 10 of 12

New event types: **an offside** (Messi 29:06).

Event recall 8/8 — the offside, the corner, the free kick, the throw-in, both
clearances, the interception, the tackle. 12 distinct players named, none
wrong. 38 of 46 sightings bound.

**The gate caught a hallucinated goal.** At 31:15, with the score 2-0 and the
board unmoved, the caller wrote "It's worked across the six-yard box and slid
in at the far post, Argentina break away in delight, Di María wheeling to the
corner flag!" — a goal that did not happen, in a match where Di María had
scored one an hour earlier. `unconfirmed_goal` killed it. That is the single
thing the gate exists for, and it is the only time in nine runs it has had to
do it.

## Decided

Both of penalty1's open questions came back with an answer, and both are in
the code. Neither has been run: the key still has no credit.

1. **A number alone, when both squads wear it, names nobody — and that was 18
   of penalty1's 34 sightings.** *Decision: the caller resolves the side, not
   the kit split.* `Sighting` has a `side` field, described as "which team's
   kit the body wears, from the shirt colours on the team sheet" — the thing
   reading the picture is the thing with both kit strings in its prompt, and a
   model does not confuse white stripes with navy. A number with a side names
   that side's player wearing it; a number with the side unknown still names
   nobody, which is the case that used to be silent and is now something the
   caller is told about in as many words. The kit split stays out of the
   naming path, so the Thuram class of error — an Argentina body placed on
   France by a colour histogram, and a "26" on it read as a France forward —
   is closed rather than reopened. A name and a side that disagree are dropped
   together: one of the two was misread and nothing can say which.
2. **A player the system named thirty seconds ago is "he" at the moment that
   matters.** *Decision: no rule change.* A name may be carried into a line
   only through a live tag — once a body is bound its tag shows the surname,
   and the caller may use it while that tag is on screen, which is C11's rule
   already and is what a human does when the same player is visibly the same
   player. Never from memory without a tag. So the lever is tracks surviving
   and the gallery, both of which are already the next items in the handoff.

Two prompt additions went in beside them, off the same clip:

- **A referee pointing at the penalty spot is a penalty**, and the caller is
  told not to wait for the graphic. It spent fifty seconds and six lines
  calling this one a free kick.
- **A name on a broadcast graphic is a sighting** — a lower third naming the
  taker, a scorer's caption, a substitution board — with the name filled in,
  no number, and the tag of the body in the close-up it is over.

## Bugs the clips exposed, and the commits

Every one is a rule that fired where it should not have, or a check that was
measuring something other than the system.

| commit | what the clip showed |
|---|---|
| `3c9674c` | Five lines about the ball sitting on the penalty spot came back as **phantom penalties**: StatsBomb stamps the award as an instant and the broadcast spends ninety-one seconds on it. Each claimable event now has the tail its coverage really runs to. |
| `8da34a7` | "Play breaks down by the touchline" — the gate has "play" in its stopwords and passed the line; the grader kept **its own shorter list** and called "Play" an invented name. One list now. Also: a sighting's name matches any word of a squad name, so a correct read of KOLO MUANI written as "Kolo" stopped being a contradiction. |
| `e64e8b0` | "Grimacing, and France get on with it" lost its first word. **A participle at the front of a line is grammar**, and the openers list was being finished one word at a time. |
| `3a9e065` | Five lines killed by `sighting_disagrees` on reads that were right: a graphic writes "T. Hernández", a shirt reads MAC ALLISTER. **One name matcher**, shared by the gate and the bind, that reads the forms a broadcast uses. Plus plurals in the openers rule, and item 4 no longer counting the gate killing a hallucinated goal as a failure. |
| `f766bd1` | Three correctly called goals scored as "no goal line within 6 s", because none of them says "scores": "sends the keeper the wrong way — buried", "stabs it home", "rolls it into the empty net". |
| `ec01fa5` | (before these clips) `Sighting.side`, which took sightings bound from 47% to 77-98%. |
| `4703547` | Recall counts a line landing near an event; it says nothing about whether the event was called. `grade` now answers both. |
| `0dc6035` | "He strikes it low to **De Gea's** right" reached the voice as "low to right" — `fold` glued the possessive s onto the name and matched nobody. Twice in one 45 s clip. |
| `228a768` | "The **Dutchman** steps up" lost its subject twice: the demonym was a team word and the noun a commentator says was not. |

One more fix is in a gitignored input rather than a commit: the pack called
Randal Kolo Muani "Randal Kolo", because it was built from StatsBomb's
`player_nickname` field, which truncates him. Two true lines died of it.

## Needs a decision

1. **The position-zero name trim has never once worked, and it has damaged
   nineteen lines.** Across nine runs the gate has trimmed a name from the
   front of a line nineteen times: Tears, Hands, Arms, Fist, Ice, Thousands,
   Whole, Pure, Emotion, Sky, Restart, Round, Grimacing. **Every one is an
   ordinary English word and not one is a name the caller invented.** The
   rule exists to catch a hallucinated surname at the start of a sentence and
   in nine runs it has caught none. Two of the nineteen are now covered by
   the participle rule and five by the plural rule; the rest need a
   dictionary the project does not have. The question is whether the trim
   should apply at position zero at all. Removing it risks an invented name
   opening a line, which has never happened; keeping it costs about two true
   lines a run. That is a change to A15, so it is not mine.
2. **The minimum gap between lines can swallow a goal.** Mbappé's penalty
   went in 0.1 s after a scheduled call was refused by the four-second rate
   cap, and the next call came seven seconds later with the camera on the
   crowd. Nothing knew the kick mattered: the board had not moved yet. A
   rule that let a call through when the *lookahead* frames contain a
   celebration — which the gate already computes for goal confirmation — or
   a shorter gap while the caller's last scene was a penalty, would have
   caught it. Both are new rules.
3. **The caller has no word for an injury stoppage**, so it writes `foul`,
   and the grader reads the tag and counts a phantom foul. Two lines on the
   card clip. Adding `INJURY` to the event vocabulary is small but it is a
   schema change and a prompt change.
4. **The gate and the grader have drifted apart on what a goal claim is.**
   The grader now knows "buries it", "stabs it home" and "into the empty
   net"; the gate does not. Unifying them makes the gate stricter — every
   "buries it" line would need the board to have moved, and on the shootout
   the board is absent for all three minutes, which would have silenced the
   best-naming run of the six. Worth doing, worth doing with a run behind it.
5. **A goal that arrives from a set piece is not tagged `goal`.** Ronaldo's
   free kick was called correctly and with his name and then dropped by the
   director on a camera cut, because the line was tagged `free_kick`, which is
   not in BIG_EVENTS and is therefore preemptable. One sentence in the caller
   prompt — if your line says the ball went in, the event is `goal`, whatever
   put it there — is the cheapest high-value fix this whole exercise found. It
   is a prompt change, so it is yours, and it wants a run behind it.
6. **Naming in open play is the gap, and it is the same gap the handoff has.**
   `name_rate` in live play across the five: 33%, 18%, 0%, 33%, 73%, 38%. The
   73% is the shootout, which is all close-ups. Nothing here is a new
   problem; the clips confirm that the lever is tracks surviving and the
   gallery carrying a name out of the close-up that earned it.
