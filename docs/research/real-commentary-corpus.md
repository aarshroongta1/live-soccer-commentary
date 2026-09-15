# What a whole match of real commentary sounds like

Seven broadcast feeds, 12.4 hours of live-window caption, 7,878 reconstructed
utterances, 78,963 words. Six are British club football; the seventh is the 2022 World Cup
final that `runs/prompt-name/REAL_COMMENTARY.md` measured, kept here as the
control so that the earlier study's numbers and this one's sit on the same
axis.

Everything below is counted off files in `clips/`. Every quote is verbatim from
a caption file, with the file and the video timestamp beside it. Nothing is
invented and nothing is drawn from memory of a match.

**Method, in one paragraph.** `commentary.grading.captions.load_json3` reads
each `.en.json3`; `scripts/build_commentary_examples.utterances` reassembles the
rolling caption stream into utterances by splitting on sentence-final stops, on
`>>` speaker-change markers, and on inter-segment gaps of 2 s or more. That is
exactly the reconstruction the earlier study used, and the control file
reproduces its figures to within a tenth of a point (1,537 utterances, median 6
words, mean 7.4, 8.2% one-word, 35.2% four-or-fewer). Bracketed tokens — `[Applause]`,
`[Music]`, `[cheering]` — are stripped before counting and an utterance that is
nothing but a bracket is dropped. Four of the six club matches are aligned to
StatsBomb event data by cross-correlating roster surnames against event times
(section 10); the two NBC matches are aligned to ESPN key events at minute
precision.

The headline, before the detail: **the earlier study measured one atypical
feed and the system was built to it.** The 2022 final is the sparsest,
shortest-lined, most fragmentary commentary in this corpus. British club
football is roughly 60% wordier per minute, uses a bare surname a third as
often, and fills the ninety minutes with numbers, threads and a second voice
that the system has almost none of.

---

## 1. The corpus

Live window only: pre-match studio and post-match wrap are excluded by the
window in the second column, chosen from the text (section 10 gives each).

| file | match | feed | live window (video s) | minutes | utterances | words | median words | mean | 1 word | ≤4 words | ≥9 words | ≥16 words |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `clips/pl-liv-mun-2025.en.json3` | Liverpool 2-2 Man Utd, PL, 5 Jan 2025 | NBC | 152–6110 | 99.3 | 923 | 9,291 | 8 | 10.1 | 6.0% | 26.3% | 46.9% | 20.2% |
| `clips/pl-tot-che-2024.en.json3` | Tottenham 3-4 Chelsea, PL, 8 Dec 2024 | NBC | 168–6150 | 99.7 | 975 | 9,788 | 8 | 10.0 | 6.1% | 25.0% | 46.2% | 18.6% |
| `clips/lei-mun-2015.en.json3` | Leicester 1-1 Man Utd, PL, 28 Nov 2015 | LCFC channel, **radio** | 70–5760 | 94.8 | 1,239 | 18,499 | 11 | 14.9 | 2.8% | 14.2% | 62.1% | 32.5% |
| `clips/lei-avl-2015.en.json3` | Leicester 3-2 Aston Villa, PL, 13 Sep 2015 | LCFC channel | 15–6100 | 101.4 | 811 | 7,671 | 7 | 9.5 | 8.1% | 33.7% | 41.9% | 17.8% |
| `clips/clasico-2017.en.json3` | Real Madrid 2-3 Barcelona, La Liga, 23 Apr 2017 | FCB channel, English | 282–6145 | 97.7 | 1,089 | 10,880 | 8 | 10.0 | 0.6% | 19.4% | 48.0% | 17.5% |
| `clips/bar-mal-2019.en.json3` | Barcelona 5-2 Mallorca, La Liga, 7 Dec 2019 | FCB channel, English | 243–6000 | 96.0 | 1,341 | 11,864 | 7 | 8.8 | 2.3% | 25.1% | 37.4% | 13.8% |
| **pooled, six club matches** | | | | **589** | **6,378** | **67,993** | **8** | **10.7** | **4.0%** | **23.3%** | **47.3%** | **20.2%** |
| `clips/argfra-dimaria.en.json3` | Argentina 3-3 France, WC final 2022 | FIFA | 60–9250 | 153.2 | 1,500 | 10,970 | 6 | 7.3 | 8.2% | 35.7% | 30.7% | 7.9% |

### What is different about British club feeds

| measure | 2022 FIFA final | pooled club football | ratio |
|---|---:|---:|---:|
| words per minute | 72 | 115 | ×1.6 |
| utterances per minute | 9.8 | 10.8 | ×1.1 |
| median words per utterance | 6 | 8 | ×1.33 |
| one-word utterances | 8.2% | 4.0% | ×0.49 |
| four words or fewer | 35.7% | 23.3% | ×0.65 |
| nine words or more | 30.7% | 47.3% | ×1.54 |
| sixteen words or more | 7.9% | 20.2% | ×2.56 |

Three things follow, and all three cut against the phraser's current prompt.

1. **The rate of utterances is nearly identical; the length is not.** Club
   commentary does not speak more often than the final did. It speaks at
   almost exactly the same rhythm and puts half again as many words into each
   turn. So the cadence the system was built for is roughly right and the
   *length* is not.
2. **The bare surname is a World Cup habit, not a football habit.** One word in
   twelve at the final; one in twenty-five in club football, and one in 160 in
   the Clásico feed. The phraser's prompt says "a good proportion of it should
   be one name … and nothing else", quoting 19%. That figure came from ten
   hand-picked live-play windows of one match.
3. **Long utterances are normal.** One club utterance in five is sixteen words
   or longer; at the final it was one in thirteen. The phraser's cap is 16
   words (`PhraserConfig.max_words`), which excludes a fifth of real commentary
   by construction.

### The long tail, and the word cap

| file | 95th percentile words | 99th percentile | longest utterance |
|---|---:|---:|---:|
| pl-liv-mun-2025 | 27 | 36 | 50 |
| pl-tot-che-2024 | 26 | 36 | 86 |
| lei-mun-2015 (radio) | 37 | 73 | 119 |
| lei-avl-2015 | 25 | 41 | 83 |
| clasico-2017 | 22 | 31 | 57 |
| bar-mal-2019 | 22 | 33 | 54 |
| argfra-dimaria (FIFA) | 18 | 26 | 39 |

The earlier study concluded that "the longest thing anyone said in half an hour
of live football was 28 words, which is exactly the current `max_words` cap —
the cap is right". Over whole matches that is wrong in both directions. The 2022
final's 95th percentile is 18 words and its longest utterance in 153 minutes is
39. In club football the 95th percentile is 22–27 and the longest utterance runs
to 50–86 words: the substitution announcement, the post-goal rebuild of the
move, and the colour voice's argument about a card all exceed 28 routinely. The
caller's cap of 28 is a reasonable backstop. The phraser's cap of 16 removes the
top fifth of the distribution.

`lei-mun-2015` is the outlier in the table and the reason is not noise: it is
**radio commentary**, not television. The commentator describes what a listener
cannot see — "The two teams are out and are shaking hands just in front of us in
the West End at the moment" (`lei-mun-2015` 0:04), "as we look out from the west
stand of the King Power Stadium" (1:39). At 195 words a minute it is nearly
twice as dense as any television feed here. It is included because its *event
coverage* and *thread* behaviour are excellent evidence, and excluded from the
register conclusions wherever that matters; where a number would be distorted
by it, the per-match column shows it.

---

## 2. Cadence

### 2.1 The gap number in the handoff is measuring caption segments, not speech

`docs/HANDOFF.md` section 4 says "Real commentary's median gap is 2.4 s but 22%
of gaps are over four seconds". Measured four ways on the same control file:

| definition | n | median | mean | share > 4 s |
|---|---:|---:|---:|---:|
| caption segment start → next segment start | 2,688 | **2.44 s** | 3.50 | **21.9%** |
| caption segment end → next segment start | 2,688 | 0.00 s | 1.09 | 8.1% |
| **utterance start → next utterance start** | 1,536 | **4.60 s** | 6.12 | **56.7%** |
| utterance end → next start (0.35 s/word) | 1,536 | 2.10 s | 3.74 | 29.0% |

The handoff's figure is row one: YouTube caption segments, of which there are
1.75 per utterance. The unit the system actually emits is an utterance, and
**the real inter-utterance gap is 4.6 s at the final and 4.3 s pooled across
club football, with more than half of all gaps over four seconds.**
`CallerConfig.min_gap_floor_s = 1.5` cites the 2.4 s figure in its docstring;
that floor is calibrated against the wrong unit.

### 2.2 Gap distribution, live window, per match

| match | gaps | median | mean | p25 | p75 | p90 | >4 s | >6 s | >10 s | >20 s | longest |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pl-liv-mun-2025 | 922 | 5.0 | 6.4 | 2.6 | 8.6 | 13.8 | 59% | 40% | 19% | 3% | 37 s |
| pl-tot-che-2024 | 974 | 4.7 | 6.1 | 2.2 | 8.5 | 13.2 | 56% | 39% | 19% | 3% | 34 s |
| lei-mun-2015 (radio) | 1,238 | 3.6 | 4.6 | 1.9 | 6.1 | 9.3 | 45% | 25% | 8% | 1% | 38 s |
| lei-avl-2015 | 810 | 5.9 | 7.5 | 3.0 | 10.1 | 15.9 | 66% | 49% | 25% | 5% | 41 s |
| clasico-2017 | 1,088 | 4.3 | 5.4 | 2.5 | 7.0 | 10.7 | 53% | 31% | 12% | 2% | 33 s |
| bar-mal-2019 | 1,340 | 3.5 | 4.3 | 2.1 | 5.9 | 9.0 | 43% | 24% | 6% | 0% | 26 s |
| **pooled club** | **6,372** | **4.3** | **5.5** | | | | **53%** | **33%** | **14%** | | |
| argfra-dimaria (FIFA) | 1,499 | 4.6 | 6.1 | 2.6 | 7.6 | 12.8 | 57% | 37% | 16% | 3% | 50 s |

### 2.3 By phase

Phases are assigned from StatsBomb events on the four aligned matches, by
priority: a replay/VAR keyword beats a stoppage, which beats an after-goal
window, which beats a dead-ball window, which beats an attacking window, and
anything unclaimed is build-up. `attacking` is ±10 s around a shot or ±5 s
around a cross or corner delivery; `dead_ball` is from a foul/throw/corner/goal
kick award to its delivery; `after_goal` is the 30 s after a goal; `stoppage`
covers cards, injuries and substitutions; `replay_var` is keyed off the
commentator's own language ("as we see that … once more", "having seen the
replay", "watch this").

| phase | gaps | share of match | median gap | mean | >4 s | >6 s | >10 s | median words/utt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| build-up possession | 2,517 | 54.6% | **4.2 s** | 5.3 | 53% | 31% | 11% | 8 |
| dead ball / restart | 1,119 | 24.3% | **4.5 s** | 5.8 | 55% | 36% | 15% | 10 |
| attacking move | 569 | 12.3% | **2.8 s** | 3.6 | 31% | 15% | 5% | **7** |
| stoppage (card, injury, sub) | 255 | 5.5% | 4.6 s | 5.8 | 56% | 36% | 13% | 9 |
| after a goal | 123 | 2.7% | 4.1 s | 4.8 | 51% | 29% | 7% | 7 |
| replay / VAR | 26 | 0.6% | 4.2 s | 4.5 | 62% | 27% | 8% | 10 |

**The shape is the opposite of a fixed rate.** In an attacking move the
commentator speaks half again as fast and says less each time: median gap 2.8 s,
median 7 words. In build-up and at a restart the gap opens to 4.2–4.5 s and the
lines get *longer*, because that is where the context goes. A system that runs
one cadence through both phases will sound too slow in the box and too busy on
the halfway line, and no prompt wording changes that — it is a rate decision.

### 2.4 After a goal there is no silence

The earlier study observed "Then silence for five seconds" after the Di María
goal. Across 19 goals in the four aligned matches, in the 30 s after the goal
event:

| | median | mean |
|---|---:|---:|
| words spoken | **60** | 63.3 |
| utterances | **7** | 7.6 |
| gap from the call to the next line | **2.0 s** | |
| longest gap anywhere in the 30 s | **7.1 s** | |
| utterances carrying a number | **2** | |

Sixty words in thirty seconds is the *fastest* sustained talking in the match.
The five seconds of silence at the 2022 final is a World Cup final's pause for
a crowd, and it is not what happens in a league game. Verbatim, all six of the
first goals in the aligned matches:

```
bar-mal-2019  10:26  Griezmann, he's in a one-on-one.
              10:28  GRIEZMANN CHIPS THE KEEPER.
              10:28  It's 1-0 to Barcelona.
              10:33  Quick thinking by ter Stegen and Griezmann celebrates.
              10:49  A quick ball out by ter Stegen and Griezmann gets his fifth goal of the season.

clasico-2017  32:27  IT'S OFF THE POST.
              32:27  SURELY NOW IT'S CASEMIRO.
              32:31  WELL, it's the first goal of the game.
              32:34  And it's scored by Casemiro.
              32:38  It's his third goal of this La Liga campaign.
              32:42  It was an excellent cross that was put back into the box by Marcelo.
              32:49  The shot rebounded off the post and fell very kindly for Carlos Henrique Casemiro who slotted the ball home

lei-mun-2015  24:40  CHANCE FOR A SHOT.
              24:40  BODY.
              24:40  IT'S IN.
              24:44  LISTEN TO THE NOISE.
              24:44  THAT'S THE SOUND OF PREMIER LEAGUE HISTORY BEING MADE BY JAMIE BARDY.
              24:53  11 CONSECUTIVE GOALS IN PREMIER League games.
              24:56  IT'S A MAGNIFICENT RECORD AND LEICESTER CITY ARE ONE UP AGAINST MANCHESTER United after 24 minutes.

bar-mal-2019  46:26  SUáREZ OH MY!
              46:26  OH MY!
              46:26  OH MY!
              46:30  SUáREZ HAS MADE IT 4-1.
              46:30  It's not It's the way he did it.
              46:36  He's backheeled the ball into the goal.
              46:41  And another standing ovation.
              46:45  It's exhibition stuff.
              46:47  Welcome to the home of the Harlem Globetrotters.

clasico-2017 101:03  OH!
             101:03  HE'S DONE IT!
             101:05  LEO MESSI, WITH WHAT COULD BE THE LAST KICK OF THE GAME, BURIES IT INTO THE BACK OF THE NET.
             101:10  IT'S HEARTBREAK FOR REAL MADRID, BUT IT'S JUBILATION FOR BARCA.
             101:17  WELL, they've done a Real Madrid.
             101:18  They've scored a last-minute goal.
             101:22  92 minutes are on the clock and Barca have put themselves back in front.
             101:26  And it's Leo Messi with his 500th Barca goal, his 23rd goal in El Clasico, and his 31st strike of the season

lei-avl-2015  39:09  Sanchez struck it.
              39:09  It's come back to Richards again.
              39:10  And Greenish.
              39:14  This time it happens for the youngster.
              39:17  And Jack Greenish, just three days after his 20th birthday, gets the perfect present for a Villa fan.
              39:24  His first ever goal for the club.
```

Note the structure, which is the same every time: a shout of two to five
fragments, then within about four seconds a **number** (the scoreline, the
scorer's tally, or a record), then within about fifteen seconds a **rebuild of
the move in past tense** naming two or three players, then a colour line.

### 2.5 After a shot or a save

| event | window | median words | median utterances | median first gap | median longest internal gap |
|---|---:|---:|---:|---:|---:|
| shot saved / keeper save | 20 s | 41 | 4 | 3.2 s | 6.7 s |
| shot off target | 20 s | 45 | 4 | 3.7 s | 7.3 s |
| foul | 15 s | 28 | 3 | 4.1 s | 6.5 s |
| corner awarded | 15 s | 35 | 3 | 3.8 s | 6.4 s |

### 2.6 The longest silences, and what was happening

Every gap over 30 s in the six club matches is either a dead ball, a stoppage,
or ordinary midfield build-up with nothing in it. Six examples with the line
either side:

```
lei-mun-2015  37.8 s at 67:22 [dead ball — Rooney limping]
   before: You would think that because Rooney is still just walking gingerely across the pitch...
   after : He's having a long chat with with Gray Shakespeare down there, isn't he?

lei-mun-2015  37.2 s at 7:29 [build-up]
   before: Here's Chris Mlin who's parted this Manchester United defense that's conceded just nine goals this season
   after : He tries to get away from Michael Brighton who was trying to get the ball back

lei-mun-2015  33.0 s at 30:36 [build-up]
   before: He's got the ball on the right hand side if he can find Riyad Mahrez but it's intercepted by Young
   after : He really does look out of touch, Michael Carrick.

pl-liv-mun-2025  37.1 s at 45:29 [end of the half]
   before: Not necessarily at the break, but before too long of the second half.
   after : Maz Rawi Robertson that won it.

pl-tot-che-2024  33.1 s at 69:53 [build-up]
   before: Porro.
   after : Venna again.

lei-avl-2015  40.7 s at 52:17 [after a chance, into replay]
   before: Jamie Vardy that was the only one open to distinctive moment back from Jamie Vardy...
   after : get the crowd behind in this les team
```

The pattern worth copying: **a long silence is preceded by a long line and
followed by a short one.** The commentator finishes a thought, stops, and comes
back in with a bare name or a fragment when the ball next matters.

---

## 3. What is said for each kind of event

All rates below are measured on the four StatsBomb-aligned matches
(`lei-mun-2015`, `lei-avl-2015`, `clasico-2017`, `bar-mal-2019`) by mapping each
StatsBomb event to video time and looking at the caption utterances that start
in a window around it. The default window is [−1 s, +5 s]; goals, cards,
substitutions, injuries and offsides use wider windows because the reaction is
slower (goal, card, offside: [−2, +12]; substitution: [−5, +20]; injury: [−2, +15]).

Four columns, and they mean different things:

- **any%** — at least one utterance starts in the window. Because commentary is
  near-continuous this is close to a measure of how busy the feed is, not of
  coverage. Read it only as a ceiling.
- **silent%** — *nothing at all* is said within ±3 s of the event. This is the
  real "passed in silence" number.
- **named%** — the StatsBomb player of the event is named in the window, matched
  fuzzily (difflib ratio ≥ 0.75) to tolerate auto-caption mangling. This is a
  **floor**: the captions render Grealish as "Greenish", Vardy as "Bardy" and
  Kulusevski as "Kulleski", and some of those fall below the threshold.
- **label%** — the event is named by its own vocabulary (a corner is called a
  corner, a save a save).

| kind | n | any% | silent% | named% | label% | median words in window |
|---|---:|---:|---:|---:|---:|---:|
| carry, build-up | 3,121 | 77 | **24** | **38** | — | 14 |
| pass, build-up (<25 m) | 2,306 | 77 | **24** | **29** | — | 14 |
| throw-in, delivery | 173 | 57 | **37** | 11 | 14 | 18 |
| throw-in, award | 172 | 75 | 28 | — | 20 | 16 |
| tackle | 168 | 74 | 21 | 28 | 12 | 17 |
| block | 147 | 80 | 19 | 22 | 8 | 16 |
| clearance | 139 | 79 | 14 | 32 | 24 | 20 |
| free kick, delivery | 91 | 67 | 33 | 15 | 8 | 15 |
| free kick, award | 87 | 74 | 24 | — | 20 | 16 |
| cross | 84 | **90** | **11** | **50** | 21 | 18 |
| switch of play | 82 | 74 | 26 | 29 | 12 | 12 |
| foul | 78 | 74 | 28 | 27 | 23 | 13 |
| goal kick, award | 69 | 81 | 16 | — | 14 | 17 |
| goal kick, taken | 69 | 61 | **43** | 4 | **1** | 18 |
| interception | 60 | 82 | 17 | 37 | 13 | 15 |
| shot off target | 43 | 88 | 9 | 49 | 47 | 16 |
| keeper save | 42 | 86 | 7 | 38 | 26 | 16 |
| corner, award | 41 | 85 | 20 | — | 44 | 16 |
| corner, delivery | 41 | 85 | 17 | 22 | 37 | 14 |
| shot saved | 40 | 88 | **0** | 60 | 30 | 16 |
| kickoff / restart | 26 | 85 | 31 | 0 | 23 | 18 |
| shot blocked | 21 | 90 | 5 | 71 | 33 | 15 |
| substitution | 20 | 100 | 20 | 55 | 55 | 50 |
| goal | 19 | 100 | 11 | **79** | 68 | 34 |
| half start | 16 | 100 | 12 | — | — | 17 |
| half end | 16 | 88 | 0 | — | 12 | 15 |
| yellow card | 12 | 92 | 33 | 58 | 42 | 32 |
| injury stoppage | 11 | 100 | 45 | 36 | 55 | 26 |
| offside | 2 | 100 | 0 | 50 | 50 | 39 |
| red card | 1 | 100 | 0 | 100 | 100 | 50 |

### 3.1 The three findings that matter most

**a. A quarter of all touches pass in complete silence and two-thirds are never
named.** 24% of carries and passes in build-up have nothing said within ±3 s,
and even counting generously only 38% of carries have their player named. The
phraser's rule "Name first. If the form gives you a player, the line starts with
that player" describes what happens to roughly a third of touches, not all of
them.

**b. A goal kick is the commentator's slot for something else.** 43% pass in
silence, and when there *is* speech the event word "goal kick" appears in 1% of
cases. What actually happens is that the goal kick is where the storyline goes:

```
lei-avl-2015   7:56  been in fine goal scoring form for Villa Scott Sinclair with five in four
                     appearances so far this season.
               8:01  helped of course by the hattick he managed in the Capital One Cup against Knots County
bar-mal-2019  23:01  I mean they Again, with all respect to Mallorca, there's a difference, isn't there?
              23:04  I mean, if you look for example at Barcelona's wage bill, it's 671 million and
                     Mallorca's is under 30.
lei-mun-2015  33:34  Well, what's interesting about that is Vardy goes away from the ball as far away
                     from the ball as he can to leave the space
clasico-2017  17:35  And actually should potentially have been sent off before Arturo Vidal was shown a
                     second yellow card.
```

**c. The crossing of the ball is the one thing that is never missed.** The cross
has the highest any% (90%), the lowest silence (11%) and the highest naming rate
of any build-up event (50%). If one event kind has to be got right, it is the
ball into the box.

### 3.2 Verbatim, by kind

**Build-up: a pass, a carry.** The dominant shapes are a name chained with a
comma, `Here's <Name>`, `Now <Name>`, and `<X> to <Y>`. Note that the participle
form the phraser prompt leans on is comparatively rare here (section 8.3).

```
pl-liv-mun-2025   9:20  Mati Delft.
                  9:59  Diaz, Robertson,
                 10:09  Mallister, Kenate, who's missed eight games with a knee injury.
                 10:16  Jones.
                  5:00  Diego Dalo to Bruno Fernandez and back to Kobe Mayu.
                  7:38  Here's Salah.
                 17:08  Here's Mallister.
pl-tot-che-2024  69:53  Porro.
                 70:42  Silanki.
                 70:42  Kaisedo.
bar-mal-2019      8:42  Back to the action, it's De Jong to Messi, back to De Jong on the edge of the area.
                 13:51  Now, Griezmann.
clasico-2017     10:41  Sergio Busquets into Samuel Umtiti.
                 10:44  Here's Andres Iniesta.
                 56:22  Picked up by Leo Messi.
lei-mun-2015      2:58  Here's Wes Morgan again.
                  2:58  Patient from Leicester City as Drinkwater drops deep to get the ball.
```

**Switch of play.** Shortest median line of any kind (12 words). The word
"switch" itself is used 15 times in 68,000 words; the commoner form is the
direction.

```
pl-liv-mun-2025  12:18  It was good possession as Rawi switching the ball to the left.
clasico-2017     63:04  Gerard Piqué tries to switch play over to the far side, but Jordi Alba can't keep that one in play
lei-mun-2015     10:18  which switches play finds McNair now Chris Smalling United happy to play
lei-mun-2015     58:32  does well to find Carrick who under pressure from Drinkwater plays it to McNair
                        who sprays it out to the right where Darmian can pick it up
pl-liv-mun-2025   4:38  Down the left now.
clasico-2017     74:34  Play switched to the far side by Real Madrid, and it's picked up by As[ensio]
```

**Cross.**

```
bar-mal-2019      6:59  Goes all the way across, headed back into the penalty spot, and headed away by Salva Sevilla.
bar-mal-2019     10:08  Swung in and all the way across.
lei-mun-2015     82:26  Dangerous cross to the back post towards Mata onto his left foot
lei-mun-2015     28:46  And it could be worse here for Manchester United as dangerous ball goes in from Buch's
pl-liv-mun-2025  22:26  Yeah, excellent ball in by Dalow.
pl-liv-mun-2025  14:22  What a ball into Jones.
lei-avl-2015     38:38  Crosswoods corner in towards Micah Richards.
lei-mun-2015     45:25  Oh, it's a dangerous ball in and I think it came up ahead of Wes Morgan last.
```

**Shot off target.** 47% get the event word; the recurring shape is
`<Name> <verb>` then a direction fragment.

```
clasico-2017     14:49  SUAREZ STRIKES ONE.
                 14:51  And it's wide of the far post, but that's the first opportunity we've had so far in this game.
clasico-2017     27:59  He's annoyed with himself as he blasts it over the crossbar and into the stands behind.
clasico-2017     22:48  It was a sort of half-volley on the edge of the area.
                 22:50  But he sprayed the ball well wide of the target.
bar-mal-2019     31:14  NO, SUAREZ SUAREZ OFF the inside of the post.
lei-avl-2015      7:21  tries to land it up for Srira and the angle was just too tight for him.
lei-avl-2015      3:33  And Kuzan very nearly caught on the hop there by Danny Drinkwater.
pl-tot-che-2024  26:09  It's a lovely strike just slightly underneath it.
lei-mun-2015     31:50  MORRIS WITH THE LOW DRIVE.
```

**Shot saved / keeper save.** 0% silent — this is the one event kind that is
always called.

```
clasico-2017     24:28  Ronaldo shot, good save from Ter Stegen and hooked clear by Samuel Umtiti.
clasico-2017     22:36  It's a comfortable save yet again for Ter Stegen.
                 22:38  The Frenchman didn't get a lot behind it in fairness.
clasico-2017     10:04  He shoots and it's a comfortable save in the end for Ter Stegen,
                        although the ball did bounce just in front of him.
lei-avl-2015     24:14  and he strikes it straight at Casper Schmeichel.
lei-avl-2015     88:17  Riyad Mahrez takes on Ami looks to bend one and Kuzan sees it all the way.
lei-mun-2015     69:25  OH, HIS SHOT SAVED BY THE LEGS OF DE GEA.
lei-mun-2015     52:55  Good save from Schmeichel, but right at Rooney, tried to adjust his body,
                        Huth who did just enough to put him off.
bar-mal-2019     83:12  OH, WHAT A SAVE BY TER STEGEN.
bar-mal-2019     25:42  Piqué with the header but uh it was a tame one in the end and no problems there for Reina.
```

**Block, tackle, interception.** Highest naming rate of all (71% for blocks)
because the point of the line *is* who did it.

```
bar-mal-2019      7:15  Messi inside the area, shoots.
                  7:17  It's blocked.
                  7:17  Um but it'll fall now to Griezmann.
bar-mal-2019     80:10  Now, the shot is blocked.
bar-mal-2019     38:13  Suarez advances, shoots, comes off Gamez.
clasico-2017     13:57  And it was blocked by Nacho.
lei-mun-2015     72:05  Chance to shoot here with his right foot, but it's charged down by Michael Carrick
                        and will go out for a throw in
lei-mun-2015     22:35  Excellent block from Wes Morgan I think it was.
lei-avl-2015      1:44  And now Okazaki hits that one into the face of Lecot.
pl-liv-mun-2025   7:41  Cut out by Maguire.
pl-liv-mun-2025  25:32  It's well blocked initially by Kenate
lei-mun-2015      8:22  it's well read led by Mark Albright who gets a foot in but does concede the corner
```

**Corner, awarded.** 44% name it as a corner; the line is usually *how it came
about*, not "corner".

```
bar-mal-2019     17:30  Roberto plays it in.
                 17:30  Comes off a defender.
                 17:33  It'll be a corner to Barcelona.
bar-mal-2019      9:29  OOH, and that goes out for a corner, and almost could have been an own goal there.
bar-mal-2019      6:34  Barca with a corner on the right-hand side as we look at goal.
clasico-2017     51:15  And Barca have got a corner.
                 51:15  They're going to have to take it quickly.
lei-mun-2015     23:08  In the end it comes off the legs of Danny Simpson and out for a United corner,
                        the second of the game.
lei-mun-2015     80:29  and in the end I think it was Danny Simpson that was forced to head it behind
                        for a seventh corner of the game.
lei-avl-2015     15:39  Away.
```

**Corner, delivered.** Note "In by X" and "X's corner" as the two commonest
forms, and that the taker is named only 22% of the time.

```
lei-avl-2015     35:26  Brighton's corner.
                 35:26  Richards gets up ahead of Wes Morgan this time.
lei-mun-2015      8:58  Corner will be taken by the left foot of Daley Blint.
lei-mun-2015     56:01  comes in from Daily Blind.
                 56:03  It's another dangerous one and it almost is allowed to bounce by Hood
lei-mun-2015     82:37  This might be United's best spell for a while and oh corner taken quickly
bar-mal-2019     10:08  Swung in and all the way across.
clasico-2017     51:56  Sends it sailing in towards the back post and what a chance.
                 51:58  What a chance for Leo Messi.
clasico-2017     32:20  It's punched clear by Ter Stegen.
lei-avl-2015     57:54  We have the corner.
lei-mun-2015     46:12  Corner in from Blind.
```

**Free kick.** Only 8% of deliveries get called a free kick; the award is
described as a foul.

```
lei-avl-2015     84:11  Gat is taken down by Anovi and concedes a free kick.
                 84:14  That's as good as a corner for Leicester.
clasico-2017     74:34  And it's a foul by Paco Alcácer on Casemiro and a free kick for Real Madrid.
lei-mun-2015     35:54  Not really any appeals for a free kick from the England captain
clasico-2017     18:15  The free kick goes Barca's way.
lei-mun-2015     15:48  He wants a free kick to the Leicester fans not given.
bar-mal-2019     79:57  uh Mallorca needed a free kick there, but the advantage was played.
lei-mun-2015      3:28  and Craig Pson's got a pull play back for a foul by Okazaki on Patty McNair.
```

**Throw-in.** 37% of deliveries pass in silence; the award is described, the
throw itself almost never is.

```
lei-mun-2015      1:22  Three Leicester players trying to get a challenge and in the end it's Jamie Vardy
                        who snaps in his heels and gets a throw in for his efforts
lei-mun-2015     12:17  He is forced to put the ball out for a throw in level with the 18 yd area.
lei-mun-2015      5:44  It's won first by Santio Darmian who puts the ball out for a throwing under pressure
clasico-2017      5:01  It's a poor pass from Sergio Ramos out for a Barca throw on the far side.
clasico-2017      5:30  Too much on it and it's out of play for a Barca throw.
clasico-2017      7:06  This time the Frenchman wins the ball and puts it out for a throw-in.
bar-mal-2019      5:51  He wins a throw-in off Sergi Roberto, halfway inside the Barcelona half.
lei-mun-2015     12:54  Looks like it might be a long throw from the Austrian fullback
lei-avl-2015      1:28  Riyad Mahrez actually took the throw in while he was on the ground, but the referee
                        did want to check first whether medical treatment was needed.
lei-avl-2015      1:35  So Richie Dat now does take the throw.
```

**Goal kick.** See 3.1b. When it *is* called it is called in passing:

```
lei-mun-2015     13:45  It's a deep one to the back post and it will miss everybody and go out for a goal kick.
lei-mun-2015     27:28  Back in on the volley by Stiger but that was poor by his standards and out for a goal kick.
lei-mun-2015     35:54  as the ball runs down for a goal kick for Leicester.
pl-liv-mun-2025  75:19  It's a goal kick that's given as Alexander Arnold ends up crouched over a pile of snow
clasico-2017     90:31  The linesman missed it, so too did the referee and awarded Real Madrid the goal
                        Barca the goal kick.
```

**Foul.** 23% get the word; the shape is `<Name> goes down under <something>`.

```
bar-mal-2019     29:38  combining with Budimir who goes down under the attentions of Junior.
bar-mal-2019     36:16  That could Ooh, Busquets has been knocked over there, and uh the fans wanting a
                        yellow card for Joan Sastre.
clasico-2017     29:50  Handball by Modric according to the referee.
lei-avl-2015     11:42  Morgan Schlo had a push in the back then from Michael Richards.
lei-avl-2015     27:12  Richards makes the tackle and it was an earlier tackle actually on Danny Drinkw[ater]
lei-mun-2015     30:02  Leicester City are pressing really high up the pitch, which is good to see.
pl-liv-mun-2025  13:28  given up by Mayuatte catching Gako
pl-liv-mun-2025  67:36  Oh, they're appealing for a hand ball in there.
```

**Booking.** The highest-variance kind: 33% of yellow cards pass in silence
within ±3 s (the card is shown well after the foul), but the window that does
catch it runs long (median 32 words) and is dominated by the colour voice
arguing about it.

```
lei-avl-2015     80:45  Are you fouled by D?
                 80:56  And I think for the protest as much as the offense, Dier is booked.
clasico-2017     16:20  It's a really poor challenge from Casemiro.
                 16:27  To be fair to the Real Madrid midfielder, he went over to Leo Messi immediately
                        helping the Argentine to his feet
clasico-2017     42:46  He's challenged by Samuel Umtiti.
                 42:50  Well, it could be a yellow card for the Frenchman.
                 42:51  That's certainly what the Real Madrid players are suggesting.
clasico-2017     87:07  It's a ridiculous challenge from the Real Madrid captain.
                 87:11  It looks worse every time you see it.
                 87:13  And I think the referee's got that one absolutely right.
bar-mal-2019     11:44  But now Mallorca on the attack and that that's possibly a yellow card there for Sergi Roberto.
                 11:51  We know that uh referees don't like you pulling players back.
bar-mal-2019     72:56  As ever, Piqué gets a head to it, but uh did he lead with his elbow?
                 73:01  That's what the referee seems to think.
clasico-2017     43:25  So, two players in the referee's book.
                 43:27  Samuel Umtiti for Barca and Casemiro for Real Madrid.
```

**Offside.** Only two in the StatsBomb data for these four matches, but both get
a long window (median 39 words) and both quote a flag.

```
lei-mun-2015     52:20  It was an excellent save by Schmeichel in the first effort to deny the German and
                        then I think Rooney was given offside.
lei-avl-2015     93:56  Whether the linesman flag for offside here, but he knew he was going to get hit by it.
argfra-dimaria  144:01  AND THE FLAG IS UP.
argfra-dimaria   30:44  Chased up by Tagliafico. It's Di María who's offside.
argfra-dimaria  129:56  OFFSIDE.
argfra-dimaria  130:00  THE REFEREE'S GIVEN THE GOAL.
```

**Substitution.** The longest windows in the corpus (median 50 words). The
structure is: name off, name on, one fact about the man coming on.

```
lei-avl-2015     68:30  Ganin replaced by Angolo Kante and Leonardo Aaha coming on in place of former Villa
                        man Mark Allbrighton
lei-avl-2015     49:26  But Nathan Dy making his first appearance for Leicester, but his 350th in English
                        competitive football
clasico-2017     78:34  He's going to be replaced by Kovacic.
                 78:46  There's a change, too, for Football Club Barcelona.
                 78:49  Paco Alcácer has been taken off and replaced by André Gomes.
clasico-2017     43:25  And it's Gareth Bale who's forced off to be replaced by Marco Asensio.
bar-mal-2019     64:43  And on will come Hernández for his debut.
                 64:52  Cucho Hernández on loan from Watford.
bar-mal-2019     77:15  25 minutes gone now in the second half, and Aleñá will come on for Rakitić.
pl-tot-che-2024  80:49  Substitution for Chelsea number 15.
pl-tot-che-2024  95:16  Palmer number 14
lei-mun-2015     63:53  So he's on and he certainly poses a different kind of threat.
```

**Injury and stoppage.** 45% silent within ±3 s — the commentator waits, then
narrates.

```
lei-mun-2015     81:11  Went off for the ball, but that was a clash of heads between he and Robert Huth.
                 81:13  Both down at the minute, although Huth is just
                 81:17  getting to his feet.
lei-avl-2015     25:04  Wonder if this is a a scheduled water break or not, but certainly one or two of the
                        players taking on the opportunity to [drink]
clasico-2017     24:43  He's spitting blood onto the field of play.
pl-liv-mun-2025   2:43  And Manchester United already have a casualty down on the turf.
                  2:47  Plays allowed to continue for a while by our match referee.
                  2:53  Yeah, it's Mayu who's down, John.
                  2:53  And he's not moving at all at the moment.
                  3:02  It's just that left boot there catching Mayu in the head.
                  3:04  Complete accident.
pl-tot-che-2024  48:33  Romero with treatment to his thigh.
lei-avl-2015      1:27  But he's been very slow getting back to his feet.
```

**VAR and replay talk.** Rare and formulaic. 26 utterances across the four
aligned matches match the replay vocabulary. The commonest openers are "As we
see …", "Having seen the replay …", "Watch this."

```
clasico-2017     41:40  As we see Leo Messi's goal.
                 41:43  Wonderful turn to get beyond Dani Carvajal.
                 41:47  It was a good first time pass as well from Ivan Rakitić.
                 41:50  Brilliantly finished off by Leo Messi.
clasico-2017     37:24  Watch this.
                 37:24  Rakitic into Messi.
                 37:24  Brilliant touch from Messi to get away from Dani Carvajal and a fine finish beyond Keylor Navas
clasico-2017      6:41  Let's look at that incident once more.
clasico-2017      8:31  As we see that penalty appeal once more and every time you look at it, it looks less
                        and less like there was enough contact
clasico-2017     71:09  Having seen the replay, Suárez played the ball while he was down on the ground.
clasico-2017     75:31  You see in the replay the ball actually came off Ronaldo's knee.
clasico-2017     41:24  Well, we've just seen confirmation there that Sergio Ramos certainly wasn't offside.
bar-mal-2019     40:29  Like to see it again, Sastre.
bar-mal-2019     39:33  So, the goal is being checked, and now we'll go up on the scoreboard.
pl-tot-che-2024  13:04  There's a little check about Kun Song being potentially in the way.
pl-tot-che-2024  33:30  The check, the shot, the smart reactions from Forster.
```

**Kickoff and restart.** Never names the player taking it (0%).

```
lei-mun-2015      1:13  We're underway at the King Power Stadium.
lei-mun-2015      1:18  Referee Craig Porson has blown his whistle and Manchester United have the ball
                        inside their own half with Daily Blind on the left hand side.
lei-avl-2015     49:20  We are ready to go for the second half.
pl-liv-mun-2025  49:39  And we're back underway with Liverpool as they like to do attacking the top end away
                        to the right in the second half.
clasico-2017     52:34  Hello and welcome back to the Santiago Bernabéu to the second half of this enthralling
                        clásico between Real Madrid and Football Club Barcelona.
clasico-2017      5:10  Barca kicking from right to left in this first half.
clasico-2017     54:36  Barca in the second half playing from left to right.
lei-avl-2015     75:47  The referee's pointed back to the center circle
```

**Half-time and full-time whistle.**

```
lei-mun-2015     48:12  Half time at the King Power Stadium.
clasico-2017     52:22  An entertaining first half of football here at the Santiago Bernabéu and it finishes
                        Real Madrid 1, Barca 1.
clasico-2017    102:20  The final whistle has gone and this goal from Leo Messi has handed Barcelona
                        all three points.
pl-liv-mun-2025 101:48  It's finished 2-2 and Graeme Lasso, hats off to the lot of them in dreadful conditions
                        for a terrific second half.
pl-tot-che-2024  61:36  4-1 Chelsea at finished.   [a score from another ground]
pl-liv-mun-2025  46:50  There will be two minutes of Lexus stoppage time.
pl-tot-che-2024  95:41  and an overworked fourth official Lewis Smith actually has amended did eight to
                        7 minutes of Lexus stoppage time.
bar-mal-2019     51:13  Just 30 seconds of additional time to be played at the end of this first half.
```

**Score and clock restatement.** Section 5.3.

---

## 4. Lead versus colour

### 4.1 Telling the two apart

YouTube writes a `>>` speaker-change marker into some files and not others:

| file | markers | share of utterances that begin a marked turn |
|---|---:|---:|
| argfra-dimaria | 544 | 24% |
| lei-mun-2015 | 205 | 16% |
| pl-tot-che-2024 | 154 | 12% |
| pl-liv-mun-2025 | 103 | 10% |
| lei-avl-2015 | 18 | 2% |
| clasico-2017 | 0 | 0% |
| bar-mal-2019 | 0 | 0% |

The markers are under-inserted everywhere and absent in two files, so they
cannot be the classifier. They are usable as **validation**. Define a *colour
entry* as an utterance opening with one of `well`, `yeah`, `yes`, `i think`,
`i mean`, `you know`, `absolutely`, `exactly`, `for me`, `listen`. Then:

| file | colour-opener rate, all utterances | rate among `>>`-marked turn starts | lift |
|---|---:|---:|---:|
| pl-liv-mun-2025 | 8.0% | 29.1% | ×3.6 |
| argfra-dimaria | 5.6% | 17.1% | ×3.1 |
| pl-tot-che-2024 | 7.3% | 20.7% | ×2.8 |
| lei-mun-2015 | 14.1% | 36.6% | ×2.6 |

A three-fold lift on every file with markers. The opener is not a perfect
speaker label, but it is a strong one, and it is the one used for the rest of
this section. The word "Well" alone accounts for 184 utterance-openings in the
club corpus (2.77%) and "Yeah" 148 (2.23%); at `>>`-marked turn starts "yeah" is
the single most common first word in all four marked files.

The other reliable tell is that **the two voices address each other by name**.
This is worth copying directly:

```
pl-liv-mun-2025   0:30  And I think we all know about the front line, Rebecca, how well that's working.
pl-liv-mun-2025   0:58  Robbie, what about the personnel?
pl-liv-mun-2025   1:33  It was a line earlier on John Champion News, but that's worth repeating.
pl-liv-mun-2025   2:24  Grae Lau alongside me.
                  2:24  Can you construct a case for Manchester United getting anything here today?
pl-liv-mun-2025   2:53  Yeah, it's Mayu who's down, John.
pl-liv-mun-2025 101:48  It's finished 2-2 and Graeme Lasso, hats off to the lot of them
pl-tot-che-2024   2:52  And Graeme Lo, you were an integral part of that rivalry.
                  2:53  What does Tottenham Chelsea mean to you?
lei-mun-2015      0:42  And Mickey, what do you think of the way that Manchester United are lining up?
lei-mun-2015     30:04  The thoughts of Mickey Adams and Neville Fulgger on Fox's player HD
lei-avl-2015     49:22  And let's hand you back to Alan Smith and Rob Paw.
```

### 4.2 When the colour voice speaks

| phase | utterances in phase | colour entries | entries per 100 utterances |
|---|---:|---:|---:|
| replay / VAR | 26 | 4 | **15.4** |
| stoppage (card, injury, sub) | 255 | 30 | **11.8** |
| after a goal | 123 | 13 | 10.6 |
| dead ball / restart | 1,119 | 117 | **10.5** |
| build-up | 2,521 | 149 | 5.9 |
| attacking move | 569 | 12 | **2.1** |

**The colour seat is five times less likely to speak during an attacking move
than during a stoppage.** It is not on a timer; it is gated on the ball being
dead or the picture being a replay.

### 4.3 How long after the event

For each big event, the delay to the first colour entry within 60 s:

| event | n with a colour entry inside 60 s | median delay | p25 | p75 | within 6 s |
|---|---:|---:|---:|---:|---:|
| goal | 13 | **16.7 s** | 4.4 | 25.8 | 31% |
| shot saved | 22 | 13.6 s | 9.4 | 25.1 | 9% |
| shot off target | 20 | 15.5 s | 11.2 | 31.9 | 5% |
| corner awarded | 22 | 15.5 s | 9.1 | 38.6 | 14% |
| foul | 37 | 16.7 s | 12.1 | 31.9 | 5% |
| yellow card | 7 | 17.0 s | 5.6 | 32.0 | 29% |
| substitution | 7 | 17.1 s | 6.4 | 19.3 | 29% |
| injury | 8 | 17.9 s | 14.9 | 18.7 | 0% |
| shot blocked | 9 | 37.3 s | 24.4 | 44.9 | 11% |
| **pooled, nearest big event** | 174 | **21.4 s** | 12.3 | 34.8 | **7%** |

`docs/HANDOFF.md` section 3e specifies a colour seat that "reacts two to four
seconds after a big moment". **The corpus says twelve to thirty-five seconds,
and only 7% of colour entries land within six seconds of an event.** The two-to-
four-second window is where the *lead* is still talking — rebuilding the move,
giving the tally, restating the score. The colour seat comes in after that, when
the replay is up or the ball is being placed.

The exception is the goal, where 31% of colour entries do arrive inside six
seconds — but what arrives is a reaction fragment, not analysis:

```
clasico-2017     32:31  (+3.2 s)  WELL, it's the first goal of the game.
clasico-2017     94:19  (+4.4 s)  Well, well, well.
bar-mal-2019     39:02  (+3.8 s)  Well, they they didn't see that one coming.
lei-mun-2015     24:44  (+8.5 s)  LISTEN TO THE NOISE.
```

### 4.4 How long it runs and what it says

After a colour entry the voice holds the microphone for a **median of 4
utterances** (mean 4.4, n = 325). 20% are a single utterance; **51% run four or
more**. That is a long turn by the standards of this system:
`AnalystConfig.max_words = 30` is one utterance's worth, and 30 words is about
what the colour voice says in its *first* utterance alone.

Fifteen colour entries with the lead's line before and the colour voice's
continuation after:

```
[lei-avl-2015 67:32, +19.0 s after a goal]
  before : They've been under real pressure since the second half began but they break and
           that was a deadly finish.
  COLOUR : Well, I've got to say if anything that was even better than Jack Greish's effort
           as the ball's traveling to him, not slowly by any means of the control on this
           shot is absolutely first class.
  then   : Schmeichel yet again cannot get a glove on it. / Another example, a pure quality
           finish and it starts outside the post and curls in. / What a beauty.

[lei-avl-2015 86:16, +25.8 s after Vardy's equaliser]
  before : But from somewhere they summon the spirit and JD Vardy comes up with the equalizer.
  COLOUR : Well, it's classic Leicester City, isn't it?
  then   : The Leicester City we've come to know over the last 6 months or so. /
           Morris is the man. / He's the man here.

[clasico-2017 101:17, +14.4 s after Messi's winner]
  before : IT'S HEARTBREAK FOR REAL MADRID, BUT IT'S JUBILATION FOR BARCA.
  COLOUR : WELL, they've done a Real Madrid.
  then   : They've scored a last-minute goal. / 92 minutes are on the clock and Barca have
           put themselves back in front.

[bar-mal-2019 71:01, +43.1 s, over a VAR check]
  before : Um So, the scoreboard still shows 4-1, but can't see why that wouldn't be a goal.
  COLOUR : I think unless there was a offside on Gamez.
  then   : Um Nobody claimed it. / Nobody claimed it cuz it didn't exist.

[lei-mun-2015 9:33, +12.0 s after a save]
  before : Either way, Casper Schmeichel clutches the ball to his chest.
  COLOUR : Well, Lester have got to be very, very careful.
  then   : You know, you give the ball away cheaply to good uh players then, you know,
           you could get punished there. / I agree with you.

[lei-mun-2015 52:55, +41.0 s after a save]
  before : Good save from Schmeichel, but right at Rooney, tried to adjust his body,
           Huth who did just enough to put him off.
  COLOUR : Yeah, unlike the corners there from Leicester's point of view, they've uh
           they've gone zonally rather than man-to-man there.
  then   : And you know, any sort of movement across them will cause them problems.

[lei-avl-2015 13:21, +7.2 s after a save]
  before : Mahrez.
  COLOUR : Well, you don't want to be giving him a side of goal the way he's playing this season.
  then   : No. / And he won't hesitate to let fly the confidence that he's playing with.

[lei-avl-2015 50:26, +10.7 s after a shot]
  before : It's all Bright with a shot this time.
  COLOUR : Well, again, you'd expect a quick start.
  then   : The response from Leicester first five or 10 minutes. /
           Sure, Tim Sherwoods mentioned that in the dressing room. / Be ready for it.

[clasico-2017 10:28, +23.5 s after a save]
  before : But it gets saved from Ter Stegen.
  COLOUR : Well, what a start to this game.
  then   : Barca finally managing to get their foot on the ball here.

[clasico-2017 22:46, +9.6 s after a shot]
  before : The Frenchman didn't get a lot behind it in fairness.
  COLOUR : Well, it's end-to-end stuff.

[lei-avl-2015 93:35, +36.2 s after the winner]
  before : He's really hurt himself in getting there.
  COLOUR : Well, concern here for Nathan Dier as you would expect.

[lei-mun-2015 3:45, in quiet build-up]
  COLOUR : Yes, I think in the early stage we were already seeing what we expected.
  then   : Man United playing with a very strong defense and and protection in front of them,
           but already Kante showing what he can do in midfield.

[lei-mun-2015 4:08, quiet build-up, humour]
  before : You don't get much more experience than Michael Carrick and World Cup winner
           Bastian Steinsteiger.
  COLOUR : Oh, you don't.
  then   : No. / Looking at United though, I think they've lined up with three at the back.

[pl-tot-che-2024 3:23, quiet build-up]
  before : It's a corner.
  COLOUR : Spurs might have made more of that.
  then   : Well, they should have made more of it.

[pl-liv-mun-2025 1:00, pre-kickoff]
  before : Robbie, what about the personnel?
  COLOUR : Yeah, and actually on a slight in that interview did say that that he thinks this
           is their strongest lineup and I tend to agree.

[lei-mun-2015 32:28, +43.2 s after a chance]
  COLOUR : You know that's how good it was.
  then   : Uh but who said that they can't play you know in the last third of the pitch there.
```

### 4.5 What the colour voice talks about in quiet build-up

Five recurring kinds, in rough order of frequency:

1. **A pattern that has now repeated.** "unlike the corners there from
   Leicester's point of view, they've gone zonally rather than man-to-man there"
   (`lei-mun-2015` 52:55); "Kaisedo started off on paper as a right back. As soon
   as Chelsea get the ball, he moves into midfield" (`pl-tot-che-2024` 4:39).
2. **A shape or personnel observation.** "Looking at United though, I think
   they've lined up with three at the back. You know, I think Ashley Young's
   playing ever so deep" (`lei-mun-2015` 4:08); "Change in shape for Chelsea in
   midfield" (`pl-tot-che-2024` 52:58).
3. **A prediction with a time-box.** "Alston Villa know they should know that
   Leicester are going to start quickly here. Try and build momentum in the
   first 10 minutes" (`lei-avl-2015` 1:53).
4. **A storyline picked back up.** Section 7.
5. **Humour and the direct address.** "Welcome to the home of the Harlem
   Globetrotters" (`bar-mal-2019` 46:47); "Oh, you don't. / No."
   (`lei-mun-2015` 4:08); "You could put 14 on sometimes, couldn't you?"
   (`bar-mal-2019` 85:58).

### 4.6 Handover

The lead hands over with a direct question ("Robbie, what about the personnel?",
"And Mickey, what do you think of the way that Manchester United are lining
up?", "Can you construct a case for Manchester United getting anything here
today?"). The colour voice hands back by **stopping mid-thought when the ball
moves**; there is no verbal hand-back in the corpus. The lead simply starts a
play line, and the commonest first words of that line are `Here's`, `Now`, `And`
and a bare name. In the phase data this shows as the colour rate collapsing from
10.5 per hundred at a dead ball to 2.1 per hundred once the move starts.

---

## 5. Numbers, statistics and context

### 5.1 Rate

| file | utterances | carrying a number | share | per minute | carrying an off-pitch fact | per minute |
|---|---:|---:|---:|---:|---:|---:|
| pl-liv-mun-2025 | 983 | 158 | 16.1% | 1.53 | 29 | 0.28 |
| pl-tot-che-2024 | 1,038 | 205 | 19.7% | 1.95 | 31 | 0.29 |
| lei-mun-2015 | 1,262 | 250 | 19.8% | 2.59 | 29 | 0.30 |
| lei-avl-2015 | 825 | 145 | 17.6% | 1.42 | 30 | 0.29 |
| clasico-2017 | 1,141 | 219 | 19.2% | 2.13 | 23 | 0.22 |
| bar-mal-2019 | 1,385 | 226 | 16.3% | 2.25 | 21 | 0.21 |
| argfra-dimaria | 1,537 | 225 | 14.6% | 1.44 | 34 | 0.22 |

"A number" is any digit or number word including ordinals and "hat-trick". "An
off-pitch fact" is a tighter pattern: a year, a record, a run of results, a
transfer, an age, a previous season, a league position.

**One utterance in six carries a number, and that is roughly two a minute for
the whole ninety.** The system's v3 trace produced two numbers in 27 lines, one
of which was the scoreline and one of which came from a pack note.

Colour-opener lines are *not* where the numbers live. The share of
number-carrying lines that open like the colour voice is 3.4–12.8%, against a
base rate of 2.4–14.1% — within noise on every file. **Numbers are the lead's
job**, dropped into the flow of play description, not the analyst's set piece.

### 5.2 What kind of number

Pooled over the six club matches, 1,203 number-carrying utterances:

| category | count | share of number-carrying lines |
|---|---:|---:|
| scoreline | 276 | 22.9% |
| clock (minutes gone, remaining, stoppage) | 116 | 9.6% |
| form (run, unbeaten, points, table) | 95 | 7.9% |
| a player's tally (goals, assists, appearances) | 50 | 4.2% |
| history (since a year, a record, last season) | 40 | 3.3% |
| shirt number / position | 8 | 0.7% |
| money / transfer | 4 | 0.3% |
| age | 3 | 0.2% |
| distance | 2 | 0.2% |
| attendance | 1 | 0.1% |

(Categories overlap and the remainder is numbers used in ordinary speech — "one
or two", "the first half".)

Verbatim, by category:

```
tally     bar-mal-2019  21:47  That's his 12th goal of the season, his 10th league goal and it's
                               his 13th in 13 games against Mallorca.
tally     bar-mal-2019  10:49  A quick ball out by ter Stegen and Griezmann gets his fifth goal of the season.
tally     clasico-2017  32:38  It's his third goal of this La Liga campaign.
tally     clasico-2017  81:39  It's his seventh La Liga goal of the season.
tally     clasico-2017 101:26  And it's Leo Messi with his 500th Barca goal, his 23rd goal in El Clasico,
                               and his 31st strike of the season
tally     pl-tot-che-2024 89:50  12 out of 12 for the club in total.
tally     pl-tot-che-2024 65:54  10 out of 10.
tally     pl-liv-mun-2025  0:44  17 goals and 13 assists in 18 Premier League.
tally     lei-mun-2015   2:12  13 goals from 13 games this season including 10 in consecutive games
                               equally Rubisto's record for Manchester United.

form      pl-liv-mun-2025  0:51  They're unbeaten in their last 23 in all competitions.
form      pl-liv-mun-2025  0:53  It's their fifth longest unbeaten run in club history.
form      pl-liv-mun-2025  0:49  Liverpool on pace for 95 Premier League points.
form      pl-liv-mun-2025  2:17  The nervous visitors, Manchester United in white, five defeats in their
                               last six league games.
form      pl-tot-che-2024 61:29  It's the end of Tottenham's unbeaten start to the season of 10 matches
form      clasico-2017   0:36  They're three points behind Real Madrid going into this game.
form      clasico-2017 102:37  Real Madrid and Barca both have 75 points.
form      bar-mal-2019  66:48  Mallorca just one point outside the relegation zone.

history   pl-liv-mun-2025  1:40  They've not gone four without scoring since 1909.
history   pl-liv-mun-2025  0:00  Manchester United haven't scored a goal since Jesse Lingard strike six years ago.
history   lei-mun-2015  85:31  Incidentally, the record from Jimmy Dunn was set by uh Sheffield United player.
history   bar-mal-2019  19:13  Although, he did score, of course, here a month ago in the 4-1 victory over Celta Vigo.
history   bar-mal-2019  24:13  on their last two visits here, Mallorca have um come away with 5-0 defeats
history   pl-tot-che-2024 4:00  He scored a hat-tick on this ground last season in a chaotic 4-1 Chelsea victory.
history   pl-tot-che-2024 91:56  They were two up at Brighton at Halime and lost 3-2.
history   argfra-dimaria 139:29  THE FIRST WORLD CUP FINAL HAT-TRICK SINCE HURST IN '66.

age       lei-avl-2015   1:40  Clearance there by Jack Greenish who's just turned 20 years of age.
age       lei-avl-2015  39:17  And Jack Greenish, just three days after his 20th birthday, gets the perfect
                               present for a Villa fan.
money     bar-mal-2019  23:04  if you look for example at Barcelona's wage bill, it's 671 million and
                               Mallorca's is under 30.
career    lei-avl-2015  49:26  Nathan Dy making his first appearance for Leicester, but his 350th in English
                               competitive football
other     pl-tot-che-2024 81:45  Daniel Leik chairman here for the last 23 years.
                        81:50  He's gone through a whole series of managers, 11 in total.
```

### 5.3 How the score and the clock are restated

Two different broadcast cultures, and the split is sharp:

| file | score restatements | one every | clock restatements | one every |
|---|---:|---:|---:|---:|
| bar-mal-2019 | 40 | **150 s** | 21 | **287 s** |
| lei-mun-2015 | 22 | 263 s | 11 | 526 s |
| clasico-2017 | 13 | 474 s | 7 | 881 s |
| pl-tot-che-2024 | 13 | 486 s | 3 | 2,105 s |
| pl-liv-mun-2025 | 10 | 618 s | 3 | 2,059 s |
| lei-avl-2015 | 9 | 683 s | 0 | — |
| argfra-dimaria | 8 | 1,176 s | 4 | 2,352 s |

The club-channel feeds (`bar-mal-2019`, `clasico-2017`) restate constantly for
viewers joining late; the domestic NBC feeds barely do, because the score bug is
on screen. `bar-mal-2019` does it on an almost literal five-minute timer and the
shape is always the same **two short utterances, clock then score**:

```
 14:21  10 minutes gone.
 14:21  1-0 to Barcelona.
 19:28  15 minutes gone.
 19:33  1-0 to Barcelona.
 24:07  20 minutes gone in the game.
 24:10  Now it's 2-0 to Barcelona.
 27:20  Halfway through the first half, Barcelona leading Mallorca 2-0.
 29:46  It's two 25 minutes gone in the first half, Saturday night at the Camp Nou.
 29:54  Barca leading 2-0.
 32:53  Back to Valjent, and then good back to the scoreline here, 2-0 to Barcelona.
 34:54  Half hour gone in the Camp Nou, Barcelona still winning this game 2-0 as Messi
        receives from De Jong, plays into Griezmann.
 62:04  10 minutes into the second half, still Barcelona four, Mallorca one.
 65:18  13 minutes gone in the second half.
 65:18  It's 4-1 to Barcelona.
 66:41  15 minutes into the second half, still 4-1 and they come now.
 70:40  19 minutes into the second half.
 75:56  24 minutes into the second half.
 77:15  25 minutes gone now in the second half, and Aleñá will come on for Rakitić.
 86:39  Barca leading 4-2 as we approach the last 10 minutes.
 87:18  We're into the last 10 minutes.
 94:01  It's 5-2 to Barcelona.
 95:39  We're into the last minute here.
```

The vocabulary is a small closed set: `N minutes gone`, `N minutes into the
second half`, `halfway through the first half`, `half hour gone`, `we're into
the last ten minutes`, `as we approach the last N minutes`, `N-N to <team>`,
`<team> leading <team> N-N`, `still N-N`, `<team> still winning this game N-N`.
Numbers are always words or digits read as words; never "the score is".

---

## 6. Names

Measured on the four StatsBomb-aligned matches, against the actual team sheets.

| file | utterances | carrying a roster surname | bare name only (≤2 words) | share of named utterances that are bare | first name + surname | substitutes for a name |
|---|---:|---:|---:|---:|---:|---|
| lei-mun-2015 | 1,262 | 537 (43%) | 15 (1.2%) | 2.8% | 149 (27.7%) | nationality 17, position 8, keeper 5, age 1 |
| lei-avl-2015 | 825 | 187 (23%) | 35 (4.2%) | 18.7% | 40 (21.4%) | age 2, keeper 1 |
| clasico-2017 | 1,141 | 643 (56%) | 41 (3.6%) | 6.4% | 312 (48.5%) | nationality 31, position 4, keeper 2 |
| bar-mal-2019 | 1,385 | 777 (56%) | 70 (5.1%) | 9.0% | 132 (17.0%) | nationality 20, keeper 10, position 3 |

Five findings.

**a. The bare surname is much rarer than the earlier study implied.** 1.2–5.1%
of all utterances against the 19% the phraser prompt quotes. The 19% came from
ten hand-picked live-play windows at a World Cup final; over a whole club match
the figure is a quarter of that. The correct statement for a prompt is: *a bare
surname is a legal and idiomatic line, and it is roughly one line in twenty-five,
not one in five.*

**b. The full name is the club-football habit.** Between 17% and 48% of
name-carrying utterances use first name plus surname. The Clásico feed says "Leo
Messi" and "Ivan Rakitić" and "Carlos Henrique Casemiro" almost every time. NBC
alternates: the first mention in a passage is full ("Here's Bruno Fernandez",
"And Jack Greenish, just three days after his 20th birthday"), later ones bare.

**c. The substitutes are nationality first, then position.** Across the four
matches the corpus uses `the Frenchman`, `the Argentine`, `the Croatian
forward`, `the Austrian fullback`, `the German`, `the Uruguayan`, `the
Welshman`, `the England captain`, `the Real Madrid midfielder`, `the Leicester
winger`, `the youngster`, `the keeper`, `the big center back`, `the Chelsea
loanee`. Nationality is the commonest by roughly four to one over position.

```
lei-mun-2015  12:54  Looks like it might be a long throw from the Austrian fullback
clasico-2017  22:38  The Frenchman didn't get a lot behind it in fairness.
clasico-2017   7:06  This time the Frenchman wins the ball and puts it out for a throw-in.
clasico-2017  16:27  To be fair to the Real Madrid midfielder, he went over to Leo Messi immediately
clasico-2017  87:07  It's a ridiculous challenge from the Real Madrid captain.
bar-mal-2019   5:33  Budimir, the Croatian forward, plays it back to Joan Sastre.
bar-mal-2019  70:34  The Croatian forward celebrating as uh well, he should.
bar-mal-2019   7:57  There's picked up now by Baba, the Chelsea loanee
bar-mal-2019  10:33  Quick thinking by ter Stegen and Griezmann celebrates.
bar-mal-2019  10:55  Piqué congratulating the German keeper.
lei-mun-2015  35:54  Not really any appeals for a free kick from the England captain
lei-avl-2015  39:14  This time it happens for the youngster.
lei-mun-2015  72:05  Oh, one by Albright and great play from the Leicester winger.
clasico-2017   6:52  There is contact from the big center back on Ronaldo.
```

**d. A name carries for about ten seconds.** The median interval between two
mentions of the same surname within 30 s is 8.3 s (`bar-mal-2019`), 9.8 s
(`clasico-2017`), 11.8 s (`lei-mun-2015`) and 12.4 s (`lei-avl-2015`). After
that the commentator re-names rather than using a pronoun. Inside that window
"he" and "him" carry it.

**e. When the commentator cannot identify a player, he says so, in one of four
fixed forms.** This is the single most copyable behaviour in the corpus for a
system whose open-play naming is bounded by pixels.

```
hedge on the name
  lei-mun-2015  22:35  Excellent block from Wes Morgan I think it was.
  lei-mun-2015  25:54  I think it was Fes, wasn't it?
  lei-mun-2015  39:03  in the end it was I think it was Scheider in there and got the final touch
  lei-mun-2015  44:30  I think it was Rooney who finally stopped him.
  lei-mun-2015  53:19  I think it was whose boot caught him in the in the stomach.
  lei-mun-2015  63:38  Well, I think it was Simpson there that was challenging him
  argfra-dimaria 30:38  In by Messi, out I think by Giroud from the near post.
  bar-mal-2019  46:11  It's a level 42 on the scoreboard.

refuse the name
  lei-mun-2015  25:06  I'm not really sure who played it through, but the first time that Vardy has
                       been allowed to get beyond the Manchester United back three
  lei-mun-2015  42:34  That was a terrific kick out that found well it was helped on its way by one
                       of the defenders
  lei-mun-2015  45:25  Oh, it's a dangerous ball in and I think it came up ahead of Wes Morgan last.

name the set, not the man
  lei-mun-2015  36:49  One of them is Martial who gets away from Wes Morgan.
  lei-mun-2015  69:44  One of them Wayne Rooney is now on the ball on the right hand side.

name the act, not the actor
  lei-mun-2015   9:28  Was it the cross?
  lei-mun-2015   9:28  Was it the shot?
  bar-mal-2019  70:13  Headed down.
  bar-mal-2019  70:13  Oh, it's another goal.
  lei-avl-2015  15:39  Away.
```

---

## 7. Threads

A thread is a fact planted early, paid off at an event, and picked up again
later. Every one below is a real recurrence with timestamps. This is what
"picking something back up" looks like.

### 7.1 Vardy's consecutive-scoring record (`lei-mun-2015`, 13 mentions over 94 min)

The archetype: **set up in the second minute, paid at the goal, revisited twice
in the second half, closed at full time.**

```
  2:12  13 goals from 13 games this season including 10 in consecutive games equally
        Rubisto's record for Manchester United.
  2:22  He can set a new one today by scoring in an 11th [Small sends the ball up...]
 24:44  THAT'S THE SOUND OF PREMIER LEAGUE HISTORY BEING MADE BY JAMIE BARDY.
 24:53  11 CONSECUTIVE GOALS IN PREMIER League games.
 24:56  IT'S A MAGNIFICENT RECORD AND LEICESTER CITY ARE ONE UP AGAINST MANCHESTER
        United after 24 minutes.
 26:15  Jamie Vardy, where you just can't stop him scoring in the minute.
 35:35  Top of the league and the record scorer.
 84:44  He scored in 11 consecutive games now.
 84:49  The all-time record is actually 12 set by Jimmy Dunn in the 1930s.
 85:31  Incidentally, the record from Jimmy Dunn was set by uh Sheffield United player.
 96:07  He scored for the 11th consecutive Premier League game.
 96:10  A remarkable record set by the England international.
```

### 7.2 Cucurella's two slips (`pl-tot-che-2024`, 15 mentions over 99 min)

**A mistake becomes a running thread, is checked when the player does the same
thing again, and is recalled at full time.**

```
  7:40  Cookera with a massive part to play in terms of the slip.
 12:55  TWO SLIPS BY KUKERA.
 13:11  He's got to change his boots.
 14:18  He should know in the warm up that that footwear isn't going to work.
 28:48  Cookareo who now has the right boots on didn't slip.
 52:12  One of the talking points raised by the first period was the fact that both
        Tottenham goals stem from moments where the Chelsea left back Kukera slipped
        and then he went sheepishly to change his footwear but by that stage his team
        was two down.
 93:56  and to do so having conceded two very sloppy early goals both a result of their
        left back cooker slipping makes it even more impressive
 95:01  So another Chelsea player slipping over in a key area of the pitch.
```

### 7.3 Casemiro on a booking (`clasico-2017`, 5 mentions over 62 min)

**A card is carried for an hour and is the reason a substitution is explained.**

```
 16:17  Great skill from Leo Messi to get away from Casemiro and that should be a yellow card.
 16:54  Casemiro already in the referee's book.
 54:09  A bit of controversy at the end of the first half when he could have potentially
        picked up a second one and been given a red.
 54:24  Although Casemiro will have to be very wary in this second half.
 78:24  Casemiro, who's on a yellow card, is making way.
```

### 7.4 Messi's 500th Barcelona goal (`clasico-2017`, 6 mentions over 101 min)

**Set up before kickoff as a conditional, and closed by the last kick.**

```
  3:01  He's also on 498 Barca goals, so a brace here today would see him reach 500.
 36:49  [first goal — not counted aloud at the time]
 52:54  before half-time Leo Messi leveled things with his 22nd goal in this fixture.
101:03  LEO MESSI, WITH WHAT COULD BE THE LAST KICK OF THE GAME, BURIES IT INTO THE BACK
        OF THE NET.
101:26  And it's Leo Messi with his 500th Barca goal, his 23rd goal in El Clasico, and his
        31st strike of the season, who makes it Real Madrid 2, Barca 3.
```

### 7.5 Manchester United's goalless run (`pl-liv-mun-2025`)

```
  0:00  to Anfield where Manchester United haven't scored a goal since Jesse Lingard
        strike six years ago.
  1:35  Three straight Premier League defeats for Manchester United without scoring.
  1:40  They've not gone four without scoring since 1909.
  1:42  Will today be the day?
 83:52  MANCHESTER UNITED HAVE DONE IT AGAIN.
 84:00  Well, THEY HAVE STRUGGLED FOR GOALS ALL SEASON LONG, BUT THEY'VE MANAGED TWO
        AT ANFIELD.
```

Note the payoff line at 84:00: it is the colour voice, and it closes a thread
set up before kickoff, 83 minutes earlier.

### 7.6 The snow (`pl-liv-mun-2025`, 9 mentions over 98 min)

**A condition thread, referred to whenever the pictures show it.**

```
  3:49  We had significant overnight snow in much of the UK
  3:54  You'll see snow piled up around the pitch.
  3:56  When we arrived 4 hours ahead of kickoff, it was still subject to a light covering.
  5:56  Just on the borderline between freezing rain and snow at the moment
 30:18  Neu going tumbling in the snow.
 54:39  I said to you at Halime, I just wonder if the conditions are playing into the poor
        quality that we've seen
 55:09  I think there snow shovel probably hindering rather than helping the ground staff
 75:19  It's a goal kick that's given as Alexander Arnold ends up crouched over a pile of snow
101:48  hats off to the lot of them in dreadful conditions for a terrific second half.
```

### 7.7 Grealish's 20th birthday (`lei-avl-2015`)

```
  1:40  Clearance there by Jack Greenish who's just turned 20 years of age.
 39:17  And Jack Greenish, just three days after his 20th birthday, gets the perfect
        present for a Villa fan.
 39:24  His first ever goal for the club.
 67:32  Well, I've got to say if anything that was even better than Jack Greish's effort
```

### 7.8 Palmer's penalties (`pl-tot-che-2024`)

```
 65:18  Chelsea's undisputed penalty king.
 65:48  Phil Palmer who scored a penalty on this ground last season.
 65:54  Perfect from the spot.
 65:54  10 out of 10.
 88:36  for the second time in 22 minutes, it's Cole Palmer.
 89:48  Two out of two from the spot today.
 89:50  12 out of 12 for the club in total.
```

**The shape of a thread, generalised.** Three slots, and the corpus fills all
three every time: a *setup* before or early in the match with the number in it;
a *payoff* within four seconds of the event, still carrying the number; and one
to four *callbacks*, each of which restates the number in a new form ("11
consecutive games" → "the record scorer" → "he scored in 11 consecutive games
now" → "the 11th consecutive Premier League game"). A thread that fires once is
not a thread.

---

## 8. Register

### 8.1 A British vocabulary, counted

71,252 words of club football against 11,384 words of the 2022 final — whole
files here, not the live windows of section 1, because a word's rate does not
depend on where kickoff was. Rates per 10,000 words.

| word | club count | per 10k | final count | per 10k |
|---|---:|---:|---:|---:|
| area | 118 | 16.6 | 4 | 3.5 |
| oh | 76 | 10.7 | 16 | 14.1 |
| box | 59 | 8.3 | 7 | 6.1 |
| post | 55 | 7.7 | 3 | 2.6 |
| headed | 48 | 6.7 | 0 | 0.0 |
| head / heads | 53 | 7.4 | 9 | 7.9 |
| clear / cleared / clears | 68 | 9.5 | 6 | 5.3 |
| poor | 36 | 5.1 | 0 | 0.0 |
| brilliant | 29 | 4.1 | 3 | 2.6 |
| lovely | 25 | 3.5 | 2 | 1.8 |
| quality | 18 | 2.5 | 1 | 0.9 |
| keeper | 16 | 2.2 | 2 | 1.8 |
| crossbar | 12 | 1.7 | 0 | 0.0 |
| equalizer / equaliser | 11 | 1.5 | 0 | 0.0 |
| press / pressing | 16 | 2.2 | 0 | 0.0 |
| magnificent | 10 | 1.4 | 2 | 1.8 |
| terrific | 10 | 1.4 | 0 | 0.0 |
| shield | 10 | 1.4 | 0 | 0.0 |
| switch / switched / switches / swung | 20 | 2.8 | 1 | 0.9 |
| flick / flicked / flicks | 17 | 2.4 | 1 | 0.9 |
| booked / booking / cautioned | 16 | 2.2 | 0 | 0.0 |
| wonderful | 8 | 1.1 | 2 | 1.8 |
| beauty | 7 | 1.0 | 1 | 0.9 |
| slot / slots | 9 | 1.3 | 0 | 0.0 |
| overlap | 7 | 1.0 | 0 | 0.0 |
| robbed / rob | 10 | 1.4 | 0 | 0.0 |
| ooh | 6 | 0.8 | 0 | 0.0 |
| outstanding | 5 | 0.7 | 0 | 0.0 |
| chips / chipped | 8 | 1.1 | 0 | 0.0 |
| sloppy | 4 | 0.6 | 0 | 0.0 |
| scramble | 2 | 0.3 | 0 | 0.0 |
| rifles | 2 | 0.3 | 0 | 0.0 |
| scuffed | 2 | 0.3 | 0 | 0.0 |

Words the **FIFA feed** used more than club football, which is a list of words
that have leaked into the phraser's examples and should not dominate them:

| word | final per 10k | club per 10k |
|---|---:|---:|
| glorious | 1.8 | 0.0 |
| sumptuous | 1.8 | 0.0 |
| dragged | 1.8 | 0.0 |
| dispossessed | 1.8 | 0.1 |
| nick / nicked | 2.7 | 0.7 |
| rhythm | 2.6 | 0.6 |
| superb | 2.6 | 0.4 |
| clever | 1.8 | 0.6 |
| bursts | 1.8 | 0.1 |
| dink | 0.9 | 0.0 |
| clip | 0.9 | 0.0 |
| whipped | 0.9 | 0.0 |

The 2022 final's commentator writes in a lyrical register — "sumptuous",
"glorious", "a nation re-infused with hope", "amid family, in front of the whole
wide world" — which is what a World Cup final gets and a Tuesday-night league
game does not. Every one of those words is in the phraser's example set.

### 8.2 Openers

The 30 most common first words of an utterance, pooled club football, whole
files (6,634 utterances, 1,101 distinct first words, top 30 covering 50%):

| rank | word | count | share | | rank | word | count | share |
|---:|---|---:|---:|---|---:|---|---:|---:|
| 1 | and | 416 | 6.27% | | 16 | oh | 67 | 1.01% |
| 2 | it's | 298 | 4.49% | | 17 | that's | 61 | 0.92% |
| 3 | he | 207 | 3.12% | | 18 | this | 61 | 0.92% |
| 4 | well | 184 | 2.77% | | 19 | a | 60 | 0.90% |
| 5 | he's | 174 | 2.62% | | 20 | back | 56 | 0.84% |
| 6 | the | 159 | 2.40% | | 21 | just | 55 | 0.83% |
| 7 | now | 157 | 2.37% | | 22 | that | 53 | 0.80% |
| 8 | **here's** | 152 | **2.29%** | | 23 | as | 53 | 0.80% |
| 9 | yeah | 148 | 2.23% | | 24 | what | 49 | 0.74% |
| 10 | i | 143 | 2.16% | | 25 | there's | 46 | 0.69% |
| 11 | it | 128 | 1.93% | | 26 | messi | 40 | 0.60% |
| 12 | so | 124 | 1.87% | | 27 | they're | 37 | 0.56% |
| 13 | but | 113 | 1.70% | | 28 | they've | 36 | 0.54% |
| 14 | you | 96 | 1.45% | | 29 | there | 34 | 0.51% |
| 15 | they | 73 | 1.10% | | 30 | good | 34 | 0.51% |

The same list for the 2022 final is materially different: `and` 8.00%, `he`
2.93%, `well` 2.54%, `yeah` 2.54%, `the` 2.47%, `he's` 2.47%, **`messi` 2.41%**,
`it's` 2.21%, `they` 1.89%. A player surname is the seventh most common opener
at the final and the twenty-sixth in club football.

The practical reading: **`Here's`, `Now` and `Back` are the club-football play
openers the system has never been shown.** They account for 152 + 157 + 56 = 365
line openings, 5.5% of the corpus, and they are missing from the phraser's
example set entirely.

### 8.3 Syntactic shapes, counted

| shape | club football | 2022 final | ratio |
|---|---:|---:|---:|
| `Here's` / `There's` + Name | **2.16%** | 0.33% | ×6.5 |
| `Now` + Name | **1.46%** | 0.33% | ×4.4 |
| `to` + Name (a pass named by destination) | **5.53%** | 0.98% | ×5.6 |
| `Name to Name` | **1.03%** | 0.07% | ×15 |
| `It's` + … | **4.40%** | 2.08% | ×2.1 |
| possessive `Name's` | **14.14%** | 8.59% | ×1.6 |
| `Name` + present verb | 9.66% | 8.20% | ×1.2 |
| `And` + clause | 5.85% | 7.29% | ×0.8 |
| bare name, ≤2 words | 5.61% | **7.81%** | ×0.7 |
| rhetorical question | 2.20% | **3.77%** | ×0.6 |
| **participle + `by` + Name** | 0.45% | **1.82%** | **×0.25** |
| `Oh` / `Ooh` opener | 0.92% | 0.59% | ×1.6 |
| `What a` … | 0.54% | 0.33% | ×1.6 |

**The participle-plus-by form that the phraser prompt calls out by name —
"Played by Molina, collected by Upamecano" — is four times rarer in club
football than at the 2022 final.** It is a real shape (30 instances) but it is
not the workhorse. The workhorses are `Here's X`, `Now X`, `X to Y`, and the
possessive.

```
participle + by
  pl-liv-mun-2025   7:10  Outwitted by Mayu, Bruno Fernandez, Amadalo.
  pl-liv-mun-2025   7:41  Cut out by Maguire.
  pl-liv-mun-2025  14:22  Started by Mallister.
  pl-liv-mun-2025  92:14  given away by Kate
  pl-liv-mun-2025  93:40  Scrambled out by Gravenber to Jotar.
  clasico-2017     56:22  Picked up by Leo Messi.
  lei-mun-2015      2:39  Picked up by Blint, who sends it down the line towards Junk
  bar-mal-2019      6:59  headed back into the penalty spot, and headed away by Salva Sevilla.

Here's / There's / Now / Back
  pl-liv-mun-2025   7:38  Here's Salah.
  pl-liv-mun-2025  17:08  Here's Mallister.
  pl-liv-mun-2025  42:16  Here's Bruno Fernandez.
  clasico-2017     10:44  Here's Andres Iniesta.
  clasico-2017      5:59  Here's Sergio Ramos now for Real Madrid.
  clasico-2017     28:51  Here's Toni Kroos.
  clasico-2017     84:21  Here's Modrić.
  clasico-2017     90:31  Here comes Iniesta.
  lei-mun-2015      2:58  Here's Wes Morgan again.
  lei-mun-2015     37:46  Here's McNair, one of the three center halves for Manchester United.
  lei-mun-2015     70:20  Here's Rooney on the right hand side.
  pl-tot-che-2024  87:38  Now Palmer does have it.
  pl-tot-che-2024  16:07  Now Romero is in trouble here.
  bar-mal-2019     13:51  Now, Griezmann.
  bar-mal-2019     79:09  Now, Vidal.
  lei-mun-2015      5:31  Back to Robert Hook.
  lei-mun-2015      6:52  Back to Wes's Morgan
```

### 8.4 How the goal call is built

YouTube renders shouted speech in capitals, which gives a free marker for the
call. The corpus has 6 all-caps utterances in `pl-liv-mun-2025`, 3 in
`pl-tot-che-2024`, 6 in `lei-mun-2015`, 8 in `clasico-2017`, 8 in
`bar-mal-2019`, 0 in `lei-avl-2015` and 94 in `argfra-dimaria`.

The structure, in order, and the corpus fills every slot:

1. **A fragment of anticipation before the ball crosses.** `CHANCE FOR A SHOT.`
   `BODY.` / `Griezmann, he's in a one-on-one.` / `Messi outside THE D.` /
   `Suárez back into Rakitić.` / `Now, the shot is blocked.`
2. **The strike, one to three words.** `IT'S IN.` / `GRIEZMANN CHIPS THE
   KEEPER.` / `MESSI STRIKES!` / `SUáREZ OH MY!` / `SUAREZ STRIKES ONE.` /
   `OH, YES!` / `DROPS OFF LEO MESSI.`
3. **Repetition of the name, often with an intensifier.** `OH MY! / OH MY!` /
   `LEO MESSI DOES IT AGAIN.` / `SURELY NOW IT'S CASEMIRO.` / `HE'S DONE IT!`
4. **The score, as a separate short utterance, within about five seconds.**
   `It's 1-0 to Barcelona.` / `SUáREZ HAS MADE IT 4-1.` / `LEO Messi makes it
   two.` / `Mallorca have got a second.` / `Two goals for Tottenham.` /
   `Villa take a two-0 lead.`
5. **A number about the scorer, within about fifteen seconds.** `It's his third
   goal of this La Liga campaign.` / `Griezmann gets his fifth goal of the
   season.` / `11 CONSECUTIVE GOALS IN PREMIER League games.` / `OH, IT'S A
   HAT-TRICK FOR LEO MESSI.` / `His first ever goal for the club.`
6. **The move, rebuilt in the past tense, naming two or three players.**
   `The shot rebounded off the post and fell very kindly for Carlos Henrique
   Casemiro who slotted the ball home` / `Ivan Rakitić rifles a left-footed drive
   into the top corner beyond Keylor Navas` / `Schweinsteiner made the run. He
   beat Aazaki and headed it past a stranded Casper Schmeichel.`
7. **The colour voice**, 4 to 26 seconds in.

The excitement markers are: all-caps (volume), one- and two-word utterances
stacked with zero gap between them, literal repetition (`OH MY! / OH MY! / OH
MY!`, `It was. / It was over.`), the interjection (`OH!`, `OOH`, `Oh, yes!`,
`Well, well, well.`), and the superlative used bare (`What a beauty.`, `What a
strike that is.`, `It's exhibition stuff.`).

---

## 9. Gap analysis against the system

Ranked by how far each would move the system's output toward the corpus. Each
gap gives the corpus evidence, the system evidence, the change, and the offline
measurement.

The system traces quoted are `runs/rephrased/mbappe-v3/file-20260913-185228-phrased.jsonl`
(27 lines over 202 s of the 2022 final) and
`runs/rephrased/e07b-freekick-v3/file-20260913-042354-phrased.jsonl` (6 lines
over 29 s).

---

### Gap 1 — Numbers almost never reach air, and they are the lead's job, not the analyst's

**Corpus.** 16.1–19.8% of utterances carry a number; 1.4–2.6 per minute. Pooled,
that is 1,203 number-carrying utterances in 589 minutes. The category split is
scoreline 22.9%, clock 9.6%, form 7.9%, a player's tally 4.2%, history 3.3%.
Number-carrying lines open like the colour voice at the base rate, so the lead
says them. After a goal the median 30-second window carries **two**
number-carrying lines.

**System.** Two lines in 27 carry a number, both in the mbappé trace:

```
  59.2  Mbappé steps up. Five in the tournament.
 110.9  Argentina to restart. Ten minutes remaining.
```

That is 7.4% of lines against a corpus 16–20%, and one of the two is the clock.
The phraser prompt's context section says "You may ignore it. Most lines
should." `clips/pack-argfra-2022.json` holds 13 notes about seven players; the
mbappé trace has Upamecano on the ball for three consecutive lines and the pack
says nothing about him.

**Change.**
1. Invert the note permission for quiet events. In `PHRASER_RULES`, the context
   section currently reads "your line MAY carry one of those clauses" and "Most
   lines should [ignore it]". For `build_up`, `pass`, `carry` and `dead_ball`
   forms with a `context:` block present, make it "if the form carries a
   `context:` clause about a player your line names, use it" — with the existing
   one-clause, exact-numbers, subject-must-be-named limits unchanged.
2. Add the corpus's own number shapes to the examples under a new `numbers`
   kind, drawn from section 5.2 above, so the model has "his third goal of this
   La Liga campaign" and "five in four appearances so far this season" and
   "12 out of 12 for the club in total" in front of it rather than inventing a
   form.
3. Widen the pack. A note per starting XI player, not thirteen for the match.
   `commentary notes` already appends to a finished pack.

**Measure.** Re-run `commentary rephrase` over the two existing traces with a
widened `pack-argfra-2022.json` and count number-carrying lines with the regex
in section 5.1. Target: from 7.4% to 16%. The gate's `note_claim` already checks
any non-scoreline number against the notes about the people the line names, so a
rise in the number rate with zero `note_claim` refusals is the pass condition.

---

### Gap 2 — The colour seat fires on a timer; it should fire on the ball being dead

**Corpus.** Colour entries per 100 utterances: 2.1 in an attacking move, 10.5 at
a dead ball, 11.8 at a stoppage, 15.4 over a replay. Median delay after a big
event 21.4 s; only 7% land within 6 s. Median run 4 utterances, 51% run four or
more. It addresses the lead by name.

**System.** `AnalystConfig` is `lull_s = 7.0`, `min_gap_s = 40.0`,
`max_words = 30`. It is a silence timer with no notion of phase. On the mbappé
trace it spoke three times in 202 s, at 41.7 s, 95.2 s and 144.9 s, each a
single 25–35 word utterance:

```
 41.7  Otamendi's taking no chances with the fresh legs off the French bench — that's
       twice now he's simply stopped the run rather than let it get behind him.
 95.2  Ten minutes left for the holders, and Deschamps has already spent his changes —
       whatever gets France level has to come from the players out there now.
144.9  Argentina have dropped everything behind the halfway line and invited France to
       carry it — that hands Mbappe and Kolo Muani room to run into, which is the one thing
```

The 144.9 s line is truncated at the 30-word cap mid-clause. The 95.2 s line
lands 8 seconds after the goal call at 86.8 s — inside the window the corpus
says belongs to the lead.

**Change.**
1. Replace `lull_s` with a phase gate. The predictor already classifies the
   picture (`live_play`, `stoppage`, `replay`, `close_up`, `crowd` appear in the
   caller trace). Allow the analyst only when the last two caller forms were
   `stoppage`, `replay`, `close_up`, `crowd`, or a dead-ball event
   (`corner`, `free_kick`, `throw_in`, `goal_kick`, `kickoff`), **or** when 20 s
   or more have passed since the last big event. Forbid it outright inside 12 s
   of a `goal`, `shot`, `save` or `penalty` form.
2. Make it a turn, not a line. Raise `max_words` to about 45 and let it return
   two to four short utterances rather than one long one, matched to the median
   run of 4. A 30-word cap that truncates mid-clause is worse than a shorter
   cap: the corpus's colour entries are 3 to 12 words each and simply come in
   sequence.
3. Give it its own example set from section 4.4, and the `Well,` / `Yeah,` /
   `I think` opener, which is how a listener knows the voice has changed before
   the timbre tells them.
4. Add the lead's name as an addressable token so a handover question is
   possible. The corpus does this on every file with two identified voices.

**Measure.** Replay the mbappé and freekick traces through the analyst offline
with the phase gate on, and count: colour turns inside 12 s of a big event
(target 0), median utterances per turn (target 3–4), and share of turns opening
with a colour cue (target > 60%).

---

### Gap 3 — Cadence is one rate; it should be two, and the slow one is slower than the code thinks

**Corpus.** Median inter-utterance gap 4.3 s pooled, 2.8 s in an attacking move,
4.2 s in build-up, 4.5 s at a dead ball. 53% of all gaps exceed 4 s; 33% exceed
6 s; 14% exceed 10 s. Words per utterance move the other way: 7 in an attacking
move, 8 in build-up, 10 at a dead ball.

**System.** `CallerConfig.min_gap_s = 4.0`, `min_gap_floor_s = 1.5`, and the
floor's docstring cites the wrong unit ("a median 2.4s apart" — that is caption
segments, section 2.1). The mbappé trace's realised gaps between phrased lines,
excluding the three long stoppage waits, run 3.8–8.1 s with a median of 4.3 s
and **no gap under 3.8 s anywhere in the trace**. The system is flat: it has the
build-up rate and never reaches the attacking rate, and its build-up gaps are
right for the wrong reason.

**System evidence, lines 115.3 to 176.7 of the mbappé trace:**

```
115.3  +4.3  Through the middle now.
119.6  +4.3  Down the right, France set.
124.6  +5.0  Players appealing at the halfway line.
128.9  +4.3  Scaloni roaring at his players.
133.2  +4.3  Upamecano out from the back.
137.5  +4.3  Upamecano, striding out.
153.1 +15.6  Upamecano squeezes it infield.
157.1  +4.1  France probe across the halfway line.
161.5  +4.4  Messi, head up.
165.7  +4.1  Messi dispossessed on the touchline.
171.5  +5.9  France arriving in numbers.
176.7  +5.1  Mbappé! Off the ground! Two-two.
```

Twelve lines, eleven gaps, ten of them between 4.1 and 5.9 s. Real build-up has
this median but a much fatter tail: 11% of build-up gaps exceed 10 s and the
system has one such gap, forced by a stoppage.

**Change.**
1. Make `min_gap_s` a function of the caller's form. Attacking forms (`shot`,
   `save`, `goal`, `penalty`, `corner` delivery, a `cross` if the caller learns
   to emit one) get 2.5 s; `build_up`, `pass`, `carry` get 4.5 s; dead-ball and
   stoppage forms get 5.0 s.
2. Give the phraser the right to return silence on a build-up form. The handoff
   already names this: "The phraser cannot choose silence." The corpus number to
   target is 24% of build-up touches passing with nothing said. A cheap version:
   if the form is `build_up`/`pass`/`carry` and the previous line was also one of
   those and named the same player, return empty.
3. Correct the docstring on `min_gap_floor_s` so the next person does not
   recalibrate against caption segments.

**Measure.** `commentary rephrase` does not change trigger timing, so this one
needs a re-run of `commentary run --source file` over `clips/mbappe.mp4` with
the caller cached, or a simulation over the existing caller rows. Measure: the
gap distribution split by form, against the table in section 2.3. Pass when the
attacking median is under 3.2 s, the build-up median is 4.0–4.6 s, and at least
10% of build-up gaps exceed 10 s.

---

### Gap 4 — The line is too short, and the cap makes it structural

**Corpus.** Club football median 8 words, mean 10.7; 47.3% of utterances are
nine words or more and 20.2% are sixteen or more. Even at the 2022 final the
median was 6 and 30.7% ran to nine words or more.

**System.** `PhraserConfig.max_words = 16`. The mbappé v3 trace: median 5,
mean 4.5, **zero lines of nine words or more**, zero one-word lines. The
freekick trace: median 4, mean 4.7, one one-word line, zero nines.

So the system's distribution is not a compressed version of the corpus — it is
a different distribution with almost no overlap in the upper half. The prompt
says "Most of what you send back should be under eight words"; 100% of it is.

**Change.**
1. Raise `max_words` to 28, matching the caller's, and change the prompt's
   length paragraph from the 2022-final figures to the club figures: median 8,
   nearly half of all lines nine words or more, one in five sixteen or more, the
   95th percentile 22–27, and the longest in a match 50 or more.
2. Tie length to phase in the prompt the way the corpus does: shortest in the
   box, longest at a restart. "A line at a dead ball or after a goal is the long
   one — the tally, the rebuild of the move, the note. A line while the ball is
   moving into the box is three words."
3. Rebuild `commentary_examples.py` from all seven files rather than one, with
   the sampling weighted to club football, so the length distribution of the
   examples matches the target. The script already takes `--captions` and
   `--pack`; it needs to take several.

**Measure.** `commentary rephrase` over both traces, then the word-length
histogram against section 1's pooled row. Pass when the median is 7–9, at least
30% of lines are nine words or more, and at least 8% are sixteen or more.

---

### Gap 5 — The register is the 2022 final's, and the 2022 final is the odd one out

**Corpus.** Section 8.1 and 8.3. Club football's play openers are `Here's`
(2.29%), `Now` (2.37%), `Back`, and `X to Y` (1.03%); the participle-plus-`by`
form is 0.45%. The 2022 final inverts this: participle-plus-`by` 1.82%,
`Here's` 0.33%, `X to Y` 0.07%.

**System.** `commentary_examples.py` is built entirely from
`clips/argfra-dimaria.en.json3` by `scripts/build_commentary_examples.py`, and
`PHRASER_RULES` names the participle form explicitly: "The participle form
carries most of the build-up that is not a bare name: 'Played by Molina,
collected by Upamecano.'" The examples header says "REAL COMMENTARY, FROM THE
CAPTIONS OF A WORLD CUP FINAL". So the system has been taught, on purpose, the
one feed in this corpus that is least like a league match — and the target
fixture is a league match.

**Change.**
1. Extend `scripts/build_commentary_examples.py` to take several
   `--captions`/`--pack` pairs and pool them. The roster-cleanliness filter
   already needs a pack per file; the six new matches have StatsBomb lineups for
   four and ESPN rosters for two (`clips/espn-lineups-*.json`, fetched and
   written as part of this study).
2. Weight the pool toward club football — the 2022 final at most one sixth of
   the examples, matching its share of the corpus.
3. Drop `lei-mun-2015` from the *register* pool. It is radio: 195 words a minute
   and a median of 11 words, and it will drag the length distribution. Keep it
   for the event-coverage and thread work, where it is the best file here.
4. Replace the participle sentence in `PHRASER_RULES` with the club shapes:
   `Here's <Name>.` / `Now <Name>.` / `Back to <Name>.` / `<X> to <Y>.` /
   `<Name>'s <noun>.` Keep the participle as one option among several rather
   than "carries most of the build-up".

**Measure.** After rebuilding the examples, `commentary rephrase` both traces
and count the shape rates from section 8.3 against the club column. Pass when
`Here's`/`Now` openers appear at all (they currently never do) and
participle-plus-`by` is under 1%.

---

### Gap 6 — No thread runs across the match

**Corpus.** Section 7. Every match has three to five threads, each with a setup,
a payoff within four seconds of the event, and one to four callbacks spread over
the rest of the match. The Vardy record is mentioned twelve times over 94
minutes; the Cucurella slips fifteen times over 99; the snow nine times over 98.
Each callback restates the same number in a new form.

**System.** `CallerConfig.recent_lines = 5` and `PhraserConfig.recent_lines = 4`.
Each call sees the last four or five lines and nothing else. Nothing in the
trace is ever picked back up. In the mbappé trace "Five in the tournament" is
said once at 59.2 s and never again, including at 176.7 s when Mbappé scores the
goal that makes it six.

**Change.**
1. Add a `threads` list to the run state: `{note_id, subject, last_said_ts,
   times_said, status}`. Seed it from the pack's notes at kickoff with
   `times_said = 0`.
2. When the phraser uses a note (the gate's `note_claim` already detects this),
   increment `times_said` and stamp `last_said_ts`.
3. Feed back into the phraser's `context:` block, in priority order: a note
   about the player the line names that has been said **once before** and not in
   the last 300 s, ahead of an unused note. That is the callback, and it is what
   makes a thread rather than a list of facts.
4. On a goal, force the scorer's unused notes to the top of the context block.
   Every goal in the corpus is followed within fifteen seconds by a number about
   the scorer.

**Measure.** `commentary rephrase` over a longer trace than exists today — this
needs one of the 90-minute caption files replayed, or the mbappé trace extended.
Count: distinct notes used, and how many are used more than once. The corpus
target is that roughly half of used notes recur. Today the answer is one note
used once.

---

### Gap 7 — The goal call is two beats where the corpus has seven, and the third beat is lost to the gate

**Corpus.** Section 8.4. Median 60 words in the 30 s after the goal, 7
utterances, longest internal gap 7.1 s, two number-carrying lines. The score
arrives as its own short utterance within about five seconds; the scorer's tally
within fifteen; the move is rebuilt in past tense naming two or three players.

**System.** The mbappé trace's two goals:

```
 82.5  Mbappé! Buried past Martínez! Two-one.          gate passed
 86.8  Mbappé! Three-two.                              gate refused: scoreline_mismatch
176.7  Mbappé! Off the ground! Two-two.                gate passed
180.5  Mbappé! The volley! Three-two.                  gate refused: scoreline_mismatch
```

Two lines each, the second refused both times, and nothing after. No tally, no
rebuild, no score restatement, no colour.

**Change.** The append-in-code shape is already agreed in `docs/HANDOFF.md`
section 3d and not built: the model writes name and how, code computes and
appends the scoreline, and a check strips any number the model wrote anyway.
Build that, and then add the beats the corpus has and the system does not:

1. **A `goal_followup` state**, armed by the gate when a goal line passes,
   lasting 30 s. While it is armed the phraser is handed a different instruction
   block: beat 2 is the score (appended by code), beat 3 is a note about the
   scorer if one exists, beat 4 is a past-tense rebuild of the move naming the
   players the caller saw. These are four separate lines at 2–5 s spacing, not
   one long one.
2. **Ban the scoreline from every line but the first**, which the gate already
   effectively does by refusing them; make it a prompt rule and a code strip so
   the line is not dropped whole. A dropped line is a lost beat, which is
   exactly what happened at 86.8 s and 180.5 s.

**Measure.** `commentary rephrase` over the mbappé trace. Count the words spoken
in the 30 s after each goal (corpus median 60, system currently 7 and 9), the
number of utterances (corpus 7, system 2), and `scoreline_mismatch` refusals
(target 0).

---

### Gap 8 — Whole event kinds have no examples and no form

**Corpus.** Section 3. Crosses are the most reliably called event in football
(90% any, 11% silent, 50% named). Switches of play get the shortest lines in the
corpus. Throw-ins, goal kicks and free-kick deliveries are mostly *not* called,
and the goal kick is where the storyline goes. Offside gets the longest window
after a substitution.

**System.** `Event` in `src/commentary/schemas.py` has 20 members and includes
`corner`, `free_kick`, `throw_in`, `offside`, `card`, `substitution`,
`kickoff`, `stoppage`, `interception`, `clearance`, `tackle`. It does **not**
have `cross` or `switch`, which are the two build-up events the corpus treats as
worth naming. `commentary_examples.py` has eight kinds — `build_up`, `pass`,
`shot`, `save`, `goal`, `foul`, `dead_ball`, `aside` — so a corner, a free kick,
a throw-in and a goal kick all draw from one `dead_ball` bucket of examples, and
there are no offside, card, substitution or kickoff examples at all.

**Change.**
1. Add `CROSS` and `SWITCH` to `Event`. Both are legible from six frames and
   both are things the corpus always names.
2. Split `dead_ball` into `corner`, `free_kick`, `throw_in`, `goal_kick` in
   `commentary_examples.py`, and add `card`, `offside`, `substitution`,
   `kickoff`, `injury` and `restatement` kinds. The examples for each exist
   verbatim in section 3.2 of this document and can be lifted straight in, or
   regenerated from the caption files with a keyword classifier per kind.
3. Teach the phraser that some kinds are usually silent. The corpus rates are
   in section 3: goal kick 43% silent, throw-in delivery 37%, free-kick delivery
   33%, kickoff 31%. Currently the phraser returns a line for every call.
4. Add the score-and-clock restatement as a **code-generated** line on a timer,
   not a model line. Section 5.3 shows it is formulaic — `N minutes gone.` /
   `N-N to <team>.` — and the handoff's rule is already "code writes numbers, the
   model writes words". `bar-mal-2019` does it every 150 s; a five-minute timer
   is a reasonable default for a feed with no score bug and can be disabled for
   one that has.

**Measure.** For the event kinds, `commentary rephrase` cannot help because the
caller's forms are fixed in the existing traces. Measure instead on a fresh run
over `clips/e11-corner.mp4`, `clips/e09-offside.mp4`, `clips/e06-yellow.mp4` and
`clips/e12-foul.mp4`: does the phrased line use the event's own vocabulary at
roughly the corpus's `label%`? For the restatement timer, count restatements per
hour in a replayed trace against section 5.3.

---

### Lower-ranked, listed for completeness

**Names.** The prompt's "a good proportion of it should be one name, or two
names, and nothing else" should become one line in twenty-five, and the full
name should be offered for a first mention in a passage (17–48% of named
utterances in the corpus). The nationality substitute (`the Frenchman`, `the
Croatian forward`) is the commonest way a commentator avoids repeating a name,
and it is derivable from the pack. And the four hedge forms in section 6e —
`I think it was X`, `I'm not really sure who`, `one of them is X`, and naming the
act not the actor — are directly useful to a system whose open-play naming is
bounded by pixels: it can say the thing and withhold the name idiomatically
rather than withholding the line.

**Replay talk.** 26 instances across four matches, all formulaic (`As we see …`,
`Having seen the replay …`, `Watch this.`, `Let's look at that incident once
more.`). The caller already emits a `replay` scene kind and currently sets
`speak=False` on all of them (8 of 8 replay rows in the mbappé trace). The
corpus says a replay is worth one line, and that line is the past-tense rebuild
of the move — which is also beat 4 of the goal call in Gap 7.

**Excitement.** The system's curve runs 0.2 to 1.0 and the values in the mbappé
trace are sensible. The corpus's marker that is missing is **repetition**: `OH
MY! / OH MY! / OH MY!`, `It was. / It was over.`, `SURELY NOW IT'S CASEMIRO.`
Stacked identical short utterances with zero gap are how volume is written, and
the prompt currently forbids repeating the previous line under every
circumstance. The ban should exempt excitement above about 0.9.

---

## 10. Appendix: alignment and caption caveats

### 10.1 Kickoff offsets

Four matches are aligned by cross-correlating roster surnames from the StatsBomb
lineup files against StatsBomb event times, searching the offset that maximises
the count of events whose player is named within ±6 s. The hit rate at the
optimum is the confidence.

| file | StatsBomb id | 1st-half kickoff (video s) | 2nd-half kickoff (video s) | hit rate p1 / p2 | corroboration |
|---|---|---:|---:|---|---|
| lei-mun-2015 | 3754186 | **70.0** | **2936.0** | 35% / 42% | "We're underway at the King Power Stadium" at 73.9 s; "Half time at the King Power Stadium" at 2892.0 s |
| lei-avl-2015 | 3754106 | **15.0** | **2966.0** | 15% / 10% | "We are ready to go for the second half" at 2960.6 s; "three added minutes in the first half" at 2723.8 s |
| clasico-2017 | 267569 | **282.5** | **3256.0** | 42% / 39% | Casemiro 27:46 → call at 1947.4 s; Messi 91:47 → `LEO MESSI!` at 6060.2 s |
| bar-mal-2019 | 303451 | **243.0** | **3092.0** | 53% / 55% | Griezmann 6:23 → `GRIEZMANN CHIPS THE KEEPER` at 628.2 s; Suárez 42:22 → `SUáREZ OH MY!` at 2786.2 s |

The two NBC files have no StatsBomb data. ESPN summaries were fetched during
this study and written to `clips/espn-lineups-pl-liv-mun-2025.json` (event
704475) and `clips/espn-lineups-pl-tot-che-2024.json` (event 704420); they carry
rosters and minute-precision key events, which gives ±30 s alignment at best.

| file | 1st-half kickoff | 2nd-half kickoff | how |
|---|---:|---:|---|
| pl-liv-mun-2025 | **≈152 s** (±15) | **2977 s** (±5) | "the first through ball of the game" at 162.1 s; "And we're back underway with Liverpool" at 2979.6 s. Checked against ESPN: Salah's 70' penalty → `SALAH ENSURES THEY LEAD` at 4426.2 s = 69:09 match; Amad Diallo's 80' → `MANCHESTER UNITED HAVE DONE IT AGAIN` at 5032.3 s = 79:13 match. |
| pl-tot-che-2024 | **≈165 s** (±20) | **≈3035 s** (±30) | Three ESPN goal anchors give 156, 164 and 191 s; the spread is minute-rounding. Palmer's 61' penalty at 3963.5 s gives the second-half figure. |

**Both NBC uploads cut the half-time interval out of the video.** In
`pl-liv-mun-2025` the largest gap anywhere in the file is 37.1 s and the second
half restarts 7 seconds after the last first-half line; in `pl-tot-che-2024` the
gap between "It's been a fascinating first half" (2859.9 s) and "Welcome back to
North London" (3008.1 s) is 148 s, far shorter than a real interval. Any analysis
that treats video time as continuous match time across the break will be wrong by
about fifteen minutes.

### 10.2 The Leicester–Manchester United file

`lei-mun-2015` has 18,499 words in its live window against 7,671–11,864 for the
others. The cause is **not** studio content: the file starts 70 seconds before
kickoff ("So Clauddio Raneri ready for a challenge this afternoon" at 0:00, "So
we're ready for kickoff" at 0:16, "We're underway" at 1:13) and ends shortly
after the final whistle. The live window 70–5760 s is genuinely live play
throughout.

The cause is that **it is radio commentary**, from the Leicester City club
channel, and the commentator is describing the match for listeners who cannot
see it:

```
   0:04  The two teams are out and are shaking hands just in front of us in the West End
         at the moment.
   1:39  The Foxes attacking from right to left as we look out from the west stand of the
         King Power Stadium as Huth gets under a high ball and heads it forward.
   1:18  Referee Craig Porson has blown his whistle and Manchester United have the ball
         inside their own half with Daily Blind on the left hand side.
   2:50  He could just dribble the ball on the edge of his own area and play a 10yd pass
         just further forward to his central defensive partner, Robert Huth.
```

At 195 words a minute it is 70% denser than the next-densest file. It is used
throughout this document for event coverage, threads and names, where it is the
richest source here, and it is flagged wherever it would distort a register
number. **It should be excluded from the phraser's example pool.**

### 10.3 Other caption caveats

- **Bracketed non-speech is very unevenly transcribed.** `pl-tot-che-2024` has
  266 bracketed tokens, `argfra-dimaria` 196, `lei-avl-2015` 166,
  `pl-liv-mun-2025` 47, `lei-mun-2015` 13, and `clasico-2017` and
  `bar-mal-2019` have none at all. Their absence in a file does not mean the
  crowd was quiet; it means the ASR did not label it. They are dropped from all
  counts.
- **A gap in the captions is ambiguous**, as `commentary/grading/captions.py`
  already documents: it means either that the commentator went quiet or that
  the ASR gave up in crowd noise. Every silence figure in this document is
  therefore a **ceiling** on the real one, and the ones after goals — where the
  crowd is loudest — are the least trustworthy.
- **Surnames are mangled constantly.** Grealish appears as "Greenish", "Greish"
  and "Gish"; Vardy as "Bardy"; Kulusevski as "Kulleski", "Kukera" and
  "Kukare"; Cucurella as "Cookera", "Cookareo" and "Cooker"; Schweinsteiger as
  "Feinsteiger", "Strike Stiger" and "Schweinsteiner". Every `named%` figure in
  section 3 is a floor. No factual claim in this document rests on a name read
  out of a caption: the alignment uses StatsBomb and ESPN for the facts and the
  captions only for what was said and when.
- **All-caps rendering marks shouted speech** and is a reliable free goal-call
  detector in six of the seven files. `lei-avl-2015` has zero all-caps
  utterances despite five goals, so its goal calls were found through the
  StatsBomb alignment instead.
- **`clasico-2017` and `bar-mal-2019` have no `>>` speaker markers at all**, so
  the two-voice analysis on those two files rests entirely on the colour-opener
  heuristic validated in section 4.1.
