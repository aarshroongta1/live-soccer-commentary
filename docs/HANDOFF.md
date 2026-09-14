# Handoff

Start here, then read `docs/CLIPS.md`. This file is the state; that one is the
evidence.

**HEAD:** `main` in `/Users/Aarsh/Desktop/commentary`. One repo, one branch, no
worktrees, no PRs. `.env` at the root. `clips/` and `runs/` are in the repo and
gitignored — the clips are 26 MB each and the traces are somebody's API spend.

**Gates, green at every commit:** `uv run pytest -q` · `uv run ruff check .` ·
`uv run mypy`. Run `uv sync --all-extras --dev` first: without the `tools`
extra, mypy reports nine import errors in `mcp_server.py` that are the missing
package and not the code. There is no `vision` or `audio` extra any more;
sections 7 and 8 say why. HEAD as of 13 Sep 2026 is `31b6a88`, pushed.

## 1. What runs

```
uv run python -m commentary run --source file --path clips/<clip>.mp4 \
    --backend anthropic --pack clips/pack-<match>.json \
    --seconds 65 --delay 8 --out runs/<name>
uv run python -m commentary grade runs/<name>/*.jsonl \
    --pack clips/pack-<match>.json \
    --statsbomb clips/statsbomb-events-<match>.json \
    --lineups clips/statsbomb-lineups-<match>.json
```

Opus 5 calls, Haiku 4.5 reads the board, Opus 5 on the analyst (`.env`).
`CALLER_FRAME_WIDTH` (default 768) is the width the caller's frames go out
at; 1280 costs 1.75x and two clips could not tell it from noise (section 9).
**About $0.30 a minute of video**: a three-minute clip costs $0.95 to $1.07, a
forty-five-second clip $0.15 to $0.25. On Haiku for the caller and analyst a
forty-five-second clip is $0.04 to $0.06. Nothing runs locally but ffmpeg,
OpenCV and numpy; nothing downloads at run time.

`grade` aligns StatsBomb onto video time from the trace's own board readings
and **refuses to grade an alignment it cannot trust**. A shootout has no clock
on any broadcast, so those need `--offset <seconds>` measured off a frame by
hand, and the output says so above every number.

## 2. Where it got to

Twenty-one runs on eighteen clips from six matches and four broadcasters:
**182 spoken lines, 124 player namings, and not one wrong name.** No trace in
the set contains a word that is off the roster. Zero phantom events in any
clip. **$8.90 all in** ($8.19 of it after the key was topped up mid-way).

| pass | clips | events called | detail |
|---|---|---|---|
| the tuned clip | 1 (re-run many times) | goal called and named | best run 10 of 12, `runs/A` |
| same broadcast | 5 × 3 min | 5 of 6 headline events | `CLIPS.md`, "The first pass" |
| unseen matches | 12 × 45 s | 8 of 12, 9 counting e01's first run | `CLIPS.md`, "The short clips" |
| after the fixes | 3 reruns | e07 and e02 fixed, e01 inconclusive | `CLIPS.md`, "The twelve, and the three rerun" |

I cannot reproduce a 15-of-20 count; by headline event per clip it is 13 or 14
of 18 depending on whether e01 is scored on its first run or its rerun. The
per-clip tables in `CLIPS.md` are the source.

What generalised from the tuned clip: the event vocabulary, the gate, and the
board reader — which read a **two-row CONMEBOL bug with the clock underneath
the score** 13 times out of 13, and a 2018 bug with the clock on the right
perfectly, having never seen either. What did not: naming a player in open
play.

## 3. What changed in the pass before this one

| commit | |
|---|---|
| `4703547` | `grade` answers "was the event called", not just "did a line land near it". Recall read 67-100%; called read 17-75%. |
| `0dc6035` | `fold` glued a possessive s onto the name — "De Gea's" became "de geas" — and trimmed the keeper out of a penalty, twice. |
| `228a768` | "the Dutchman" read as an invented surname; the demonym's -man form goes in beside it. |
| `40e8040` | The position-zero name trim is gone. Nineteen firings in nine runs, zero catches. The trade is real and a test asserts it: an invented surname put first now reaches the microphone. |
| `1a54a3a` | One goal-claim definition in the gate, and the runtime asks it: a goal from a set piece is a goal to the director whatever the caller tagged. |
| `dd85151` | A line that claimed a goal lifts the four-second rate cap for the next call. |
| `d18358c` | `stoppage` in the event vocabulary, so a player lying injured is not a phantom foul. |
| `012c232` | Carry rule: a name stays on the ball for eight seconds, same phase, same side. Plus `name_withheld` logged when a line says "the taker" and its own sightings name him. |
| `9bda1db` | "and it is in" written out, which unifying the goal definitions had lost. |
| `clips/build_pack.py` | `NAME_FIXES`: StatsBomb's nickname field calls Randal Kolo Muani "Randal Kolo", and two true lines died of it. |

## 3b. What changed on 13 September

Six commits, all on the strength of traces already on disk or a run under
$1.20. Sections 7, 8 and 9 carry the numbers.

| commit | |
|---|---|
| `99dc36c` | The player tracker, the gallery, the kit split and the marks are gone, with torch. A real-clip ablation: without them the caller named 20 distinct players against 8 with them, and bound 96% of sightings against 30%. Section 7. |
| `2b96b44` | The whistle and roar detectors and the audio pipeline are gone. Whistle fired once in 58 runs; roar fired every five seconds regardless and missed the tuned clip's goal nine runs out of nine; it was also holding the goal gate open half the time when the board was unreadable. Section 8. |
| `f43d5ed` | `CALLER_FRAME_WIDTH`. Two clips at 1280 went one up, one down, at 1.75x the cost. Default stays 768. |
| `e1166e0` | The caller's examples are real broadcast lines; "one sentence" is gone from the rules; the three "use the name" paragraphs are one. And a gate bug: the minimum-words rule applied to every line, not only to trimmed ones, so any line under three words died. |
| `193ab26` | The rate cap is earned by the last line's length (floor 1.5 s, cap 4 s). Two trims the new multi-sentence lines exposed: a capitalised word opening a second sentence was read as a name, and a plural demonym ("Argentines") was off-roster to the grader. Both fixed in gate and grader. |
| `31b6a88` | The similarity veto on repeats is gone from both voices. Zero firings in 63 runs and 1,062 calls; the prompt's "last five lines" is what stops a repeat. The analyst's echo check against the caller stays. |

- **14 September — the gate does the arithmetic now.** In
  `runs/voice/dimaria-goal/file-20260914-015533.jsonl` the caller called the
  second goal twice more during the celebration, "Argentina's third" and then
  "Argentina's fourth", with the state holding 2-0. Both went out with no gate
  reason at all: a goal *was* in the state, which is cover for any line about a
  goal, and it turned out to be cover for counting as well. A side that has
  scored *n* may now be said to have scored *n*, and *n+1* only while a goal is
  being called that the state has not taken in yet — the new `score_claim`
  rejection, whole line, no trim. The scoreline rule shares the arithmetic and
  gains the same latitude, which it never had; it also hears "3 nil" and "2 to
  1" now. An ordinal that counts anything but goals ("their first real chance",
  "chasing a third") is left alone.

## 4. Known gaps

- **Open-play naming, and what it actually is.** Across 208 live-play caller
  lines on Opus, 57% had no sighting at all — nothing legible in the frame —
  10% had sightings that bound to nobody, and 33% had a bound name. Of those
  69, 45 said the name and 24 withheld it, mostly correctly (the bound name
  was van Dijk on the edge while Messi took the penalty). So the gap is
  pixels, not prompt: a shirt number in a wide shot at 768 wide is four
  pixels tall. 1280 did not settle it in two runs (section 9). The carry
  rule and the registry are the room that is left.
- **A call refused just before the kick.** Still open. The board reader runs
  eight seconds ahead of the cursor but only fires a trigger on the third
  agreeing read; firing on the *first* differing read would land as the
  cursor reaches the kick, and the caller's lookahead would show the ball in.
  Deterministic, no model call, one unit test. Not done yet.
- **Fragments do not come.** With real fragment examples in the prompt and a
  cadence that rewards short lines, the Opus caller wrote no line under seven
  words on the Mbappé clip. Three gates stood in front of the style; two are
  open now (section 9); the third is the caller's own choice.
- **No run longer than 3.5 minutes.** Nothing has shown the state, the cost
  cap, the goal-talk cap or the analyst's spacing over a half.
- **Live capture and voice have never been switched on.** `ScreenCapture`
  and `ElevenLabsSpeaker` are built and unit-tested; every result in this
  file came from `--source file` with `--voice log`.
- **Shootouts.** The FIFA feed shows a tally strip at the bottom centre,
  intermittently, outside the board reader's crop. A second crop would read
  it. Decided not to build it; a league match has no shootout and a live
  wire is the likelier answer if it ever matters.
- **StatsBomb's card timestamp is not when the card is shown.** Unchanged;
  it cost two card clips.
- **Grading a 2026 fixture.** StatsBomb open data is historical. The ESPN
  feed adapter (`commentary feed`) exists and is unverified against a
  current match. The Opus judge can score a trace without a feed.

## 5. Next steps, in this order

The target moved: not October, a Premier League or La Liga match in the
next two or three days, then finish. Roughly three days and $40.

**Today, from clips on disk, no match needed.**

1. **Live capture smoke test.** Play a clip full-screen in a video player and
   run `--source screen`. Get moment-to-spoken lag and whether the watch
   page's delayed video holds sync. This path has never been run.
2. **Voice on a real clip.** `--voice elevenlabs` on the Di María clip: two
   voices, the goal preempting the analyst mid-sentence, synthesis lag on top
   of the buffer. Needs ElevenLabs credit; pennies.
3. **Fire the board trigger on the first differing read** (gap 2 above).
   Then rerun `clips/mbappe.mp4` once, $1.10, and look for the penalty called
   as it happens.
4. **Sonnet versus Opus**, six short clips on Sonnet, about $1, against the
   Opus runs on disk. Haiku already showed the floor is real (4 of 22 events);
   if Sonnet holds, a match is $11 instead of $27.

**Once the match is picked.**

5. **Which app or site the broadcast plays in** decides the capture device
   and whether the picture is capturable at all. Ask before anything else.
6. **Build the pack with the researcher** for the real fixture and check the
   numbers and kits by hand. A wrong number here is a wrong name on air.
7. **Set the score-bug crop** for that broadcaster with `commentary crop` on
   any recent frame of its feed. The reader read three unfamiliar layouts
   perfectly; the crop still has to point at the corner.
8. **One 20-minute dry run** of a recent match off the same broadcaster if a
   recording is obtainable, about $6. First run past four minutes.

**Match day.**

9. **The live run**: `MAX_USD_PER_MATCH` set, trace on, screen-record the
   watch page with audio for the demo. Fallback if capture fails: record the
   broadcast and run it as a file twenty minutes behind; the demo survives.

**After.**

10. **Demo video, README numbers, this file.** README has the clip results;
    it needs the live lag, the cost, the demo, and the honest line about
    what was not graded. Judge the trace with `commentary grade` sans feed
    and the Opus judge.

Skip: grading the live match against a feed; the shootout crop; anything
tracker-shaped; more work on fragments; the simulator ablation table.

## 6. Do not do these again

- **No batch scripts that cannot be interrupted.** Five runs fired from one
  script before a change of plan landed, and $4.90 went on clips that had just
  been deprioritised.
- **`--seconds` must be the clip length plus at least twenty.** The cursor is
  pinned at the live edge minus the delay, so the last eight seconds of a clip
  are never narrated whatever you pass, and the caller needs another twelve on
  top to write about what it saw. `--seconds 50` on a 45 s clip lost a goal
  that the trace shows the system had already called internally.
- **Do not centre a card clip on the StatsBomb timestamp.** Check the frame
  first; the broadcaster shows the card later, often after the ball goes out.
- **Do not take a simulator A/B as evidence about vision or audio.** The sim
  oracle decodes a timestamp and never reads a pixel or hears a sound. The
  marks A/B in the old README was 12 lines on the sim and it was the whole
  case for a stack that lost on real clips.
- **Ask the traces before building.** The tracker, the whistle, the roar and
  the repetition veto were each answered from `runs/` for nothing before any
  code moved. `runs/readtrace.py` prints a trace; the trigger, sighting,
  gate and spoken rows carry everything needed.

## 7. The tracker is gone

The local vision stack — RF-DETR, ByteTrack, the HSV kit split and the SigLIP
gallery, the whole `vision` extra with torch under it — was removed after a
real-clip ablation, not on taste. Six 45 s clips (e01, e02, e07, e08, e09,
e11), two runs each, tags on versus tags off, Haiku 4.5 as caller and analyst
in both arms, $1.15 all in. Traces, driver and table are in `runs/abl/`.

| | tags on | tags off |
|---|---:|---:|
| runs | 12 | 12 |
| spoken lines | 40 | 45 |
| lines naming a roster player | 26 | 43 |
| distinct players named | 8 | 20 |
| sightings bound / made | 70 / 235 | 115 / 120 |
| wrong or off-roster names | 0 | 0 |
| grader factual errors | 1 | 0 |

With tags the caller named the close-ups (Ronaldo, Messi, Henderson) and
little else, and filed letter tags as sightings that bound to nobody, which
is the "tag looks like a claim" failure this file already described. The
median track lived 2 to 2.8 s, no longer than the caller's own frame window,
and the kit split had already been measured discarding 94% of close-ups. The
only evidence ever in its favour was a 12-line simulator A/B that the README
itself said could not measure marks, and one Mbappé anecdote.

What replaced it is nothing new: the caller reads shirts, `_bind_sightings`
checks the read against the team sheet and tells the registry, and the
eight-second carry rule (`012c232`) keeps a spoken name on the ball. That is
the "tags off" arm above. `Sighting.mark`, `Topic.TRACKS`, `Topic.GALLERY`,
`--marks`, the `no-marks` baseline, checklist item 7's "carried on a mark"
and item 12's tracker rate are gone with it; the `tracks` and `gallery` rows
in older traces are history.

Untested and worth $1.20 if anyone cares: the same six clips on Opus with
tags off, to see whether an Opus caller alone clears the 60% naming bar.

## 8. The audio triggers are gone too

The whistle and crowd-roar detectors, the audio ring, the second ffmpeg
process that fed them and the simulator's synthetic sound are removed. The
camera-cut detector stays; it is video. Measured across all 58 real-clip
traces on disk, no new spend:

- **Whistle**: fired once in 58 runs. A 2.2-4.2 kHz tone never stands out of
  a broadcast mix with two commentators in it.
- **Roar**: a median 4 firings a minute on every clip, event or no event. On
  the Di María clip (ball over the line at video 58, celebration to 80) it
  fired at 31, 35, 40, 44, then 81, 104, 108, 115 — nine runs, zero roars in
  47-80. Lines it triggered alone were about an event 32% of the time; the
  silence timer's lines, 32%; camera cuts, 60%. Roar-only calls fired a
  median 4.4 s after the previous one, which is the rate cap, so they
  advanced nothing.
- **Roar as goal evidence**: when the board was unreadable a roar anywhere
  in a nine-second window let a goal claim through the gate. That window
  was open a median 50% of the time and up to 96%. In the three shootout
  clips every goal and penalty claim passed on it. They happened to be
  right.

So `Trigger.WHISTLE`, `Trigger.ROAR`, `roar_ratio`, `whistle_band_hz`,
`whistle_ratio`, `AudioChunk`, `AudioRing`, `FileCapture.audio`,
`SimSource.audio`, `MatchAudio`, `_celebration_ahead` and the gate's
`lookahead_celebration` route are gone. **A goal now needs the board or the
wire, full stop.** In a shootout, where no broadcaster shows a clock or a
running score, that means goal claims are refused until a graphic changes.
That is honest, and it will cost shootout lines. The cheap fix if it
matters: a Haiku yes-or-no on the two lookahead frames, "is this a goal
celebration", as the independent second source the roar was pretending to
be — well under a cent a claim.

## 9. The prompt, the cadence, and what did not happen

**Real commentary, measured.** From the 2022 final's YouTube captions
(`clips/argfra-dimaria.en.json3`) and local Whisper transcripts of three
short clips; the analysis with thirty quoted lines is in
`runs/prompt-name/REAL_COMMENTARY.md`. Ten windows of live open play, 115
utterances: median 5 words, 24% two words or fewer, 19% a bare surname,
longest 28 (the existing cap). Whole match: median gap between utterances
2.4 s, 78% of gaps under 4 s. This system's floor was 4 s between lines.

**What was done with that.** The caller's Good examples are now real lines
("De Paul." / "Messi, Álvarez." / "Save. The deflection off Varane flies
wide."); "one sentence" is gone; the subject is the player on the ball,
named if read. The rate cap is earned by the last line (words over
`WORDS_PER_SECOND` plus 0.8 s, floored at 1.5, capped at 4.0). The
minimum-words gate rule now applies only after a trim. The similarity veto
is gone.

**What happened: nothing.** Two short clips with the new prompt, then the
Mbappé clip with the new cadence as well, all Opus: no line under seven
words. Every rate-cap refusal printed the 4.0 s cap because no line was
short enough to earn less. Naming did not move on the two short clips (1 of
5 to 1 of 4; 1 of 3 to 1 of 3). The Mbappé run named players in 50% of live
lines against 33% and called 6 of 9 events against 3 of 8, including the
penalty the old run missed — but that baseline is three code generations
back and proves nothing about cadence.

**What the new style did break, both fixed with tests.** Two-sentence lines
put a capitalised word after a full stop, and the gate read it as a name:
"Everything in this final waits on him" went out as "in this final waits on
him". Any word that opens a sentence in the line is now exempt, in the gate
and the grader. And "the Argentines" was an off-roster name to the grader;
demonym plurals are team words now.

**1280.** Corner clip: 7 sightings and 1 of 5 lines named at 768; 17 and 3
of 5 at 1280. Offside clip: 13 and 2 of 4 at 768; 10 and 0 of 3 at 1280 —
that run read four names in one call and used none. 1.75x the cost. Six
clips twice each, about $4, would settle it; not worth it before the live
run.

Traces for all of this: `runs/abl/`, `runs/w1280/`, `runs/prompt-name/`,
`runs/cadence/`.

## Things the next person will trip over

- **The simulator's timestamp strip is in the top-left corner**, and the
  oracle decodes it from the frames the caller is given. Nothing draws on
  those frames any more; if that changes, keep that corner clear.
- **Three runtime tests are timing-sensitive** because the sim runs flat out
  against wall-clock loops: `test_a_goal_is_never_announced_before_the_board_confirms_it`,
  `test_only_the_wire_can_put_a_lied_about_score_right`, and anything
  comparing single-digit error counts between two short runs.
  Both were rewritten this session to assert properties rather than
  statistics; keep it that way.
- **Commit `9b76685` contains a mid-flight snapshot** of
  `perception/players.py` swept in by a broad `git add` while a parallel agent
  was still writing it. That one commit does not pass mypy on its own. HEAD
  does. Left as is rather than rewriting history.
- **The brief's opening line points at a worktree that no longer exists.**
  Left as the user wrote it; this file has the current layout.
- **The openers list is gone and so is the position-zero trim** (`40e8040`),
  and since `193ab26` the same exemption covers a word that opens any
  sentence inside the line.
  Anything that reintroduces "strip an ordinary word off the front of a line"
  is reintroducing nineteen damaged lines for zero catches.
- **`tests/test_baselines.py::test_only_the_wire_can_put_a_lied_about_score_right`
  failed once in about fifteen runs**, and only while a second full suite was
  running against the same machine. `blind` came back 1-1 rather than 0-0.
  Its docstring says the board reader lies on every read at `error_rate=1.0`,
  and `SimOracle._board` does not lie at all — the rate only injects errors
  into caller lines — so what holds the score at 0-0 is the clip ending
  before three agreeing post-goal reads land, which is a race. Nothing to do
  with A16: eight runs of it after that change all passed.
- **There is no fixed window on talking about a goal.** It runs from the
  cursor the state applied it at until play restarts — a kickoff line — with
  `GOAL_TALK_CAP_S` (150 s) as the backstop. A test asserting that a celebration line expires
  on a clock is asserting the bug that rejected two correct lines.
- **The board tracker now believes a score change it saw before a replay.**
  A pending change survives absent reads and confirms whenever the bug comes
  back, stamped at the first read that saw it. Anything asserting the old
  "a replay interrupts the evidence" behaviour is asserting a bug.

---

## Where things deviate from the brief

Six, all deliberate:

1. **The team sheets print the full name**, `#11 Ángel Di María (LW)`, not the
   `rsplit`-derived surname the brief specified. That would print "María" and
   "Bruyne" — exactly the names A13 is about — and tell the caller to say the
   wrong one. The rule still says to use the surname.
2. **A13's grader half is name-token membership, not the suffix rule.**
   `factual_errors` scans single capitalised words, so "Di" arrives alone and
   a suffix rule cannot match it. The gate got the suffix rule as written.
3. **The wire is polled from the frame loop, not the board loop.** Tying
   release to board-read cadence made a goal's arrival depend on when the
   score bug was last glanced at, which is nothing to do with it.
4. **`WireSync` will not re-stamp an event that arrived with a `video_ts`** —
   the simulator's own truth. Otherwise the kickoff offset moves every event.
5. **A16's account of the second run is half right, and the fix is right
   anyway.** The brief has the bug absent from 70 to 132 and the state stuck
   at 1-0 through the celebration. The trace has one absent read at 70.2 at
   confidence 0.15 — discarded before it could clear anything — and the state
   reaching 2-0 at cursor 66.8; the eleven confident absent reads run 91.0 to
   132.4, after confirmation. So the pending-survives-a-replay fix does not
   rescue the two late rejections at 128.1 and 140.1 in *this* run. It is in
   because a confident absent read at 70.2, which is what the board reader
   will usually return, would have held confirmation until 136 with the state
   saying 1-0 throughout.
6. **B10's wire error-count assertion was replaced.** At `error_rate=1.0` each
   run speaks one or two lines, so the comparison failed about one run in
   three on noise. What is tested instead is deterministic and stronger: a
   board lying on every read cannot move the score at all, and the wire moves
   it to exactly the truth.

Three fixes were not in the brief and came out of actually running the thing:
an ffmpeg deadlock when one of two streams goes unread, the gate trimming the
kit colours it now instructs the caller to use, and both trackers owning a
registry the runtime never saw — so the marks were drawn and
`state.identified` stayed empty all run.
