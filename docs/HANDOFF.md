# Handoff

Start here, then read `docs/CLIPS.md`. This file is the state; that one is the
evidence.

**HEAD:** `main` in `/Users/Aarsh/Desktop/commentary`. One repo, one branch, no
PRs. `.env` at the root. `clips/` and `runs/` are in the repo and gitignored —
the clips are 26 MB each and the traces are somebody's API spend. HEAD as of
14 Sep 2026 is `fd699e4`.

**Gates, green at every commit:** `uv run pytest -q` · `uv run ruff check .` ·
`uv run mypy`. Run `uv sync --all-extras --dev` first: without the `tools`
extra, mypy reports nine import errors in `mcp_server.py` that are the missing
package and not the code. There is no `vision` extra any more; section 6 says
why. There *is* an `audio` extra again since `3b7a99a` — it is PortAudio
(`sounddevice`) and nothing else, and it is what plays raw PCM in process.

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

`--source screen` is the live path and it works (section 3c).
`--voice log|say|elevenlabs` on `run`; `--serve` adds the watch page on
`:8000`.

**Replaying a finished run costs nothing**, calls no model and needs no key:

```
uv run python -m commentary replay --trace runs/<name>/<run>.jsonl \
    --path clips/<clip>.mp4 --start 20 --serve --port 8000
```

`--start` is how far into the clip the original run began; a `--source file`
run began at 0, the screen runs played a clip in a player and started part-way
in. `--loop` plays the trace again when the clip ends, and speaks on every lap.
`--voice say|elevenlabs` hands the trace's beats to a real director and a real
speaker, so a change to a voice, a sink or the director can be heard on lines
that were already paid for; `--out` writes a trace of the playback, which is
where `seconds` and `first_audio_s` come from.

**Env that matters.** `CALLER_MODEL` / `BOARD_MODEL` / `ANALYST_MODEL` in
`.env`: Opus 5 calls, Haiku 4.5 reads the board, Opus 5 on the analyst.
`DELAY_S` 8. `MAX_USD_PER_MATCH` 35. `CALLER_FRAME_WIDTH` (default 768) is the
width the caller's frames go out at; 1280 costs 1.75x and two clips could not
tell it from noise (section 7). `PRESENT_OFFSET_S` (default 3.5) holds the
picture served to the viewer that far behind the narration cursor; it is
clamped to the oldest frame the buffer still has, and `CaptureConfig` refuses
an offset deeper than `history_s` (6.0). `AVFOUNDATION_DEVICE` defaults to the
**name** `"Capture screen 0"` — never an index, which moves.

**About $0.30 a minute of video** on Opus: a three-minute clip costs $0.95 to
$1.11, a forty-five-second clip $0.15 to $0.25. On Haiku for the caller and
analyst a forty-five-second clip is $0.04 to $0.06. Nothing runs locally but
ffmpeg, OpenCV and numpy; nothing downloads at run time.

`grade` aligns StatsBomb onto video time from the trace's own board readings
and **refuses to grade an alignment it cannot trust**. A shootout has no clock
on any broadcast, so those need `--offset <seconds>` measured off a frame by
hand, and the output says so above every number.

## 2. Where it got to

Twenty-two runs on eighteen clips from six matches and four broadcasters:
**no wrong player name has ever reached air, on any trace, on Opus.** Zero
phantom events on any Opus clip. Phantom *goals* did reach air once, on Haiku,
during a replay — section 3c, and the gate rule that now stops it.

| pass | clips | events called | detail |
|---|---|---|---|
| the tuned clip | 1 (re-run many times) | goal called and named | best run 10 of 12, `runs/A` |
| same broadcast | 5 × 3 min | 5 of 6 headline events | `CLIPS.md`, "The first pass" |
| unseen matches | 12 × 45 s | 8 of 12, 9 counting e01's first run | `CLIPS.md`, "The short clips" |
| after the fixes | 3 reruns | e07 and e02 fixed, e01 inconclusive | `CLIPS.md`, "The twelve, and the three rerun" |
| trigger rerun | `clips/mbappe.mp4` | 5 of 8, both goals, 12 of 12 | `runs/trigger/mbappe` |

By headline event per clip it is 13 or 14 of 18 depending on whether e01 is
scored on its first run or its rerun. The per-clip tables in `CLIPS.md` are the
source.

What generalised from the tuned clip: the event vocabulary, the gate, and the
board reader — which read a **two-row CONMEBOL bug with the clock underneath
the score** 13 times out of 13 and a 2018 bug with the clock on the right
perfectly, having seen neither before. A rescaled 4K screen grab is no harder
than a file: 16 reads out of 16 on the live path. What did not generalise:
naming a player in open play.

## 3. What changed in the pass before this one

| commit | |
|---|---|
| `4703547` | `grade` answers "was the event called", not just "did a line land near it". Recall read 67-100%; called read 17-75%. |
| `0dc6035` | `fold` glued a possessive s onto the name — "De Gea's" became "de geas" — and trimmed the keeper out of a penalty, twice. |
| `228a768` | "the Dutchman" read as an invented surname; the demonym's -man form goes in beside it. |
| `40e8040` | The position-zero name trim is gone. Nineteen firings in nine runs, zero catches. A test asserts the trade: an invented surname put first now reaches the microphone. |
| `1a54a3a` | One goal-claim definition in the gate, and the runtime asks it: a goal from a set piece is a goal to the director whatever the caller tagged. |
| `dd85151` | A line that claimed a goal lifts the four-second rate cap for the next call. |
| `d18358c` | `stoppage` in the event vocabulary, so a player lying injured is not a phantom foul. |
| `012c232` | Carry rule: a name stays on the ball for eight seconds, same phase, same side. Plus `name_withheld` logged when a line says "the taker" and its own sightings name him. |
| `9bda1db` | "and it is in" written out, which unifying the goal definitions had lost. |
| `clips/build_pack.py` | `NAME_FIXES`: StatsBomb's nickname field calls Randal Kolo Muani "Randal Kolo", and two true lines died of it. |

## 3b. What changed on 13 September

| commit | |
|---|---|
| `99dc36c` | The player tracker, the gallery, the kit split and the marks are gone, with torch. A real-clip ablation: without them the caller named 20 distinct players against 8 with them, and bound 96% of sightings against 30%. Section 6. |
| `2b96b44` | The whistle and roar detectors and the audio pipeline are gone. Whistle fired once in 58 runs; roar fired every five seconds regardless and missed the tuned clip's goal nine runs out of nine; it was also holding the goal gate open half the time when the board was unreadable. Section 6. |
| `f43d5ed` | `CALLER_FRAME_WIDTH`. Two clips at 1280 went one up, one down, at 1.75x the cost. Default stays 768. |
| `e1166e0` | The caller's examples are real broadcast lines; "one sentence" is gone from the rules; the three "use the name" paragraphs are one. And a gate bug: the minimum-words rule applied to every line, not only to trimmed ones, so any line under three words died. |
| `193ab26` | The rate cap is earned by the last line's length (floor 1.5 s, cap 4 s). Two trims the new multi-sentence lines exposed: a capitalised word opening a second sentence was read as a name, and a plural demonym ("Argentines") was off-roster to the grader. Both fixed in gate and grader. |
| `31b6a88` | The similarity veto on repeats is gone from both voices. Zero firings in 63 runs and 1,062 calls; the prompt's "last five lines" is what stops a repeat. The analyst's echo check against the caller stays. |

## 3c. What changed on 13-14 September

Everything here came out of traces on disk or runs under $1.20. **$1.76 of
Anthropic spend** across the traces written after the last handoff
(`runs/trigger`, `runs/sonnet`, `runs/screen`, `runs/voice/dimaria-goal`), and
**about 510 ElevenLabs credits** across the two voiced sessions, counted at
0.5 credit a character on Flash over their spoken and preempted rows.

| commit | |
|---|---|
| `46552ec` | The *first* board read that disagrees with the settled score in a goal's direction fires the trigger; confirmation still needs three and still fires again. This closes the "call refused just before the kick" gap. |
| `6447d5f` | The screen source emitted about 470 frames a second whatever `-framerate` asked for, because ffmpeg's constant-rate output duplicates to the device timebase; the buffer's capacity is a frame *count*, so it held under half a second and `ready` never came. `-r` after `-i` pins it at 15. Device default is now the name `"Capture screen 0"`: the screen was index 1 until an iPhone was plugged in, then 3. |
| `e022812` | The watch page was parsing a stream the runtime stopped sending: `TriggerName` still listed whistle and roar, `names_read` had been `sightings` for two generations so the panel said "nothing legible" all match, and six caller events were coerced to `none`. Sightings and corrections get panels of their own. |
| `a66128d`, `55a34a1` | `commentary replay`: a trace back onto the bus with its clip through the same `FileCapture` and `DelayBuffer`, satisfying `RuntimeHandle` so `create_app` is reused unchanged. Free. README "Playground" section. |
| `356750d`, `7c48099` | `PRESENT_OFFSET_S`, default 3.5 s — the measured caller round trip. `/api/video` serves the frame at `cursor − offset`, clamped to the oldest frame held; `/api/state` reports the figure so the stage can label it, and says nothing extra at 0. Replay schedules rows carrying a `live_ts` at `live_ts − delay_s`, the cursor time the run actually published them at. |
| `dfa177f` | `--voice say`: `SaySpeaker` runs macOS's `say` as a cancellable subprocess, same contract as `ElevenLabsSpeaker`, no key. |
| `d7d7e43`, `fd699e4` | `replay --loop` seeks the clip back and replays the trace until Ctrl-C or `--seconds`, bus and server up across the seam, and speaks on every lap because `_restart_pass` re-stamps `created_ts`. |
| `5636351` | The gate's `score_claim` rule: a number put on the score must be a number the state holds. Below. |
| `3b7a99a`, `fccbaa8` | ElevenLabs streams `pcm_22050` into an in-process `sounddevice` sink; `FFplaySink` stays for the mp3 formats. `first_audio_s` on every spoken row. The write path gives up after five seconds of a card taking nothing, and `finish` aborts rather than drains. |
| `d06e1cc` | `replay --voice`: the trace's beats go to a real director and speaker, `created_ts` rewritten to now or every line is hours past its ageing limit, and the trace's own `spoken`/`preempted` rows are dropped in favour of this playback's. |

**The Mbappé rerun** (`runs/trigger/mbappe/file-20260913-185228.jsonl`,
`clips/mbappe.mp4`, Opus, 210 s). The board bug flipped at live 86.1; the old
trigger waited for confirmation and the goal went out at 87.9, after a line
about the run-up at 76.6 and a lull handed to the analyst at 80.8.

| | before | after |
|---|---:|---:|
| penalty called at cursor | 87.9 s | **82.5 s** |
| cost | $1.11 | $1.11 |
| grade: factual errors | 0 | 0 |
| grade: definition of done | — | 12 of 12 |
| events called | — | 5 of 8 said, both goals |
| names spoken / wrong | — | 26 / 0 |

**The live screen path, first run**
(`runs/screen/dimaria/screen-20260913-222906.jsonl`). 60 s, Haiku caller, a
full-screen 4K playback of `clips/argfra-dimaria.mp4` from 20.2 s in, at the
default crop, `--voice log`.

| | |
|---|---:|
| board reads legible | 16 of 16, confidence 0.95 |
| lines: spoken / judged / passed | 9 / 9 / 9 |
| cost | $0.075 |
| caller round trip (`live_ts − video_ts − 8`) | median **3.4 s**, range 1.9 to 5.6 |

The same median holds on Opus over file runs, which is where the 3.5 s default
for `PRESENT_OFFSET_S` comes from. That round trip is the whole reason the
offset exists: a caller line is stamped with the cursor when the call *begins*
and reaches the viewer when the model answers, so without it every line landed
three and a half seconds after its moment had gone past on screen.

**The goal on that run, measured against the clip.** The ball crosses at video
37.8 s (clip 58.0, offset 20.2 confirmed off the bug's clock). "Di María
strikes." went out at cursor 39.3; "Glorious goal. Di María." at cursor 43.2,
**5.4 s after the ball crossed**, and half a second before the state took the
change in at 43.9. So the goal line still lands on the celebration, not the
crossing — see section 4.

**The first phantom goals ever on air.**
`runs/voice/dimaria-goal/file-20260914-015533.jsonl`, Haiku caller, during a
voiced replay at 2-0: the caller wrote "Messi, from the rebound! Argentina's
third." and then "De Paul! Argentina's fourth!", and the gate passed both with
no reason recorded. Nothing was wrong by the goal rule — a goal *was* in the
state, and a goal in the state was cover for every line that came after one. It
was not cover for the arithmetic, and nothing was checking it. `5636351`: an
ordinal is a scoreline with one number left out, so it is checked like one. A
side that has scored *n* may be said to have scored *n*, and *n+1* only while a
goal is being called that the state has not taken in yet — the same latitude
the caller already had for speaking ahead of the graphic. The scoreline rule
shares that arithmetic and gains the latitude it never had; it also hears "3
nil" and "2 to 1" now. An ordinal counting anything but goals ("their first
real chance") is left alone. Rejection is whole-line, no trim.

**What the PCM sink bought.** One real replay,
`runs/voice/replay-pcm/replay-20260914-022058.jsonl`, against the ffplay run
above.

| | ffplay (mp3) | PCM sink |
|---|---:|---:|
| the four lines that hit the drain timeout | 5.10 / 5.22 / 4.92 / 5.50 s | 4.48 / 4.10 / 4.53 s and one cut |
| mean seconds per line | 4.01 | 3.78 |
| first audio, once warm | not recorded | 0.22 to 0.39 s |
| first audio, first stream of the process | — | 1.81 s |

Per-line seconds barely moved, and that is the finding: the free tier's PCM
stream arrives at about playback speed, so a six-word line still holds the
channel 2.8 to 3.2 s. Section 4.

**Sonnet versus Opus.** `runs/sonnet/RESULTS.md` has the table. Sonnet is about
half the cost ($0.10 a clip against $0.20) and called 4 of 6 events against
Opus's 5 of 11 on the same clips — but **8 of 23 Sonnet spoken lines carry code
tokens** glued to the line field (`.replace('','')`, `.strip()`, `.line`), and
the gate let every one through. Opus produced none in 63 runs. **Stay on
Opus.** The two untried fixes: a gate rule that cuts a line at the first
non-prose token, and a stricter output schema for the caller.

**ElevenLabs, what a match would cost.** Spoken volume measured across four
traces runs 256 to 711 characters a minute, so ninety minutes is roughly 23,000
to 64,000 characters, which at Flash's 0.5 credit a character is **12,000 to
32,000 credits**. The free tier is 10,000 credits a month and carries no
commercial licence; the Creator plan (about $11 for the first month, 121,000
credits) covers the demo with room to spare.

## 4. Known gaps

- **Open-play naming, and what it actually is.** Across 208 live-play caller
  lines on Opus, 57% had no sighting at all — nothing legible in the frame —
  10% had sightings that bound to nobody, and 33% had a bound name. Of those
  69, 45 said the name and 24 withheld it, mostly correctly. So the gap is
  pixels, not prompt: a shirt number in a wide shot at 768 wide is four pixels
  tall. 1280 did not settle it in two runs (section 7). The carry rule and the
  registry are the room that is left.
- **The goal line still lands after the cut, not on the crossing.** `46552ec`
  moved the trigger five seconds earlier on the Mbappé penalty, where the
  caller was watching the run-up and the kick was in the lookahead. It did not
  fix the open-play case: on the screen run the ball crossed at video 37.8 and
  the goal line went out at cursor 43.2, by which point the broadcast had cut
  to a close-up. The trigger fires on the bug, and a broadcaster's bug lags the
  ball by about six seconds. Nothing cheap is left here except a second source
  of evidence — the Haiku yes-or-no on the lookahead frames described in
  section 6.
- **The ElevenLabs stream is a floor under line length.** The cadence work
  bought a 1.5 s rate-cap floor and real commentary's median utterance is five
  words. On the free tier's `pcm_22050` stream, no line in the replay came out
  under 2.8 s and a six-word line took 3.2 s, because the bytes arrive at about
  playback speed. Until that changes, short lines buy less than the cadence
  numbers suggest. A paid tier may or may not stream faster; untested.
- **The scoreboard clock on the page freezes between state publishes.** A
  `state` row goes out only when something changes: 2 rows in the 60 s screen
  run, 7 in the 210 s Mbappé run. Between them the page shows a stopped clock.
  Either tick it client-side off `clock_s` and the row's timestamp, or publish
  state on a timer.
- **Haiku phantoms.** The `score_claim` rule (`5636351`) stops the two lines
  that got out, and a test asserts it. It has not been re-run against a Haiku
  caller on a goal clip; that costs about $0.06 and has not been done.
- **Fragments do not come.** With real fragment examples in the prompt and a
  cadence that rewards short lines, the Opus caller wrote no line under seven
  words on the Mbappé clip. Three gates stood in front of the style; two are
  open now (section 7); the third is the caller's own choice, and the fourth is
  the audio stream above.
- **No run longer than 3.5 minutes.** The Mbappé rerun reached cursor 202 s.
  Nothing has shown the state, the cost cap, the goal-talk cap or the analyst's
  spacing over a half.
- **Voice has never run against a live source.** `--voice elevenlabs` has run
  on a file and on a replay. It has not run with `--source screen`, and the
  screen path has not run longer than 60 s.
- **Shootouts.** The FIFA feed shows a tally strip at the bottom centre,
  intermittently, outside the board reader's crop. A second crop would read it.
  Decided not to build it; a league match has no shootout.
- **StatsBomb's card timestamp is not when the card is shown.** Unchanged; it
  cost two card clips.
- **Grading a 2026 fixture.** StatsBomb open data is historical. The ESPN feed
  adapter (`commentary feed`) exists and is unverified against a current match.
  The Opus judge can score a trace without a feed.

## 5. Next steps, in this order

The target is not a full match. It is **testing on 60-90 second clips with
voice**, then a fixture. The fixture and the app it streams in are both still
unchosen, and that choice blocks everything from step 4 on.

**Now, from clips on disk, no match needed and no model called.**

1. **60-90 second voiced replays.** `replay --voice elevenlabs` over the
   traces already in `runs/` — the Di María goal, the Mbappé penalty, two or
   three of the twelve short clips. ElevenLabs credits only, about 260 a
   minute of commentary. What to watch: whether the goal cuts the analyst
   cleanly, whether `first_audio_s` stays under half a second across a longer
   trace, and whether the line-length floor above shows up as dead air.
2. **The same, with `--serve`, screen-recorded.** This is the demo artefact,
   and it costs nothing but credits. Fix the frozen page clock first if it is
   visible on camera.
3. **One Haiku goal clip through the new gate rule**, about $0.06, to see the
   `score_claim` rejection fire on the lines that got out.

**Once the fixture is picked. Ask before any of this — it all spends.**

4. **Which app or site the broadcast plays in** decides the capture device and
   whether the picture is capturable at all. Ask first.
5. **Build the pack with the researcher** for the real fixture and check the
   numbers and kits by hand. A wrong number here is a wrong name on air.
6. **Set the score-bug crop** for that broadcaster with `commentary crop` on
   any recent frame of its feed. The reader has read three unfamiliar layouts
   perfectly; the crop still has to point at the corner.
7. **One 20-minute dry run** off the same broadcaster if a recording is
   obtainable, about $6. First run past four minutes.
8. **The live run**: `MAX_USD_PER_MATCH` set, trace on, screen-record the watch
   page with audio. Fallback if capture fails: record the broadcast and run it
   as a file twenty minutes behind; the demo survives.

**After.** Demo video, README numbers, this file. Judge the trace with
`commentary grade` sans feed and the Opus judge.

Skip: grading the live match against a feed; the shootout crop; anything
tracker-shaped; more work on fragments; the simulator ablation table.

## 6. Do not do these again

- **Ask before any paid run.** Every number in section 3c came from a trace
  already on disk or from a run that was agreed first. The one time that
  slipped, five runs fired from one script before a change of plan landed and
  $4.90 went on clips that had just been deprioritised.
- **Discuss before launching implementation.** More than one thing in section
  3c was built before it was clear it was wanted.
- **Never cut `say` mid-word on this Mac.** Replaying a whole trace through
  `--voice say` cuts `say` off on every preemption, and two of those wedged
  CoreAudio: nothing on the machine would play — not `say`, not PortAudio, not
  the browser — until `coreaudiod` exited. The sink now gives up on a stalled
  device after five seconds so the match keeps calling, but the sound does not
  come back on its own. Use `--voice say` to check the plumbing, not to
  rehearse.
- **`EnterWorktree` grabs the Desktop repo.** `~/Desktop` is itself a git repo
  and claims this directory, so the tool can land you in a worktree of the
  wrong project. Make one by hand:
  `git -C /Users/Aarsh/Desktop/commentary worktree add -b <branch> .worktrees/<name> main`.
- **No batch scripts that cannot be interrupted.** See the first item.
- **`--seconds` must be the clip length plus at least twenty.** The cursor is
  pinned at the live edge minus the delay, so the last eight seconds of a clip
  are never narrated whatever you pass, and the caller needs another twelve on
  top to write about what it saw. `--seconds 50` on a 45 s clip lost a goal the
  trace shows the system had already called internally.
- **Do not centre a card clip on the StatsBomb timestamp.** Check the frame
  first; the broadcaster shows the card later, often after the ball goes out.
- **Do not take a simulator A/B as evidence about vision or audio.** The sim
  oracle decodes a timestamp and never reads a pixel. The marks A/B in the old
  README was 12 lines on the sim and it was the whole case for a stack that
  lost on real clips.
- **Ask the traces before building.** The tracker, the whistle, the roar and
  the repetition veto were each answered from `runs/` for nothing before any
  code moved. `runs/readtrace.py` prints a trace; the trigger, sighting, gate
  and spoken rows carry everything needed.

## 7. What was removed, and why it stays removed

**The tracker** (`99dc36c`). RF-DETR, ByteTrack, the HSV kit split, the SigLIP
gallery and the whole `vision` extra with torch under it, removed after a
real-clip ablation: six 45 s clips, two runs each, tags on against tags off,
Haiku in both arms, $1.15 all in. Traces and table in `runs/abl/`.

| | tags on | tags off |
|---|---:|---:|
| spoken lines | 40 | 45 |
| lines naming a roster player | 26 | 43 |
| distinct players named | 8 | 20 |
| sightings bound / made | 70 / 235 | 115 / 120 |
| wrong or off-roster names | 0 | 0 |

With tags the caller named the close-ups and little else, and filed letter tags
as sightings that bound to nobody. The median track lived 2 to 2.8 s, no longer
than the caller's own frame window. What replaced it: the caller reads shirts,
`_bind_sightings` checks the read against the team sheet, and the eight-second
carry rule keeps a spoken name on the ball. `Sighting.mark`, `Topic.TRACKS`,
`Topic.GALLERY`, `--marks` and the `no-marks` baseline are gone with it; the
`tracks` and `gallery` rows in older traces are history.

**The audio triggers** (`2b96b44`). Measured across all 58 real-clip traces on
disk, no new spend. Whistle fired once in 58 runs — a 2.2-4.2 kHz tone never
stands out of a broadcast mix. Roar fired a median 4 times a minute on every
clip, event or no event, and on the Di María clip fired at 31, 35, 40, 44, then
81 — nine runs, zero roars in the 47-80 window where the goal actually was.
Worse, when the board was unreadable a roar anywhere in a nine-second window
let a goal claim through the gate; that window was open a median 50% of the
time and up to 96%.

**So a goal now needs the board or the wire, full stop.** In a shootout, where
no broadcaster shows a clock or a running score, goal claims are refused until
a graphic changes. That is honest and it will cost shootout lines. The cheap
fix if it ever matters — and the same fix the goal-timing gap in section 4
wants — is a Haiku yes-or-no on the two lookahead frames, "is this a goal
celebration", as the independent second source the roar was pretending to be.
Well under a cent a claim.

**1280-wide caller frames** (`f43d5ed`, default stays 768). Corner clip: 7
sightings and 1 of 5 lines named at 768; 17 and 3 of 5 at 1280. Offside clip:
13 and 2 of 4 at 768; 10 and 0 of 3 at 1280. 1.75x the cost, and six clips
twice each (about $4) would settle it. Not worth it before the live run.

**Real commentary, measured** (`runs/prompt-name/REAL_COMMENTARY.md`). From the
2022 final's YouTube captions and local Whisper transcripts of three short
clips: 115 utterances over ten windows of live open play, median 5 words, 24%
two words or fewer, 19% a bare surname, longest 28. Whole match: median gap
2.4 s, 78% of gaps under 4 s. This system's floor was 4 s. That is what the
cadence work (`e1166e0`, `193ab26`) was aimed at, and section 4 says how far it
got.

## Things the next person will trip over

- **Screen capture needs macOS Screen Recording permission** for the
  application running the session — the terminal, or the agent harness, not
  ffmpeg. Without it ffmpeg hangs silently: no error, no frames, no timeout.
- **The avfoundation device index shifts** when an iPhone or anything else
  capturable is plugged in. The screen was index 1, then 3. Use the name;
  `bash scripts/list_devices.sh` prints it.
- **The simulator's timestamp strip is in the top-left corner**, and the oracle
  decodes it from the frames the caller is given. Nothing draws on those frames
  any more; if that changes, keep that corner clear.
- **Three runtime tests are timing-sensitive** because the sim runs flat out
  against wall-clock loops:
  `test_a_goal_is_never_announced_before_the_board_confirms_it`,
  `test_only_the_wire_can_put_a_lied_about_score_right`, and anything comparing
  single-digit error counts between two short runs. They were rewritten to
  assert properties rather than statistics; keep it that way.
- **Commit `9b76685` contains a mid-flight snapshot** of
  `perception/players.py` swept in by a broad `git add` while a parallel agent
  was still writing it. That one commit does not pass mypy on its own. HEAD
  does. Left as is rather than rewriting history.
- **The openers list is gone and so is the position-zero trim** (`40e8040`),
  and since `193ab26` the same exemption covers a word that opens any sentence
  inside the line. Anything that reintroduces "strip an ordinary word off the
  front of a line" is reintroducing nineteen damaged lines for zero catches.
- **There is no fixed window on talking about a goal.** It runs from the cursor
  the state applied it at until play restarts — a kickoff line — with
  `GOAL_TALK_CAP_S` (150 s) as the backstop. A test asserting that a
  celebration line expires on a clock is asserting the bug that rejected two
  correct lines.
- **The board tracker believes a score change it saw before a replay.** A
  pending change survives absent reads and confirms whenever the bug comes
  back, stamped at the first read that saw it. Anything asserting the old "a
  replay interrupts the evidence" behaviour is asserting a bug.
- **A replay's `cost` rows are the original run's**, republished. Do not add
  them up as new spend: `runs/voice/replay-pcm` reports $0.10 and called
  nothing.

---

## Where things deviate from the brief

Six, all deliberate, and none of them has been revisited:

1. **The team sheets print the full name**, `#11 Ángel Di María (LW)`, not the
   `rsplit`-derived surname the brief specified. That would print "María" and
   "Bruyne" and tell the caller to say the wrong one. The rule still says to use
   the surname.
2. **A13's grader half is name-token membership, not the suffix rule.**
   `factual_errors` scans single capitalised words, so "Di" arrives alone and a
   suffix rule cannot match it. The gate got the suffix rule as written.
3. **The wire is polled from the frame loop, not the board loop.** Tying release
   to board-read cadence made a goal's arrival depend on when the score bug was
   last glanced at.
4. **`WireSync` will not re-stamp an event that arrived with a `video_ts`** —
   the simulator's own truth. Otherwise the kickoff offset moves every event.
5. **A16's account of the second run is half right, and the fix is right
   anyway.** The trace has one absent read at 70.2 at confidence 0.15,
   discarded before it could clear anything, and the state reaching 2-0 at
   cursor 66.8. So the pending-survives-a-replay fix does not rescue that run's
   two late rejections. It is in because a *confident* absent read at 70.2,
   which is what the board reader will usually return, would have held
   confirmation until 136 with the state saying 1-0 throughout.
6. **B10's wire error-count assertion was replaced.** At `error_rate=1.0` each
   run speaks one or two lines, so the comparison failed about one run in three
   on noise. What is tested instead is deterministic and stronger: a board lying
   on every read cannot move the score at all, and the wire moves it to exactly
   the truth.

Three fixes were not in the brief and came out of actually running the thing:
an ffmpeg deadlock when one of two streams goes unread, the gate trimming the
kit colours it now instructs the caller to use, and both trackers owning a
registry the runtime never saw.
