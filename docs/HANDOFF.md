# Handoff

Start here, then read `docs/CLIPS.md`. This file is the state; that one is the
evidence.

**HEAD:** `main` in `/Users/Aarsh/Desktop/commentary`. One repo, one branch, no
worktrees, no PRs. `.env` at the root. `clips/` and `runs/` are in the repo and
gitignored — the clips are 26 MB each and the traces are somebody's API spend.

**Gates, green at every commit:** `uv run pytest -q` · `uv run ruff check .` ·
`uv run mypy`. Run `uv sync --all-extras --dev` first: without the `tools`
extra, mypy reports nine import errors in `mcp_server.py` that are the missing
package and not the code. There is no `vision` extra any more; see section 7.

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

## 3. What changed in this pass

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

## 4. Known gaps

- **Open-play naming.** `name_rate` in live play across the three-minute
  clips: 33%, 18%, 0%, 33%, 73%, 38%. The 73% is the shootout, which is all
  close-ups. The system names players when the camera is close enough to read
  a shirt, and open play is where a listener most wants a name.
- **A call refused just before the kick.** Mbappé's penalty was missed because
  the call that would have covered it was refused one tenth of a second under
  the rate cap, when nothing — board, caller, lookahead — knew a goal was
  coming. `dd85151` does not fix this and its message says so.
- **StatsBomb's card timestamp is not when the card is shown.** e06's yellow is
  stamped 89:05 and the frame at 89:07 is live play with no referee in it. It
  cost two card clips.
- **Shootouts have no board.** Three shootout clips on two matches: zero
  usable readings in all three.
- **The e01 rerun is inconclusive.** It produced no goal line at all, which is
  variance rather than regression — the carry worked elsewhere in the same run
  — but one run of something this variable proves nothing.
- **`name_withheld` is new.** Only the three reruns have it in their traces;
  counting it across the older ones will find nothing.

## 5. Next steps, in this order

1. **Open-play naming.** Start free: count `name_withheld` across traces to
   split "never identified" from "identified and not used" — the Messi penalty
   was the second, and that is a different problem from the first. The
   tracker is not the answer (section 7); the carry rule and the registry are
   where the room is.
2. **Use the eight-second lookahead before refusing a call.** The director
   should scan the lookahead frames for a celebration or a board change before
   the rate cap says no. It covers the Mbappé case and also the card shown
   after the referee's decision.
3. **One fifteen-minute run on an unseen match, $5-6**, as the gate before
   anything goes live.
4. **A live source** — yt-dlp or capture — with measured end-to-end
   latency.
5. **A Sonnet-versus-Opus caller A/B** over a fifteen-minute segment, once
   naming is stable.

Leave the gate, the grader and the wire alone.

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

---

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
- **The openers list is gone and so is the position-zero trim** (`40e8040`).
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
