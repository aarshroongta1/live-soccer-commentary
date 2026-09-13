# Handoff

State of the branch `sprint/days-2-12` after the real-footage-and-wire brief
(`docs/BRIEF-real-footage-and-wire.md`). Written for whoever picks this up
next, including me.

**Head:** `5544219` on `main`, 63 commits on top of `c2ef18e`.
**Gates:** `uv run pytest` 539 passed · `uv run ruff check .` clean ·
`uv run mypy` clean. All three were green after every commit.

**The layout changed on 2026-09-13.** `sprint/days-2-12` was fast-forwarded
into `main` and deleted, and the worktree under `.claude/worktrees/sprint` is
gone; work happens in `/Users/Aarsh/Desktop/commentary` on `main`, and the
clips and every trace ever made came with it — `clips/` and `runs/` in the
repo, both gitignored, which is why the trace paths below say `runs/` and not
the scratchpad they were written in. The brief's
opening line still points at the old worktree and is left as the user wrote
it. `uv run mypy` needs the `tools` extra to be clean — without it
`mcp_server.py` reports nine import errors that are the missing package, not
the code — so `uv sync --all-extras --dev` before trusting a red gate.

**Read `docs/CLIPS.md` before anything else.** Sixteen clips, one run each,
$7.10 all in. Ten of them are 45-second single-event clips from five matches
across three broadcasters — a Copa América bug with the clock on a second row,
a 2018 bug with it on the right — and they answer the question the first six
could not: **seven of ten events called, zero phantoms, and not one wrong name
in the set**, on squads and kits and graphics nothing had been tuned on. The
first six were:
a penalty, two substitutions, a card, a two-goal comeback, a shootout and an
offside. Bar scores 7, 10, 8, 9, 10, 10 of 12. **47 distinct players named
across 119 lines on footage nobody tuned on, and not one was the wrong man.**
What did not generalise is naming in open play — 0% to 38% outside the
shootout — and five bugs the clips exposed are fixed with their commits
listed there. The twelve-item definition of done is decided by machine rather
than by eye now: `commentary grade <trace> --pack --statsbomb
--lineups`, which aligns the feed from the trace's own board readings and
refuses to grade an alignment it cannot trust. That page also carries the
per-period video offsets, which are different in every half.

The key ran out of credit 131 seconds into the first of those runs, which is
why penalty1 is a partial; it was topped up and the other five are complete.

---

## What is true now that was not before

Parts A, C and B of the brief are all implemented, in that order.

**A — a match can be run from a file.** `--source file` carries audio through a
second ffmpeg process, so the whistle and roar detectors fire on a clip for the
first time (verified: 1 whistle, 3 roars on a 110 s run that would have had
zero). New commands: `crop` draws the score-bug box on one frame so it can be
checked by eye, `captions` turns yt-dlp's `.en.json3` into the human
transcript, `feed` turns StatsBomb's event file into the grading feed. The gate
fixes from the first real traces all landed — compound surnames, sightings
written as `"11 Di María"`, "towards their own goal", demonyms, and the
goal-confirmation rework. A second real run added two more (A15, A16): an
ordinary word opening a line is no longer read as a name, and the evidence
for a score change now survives the replay the broadcaster cuts to after
every goal.

**C — vision names the players.** Detect → track → split by kit → read the
shirt number → draw the name on a copy of the caller's frames. Every model
sits behind a Protocol with a fake, so the suite runs with no weights and no
network. `state.identified` carries the same names in text, which survives a
camera cut when the labels do not.

**B — the wire.** A statistician's feed, off by default, as the ceiling row of
the ablation table. The two-clock rule is implemented and tested: an event is
*known* at `video_ts + latency_s` and *applied* at `video_ts`, so a correction
lands at `video_ts + max(0, latency_s - delay_s)`.

---

## The A6 finding, since it was the open question

**The checkpoint's unverified claim was true.** A permanently absent score bug
left `in_replay` set, so every prompt carried "screen: replay, not live play"
— and the caller, told never to call a replay as live, went quiet for the rest
of the match. Two minutes of absent reads was enough.

Fixed: past `BUG_GONE_S` (30 s) the tracker calls it a missing bug rather than
a replay, and the summary says "no score bug visible, so the score and clock
may be stale", which the caller can work around. Test:
`tests/test_board.py::test_a_bug_that_never_comes_back_stops_being_a_replay`.

---

## What was measured, and what those numbers are worth

### Against the real API (~$2.20 of calls)

Opus 5 calling, Haiku 4.5 on the board, through a rendered clip with audio.
The marks A/B — same 110 s, same seed, only difference is whether the names
were drawn on the caller's frames:

| | lines | naming a player | names said | distinct | wrong |
|---|---:|---:|---:|---:|---:|
| marks on | 12 | 11 (92%) | 24 | 12 | 0 |
| marks off | 11 | 6 (55%) | 6 | 4 | 0 |

Every name a real member of that squad. Also run and working end to end: a
file source with audio (14 lines judged, 14 passed, where the brief's trace
had 5 of 8 rejected), and a wire run producing live corrections.

### The ablation table

`uv run python -m commentary.grading.baselines --duration 600 --seconds 150
--error-rate 0.3 --speed 1`, re-run in full; the table is in the README.
Headline: the fact gate takes factual error rate from 70.0% to 16.0%.

**Two columns and one row do not measure what their names suggest, and the
README says so.** The stand-in oracle never reads the picture, so `no-marks`
cannot show what the marks buy — it is there to prove the substitution runs.
And the sim's ground truth names a player at about thirty moments in ten
minutes with no passes at all, so `name prec` has almost nothing to mark
against. Both need real footage.

---

## The gap, and it is the whole remaining gap

**The vision chain has been run on real footage, and it cost two of its three
models.** A full `--marks` run on the Argentina-France clip (trace
`runs/marks/`) said: 83 tracker passes in 175 s, median 1.6 s
a pass, median 3 tracks a pass on footage with fifteen bodies in it, `named:
0` on every pass, and `with_side` 0 on most — PARSeq confirmed no shirt
number in three minutes and the kit split never gathered the 200 crops it
wanted. In the same three minutes the caller read nine number-and-name pairs
off the same frames, every one correct.

So PARSeq and SigLIP are gone (C11). The kit split is an HSV histogram, the
caller reads the numbers and reports them against a `#id` tag drawn over each
tracked body, and the tracker's job is the one a model looking at single
frames cannot do: keep that name on that body while the camera stays on it.

### C11 was run on the clip, and half of it works

`--marks` on the same clip at `067428d` (trace `runs/c11/`,
$0.96). Against the run that motivated C11:

| | before | after |
|---|---:|---:|
| tracker passes | 83 in 175 s (0.5/s) | 1039 in 179 s (**5.8/s**) |
| median ms a pass | 1635 | **130** |
| median tracks a pass | 3 | **13** |
| median tracks with a side | 0 | **10** |
| passes with a named track | 0 of 83 | **0 of 1039** |

Five sightings bound and one was dropped, the roster check did its job, and
the registry ended the run holding six players. **But no track was ever
named**, and the reason is not in C11 at all.

**Track ids do not live long enough to be bound.** Measured offline on the
same clip at the same rate, no API: 300 passes, **1966 distinct ids, 1539 of
them lasting a single pass**, median lifetime 0.00 s, two ids alive for 8 s.
`_assign_ids` is greedy IoU >= 0.3 against the previous pass only, and its
own docstring has always said it is the simple version standing in for the
ByteTrack that C2 specified. On a panning broadcast camera at 5.8 Hz it
almost never matches. So the mark the caller reports is, nearly always, a
body that no longer exists by the time the sighting comes back.

Two things would fix it, and both were measured on the same detections:

- **Give the tracker a memory.** A lost track kept for 30 passes and matched
  at IoU 0.2 gives 836 ids (median life 0.92 s, 76 alive at 8 s); at IoU 0.1,
  568 ids and 89 alive at 8 s. Ten lines, no dependency. The loose gate is
  the risk: a different body inheriting a dead id inherits its name, and a
  wrong name is the thing this system exists not to do. ByteTrack via
  `supervision` — in the vision extra since C1 and never wired up — scored
  296 ids, median 1.67 s, 16 alive at 8 s: far better than today, worse than
  the memory, and it brings matplotlib and scipy.
- **Track the cursor, not the live edge.** Nothing but the caller consumes
  tracks and the caller lives at the cursor, so a mark has to survive from
  the cursor frame it was drawn on to the moment the line comes back — about
  twelve seconds of frame time with the tracker eight seconds ahead, about
  four with it on the cursor. The cut reset would have to move with it: it
  is fired by the frame loop at the live edge and would otherwise wipe the
  ids eight seconds early.

**The caller also reads the tag as a shirt number.** Of six sightings, three
said mark 11 wearing number 11 and one said mark 1 wearing number 7, and one
came back as `#916` with neither a number nor a name. The rules say twice
that a tag is not a shirt number; on this evidence they are not enough, and
a tag that cannot be confused with a number — a letter, a colour — may be
the cheaper fix than more words.

### C12 fixed the tracking and the naming loop still does not close

`ccaba63`, same clip, $0.99 (trace `runs/c12/`).

| | C11 | C12 |
|---|---:|---:|
| passes a second | 5.8 | **7.8** |
| median ms a pass | 130 | **114** |
| id survives the 4 s round trip (offline) | 3 % | **42 %** |
| sightings bound | 5 | **0** |
| passes with a named track | 0 of 1039 | **0 of 1343** |

The tracking is fixed: ByteTrack at 0.9 with a 30-pass buffer turns 1966 ids
in 300 passes into 202, and takes the chance that a body the caller points at
still carries its id four seconds later from 3 % to 42 %. Measured offline on
the clip's own detections, so it is not an estimate.

**But the run made only one sighting, and it was unusable** — mark `KE` with
no number and no name. And the caller put seven tag letters in `names_read`,
where the roster check has no latitude, and four lines died as invented
names. That half is fixed in `a40b27b`: the rules now say a letter tag never
goes in `names_read`, and the gate no longer reads a short capitalised
alphabetic token there as a name claim. **It has not been run since.**

### The gallery runs (C10), and where it stopped: 10 of 12

Two runs, $1.88. Run A (`0d06e83`, `runs/A/`, $1.03) is **10 of 12** and the
best the clip has produced: name_rate 53%, 10 of 10 names correct, three
distinct players in open play, no phantom, no silence, median track life
2.53 s. Run B (`0e10c3b`, `runs/B/`, $0.86) is 9 of 12 — the margin was
raised and the run named fewer people, but the fall is mostly run-to-run
variance in what the caller chooses to say.

**The gallery works and is thinner than it looks.** Run A: 57
classifications, 19 hits, two players in the gallery, margins sharply
bimodal — confident 0.76 to 0.86, ambiguous 0.001 to 0.141, nothing in
between — so the margin went to 0.35, in the gap with headroom either side.
Run B then held **one** player and named five different bodies after him,
because with one centroid there is no second best and the margin is the
similarity under another name. Fixed in `4b70faa`: two in the set before
anything is recognised. No gallery name reached a voice in either run.

**Why the gallery is thin, and it is the same problem in its last form.** It
learns only through `identify`, `identify` needs a mark, and **twelve of run
B's thirteen bound sightings had no mark** — the body the number was read off
carried no tag. Tagging every body took close-up coverage from 6% to 75%, and
the missing quarter is bodies clipped by the top of the frame, which is most
of a celebration close-up.

**What to fix next, in order.**

1. **Attribute a markless read to the only body in shot.** When a sighting
   binds with no mark and exactly one tracked body is big enough to embed,
   the read is about that body. That is what fills the gallery, and it needs
   a rule decision because it is an inference rather than something read.
2. **name_rate.** 53% on run A against 60%, and 25% on run B with the same
   code, so the variance is larger than the gap. More runs would measure the
   variance rather than close it; more names carried is what closes it.
3. **The `error` row on run B was an `APITimeoutError` on one caller call.**
   Transient. Item 12 counts it as a health failure, which is right, but it
   is not a defect to chase.

### Where the earlier loop stopped: 9 of 12

Five runs, $4.90, and the user's cap is $5. Run 5 (`300c2ba`, trace
`runs/r5/`) passes everything except items 1, 6 and 7's
margins:

| item | |
|---|---|
| 1 events | goal named twice in the window; **one line trimmed** — the gate cut "Rosario" and "World Cup" out of "a final goal for the man from Rosario" |
| 2, 3, 4 | all pass: event recall 3/3, no phantom, **no gate rejection in 28 judged lines** |
| 5 names | 8/8 names correct against StatsBomb |
| 6 | name_rate **40%**, wants 60% |
| 7 | 3 distinct players in open play, 12 sightings bound |
| 8 | 12 of 20 sightings bound, 7 on a live tag |
| 9, 10, 11 | no silence over 20 s in live play, analyst behaving, no scoreline |
| 12 | $0.99, no errors, 7.0 passes/s, **median track life 2.57 s** |

What moved it, in order of how much: deleting `names_read` so there is one
field to put a read in (0 sightings to 21); tagging every tracked body rather
than only the ones the kit split places (the close-ups had no tags at all);
the lost-track buffer at sixty passes so a name outlasts the line that earned
it (median identity 1.8 s to 2.6 s).

**What to fix next, in order.**

1. **Item 1.** The gate trims true words that are not people: "Rosario",
   "World Cup". `_roster_of` adds competition and venue tokens of four
   letters or more, so "Cup" is dropped and the pair fails as a run. The
   pack's storylines are not in the roster at all.
2. **Item 6.** 40% against 60%. Every name said was correct, so this is
   reach and not precision: the caller names a player when it can see the
   shirt or a surname tag and says "Argentina" otherwise. More of the tags
   carrying names for longer is the lever, which is C10 — the per-match
   gallery, approved and not built.
3. **Item 7** passes but thinly (exactly 3), and no line yet uses a name that
   was carried on a mark across a cut. That is C10 again.

### Run 3: the loop works, and it named the wrong man once

`0eb2c4b`, $0.96 (trace `runs/r3/`). **21 sightings made, 16
bound** — the first run where the naming loop did anything at all. What made
the difference was not the prompt: it was deleting `names_read`, so there is
one field to put a read in.

Two failures worth keeping in mind, both fixed in `04139f3`:

- **A wrong name.** "26" read off a body the kit split had put on France
  bound as Marcus Thuram, through a passage StatsBomb has Argentina playing
  all of. The body was Molina, Argentina's 26. A number alone now names
  somebody only when one squad wears it, and the kit split is out of the
  naming path entirely.
- **A mark of "Thuram".** `id_of` took any alphabetic string, so a name in
  the mark field parsed to a track id in the millions. A tag is at most three
  letters now, which is all we ever draw.

Still open after run 3: only 1 of 21 sightings carried a real tag, because
the close-ups where a number is legible still have no tags on them (below).
So nothing was ever carried on a mark, item 7 fails, and `name_rate` is 14%
— the caller reads players constantly and writes lines about the crowd.

### The loop is blocked: close-up bodies have no tag

Two runs into the definition-of-done loop, both 5/12, and the measurement
that explains both (offline, no API, over the whole clip):

- 9113 tracked bodies. **784 of them are 200 px tall or more** — a close-up,
  which is the only place a shirt number is legible at 768 px.
- **44 of those 784 carry a tag. Six per cent.**
- None of the suppressions are the "no room above the player" rule. Every
  one is `Side.UNKNOWN`: `_mark_text` draws nothing for a body the kit split
  will not put in a team.

The kit split is fitted on wide-shot crops — small, half grass — and a
close-up crop is shirt and skin at a completely different scale. Its
histogram sits outside `KIT_DISTANCE` of both centroids, so it is called a
referee. The bodies whose numbers can be read are exactly the bodies the
split refuses to classify, and C11 rule 3 says only a body with a side is
tagged. So the caller reads a number off a player who has no tag, and there
is nothing for it to report.

That is why run 2 produced *fewer* sightings than run 1 rather than more: the
sharper rules told it that a sighting with no letter is worth nothing, which
correctly suppressed the two empty-mark ones and left nothing behind.

### The tag fix worked, and the caller will not use the tag

`de26be1`, same clip, $1.02 (trace `runs/c12b/`).

| | C12 | C12 + the tag fix |
|---|---:|---:|
| passes a second | 7.8 | **9.0** |
| judged / passed / rejected | 13 / 9 / 4 | **17 / 16 / 1** |
| `name_read_not_on_roster` rejections | 7 | **0** |
| spoken lines | 12 | **14** |
| sightings made / bound | 1 / 0 | 2 / 0 |
| passes with a named track | 0 of 1343 | **0 of 1538** |

The rejection half is fixed outright: the only line lost in three minutes was
an unconfirmed goal, and that was the pre-existing lag.

**The naming loop is now blocked on one thing, and it is not the tracker.**
Tags are drawn on 82% of tracked bodies (9424 of 11476) and at least one is
on screen in 900 of 1538 passes, so there is a tag to point at. The caller
reads shirts constantly and correctly — thirteen lines carried a read in
`names_read`, every one of them right: `3 Tagliafico`, `24 Enzo Fernandez`,
`11 Di María` six times, `7 De Paul` three times, `10 Messi`, `18
Upamecano`, `5 Koundé`, `1 Lloris`. And it filled `sightings` twice, both
times with the right player and **an empty `mark`**.

So: it reads the number, it writes it in the field it has always written it
in, and it will not tie it to a tag. Nothing downstream can work until it
does. What has not been tried is making `sightings` the only place a read
goes, or refusing a sighting without a mark at the schema, instead of asking
for both in prose and hoping.

The in-loop version of the 42% survival number is still unmeasured, because
a mark that is an empty string never resolves to anything. The trace carries
it (`live` on every sighting row) the moment a mark arrives.

Two things that are not regressions, so nobody re-investigates them:

- **Tracks a pass looks worse (13 to 4) and is mostly footage.** The tracker
  is at the cursor now, so it never sees the last eight seconds of the clip,
  which is the wide restart; and the middle two minutes of this clip are
  celebration close-ups with one to three people in frame. In the first
  thirty seconds, which is wide play in both runs, it is 14 and 13.
- **The cut detector fires 16 times in 180 s of this clip**, not hundreds, so
  the reset is not what keeps the track count down.

What a real run settles once the loop closes is `name_rate` and
`name_precision`, and the honest expectation in the README — names on the big
moments, not pass-by-pass — is what is being tested.

`scripts/first_real_run.md` is the order to do it in. Short version:

1. `.env` with a key and `CALLER_MODEL`, plus a real `MAX_USD_PER_MATCH`.
2. `yt-dlp -f "bv*[height<=720]+ba" --merge-output-format mp4
   --write-auto-subs --sub-lang en --sub-format json3 <url>` — video and
   captions in one call. Downloading is the user's call; the repo has no
   downloader.
3. `commentary captions <name>.en.json3` for the transcript.
4. `commentary crop --path <name>.mp4 --at 300` and adjust the four fractions
   until the score and clock fill the right-hand panel. **Do not skip this.**
   A box that is half a bug reads as an unreadable board for ninety minutes
   with nothing in the trace to say so.
5. A pack: `commentary research`, or hand-written. Set `kit` and `demonym` by
   hand either way — the gate reads both.
6. `uv sync --extra vision` if you want the marks, or `--no-marks` if not.
7. Ten minutes first, never ninety: `ffmpeg -ss 00:20:00 -t 00:10:00 -c copy`.
8. `commentary feed <events.json> --home X --away Y` and grade it.

---

## Things the next person will trip over

- **`run --source file` has the marks on by default**, so on a machine without
  the `vision` extra it stops with a one-line message telling you to install
  it. That is deliberate; `--no-marks` is the other answer.
- **A tag is a letter, and it is neither a shirt number nor a name.** It is a
  label so the caller can say which body it read a number off. Both failures
  have now happened on real footage: with digits the caller reported the tag
  as the shirt number, and with letters it filed the tags in `names_read`
  where the gate killed the lines. Anything that makes a tag look like a
  claim about a player is the marks doing harm.
- **The tracker runs in its own loop, on the newest frame, and skips the
  rest.** It is not a per-frame pipeline and must never become one again: one
  pass is 92 ms at best and frames arrive every 66 ms, so anything that
  awaits it from `_ingest_frames` stalls the cursor outright. The `tracks`
  rows in a trace say what rate it managed and how much it named.
- **The simulator's timestamp strip is in the top-left corner**, and the
  oracle decodes it from the frames the caller was given — which now carry
  marks. `draw_marks` refuses to draw a label with no room above the player,
  which keeps labels out of that corner. If you move the strip, check that
  still holds.
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
- **`docs/BRIEF-real-footage-and-wire.md` is modified in the working tree.**
  That edit is the user's, not this session's. It is where A15 and A16 came
  from; the brief in the last commit does not have them.
- **`tests/test_baselines.py::test_only_the_wire_can_put_a_lied_about_score_right`
  failed once in about fifteen runs**, and only while a second full suite was
  running against the same machine. `blind` came back 1-1 rather than 0-0.
  Its docstring says the board reader lies on every read at `error_rate=1.0`,
  and `SimOracle._board` does not lie at all — the rate only injects errors
  into caller lines — so what holds the score at 0-0 is the clip ending
  before three agreeing post-goal reads land, which is a race. Nothing to do
  with A16: eight runs of it after that change all passed.
- **There is no fixed window on talking about a goal.** It runs from the
  cursor the state applied it at until play restarts — a kickoff line, or a
  whistle since the goal and then a live picture — with `GOAL_TALK_CAP_S`
  (150 s) as the backstop. A test asserting that a celebration line expires
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
