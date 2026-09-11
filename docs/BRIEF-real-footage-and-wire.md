# Brief: real footage readiness, then the wire

For an implementation session. Read all of it first. Work in this worktree
(`sprint/days-2-12`, at `.claude/worktrees/sprint`), not the main checkout.
Small commits. No `Co-Authored-By` or session links in commits or PRs.
`uv run pytest`, `uv run ruff check .`, `uv run mypy` green after every
commit. The shell here refuses paths containing the token `e-v-a-l`; that is
why the grading package is `grading/`. Do not create anything with it.

**Keep it simple.** One way to do each thing. No fallbacks, no presets, no
flags nobody asked for, no "in case". If a feature in this brief can be done
in fewer moving parts than described, do that. Match the surrounding code's
style: short modules, docstrings that say why, no abstraction for one caller.

Part A unblocks the first run on real YouTube full-match footage, which is
imminent. Do it first. Part B is decided; build it after A is committed.

Rules that must survive:

1. `src/commentary/grading/` is imported by nothing under `src/commentary/`
   except `grading/*` and, lazily inside a command handler, `__main__.py`.
2. The default runtime uses no outside data. Every new source is opt-in,
   off by default, visible in config and trace.
3. Ablation variants are substitutions of the same type, never branches in
   the match loop (see `grading/baselines.py` docstring).

---

## Part A: run a full match from a file

### A1. `FileCapture` carries audio

`--source file` yields frames only. The runtime only starts the audio loop
when `hasattr(source, "audio")`, so on a file there is no whistle, no roar,
and `_celebration_ahead` never fires. Give `FileCapture` an `audio()` async
iterator yielding `AudioChunk` (type and sample rate: `capture/audio.py`,
`capture/buffer.py`). A second `ffmpeg` process, `-vn -ac 1 -ar <rate> -f
s16le`, paced the same way as the frames. Timestamps on the video clock
from 0 at the start of the file. Test with a short file the test writes
(sim frames muxed with a synthesised wav).

### A2. Crop flag and preview

`BoardConfig.crop` is one box. Add `run --crop x0,y0,x1,y1` (fractions) and
a `crop` subcommand: `python -m commentary crop --path m.mp4 --at 300 --crop
0,0,0.42,0.16 --out /tmp/bug.png` writes the frame at that second with the
box drawn and the cropped bug beside it, so the box is checked by eye before
a run spends money. Reuse `crop_score_bug`.

### A3. Captions become the transcript

`grading/transcripts.py` has `Transcript`/`Segment`/`load`/`save`. The
YouTube captions are the human-commentator transcript: ground truth for
timing and naturalness, never for facts, so auto-caption name errors do not
matter for what they are used for. Say so in the docstring.

The captions come from yt-dlp in the same call that pulls the video
(`--write-auto-subs --sub-lang en --sub-format json3`), which leaves a
`<name>.en.json3` file beside the mp4. No new dependency. Add
`grading/captions.py` with one function, `load_json3(path) -> Transcript`:
one `Segment` per JSON3 event (`tStartMs`, `dDurationMs`, text joined from
`segs[].utf8`), skip events whose text is empty or only a newline.
`Transcript.model = "youtube-auto"`. CLI: `python -m commentary captions
<file.json3> --out transcript.json`, writes the `save` shape. No network
anywhere. Test with a short hand-written JSON3 fixture under
`tests/fixtures/`.

Docstring notes: captions are on the video clock, so no alignment (the feed
needs it, see `grading/feed.align`). Caption gaps are ambiguous between "the
commentator went quiet" and "ASR gave up in crowd noise"; state that as a
caveat where silence ratio is compared.

### A4. StatsBomb events to the feed shape

One converter, `grading/statsbomb.py`, `convert(rows, home, away) -> dict`
producing the project shape `load_feed` already reads (`{"events": [{"clock",
"type", "team", "player", "home_score", "away_score"}]}`). StatsBomb rows:
`type.name`, `minute`, `second`, `period`, `team.name`, `player.name`. Goals:
`type.name == "Shot"` and `shot.outcome.name == "Goal"`; own goals `type.name
== "Own Goal Against"` count for the other side. Cards:
`foul_committed.card.name` or `bad_behaviour.card.name`. Subs: `type.name ==
"Substitution"` with `substitution.replacement.name` as the player. Clock text
`"{minute}:{second:02d}"`. Scores run forward. Drop everything else. CLI:
`python -m commentary feed <events.json> --home X --away Y --out feed.json`.
Small hand-written fixture. `parse_feed` is not touched.

### A5. First-run checklist

`scripts/first_real_run.md`, short: `.env` with `ANTHROPIC_API_KEY` and
`CALLER_MODEL=claude-haiku-4-5`; `brew install yt-dlp` and one command for
video plus captions: `yt-dlp -f "bv*[height<=720]+ba" --merge-output-format
mp4 --write-auto-subs --sub-lang en --sub-format json3 <url>`; `captions`
on the `.en.json3`; `crop`;
a pack via `research` or a hand-written JSON; a ten-minute cut first
(`ffmpeg -ss 00:20:00 -t 00:10:00 -c copy`) with `--seconds 600 --delay 8`;
the cost cap. Downloading is the user's call; the repo has no downloader.

### A6. Verify the score-bug claim

Last session's checkpoint says, unverified: a permanently absent bug reads
as a permanent replay and poisons the state summary. Write the test in
`tests/test_board.py`: feed `BoardTracker` a long run of `bug_visible=False`
reads and assert what `in_replay` and `MatchStateTracker.summary()` do. If
true, fix it in the smallest way that makes the summary say "score bug not
visible" rather than "replay" after the bug has been gone longer than any
replay lasts (a constant, say 30 s). Report the finding either way.

### A7. The caller looks for numbers before it settles for a role

`prompts/caller.py` `CALLER_RULES`, the "Names" paragraph. Today it says say
a name only if legible, then gives a *bad* example of naming. Rewrite that
paragraph so the model is told to look before it gives up: on every call,
inspect the shirt number of the ball carrier, the player it is passed to,
the shooter and the keeper; if a number is readable and the kit matches a
team sheet, use that player's surname; if not, use the role and the kit
("the left-back in white"). Keep "a wrong name is the worst thing you can
do" and keep `names_read` as the record of what was legible. Replace the
"Odegaard picks out Havertz" bad example with one that shows the right
thing: a name used *with* the number that justified it in `names_read`.
The gate and the roster check do not change. This is the one thing
worldcupvoice's prompt does better than ours, and it is a prompt edit, not
a design change. Also make sure `_pack_section` prints every player as
`#<number> <surname> (<position>)` on one line per player and the kit
colours for both sides, since that is what the instruction refers to.

---

## Part B: the wire

**Decision (2026-09-11): the picture decides what to say, the feed decides
who.** A vision model cannot name the player on the ball from a wide shot,
and commentary that cannot say "Otamendi to Mac Allister" is not commentary.
So a play-by-play feed becomes a runtime input, opt-in, carrying passes,
carries and shots with player names, and match state tracks who is on the
ball. The fact gate is unchanged: every name still has to be on the roster,
and now the names it sees come from data.

This rewrites rule 2 above: the default runtime (`--wire` absent) still uses
no outside data, and that variant stays in the ablation table as "vision
only". The full system is vision plus wire. README and PLAN say so plainly.

Module: `commentary/wire.py`. One StatsBomb reader, `commentary/statsbomb.py`,
used by both the wire and `grading/statsbomb.py` (so A4 becomes a thin
conversion over it; do not write the parser twice). It reads a saved file;
nothing fetches.

### B1. Types (`schemas.py`)

```python
class Event(StrEnum):   # add
    PASS = "pass"
    CARRY = "carry"

class WireEvent(BaseModel):
    event: Event
    side: Side
    player: str | None = None
    recipient: str | None = None      # passes
    detail: str = ""                  # "yellow", "red", "own goal"
    home_score: int = 0
    away_score: int = 0
    clock_s: float                    # match clock, seconds
    period: int
    duration_s: float = 0.0           # passes and carries
    video_ts: float | None = None     # set for sim truth; resolved otherwise

class Possession(BaseModel):          # who is on the ball, from the wire
    player: str
    side: Side
    since_ts: float
    from_player: str | None = None    # who passed it, when known

class Incident(BaseModel):            # goals, cards, subs
    event: Event
    side: Side
    player: str | None
    video_ts: float
    source: str                       # "board" | "wire"
```

`MatchState` gains `ball: Possession | None` and `incidents: list[Incident]`.

### B2. `commentary/statsbomb.py`

`read(path, home, away) -> list[WireEvent]`. StatsBomb rows: `type.name`,
`minute`, `second`, `period`, `duration`, `team.name`, `player.name`. Keep:
`Pass` (recipient from `pass.recipient.name`; skip if none), `Carry`, `Shot`
(`shot.outcome.name == "Goal"` is a GOAL, else SHOT), `Own Goal Against`
(GOAL for the other side), cards from `foul_committed.card.name` /
`bad_behaviour.card.name` (CARD, detail lower-cased), `Substitution` (player
= `substitution.replacement.name`). Drop everything else. Player names: use
the short form StatsBomb gives in the lineups file when present
(`player_nickname`), so "Lionel Andrés Messi Cuccittini" is "Lionel Messi";
`read` takes the lineups path too. Scores run forward. `clock_s = minute *
60 + second`. Team name to side by the two names passed in. Hand-written
fixture of ~12 rows for the test.

### B3. `Wire` and `ReplayWire`

```python
class Wire(Protocol):
    latency_s: float
    def due(self, live_ts: float) -> list[WireEvent]: ...
```

`due` returns, once each, events whose resolved `video_ts + latency_s <=
live_ts`. `ReplayWire(events, latency_s)`. Events without `video_ts` are
resolved by the sync before `due` can release them (below).
`ReplayWire.from_truth(list[GroundTruthEvent], latency_s)` for the sim.

### B4. `WireSync`

- `observe_clock(clock_s, period, video_ts)`: from the board loop on every
  readable read. Keeps the last 10 `(video_ts - clock_s)` per period and
  uses the median as that period's offset. Ten and median rather than one
  read because passes are two seconds apart and the board reader misreads
  a digit now and then; a single bad read must not rename every touch for
  the next two seconds. Resolves `video_ts` on every event of that period.
- `poll(live_ts)`: pulls `wire.due(live_ts)` into the queue.
- `apply_due(cursor_ts, tracker) -> list[Correction]`: applies queued
  events with `video_ts <= cursor_ts`, in time order.

The two-clock rule: an event is *known* when the live edge passes `video_ts
+ latency_s`; it is *applied* when the cursor passes `video_ts`. So a
correction lands at cursor `>= video_ts + max(0, latency_s - delay_s)`.
With `latency_s <= delay_s` the buffer absorbs the feed's latency and the
carrier is right at the cursor; at `delay_s = 0` it is `latency_s` stale.
This is the argument for the delay, stated in the docstring and tested.

`Correction` (dataclass): `video_ts`, `what` (one string: `"score 1-0 ->
1-1"`, `"card France Rabiot"`), `event`. Emitted only when the wire changed
something other than possession; possession changes are too frequent to
trace individually and are visible in `state` rows.

### B5. `MatchStateTracker.apply_wire(event, ts) -> str | None`

- PASS: `ball = Possession(player=event.player, side, since_ts=ts)` at `ts`;
  the recipient takes over at `ts + duration_s` (queue it as a second apply
  inside the tracker, or have the sync split a pass into two applies; pick
  the simpler). `from_player` is the passer.
- CARRY: `ball` = player if not already.
- SHOT: `ball` = shooter; append SHOT to `last_events`.
- GOAL: set the score (the only path other than the board that may); an
  `Incident(source="wire")`, or the player name onto an existing board goal
  for the same side within 15 s.
- CARD, SUBSTITUTION: `Incident(source="wire")`; a sub with a number in the
  pack calls `registry.believe(number, name, ts, side=side)`.

`summary()` adds, after the scoreline: `on the ball: Mac Allister (ARG),
from Otamendi` when `ball` is set and fresh (under 6 s old at the cursor),
and one line each for goals, cards, subs. Nothing else changes.

### B6. Prompt

`CALLER_RULES` "Names" paragraph (on top of A7): add that MATCH STATE may
carry "on the ball" and "from" from a statistician; those names may be used
as given, and are the only names allowed without a legible number or
graphic. Change "no data feed, no statistician" at the top to say the
statistician may be present and, when they are, speaks only through MATCH
STATE. Byte-stable: the rule text does not depend on whether a wire is set;
the state line simply appears or does not.

### B7. Runtime and gate

- `Runtime.wire: Wire | None = None`; `__post_init__` builds `WireSync`.
- `_read_board`: after a readable read with a clock, `observe_clock`, then
  `poll(frame.ts)`.
- `_ingest_frames`: beside `_apply_due_board_changes()`, `_apply_due_wire()`;
  each `Correction` on new `Topic.CORRECTION`, then a `Topic.STATE`.
- Board goals become `Incident(source="board")` in `_apply_due_board_changes`.
- Gate: `judge(..., wire_confirmed=False)`; a goal passes on `board_changed
  or lookahead_celebration or wire_confirmed`. `_wire_confirms_goal(cursor)`:
  a known wire GOAL with `cursor - GOAL_GRAPHIC_LAG_S <= video_ts <= cursor
  + 2.0`. `OpenGate._judge` gets the kwarg. Names in the line are checked
  against the roster exactly as now; `state.ball` names are roster names by
  construction.

### B8. Ablation and grading

- `Variant.wire_latency_s: float | None = None`. `wire(base)` is `wire-10s`,
  `DEFAULT_WIRE_LATENCY_S = 10.0`, note "vision plus the play-by-play feed
  at 10 s modelled latency". Sixth row of `standard_variants`. For the sim,
  `ReplayWire.from_truth(sim.ground_truth, latency_s)`; the sim's truth has
  no passes, so on the sim this row shows goals, cards and subs only, and
  the note says so.
- `metrics.names(run, wire_events, window_s=3.0) -> Names(lines_with_name,
  names, correct)`: for each spoken line, every capitalised roster surname
  in it counts as a name; it is correct if that player is the passer,
  recipient, carrier or shooter of a wire event within `window_s` of the
  line's `video_ts`. `Scorecard.name_rate` (lines with a name / lines) and
  `Scorecard.name_precision` (correct / names). Two new columns in the main
  table; this is the number that answers "does it name players and is it
  right". `metrics.corrections(run) -> int` in `report.detail`.

### B9. CLI

`run --wire statsbomb-events.json --lineups statsbomb-lineups.json
--wire-latency 10` for `--source file|screen`: `statsbomb.read` in
`cmd_run`, `ReplayWire` into the runtime. Team names come from the pack.
For `--source sim`, `--wire-latency N` alone builds it from the sim truth.
Post-run summary prints corrections and, when a wire was given, the names
table.

### B10. Tests

- `tests/test_statsbomb.py`: the fixture reads to the expected events;
  nickname used; own goal credited to the other side; a pass with no
  recipient is dropped.
- `tests/test_wire.py`: `due` releases once, in order; the two-clock rule at
  `latency_s` below, equal to and above `delay_s`; median offset survives
  one misread; possession moves to the recipient after `duration_s`.
- `tests/test_state.py`: `apply_wire` moves the score; a wire goal names a
  board goal instead of duplicating it; summary shows "on the ball" only
  while fresh.
- `tests/test_runtime.py`: with `wire=ReplayWire.from_truth(truth, 2.0)`
  and `delay_s=4.0`, every goal in the watched window is an incident by
  the end; `test_the_score_only_ever_comes_from_the_board` still passes for
  the default runtime.
- `tests/test_baselines.py`: `wire-10s` runs, writes `correction` rows, and
  at `error_rate=1.0` its `error_count` is `<=` `full`'s.
- `tests/test_gate.py`: `wire_confirmed=True` passes a goal claim alone.
- `tests/test_grading.py`: `names` scores a made-up trace against a
  made-up wire: one right name, one wrong, one line without a name.

### B11. Docs

README: rewrite the opening claim to "the picture decides what to say, the
feed decides who"; the vision-only variant stays in the table as the
ablation. A "The wire" subsection: the two-clock rule, why the delay should
be at least the feed's latency, the name-precision columns. PLAN.md: the
same paragraph under section 6, and section 1's "Nothing else." becomes
"Nothing else, except the play-by-play feed when `--wire` is given."

---

## Report back

Commits; the A6 finding and the test that shows it; the ablation table re-run
with the `wire-10s` row (same command as the README table); anything done
differently from this brief and why.
