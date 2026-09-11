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

A play-by-play feed as a second source of state, as an ablation and a
ceiling, off by default. `board.py`'s docstring says a commentator does not
receive the score over a wire. This is the wire. Module: `commentary/wire.py`.

### B1. Types (`schemas.py`)

```python
class WireEvent(BaseModel):
    event: Event
    side: Side = Side.UNKNOWN
    player: str | None = None
    detail: str = ""            # "yellow", "red", "own goal"
    home_score: int = 0
    away_score: int = 0
    video_ts: float | None = None   # known for sim truth
    clock_s: float | None = None    # match clock, for a saved feed
    period: int = 1

class Incident(BaseModel):      # goals, cards, subs only
    event: Event
    side: Side
    player: str | None
    video_ts: float
    source: str                 # "board" | "wire"
```

`MatchState.incidents: list[Incident]`. `MatchStateTracker.summary()` prints
them after the scoreline, one line each kind: `goals: Ashcombe 12' Owen
Kasper` / `cards: Verity 34' yellow Nils Marek` / `subs: Ashcombe 61' Teo
Felix`. Board-confirmed goals (`BoardChange.is_goal`, `scoring_side`) are
recorded as `source="board"` incidents with no player. That is the
vision-only state; the wire fills in names and adds what vision missed.

### B2. `Wire` and `ReplayWire`

```python
class Wire(Protocol):
    latency_s: float
    def due(self, live_ts: float) -> list[WireEvent]: ...
```

`due` returns, once each, events with `video_ts + latency_s <= live_ts`.
`ReplayWire(events, latency_s)` is the one implementation.
`ReplayWire.from_truth(events: list[GroundTruthEvent], latency_s)`. A live
polling wire is out of scope; one sentence in the docstring saying it would
implement the same `due`.

### B3. `WireSync`

Holds the wire, a queue of known events, and the latest clock fix.

- `observe_clock(clock_s, period, video_ts)`: called from the board loop on
  every readable read. Stores `offset[period] = video_ts - clock_s` from the
  latest read. No fitting. Resolves `clock_s`-only events to `video_ts`.
- `poll(live_ts)`: pulls `wire.due(live_ts)` into the queue.
- `apply_due(cursor_ts, tracker) -> list[Correction]`: applies queued events
  with resolved `video_ts <= cursor_ts`, in time order, via
  `tracker.apply_wire(event, ts)`.

The two-clock rule: an event is *known* when the live edge passes
`video_ts + latency_s`; it is *applied* when the cursor passes `video_ts`.
So a correction lands at cursor `>= video_ts + max(0, latency_s -
delay_s)`. With `latency_s <= delay_s` the buffer absorbs the feed's latency
and corrections are on time; at `delay_s = 0` they are `latency_s` late.
Docstring and a direct test.

`Correction` (dataclass): `video_ts`, `what` (one short string, e.g.
`"score 1-0 -> 1-1"`, `"card Verity Nils Marek"`), `event`. Emitted only
when the wire changed state.

### B4. `MatchStateTracker.apply_wire(event, ts) -> str | None`

Sets the score from the event (the only path other than the board that may
move it; docstring says it is the ablation path, off by default). Appends an
`Incident(source="wire")` for goals, cards, subs; a board goal already
recorded for the same side within 15 s gets the player name instead of a
second incident. For a sub with a number in the pack:
`registry.believe(number, name, ts, side=side)`. Returns the `what` string
if anything changed, else `None`.

### B5. Runtime

- `Runtime.wire: Wire | None = None`; `__post_init__` builds
  `self.wire_sync = WireSync(wire)` when set.
- `_read_board`: after a readable read with a clock,
  `wire_sync.observe_clock(...)`, then `wire_sync.poll(frame.ts)`.
- `_ingest_frames`: beside `_apply_due_board_changes()`, `_apply_due_wire()`
  publishes each `Correction` on new `Topic.CORRECTION` and then a
  `Topic.STATE`.
- Board goals become incidents in `_apply_due_board_changes`.
- Gate: `FactGate.judge(..., wire_confirmed: bool = False)`; a goal passes
  when `board_changed or lookahead_celebration or wire_confirmed`.
  `_wire_confirms_goal(cursor)`: any known wire GOAL with `cursor -
  GOAL_GRAPHIC_LAG_S <= video_ts <= cursor + 2.0`. Known means already
  pulled by `poll`, never peeked. `OpenGate._judge` gets the same kwarg.
- The scoreline check reads state, so a wire-corrected score is what a
  stated score is checked against. Nothing to change; a comment.

### B6. Ablation

- `Variant.wire_latency_s: float | None = None`.
- `wire(base)`: name `wire-10s`, `DEFAULT_WIRE_LATENCY_S = 10.0`, note "full
  system plus a play-by-play feed at 10 s modelled latency: the ceiling".
  Sixth row of `standard_variants`.
- `run_variant`: when set, `wire=ReplayWire.from_truth(sim.ground_truth,
  latency_s)` into `BaselineRuntime`. The runtime never sees the sim.
- Grader: `metrics.corrections(run) -> int` from `correction` rows;
  `Scorecard.corrections`; one line in `report.detail`.

### B7. CLI

`run --wire feed.json --wire-latency 10` for `--source file|screen`:
`grading.feed.load_feed` lazily in `cmd_run`, `FeedEvent` to `WireEvent` on
match clock, `ReplayWire` into the runtime. For `--source sim`,
`--wire-latency N` builds it from `sim.ground_truth`. Print corrections in
the post-run summary.

### B8. Tests

- `tests/test_wire.py`: `due` releases once, in order; the two-clock rule at
  `latency_s` below, equal to and above `delay_s` (assert the cursor at
  which the correction lands); clock-only events resolve after a clock fix
  and not before; no correction when state already agrees; a sub updates
  the registry.
- `tests/test_state.py`: `apply_wire` moves the score; a wire goal names a
  board goal instead of duplicating it; summary prints incidents and stays
  short.
- `tests/test_runtime.py`: with `wire=ReplayWire.from_truth(truth, 2.0)` and
  `delay_s=4.0`, every goal in the watched window is an incident by the
  end; `test_the_score_only_ever_comes_from_the_board` still passes for the
  default runtime.
- `tests/test_baselines.py`: `wire-10s` runs, writes `correction` rows, and
  at `error_rate=1.0` its `error_count` is `<=` `full`'s.
- `tests/test_gate.py`: `wire_confirmed=True` passes a goal claim alone.

### B9. Docs

README: a short "The wire" subsection under Ablations: the ceiling row, the
two-clock rule, the default runtime never loads one. Update the test count.
PLAN.md: one paragraph under section 6.

---

## Report back

Commits; the A6 finding and the test that shows it; the ablation table re-run
with the `wire-10s` row (same command as the README table); anything done
differently from this brief and why.
