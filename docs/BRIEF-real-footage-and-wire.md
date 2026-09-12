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

Part A unblocks the first run on real YouTube full-match footage. Do it
first. Part C is the product path for naming players from the picture; build
it second. Part B is the play-by-play wire, kept only as the off-by-default
ceiling ablation; build it last.

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

### A8. Goal confirmation, from the first real run

First real run (Argentina v France 2022, video 36:00-39:00, Opus 5 caller,
delay 8 s, trace `scratchpad/run/runs/opus/`): the caller called Di María's
goal correctly at cursor 56.5 with the finish in the lookahead, and the
gate rejected it, `unconfirmed_goal`. Then it rejected three more lines
about the same goal over the next 80 s. Measured facts behind that:

- The FIFA score bug moved 1-0 -> 2-0 at video 63.7, about 7 s after the
  ball crossed the line (StatsBomb 35:22 ≈ video 58). `GOAL_GRAPHIC_LAG_S`
  is 5.0. The window `[cursor - 2, cursor + 5]` = `[54.5, 61.5]` missed it.
- `BoardTracker` needs 3 agreeing reads, and reads land every ~3.5 s (2 s
  sleep plus ~1.5 s of Haiku), so the change was *confirmed* at 70.0, 13 s
  after the goal. No delay we would run at covers that.
- `_board_changed_near` only looks near the cursor, so a line about a goal
  that the state already holds (celebration, replay talk, "the scorer")
  is rejected as a phantom goal 10, 20, 80 s later.

Fix, three parts, all in `runtime.py` / `gate.py`:

1. `GOAL_GRAPHIC_LAG_S = 10.0`, docstring citing the measurement above.
2. The gate accepts the board's *pending* change as corroboration. Expose
   `BoardTracker.pending` (exists) and let `_board_changed_near` also match
   a pending score increase whose `first_ts` is in the window. One board
   read agreeing with an independent caller claim is two sources; state
   still moves only on three reads. Say that in the docstring.
3. A goal claim is also allowed when the state's score changed within the
   last 45 s of cursor time (track `_last_goal_ts` when a board goal is
   applied). That is the case for every line *about* a goal after it is
   confirmed. A claim with neither is still `unconfirmed_goal`.

Tests: the three rejected lines from the trace as fixtures (times and
board reads as above) must pass; a goal claim at cursor 200 with the last
board goal at 100 and nothing pending must still fail.

### A9. `names_read` in the sighting form is rejected

Trace, cursor 56.5: `names_read=['11 Di María']`, gate reason
`name_read_not_on_roster: 11 Di María`. The caller writes sightings the
way `state.parse_sighting` reads them ("11 Di María", "Di María (11)");
`FactGate._check_names_read` does not parse them and matches the whole
string against the roster. Fix: run `parse_sighting` first; a parsed
sighting is checked as number-in-squad AND name-on-roster; only an
unparseable token is matched whole. Test with "11 Di María", "Di María
(11)", "11", "Di María", "Zaltimore (11)" (number fine, name not).

### A10. "towards their own goal" is not a goal claim

Trace, cursor 50.0: "...France scrambling back towards their own goal",
`event=build_up`, rejected `unconfirmed_goal`. `_NOT_A_GOAL` strips "towards
the goal" but not "towards their own goal" / "his goal" / "the French
goal". Fix: drop the bare `\bgoal\b` pattern from `_GOAL_CLAIMS`; a goal
claim is `event is Event.GOAL` or one of the strong phrases ("scores",
"scored", "it's in", "back of the net", "finds the net", "makes it N",
"equalis"). `grading/metrics.factual_errors` uses the spoken row's `event`
field the same way instead of the bare word. Test with the trace line.

### A11. Demonyms

Trace, cursor 0.3: "the French lines" trimmed to "the lines" as
`name_not_on_roster: French`. Add `TeamSheet.demonym: str = ""`
("French", "Argentine"); the gate's `_roster_of` adds the folded demonym
to team words; the researcher prompt asks for it; the StatsBomb pack path
sets it by hand in the pack file. Test: "the French lines" passes untrimmed.

### A12. The analyst's clock arithmetic

Trace, 88.5 s: "Deschamps has barely half an hour of the first half left"
at 35:52. Add to the analyst rules: the MATCH STATE clock is the match
clock, a half is 45 minutes, and time remaining is computed from it or not
mentioned. Prompt-only.

### A13. Multi-word surnames fail the roster check

Sonnet run, five lines: `names_read=['Di María']` rejected
`name_read_not_on_roster: Di María`, and the same would happen to De Paul,
Mac Allister, Van Dijk, De Bruyne. `Player.surname` is `rsplit(" ", 1)`,
so Di María's surname is "María"; `_roster_of` adds only tokens of three
letters or more, so "di" is never known; and the similarity ratio of "di
maria" to "angel di maria" is 0.73, under the 0.86 threshold. Fix in one
place: `_matches_roster` also accepts a folded candidate that is a
suffix of a known full name on a word boundary (`known.endswith(" " +
candidate)`). Apply the same rule in `grading/metrics.roster_names` /
`factual_errors` so the grader stops counting "Di" as a name off the
roster (it would today). Tests: "Di María", "De Paul", "Mac Allister"
pass against the Argentina sheet; "María" alone still passes; "Di" alone
does not.

Also from the Sonnet run: the caller said "before the half-hour mark is
even done" at 35:5x. The A12 clock rule goes in the caller prompt too.

### A14. The analyst on a shorter leash

Sonnet run: the analyst spoke 5 of 7 lines, 40 to 50 words each, and
editorialised ("No holder has ever wanted a half-time whistle more than
this one"). `AnalystConfig`: `max_words` 45 -> 30, `min_gap_s` 25 -> 40.
Analyst rules: one observation per line, no rhetorical flourishes, and
nothing about how a manager feels. Prompt and config only.

### Measured on the run (for the README later)

Opus 5 caller: 52 board reads, all confident, score and replay (bug absent 91-131 s)
tracked correctly; 27 caller calls, 0 timeouts on Opus 5 at default
effort, 12 judged, 8 passed; 10 lines spoken in 172 s of cursor (6 caller,
4 analyst); every spoken name correct against StatsBomb (Tagliafico at
34:25-34:30, Di María 35:22); $0.75 for three minutes. Sonnet 5 caller, same clip: 54 board
reads, 43 caller calls, 8 judged, 3 passed, 5 rejected (all the goal and
every `Di María` sighting); 7 spoken (2 caller, 5 analyst); $0.48. Model
choice is open until the gate fixes land and both are re-run.

---

## Part C: names from the picture

**Decision (2026-09-11): vision names the players.** The research on this
(SoccerNet jersey-number and game-state work, Roboflow's sports pipelines,
Set-of-Mark prompting) converges on one recipe, and it is not "ask the
vision model who that is". It is: track every body, split by kit, read the
shirt number once when it is legible and carry it on the track, resolve
(team, number) to a name from the lineup, and **draw the name on the frame**
so the language model reads it instead of guessing. Claude stays the
writer; open models do the "is that a 7 or a 1" layer, locally, free.

Expectation, stated in the README when it ships: names on the big moments
(shooter, scorer, the fouled and the fouler, the booked player, the sub),
names through a sustained shot once a number has been read, role and kit
after a cut until the next legible number. Not pass-by-pass naming from a
wide shot. `name_rate` and `name_precision` (B8) against StatsBomb say how
far it gets.

### C1. Dependencies

A `vision` optional group in `pyproject.toml`: `torch`, `rfdetr`
(Apache 2.0; not Ultralytics YOLO, which is AGPL), `supervision` (ByteTrack
lives there), `open_clip_torch` or `transformers` for SigLIP (pick the one
with fewer transitive dependencies), and PARSeq via `torch.hub.load(
"baudm/parseq", "parseq", pretrained=True)`. Model weights download on
first use to the default cache; never in tests. Every model sits behind a
small Protocol (`Detector`, `Embedder`, `NumberReader`) with a fake in
tests, so the suite runs with no weights and no network. That is the one
place this brief allows an abstraction for one caller: the alternative is
downloading a gigabyte in CI.

### C2. `perception/players.py`: tracks

```python
@dataclass
class Track:
    id: int
    side: Side            # from kit clustering; UNKNOWN for referees/others
    box: tuple[int, int, int, int]
    number: int | None    # confirmed by voting, else None
    name: str | None      # registry lookup of (side, number)

class PlayerTracker:
    def update(self, frame: Frame) -> list[Track]: ...
    def reset(self) -> None: ...   # on a scene cut
```

Per frame at the live edge, on a copy downscaled to 640 wide: RF-DETR
(COCO "person" is enough), ByteTrack for IDs, SigLIP embedding of each crop
and k-means with k=2 fitted on the first ~200 crops of the match and
refitted every 5 minutes; a crop farther than a threshold from both
centroids is `Side.UNKNOWN` (referees, staff). Which cluster is home: the
pack's `kit` strings name a colour; compare the mean crop colour of each
cluster to the first colour word in each kit string. Say in the docstring
that this is the weakest link and log the assignment once per fit.

Run detection on every other frame (7.5 Hz is plenty for IDs) in a thread
executor so the asyncio loops are not blocked; tracks are stored per frame
ts in a bounded dict the caller reads from at cursor time. `reset()` on
every cut the `CutDetector` finds; the tracker starts fresh IDs and the
registry keeps the names.

### C3. Shirt numbers, read once and voted

For each track, when its box is at least 110 px tall at 720p: crop the
upper half (torso), run PARSeq, keep results that are 1 or 2 digits, 1-99,
confidence >= 0.8. Keep the last 5 reads per track; a number is confirmed
when 2 reads agree. No legibility classifier; PARSeq's confidence gate is
the simple version and the voting absorbs the rest. On confirmation:
`registry.believe(number, name, ts, side=side)` where `name` is the pack's
player with that number on that side, and the `EntityRegistry` already
gives it decay and sub handling. If the pack has no such number, the track
stays numbered but unnamed and the caller sees "ARG #14".

### C4. Marks on the caller's frames

In `prompts/caller.caller_blocks`, before `encode_frame`, draw on a *copy*
of each cursor and lookahead frame a small label above each track that has
a side: the surname when `name` is set, else `<short> #<number>` when only
the number is known, else nothing (an unlabelled body means "unknown"; do
not draw "?" labels, they are noise). White text on a dark rounded box,
small, `cv2.putText`, never on the frames in the buffer (the board reader
and the analyst get clean frames). Tracks are looked up by the frame's
`ts` (nearest tracked frame within 0.2 s).

Caller rules, "Names" paragraph, on top of A7: players may carry a small
label above them with a surname or a team and number. A surname label is a
name you may use for that player and nothing else. A team-and-number label
means the number was read but the player is not on the sheet; say the
team. No label means unknown; role and kit. Put every label you used in
`names_read` exactly as printed. Byte-stable: the rule text does not depend
on whether marks are on.

### C5. State

`Track.name` reads come through the registry, so `state.on_pitch` already
reflects confirmed identities with decay. `summary()` gains one line,
`identified: ARG 11 Di María, 7 De Paul · FRA 10 Mbappé`, from `on_pitch`
split by side, so the caller has the names even on frames where the label
is off screen. No `ball` field from vision in this version: there is no
ball detector, and the caller can see who has it once the bodies are named.
Say so; add ball detection only if the measurement says naming the carrier
is the gap.

### C6. Ablation

`Variant.marks: bool = True`. `no-marks` is the full system with a
`NullTracker` (same type, returns no tracks) substituted, so the table shows
what the marks buy. For the simulator, RF-DETR will not find dots, so
`SimTracker` (same type) reads tracks from the sim's ground truth
(positions, sides, numbers), and the renderer's font size decides
legibility: a dot whose number is drawn at fewer than N px is "unread".
That keeps the suite runnable offline and gives the marks row a number on
the sim before any footage.

### C7. Runtime

- `Runtime.tracker: PlayerTracker` built in `__post_init__` from the vision
  extra when `--marks` is on (default on for `--source file|screen`,
  `SimTracker` for `--source sim`); `NullTracker` otherwise.
- `_ingest_frames`: every other frame, `await loop.run_in_executor(None,
  tracker.update, frame)`; store the tracks by ts; on a cut, `reset()`.
- The caller gets `tracks_for(ts)` via a callable passed into
  `caller_blocks`, which draws the marks.

### C8. Tests

No weights: `FakeDetector` returns boxes from a script, `FakeEmbedder`
returns a vector per "kit colour", `FakeNumberReader` returns scripted
(text, confidence). Test: k-means splits two kits and leaves the referee
unknown; a number confirms on the second agreeing read and not the first;
a confirmed number believes the name in the registry and `on_pitch` shows
it; a cut resets track IDs but not the registry; `caller_blocks` draws
labels only where identity is known and leaves the buffer frame
byte-identical; `SimTracker` names the sim's dots and the `no-marks`
variant runs and writes a comparable trace.

### C9. Docs

README: a "Names" subsection: the recipe above in five lines, the honest
expectation, the `name_rate` / `name_precision` columns, and that every
model in the chain is local and open. PLAN.md section 1 stays as is; add
the tracker row to section 2's table.

---

## Part B: the wire (ceiling ablation only)

**Decision (2026-09-11, superseding the earlier one): vision names the
players. The feed stays behind the grading wall as ground truth and appears
at runtime only as the off-by-default ceiling row in the ablation table.**
The user's call: a statistician in the ear is not what a human commentator
has, and the thesis is the picture, the sound, and notes. Part C is how
names come from the picture. Part B exists so the writeup can show what a
feed would buy next to what vision achieves.

Everything below stays as specified, with one reading: `--wire` is an
ablation switch, the README's claim does not change, and the state fields
the wire fills (`ball`, `named`, `incidents`) are the same fields Part C
fills from vision, so the caller prompt is one prompt whichever source is
on.

Module: `commentary/wire.py`. One StatsBomb reader, `commentary/statsbomb.py`,
used by both the wire and `grading/statsbomb.py` (so A4 becomes a thin
conversion over it; do not write the parser twice). It reads a saved file;
nothing fetches.

### B1. Types (`schemas.py`)

```python
class Event(StrEnum):   # add
    PASS = "pass"
    CARRY = "carry"
    INTERCEPTION = "interception"
    CLEARANCE = "clearance"
    TACKLE = "tackle"

class WireEvent(BaseModel):
    event: Event
    side: Side
    player: str | None = None
    recipient: str | None = None      # pass recipient; fouled player; sub off
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

`read(events_path, lineups_path, home, away) -> list[WireEvent]`. The wire
carries every StatsBomb event that names a player, because the commentary
has to say who fouled whom, who the card is for and who was offside, not
only who has the ball. StatsBomb rows: `type.name`, `minute`, `second`,
`period`, `duration`, `team.name`, `player.name`, `related_events`. Map:

| StatsBomb | `event` | `player` | `recipient` | `detail` |
|---|---|---|---|---|
| Pass (with `pass.recipient`) | PASS | passer | recipient | |
| Pass with `pass.outcome.name == "Pass Offside"` | OFFSIDE | passer | recipient (the one offside) | |
| Carry | CARRY | player | | |
| Shot, `shot.outcome.name == "Goal"` | GOAL | shooter | | `shot.type.name` if "Penalty" |
| Shot, other outcome | SHOT | shooter | | outcome name, lower |
| Own Goal Against | GOAL (other side) | | | "own goal" |
| Goal Keeper with `goalkeeper.type.name` containing "Save" | SAVE | keeper | | |
| Foul Committed | FOUL | fouler | fouled (the `Foul Won` row in `related_events`) | card colour if `foul_committed.card`, lower |
| Foul Committed with a card, or Bad Behaviour with `bad_behaviour.card` | also emit CARD | player | | "yellow" / "red" / "second yellow" |
| Offside | OFFSIDE | player | | |
| Interception, Clearance, Block | INTERCEPTION / CLEARANCE / CLEARANCE | player | | |
| Duel with `duel.type.name == "Tackle"` | TACKLE | player | | outcome name, lower |
| Substitution | SUBSTITUTION | `substitution.replacement.name` (on) | player (off) | |

Drop everything else (Ball Receipt, Pressure, Dribble, Ball Recovery,
Dispossessed, Miscontrol, Shield, 50/50, Injury Stoppage, Tactical Shift,
Half Start/End, Starting XI, Player Off/On, Error, Referee Ball-Drop,
Dribbled Past). Player names: the lineups file's `player_nickname` when
present, else `player_name`, so "Lionel Andrés Messi Cuccittini" is "Lionel
Messi". Scores run forward. `clock_s = minute * 60 + second`. Team name to
side by the two names passed in. Fixture of ~15 hand-written rows covering
a pass, an offside pass, a foul with its foul-won pair and a yellow, a
save, a goal, a sub.

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
- FOUL, OFFSIDE, SAVE, TACKLE, INTERCEPTION, CLEARANCE: append to a new
  `state.named: list[NamedEvent]` (event, side, player, recipient, detail,
  video_ts), kept to the last 6. Also append the event to `last_events`.

`summary()` adds, after the scoreline: `on the ball: Mac Allister (ARG),
from Otamendi` when `ball` is set and fresh (under 6 s old at the cursor);
then `just now:` with the named events of the last 20 s, newest last, one
per line in the form `foul by Rabiot (FRA) on Messi` / `yellow card Rabiot
(FRA)` / `offside Giroud (FRA)` / `save Martínez (ARG)`; then one line each
for goals, cards, subs so far. Nothing else changes.

### B6. Prompt

`CALLER_RULES` "Names" paragraph (on top of A7): add that MATCH STATE may
carry "on the ball", "from" and a "just now" list from a statistician; those
names may be used as given, for exactly the thing the statistician says
they did, and are the only names allowed without a legible number or
graphic. A foul in the picture with "foul by Rabiot on Messi" in the state
is called with both names; a foul with nothing in the state is called by
kit and role. Change "no data feed, no statistician" at the top to say the
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
  in it counts as a name; it is correct if that player is the `player` or
  `recipient` of any wire event within `window_s` of the line's `video_ts`. `Scorecard.name_rate` (lines with a name / lines) and
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
  recipient is dropped; a foul carries the fouled player from its related
  Foul Won row and emits a CARD alongside when carded; an offside pass
  names the recipient as the one offside; a sub carries on and off.
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

README: the opening claim does not change. A "The wire" subsection under
Ablations: the ceiling row, the two-clock rule, why a live feed's latency
sets the minimum delay, the name-precision columns, and that the default
runtime never loads one. PLAN.md: one paragraph under section 6.

---

## Report back

Commits; the A6 finding and the test that shows it; the ablation table re-run
with the `wire-10s` row (same command as the README table); anything done
differently from this brief and why.

---

## Addendum: caller model and latency

The caller runs on `claude-opus-5` (`.env`), Haiku stays on the board
reader. Opus 5 thinks by default; the caller has to answer inside the delay
window, so `Caller.call` must pass `effort="low"` to the backend (the
backend already forwards it; check `caller.py` passes it and add it if
not). Keep the 8 s timeout. Do not disable thinking (it makes the model put
tool calls in visible text on Opus 5); low effort is the lever. The
analyst keeps whatever it has; it is not on the clock.

## Addendum: after the simplification pass

The simplification commits (`5fa89d6`..`90de9ce`) removed
`BoardChange.scoring_side` and `EntityRegistry.number_for` as dead code.
Part B needs the first one back (board goals become incidents with a side);
re-add it when B7 needs it, not before. `number_for` is not needed.
`MatchStateTracker.apply_board` now takes only a `ConfirmedBoard`.
