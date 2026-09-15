"""The ablations, and the table that compares them.

The project claims that four additions to a naive vision-narration loop make
the commentary measurably more truthful: a delay buffer with lookahead, a
scoreboard reader, a fact gate, and a second voice. A claim like that is worth
nothing without the runs that take each addition away again, so this module
turns every one of them into a named variant that can be run with one command
against the same match.

Two rules govern everything here.

The first is that an ablation must differ from the full system in exactly one
way. That is why the switches are substitutions rather than branches: after
``Runtime`` has built itself, the gate, the predictor, the caller or the
director is replaced by an object of the same type that behaves differently.
Nothing in the match loop learns that it is running an ablation, so a variant
cannot accidentally take a second, unmeasured path through the code.

The second is that every variant must write a trace the ordinary grader can
read. A bypassed gate that simply skipped ``judge`` would leave no ``gate``
rows behind, and the comparison would then be between traces of two different
shapes, which is not a comparison. So the bypass is a null object that still
answers, still records, and always says yes.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from commentary.agents.caller import Caller
from commentary.capture.buffer import DelayBuffer, Frame, now
from commentary.config import SETTINGS, CallerConfig, PredictorConfig, Settings
from commentary.director import Director
from commentary.gate import CountFact, FactGate
from commentary.grading import report
from commentary.grading.report import Scorecard
from commentary.predictor import SpeakPredictor
from commentary.runtime import Runtime
from commentary.schemas import (
    Beat,
    CallerLine,
    Event,
    GateVerdict,
    KnowledgePack,
    MatchState,
    Note,
    SpeakDecision,
    Trigger,
    Voice,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.sim.oracle import DEFAULT_OUTCOME_GUESS_ERROR
from commentary.trace import RunTrace
from commentary.voice.speaker import WORDS_PER_SECOND, LogSpeaker
from commentary.wire import ReplayWire

#: The headline chart sweeps the buffer depth over these, in seconds.
DELAY_DEPTHS: tuple[float, ...] = (0.0, 2.0, 4.0, 8.0)

#: ``speed=0`` runs the match as fast as the machine will render it. How much
#: faster than real time that turns out to be is not knowable in advance, so
#: the voice is simply made fast enough never to be the bottleneck.
FLAT_OUT_SPEECH_FACTOR = 60.0

#: worldcupvoice speaks every four seconds whether or not anything happened.
#: This is the number the whole speak predictor exists to replace, so it is
#: written down once and never tuned.
WORLDCUPVOICE_CADENCE_S = 4.0

DEFAULT_SEED = 11
#: The oracle's lying is seeded separately from the match, so two variants meet
#: the same match *and* the same opening sequence of injected errors.
ORACLE_SEED = 17


# -- the switches -------------------------------------------------------


class OpenGate(FactGate):
    """A fact gate that passes everything, for the ``no-gate`` ablation.

    It is a null object rather than a missing call. The grader counts gate rows
    in the trace, and a variant that stopped writing them would differ from the
    full system in two ways at once. So the question is still asked, the answer
    is still recorded through the same stats object, and the answer is always
    yes.
    """

    def _judge(
        self,
        line: CallerLine,
        state: MatchState,
        pack: KnowledgePack | None,
        *,
        board_changed: bool,
        wire_confirmed: bool = False,
        goal_in_state: bool = False,
        carried: str | None = None,
        at: float | None = None,
        notes: Sequence[Note] | None = None,
        ledger: Sequence[CountFact] = (),
    ) -> GateVerdict:
        return GateVerdict(
            passed=True,
            reasons=["gate_bypassed: the fact gate is off in this ablation"],
            line=line.line.strip(),
        )


class FixedCadence(SpeakPredictor):
    """worldcupvoice's timer: a line every N seconds, triggers ignored.

    The naive loop has no notion of a moment worth calling. It wakes on a
    schedule, looks at whatever is on screen, and talks. Keeping the same
    ``SpeakPredictor`` type means the tick loop, the trigger row in the trace
    and the rate accounting are untouched; only the verdict changes.
    """

    def __init__(
        self,
        cadence_s: float = WORLDCUPVOICE_CADENCE_S,
        cfg: PredictorConfig = SETTINGS.predictor,
        caller: CallerConfig = SETTINGS.caller,
    ) -> None:
        super().__init__(cfg, caller)
        self.cadence_s = cadence_s
        self.last_call_ts: float | None = None

    def decide(
        self,
        now_ts: float,
        triggers: Sequence[Trigger],
        last_spoken_ts: float | None,
        *,
        after_goal: bool = False,
        last_spoken_seconds: float | None = None,
        last_event: Event | None = None,
        last_quiet_ts: float | None = None,
        colour_stretch: float = 0.0,
    ) -> SpeakDecision:
        """Due every ``cadence_s`` of video time, and never for any other reason.

        Every keyword is ignored on purpose: the baseline is a timer, and a
        timer that made an exception for goals, or that shortened its wait
        after a short line, or that ran at one rate in the box and another on
        the halfway line — or that opened a hole because a second voice was
        short of the channel — would not be the baseline.

        The timer runs off this object's own last attempt rather than off the
        runtime's last spoken line. worldcupvoice narrates every four seconds;
        when its sentence is dropped as a repeat it waits another four, it does
        not try again on the next tick. Reading ``last_spoken_ts`` instead would
        turn every suppressed line into a burst of retries, and the cadence
        this baseline exists to reproduce would not be a cadence.
        """
        self.ticks += 1
        since = float("inf") if self.last_call_ts is None else now_ts - self.last_call_ts
        due = since >= self.cadence_s
        if due:
            self.last_call_ts = now_ts
        return self._record(
            SpeakDecision(
                should_call=due,
                triggers=[Trigger.SCHEDULED],
                urgency=1.0 if due else 0.0,
                reason=f"fixed cadence {self.cadence_s:.1f}s: {'due' if due else 'waiting'}",
            )
        )


class StatelessCaller(Caller):
    """A caller shown the pictures and nothing else.

    worldcupvoice sends frames and asks for a sentence. It has no match state
    to put in front of the model and no reason-for-asking, because it never had
    a reason. Both are withheld here rather than blanked downstream, so the
    prompt really is the naive one.
    """

    async def call(
        self,
        buffer: DelayBuffer,
        state_summary: str,
        triggers: list[Trigger],
        lookahead_until: float | None = None,
    ) -> CallerLine | None:
        # The cut guard is withheld too. worldcupvoice has no scene detection,
        # so its lookahead runs straight across a cut into whatever the
        # broadcaster went to next — which is part of what makes it the
        # baseline rather than the system.
        return await super().call(buffer, "", [], lookahead_until=None)


class CallerOnlyDirector(Director):
    """One voice on the channel: a beat from anyone else is refused.

    The switch that actually suppresses the analyst is the runtime's own
    ``with_analyst``, which stops it being called at all and so keeps the
    ablation's cost column honest. This is the backstop underneath it: every
    voice reaches the speaker through the director, so with this in place the
    single-voice claim is structural rather than a hope about configuration.
    """

    def submit(self, beat: Beat) -> bool:
        if beat.voice is not Voice.CALLER:
            return False
        return super().submit(beat)


# -- pacing -------------------------------------------------------------


class PacedSource:
    """A simulator source played back at a chosen multiple of real time.

    This exists because the system runs on two clocks. Frames carry video
    time, but the board reader's two-second interval, the predictor's tick and
    the director's staleness cut are all ``asyncio.sleep`` against the wall.
    Run the simulator flat out and those loops do not speed up with it: the
    board gets read once per minute of match instead of once every two
    seconds, and every goal then fails the gate for want of a board change.
    That is an artefact of the harness and it would be reported as a result.

    So the pacing is a dial rather than a switch. ``speed=1`` is a real match
    in real time and every threshold in ``Settings`` means what it will mean in
    October; ``speed=4`` distorts the wall-clock loops by four and finishes
    four times sooner; ``speed=0`` is flat out, which is fast enough for a
    smoke run and honest only about the things measured in video time.
    """

    def __init__(self, inner: SimSource, speed: float) -> None:
        if speed < 0:
            raise ValueError("speed must be >= 0")
        self.inner = inner
        self.speed = speed
        self._start = 0.0

    async def __aenter__(self) -> PacedSource:
        await self.inner.__aenter__()
        self._start = now()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.inner.__aexit__(*exc)

    async def _hold(self, ts: float) -> None:
        if self.speed <= 0:
            return
        behind = ts / self.speed - (now() - self._start)
        if behind > 0:
            await asyncio.sleep(behind)

    async def frames(self) -> AsyncIterator[Frame]:
        async for frame in self.inner.frames():
            await self._hold(frame.ts)
            yield frame



def speaker_for(speed: float) -> LogSpeaker:
    """A voice that talks at the same multiple of real time as the match.

    ``LogSpeaker`` spends wall-clock time at a human speaking rate, which is
    exactly right at ``speed=1`` and ruinous above it: a three second line
    would hold the channel for half a minute of match time and the run would
    be one long backlog. Speeding the voice up with the match keeps the number
    of *video* seconds a line occupies the same, which is what the silence
    ratio and the director's gaps are about.
    """
    factor = speed if speed > 0 else FLAT_OUT_SPEECH_FACTOR
    return LogSpeaker(words_per_second=WORDS_PER_SECOND * factor)


# -- the variants -------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One row of the results table: a name, a configuration, and the switches.

    Most of an ablation is expressible as ``Settings``, which is the point of
    having them. The switches below are the rest: the things settings cannot
    say, because the full system has no configuration for being less than
    itself.
    """

    name: str
    settings: Settings = SETTINGS
    #: False replaces the fact gate with :class:`OpenGate`.
    fact_gate: bool = True
    #: False turns the second voice off and refuses its beats at the director.
    analyst: bool = True
    #: A number here replaces the speak predictor with a metronome.
    fixed_cadence_s: float | None = None
    #: False withholds the match state and the triggers from the prompt.
    match_state_in_prompt: bool = True
    #: False runs with no pre-match notes at all, as the naive loop does.
    knowledge_pack: bool = True
    #: A number here gives the run a statistician's feed at that modelled
    #: latency. ``None`` everywhere but the ceiling row: the default runtime
    #: never loads one, and the README's claim does not depend on it.
    wire_latency_s: float | None = None
    note: str = ""

    @property
    def delay_s(self) -> float:
        return self.settings.capture.delay_s


def with_delay(base: Settings, delay_s: float) -> Settings:
    return replace(base, capture=replace(base.capture, delay_s=delay_s))


def worldcupvoice(base: Settings = SETTINGS) -> Variant:
    """Baseline one: the loop this project is a reply to.

    Frames in, one sentence out, every four seconds, with no delay, no
    lookahead, no match state, no notes, no fact gate and one voice. Its word
    cap is kept, because that it does have and removing it would be building
    a worse baseline than the real thing.
    """
    settings = replace(
        base,
        capture=replace(base.capture, delay_s=0.0),
        caller=replace(
            base.caller,
            frames_lookahead=0,
            min_gap_s=WORLDCUPVOICE_CADENCE_S,
        ),
    )
    return Variant(
        name="worldcupvoice",
        settings=settings,
        fact_gate=False,
        analyst=False,
        fixed_cadence_s=WORLDCUPVOICE_CADENCE_S,
        match_state_in_prompt=False,
        knowledge_pack=False,
        note="reproduced naive loop: fixed 4s cadence, no delay, no state, no gate",
    )


def full(base: Settings = SETTINGS) -> Variant:
    return Variant(name="full", settings=base, note="the system as designed")


def no_delay(base: Settings = SETTINGS) -> Variant:
    return Variant(
        name="no-delay",
        settings=with_delay(base, 0.0),
        note="full system, delay_s = 0 (which also removes the lookahead frames)",
    )


def no_gate(base: Settings = SETTINGS) -> Variant:
    return Variant(
        name="no-gate",
        settings=base,
        fact_gate=False,
        note="full system with the fact gate bypassed",
    )


def single_voice(base: Settings = SETTINGS) -> Variant:
    return Variant(
        name="single-voice",
        settings=base,
        analyst=False,
        note="full system with the analyst suppressed",
    )


#: What a live play-by-play feed costs in lag. Ten seconds is the number the
#: delay argument is made against: with the buffer at eight the feed is still
#: two seconds stale at the cursor, and with no buffer it is ten.
DEFAULT_WIRE_LATENCY_S = 10.0


def wire(base: Settings = SETTINGS) -> Variant:
    """The ceiling: everything the system has, plus a statistician.

    Not a row the project is claiming. It is here so the writeup can put a
    number next to what vision achieves — what a feed would have bought, at
    the cost of the one thing a human commentator does not have.
    """
    return Variant(
        name=f"wire-{DEFAULT_WIRE_LATENCY_S:g}s",
        settings=base,
        wire_latency_s=DEFAULT_WIRE_LATENCY_S,
        note=(
            f"vision plus the play-by-play feed at {DEFAULT_WIRE_LATENCY_S:g}s modelled "
            "latency; on the sim the feed has no passes, so this is goals, cards and subs"
        ),
    )


def standard_variants(base: Settings = SETTINGS) -> list[Variant]:
    """The runs of PLAN section 9, in the order the table wants them."""
    return [
        worldcupvoice(base),
        full(base),
        no_delay(base),
        no_gate(base),
        single_voice(base),
        wire(base),
    ]


def delay_sweep(base: Settings = SETTINGS, depths: Sequence[float] = DELAY_DEPTHS) -> list[Variant]:
    """The headline chart: the full system at each buffer depth."""
    return [
        Variant(
            name=delay_name(depth),
            settings=with_delay(base, depth),
            note=f"full system, buffer depth {depth:g}s",
        )
        for depth in depths
    ]


def delay_name(depth: float) -> str:
    return f"delay-{depth:g}s"


def delay_of(name: str) -> float | None:
    """The depth encoded in a sweep variant's name, or None for the others.

    A ``Scorecard`` carries a name and numbers and nothing else, so the sweep
    table recovers its x axis from the naming convention defined just above.
    """
    if not name.startswith("delay-") or not name.endswith("s"):
        return None
    try:
        return float(name[len("delay-") : -1])
    except ValueError:
        return None


# -- running ------------------------------------------------------------


@dataclass
class BaselineRuntime(Runtime):
    """A ``Runtime`` wearing one variant's switches.

    Every switch is a part swapped for another part of the same type, after
    the runtime has finished building itself. Nothing below this line knows an
    ablation is running, which is what keeps the traces comparable.
    """

    variant: Variant = field(default_factory=lambda: Variant(name="full"))

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.variant.fact_gate:
            self.gate = OpenGate(self.settings.gate)
        if self.variant.fixed_cadence_s is not None:
            self.predictor = FixedCadence(
                self.variant.fixed_cadence_s,
                self.settings.predictor,
                self.settings.caller,
            )
        if not self.variant.match_state_in_prompt:
            self.caller = StatelessCaller(
                self.backend,
                config=self.settings.caller,
                pack=self.pack,
            )
        if not self.variant.analyst:
            self.director = CallerOnlyDirector(
                speaker=self.speaker, cfg=self.settings.director, bus=self.bus
            )


def trace_file(out_dir: Path, name: str) -> Path:
    return out_dir / f"{name}.jsonl"


async def run_variant(
    variant: Variant,
    sim: MatchSim,
    *,
    seconds: float,
    out_dir: Path,
    error_rate: float = 0.0,
    outcome_guess_error: float = DEFAULT_OUTCOME_GUESS_ERROR,
    speed: float = 0.0,
    oracle_seed: int = ORACLE_SEED,
) -> Scorecard:
    """Run one variant against a given match and grade the trace it leaves.

    The match is passed in rather than built here so that every variant in a
    suite sees the identical ninety minutes. A comparison in which the baseline
    got an easier match is not a comparison.
    """
    path = trace_file(out_dir, variant.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Traces are opened for append, so a file left over from an earlier suite
    # would be graded together with this one.
    path.unlink(missing_ok=True)

    inner = SimSource(sim, variant.settings.capture, realtime=False)
    source = PacedSource(inner, speed)
    oracle = SimOracle(
        sim=sim,
        error_rate=error_rate,
        outcome_guess_error=outcome_guess_error,
        seed=oracle_seed,
    )
    pack = sim.knowledge_pack if variant.knowledge_pack else None

    with RunTrace(path=path) as trace:
        runtime = BaselineRuntime(
            source=source,
            backend=oracle,
            pack=pack,
            settings=variant.settings,
            speaker=speaker_for(speed),
            trace=trace,
            with_analyst=variant.analyst,
            wire=(
                None
                if variant.wire_latency_s is None
                else ReplayWire.from_truth(sim.ground_truth, variant.wire_latency_s)
            ),
            variant=variant,
        )
        await runtime.run(seconds=seconds)

    # Grade against the stretch of match this variant actually called. A run
    # cut off after forty-five seconds is not answerable for the goals in the
    # four minutes it never saw a frame of, and scoring it against the whole
    # fixture drags every variant's recall down by the same large amount —
    # which looks like a finding and is an artefact of the run length.
    #
    # The window ends at the narration cursor rather than at the live edge.
    # The cursor is how far the commentary got; the last ``delay_s`` seconds
    # were ingested but never called, by construction. Charging a variant for
    # those would take marks off in direct proportion to its buffer depth,
    # which is the single thing the delay sweep is trying to measure.
    watched = max(0.0, runtime.cursor_ts)
    truth = [event for event in sim.ground_truth if event.video_ts <= watched]

    # The truth and the notes handed to the grader are the sim's own, never the
    # variant's: worldcupvoice runs without a roster but is still judged
    # against one, exactly as it would be against a real match.
    #
    # The same truth doubles as the feed the names are scored against, which
    # is the best this fixture can do and is a floor rather than the number:
    # the sim's script names a player at a goal, a save, a foul, a card and a
    # substitution and at nothing else, so a correct name said during an
    # ordinary passage of play has nothing to be marked right against. On real
    # footage this column is scored against StatsBomb, which names every touch.
    card = report.score(
        variant.name,
        path,
        truth,
        sim.knowledge_pack,
        duration_s=watched,
        wire_events=ReplayWire.from_truth(truth, 0.0).events,
    )
    card.watched_s = watched
    return card


async def run_suite(
    variants: Sequence[Variant],
    *,
    seed: int = DEFAULT_SEED,
    duration_s: float = 180.0,
    seconds: float = 20.0,
    error_rate: float = 0.2,
    outcome_guess_error: float = DEFAULT_OUTCOME_GUESS_ERROR,
    out_dir: Path = Path("runs"),
    speed: float = 0.0,
) -> list[Scorecard]:
    """Run every variant against one match, one after another.

    Sequentially, deliberately. The runs share wall-clock pacing assumptions —
    the speaker occupies real time, the director's gaps are real gaps — so two
    of them on one event loop would distort the latency and silence numbers,
    which are among the things being measured.
    """
    sim = MatchSim(seed=seed, duration_s=duration_s)
    cards: list[Scorecard] = []
    for variant in variants:
        cards.append(
            await run_variant(
                variant,
                sim,
                seconds=seconds,
                out_dir=out_dir,
                error_rate=error_rate,
                outcome_guess_error=outcome_guess_error,
                speed=speed,
            )
        )
    return cards


# -- reporting ----------------------------------------------------------


def error_count(card: Scorecard) -> int:
    """How many factual errors survived to a microphone, in absolute terms.

    The rate is what the table shows, but an ablation that speaks less can
    flatter its rate, so the comparisons in the tests are made on the count.
    """
    return sum(card.errors_by_kind.values())


SWEEP_HEADER = (
    "| delay depth | lines | factual err | errors | recall | lag p50/p95 s |\n"
    "|---|---:|---:|---:|---:|---:|"
)


def sweep_table(cards: Sequence[Scorecard]) -> str:
    """Factual error rate against delay depth: the headline chart, as numbers."""
    rows = []
    for card in cards:
        depth = delay_of(card.name)
        if depth is None:
            continue
        rows.append(
            f"| {depth:g} s | {card.lines} | {card.factual_error_rate:.1%} | "
            f"{error_count(card)} | {card.event_recall:.0%} | "
            f"{card.lag_p50:.1f} / {card.lag_p95:.1f} |"
        )
    return "\n".join([SWEEP_HEADER, *rows]) if rows else ""


#: Two runs whose windows differ by less than this watched the same match for
#: reporting purposes. Wall-clock scheduling alone moves the end of a run by a
#: second or two, and reporting that as a difference would be noise.
SPAN_TOLERANCE_S = 5.0

#: Travels with the table, because the table will be pasted somewhere without
#: it otherwise.
CAVEAT = (
    "_These are simulator numbers against a stand-in oracle, not a vision model on "
    "real footage. They compare variants against each other; they do not measure "
    "how well the system calls a football match._"
)


def coverage(cards: Sequence[Scorecard]) -> str:
    """How much of the match each variant watched, in one line.

    Recall and silence ratio are fractions of this window and of nothing else,
    so the window has to be printed beside them. When the windows differ the
    line says so instead of quietly implying the columns are comparable.
    """
    spans = [card.watched_s for card in cards]
    if not spans:
        return ""
    if max(spans) - min(spans) <= SPAN_TOLERANCE_S:
        return (
            f"Each variant called the first {max(spans):.0f}s of the match. Recall and "
            "silence are fractions of that window, not of the whole fixture."
        )
    listing = ", ".join(f"{card.name} {card.watched_s:.0f}s" for card in cards)
    return (
        "Variants called different stretches of the match, so their recall and silence "
        f"figures are not directly comparable: {listing}."
    )


def render(cards: Sequence[Scorecard]) -> str:
    """The results section: the comparison, the breakdowns, then the sweep."""
    parts: list[str] = ["## Results", "", CAVEAT, ""]
    window = coverage(cards)
    if window:
        parts.extend([window, ""])
    parts.extend([report.table(list(cards)), ""])
    for card in cards:
        parts.extend([report.detail(card), ""])
    sweep = sweep_table(cards)
    if sweep:
        parts.extend(["### Factual error rate against delay depth", "", sweep, ""])
    return "\n".join(parts).rstrip() + "\n"


# -- command line -------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m commentary.grading.baselines",
        description="Run the ablations against the simulator and print the results table.",
    )
    parser.add_argument(
        "--duration", type=float, default=180.0, help="seconds of simulated match (default 180)"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=20.0,
        help="wall-clock seconds to let each variant run (default 20)",
    )
    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.2,
        help="fraction of oracle calls that come back deliberately wrong (default 0.2)",
    )
    parser.add_argument(
        "--outcome-guess-error",
        type=float,
        default=DEFAULT_OUTCOME_GUESS_ERROR,
        help="how often a caller with no lookahead calls the end of a move wrongly. "
        "This is the simulator's modelling assumption and the delay chart is only "
        "as good as it, so it is a dial rather than a constant "
        f"(default {DEFAULT_OUTCOME_GUESS_ERROR:g})",
    )
    parser.add_argument("--out", type=Path, default=Path("runs"), help="where traces are written")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="match seed")
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="playback speed: 1 is real time and the only setting under which the "
        "wall-clock loops (board reader, tick, staleness) behave as they will "
        "live; 0, the default, is flat out and finishes in a couple of minutes",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        metavar="NAME",
        help="run just these variants by name",
    )
    parser.add_argument(
        "--no-sweep", action="store_true", help="skip the delay sweep and run the five variants"
    )
    return parser


def suite_for(args: argparse.Namespace) -> list[Variant]:
    """Which variants to run. ``--only`` picks from everything this module knows.

    Naming a sweep depth under ``--only`` selects it whether or not
    ``--no-sweep`` was passed, because asking for a variant by name is a
    clearer statement of intent than a flag about the default set.
    """
    catalogue = [*standard_variants(), *delay_sweep()]
    if args.only:
        wanted = set(args.only)
        return [v for v in catalogue if v.name in wanted]
    if args.no_sweep:
        return standard_variants()
    return catalogue


def provenance(args: argparse.Namespace, variants: Sequence[Variant]) -> str:
    """What produced these numbers, so a table in a README can be re-run."""
    pacing = (
        "flat out: the board reader, the tick loop and the director's staleness "
        "cut run on the wall clock and so see far less of the match than they will live"
        if args.speed <= 0
        else f"{args.speed:g}x real time"
    )
    return "\n".join(
        [
            f"Simulator, seed {args.seed}, {args.duration:g}s of match, "
            f"oracle hallucination rate {args.error_rate:g}, "
            f"outcome guess error {args.outcome_guess_error:g}.",
            f"Each variant run for up to {args.seconds:g}s of wall clock, {pacing}.",
            f"Traces: {args.out}/<variant>.jsonl",
            "",
            *(f"- `{v.name}` — {v.note}" for v in variants),
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = suite_for(args)
    if not variants:
        print("no variants selected", flush=True)
        return 2
    cards = asyncio.run(
        run_suite(
            variants,
            seed=args.seed,
            duration_s=args.duration,
            seconds=args.seconds,
            error_rate=args.error_rate,
            outcome_guess_error=args.outcome_guess_error,
            out_dir=args.out,
            speed=args.speed,
        )
    )
    print(provenance(args, variants))
    print()
    print(render(cards))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
