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
from commentary.capture.buffer import AudioChunk, DelayBuffer, Frame, now
from commentary.config import (
    SETTINGS,
    AnalystConfig,
    CallerConfig,
    PredictorConfig,
    Settings,
)
from commentary.director import Director
from commentary.gate import FactGate
from commentary.grading import report
from commentary.grading.report import Scorecard
from commentary.predictor import SpeakPredictor
from commentary.runtime import Runtime
from commentary.schemas import (
    Beat,
    CallerLine,
    GateVerdict,
    KnowledgePack,
    MatchState,
    SpeakDecision,
    Trigger,
    Voice,
)
from commentary.sim import MatchSim, SimOracle, SimSource
from commentary.trace import RunTrace
from commentary.voice.speaker import WORDS_PER_SECOND, LogSpeaker

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
        lookahead_celebration: bool,
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

    def decide(
        self,
        now_ts: float,
        triggers: Sequence[Trigger],
        last_spoken_ts: float | None,
        last_line_salience: float = 0.0,
    ) -> SpeakDecision:
        self.ticks += 1
        silence_s = float("inf") if last_spoken_ts is None else now_ts - last_spoken_ts
        due = silence_s >= self.cadence_s
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
    ) -> CallerLine | None:
        return await super().call(buffer, "", [])


class CallerOnlyDirector(Director):
    """One voice on the channel: a beat from anyone else is refused.

    Suppressing the analyst at the director rather than at its source is what
    makes this ablation survive however the second voice ends up being wired
    in, since every voice reaches the speaker through here.
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

    async def audio(self) -> AsyncIterator[AudioChunk]:
        async for chunk in self.inner.audio():
            await self._hold(chunk.ts)
            yield chunk


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
    having them. The four booleans are the things settings cannot say, because
    the full system has no configuration for being less than itself.
    """

    name: str
    settings: Settings = SETTINGS
    #: False replaces the fact gate with :class:`OpenGate`.
    fact_gate: bool = True
    #: False drops every beat that is not the play-by-play voice.
    analyst: bool = True
    #: A number here replaces the speak predictor with a metronome.
    fixed_cadence_s: float | None = None
    #: False withholds the match state and the triggers from the prompt.
    match_state_in_prompt: bool = True
    #: False runs with no pre-match notes at all, as the naive loop does.
    knowledge_pack: bool = True
    note: str = ""

    @property
    def delay_s(self) -> float:
        return self.settings.capture.delay_s


def _mute_analyst(cfg: AnalystConfig) -> AnalystConfig:
    """Thresholds a lull can never reach, as a second lock on the analyst.

    The director already refuses the beats. This makes the ablation cost
    nothing as well as say nothing, so the single-voice row's cost column is
    the cost of one voice rather than of two with one thrown away.
    """
    return replace(cfg, lull_s=1.0e9, min_gap_s=1.0e9)


def with_delay(base: Settings, delay_s: float) -> Settings:
    return replace(base, capture=replace(base.capture, delay_s=delay_s))


def worldcupvoice(base: Settings = SETTINGS) -> Variant:
    """Baseline one: the loop this project is a reply to.

    Frames in, one sentence out, every four seconds, with no delay, no
    lookahead, no match state, no notes, no fact gate and one voice. Its
    repetition gate and word cap are kept, because those it does have and
    removing them would be building a worse baseline than the real thing.
    """
    settings = replace(
        base,
        capture=replace(base.capture, delay_s=0.0),
        caller=replace(
            base.caller,
            frames_lookahead=0,
            min_gap_s=WORLDCUPVOICE_CADENCE_S,
        ),
        analyst=_mute_analyst(base.analyst),
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
        settings=replace(base, analyst=_mute_analyst(base.analyst)),
        analyst=False,
        note="full system with the analyst suppressed",
    )


def standard_variants(base: Settings = SETTINGS) -> list[Variant]:
    """The five runs of PLAN section 9, in the order the table wants them."""
    return [
        worldcupvoice(base),
        full(base),
        no_delay(base),
        no_gate(base),
        single_voice(base),
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
            self.caller = StatelessCaller(self.backend, config=self.settings.caller, pack=self.pack)
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

    source = PacedSource(SimSource(sim, variant.settings.capture, realtime=False), speed)
    oracle = SimOracle(sim=sim, error_rate=error_rate, seed=oracle_seed)
    pack = sim.knowledge_pack if variant.knowledge_pack else None

    with RunTrace(path=path) as trace:
        runtime = BaselineRuntime(
            source=source,
            backend=oracle,
            pack=pack,
            settings=variant.settings,
            speaker=speaker_for(speed),
            trace=trace,
            variant=variant,
        )
        await runtime.run(seconds=seconds)

    # The truth and the notes handed to the grader are the sim's own, never the
    # variant's: worldcupvoice runs without a roster but is still judged
    # against one, exactly as it would be against a real match.
    return report.score(variant.name, path, sim.ground_truth, sim.knowledge_pack)


async def run_suite(
    variants: Sequence[Variant],
    *,
    seed: int = DEFAULT_SEED,
    duration_s: float = 180.0,
    seconds: float = 20.0,
    error_rate: float = 0.2,
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


def render(cards: Sequence[Scorecard]) -> str:
    """The results section: the comparison, the breakdowns, then the sweep."""
    parts: list[str] = ["## Results", "", report.table(list(cards)), ""]
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
    variants = standard_variants()
    if not args.no_sweep:
        variants += delay_sweep()
    if args.only:
        wanted = set(args.only)
        variants = [v for v in variants if v.name in wanted]
    return variants


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
            f"oracle error rate {args.error_rate:g}.",
            f"Each variant run for up to {args.seconds:g}s of wall clock, {pacing}.",
            f"Variants: {', '.join(v.name for v in variants)}.",
            f"Traces: {args.out}/<variant>.jsonl",
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
