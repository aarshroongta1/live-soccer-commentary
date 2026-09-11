"""The commands. Five of them, and the one that matters is ``run``.

    uv run python -m commentary capture 10           # day one: frames reach Python
    uv run python -m commentary sim                  # watch the fake broadcast
    uv run python -m commentary research Arsenal PSG # pre-match notes, once
    uv run python -m commentary run --serve          # call a match, in a browser
    uv run python -m commentary grade runs/*.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path

from commentary.capture import DelayBuffer, FileCapture, ScreenCapture
from commentary.config import SETTINGS, Settings
from commentary.llm.base import LLMBackend, LLMError
from commentary.runtime import Runtime, trace_path
from commentary.trace import RunTrace
from commentary.voice import LogSpeaker, Speaker

# -- capture ------------------------------------------------------------


async def cmd_capture(args: argparse.Namespace) -> int:
    """Prove that whatever is on screen reaches Python."""
    cap = SETTINGS.capture
    buf = DelayBuffer(cap.fps, cap.delay_s, cap.history_s)
    count = 0
    async with ScreenCapture() as source:
        async for frame in source.frames():
            buf.append(frame)
            count += 1
            if count % cap.fps == 0:
                cursor = buf.cursor_ts
                print(
                    f"{frame.ts:6.1f}s  frames={count:5d}  "
                    f"fps={count / max(frame.ts, 1e-9):5.1f}  "
                    f"buffered={len(buf):4d}  "
                    f"cursor={'-' if cursor is None else f'{cursor:6.1f}s'}  "
                    f"ready={buf.ready}",
                    flush=True,
                )
            if frame.ts >= args.seconds:
                break
    if count == 0:
        print("no frames: check AVFOUNDATION_DEVICE, see scripts/list_devices.sh")
        return 1
    return 0


# -- sim ----------------------------------------------------------------


async def cmd_sim(args: argparse.Namespace) -> int:
    """Render the synthetic broadcast, to a window-less MP4 or to stats."""
    from commentary.sim import MatchSim

    sim = MatchSim(seed=args.seed, duration_s=args.duration)
    print(f"simulated match: {sim.knowledge_pack.home.name} v {sim.knowledge_pack.away.name}")
    print(f"{len(sim.ground_truth)} scripted events over {sim.duration_s:.0f}s")
    for event in sim.ground_truth[:20]:
        print(f"  {event.video_ts:7.1f}s  {event.event.value:<13} {event.side.value}")
    if len(sim.ground_truth) > 20:
        print(f"  ... {len(sim.ground_truth) - 20} more")

    if args.out:
        path = Path(args.out)
        await _write_video(sim, path, args.seconds)
        print(f"wrote {path}")
    return 0


async def _write_video(sim: object, path: Path, seconds: float) -> None:
    """Dump rendered frames to an MP4 so the sim can be watched like a clip."""
    import cv2

    from commentary.sim import SimSource

    cfg = SETTINGS.capture
    source = SimSource(sim, cfg, realtime=False)  # type: ignore[arg-type]
    fourcc = int(cv2.VideoWriter.fourcc(*"mp4v"))
    writer = cv2.VideoWriter(str(path), fourcc, cfg.fps, (cfg.width, cfg.height))
    try:
        async with source:
            async for frame in source.frames():
                writer.write(frame.image)
                if frame.ts >= seconds:
                    break
    finally:
        writer.release()


# -- run ----------------------------------------------------------------


def _settings(args: argparse.Namespace) -> Settings:
    capture = SETTINGS.capture
    if args.delay is not None:
        capture = replace(capture, delay_s=args.delay)
    return replace(SETTINGS, capture=capture)


def _backend(args: argparse.Namespace, sim: object | None) -> LLMBackend:
    """Anthropic when a key is around, the simulator's oracle when not.

    The oracle is not a mock standing in for a missing feature. It is how the
    pipeline is exercised without spending money on every run, and how the
    fact gate is measured against errors whose ground truth is known.
    """
    if args.backend == "oracle":
        if sim is None:
            raise SystemExit("--backend oracle only works with --source sim")
        from commentary.sim import SimOracle

        return SimOracle(sim=sim, error_rate=args.error_rate)  # type: ignore[arg-type]
    from commentary.llm import default_backend

    return default_backend()


def _speaker(args: argparse.Namespace) -> Speaker:
    """A real voice if asked for and possible, otherwise the printed one.

    Falling back rather than failing is deliberate: discovering at kickoff
    that a key has expired should cost the sound, not the match.
    """
    if args.voice == "elevenlabs":
        from commentary.voice import speaker_from_env

        spoken = speaker_from_env()
        if spoken is not None:
            return spoken
        print("no ELEVENLABS_API_KEY: falling back to the printed voice")
    return LogSpeaker(echo=True)


async def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    sim = None
    pack = None

    if args.source == "sim":
        from commentary.sim import MatchSim, SimSource

        sim = MatchSim(seed=args.seed, duration_s=args.duration)
        source: object = SimSource(sim, settings.capture, realtime=True)
        pack = sim.knowledge_pack
    elif args.source == "file":
        if not args.path:
            raise SystemExit("--source file needs --path")
        source = FileCapture(args.path, settings.capture)
    else:
        source = ScreenCapture(settings.capture)

    if args.pack:
        from commentary.agents.researcher import load_pack

        pack = load_pack(Path(args.pack))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = trace_path(out, args.source)

    with RunTrace(path=path) as trace:
        runtime = Runtime(
            source=source,
            backend=_backend(args, sim),
            pack=pack,
            settings=settings,
            speaker=_speaker(args),
            trace=trace,
        )
        server_task = asyncio.create_task(_serve(runtime, args.port)) if args.serve else None
        if server_task is not None:
            print(f"watch at http://127.0.0.1:{args.port}")
        try:
            await runtime.run(seconds=args.seconds)
        finally:
            if server_task is not None:
                server_task.cancel()

    print()
    print(runtime.gate.stats.table())
    print(f"trace: {path}")
    print(f"cost:  ${runtime.backend.total.cost_usd:.3f}")

    if sim is not None and pack is not None:
        from commentary.grading import report

        # Grade against the stretch of match that was actually watched. A run
        # cut short at twenty seconds is not answerable for the goals in the
        # eighty minutes it never saw, and scoring it against the whole
        # fixture makes every number meaningless in the same direction.
        watched = runtime.live_ts
        truth = [e for e in sim.ground_truth if e.video_ts <= watched]
        card = report.score("run", path, truth, pack, duration_s=watched)
        print()
        print(report.table([card]))
        print()
        print(report.detail(card))
    return 0


async def _serve(runtime: Runtime, port: int) -> None:
    import uvicorn

    from commentary.server import create_app

    config = uvicorn.Config(
        create_app(runtime), host="127.0.0.1", port=port, log_level="warning"
    )
    await uvicorn.Server(config).serve()


# -- serve --------------------------------------------------------------


async def cmd_research(args: argparse.Namespace) -> int:
    """Write the pre-match notes, once, before anybody is waiting.

    This is the only command that reaches the outside world, and it is meant
    to be run well before kickoff — a pack costs a real model call, and a
    live match is not the time to discover the squad numbers are wrong.
    """
    from commentary.agents.researcher import Researcher, pack_path, save_pack
    from commentary.llm import default_backend

    backend = default_backend()
    researcher = Researcher(backend)
    pack = await researcher.research(
        args.home, args.away, competition=args.competition, when=args.when
    )
    path = Path(args.out) if args.out else pack_path(args.home, args.away)
    save_pack(pack, path)

    searched = "with web search" if getattr(researcher, "used_search", False) else "from memory"
    print(f"{pack.home.name} v {pack.away.name} — researched {searched}")
    for sheet in (pack.home, pack.away):
        numbered = sum(1 for p in sheet.squad if p.number is not None)
        print(f"  {sheet.name}: {len(sheet.squad)} players, {numbered} with shirt numbers")
    print(f"  {len(pack.storylines)} storylines")
    print(f"wrote {path}")
    print(f"cost ${backend.total.cost_usd:.3f}")
    return 0


async def cmd_grade(args: argparse.Namespace) -> int:
    """Score one or more traces. Needs a pack and a ground truth to grade against."""
    from commentary.grading import metrics

    for raw in args.traces:
        path = Path(raw)
        run = metrics.load_run(path)
        print(f"{path.name}: {len(run.lines)} lines, cost ${run.cost_usd:.3f}")
        print(f"  lag p50/p95: {metrics.lag(run).as_dict()}")
        print(f"  silence:     {metrics.silence_ratio(run):.0%}")
        print(f"  repetition:  {metrics.repetition_rate(run):.1%}")
        table = metrics.gate_table(run)
        if table:
            print(f"  gate:        {table}")
    return 0


# -- wiring -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="commentary",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="stream the screen into Python and print stats")
    cap.add_argument("seconds", nargs="?", type=float, default=10.0)
    cap.set_defaults(func=cmd_capture)

    sm = sub.add_parser("sim", help="describe or render the synthetic broadcast")
    sm.add_argument("--seed", type=int, default=11)
    sm.add_argument("--duration", type=float, default=180.0)
    sm.add_argument("--seconds", type=float, default=30.0, help="how much video to write")
    sm.add_argument("--out", help="write an mp4 here")
    sm.set_defaults(func=cmd_sim)

    run = sub.add_parser("run", help="call a match")
    run.add_argument("--source", choices=["sim", "screen", "file"], default="sim")
    run.add_argument("--path", help="video file, with --source file")
    run.add_argument("--backend", choices=["oracle", "anthropic"], default="oracle")
    run.add_argument("--voice", choices=["log", "elevenlabs"], default="log")
    run.add_argument("--pack", help="knowledge pack JSON written by the researcher")
    run.add_argument("--seconds", type=float, default=60.0, help="wall-clock run length")
    run.add_argument("--duration", type=float, default=600.0, help="sim match length")
    run.add_argument("--delay", type=float, default=None, help="override the buffer depth")
    run.add_argument("--error-rate", type=float, default=0.0, help="oracle lie rate")
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--serve", action="store_true", help="also serve the watch page")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--out", default="runs")
    run.set_defaults(func=cmd_run)

    res = sub.add_parser("research", help="write the pre-match notes (needs a key)")
    res.add_argument("home")
    res.add_argument("away")
    res.add_argument("--competition", default="")
    res.add_argument("--when", default="")
    res.add_argument("--out", help="where to write the pack; defaults under packs/")
    res.set_defaults(func=cmd_research)

    grade = sub.add_parser("grade", help="print metrics for saved traces")
    grade.add_argument("traces", nargs="+")
    grade.set_defaults(func=cmd_grade)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = asyncio.run(args.func(args))
        return code
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 130
    except LLMError as exc:
        # A missing key is a thing to fix, not a thing to debug. The message
        # already says what to do, so a traceback only buries it.
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
