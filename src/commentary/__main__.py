"""The commands. Five of them, and the one that matters is ``run``.

    uv run python -m commentary capture 10           # day one: frames reach Python
    uv run python -m commentary crop --path m.mp4    # check the score-bug box by eye
    uv run python -m commentary sim                  # watch the fake broadcast
    uv run python -m commentary research Arsenal PSG # pre-match notes, once
    uv run python -m commentary notes --pack clips/pack-x.json   # add context to one
    uv run python -m commentary run --serve          # call a match, in a browser
    uv run python -m commentary replay --trace ... --path clip.mp4 --serve
    uv run python -m commentary rephrase --trace ... --pack clips/pack-x.json
    uv run python -m commentary register runs/rephrased/x/*.jsonl --no-model
    uv run python -m commentary grade runs/*.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from commentary.bus import Bus
from commentary.capture import DelayBuffer, FileCapture, FrameSource, ScreenCapture
from commentary.config import JUDGE_MODEL, SETTINGS, Settings
from commentary.llm.base import LLMBackend, LLMError
from commentary.runtime import Runtime, trace_path
from commentary.trace import RunTrace
from commentary.voice import LogSpeaker, Speaker, VoiceUnavailable

if TYPE_CHECKING:
    # Imported for types only: the simulator pulls in the renderer and cv2,
    # and `commentary capture` should not pay for either.
    from commentary.schemas import KnowledgePack
    from commentary.server import RuntimeHandle
    from commentary.sim import MatchSim
    from commentary.wire import Wire

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


async def _write_video(sim: MatchSim, path: Path, seconds: float) -> None:
    """Dump rendered frames to an MP4 so the sim can be watched like a clip."""
    import cv2

    from commentary.sim import SimSource

    cfg = SETTINGS.capture
    source = SimSource(sim, cfg, realtime=False)
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


def crop_box(text: str) -> tuple[float, float, float, float]:
    """``x0,y0,x1,y1`` as fractions of the frame, for argparse.

    Fractions rather than pixels so the same four numbers work on a 720p and
    a 1080p grab of the same broadcast, which is also how ``BoardConfig``
    stores them.
    """
    parts = text.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"crop needs four numbers, got {text!r}")
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"crop must be four numbers: {exc}") from exc
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise argparse.ArgumentTypeError(f"crop must be an ordered box inside 0..1, got {text!r}")
    return x0, y0, x1, y1


def _settings(args: argparse.Namespace) -> Settings:
    capture = SETTINGS.capture
    if args.delay is not None:
        capture = replace(capture, delay_s=args.delay)
    board = SETTINGS.board
    if args.crop is not None:
        board = replace(board, crop=args.crop)
    return replace(SETTINGS, capture=capture, board=board)


def _backend(args: argparse.Namespace, sim: MatchSim | None) -> LLMBackend:
    """Anthropic when a key is around, the simulator's oracle when not.

    The oracle is not a mock standing in for a missing feature. It is how the
    pipeline is exercised without spending money on every run, and how the
    fact gate is measured against errors whose ground truth is known.
    """
    if args.backend == "oracle":
        if sim is None:
            raise SystemExit("--backend oracle only works with --source sim")
        from commentary.sim import SimOracle

        return SimOracle(sim=sim, error_rate=args.error_rate)
    from commentary.llm import default_backend

    return default_backend()


def _wire(
    args: argparse.Namespace, sim: MatchSim | None, pack: KnowledgePack | None
) -> Wire | None:
    """A statistician, if one was asked for. Nothing loads one by default.

    This is the ablation's ceiling row reachable from the command line, and
    it is the only place in the runtime where information that did not come
    off the screen can get in.
    """
    from commentary.wire import ReplayWire

    if args.wire:
        if pack is None:
            raise SystemExit("--wire needs --pack: the feed's team names come from it")
        if not args.lineups:
            raise SystemExit("--wire needs --lineups, for the names the feed uses")
        from commentary import statsbomb

        events = statsbomb.read(
            Path(args.wire), Path(args.lineups), pack.home.name, pack.away.name
        )
        print(f"wire: {len(events)} events at {args.wire_latency:g}s modelled latency")
        return ReplayWire(events, args.wire_latency)
    if args.wire_latency is not None and sim is not None:
        return ReplayWire.from_truth(sim.ground_truth, args.wire_latency)
    return None


def _speaker(args: argparse.Namespace) -> Speaker:
    """The voice that was asked for, or an error saying why there isn't one."""
    if args.voice == "elevenlabs":
        from commentary.voice import ElevenLabsSpeaker

        return ElevenLabsSpeaker()
    if args.voice == "say":
        from commentary.voice import SaySpeaker

        return SaySpeaker()
    return LogSpeaker(echo=True)


def _pack_for_air(
    path: Path,
    *,
    trust_unchecked: bool = False,
    trust_floor: float = SETTINGS.researcher.trust_floor,
) -> KnowledgePack:
    """The pack as a match may use it: checked notes only, unless told otherwise.

    A note is a figure said out loud in a confident voice, and the fact gate
    lets it through for one reason — it is in the pack. So being in the pack
    has to mean a person read it beside its source and agreed. A freshly
    researched pack is forty-odd claims nobody has looked at yet, and the
    honest default is that they stay off air until somebody ticks them.

    ``--trust-unchecked`` is the rehearsal switch. It is how a rephrase over
    a trace finds out whether a bigger pack changes anything before the
    twenty minutes of checking are spent, and it is never the right flag for
    a broadcast. It does not mean "say all of it" any more: ``trust_floor``
    is the line below which even the rehearsal declines an unchecked note,
    because a note the researcher itself marked at confidence 0.1 is not
    "unverified but probably fine", it is a guess the researcher is on record
    doubting.

    What was skipped is printed rather than logged, because a run that
    silently has no context looks exactly like a run whose notes were all
    unusable, and those want different fixes.
    """
    from commentary.agents.researcher import above_trust_floor, load_pack, only_checked

    pack = load_pack(path)
    if trust_unchecked:
        trusted, below_floor = above_trust_floor(pack, floor=trust_floor)
        if below_floor:
            print(
                f"--trust-unchecked: {below_floor} of {len(pack.notes)} notes are unchecked and "
                f"below the trust floor ({trust_floor:g}) and will not be said"
            )
        unchecked = sum(1 for note in trusted.notes if not note.checked)
        if unchecked:
            print(
                f"--trust-unchecked: {unchecked} of {len(trusted.notes)} notes nobody has checked"
            )
        return trusted
    trimmed, skipped = only_checked(pack)
    if skipped:
        print(
            f"{skipped} of {len(pack.notes)} notes are unchecked and will not be said; "
            "tick them in the pack or pass --trust-unchecked"
        )
    return trimmed


async def cmd_run(args: argparse.Namespace) -> int:
    settings = _settings(args)
    sim = None
    pack = None
    sim_source = None

    if args.source == "sim":
        from commentary.sim import MatchSim, SimSource

        sim = MatchSim(seed=args.seed, duration_s=args.duration)
        sim_source = SimSource(sim, settings.capture, realtime=True)
        source: FrameSource = sim_source
        pack = sim.knowledge_pack
    elif args.source == "file":
        if not args.path:
            raise SystemExit("--source file needs --path")
        source = FileCapture(args.path, settings.capture)
    else:
        source = ScreenCapture(settings.capture)

    if args.pack:
        pack = _pack_for_air(
            Path(args.pack), trust_unchecked=args.trust_unchecked, trust_floor=args.trust_floor
        )

    wire = _wire(args, sim, pack)

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
            wire=wire,
        )
        watch = _Watch(runtime, args.port) if args.serve else None
        if watch is not None:
            print(f"watch at http://127.0.0.1:{args.port}")
        try:
            await runtime.run(seconds=args.seconds)
        finally:
            if watch is not None:
                await watch.close()

    print()
    print(runtime.gate.stats.table())
    print(f"trace: {path}")
    print(f"cost:  ${runtime.backend.total.cost_usd:.3f}")
    if wire is not None:
        from commentary.grading import metrics

        print(f"wire corrections: {metrics.corrections(metrics.load_run(path))}")

    if sim is not None and pack is not None:
        from commentary.grading import report

        # Grade against the stretch of match that was actually watched. A run
        # cut short at twenty seconds is not answerable for the goals in the
        # eighty minutes it never saw, and scoring it against the whole
        # fixture makes every number meaningless in the same direction.
        watched = runtime.live_ts
        truth = [e for e in sim.ground_truth if e.video_ts <= watched]
        card = report.score(
            "run",
            path,
            truth,
            pack,
            duration_s=watched,
            wire_events=wire.events if wire is not None else None,
        )
        print()
        print(report.table([card]))
        print()
        print(report.detail(card))
    return 0


class _Watch:
    """The watch page, served alongside a run for as long as the run lasts.

    Shut down rather than cancelled. Cancelling a uvicorn task with an open
    MJPEG stream and an open event stream on it prints two pages of
    ``CancelledError`` after the results table, which reads as a crash and is
    only the browser still being attached. ``force_exit`` is the right flag
    for these two endpoints in particular: neither ever ends on its own, so a
    graceful shutdown would wait for the tab to be closed.
    """

    def __init__(self, runtime: RuntimeHandle, port: int) -> None:
        import uvicorn

        from commentary.server import create_app

        config = uvicorn.Config(
            create_app(runtime),
            host="127.0.0.1",
            port=port,
            log_level="warning",
            # The app has no startup or shutdown handlers, and the lifespan
            # task is the one that prints a CancelledError traceback on the
            # way out. Nothing to run, so nothing to cancel.
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve(), name="watch")

    async def close(self) -> None:
        self._server.should_exit = True
        self._server.force_exit = True
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(self._task), timeout=2.0)
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task


# -- replay -------------------------------------------------------------


async def _record(bus: Bus, trace: RunTrace) -> None:
    """Write everything a replay publishes to a trace of its own.

    The same subscribe-and-write the runtime uses, and for the same reason:
    the director publishes its own account of what was said, and a second
    write path would be a thing the eval silently never sees. A replay with a
    voice attached is the only case that has anything new to record — the
    ``spoken`` rows are about this playback, not the run that paid for the
    lines — but recording the rest along with them costs nothing and keeps the
    file the same shape as any other trace.
    """
    async for message in bus.subscribe():
        trace.write(message)


async def cmd_replay(args: argparse.Namespace) -> int:
    """Play a finished run back: its clip through the buffer, its trace onto the bus.

    The watch page cannot tell this from a live run, which is the whole
    point — a run costs real money and happens once, and everything anybody
    needs to look at afterwards is already on disk.

    ``--voice`` makes it the cheap way to hear a change too. The trace's lines
    go through a real director and a real speaker, so the audio, the queueing
    and the mid-word cut on a goal are the live article; only the thinking was
    paid for, once, weeks ago.
    """
    from commentary.replay import from_files

    capture = SETTINGS.capture
    if args.delay is not None:
        capture = replace(capture, delay_s=args.delay)
    settings = replace(SETTINGS, capture=capture)

    speaker = _speaker(args) if args.voice != "log" else None
    # A silent replay says nothing the trace it was given does not already
    # hold, so writing a second copy of it would be litter. A spoken one is
    # the only record of how long each line actually took to come out, and
    # that number is the whole reason to run it, so it gets somewhere to land
    # whether or not anybody said where.
    out = args.out or ("runs" if speaker is not None else None)
    replay = from_files(
        args.trace,
        args.path,
        start_s=args.start,
        settings=settings,
        loop=args.loop,
        speaker=speaker,
    )
    span = max((cue.ts for cue in replay.cues), default=0.0)
    print(f"{len(replay.cues)} rows over {span:.0f}s, against {args.path} from {args.start:g}s in")
    print(
        f"buffer {settings.capture.delay_s:g}s, picture held "
        f"{settings.capture.present_offset_s:g}s behind the cursor: "
        "lines land when the run published them"
    )
    if speaker is not None:
        print(
            f"voice {args.voice}: the lines are being said again now, so the trace's own "
            "spoken rows are dropped and the director's are published instead"
        )

    trace = RunTrace(path=trace_path(Path(out), "replay")) if out else None
    if trace is not None:
        recorder = asyncio.create_task(_record(replay.bus, trace), name="trace")
        # Let the subscription register before the first frame is decoded;
        # the bus only fans out to whoever is already listening.
        await asyncio.sleep(0)

    watch = _Watch(replay, args.port) if args.serve else None
    if watch is not None:
        print(f"watch at http://127.0.0.1:{args.port}")
    try:
        await replay.run(seconds=args.seconds)
    finally:
        if watch is not None:
            await watch.close()
        if trace is not None:
            await asyncio.sleep(0.05)
            recorder.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recorder
            trace.close()
            print(f"trace: {trace.path}")

    print(f"replayed {replay.played} of {len(replay.cues)} rows")
    if replay.remaining:
        # The cursor never reached the end of the trace: the clip is shorter
        # than the run was, or --seconds cut it off. Said out loud, because a
        # demo that quietly stops two rows before the goal reads as a bug in
        # the system rather than in the arguments.
        print(f"{replay.remaining} rows the cursor never reached")
    return 0


# -- rephrase -----------------------------------------------------------


async def cmd_rephrase(args: argparse.Namespace) -> int:
    """Say a finished run's lines again, in the register, for about a cent.

    The lines were paid for once, on Opus, and the phrasing stage is a
    rewrite of words the caller already got right. So the cheap way to find
    out whether it helps is to read the trace rather than run the match: the
    caller's form, the state it was looking at and what had just been said
    are all in the file, and the only model called is the phraser.

    What comes out is a trace, so ``commentary replay`` plays it — with a
    voice, which is the only way to actually hear the difference.
    """
    from commentary.llm import default_backend
    from commentary.rephrase import load, rephrase

    rows = load(args.trace)
    pack = (
        _pack_for_air(
            Path(args.pack), trust_unchecked=args.trust_unchecked, trust_floor=args.trust_floor
        )
        if args.pack
        else None
    )
    if pack is None:
        # Said out loud: the gate's roster comes from the pack, and without
        # one every surname in every phrased line is trimmed as unverified.
        print("no --pack: the gate has only the two team names to check against")

    backend = default_backend()
    result = await rephrase(
        rows,
        backend,
        pack=pack,
        settings=SETTINGS,
        model=args.model,
        colour=args.colour,
        colour_model=args.colour_model,
    )

    out = Path(args.out)
    name = Path(args.trace).stem
    path = result.write(out / f"{name}-phrased.jsonl")

    print()
    print(result.table())
    print()
    spent = backend.total.cost_usd or result.cost_usd
    used = backend.total
    print(
        f"{len(result.lines)} lines, {result.rejected} rejected by the gate, "
        f"${spent:.4f} on {args.model or SETTINGS.phraser.model}"
    )
    # Said out loud because the example set lives in the cached prefix and is
    # most of the input: a cache_read that stays at zero means every call is
    # paying full price for two hundred utterances, which is a four-fold
    # difference over ninety minutes.
    print(
        f"tokens: in {used.input_tokens} out {used.output_tokens} "
        f"cache read {used.cache_read_tokens} write {used.cache_write_tokens}"
    )
    if result.colour is not None:
        print()
        print(result.colour.table())
        print()
        counts = result.colour.counts()
        print(
            f"colour: {counts['turns_spoken']:.0f} turns of "
            f"{counts['turns_offered']:.0f} offered, {counts['utterances']:.0f} utterances, "
            f"{counts['inside_12s_of_a_big_event']:.0f} inside 12 s of a big event "
            f"(target 0), {counts['sanctioned_goal_reactions']:.0f} sanctioned goal reactions"
        )
        print(
            f"        median {counts['median_utterances_per_turn']:.1f} utterances a turn "
            f"(target 3-4), median {counts['median_words']:.0f} words, "
            f"{counts['cue_share']:.0%} open on a cue (target >60%), "
            f"{counts['refused']:.0f} refused, ${counts['usd']:.4f}"
        )
    print(f"trace: {path}")
    print(f"hear it: uv run python -m commentary replay --trace {path} --path <clip> --voice say")
    return 0


# -- crop ---------------------------------------------------------------


async def cmd_crop(args: argparse.Namespace) -> int:
    """Show what the board reader will be looking at, before a run spends money.

    A score-bug box that is half a bug reads as an unreadable board for
    ninety minutes and there is nothing in the trace that says so. One frame
    with the box drawn on it settles that in a second.
    """
    import cv2

    cap = replace(SETTINGS.capture, fps=1)
    async with FileCapture(args.path, cap, start_s=args.at, realtime=False) as source:
        frame = await anext(aiter(source.frames()), None)
    if frame is None:
        print(f"no frame at {args.at:.0f}s in {args.path}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), _crop_preview(frame.image, args.crop))
    print(f"{args.path} at {args.at:.0f}s, crop {','.join(str(c) for c in args.crop)}")
    print(f"wrote {out}")
    return 0


def _crop_preview(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """The frame with the box drawn on it, and the cut-out bug beside it.

    The bug is scaled up to fill the panel because the question being asked
    is whether a cheap vision model can tell a 3 from an 8 in it, and at its
    native size in the corner of a 720p grab nobody can answer that by eye.
    """
    import cv2

    from commentary.perception import crop_score_bug

    marked = image.copy()
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    cv2.rectangle(
        marked,
        (int(x0 * width), int(y0 * height)),
        (int(x1 * width), int(y1 * height)),
        (0, 0, 255),
        2,
    )

    bug = crop_score_bug(image, box)
    panel_width = width // 2
    panel = np.zeros((height, panel_width, 3), dtype=image.dtype)
    scale = min(panel_width / bug.shape[1], height / bug.shape[0])
    shown = cv2.resize(
        bug, (max(1, int(bug.shape[1] * scale)), max(1, int(bug.shape[0] * scale)))
    )
    panel[: shown.shape[0], : shown.shape[1]] = shown
    return np.asarray(np.hstack([marked, panel]))


# -- serve --------------------------------------------------------------


def _researcher_backend() -> LLMBackend:
    """The client a research pass needs, which is not the one a match needs.

    :func:`default_backend` is tuned for the caller: eight seconds and one
    retry, because a line that arrives after the moment has passed is worse
    than no line. A researcher writing fifty notes issues eight web searches
    and thinks between them, which takes minutes, so on that client every
    pack would time out having already been billed. ``max_retries=0`` for the
    same reason the judge uses it: a timed-out request has very likely been
    charged, and a retry pays twice for an answer nobody sees.
    """
    from commentary.llm import grading_backend

    return grading_backend(
        timeout_s=1200.0,
        no_key_hint="this is the one command that cannot run offline",
    )


def _search_tool(max_searches: int) -> dict[str, object] | None:
    """The web search tool this pass may use, or ``None`` for none at all.

    Every search puts its results into the next turn's input, and the turns
    accumulate, so the number of searches is the one dial that decides what a
    research pass costs. Eight is right for a fixture nobody has seen; a pass
    over a match the model already knows, checking rather than discovering,
    should be told to use fewer.
    """
    from commentary.agents.researcher import WEB_SEARCH_TOOL

    if max_searches <= 0:
        return None
    return {**WEB_SEARCH_TOOL, "max_uses": max_searches}


async def cmd_research(args: argparse.Namespace) -> int:
    """Write the pre-match notes, once, before anybody is waiting.

    This is the only command that reaches the outside world, and it is meant
    to be run well before kickoff — a pack costs a real model call, and a
    live match is not the time to discover the squad numbers are wrong.
    """
    from commentary.agents.researcher import Researcher, pack_path, save_pack

    backend = _researcher_backend()
    researcher = Researcher(backend, search_tool=_search_tool(args.max_searches))
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


async def cmd_notes(args: argparse.Namespace) -> int:
    """Add the spoken-context notes to a pack that already has its team sheets.

    Separate from ``research`` because the two ages differ. Team sheets go
    stale an hour before kickoff and notes do not, and a pack whose squad
    numbers were checked against two sources last night should not have to
    risk them to gain a sentence about who is in form. One model call, the
    researcher's model, and the pack is rewritten in place unless ``--out``
    says otherwise.
    """
    from commentary.agents.researcher import (
        Researcher,
        hand_check_list,
        load_pack,
        researched_path,
        save_pack,
    )

    source = Path(args.pack)
    pack = load_pack(source)
    path = Path(args.out) if args.out else researched_path(source)
    checked = sum(1 for note in pack.notes if note.checked)
    if path == source and checked:
        raise SystemExit(
            f"{source} holds {checked} hand-checked notes; writing over it would put "
            "a model's output where somebody's checking is. Leave --out off, or name "
            "a different file."
        )

    backend = _researcher_backend()
    researcher = Researcher(backend, search_tool=_search_tool(args.max_searches))
    updated = await researcher.write_notes(pack)
    save_pack(updated, path)

    searched = "with web search" if researcher.used_search else "from memory"
    print(f"{updated.home.name} v {updated.away.name} — {len(updated.notes)} notes, {searched}")
    merge = researcher.last_merge
    if merge is not None:
        print(
            f"  {merge.kept} hand-checked kept, {merge.added} added, "
            f"{merge.clauses} no-number forms filled in, "
            f"{len(merge.duplicates)} repeats dropped"
        )
    if researcher.dropped_notes:
        print(f"  dropped {len(researcher.dropped_notes)} about names on no team sheet")
    subjects = {note.about for note in updated.notes}
    starters = [p.name for sheet in (updated.home, updated.away) for p in sheet.starters]
    uncovered = [name for name in starters if name not in subjects]
    if uncovered:
        print(f"  {len(uncovered)} starters with no note: {', '.join(uncovered)}")
    print(f"wrote {path}")
    print(f"cost ${backend.total.cost_usd:.3f}")
    print()
    print(hand_check_list(updated))
    return 0


async def cmd_captions(args: argparse.Namespace) -> int:
    """Turn the broadcast's own captions into the transcript the eval reads.

    yt-dlp leaves a ``.en.json3`` beside the video when it is asked for one,
    which is ninety minutes of professional commentary already timestamped on
    the video clock. Nothing here reaches the network.
    """
    from commentary.grading import captions, transcripts

    transcript = captions.load_json3(Path(args.path))
    out = Path(args.out)
    transcripts.save(transcript, out)
    print(f"{len(transcript)} segments over {transcript.duration_s:.0f}s")
    print(f"wrote {out}")
    return 0


async def cmd_feed(args: argparse.Namespace) -> int:
    """StatsBomb's event file to the feed the grader reads. Never fetches."""
    import json

    from commentary.grading import statsbomb

    rows = json.loads(Path(args.path).read_text(encoding="utf-8"))
    document = statsbomb.convert(rows, args.home, args.away)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} rows in, {len(document['events'])} events out")
    print(f"wrote {out}")
    return 0


def _judge_backend(timeout_s: float) -> LLMBackend:
    """The ``register`` command's backend: see :func:`commentary.llm.grading_backend`.

    A thin wrapper rather than a second copy — :mod:`commentary.grading.judge`
    needed the identical long-timeout, zero-retry treatment and the
    construction is shared from the ``llm`` package now. Only the
    credentials-missing hint stays local, because it is this command's own
    flag.
    """
    from commentary.llm import grading_backend

    return grading_backend(timeout_s, no_key_hint="or use --no-model")


async def cmd_register(args: argparse.Namespace) -> int:
    """Score a trace's commentary against real commentary, and say what it cost.

    Two layers, printed side by side and never averaged. The free one counts
    line lengths, bare names, repeated openers, gaps and numbers, each beside
    what a real broadcast measures. The paid one is a single Opus call over
    the whole line sequence, scoring nine dimensions of register out of ten
    and quoting the three worst lines back.

    ``--no-model`` runs only the free layer, which is the one to watch from
    iteration to iteration: it costs nothing, so it can be run on every trace
    after every phraser change without asking anybody.
    """
    from commentary.agents.researcher import load_pack
    from commentary.grading import register

    pack = load_pack(Path(args.pack)) if args.pack else None
    backend = None if args.no_model else _judge_backend(args.timeout)
    if pack is None:
        # Said out loud, like `rephrase` does: without a team sheet the only
        # names to check a line against are the ones the trace itself read
        # off the pictures, so the bare-name and name shares are floors.
        print("no --pack: names come from the trace's own sightings, so name shares are floors")

    for raw in args.traces:
        path = Path(raw)
        report = await register.score_trace(
            path, pack=pack, backend=backend, model=args.model
        )
        out = register.report_path(path)
        report.write(out)
        print()
        print(report.table())
        print(f"report: {out}")
    return 0


async def cmd_grade(args: argparse.Namespace) -> int:
    """Score traces. With a pack and StatsBomb's files, the whole definition of done.

    Bare, it prints the numbers a trace can give on its own. Given the two
    StatsBomb files and the pack, it does the thing a real-footage run is only
    worth its cost for: aligns the feed onto video time from the trace's own
    board readings, builds the truth for the window that was actually watched,
    and answers the brief's twelve questions with the evidence for each.

    The alignment is printed first and checked first. An offset four seconds
    out produces a complete, plausible table in which every line has missed
    its event, so a suspect fit refuses to grade rather than reporting one.
    """
    from commentary.grading import metrics

    if not args.statsbomb:
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

    if not args.pack:
        print("--statsbomb needs --pack: the roster is what a name is checked against")
        return 2
    if not args.lineups:
        print("--statsbomb needs --lineups: it is where the names a commentator says come from")
        return 2

    code = 0
    for raw in args.traces:
        code = max(code, _grade_fully(Path(raw), args))
    return code


def _grade_fully(path: Path, args: argparse.Namespace) -> int:
    """One trace, graded end to end against StatsBomb. Returns a shell code."""
    from commentary import statsbomb
    from commentary.agents.researcher import load_pack
    from commentary.grading import checklist, feed, metrics, report
    from commentary.schemas import Event
    from commentary.trace import read_trace, rows_of

    pack = load_pack(Path(args.pack))
    home = args.home or pack.home.name
    away = args.away or pack.away.name
    wire = statsbomb.read(Path(args.statsbomb), Path(args.lineups), home, away)

    rows = read_trace(path)
    run = metrics.load_run(path)
    print(f"== {path}")
    if args.offset is not None:
        # A shootout has no clock on the screen at all — the bug is replaced
        # by a tally — so there is nothing to fit and the offset has to be
        # measured off a frame by hand. Said out loud, because an offset
        # somebody typed is not an offset anything checked.
        alignment = feed.Alignment(offsets={p: args.offset for p in range(1, 6)}, n=0)
        alignment.events = feed.shift(feed.from_wire(wire), args.offset)
        print(f"alignment BY HAND: {args.offset:+.1f}s, nothing fitted, nothing checked")
    else:
        alignment = feed.align(
            feed.from_wire(wire), rows_of(rows, "board"), tolerance_s=args.tolerance
        )
        print(alignment.summary())
        if not alignment.ok:
            print("refusing to grade: fix the alignment first, every number below it is fiction")
            print("(if the broadcast shows no clock at all, measure it off a frame and --offset)")
            return 1

    stamped = feed.stamp(wire, alignment)
    watched_s = args.watched or max((float(r.get("ts", 0.0)) for r in rows), default=0.0)
    # A commentator is not graded on the passes and carries between the
    # events; everything else StatsBomb has in the window is fair game.
    truth = [
        event
        for event in alignment.events
        if event.event not in (Event.PASS, Event.CARRY) and 0.0 <= event.video_ts <= watched_s
    ]
    print(f"{len(truth)} graded events in the {watched_s:.0f}s watched")

    card = report.score(
        path.stem,
        path,
        truth,
        pack,
        duration_s=watched_s,
        wire_events=stamped,
    )
    card.watched_s = watched_s
    print()
    print(report.table([card]))
    print()
    print(report.detail(card))
    print()
    state = checklist.watched(rows, run, truth, stamped, pack, watched_s=watched_s)
    items = checklist.check(state)
    print(checklist.report(items))
    print()
    made = checklist.calls(state)
    spoke = sum(1 for call in made if call.said)
    print(f"== events called ({spoke} of {len(made)} said, not merely near)")
    for call in made:
        print(call.row())
    print()
    print("== spoken")
    offset = -alignment.offset_for(1)
    for line in state.lines:
        clock = line.ts + offset
        print(f"  {int(clock // 60)}:{int(clock % 60):02d} [{line.voice}] {line.text}")
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
    run.add_argument("--voice", choices=["log", "say", "elevenlabs"], default="log")
    run.add_argument("--pack", help="knowledge pack JSON written by the researcher")
    run.add_argument(
        "--trust-unchecked",
        action="store_true",
        help="say the pack notes nobody has hand-checked; off by default",
    )
    run.add_argument(
        "--trust-floor",
        type=float,
        default=SETTINGS.researcher.trust_floor,
        help=(
            "with --trust-unchecked, the lowest confidence an unchecked note may still say "
            f"(default {SETTINGS.researcher.trust_floor:g})"
        ),
    )
    run.add_argument("--seconds", type=float, default=60.0, help="wall-clock run length")
    run.add_argument("--duration", type=float, default=600.0, help="sim match length")
    run.add_argument("--delay", type=float, default=None, help="override the buffer depth")
    run.add_argument(
        "--crop",
        type=crop_box,
        default=None,
        help="score-bug box as x0,y0,x1,y1 fractions; check it with the crop command",
    )
    run.add_argument("--error-rate", type=float, default=0.0, help="oracle lie rate")
    run.add_argument("--seed", type=int, default=11)
    run.add_argument("--wire", help="StatsBomb events JSON: the ablation's ceiling row")
    run.add_argument("--lineups", help="StatsBomb lineups JSON, for the names the feed uses")
    run.add_argument(
        "--wire-latency",
        type=float,
        default=None,
        help="modelled feed latency in seconds; on --source sim this alone builds the wire",
    )
    run.add_argument("--serve", action="store_true", help="also serve the watch page")
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--out", default="runs")
    run.set_defaults(func=cmd_run)

    rp = sub.add_parser("replay", help="watch a saved run again, from its trace and its clip")
    rp.add_argument("--trace", required=True, help="runs/<name>/<run>.jsonl")
    rp.add_argument("--path", required=True, help="the clip that run was watching")
    rp.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="seconds into the clip the run began at; a --source file run began at 0",
    )
    rp.add_argument("--delay", type=float, default=None, help="override the buffer depth")
    rp.add_argument("--seconds", type=float, default=None, help="stop after this long")
    rp.add_argument(
        "--voice",
        choices=["log", "say", "elevenlabs"],
        default="log",
        help="say the trace's lines again through a real voice; log is silent, as before",
    )
    rp.add_argument(
        "--out",
        default=None,
        help="write a trace of this replay under here; defaults to runs/ when a voice is on",
    )
    rp.add_argument("--serve", action="store_true", help="also serve the watch page")
    rp.add_argument("--port", type=int, default=8000)
    rp.add_argument(
        "--loop",
        action="store_true",
        help="once the trace ends, seek back to --start and play it again, forever",
    )
    rp.set_defaults(func=cmd_replay)

    rph = sub.add_parser(
        "rephrase", help="rewrite a saved run's lines in the commentator's register"
    )
    rph.add_argument("--trace", required=True, help="runs/<name>/<run>.jsonl")
    rph.add_argument("--out", default="runs/rephrased", help="directory for the new trace")
    rph.add_argument(
        "--pack",
        help="knowledge pack the run used; without it the gate trims every surname",
    )
    rph.add_argument(
        "--trust-unchecked",
        action="store_true",
        help="say the pack notes nobody has hand-checked; off by default",
    )
    rph.add_argument(
        "--trust-floor",
        type=float,
        default=SETTINGS.researcher.trust_floor,
        help=(
            "with --trust-unchecked, the lowest confidence an unchecked note may still say "
            f"(default {SETTINGS.researcher.trust_floor:g})"
        ),
    )
    rph.add_argument(
        "--model",
        default=None,
        help=f"phrasing model; defaults to PHRASER_MODEL ({SETTINGS.phraser.model})",
    )
    rph.add_argument(
        "--colour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="offer the colour seat a turn wherever its phase gate allows (default on)",
    )
    rph.add_argument(
        "--colour-model",
        default=None,
        help=f"the colour seat; defaults to COLOUR_MODEL ({SETTINGS.colour.model})",
    )
    rph.set_defaults(func=cmd_rephrase)

    cr = sub.add_parser("crop", help="draw the score-bug box on one frame of a file")
    cr.add_argument("--path", required=True, help="video file")
    cr.add_argument("--at", type=float, default=300.0, help="seconds into the file")
    cr.add_argument("--crop", type=crop_box, default=SETTINGS.board.crop)
    cr.add_argument("--out", default="crop.png")
    cr.set_defaults(func=cmd_crop)

    res = sub.add_parser("research", help="write the pre-match notes (needs a key)")
    res.add_argument("home")
    res.add_argument("away")
    res.add_argument("--competition", default="")
    res.add_argument("--when", default="")
    res.add_argument("--out", help="where to write the pack; defaults under packs/")
    res.add_argument(
        "--max-searches",
        type=int,
        default=8,
        help="how many web searches the pass may make; 0 researches from memory",
    )
    res.set_defaults(func=cmd_research)

    nts = sub.add_parser("notes", help="add spoken-context notes to an existing pack (needs a key)")
    nts.add_argument("--pack", required=True, help="knowledge pack JSON to read")
    nts.add_argument(
        "--out",
        help="where to write it; defaults to <pack>-researched.json, never over the pack",
    )
    nts.add_argument(
        "--max-searches",
        type=int,
        default=8,
        help="how many web searches the pass may make; 0 researches from memory",
    )
    nts.set_defaults(func=cmd_notes)

    caps = sub.add_parser("captions", help="a yt-dlp .en.json3 caption file to a transcript")
    caps.add_argument("path")
    caps.add_argument("--out", default="transcript.json")
    caps.set_defaults(func=cmd_captions)

    fd = sub.add_parser("feed", help="a StatsBomb events file to the feed shape")
    fd.add_argument("path")
    fd.add_argument("--home", required=True, help="team name as StatsBomb spells it")
    fd.add_argument("--away", required=True)
    fd.add_argument("--out", default="feed.json")
    fd.set_defaults(func=cmd_feed)

    reg = sub.add_parser(
        "register", help="score a trace's commentary against real commentary (needs a key)"
    )
    reg.add_argument("traces", nargs="+", help="trace .jsonl files; a rephrased one for the words")
    reg.add_argument("--pack", help="knowledge pack JSON: the roster a name is counted against")
    reg.add_argument(
        "--no-model",
        action="store_true",
        help="the counted layer only. Free, no key, no call, and the one to watch per iteration",
    )
    reg.add_argument("--model", default=JUDGE_MODEL, help="the judge; one call for the whole trace")
    reg.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="seconds to wait for the judge. Grading has no deadline; the runtime's eight is wrong",
    )
    reg.set_defaults(func=cmd_register)

    grade = sub.add_parser("grade", help="score saved traces against StatsBomb")
    grade.add_argument("traces", nargs="+")
    grade.add_argument("--pack", help="knowledge pack JSON: the roster names are checked against")
    grade.add_argument("--statsbomb", help="StatsBomb events JSON; without it, trace metrics only")
    grade.add_argument("--lineups", help="StatsBomb lineups JSON, for the names people are called")
    grade.add_argument("--home", help="home team as StatsBomb spells it; defaults to the pack")
    grade.add_argument("--away", help="away team as StatsBomb spells it; defaults to the pack")
    grade.add_argument("--watched", type=float, default=None, help="seconds of clip watched")
    grade.add_argument(
        "--tolerance", type=float, default=2.0, help="alignment residual to still grade at"
    )
    grade.add_argument(
        "--offset",
        type=float,
        default=None,
        help="video_ts - match_clock_s, measured by hand; for a clip whose board has no clock",
    )
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
    except (LLMError, VoiceUnavailable) as exc:
        # A missing key or a missing optional extra is a thing to fix, not a
        # thing to debug. The message already says what to do, so a traceback
        # only buries it.
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
