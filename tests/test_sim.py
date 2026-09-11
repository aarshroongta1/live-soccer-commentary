import asyncio

import cv2
import numpy as np
import pytest

from commentary.capture import Frame, FrameSource
from commentary.config import SETTINGS
from commentary.llm.base import LLMError, encode_frame, image_block, text_block
from commentary.perception.board import BoardReader, crop_score_bug
from commentary.schemas import AnalystLine, BoardRead, CallerLine, Event, Scene
from commentary.sim import (
    ERROR_KINDS,
    TS_BLOCKS,
    BroadcastRenderer,
    MatchAudio,
    MatchSim,
    SimOracle,
    SimSource,
    band_energy,
    decode_ts,
    ts_block_size,
)

#: Every capture size the rest of the system actually runs at.
SIZES = ((1280, 720), (640, 360))


def jpeg_round_trip(image: np.ndarray) -> np.ndarray:
    """Put a frame through exactly what a model call does to it."""
    raw = np.frombuffer(encode_frame(image), dtype=np.uint8)
    return np.asarray(cv2.imdecode(raw, cv2.IMREAD_COLOR))


def board_crop(frame: np.ndarray) -> np.ndarray:
    """Exactly what the board reader sends: the real crop, not a lookalike."""
    return crop_score_bug(frame, SETTINGS.board.crop)


def below_strip(region: np.ndarray, width: int) -> np.ndarray:
    return region[ts_block_size(width) :]


def saturated_pixels(region: np.ndarray, threshold: int = 90) -> int:
    """Pixels with a strong colour cast. Replays are washed out; bugs are not."""
    hi = region.max(axis=2).astype(int)
    lo = region.min(axis=2).astype(int)
    return int(((hi - lo) > threshold).sum())


def first_phase(sim: MatchSim, scene: Scene, event: Event | None = None):
    for p in sim.phases:
        if p.scene is scene and (event is None or p.event is event):
            return p
    raise AssertionError(f"no {scene} phase in this script")


def blocks_for(frame: np.ndarray) -> list[dict]:
    return [text_block("what is happening"), image_block(encode_frame(frame))]


# ------------------------------------------------------------- determinism


def test_same_seed_gives_the_same_match():
    a, b = MatchSim(seed=4, duration_s=120.0), MatchSim(seed=4, duration_s=120.0)
    assert a.ground_truth == b.ground_truth
    assert a.whistle_times == b.whistle_times
    assert a.knowledge_pack == b.knowledge_pack
    for ts in (0.0, 3.7, 41.25, 119.9):
        assert a.at(ts) == b.at(ts)


def test_same_seed_gives_identical_pixels():
    a, b = MatchSim(seed=4, duration_s=60.0), MatchSim(seed=4, duration_s=60.0)
    ra = BroadcastRenderer(a.knowledge_pack)
    rb = BroadcastRenderer(b.knowledge_pack)
    for ts in (1.0, 17.5, 33.3):
        assert np.array_equal(ra.frame(a.at(ts)), rb.frame(b.at(ts)))


def test_different_seeds_give_different_matches():
    a, b = MatchSim(seed=4, duration_s=120.0), MatchSim(seed=9, duration_s=120.0)
    assert a.ground_truth != b.ground_truth


def test_the_script_has_a_believable_shape():
    sim = MatchSim(seed=11, duration_s=180.0)
    kinds = [g.event for g in sim.ground_truth]
    assert 2 <= kinds.count(Event.GOAL) <= 4
    assert Event.KICKOFF in kinds
    assert sim.phases[0].start == 0.0
    assert sim.phases[-1].end == pytest.approx(sim.duration_s)
    assert all(g.video_ts <= sim.duration_s for g in sim.ground_truth)
    # Score never walks backwards, and the last goal matches the final board.
    goals = [g for g in sim.ground_truth if g.event is Event.GOAL]
    assert sim.score_at(sim.duration_s) == (
        goals[-1].home_score,
        goals[-1].away_score,
    )


def test_rosters_are_complete_and_numbered():
    pack = MatchSim(seed=2).knowledge_pack
    for team in (pack.home, pack.away):
        assert len(team.squad) == 18
        assert sorted(p.number for p in team.squad if p.number) == list(range(1, 19))
    assert len(MatchSim(seed=2).roster_names) == 36


def test_substitutions_change_who_is_on_the_pitch():
    sim = MatchSim(seed=11, duration_s=180.0)
    subs = [g for g in sim.ground_truth if g.event is Event.SUBSTITUTION]
    assert subs, "every match should put at least one name graphic on screen"
    sub = subs[0]
    before = {d.name for d in sim.at(sub.video_ts - 1.0).players}
    after = {d.name for d in sim.at(sub.video_ts + 1.0).players}
    assert sub.player not in before
    assert sub.player in after


# ------------------------------------------------------------- timestamps


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_timestamp_survives_jpeg_and_the_downscale(width: int, height: int):
    sim = MatchSim(seed=11, duration_s=180.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=width, height=height)
    for ts in (0.0, 7.333, 61.2, 179.867):
        frame = renderer.frame(sim.at(ts))
        assert decode_ts(frame) == pytest.approx(ts, abs=0.001)
        assert decode_ts(jpeg_round_trip(frame)) == pytest.approx(ts, abs=0.001)


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_timestamp_survives_the_board_crop(width: int, height: int):
    """The board reader sends only the crop, so the strip has to live inside it.

    A fixed-pixel strip fits a 720p crop and overflows a 360p one, which is how
    the oracle came to return nothing at all for a whole match of board calls.
    """
    sim = MatchSim(seed=11, duration_s=60.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=width, height=height)
    for ts in (0.0, 12.4, 59.9):
        crop = board_crop(renderer.frame(sim.at(ts)))
        assert ts_block_size(width) * TS_BLOCKS <= crop.shape[1]
        assert decode_ts(crop) == pytest.approx(ts, abs=0.001)
        assert decode_ts(jpeg_round_trip(crop)) == pytest.approx(ts, abs=0.001)
        # And at the quality the board reader itself encodes with.
        sharp = encode_frame(crop, quality=88, max_width=1024)
        assert decode_ts(np.asarray(cv2.imdecode(np.frombuffer(sharp, np.uint8), 1))) == (
            pytest.approx(ts, abs=0.001)
        )


def test_a_frame_too_small_for_a_strip_says_so():
    sim = MatchSim(seed=1, duration_s=5.0)
    BroadcastRenderer(sim.knowledge_pack, width=640, height=360).frame(sim.at(1.0))
    with pytest.raises(ValueError, match="too small"):
        BroadcastRenderer(sim.knowledge_pack, width=160, height=90).frame(sim.at(1.0))


def test_a_frame_without_a_strip_decodes_to_nothing():
    assert decode_ts(np.zeros((720, 1280, 3), dtype=np.uint8)) is None
    assert decode_ts(np.full((720, 1280, 3), 128, dtype=np.uint8)) is None


# ------------------------------------------------------------ the score bug


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_score_bug_is_up_in_live_play_and_gone_in_a_replay(width: int, height: int):
    sim = MatchSim(seed=11, duration_s=180.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=width, height=height)
    live = first_phase(sim, Scene.LIVE_PLAY)
    replay = first_phase(sim, Scene.REPLAY)

    live_frame = renderer.frame(sim.at(live.start + 1.0))
    replay_frame = renderer.frame(sim.at(replay.start + 1.0))
    scale = (width / 1280.0) ** 2

    # Below the timestamp strip, the bug is the only strongly coloured thing
    # in the crop the board reader is handed.
    assert saturated_pixels(below_strip(board_crop(live_frame), width)) > 800 * scale
    assert saturated_pixels(below_strip(board_crop(replay_frame), width)) < 100 * scale
    # The replay says so in the corner the crop never reaches, and says it in
    # colour, which is the one thing the washed-out picture has none of.
    badge = (
        slice(int(24 * width / 1280), int(86 * width / 1280)),
        slice(int(-240 * width / 1280), int(-26 * width / 1280)),
    )
    assert saturated_pixels(replay_frame[badge]) > 5000 * scale
    assert saturated_pixels(live_frame[badge]) < 500 * scale


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_bug_sits_inside_the_configured_crop(width: int, height: int):
    """Whatever the capture resolution, the bug has to be in the box we crop."""
    sim = MatchSim(seed=11, duration_s=60.0)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=width, height=height)
    frame = renderer.frame(sim.at(5.0))
    inside = saturated_pixels(below_strip(board_crop(frame), width))
    whole = saturated_pixels(frame[ts_block_size(width) : int(0.17 * height), :])
    assert inside > 800 * (width / 1280.0) ** 2
    assert inside / whole > 0.3


# ------------------------------------------------------------------ audio


def test_the_crowd_roars_on_a_goal():
    sim = MatchSim(seed=11, duration_s=180.0)
    audio = MatchAudio(sim)
    goal = next(g for g in sim.ground_truth if g.event is Event.GOAL)
    quiet = next(
        ts for ts in np.arange(1.0, sim.duration_s, 0.5) if audio.roar_gain(float(ts)) < 0.05
    )
    assert audio.at(goal.video_ts + 0.5).rms > 4.0 * audio.at(float(quiet)).rms


def test_the_whistle_lands_in_the_band_the_predictor_watches():
    sim = MatchSim(seed=11, duration_s=180.0)
    audio = MatchAudio(sim)
    low, high = SETTINGS.predictor.whistle_band_hz
    foul = next(g for g in sim.ground_truth if g.event is Event.FOUL)
    silent = next(
        ts
        for ts in np.arange(1.0, sim.duration_s, 0.5)
        if all(abs(float(ts) - w) > 1.5 for w in sim.whistle_times)
    )
    blown = band_energy(audio.at(foul.video_ts + 0.1), low, high)
    assert blown > 100.0 * band_energy(audio.at(float(silent)), low, high)


def test_chunks_are_the_shape_the_ring_buffer_expects():
    audio = MatchAudio(MatchSim(seed=3, duration_s=20.0))
    chunk = audio.chunk(7)
    assert chunk.ts == pytest.approx(0.7)
    assert chunk.sample_rate == 16000
    assert chunk.samples.dtype == np.float32
    assert chunk.duration_s == pytest.approx(0.1)
    assert float(np.abs(chunk.samples).max()) <= 1.0


# ----------------------------------------------------------------- source


def test_sim_source_is_a_frame_source():
    source: FrameSource = SimSource(MatchSim(duration_s=5.0))
    assert source is not None


async def test_frames_and_audio_run_off_one_clock():
    sim = MatchSim(seed=6, duration_s=4.0)
    frames = []
    chunks = []

    async with SimSource(sim, realtime=False) as source:

        async def pull_frames() -> None:
            async for frame in source.frames():
                frames.append(frame)

        async def pull_audio() -> None:
            async for chunk in source.audio():
                chunks.append(chunk)

        await asyncio.gather(pull_frames(), pull_audio())

    assert len(frames) == int(4.0 * SETTINGS.capture.fps) + 1
    assert len(chunks) == 41
    assert [f.ts for f in frames] == sorted(f.ts for f in frames)
    assert frames[-1].ts <= sim.duration_s
    assert chunks[-1].ts <= sim.duration_s
    # Every frame knows its own time, which is the whole contract with the oracle.
    for frame in frames[::7]:
        assert decode_ts(frame.image) == pytest.approx(frame.ts, abs=0.002)


async def test_a_source_must_be_entered_before_it_yields():
    source = SimSource(MatchSim(duration_s=2.0))
    with pytest.raises(RuntimeError):
        await anext(source.frames())


# ----------------------------------------------------------------- oracle


def build_oracle(error_rate: float = 0.0, seed: int = 11) -> tuple[SimOracle, BroadcastRenderer]:
    sim = MatchSim(seed=seed, duration_s=180.0)
    return SimOracle(sim, error_rate=error_rate), BroadcastRenderer(sim.knowledge_pack)


async def test_the_oracle_reads_the_moment_off_the_picture():
    oracle, renderer = build_oracle()
    ts = first_phase(oracle.sim, Scene.LIVE_PLAY, Event.BUILD_UP).start + 1.4
    blocks = blocks_for(renderer.frame(oracle.sim.at(ts)))
    assert SimOracle.timestamp_from(blocks) == pytest.approx(ts, abs=0.001)

    read = await oracle.parse(
        model="fake", system="", blocks=blocks, output_format=BoardRead, tag="board"
    )
    state = oracle.sim.at(ts)
    assert isinstance(read.value, BoardRead)
    assert read.value.bug_visible is not state.in_replay
    assert read.value.clock == state.clock
    assert read.value.home_score == state.home_score


@pytest.mark.parametrize(("width", "height"), SIZES)
async def test_the_oracle_answers_a_real_board_reader(width: int, height: int):
    """The board reader sends the crop and nothing else, at whatever size it captures.

    This is the whole path in one test: render, crop, JPEG, back out through
    the oracle. A strip that does not fit the crop leaves every board read of
    a match raising, and the score never moves.
    """
    sim = MatchSim(seed=11, duration_s=180.0)
    oracle = SimOracle(sim)
    renderer = BroadcastRenderer(sim.knowledge_pack, width=width, height=height)
    reader = BoardReader(oracle, SETTINGS.board)

    live = first_phase(sim, Scene.LIVE_PLAY, Event.BUILD_UP).start + 1.4
    read = await reader.read(Frame(ts=live, image=renderer.frame(sim.at(live))))
    truth = sim.at(live)
    assert read.bug_visible
    assert (read.home_score, read.away_score) == (truth.home_score, truth.away_score)
    assert read.clock == truth.clock

    replay = first_phase(sim, Scene.REPLAY).start + 1.0
    gone = await reader.read(Frame(ts=replay, image=renderer.frame(sim.at(replay))))
    assert gone.bug_visible is False
    assert gone.home_score is None


async def test_the_board_read_drops_the_bug_in_a_replay():
    oracle, renderer = build_oracle()
    replay = first_phase(oracle.sim, Scene.REPLAY)
    blocks = blocks_for(renderer.frame(oracle.sim.at(replay.start + 1.0)))
    read = await oracle.parse(
        model="fake", system="", blocks=blocks, output_format=BoardRead, tag="board"
    )
    assert read.value.bug_visible is False
    assert read.value.home_score is None


async def test_the_oracle_fills_in_the_caller_and_analyst_forms():
    oracle, renderer = build_oracle()
    goal = next(g for g in oracle.sim.ground_truth if g.event is Event.GOAL)
    blocks = blocks_for(renderer.frame(oracle.sim.at(goal.video_ts + 0.5)))

    call = await oracle.parse(
        model="fake", system="", blocks=blocks, output_format=CallerLine, tag="caller"
    )
    assert isinstance(call.value, CallerLine)
    assert call.value.speak
    assert call.value.line
    assert len(call.value.line) <= 200

    colour = await oracle.parse(
        model="fake", system="", blocks=blocks, output_format=AnalystLine, tag="analyst"
    )
    assert isinstance(colour.value, AnalystLine)
    assert colour.value.line
    assert colour.value.cites
    assert oracle.total.input_tokens > 0


async def test_the_oracle_refuses_to_guess_without_a_timestamp():
    oracle, _ = build_oracle()
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    with pytest.raises(LLMError):
        await oracle.parse(
            model="fake",
            system="",
            blocks=blocks_for(blank),
            output_format=CallerLine,
            tag="caller",
        )


async def test_injected_errors_are_recorded_and_visible_in_the_line():
    oracle, renderer = build_oracle(error_rate=1.0)
    roster = oracle.sim.roster_names
    lines = []
    for ts in np.arange(5.0, 120.0, 3.0):
        blocks = blocks_for(renderer.frame(oracle.sim.at(float(ts))))
        result = await oracle.parse(
            model="fake", system="", blocks=blocks, output_format=CallerLine, tag="caller"
        )
        lines.append(result.value)

    assert len(oracle.injected) == len(lines)
    kinds = {kind for _, kind in oracle.injected}
    assert kinds <= set(ERROR_KINDS)
    assert len(kinds) == 3, "all three failure modes should appear over forty calls"

    counts = oracle.injected_kinds
    assert sum(counts.values()) == len(oracle.injected)

    # The lies have to be visible in the output, or the gate has nothing to catch.
    fake_named = [
        line
        for line in lines
        if any(name not in roster and not name.isdigit() for name in line.names_read)
    ]
    assert len(fake_named) == counts["fake_name"]
    assert any(line.event is Event.GOAL for line in lines)


async def test_no_lies_when_the_error_rate_is_zero():
    oracle, renderer = build_oracle(error_rate=0.0)
    roster = oracle.sim.roster_names
    for ts in np.arange(5.0, 60.0, 2.5):
        blocks = blocks_for(renderer.frame(oracle.sim.at(float(ts))))
        result = await oracle.parse(
            model="fake", system="", blocks=blocks, output_format=CallerLine, tag="caller"
        )
        for name in result.value.names_read:
            assert name.isdigit() or name in roster
    assert oracle.injected == []
