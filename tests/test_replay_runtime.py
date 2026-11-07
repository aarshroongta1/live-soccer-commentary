"""Causal state, concurrency, and speech scheduling without providers or TTS."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from commentary.recorded_demo import Frame, Observation
from commentary.replay.contracts import Role, RuntimeSnapshot, SpeechLine, TurnResult
from commentary.replay.evaluation import EvaluationCase, ReferenceSource, evaluate_replay
from commentary.replay.runtime import ReplayConfig, ReplayRuntime


@dataclass
class Clock:
    now: float = 0.0
    sleepers: list[tuple[float, asyncio.Future[None]]] = field(default_factory=list)

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        if delay <= 0:
            await asyncio.sleep(0)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.sleepers.append((self.now + delay, future))
        await future

    def advance(self, value: float) -> None:
        self.now = value
        for at, future in self.sleepers:
            if at <= value and not future.done():
                future.set_result(None)
        self.sleepers = [(at, f) for at, f in self.sleepers if not f.done()]


async def settle() -> None:
    for _ in range(16):
        await asyncio.sleep(0)


async def drive(task: asyncio.Task[object], clock: Clock, *, until: float = 30) -> None:
    while not task.done() and clock.now < until:
        await settle()
        clock.advance(round(clock.now + 0.25, 3))
    await settle()
    assert task.done(), "replay did not settle by its deterministic deadline"


def speech(role: Role, snapshot: RuntimeSnapshot, text: str = "A useful call.") -> TurnResult:
    return TurnResult(
        role=role,
        snapshot_version=snapshot.version,
        cutoff_s=snapshot.cutoff_s,
        line=SpeechLine(voice=role, text=text, evidence_ids=[snapshot.observations[-1].id]),
    )


async def observe(frames: list[Frame], cutoff: float) -> list[Observation]:
    return [Observation(id=f"event-{cutoff}", at_s=cutoff, kind="play", description="A pass.")]


@pytest.mark.asyncio
async def test_independent_roles_and_nonoverlapping_simulated_speech() -> None:
    clock = Clock()
    roles: list[Role] = []

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        roles.append(role)
        return speech(role, snapshot, f"{role} has a useful point.")

    runtime = ReplayRuntime(observe, run_role, monotonic=clock.monotonic, sleep=clock.sleep)
    task = asyncio.create_task(runtime.run([Frame(0, b"0")], duration_s=3))
    await drive(task, clock)
    result = task.result()
    assert roles == ["caller", "analyst"]
    assert [item.line.voice for item in result.speech] == ["caller", "analyst"]
    assert result.speech[0].ended_s <= result.speech[1].started_s
    assert all(not item.interrupted for item in result.speech)
    assert result.metadata["speech_timing"] == "simulated"
    assert result.metadata["duration_s"] == 3
    assert [item.event_at_s for item in result.speech] == [0, 0]


@pytest.mark.asyncio
async def test_concurrent_roles_cannot_repeat_same_words_for_same_event() -> None:
    clock = Clock()
    contexts: list[RuntimeSnapshot] = []

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        contexts.append(snapshot)
        text = "The pass goes forward." if role == "caller" else "  THE pass goes forward!"
        return speech(role, snapshot, text)

    runtime = ReplayRuntime(observe, run_role, monotonic=clock.monotonic, sleep=clock.sleep)
    task = asyncio.create_task(runtime.run([Frame(0, b"0")], duration_s=3))
    await drive(task, clock)
    result = task.result()
    assert len(contexts) == 2
    assert all(snapshot.recent_lines == [] for snapshot in contexts)
    assert len(result.speech) == 1
    assert result.speech[0].line.text == "The pass goes forward."
    duplicate = next(event for event in result.trace if event.event == "speech_duplicate")
    assert duplicate.role == "analyst"
    assert duplicate.data["duplicate_of"] == result.speech[0].id


@pytest.mark.asyncio
async def test_identical_goal_words_are_allowed_for_distinct_goal_evidence() -> None:
    clock = Clock()

    async def goals(frames: list[Frame], cutoff: float) -> list[Observation]:
        return [
            Observation(
                id=f"goal-{cutoff}", at_s=cutoff, kind="goal", description="Ball crosses the line."
            )
        ]

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        if role == "analyst":
            return TurnResult(
                role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s
            )
        return speech(role, snapshot, "Goal!")

    runtime = ReplayRuntime(goals, run_role, monotonic=clock.monotonic, sleep=clock.sleep)
    task = asyncio.create_task(runtime.run([Frame(0, b"0"), Frame(4, b"4")], duration_s=5))
    await drive(task, clock)
    result = task.result()
    assert [item.line.text for item in result.speech] == ["Goal!", "Goal!"]
    assert [item.line.evidence_ids for item in result.speech] == [["goal-0"], ["goal-4"]]
    assert not any(event.event == "speech_duplicate" for event in result.trace)


@pytest.mark.asyncio
async def test_clock_advances_while_roles_work_and_snapshots_are_isolated() -> None:
    clock = Clock()
    release = asyncio.Event()
    snapshots: list[tuple[Role, RuntimeSnapshot]] = []

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        snapshots.append((role, snapshot))
        if snapshot.version == 1:
            if role == "caller":
                snapshot.observations[0].description = "MUTATED"
            await release.wait()
        return speech(role, snapshot)

    runtime = ReplayRuntime(observe, run_role, monotonic=clock.monotonic, sleep=clock.sleep)
    task = asyncio.create_task(runtime.run([Frame(0, b"0"), Frame(2, b"2")]))
    await settle()
    clock.advance(2)
    await settle()
    assert len(runtime._result.observations) == 2
    assert runtime._result.observations[0].description == "A pass."
    assert all(item.at_s <= snap.cutoff_s for _, snap in snapshots for item in snap.observations)
    analyst = next(snap for role, snap in snapshots if role == "analyst")
    assert analyst.observations[0].description == "A pass."
    release.set()
    await drive(task, clock)
    result = task.result()
    assert any(event.event == "observation_available" and event.at_s == 2 for event in result.trace)


@pytest.mark.asyncio
async def test_future_and_duplicate_observations_never_enter_role_context() -> None:
    clock = Clock()
    seen: list[RuntimeSnapshot] = []

    async def invalid_observer(frames: list[Frame], cutoff: float) -> list[Observation]:
        if cutoff == 0:
            return [Observation(id="same", at_s=0, kind="play", description="Visible.")]
        if cutoff == 2:
            return [Observation(id="future", at_s=5, kind="goal", description="Unseen.")]
        return [Observation(id="same", at_s=cutoff, kind="play", description="Duplicate.")]

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        seen.append(snapshot)
        return TurnResult(role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s)

    runtime = ReplayRuntime(
        invalid_observer, run_role, monotonic=clock.monotonic, sleep=clock.sleep
    )
    task = asyncio.create_task(runtime.run([Frame(t, b"frame") for t in (0, 2, 4)]))
    await drive(task, clock)
    result = task.result()
    assert [item.id for item in result.observations] == ["same"]
    assert all([item.id for item in snap.observations] == ["same"] for snap in seen)
    assert sum(event.event == "observation_rejected" for event in result.trace) == 2


@pytest.mark.asyncio
async def test_observer_validation_rejection_and_provider_error_have_separate_metrics() -> None:
    clock = Clock()

    async def invalid_or_failed_observer(frames: list[Frame], cutoff: float) -> list[Observation]:
        if cutoff == 0:
            return [Observation(id="future", at_s=1, kind="goal", description="Not yet visible.")]
        raise RuntimeError("simulated provider failure")

    async def never_called(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        raise AssertionError("invalid observer output must not reach a role")

    runtime = ReplayRuntime(
        invalid_or_failed_observer, never_called, monotonic=clock.monotonic, sleep=clock.sleep
    )
    task = asyncio.create_task(runtime.run([Frame(0, b"0"), Frame(2, b"2")], duration_s=3))
    await drive(task, clock)
    result = task.result()
    reference = EvaluationCase(
        id="observer-failure-diagnostic",
        match_id="fictional",
        split="development",
        duration_s=3,
        source=ReferenceSource(
            kind="synthetic",
            locator="synthetic://observer-failures",
            provenance="Fictional runtime failure test.",
            usage_terms="Project-owned test fixture.",
            annotation_status="synthetic",
        ),
    )
    score = evaluate_replay(result, reference)
    assert result.observations == []
    assert result.speech == []
    assert score["execution"]["rejected_observation_batches"] == 1
    assert score["execution"]["errors"] == 1
    assert score["execution"]["rejected_turns"] == 0


@pytest.mark.asyncio
async def test_goal_preempts_analyst_and_cancels_obsolete_turns() -> None:
    clock = Clock()
    stalled = asyncio.Event()
    cancelled: list[Role] = []

    async def goal_observer(frames: list[Frame], cutoff: float) -> list[Observation]:
        return [
            Observation(
                id=f"e-{cutoff}",
                at_s=cutoff,
                kind="goal" if cutoff == 2 else "play",
                description="Ball crosses the line." if cutoff == 2 else "A pass.",
            )
        ]

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        if role == "caller" and snapshot.version == 1:
            try:
                await stalled.wait()
            except asyncio.CancelledError:
                cancelled.append(role)
                raise
        if role == "analyst" and snapshot.version == 1:
            return speech(role, snapshot, "This analyst has a long explanation of the build up.")
        return speech(role, snapshot, "Goal!" if role == "caller" else "The move worked well.")

    runtime = ReplayRuntime(
        goal_observer,
        run_role,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        config=ReplayConfig(speech_words_per_second=1),
    )
    task = asyncio.create_task(runtime.run([Frame(0, b"0"), Frame(2, b"2")]))
    await drive(task, clock)
    result = task.result()
    assert cancelled == ["caller"]
    assert result.speech[0].line.voice == "analyst"
    assert result.speech[0].interrupted
    assert result.speech[0].ended_s == 2
    assert result.speech[1].line.voice == "caller"
    assert result.speech[1].line.evidence_ids == ["e-2"]
    assert result.speech[0].ended_s <= result.speech[1].started_s


@pytest.mark.asyncio
async def test_late_outputs_and_wrong_origin_are_rejected() -> None:
    clock = Clock()
    release = asyncio.Event()

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        if role == "caller":
            await release.wait()
            return speech(role, snapshot)
        return TurnResult(
            role=role,
            snapshot_version=snapshot.version + 1,
            cutoff_s=snapshot.cutoff_s,
            line=SpeechLine(voice=role, text="Wrong snapshot.", evidence_ids=["event-0"]),
        )

    runtime = ReplayRuntime(
        observe,
        run_role,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        config=ReplayConfig(max_age_s=1),
    )
    task = asyncio.create_task(runtime.run([Frame(0, b"0")]))
    await settle()
    clock.advance(2)
    release.set()
    await drive(task, clock)
    result = task.result()
    assert result.speech == []
    assert {event.event for event in result.trace} >= {"turn_stale", "turn_rejected"}


@pytest.mark.asyncio
async def test_goal_interrupts_old_caller_and_survives_subsequent_play() -> None:
    clock = Clock()

    async def goal_observer(frames: list[Frame], cutoff: float) -> list[Observation]:
        return [
            Observation(
                id=f"e-{cutoff}",
                at_s=cutoff,
                kind="goal" if cutoff == 2 else "play",
                description="Visible action.",
            )
        ]

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        if role == "analyst":
            return TurnResult(
                role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s
            )
        if snapshot.cutoff_s == 0:
            return speech(
                role, snapshot, "The caller starts a lengthy explanation of the passing move."
            )
        if snapshot.cutoff_s == 2:
            return speech(role, snapshot, "Goal for the home side!")
        return speech(role, snapshot, "Another pass goes forward.")

    runtime = ReplayRuntime(
        goal_observer,
        run_role,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        config=ReplayConfig(speech_words_per_second=1, max_age_s=6),
    )
    task = asyncio.create_task(runtime.run([Frame(t, b"frame") for t in (0, 2, 4)]))
    await drive(task, clock)
    result = task.result()
    first, goal = result.speech[:2]
    assert first.line.voice == "caller"
    assert first.interrupted and first.ended_s == 2
    assert goal.line.text == "Goal for the home side!"
    assert goal.started_s == 2
    assert not goal.interrupted
    assert goal.ended_s <= result.speech[2].started_s


@pytest.mark.asyncio
async def test_empty_observer_batches_cannot_refresh_an_old_caller_event() -> None:
    clock = Clock()

    async def sparse_observer(frames: list[Frame], cutoff: float) -> list[Observation]:
        return await observe(frames, cutoff) if cutoff == 0 else []

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        if snapshot.cutoff_s == 0 or role == "analyst":
            return TurnResult(
                role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s
            )
        return speech(role, snapshot)

    runtime = ReplayRuntime(
        sparse_observer,
        run_role,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
        config=ReplayConfig(max_age_s=1),
    )
    task = asyncio.create_task(runtime.run([Frame(t, b"f") for t in (0, 2)]))
    await drive(task, clock)
    result = task.result()
    assert result.speech == []
    assert any(event.event == "turn_stale" for event in result.trace)


@pytest.mark.asyncio
async def test_unavailable_citations_rejected_by_coordinator() -> None:
    clock = Clock()

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        result = speech(role, snapshot)
        assert result.line is not None
        result.line.evidence_ids = ["future-id"]
        return result

    runtime = ReplayRuntime(observe, run_role, monotonic=clock.monotonic, sleep=clock.sleep)
    task = asyncio.create_task(runtime.run([Frame(0, b"0")]))
    await drive(task, clock)
    result = task.result()
    assert result.speech == []
    assert sum(event.event == "turn_rejected" for event in result.trace) == 2


@pytest.mark.asyncio
async def test_timeout_is_bounded_and_tasks_are_cleaned_up() -> None:
    cancelled: list[Role] = []

    async def run_role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(role)
            raise
        raise AssertionError("unreachable")

    runtime = ReplayRuntime(observe, run_role, config=ReplayConfig(turn_timeout_s=0.01))
    result = await runtime.run([Frame(0, b"0")])
    assert sorted(cancelled) == ["analyst", "caller"]
    assert sum(event.event == "turn_timeout" for event in result.trace) == 2
    assert result.speech == []


@pytest.mark.asyncio
async def test_slow_observer_coalesces_pending_frames_instead_of_losing_the_finish() -> None:
    clock = Clock()
    release = asyncio.Event()
    observed: list[list[float]] = []

    async def slow_observer(frames: list[Frame], cutoff: float) -> list[Observation]:
        observed.append([frame.at_s for frame in frames])
        if cutoff == 0:
            await release.wait()
        return [
            Observation(
                id=f"e-{frame.at_s}",
                at_s=frame.at_s,
                kind="goal" if frame.at_s == 2 else "celebration",
                description="The ball is in the net." if frame.at_s == 2 else "Aftermath.",
            )
            for frame in frames
        ]

    async def silent(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        return TurnResult(role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s)

    runtime = ReplayRuntime(slow_observer, silent, monotonic=clock.monotonic, sleep=clock.sleep)
    task = asyncio.create_task(runtime.run([Frame(t, b"f") for t in (0, 2, 4, 6)]))
    await settle()
    for t in (2, 4, 6):
        clock.advance(t)
        await settle()
    release.set()
    await drive(task, clock)
    result = task.result()
    assert observed == [[0], [2, 4, 6]]
    assert any(item.kind == "goal" and item.at_s == 2 for item in result.observations)
    assert sum(event.event == "batch_coalesced" for event in result.trace) == 2
    available = [event for event in result.trace if event.event == "observation_available"]
    assert available[0].data["cutoff_s"] == 0
    assert available[0].at_s >= 6


async def test_coalescing_stays_bounded_and_preserves_time_coverage() -> None:
    clock = Clock()
    release = asyncio.Event()
    batches: list[list[float]] = []

    async def slow(frames: list[Frame], cutoff: float) -> list[Observation]:
        batches.append([frame.at_s for frame in frames])
        if cutoff == 0:
            await release.wait()
        return []

    async def silent(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        return TurnResult(role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s)

    runtime = ReplayRuntime(
        slow,
        silent,
        config=ReplayConfig(max_batch_frames=4),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    task = asyncio.create_task(runtime.run([Frame(float(t), b"f") for t in range(0, 20, 2)]))
    await settle()
    for t in range(2, 20, 2):
        clock.advance(float(t))
        await settle()
    release.set()
    await drive(task, clock, until=40)
    assert batches[1][0] == 2
    assert batches[1][-1] == 18
    assert len(batches[1]) == 4
    assert all(len(batch) <= 4 for batch in batches)
    assert any(e.event == "frames_skipped" for e in task.result().trace)


async def test_late_routine_call_is_not_spoken_after_visible_celebration() -> None:
    clock = Clock()

    async def phases(frames: list[Frame], cutoff: float) -> list[Observation]:
        return [
            Observation(
                id=f"phase-{cutoff}",
                at_s=cutoff,
                kind="play" if cutoff == 0 else "celebration",
                description="Attack." if cutoff == 0 else "Celebration.",
            )
        ]

    async def role(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        if snapshot.version == 1:
            if role == "analyst":
                return speech(role, snapshot, "There is space to attack through the middle.")
            await clock.sleep(1)
            return speech(role, snapshot, "The attack is moving forward now.")
        return TurnResult(role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s)

    runtime = ReplayRuntime(
        phases,
        role,
        config=ReplayConfig(max_age_s=30),
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )
    task = asyncio.create_task(runtime.run([Frame(0, b"a"), Frame(2, b"b")], duration_s=4))
    await drive(task, clock)
    result = task.result()
    assert not any(s.line.text == "The attack is moving forward now." for s in result.speech)
    assert any(e.event == "turn_stale" and e.role == "caller" for e in result.trace)


@pytest.mark.asyncio
async def test_external_cancellation_cleans_up_role_tasks() -> None:
    clock = Clock()
    cancelled: list[Role] = []

    async def stall(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(role)
            raise
        raise AssertionError("unreachable")

    runtime = ReplayRuntime(observe, stall, monotonic=clock.monotonic, sleep=clock.sleep)
    task = asyncio.create_task(runtime.run([Frame(0, b"0")]))
    await settle()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(cancelled) == ["analyst", "caller"]


@pytest.mark.asyncio
async def test_bad_inputs_and_runtime_reuse_are_explicit_errors() -> None:
    async def silent(role: Role, snapshot: RuntimeSnapshot) -> TurnResult:
        return TurnResult(role=role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s)

    with pytest.raises(ValueError, match="positive and finite"):
        ReplayConfig(rate=float("nan"))
    runtime = ReplayRuntime(observe, silent)
    with pytest.raises(ValueError, match="strictly ordered"):
        await runtime.run([Frame(1, b"1"), Frame(0, b"0")])
    with pytest.raises(ValueError, match="duration_s"):
        await runtime.run([Frame(1, b"1")], duration_s=0)
    await runtime.run([Frame(0, b"0")])
    with pytest.raises(ValueError, match="single-use"):
        await runtime.run([Frame(0, b"0")])
