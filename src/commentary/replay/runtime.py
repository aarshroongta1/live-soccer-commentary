"""Continuously clocked causal replay and a bounded simulated speech director.

The clock advances while observer and role calls run. Speech timings are estimates,
not TTS/playback measurements; the result explicitly identifies that limitation.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from commentary.recorded_demo import Frame, Observation
from commentary.recorded_research import ResearchBrief
from commentary.replay.contracts import (
    ReplayResult,
    Role,
    RuntimeSnapshot,
    SpeechRecord,
    TraceEvent,
    TurnResult,
)

Observe = Callable[[list[Frame], float], Awaitable[list[Observation]]]
RunRole = Callable[[Role, RuntimeSnapshot], Awaitable[TurnResult]]
Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class ReplayConfig:
    """Durations are replay seconds; ``rate`` controls seconds per wall second."""

    batch_s: float = 2.0
    rate: float = 1.0
    turn_timeout_s: float = 5.0
    observer_timeout_s: float = 5.0
    max_age_s: float = 8.0
    speech_words_per_second: float = 3.0
    drain_s: float = 10.0
    line_gap_s: float = 0.25
    max_batch_frames: int = 8
    max_context_observations: int = 64

    def __post_init__(self) -> None:
        for name in (
            "batch_s",
            "rate",
            "turn_timeout_s",
            "observer_timeout_s",
            "max_age_s",
            "speech_words_per_second",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("drain_s", "line_gap_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.max_batch_frames < 1 or self.max_context_observations < 1:
            raise ValueError("frame and context limits must be positive")


@dataclass
class _ObserverJob:
    task: asyncio.Task[list[Observation]]
    frames: list[Frame]
    cutoff_s: float


@dataclass
class _RoleJob:
    task: asyncio.Task[TurnResult]
    snapshot: RuntimeSnapshot


@dataclass
class _Candidate:
    result: TurnResult
    snapshot: RuntimeSnapshot


class ReplayRuntime:
    """Single-run coordinator with one observer and at most one turn per role.

    ``monotonic`` and ``sleep`` can be replaced together by a manual test clock.
    Call timeouts use asyncio's wall clock; at an accelerated replay rate their
    wall-time budgets are shortened proportionally. Overloaded observation keeps
    only the latest pending batch, rather than accumulating work behind the clock.
    """

    def __init__(
        self,
        observe: Observe,
        run_role: RunRole,
        *,
        research: ResearchBrief | None = None,
        teams: dict[str, str] | None = None,
        config: ReplayConfig | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self.observe = observe
        self.run_role = run_role
        self.research = research.model_copy(deep=True) if research else None
        self.teams = dict(teams or {})
        self.config = config or ReplayConfig()
        self._monotonic = monotonic
        self._sleep = sleep
        self._used = False
        self._start = 0.0
        self._result = ReplayResult()
        self._version = 0
        self._goal_version = 0
        self._latest_goal_ids: set[str] = set()
        self._urgent_goal = False
        self._jobs: dict[Role, _RoleJob] = {}
        self._observer: _ObserverJob | None = None
        self._pending_batch: list[Frame] | None = None
        self._requested: dict[Role, int] = {"caller": -1, "analyst": -1}
        self._queue: dict[Role, _Candidate] = {}
        self._active: SpeechRecord | None = None
        self._next_speech_s = 0.0
        self._last_voice: Role | None = None
        self._availability_cutoff = 0.0

    def _now(self) -> float:
        return max(0.0, (self._monotonic() - self._start) * self.config.rate)

    def _trace(self, event: str, role: Role | None = None, **data: object) -> None:
        self._result.trace.append(TraceEvent(event=event, at_s=self._now(), role=role, data=data))

    def _snapshot(self) -> RuntimeSnapshot:
        # Freeze the input cutoff at the newest completed observation, not at the
        # wall clock: unseen frames cannot become visible just because time passed.
        return RuntimeSnapshot(
            version=self._version,
            cutoff_s=self._availability_cutoff,
            observations=self._result.observations[-self.config.max_context_observations :],
            research=self.research,
            recent_lines=[item.line for item in self._result.speech[-6:]],
            teams=self.teams,
        ).model_copy(deep=True)

    def _start_observer(self, frames: list[Frame]) -> None:
        cutoff_s = frames[-1].at_s

        async def bounded() -> list[Observation]:
            return await asyncio.wait_for(
                self.observe(list(frames), cutoff_s),
                self.config.observer_timeout_s / self.config.rate,
            )

        self._observer = _ObserverJob(asyncio.create_task(bounded()), frames, cutoff_s)
        self._trace(
            "observation_started",
            cutoff_s=cutoff_s,
            frame_timestamps=[frame.at_s for frame in frames],
        )

    def _submit_batch(self, frames: list[Frame]) -> None:
        if len(frames) > self.config.max_batch_frames:
            self._trace("frames_skipped", count=len(frames) - self.config.max_batch_frames)
            frames = frames[-self.config.max_batch_frames :]
        if self._observer is None:
            self._start_observer(frames)
        else:
            if self._pending_batch is not None:
                self._trace("batch_superseded", cutoff_s=self._pending_batch[-1].at_s)
            self._pending_batch = frames

    def _cancel_role(self, role: Role, reason: str) -> None:
        job = self._jobs.pop(role, None)
        if job is not None:
            job.task.cancel()
            self._cancelled.append(job.task)
            self._trace("turn_cancelled", role, reason=reason)
        self._queue.pop(role, None)

    def _observe_done(self) -> None:
        job = self._observer
        if job is None or not job.task.done():
            return
        self._observer = None
        try:
            observations = job.task.result()
            self._validate_observations(observations, job)
        except TimeoutError:
            self._trace("observer_timeout", cutoff_s=job.cutoff_s)
        except Exception as exc:
            self._trace("observer_error", error=type(exc).__name__, message=str(exc)[:240])
        else:
            self._result.observations.extend(item.model_copy(deep=True) for item in observations)
            self._availability_cutoff = job.cutoff_s
            self._version += 1
            self._trace(
                "observation_available",
                cutoff_s=job.cutoff_s,
                snapshot_version=self._version,
                observations=[item.model_dump(mode="json") for item in observations],
            )
            if any(item.kind == "goal" for item in observations):
                self._goal_version = self._version
                self._latest_goal_ids = {item.id for item in observations if item.kind == "goal"}
                self._urgent_goal = True
                self._cancel_role("caller", "new goal evidence")
                self._cancel_role("analyst", "goal priority")
                self._interrupt_for_goal()
        if self._pending_batch is not None:
            pending, self._pending_batch = self._pending_batch, None
            self._start_observer(pending)

    def _validate_observations(self, observations: list[Observation], job: _ObserverJob) -> None:
        known = {item.id for item in self._result.observations}
        previous = self._result.observations[-1].at_s if self._result.observations else -1.0
        for item in observations:
            if item.id in known:
                raise ValueError("duplicate observation ID")
            if (
                not math.isfinite(item.at_s)
                or not job.frames[0].at_s <= item.at_s <= job.cutoff_s
                or item.at_s < previous
            ):
                raise ValueError("observer supplied unordered, future, or out-of-batch evidence")
            known.add(item.id)
            previous = item.at_s

    def _start_roles(self) -> None:
        if not self._result.observations:
            return
        for role in ("caller", "analyst"):
            if role == "analyst" and self._urgent_goal:
                continue
            if role == "caller" and self._is_goal_candidate(self._queue.get("caller")):
                continue
            if role in self._jobs or self._requested[role] >= self._version:
                continue
            snapshot = self._snapshot()
            self._requested[role] = self._version
            role_view = snapshot.model_copy(deep=True)

            async def bounded(
                selected: Role = role, view: RuntimeSnapshot = role_view
            ) -> TurnResult:
                return await asyncio.wait_for(
                    self.run_role(selected, view),
                    self.config.turn_timeout_s / self.config.rate,
                )

            self._jobs[role] = _RoleJob(asyncio.create_task(bounded()), snapshot)
            self._trace(
                "turn_started", role, snapshot_version=snapshot.version, cutoff_s=snapshot.cutoff_s
            )

    def _roles_done(self) -> None:
        for role, job in list(self._jobs.items()):
            if not job.task.done():
                continue
            del self._jobs[role]
            try:
                result = job.task.result()
                self._validate_turn(result, role, job.snapshot)
            except TimeoutError:
                self._trace("turn_timeout", role, snapshot_version=job.snapshot.version)
            except ValueError as exc:
                self._trace("turn_rejected", role, error=type(exc).__name__, message=str(exc)[:240])
            except Exception as exc:
                self._trace("turn_error", role, error=type(exc).__name__, message=str(exc)[:240])
            else:
                self._result.trace.extend(result.trace)
                if self._is_candidate_stale(result, job.snapshot):
                    self._trace("turn_stale", role, snapshot_version=job.snapshot.version)
                elif result.line is None:
                    self._trace("abstained", role, reason=result.reason)
                else:
                    candidate = _Candidate(result, job.snapshot)
                    existing = self._queue.get(role)
                    if self._is_goal_candidate(existing) and not self._is_goal_candidate(candidate):
                        self._trace("speech_preserved", role, reason="queued goal has priority")
                    else:
                        self._queue[role] = candidate
                        self._trace("speech_queued", role, snapshot_version=job.snapshot.version)
            # An unsuccessful urgent turn must not indefinitely silence the analyst.
            if role == "caller" and job.snapshot.version >= self._goal_version:
                self._urgent_goal = False

    def _validate_turn(self, result: TurnResult, role: Role, snapshot: RuntimeSnapshot) -> None:
        if (
            result.role != role
            or result.snapshot_version != snapshot.version
            or result.cutoff_s != snapshot.cutoff_s
        ):
            raise ValueError("turn result does not match its role and originating snapshot")
        line = result.line
        if line is None:
            return
        if line.voice != role:
            raise ValueError("turn used the wrong voice")
        evidence = {item.id for item in snapshot.observations}
        research = {item.id for item in snapshot.research.facts} if snapshot.research else set()
        if not set(line.evidence_ids) <= evidence or not set(line.research_ids) <= research:
            raise ValueError("turn cited unavailable evidence or research")
        if role == "caller" and (not line.evidence_ids or line.research_ids):
            raise ValueError("caller requires visual evidence and cannot cite research")
        if role == "analyst" and not (line.evidence_ids or line.research_ids):
            raise ValueError("analyst requires evidence or research")

    def _is_stale(self, snapshot: RuntimeSnapshot) -> bool:
        return (
            self._now() - snapshot.cutoff_s > self.config.max_age_s
            or snapshot.version < self._goal_version
        )

    def _is_candidate_stale(self, result: TurnResult, snapshot: RuntimeSnapshot) -> bool:
        if self._is_stale(snapshot):
            return True
        if result.role == "caller" and result.line is not None:
            latest = max(
                (
                    item.at_s
                    for item in snapshot.observations
                    if item.id in result.line.evidence_ids
                ),
                default=-math.inf,
            )
            return self._now() - latest > self.config.max_age_s
        return False

    def _is_goal_candidate(self, candidate: _Candidate | None) -> bool:
        return (
            candidate is not None
            and candidate.result.role == "caller"
            and candidate.result.line is not None
            and bool(set(candidate.result.line.evidence_ids) & self._latest_goal_ids)
        )

    def _interrupt_for_goal(self) -> None:
        # Any old speech can obstruct a goal, including a long caller sentence.
        if self._active is not None:
            self._active.ended_s = min(self._active.ended_s, self._now())
            self._active.interrupted = True
            self._trace(
                "speech_interrupted",
                self._active.line.voice,
                speech_id=self._active.id,
                reason="new goal evidence",
            )
            self._active = None
            self._next_speech_s = self._now()

    def _speech(self) -> None:
        now = self._now()
        if self._active is not None:
            if now < self._active.ended_s:
                return
            self._trace("speech_ended", self._active.line.voice, speech_id=self._active.id)
            self._next_speech_s = self._active.ended_s + self.config.line_gap_s
            self._active = None
        if now < self._next_speech_s:
            return
        for role, candidate in list(self._queue.items()):
            if self._is_candidate_stale(candidate.result, candidate.snapshot):
                del self._queue[role]
                self._trace("turn_stale", role, reason="expired before speech")
        roles: tuple[Role, Role] = (
            ("analyst", "caller") if self._last_voice == "caller" else ("caller", "analyst")
        )
        # A caller carrying the newest goal must always precede queued analysis.
        caller = self._queue.get("caller")
        if self._is_goal_candidate(caller):
            roles = ("caller", "analyst")
        for role in roles:
            selected_candidate = self._queue.pop(role, None)
            if selected_candidate is None or selected_candidate.result.line is None:
                continue
            line = selected_candidate.result.line.model_copy(deep=True)
            event_times = [
                item.at_s
                for item in selected_candidate.snapshot.observations
                if item.id in line.evidence_ids
            ]
            record = SpeechRecord(
                id=f"speech-{len(self._result.speech) + 1}",
                line=line,
                snapshot_version=selected_candidate.snapshot.version,
                event_at_s=max(event_times) if event_times else None,
                started_s=now,
                ended_s=now + len(line.text.split()) / self.config.speech_words_per_second,
            )
            self._result.speech.append(record)
            self._active = record
            self._last_voice = role
            self._trace("speech_started", role, speech=record.model_dump(mode="json"))
            return

    @staticmethod
    def _batches(frames: Sequence[Frame], batch_s: float) -> list[list[Frame]]:
        batches: list[list[Frame]] = []
        bucket = -1
        previous = -1.0
        for frame in frames:
            if not math.isfinite(frame.at_s) or frame.at_s < 0 or frame.at_s <= previous:
                raise ValueError(
                    "frame timestamps must be finite, nonnegative, and strictly ordered"
                )
            selected = int(frame.at_s // batch_s)
            if selected != bucket:
                batches.append([])
                bucket = selected
            batches[-1].append(frame)
            previous = frame.at_s
        return batches

    async def run(self, frames: list[Frame], *, duration_s: float | None = None) -> ReplayResult:
        """Replay frames causally. Reusing a runtime is an error, not a state reset."""
        if self._used:
            raise ValueError("ReplayRuntime instances are single-use")
        batches = self._batches(frames, self.config.batch_s)
        if not batches:
            raise ValueError("replay needs at least one frame")
        duration_s = frames[-1].at_s if duration_s is None else duration_s
        if not math.isfinite(duration_s) or duration_s < frames[-1].at_s:
            raise ValueError("duration_s must be finite and at least the last frame timestamp")
        self._used = True
        self._start = self._monotonic()
        self._cancelled: list[asyncio.Task[TurnResult]] = []
        deadline_s = duration_s + self.config.drain_s
        self._result.metadata = {
            "duration_s": duration_s,
            "speech_timing": "simulated",
            "speech_timing_method": "word_count",
            "rate": self.config.rate,
            "deadlines": "replay_seconds",
            "max_inflight_observers": 1,
            "max_inflight_roles": 2,
        }
        index = 0
        timer: asyncio.Task[None] | None = None
        try:
            while True:
                now = self._now()
                if now >= deadline_s:
                    self._trace("replay_deadline", pending_roles=list(self._jobs))
                    break
                while index < len(batches) and batches[index][-1].at_s <= now:
                    self._submit_batch(batches[index])
                    index += 1
                self._observe_done()
                self._roles_done()
                self._start_roles()
                self._speech()
                if (
                    now >= duration_s
                    and index == len(batches)
                    and self._observer is None
                    and not self._jobs
                    and not self._queue
                    and self._active is None
                ):
                    break
                wake_s = deadline_s
                if now < duration_s:
                    wake_s = min(wake_s, duration_s)
                if index < len(batches):
                    wake_s = min(wake_s, batches[index][-1].at_s)
                if self._active is not None:
                    wake_s = min(wake_s, self._active.ended_s)
                elif self._queue:
                    wake_s = min(wake_s, self._next_speech_s)

                async def wake(delay: float = max(0.0, wake_s - now) / self.config.rate) -> None:
                    await self._sleep(delay)

                timer = asyncio.create_task(wake())
                tasks: set[asyncio.Task[object]] = {timer}
                if self._observer is not None:
                    tasks.add(self._observer.task)
                tasks.update(job.task for job in self._jobs.values())
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                if not timer.done():
                    timer.cancel()
                    await asyncio.gather(timer, return_exceptions=True)
                timer = None
        finally:
            pending: list[asyncio.Task[object]] = []
            if timer is not None:
                pending.append(timer)
            if self._observer is not None:
                pending.append(self._observer.task)
            pending.extend(job.task for job in self._jobs.values())
            pending.extend(self._cancelled)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if self._active is not None:
                self._active.ended_s = min(self._active.ended_s, self._now())
                self._active.interrupted = True
                self._trace("speech_interrupted", self._active.line.voice, reason="replay ended")
                self._active = None
        return self._result
