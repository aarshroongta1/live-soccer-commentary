"""A small, versioned boundary around the mutable match facts."""

from __future__ import annotations

from pydantic import BaseModel

from commentary.schemas import CallerLine, KnowledgePack, MatchState, WireEvent
from commentary.state import ConfirmedBoard, EntityRegistry, MatchStateTracker


class FactSnapshot(BaseModel):
    """Serializable facts supplied to one orchestration turn."""

    version: int
    cursor_s: float
    summary: str
    state: MatchState


class MatchFactStore:
    """Own the match tracker and expose versioned, isolated snapshots."""

    def __init__(self, tracker: MatchStateTracker) -> None:
        self._tracker = tracker
        self._version = 0

    @classmethod
    def from_pack(cls, pack: KnowledgePack) -> MatchFactStore:
        return cls(MatchStateTracker.from_pack(pack))

    @property
    def version(self) -> int:
        return self._version

    @property
    def state(self) -> MatchState:
        return self._tracker.state

    @property
    def registry(self) -> EntityRegistry:
        return self._tracker.registry

    def summary(self, cursor_s: float) -> str:
        self._advance_to(cursor_s)
        return self._tracker.summary(cursor_s)

    def snapshot(self, cursor_s: float) -> FactSnapshot:
        summary = self.summary(cursor_s)
        return FactSnapshot(
            version=self._version,
            cursor_s=cursor_s,
            summary=summary,
            state=self.state.model_copy(deep=True),
        )

    def apply_board(self, board: ConfirmedBoard) -> None:
        before = self.state.model_copy(deep=True)
        self._tracker.apply_board(board)
        self._note_change(before)

    def apply_caller(self, line: CallerLine, cursor_s: float) -> None:
        before = self.state.model_copy(deep=True)
        self._tracker.apply_caller(line, cursor_s)
        self._note_change(before)

    def apply_wire(self, event: WireEvent, cursor_s: float) -> str | None:
        before = self.state.model_copy(deep=True)
        result = self._tracker.apply_wire(event, cursor_s)
        self._note_change(before)
        return result

    def mark_evidence(self) -> None:
        """Record new verification evidence that does not alter match state."""
        self._version += 1

    def _advance_to(self, cursor_s: float) -> None:
        if self._tracker.advance_to(cursor_s):
            self._version += 1

    def _note_change(self, before: MatchState) -> None:
        if self.state != before:
            self._version += 1
