"""A short, structured history of the live move leading to a goal.

The caller observes overlapping windows, so the same touch can arrive more
than once and with different amounts of detail.  :class:`MoveBuffer` keeps a
small fact-only window, merges those adjacent duplicates, and freezes a
recursively immutable snapshot when the live caller reports a goal.

This module intentionally has no runtime dependency.  The ordered ingest path
can feed it after caller/identity resolution, then hand ``frozen`` to the goal
call and follow-up planners.  Replays, close-ups, and every other non-live
scene are read-only from the buffer's point of view.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from commentary.schemas import (
    Action,
    ActionBeat,
    CallerLine,
    Event,
    IdentitySource,
    PlayerIdentity,
    Scene,
    Side,
)

DEFAULT_WINDOW_S = 12.0
DEFAULT_DUPLICATE_GAP_S = 1.5
DEFAULT_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class MoveIdentity:
    """The immutable identity evidence retained with one move beat."""

    name: str | None
    number: int | None
    role: str | None
    side: Side
    confidence: float
    source: IdentitySource
    evidence: str | None

    @classmethod
    def from_player(cls, player: PlayerIdentity) -> MoveIdentity:
        return cls(
            name=player.name,
            number=player.number,
            role=player.role,
            side=player.side,
            confidence=player.confidence,
            source=player.source,
            evidence=player.evidence,
        )

    def as_player_identity(self) -> PlayerIdentity:
        """Return a fresh mutable schema value for a downstream prompt."""
        return PlayerIdentity(
            name=self.name,
            number=self.number,
            role=self.role,
            side=self.side,
            confidence=self.confidence,
            source=self.source,
            evidence=self.evidence,
        )


@dataclass(frozen=True)
class MoveBeat:
    """A recursively immutable action fact held by a move snapshot."""

    frame_index: int | None
    video_ts: float
    action: Action
    actor: MoveIdentity | None
    target: MoveIdentity | None
    origin_zone: str | None
    destination_zone: str | None
    direction: str | None
    delivery: str | None
    body_part: str | None
    outcome: str | None
    confidence: float
    evidence: str | None

    @classmethod
    def from_action(cls, beat: ActionBeat, *, fallback_ts: float) -> MoveBeat:
        return cls(
            frame_index=beat.frame_index,
            video_ts=beat.video_ts if beat.video_ts is not None else fallback_ts,
            action=beat.action,
            actor=MoveIdentity.from_player(beat.actor) if beat.actor is not None else None,
            target=MoveIdentity.from_player(beat.target) if beat.target is not None else None,
            origin_zone=beat.origin_zone,
            destination_zone=beat.destination_zone,
            direction=beat.direction,
            delivery=beat.delivery,
            body_part=beat.body_part,
            outcome=beat.outcome,
            confidence=beat.confidence,
            evidence=beat.evidence,
        )

    def as_action_beat(self) -> ActionBeat:
        """Return a fresh schema value for existing caller/planner code."""
        return ActionBeat(
            frame_index=self.frame_index,
            video_ts=self.video_ts,
            action=self.action,
            actor=self.actor.as_player_identity() if self.actor is not None else None,
            target=self.target.as_player_identity() if self.target is not None else None,
            origin_zone=self.origin_zone,
            destination_zone=self.destination_zone,
            direction=self.direction,
            delivery=self.delivery,
            body_part=self.body_part,
            outcome=self.outcome,
            confidence=self.confidence,
            evidence=self.evidence,
        )


@dataclass(frozen=True)
class MoveSnapshot:
    """The stable fact sequence exposed to goal and follow-up planning."""

    side: Side
    team: str | None
    beats: tuple[MoveBeat, ...]
    goal_ts: float | None = None

    @property
    def started_at(self) -> float | None:
        return self.beats[0].video_ts if self.beats else None

    @property
    def ended_at(self) -> float | None:
        return self.beats[-1].video_ts if self.beats else None

    @property
    def actions(self) -> tuple[ActionBeat, ...]:
        """Fresh ``ActionBeat`` copies for the current schema integration seam."""
        return tuple(beat.as_action_beat() for beat in self.beats)


class MoveBuffer:
    """Retain and freeze the useful live actions in one short possession.

    ``observe`` is intended to run only after concurrent caller results have
    been restored to cursor order.  A known change of side starts a fresh
    move.  A live goal includes any actions on its own form, freezes the
    sequence, and clears the mutable live buffer.
    """

    def __init__(
        self,
        *,
        window_s: float = DEFAULT_WINDOW_S,
        duplicate_gap_s: float = DEFAULT_DUPLICATE_GAP_S,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if not 10.0 <= window_s <= 15.0:
            raise ValueError("move window must be between 10 and 15 seconds")
        if duplicate_gap_s < 0.0:
            raise ValueError("duplicate gap cannot be negative")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("minimum confidence must be between zero and one")
        self.window_s = window_s
        self.duplicate_gap_s = duplicate_gap_s
        self.min_confidence = min_confidence
        self._side = Side.UNKNOWN
        self._team: str | None = None
        self._beats: list[MoveBeat] = []
        self._frozen: MoveSnapshot | None = None

    @property
    def current(self) -> MoveSnapshot | None:
        """Return an immutable view of the live move, if it has useful facts."""
        if not self._beats:
            return None
        return MoveSnapshot(side=self._side, team=self._team, beats=tuple(self._beats))

    @property
    def frozen(self) -> MoveSnapshot | None:
        """The last move frozen by a live goal call."""
        return self._frozen

    def observe(self, ts: float, line: CallerLine) -> MoveSnapshot | None:
        """Consume one ordered caller form and return a newly frozen move.

        Non-live forms return immediately without pruning, changing possession,
        or reacting to a goal tag.  This makes replay enrichment safe to run
        later without rewriting the live evidence it is meant to enrich.
        """
        if line.scene is not Scene.LIVE_PLAY:
            return None

        if (
            line.side is not Side.UNKNOWN
            and self._side is not Side.UNKNOWN
            and line.side is not self._side
        ):
            self._clear_live()

        if line.side is not Side.UNKNOWN:
            self._side = line.side
        if line.team is not None:
            self._team = line.team

        self._prune(ts)
        incoming = sorted(
            (
                MoveBeat.from_action(beat, fallback_ts=ts)
                for beat in line.actions
                if beat.confidence >= self.min_confidence
            ),
            key=lambda beat: beat.video_ts,
        )
        for beat in incoming:
            self._append_or_merge(beat)
        self._prune(ts)

        if line.event is Event.GOAL:
            return self.freeze(ts)
        return None

    def freeze(self, goal_ts: float) -> MoveSnapshot:
        """Freeze the active facts for an incident and start a blank live move."""
        snapshot = MoveSnapshot(
            side=self._side,
            team=self._team,
            beats=tuple(self._beats),
            goal_ts=goal_ts,
        )
        self._frozen = snapshot
        self._clear_live()
        return snapshot

    def _clear_live(self) -> None:
        self._beats.clear()
        self._side = Side.UNKNOWN
        self._team = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        self._beats = [beat for beat in self._beats if beat.video_ts >= cutoff]

    def _append_or_merge(self, beat: MoveBeat) -> None:
        # Overlapping caller windows can repeat a whole pass -> receive pair.
        # The matching pass is then not the last buffered item, so search the
        # short window backwards instead of comparing only the tail.
        for index in range(len(self._beats) - 1, -1, -1):
            existing = self._beats[index]
            if existing.video_ts < beat.video_ts - self.duplicate_gap_s:
                break
            if _duplicates(existing, beat, self.duplicate_gap_s):
                self._beats[index] = _merge(existing, beat)
                self._beats.sort(key=lambda item: item.video_ts)
                return
        self._beats.append(beat)
        self._beats.sort(key=lambda item: item.video_ts)


def _duplicates(left: MoveBeat, right: MoveBeat, max_gap: float) -> bool:
    """Whether adjacent observations describe the same touch.

    Action kind is an absolute boundary: pass → receive → cross → shot →
    finish can never collapse.  Within one kind, contradictions reject a
    merge and at least one concrete anchor must agree.
    """
    if left.action is not right.action or abs(right.video_ts - left.video_ts) > max_gap:
        return False
    if _identity_conflicts(left.actor, right.actor) or _identity_conflicts(
        left.target, right.target
    ):
        return False
    if abs(right.video_ts - left.video_ts) <= 0.35:
        return True
    return any(
        (
            _same_identity(left.actor, right.actor),
            _same_identity(left.target, right.target),
            _same_route(left, right),
            _same_text(left.evidence, right.evidence),
        )
    )


def _merge(left: MoveBeat, right: MoveBeat) -> MoveBeat:
    preferred, other = (right, left) if right.confidence > left.confidence else (left, right)
    return MoveBeat(
        frame_index=left.frame_index if left.video_ts <= right.video_ts else right.frame_index,
        video_ts=min(left.video_ts, right.video_ts),
        action=left.action,
        actor=_merge_identity(left.actor, right.actor),
        target=_merge_identity(left.target, right.target),
        origin_zone=_detail(preferred.origin_zone, other.origin_zone),
        destination_zone=_detail(preferred.destination_zone, other.destination_zone),
        direction=_detail(preferred.direction, other.direction),
        delivery=_detail(preferred.delivery, other.delivery),
        body_part=_detail(preferred.body_part, other.body_part),
        outcome=_detail(preferred.outcome, other.outcome),
        confidence=max(left.confidence, right.confidence),
        evidence=_richer(left.evidence, right.evidence),
    )


def _merge_identity(left: MoveIdentity | None, right: MoveIdentity | None) -> MoveIdentity | None:
    if left is None:
        return right
    if right is None:
        return left
    preferred, other = (right, left) if right.confidence > left.confidence else (left, right)
    return MoveIdentity(
        name=preferred.name or other.name,
        number=preferred.number if preferred.number is not None else other.number,
        role=preferred.role or other.role,
        side=preferred.side if preferred.side is not Side.UNKNOWN else other.side,
        confidence=max(left.confidence, right.confidence),
        source=(
            preferred.source if preferred.source is not IdentitySource.UNKNOWN else other.source
        ),
        evidence=_richer(left.evidence, right.evidence),
    )


def _identity_conflicts(left: MoveIdentity | None, right: MoveIdentity | None) -> bool:
    if left is None or right is None:
        return False
    if (
        left.side is not Side.UNKNOWN
        and right.side is not Side.UNKNOWN
        and left.side is not right.side
    ):
        return True
    if left.number is not None and right.number is not None and left.number != right.number:
        return True
    return bool(left.name and right.name and _normalise(left.name) != _normalise(right.name))


def _same_identity(left: MoveIdentity | None, right: MoveIdentity | None) -> bool:
    if left is None or right is None or _identity_conflicts(left, right):
        return False
    if left.name and right.name:
        return _normalise(left.name) == _normalise(right.name)
    if left.number is not None and right.number is not None:
        return left.number == right.number
    return bool(left.role and right.role and _normalise(left.role) == _normalise(right.role))


def _same_route(left: MoveBeat, right: MoveBeat) -> bool:
    pairs = (
        (left.origin_zone, right.origin_zone),
        (left.destination_zone, right.destination_zone),
        (left.direction, right.direction),
        (left.delivery, right.delivery),
    )
    matches = sum(_same_text(a, b) for a, b in pairs)
    # One exact delivery is specific enough; zones need an origin/destination
    # pair so two distinct passes merely travelling "forward" remain separate.
    return bool(_same_text(left.delivery, right.delivery) or matches >= 2)


def _same_text(left: str | None, right: str | None) -> bool:
    return bool(left and right and _normalise(left) == _normalise(right))


def _normalise(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _detail(preferred: str | None, other: str | None) -> str | None:
    return preferred if preferred is not None else other


def _richer(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    return right if len(_normalise(right)) > len(_normalise(left)) else left
