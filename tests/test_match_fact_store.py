"""The serializable boundary between live match facts and orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from commentary.orchestration import FactSnapshot, MatchFactStore
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    Player,
    Scene,
    Side,
    TeamSheet,
    WireEvent,
)


def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Arsenal",
            short="ARS",
            starters=[Player(name="Bukayo Saka", number=7), Player(name="Declan Rice", number=41)],
        ),
        away=TeamSheet(
            name="Chelsea",
            short="CHE",
            starters=[Player(name="Cole Palmer", number=20)],
        ),
    )


@dataclass(frozen=True)
class Board:
    home_score: int
    away_score: int
    clock: str
    in_replay: bool = False
    bug_missing: bool = False


def caller_line() -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.NONE,
        side=Side.HOME,
        confidence=0.9,
        speak=False,
    )


def test_a_snapshot_round_trips_and_is_isolated_from_later_changes() -> None:
    store = MatchFactStore.from_pack(pack())
    store.apply_board(Board(home_score=1, away_score=0, clock="12:34"))

    snapshot = store.snapshot(20.0)
    restored = FactSnapshot.model_validate_json(snapshot.model_dump_json())
    store.apply_board(Board(home_score=2, away_score=0, clock="13:01"))

    assert restored == snapshot
    assert snapshot.state.home_score == 1
    assert store.state.home_score == 2


def test_versions_only_advance_when_wrapped_facts_change() -> None:
    store = MatchFactStore.from_pack(pack())
    assert store.version == 0

    board = Board(home_score=0, away_score=0, clock="00:10")
    store.apply_board(board)
    after_board = store.version
    assert after_board == 1

    store.apply_board(board)
    assert store.version == after_board

    line = caller_line()
    store.apply_caller(line, 10.0)
    after_caller = store.version
    assert after_caller == after_board + 1

    store.apply_caller(line, 11.0)
    assert store.version == after_caller


def test_evidence_can_advance_the_version_without_changing_state() -> None:
    store = MatchFactStore.from_pack(pack())
    before = store.state.model_copy(deep=True)

    store.mark_evidence()

    assert store.version == 1
    assert store.state == before


def test_snapshot_advances_pending_possession_to_its_recipient() -> None:
    store = MatchFactStore.from_pack(pack())
    store.apply_wire(
        WireEvent(
            event=Event.PASS,
            side=Side.HOME,
            player="Bukayo Saka",
            recipient="Declan Rice",
            clock_s=10.0,
            period=1,
            duration_s=2.0,
        ),
        10.0,
    )
    after_pass = store.version

    before_arrival = store.snapshot(11.9)
    assert before_arrival.version == after_pass
    assert before_arrival.state.ball is not None
    assert before_arrival.state.ball.player == "Bukayo Saka"

    after_arrival = store.snapshot(12.0)
    assert after_arrival.version == after_pass + 1
    assert after_arrival.state.ball is not None
    assert after_arrival.state.ball.player == "Declan Rice"
