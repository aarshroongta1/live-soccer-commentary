"""A conservative publication check, not a goal/miss recognizer.

An unchanged or unreadable scoreboard cannot prove a miss. It can only leave a
claimed new goal unconfirmed. Score updates may lag real play; this is opt-in.
"""

from __future__ import annotations


def confirms_score_increment(
    first_left: int | None,
    first_right: int | None,
    last_left: int | None,
    last_right: int | None,
) -> bool:
    values = (first_left, first_right, last_left, last_right)
    if any(value is None or value < 0 for value in values):
        return False
    assert first_left is not None and first_right is not None
    assert last_left is not None and last_right is not None
    return (last_left - first_left, last_right - first_right) in {(1, 0), (0, 1)}


def hold_unconfirmed_goal(outcome: str, *, score_increment: bool) -> bool:
    return outcome == "goal" and not score_increment
