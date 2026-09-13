"""StatsBomb's saved event data, read into the wire's events.

This is the one parser of StatsBomb rows in the project. The wire reads it
to put a statistician in the caller's ear; :mod:`commentary.grading.statsbomb`
reads the same mapping to make a grading feed. Two parsers would be two
chances to disagree about what a row means, and the grading feed is the
thing the wire is measured against.

Nothing here fetches. Both files are ones somebody saved after the match,
for the same reason :mod:`commentary.grading.feed` reads a file: a loader
that can only open a path cannot leak a live source into a run.

The wire carries every row that names a player, not only the ones on the
ball, because commentary has to say who fouled whom and who the card is
for. Rows that name nobody — pressure, ball receipts, the half starting —
are dropped: they would be noise in the caller's prompt and cost tokens.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from commentary.schemas import Event, Side, WireEvent

#: A StatsBomb card, lowered, to the colour the wire says. A second yellow
#: is kept as itself: it is a sending-off, and the caller says so.
CARD_COLOURS = {
    "yellow card": "yellow",
    "second yellow": "second yellow",
    "red card": "red",
}

#: A StatsBomb pass type that is a restart, and the event it really is. A
#: throw-in is a throw-in before it is a pass, and a commentator is graded on
#: saying so — the grader's truth for a clip is largely made of these. The
#: pass types not here (Recovery, Interception, Goal Kick) are not restarts a
#: commentator calls, so they stay passes.
RESTARTS = {
    "Throw-in": Event.THROW_IN,
    "Corner": Event.CORNER,
    "Free Kick": Event.FREE_KICK,
    "Kick Off": Event.KICKOFF,
}


def read(events_path: Path, lineups_path: Path, home: str, away: str) -> list[WireEvent]:
    """The wire's events for one match, from the two files StatsBomb publish.

    ``home`` and ``away`` are the team names as StatsBomb spells them, which
    is how a row's ``team.name`` is placed on a side.
    """
    rows = json.loads(events_path.read_text(encoding="utf-8"))
    lineups = json.loads(lineups_path.read_text(encoding="utf-8"))
    return events(rows, home, away, player_names(lineups))


def player_names(lineups: Any) -> dict[str, str]:
    """Player id and registered name to the name a commentator would say.

    StatsBomb's ``player_name`` is the full registered one — "Lionel Andrés
    Messi Cuccittini" — and the ``player_nickname`` is what everyone calls
    him. Both the id and the registered name are keys, because a row may be
    resolved by either.
    """
    names: dict[str, str] = {}
    for team in lineups if isinstance(lineups, list) else []:
        for player in team.get("lineup", []):
            full = str(player.get("player_name") or "")
            said = str(player.get("player_nickname") or "") or full
            if not said:
                continue
            names[str(player.get("player_id"))] = said
            names[full] = said
    return names


def events(
    rows: list[dict[str, Any]], home: str, away: str, names: Mapping[str, str]
) -> list[WireEvent]:
    """StatsBomb rows to wire events, in match-clock order.

    ``names`` is the map :func:`player_names` builds; an empty one leaves
    every player under StatsBomb's own spelling. A row naming neither team
    is dropped — an event on a side we cannot identify is worse than none.
    """
    sides = {home: Side.HOME, away: Side.AWAY}
    by_id = {str(row.get("id")): row for row in rows if isinstance(row, dict)}
    wire: list[WireEvent] = []
    home_score = away_score = 0

    for row in sorted(rows, key=_sort_key):
        side = sides.get(name_at(row, "team"))
        if side is None:
            continue
        for event, event_side, player, recipient, detail in _mapped(row, side, names, by_id):
            if event is Event.GOAL:
                if event_side is Side.HOME:
                    home_score += 1
                else:
                    away_score += 1
            wire.append(
                WireEvent(
                    event=event,
                    side=event_side,
                    player=player or None,
                    recipient=recipient or None,
                    detail=detail,
                    home_score=home_score,
                    away_score=away_score,
                    clock_s=float(int(row.get("minute", 0)) * 60 + int(row.get("second", 0))),
                    period=int(row.get("period", 1)),
                    duration_s=float(row.get("duration") or 0.0),
                )
            )
    return wire


def name_at(row: Any, *keys: str) -> str:
    """``row["a"]["b"]["name"]``, or "" if any step is missing."""
    node: Any = row
    for key in keys:
        if not isinstance(node, dict):
            return ""
        node = node.get(key)
    if isinstance(node, dict):
        node = node.get("name")
    return node if isinstance(node, str) else ""


def _mapped(
    row: Any,
    side: Side,
    names: Mapping[str, str],
    by_id: Mapping[str, Any],
) -> list[tuple[Event, Side, str, str, str]]:
    """What one row becomes: nothing, one event, or a foul and its card.

    Each tuple is (event, side, player, recipient, detail); "" is absent.
    """
    kind = name_at(row, "type")
    player = _person(row, "player", names=names)
    card = CARD_COLOURS.get(
        (name_at(row, "foul_committed", "card") or name_at(row, "bad_behaviour", "card")).lower(),
        "",
    )

    if kind == "Pass":
        recipient = _person(row, "pass", "recipient", names=names)
        if name_at(row, "pass", "outcome") == "Pass Offside":
            return [(Event.OFFSIDE, side, player, recipient, "")]
        restart = RESTARTS.get(name_at(row, "pass", "type"))
        if restart is not None:
            # A restart is the event, not a pass that happens to be one: it
            # is named from the picture and it is what the grader counts.
            return [(restart, side, player, recipient, "")]
        # An incomplete pass reaches nobody, so it names nobody to talk about.
        return [(Event.PASS, side, player, recipient, "")] if recipient else []
    if kind == "Carry":
        return [(Event.CARRY, side, player, "", "")]
    if kind == "Shot":
        outcome = name_at(row, "shot", "outcome")
        if outcome != "Goal":
            return [(Event.SHOT, side, player, "", outcome.lower())]
        how = name_at(row, "shot", "type").lower()
        return [(Event.GOAL, side, player, "", how if how == "penalty" else "")]
    if kind == "Own Goal Against":
        # The row is the team whose player put it in; the goal is the other's,
        # and so is nobody on this side's to be named for.
        return [(Event.GOAL, _other(side), "", "", "own goal")]
    if kind == "Goal Keeper" and "Save" in name_at(row, "goalkeeper", "type"):
        return [(Event.SAVE, side, player, "", "")]
    if kind == "Foul Committed":
        fouled = _fouled(row, by_id, names)
        penalty = bool((row.get("foul_committed") or {}).get("penalty"))
        # The award and the kick are two moments a minute apart, and a
        # commentator is graded on both. The kick is the Shot row, which
        # arrives as a goal or a save; this is the award.
        events = [(Event.FOUL, side, player, fouled, "penalty" if penalty else card)]
        if penalty:
            events.append((Event.PENALTY, _other(side), fouled, player, ""))
        if card:
            events.append((Event.CARD, side, player, "", card))
        return events
    if kind == "Bad Behaviour":
        return [(Event.CARD, side, player, "", card)] if card else []
    if kind == "Offside":
        return [(Event.OFFSIDE, side, player, "", "")]
    if kind == "Interception":
        return [(Event.INTERCEPTION, side, player, "", "")]
    if kind in ("Clearance", "Block"):
        return [(Event.CLEARANCE, side, player, "", "")]
    if kind == "Duel" and name_at(row, "duel", "type") == "Tackle":
        return [(Event.TACKLE, side, player, "", name_at(row, "duel", "outcome").lower())]
    if kind == "Substitution":
        return [
            (
                Event.SUBSTITUTION,
                side,
                _person(row, "substitution", "replacement", names=names),
                player,
                "",
            )
        ]
    return []


def _fouled(row: Any, by_id: Mapping[str, Any], names: Mapping[str, str]) -> str:
    """Who was fouled, from the Foul Won row StatsBomb pairs with the foul."""
    for related in row.get("related_events") or []:
        other = by_id.get(str(related))
        if other is not None and name_at(other, "type") == "Foul Won":
            return _person(other, "player", names=names)
    return ""


def _person(row: Any, *keys: str, names: Mapping[str, str]) -> str:
    """A StatsBomb ``{id, name}`` person, under the name the lineups prefer."""
    node: Any = row
    for key in keys:
        if not isinstance(node, dict):
            return ""
        node = node.get(key)
    if not isinstance(node, dict):
        return ""
    said = str(node.get("name") or "")
    return names.get(str(node.get("id")), "") or names.get(said, "") or said


def _other(side: Side) -> Side:
    return Side.AWAY if side is Side.HOME else Side.HOME


def _sort_key(row: Any) -> tuple[int, int, int]:
    return (int(row.get("period", 1)), int(row.get("minute", 0)), int(row.get("second", 0)))
