"""StatsBomb's open event data, converted to the feed shape this project reads.

StatsBomb publish full event data for the men's and women's World Cups, which
is the only way to grade a run on real footage: the broadcast gives pictures
and the captions give commentary, and neither of them says who actually
scored at 35:22. The conversion is a file-to-file one — nothing here fetches,
for the same reason nothing in :mod:`commentary.grading.feed` does.

Only the events a commentator would be graded on survive: goals, cards and
substitutions. StatsBomb rows run to four thousand a match, almost all of
them passes and pressures, and a feed carrying every touch would make recall
a measure of how often the system says anything at all.
"""

from __future__ import annotations

from typing import Any

#: A StatsBomb card name to the word a saved feed uses. A second yellow is a
#: sending-off and the feed vocabulary has no separate word for one, so it is
#: written as what it is on the pitch.
_CARDS = {
    "yellow card": "yellow card",
    "second yellow": "red card",
    "red card": "red card",
}


def _name(row: Any, *keys: str) -> str:
    """``row["a"]["b"]["name"]``, or "" if any step is missing."""
    node: Any = row
    for key in keys:
        if not isinstance(node, dict):
            return ""
        node = node.get(key)
    if isinstance(node, dict):
        node = node.get("name")
    return str(node) if isinstance(node, str) else ""


def _sort_key(row: Any) -> tuple[int, int, int]:
    return (int(row.get("period", 1)), int(row.get("minute", 0)), int(row.get("second", 0)))


def convert(rows: list[dict[str, Any]], home: str, away: str) -> dict[str, Any]:
    """StatsBomb event rows to the document :func:`feed.load_feed` reads.

    ``home`` and ``away`` are the team names as StatsBomb spells them, which
    is how a row's ``team.name`` is placed on a side. A row naming neither is
    dropped: a goal on a side we cannot identify is worse than no goal at all.
    """
    sides = {home: "home", away: "away"}
    events: list[dict[str, Any]] = []
    home_score = away_score = 0

    for row in sorted(rows, key=_sort_key):
        side = sides.get(_name(row, "team"))
        if side is None:
            continue
        kind = _name(row, "type")
        player = _name(row, "player")

        if kind == "Shot" and _name(row, "shot", "outcome") == "Goal":
            event_type = "goal"
        elif kind == "Own Goal Against":
            # The row belongs to the team whose player put it in; the goal
            # belongs to the other one.
            event_type = "own goal"
            side = "away" if side == "home" else "home"
        elif card := _CARDS.get(
            (_name(row, "foul_committed", "card") or _name(row, "bad_behaviour", "card")).lower()
        ):
            event_type = card
        elif kind == "Substitution":
            event_type = "substitution"
            player = _name(row, "substitution", "replacement") or player
        else:
            continue

        if event_type in ("goal", "own goal"):
            if side == "home":
                home_score += 1
            else:
                away_score += 1

        events.append(
            {
                "clock": f"{int(row.get('minute', 0))}:{int(row.get('second', 0)):02d}",
                "type": event_type,
                "team": side,
                "player": player or None,
                "home_score": home_score,
                "away_score": away_score,
            }
        )

    return {"home_team": home, "away_team": away, "events": events}
