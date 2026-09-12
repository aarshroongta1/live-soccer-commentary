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

The reading of a StatsBomb row is not done here. It lives in
:mod:`commentary.statsbomb`, which the wire uses on the same files, and this
module is the grading half: the same events, cut down to the ones that are
graded and written in the saved feed's vocabulary.
"""

from __future__ import annotations

from typing import Any

from commentary import statsbomb
from commentary.schemas import Event, Side

#: The wire's word for a card to the one a saved feed uses. A second yellow
#: is a sending-off and the feed vocabulary has no separate word for one, so
#: it is written as what it is on the pitch.
_CARDS = {"yellow": "yellow card", "second yellow": "red card", "red": "red card"}


def convert(rows: list[dict[str, Any]], home: str, away: str) -> dict[str, Any]:
    """StatsBomb event rows to the document :func:`feed.load_feed` reads.

    ``home`` and ``away`` are the team names as StatsBomb spells them, which
    is how a row's ``team.name`` is placed on a side. A row naming neither is
    dropped: a goal on a side we cannot identify is worse than no goal at all.
    """
    events: list[dict[str, Any]] = []
    # No lineups file here — a grading feed is read by the name StatsBomb
    # writes, and convert is handed rows somebody has already loaded.
    for event in statsbomb.events(rows, home, away, {}):
        if event.event is Event.GOAL:
            kind = "own goal" if event.detail == "own goal" else "goal"
        elif event.event is Event.CARD:
            kind = _CARDS[event.detail]
        elif event.event is Event.SUBSTITUTION:
            kind = "substitution"
        else:
            continue

        events.append(
            {
                "clock": f"{int(event.clock_s // 60)}:{int(event.clock_s % 60):02d}",
                "type": kind,
                "team": "home" if event.side is Side.HOME else "away",
                "player": event.player,
                "home_score": event.home_score,
                "away_score": event.away_score,
            }
        )

    return {"home_team": home, "away_team": away, "events": events}
