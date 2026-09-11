"""What an agent is allowed to look up, and nothing else.

The analyst does not get a search box. It gets these seven questions, over
the match state the system built by watching, and over the notes the
researcher wrote before kickoff. That is the whole world available to it at
runtime, which is the point: a tool surface this narrow cannot accidentally
become a live data feed.

The same functions are exposed two ways. The analyst calls them directly,
because an in-process call costs nothing and a commentary system cannot
afford a network hop to learn its own scoreline. The MCP server next door
wraps the identical object so the tools can be inspected, demoed, and driven
from outside the process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from commentary.schemas import KnowledgePack, MatchState, Player, Side, TeamSheet


def _side(value: str) -> Side:
    text = value.strip().lower()
    if text in {"home", "h"}:
        return Side.HOME
    if text in {"away", "a"}:
        return Side.AWAY
    return Side.UNKNOWN


@dataclass
class MatchTools:
    """A read-only view of everything known, bound to a live match."""

    state: MatchState
    pack: KnowledgePack | None = None

    # -- the match itself ------------------------------------------------

    def scoreline(self) -> dict[str, Any]:
        """The score, the clock, and the half. The board is the only source."""
        return {
            "home": self.state.home,
            "away": self.state.away,
            "home_score": self.state.home_score,
            "away_score": self.state.away_score,
            "clock": self.state.clock,
            "period": self.state.period,
            "in_replay": self.state.in_replay,
        }

    def recent_events(self, count: int = 6) -> list[str]:
        """What the caller has seen happen lately, newest last."""
        return [e.value for e in self.state.last_events[-count:]]

    def possession(self) -> str:
        return self.state.possession.value

    # -- the notes -------------------------------------------------------

    def team_sheet(self, side: str) -> dict[str, Any]:
        """A starting eleven and bench with shirt numbers, as researched."""
        sheet = self._sheet(side)
        if sheet is None:
            return {"error": f"unknown side {side!r}, expected 'home' or 'away'"}
        return {
            "name": sheet.name,
            "short": sheet.short,
            "kit": sheet.kit,
            "formation": sheet.formation,
            "manager": sheet.manager,
            "starters": [_player(p) for p in sheet.starters],
            "bench": [_player(p) for p in sheet.bench],
        }

    def player(self, query: str) -> dict[str, Any]:
        """Look a player up by name, surname, or shirt number.

        Returns which side they are on, because the single most useful thing
        to know about a name you just read off a shirt is whose shirt it was.
        """
        if self.pack is None:
            return {"error": "no knowledge pack loaded"}
        wanted = query.strip().lower()
        for side, sheet in (("home", self.pack.home), ("away", self.pack.away)):
            for entry in sheet.squad:
                matches = wanted in {
                    entry.name.lower(),
                    entry.surname.lower(),
                    str(entry.number),
                }
                if matches:
                    starting = entry in sheet.starters
                    return {**_player(entry), "side": side, "team": sheet.name,
                            "starting": starting}
        return {"error": f"no player matching {query!r} on either sheet"}

    def storylines(self) -> list[str]:
        """The things worth mentioning that are not visible on the pitch."""
        return list(self.pack.storylines) if self.pack else []

    def form(self, team: str = "") -> dict[str, str]:
        """Recent results, as prepared. Empty when the researcher found none."""
        if self.pack is None:
            return {}
        if not team:
            return dict(self.pack.form)
        for key, value in self.pack.form.items():
            if team.lower() in key.lower():
                return {key: value}
        return {}

    def key_matchups(self) -> list[str]:
        return list(self.pack.key_matchups) if self.pack else []

    # -- internals -------------------------------------------------------

    def _sheet(self, side: str) -> TeamSheet | None:
        if self.pack is None:
            return None
        return self.pack.team(_side(side))


def _player(player: Player) -> dict[str, Any]:
    return {"name": player.name, "number": player.number, "position": player.position}
