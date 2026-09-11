"""The match state and the pre-match notes, over MCP.

Run this and any MCP client can interrogate a live match the way the analyst
does: what is the score, who is wearing seven, what was worth saying about
this fixture before it kicked off. It is the same :class:`MatchTools` object
the analyst holds in-process, so there is exactly one definition of what an
agent may know, and demonstrating it costs nothing but a transport.

    uv run python -m commentary.mcp_server        # stdio, for an MCP client
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer

from commentary.schemas import KnowledgePack, MatchState, TeamSheet
from commentary.tools import MatchTools

INSTRUCTIONS = """
Tools over one live soccer match, as the commentary system understands it.

Everything here was derived from the broadcast picture or written down before
kickoff. There is no live data feed behind these tools: the score comes from
reading the scoreboard on screen, and the rosters come from pre-match notes.
Treat a missing answer as genuinely unknown rather than retrying.
""".strip()


def build_server(tools: MatchTools, name: str = "commentary") -> MCPServer:
    """Wrap a live :class:`MatchTools` in an MCP server."""
    server = MCPServer(name, instructions=INSTRUCTIONS)

    @server.tool()
    def scoreline() -> dict[str, Any]:
        """Current score, clock, half, and whether a replay is on screen."""
        return tools.scoreline()

    @server.tool()
    def recent_events(count: int = 6) -> list[str]:
        """The last few events the caller saw, oldest first."""
        return tools.recent_events(count)

    @server.tool()
    def possession() -> str:
        """Which side the caller last believed had the ball."""
        return tools.possession()

    @server.tool()
    def team_sheet(side: str) -> dict[str, Any]:
        """Starting eleven and bench with shirt numbers. Side is home or away."""
        return tools.team_sheet(side)

    @server.tool()
    def player(query: str) -> dict[str, Any]:
        """Find a player by name, surname, or shirt number."""
        return tools.player(query)

    @server.tool()
    def storylines() -> list[str]:
        """What was worth saying about this fixture before it started."""
        return tools.storylines()

    @server.tool()
    def form(team: str = "") -> dict[str, str]:
        """Recent results as prepared before kickoff."""
        return tools.form(team)

    @server.tool()
    def key_matchups() -> list[str]:
        """Individual battles flagged in the pre-match notes."""
        return tools.key_matchups()

    return server


def demo_server() -> MCPServer:
    """A server over an empty match, so the tools can be inspected offline."""
    pack = KnowledgePack(home=TeamSheet(name="Home"), away=TeamSheet(name="Away"))
    state = MatchState(home=pack.home.name, away=pack.away.name)
    return build_server(MatchTools(state=state, pack=pack))


if __name__ == "__main__":
    demo_server().run()
