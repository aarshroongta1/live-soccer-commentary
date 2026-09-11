"""The tool surface is a promise about what an agent can and cannot learn."""

from __future__ import annotations

import asyncio

import pytest

from commentary.mcp_server import build_server
from commentary.schemas import Event, KnowledgePack, MatchState, Player, Side, TeamSheet
from commentary.tools import MatchTools


@pytest.fixture
def tools() -> MatchTools:
    pack = KnowledgePack(
        home=TeamSheet(
            name="Arsenal",
            short="ARS",
            starters=[Player(name="Bukayo Saka", number=7, position="RW")],
            bench=[Player(name="Kai Havertz", number=29)],
        ),
        away=TeamSheet(name="Real Madrid", short="RMA"),
        storylines=["First meeting since the quarter-final"],
        form={"Arsenal": "WWDLW"},
        key_matchups=["Saka against the left back"],
    )
    state = MatchState(
        home="Arsenal",
        away="Real Madrid",
        home_score=1,
        clock="37:12",
        possession=Side.HOME,
        last_events=[Event.CORNER, Event.SHOT, Event.SAVE],
    )
    return MatchTools(state=state, pack=pack)


def test_the_scoreline_comes_from_state(tools: MatchTools) -> None:
    assert tools.scoreline()["home_score"] == 1
    assert tools.scoreline()["clock"] == "37:12"


def test_a_player_lookup_says_whose_shirt_it_was(tools: MatchTools) -> None:
    found = tools.player("7")
    assert found["name"] == "Bukayo Saka"
    assert found["side"] == "home"
    assert found["starting"] is True


def test_a_surname_and_a_substitute_both_resolve(tools: MatchTools) -> None:
    assert tools.player("saka")["number"] == 7
    havertz = tools.player("Havertz")
    assert havertz["side"] == "home" and havertz["starting"] is False


def test_an_unknown_player_is_an_honest_miss(tools: MatchTools) -> None:
    assert "error" in tools.player("Kowalczyk")


def test_an_unknown_side_does_not_invent_a_sheet(tools: MatchTools) -> None:
    assert "error" in tools.team_sheet("neither")


def test_recent_events_are_bounded_and_ordered(tools: MatchTools) -> None:
    assert tools.recent_events(2) == ["shot", "save"]


def test_form_can_be_asked_for_one_team(tools: MatchTools) -> None:
    assert tools.form("arsenal") == {"Arsenal": "WWDLW"}
    assert tools.form("Nobody") == {}


def test_the_tools_survive_having_no_notes() -> None:
    bare = MatchTools(state=MatchState(home="A", away="B"))
    assert bare.storylines() == []
    assert "error" in bare.team_sheet("home")
    assert bare.scoreline()["home"] == "A"


def test_every_tool_is_exposed_over_mcp(tools: MatchTools) -> None:
    server = build_server(tools, name="test")
    listed = {tool.name for tool in asyncio.run(server.list_tools())}
    assert listed == {
        "scoreline",
        "recent_events",
        "possession",
        "team_sheet",
        "player",
        "storylines",
        "form",
        "key_matchups",
    }
