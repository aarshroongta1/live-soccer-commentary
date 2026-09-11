"""The researcher, exercised entirely offline.

Every call here goes through :class:`ScriptedBackend`. That is not only a
speed decision: this is the one module in the system with a route to the open
web, so a test suite that could reach it would be testing the wrong property.
The properties worth pinning down are that the notes survive a round trip to
disk, that the second run does not pay for the first run's work again, that
the pack really is read-only once the whistle has gone, and that a name read
off a substitution graphic ends up somewhere the fact gate will accept it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from commentary.agents.researcher import (
    WEB_SEARCH_TOOL,
    Researcher,
    freeze,
    is_frozen,
    load_pack,
    pack_path,
    save_pack,
    update_from_substitution,
)
from commentary.config import RESEARCHER_MODEL
from commentary.gate import FactGate
from commentary.llm.base import Block, Parsed
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.researcher import researcher_blocks, researcher_system
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    MatchState,
    Player,
    Scene,
    Side,
    TeamSheet,
)

T = TypeVar("T", bound=BaseModel)


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Arsenal",
            short="ARS",
            kit="red shirts, white sleeves, white shorts; keeper in yellow",
            formation="4-3-3",
            manager="Mikel Arteta",
            starters=[
                Player(name="David Raya", number=22, position="GK"),
                Player(name="Bukayo Saka", number=7, position="RW"),
                Player(name="Declan Rice", number=41, position="CM"),
            ],
            bench=[Player(name="Kai Havertz", number=29, position="CF")],
        ),
        away=TeamSheet(
            name="Chelsea",
            short="CHE",
            kit="all blue; keeper in green",
            formation="4-2-3-1",
            manager="Enzo Maresca",
            starters=[
                Player(name="Robert Sánchez", number=1, position="GK"),
                Player(name="Cole Palmer", number=20, position="AM"),
                # The honesty rule, exercised: an unconfirmed number is null,
                # never a plausible one.
                Player(name="Marc Cucurella", number=None, position="LB"),
            ],
        ),
        competition="Premier League",
        venue="Emirates Stadium",
        kickoff="2026-09-19 16:30 BST",
        storylines=["Saka has scored in four straight home games."],
        form={"Arsenal": "WWDWL", "Chelsea": "DLWWW"},
        key_matchups=["Saka against Cucurella down the Arsenal right"],
    )


def backend_returning(pack: KnowledgePack) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.always("researcher", pack)
    return backend


@dataclass
class BackendWithToolHook:
    """A backend that owns the one escape hatch the researcher knows about.

    Stands in for an ``AnthropicBackend`` that merges ``extra_params`` into its
    request. That hook does not exist in ``llm/`` today, so this is where the
    tool-attaching path is exercised at all.
    """

    inner: ScriptedBackend
    extra_params: dict[str, Any] = field(default_factory=dict)
    seen_tools: list[Any] = field(default_factory=list)

    @property
    def total(self) -> Any:
        return self.inner.total

    async def parse(
        self,
        *,
        model: str,
        system: str,
        blocks: list[Block],
        output_format: type[T],
        max_tokens: int = 1024,
        effort: str | None = None,
        cache_system: bool = True,
        tag: str = "",
    ) -> Parsed[T]:
        self.seen_tools.append(self.extra_params.get("tools"))
        return await self.inner.parse(
            model=model,
            system=system,
            blocks=blocks,
            output_format=output_format,
            max_tokens=max_tokens,
            effort=effort,
            cache_system=cache_system,
            tag=tag,
        )


# -- the call itself -----------------------------------------------------


async def test_the_researcher_asks_the_right_model_for_a_pack():
    backend = backend_returning(a_pack())

    pack = await Researcher(backend).research(
        "Arsenal", "Chelsea", competition="Premier League", when="Saturday"
    )

    assert pack.home.name == "Arsenal"
    calls = backend.calls_tagged("researcher")
    assert len(calls) == 1
    assert calls[0].model == RESEARCHER_MODEL
    assert calls[0].output_format is KnowledgePack
    assert "Arsenal (home) v Chelsea (away)" in calls[0].text
    assert "Premier League" in calls[0].text


# -- persistence ---------------------------------------------------------


async def test_a_researched_pack_round_trips_through_disk(tmp_path: Path):
    researcher = Researcher(backend_returning(a_pack()))
    pack = await researcher.research("Arsenal", "Chelsea")

    path = save_pack(pack, tmp_path / "arsenal-v-chelsea.json")
    reloaded = load_pack(path)

    assert reloaded == pack
    assert reloaded.home.starters[1].number == 7
    # The null survives as a null. A round trip that quietly turned an unknown
    # number into a zero would be the exact failure the prompt argues against.
    assert reloaded.away.starters[2].number is None
    assert reloaded.form == {"Arsenal": "WWDWL", "Chelsea": "DLWWW"}


async def test_load_or_research_pays_the_model_once(tmp_path: Path):
    backend = backend_returning(a_pack())
    researcher = Researcher(backend)
    path = tmp_path / "packs" / "fixture.json"

    first = await researcher.load_or_research(path, "Arsenal", "Chelsea")
    second = await researcher.load_or_research(path, "Arsenal", "Chelsea")

    assert path.exists()
    assert second == first
    assert len(backend.calls_tagged("researcher")) == 1
    assert len(backend.calls) == 1


async def test_load_or_research_falls_back_to_a_name_under_packs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = backend_returning(a_pack())
    monkeypatch.chdir(tmp_path)

    await Researcher(backend).load_or_research(None, "Arsenal", "Chelsea")

    assert pack_path("Arsenal", "Chelsea").exists()
    assert pack_path("Arsenal", "Chelsea") == Path("packs/arsenal-v-chelsea.json")


# -- the freeze ----------------------------------------------------------


async def test_a_frozen_pack_cannot_be_mutated():
    live = freeze(await Researcher(backend_returning(a_pack())).research("Arsenal", "Chelsea"))

    assert is_frozen(live)
    with pytest.raises(ValidationError):
        live.competition = "Champions League"
    with pytest.raises(ValidationError):
        live.home = TeamSheet(name="Someone else")
    # All the way down: the sheets and the players are frozen too, so the
    # obvious way to corrupt a pack mid-match is closed as well.
    with pytest.raises(ValidationError):
        live.home.kit = "green"
    with pytest.raises(ValidationError):
        live.home.starters[0].number = 99


def test_freezing_changes_nothing_but_the_mutability(tmp_path: Path):
    pack = a_pack()
    frozen = freeze(pack)

    # Field for field identical. Not ``==``: Pydantic compares classes, and a
    # frozen pack is a subclass, so the two never compare equal however alike
    # they are. Worth pinning down, because it is invisible at the call site.
    assert frozen.model_dump() == pack.model_dump()
    assert frozen != pack
    assert not is_frozen(pack)
    assert frozen.home.squad[0].surname == "Raya"
    # A frozen pack serialises and reloads as an ordinary one.
    saved = load_pack(save_pack(frozen, tmp_path / "frozen.json"))
    assert saved.model_dump() == pack.model_dump()


# -- learning a substitute -----------------------------------------------


def speaking(text: str, *names: str) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.SUBSTITUTION,
        side=Side.HOME,
        team="Arsenal",
        names_read=list(names),
        confidence=0.8,
        speak=True,
        line=text,
    )


def test_update_from_substitution_teaches_the_gate_a_name():
    pack = a_pack()
    state = MatchState(home="Arsenal", away="Chelsea")
    gate = FactGate()
    line = speaking("Trossard comes on and drives at the full-back.", "Trossard")

    before = gate.judge(line, state, pack)
    assert not before.passed
    assert any("Trossard" in reason for reason in before.reasons)

    updated = update_from_substitution(pack, 19, "Leandro Trossard", Side.HOME)

    assert gate.judge(line, state, updated).passed
    assert updated.home.bench[-1] == Player(name="Leandro Trossard", number=19)


def test_update_from_substitution_leaves_the_original_alone():
    pack = freeze(a_pack())
    bench_before = len(pack.home.bench)

    updated = update_from_substitution(pack, 19, "Leandro Trossard", Side.HOME)

    assert updated is not pack
    assert len(pack.home.bench) == bench_before
    assert len(updated.home.bench) == bench_before + 1
    # A pack updated at seventy minutes is exactly as read-only as the one
    # that kicked off; the update replaces the reference, it never edits.
    assert is_frozen(updated)
    with pytest.raises(ValidationError):
        updated.home.bench[-1].number = 21


def test_a_name_already_on_the_sheet_is_not_added_twice():
    pack = a_pack()

    assert update_from_substitution(pack, 29, "Kai Havertz", Side.HOME) is pack
    # Surnames are matched the way the gate matches them, so the caption's
    # spelling of a first name does not create a duplicate player.
    assert update_from_substitution(pack, 7, "B. Saka", Side.HOME) is pack


def test_a_substitution_needs_a_side():
    with pytest.raises(ValueError):
        update_from_substitution(a_pack(), 19, "Leandro Trossard", Side.UNKNOWN)


# -- the prompt ----------------------------------------------------------


def test_the_prompt_asks_for_the_things_a_commentator_can_see():
    rules = researcher_system().lower()

    assert "shirt number" in rules
    assert "squad number" in rules
    assert "kit colours" in rules
    assert "goalkeeper" in rules
    assert "storylines" in rules


def test_the_prompt_insists_that_unknowns_come_back_null():
    rules = researcher_system().lower()

    assert "the number is null" in rules
    assert "fact gate" in rules
    # The reason has to be in the prompt, not only in this repository: the
    # model is being asked to prefer a gap to a guess, which is the opposite
    # of what a helpful assistant does by default.
    assert "invented" in rules


def test_the_blocks_name_the_fixture_and_which_side_is_home():
    blocks = researcher_blocks("Arsenal", "Chelsea", "Premier League", "2026-09-19")
    text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")

    assert "Arsenal (home) v Chelsea (away)" in text
    assert "Premier League" in text
    assert "2026-09-19" in text
    assert not any(b.get("type") == "image" for b in blocks)


def test_missing_context_is_spelled_out_rather_than_left_blank():
    text = "\n".join(b["text"] for b in researcher_blocks("Arsenal", "Chelsea"))

    assert "No competition was given" in text
    assert "No date was given" in text


# -- the web search tool -------------------------------------------------


async def test_the_search_tool_is_attached_when_the_backend_has_a_hook():
    backend = BackendWithToolHook(backend_returning(a_pack()))
    researcher = Researcher(backend)

    await researcher.research("Arsenal", "Chelsea")

    assert researcher.used_search
    assert backend.seen_tools == [[WEB_SEARCH_TOOL]]
    assert WEB_SEARCH_TOOL["type"] == "web_search_20260209"
    # Scoped to the call. A search tool left attached to a backend the caller
    # shares would be a live outside line into a running match.
    assert "tools" not in backend.extra_params


async def test_a_backend_without_the_hook_researches_without_search():
    backend = backend_returning(a_pack())
    researcher = Researcher(backend)

    pack = await researcher.research("Arsenal", "Chelsea")

    assert pack.home.name == "Arsenal"
    assert not researcher.used_search
