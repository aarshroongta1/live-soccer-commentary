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

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, cast

import pytest
from pydantic import BaseModel, ValidationError

from commentary.agents.researcher import (
    WEB_SEARCH_TOOL,
    Researcher,
    _tools_enabled,
    check_notes,
    freeze,
    hand_check_list,
    is_frozen,
    load_pack,
    merge_notes,
    note_subject,
    only_checked,
    pack_path,
    researched_path,
    save_pack,
    settle_notes,
    update_from_substitution,
)
from commentary.config import RESEARCHER_MODEL
from commentary.gate import FactGate
from commentary.llm.anthropic_backend import AnthropicBackend
from commentary.llm.base import Block, LLMError, Parsed
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.researcher import (
    notes_blocks,
    notes_system,
    researcher_blocks,
    researcher_system,
)
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    MatchState,
    Note,
    NoteSheet,
    Player,
    Scene,
    Side,
    Sighting,
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
        sightings=[Sighting(name=n) for n in names],
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


def test_the_hook_matches_the_real_backends_attribute():
    """The one thing that silently breaks web search: a rename in ``llm/``.

    The researcher looks ``extra_params`` up by name and degrades quietly when
    it is not there, which is right at runtime and useless as a warning — a
    renamed attribute would cost every future pack its live search without
    failing anything. So the name is asserted against the real class here, with
    a stub client so that constructing it touches no key and no network.
    """
    backend = AnthropicBackend(client=cast(Any, object()))
    assert backend.extra_params == {}

    with _tools_enabled(backend, [WEB_SEARCH_TOOL]) as attached:
        assert attached
        assert backend.extra_params["tools"] == [WEB_SEARCH_TOOL]
    assert backend.extra_params == {}


async def test_the_tool_is_detached_even_when_the_call_fails():
    backend = BackendWithToolHook(ScriptedBackend())  # no handler: parse raises
    researcher = Researcher(backend)

    with pytest.raises(LLMError):
        await researcher.research("Arsenal", "Chelsea")

    assert backend.seen_tools == [[WEB_SEARCH_TOOL]]
    assert backend.extra_params == {}


async def test_a_pre_existing_tools_entry_is_put_back():
    """Restore, not clear: the hook borrows the mapping, it does not own it."""
    backend = BackendWithToolHook(backend_returning(a_pack()))
    theirs = [{"type": "something_else", "name": "not_ours"}]
    backend.extra_params["tools"] = theirs

    await Researcher(backend).research("Arsenal", "Chelsea")

    assert backend.extra_params["tools"] is theirs


# -- notes -------------------------------------------------------------------
#
# The pack has always carried context and none of it has ever been spoken: the
# storylines are paragraph-shaped and filed under nobody. A note is the same
# information cut to a clause and filed under a name, and a name is the whole
# point — the runtime looks notes up by the names on the screen and the fact
# gate checks a spoken figure against the notes about the names in the line.
# A note nobody can look up is not a note.


def some_notes() -> list[Note]:
    return [
        Note(
            about="Bukayo Saka",
            text="four goals in this competition",
            kind="stat",
            source="Premier League records",
        ),
        Note(
            about="Arsenal",
            text="have not lost at home since April",
            kind="storyline",
            source="Premier League results",
        ),
        Note(
            about="Cole Palmer",
            text="always goes to the keeper's left from the spot",
            kind="habit",
            source="Chelsea penalties, 2024 onwards",
        ),
    ]


@pytest.mark.asyncio
async def test_the_notes_the_researcher_writes_come_through_into_the_pack() -> None:
    wanted = a_pack().model_copy(update={"notes": some_notes()})
    researcher = Researcher(backend_returning(wanted))
    pack = await researcher.research("Arsenal", "Chelsea")
    assert [note.text for note in pack.notes] == [note.text for note in some_notes()]
    assert pack.notes[0].kind == "stat"
    assert pack.notes[0].source == "Premier League records"
    assert researcher.dropped_notes == []


@pytest.mark.asyncio
async def test_a_note_about_a_name_on_no_roster_is_dropped_and_said_so(caplog) -> None:
    """The one check that matters, because nothing downstream can make it.

    The runtime asks the pack for the notes about the players it can see, so
    a note filed under somebody who is not in the fixture is never asked for
    and never found. Left in the pack it is a quiet lie about how much
    context the system has; dropped, with a line in the log, it is a research
    pass that did not quite do as it was told.
    """
    invented = Note(
        about="Erling Haaland",
        text="nine goals in this competition",
        kind="stat",
        source="somewhere else entirely",
    )
    wanted = a_pack().model_copy(update={"notes": [*some_notes(), invented]})
    researcher = Researcher(backend_returning(wanted))
    with caplog.at_level("WARNING"):
        pack = await researcher.research("Arsenal", "Chelsea")

    assert "Erling Haaland" not in [note.about for note in pack.notes]
    assert len(pack.notes) == 3
    assert [note.about for note in researcher.dropped_notes] == ["Erling Haaland"]
    assert "Erling Haaland" in caplog.text
    assert "no team sheet" in caplog.text


def test_a_note_spelled_the_short_way_is_refiled_under_the_roster_name() -> None:
    """``about`` is a key, and a key that is nearly right is a key that misses."""
    pack = a_pack().model_copy(
        update={"notes": [Note(about="Saka", text="four goals in this competition", kind="stat")]}
    )
    settled, dropped = settle_notes(pack)
    assert dropped == []
    assert settled.notes[0].about == "Bukayo Saka"


def test_note_subject_answers_for_players_and_for_teams() -> None:
    pack = a_pack()
    assert note_subject(pack, "Bukayo Saka") == "Bukayo Saka"
    assert note_subject(pack, "Rice") == "Declan Rice"
    assert note_subject(pack, "Chelsea") == "Chelsea"
    assert note_subject(pack, "Mikel Arteta") is None
    assert note_subject(pack, "") is None


def test_check_notes_names_the_ones_nobody_can_look_up() -> None:
    """Reported, never raised: a hand-edited pack with one bad subject still loads."""
    stray = Note(about="the Arsenal captain", text="wears the armband", kind="storyline")
    pack = a_pack().model_copy(update={"notes": [*some_notes(), stray]})
    assert [note.about for note in check_notes(pack)] == ["the Arsenal captain"]
    assert check_notes(a_pack()) == []


def test_notes_survive_the_round_trip_to_disk(tmp_path: Path) -> None:
    pack = a_pack().model_copy(update={"notes": some_notes()})
    path = save_pack(pack, tmp_path / "pack.json")
    back = load_pack(path)
    assert [(n.about, n.text, n.kind, n.source) for n in back.notes] == [
        (n.about, n.text, n.kind, n.source) for n in some_notes()
    ]


def test_a_pack_written_before_notes_existed_loads_with_none(tmp_path: Path) -> None:
    """The default that keeps every pack on disk working."""
    path = tmp_path / "old.json"
    payload = a_pack().model_dump()
    payload.pop("notes")
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_pack(path).notes == []


def test_an_unusable_note_on_disk_is_logged_and_kept(tmp_path: Path, caplog) -> None:
    stray = Note(about="the Arsenal captain", text="wears the armband", kind="storyline")
    path = save_pack(a_pack().model_copy(update={"notes": [stray]}), tmp_path / "pack.json")
    with caplog.at_level("WARNING"):
        back = load_pack(path)
    assert len(back.notes) == 1
    assert "the Arsenal captain" in caplog.text


def test_the_notes_are_frozen_with_everything_else_at_kickoff() -> None:
    """A note edited at sixty-three minutes could only have come from outside."""
    frozen = freeze(a_pack().model_copy(update={"notes": some_notes()}))
    assert is_frozen(frozen)
    with pytest.raises(ValidationError):
        frozen.notes[0].text = "nine goals in this competition"


# -- the notes command -------------------------------------------------------


@pytest.mark.asyncio
async def test_write_notes_adds_context_without_touching_the_team_sheets() -> None:
    """The reason this is a second command and not a second research call.

    The sheets are the expensive part and the part that goes stale: a pack
    built the night before has squad numbers somebody checked against two
    sources, and re-running the full researcher to gain a sentence about form
    would put them back in the hands of a model.
    """
    pack = a_pack()
    backend = ScriptedBackend()
    backend.always("notes", NoteSheet(notes=some_notes()))
    researcher = Researcher(backend)

    updated = await researcher.write_notes(pack)
    assert [note.text for note in updated.notes] == [note.text for note in some_notes()]
    assert updated.model_dump(exclude={"notes"}) == pack.model_dump(exclude={"notes"})
    assert backend.calls_tagged("notes")[0].output_format is NoteSheet


@pytest.mark.asyncio
async def test_write_notes_replaces_rather_than_appends() -> None:
    """Twice over the same pack gives one set of notes, not two."""
    backend = ScriptedBackend()
    backend.always("notes", NoteSheet(notes=some_notes()))
    researcher = Researcher(backend)
    once = await researcher.write_notes(a_pack())
    twice = await researcher.write_notes(once)
    assert len(twice.notes) == len(some_notes())


@pytest.mark.asyncio
async def test_write_notes_drops_a_subject_nobody_can_look_up(caplog) -> None:
    stray = Note(about="Les Bleus", text="unbeaten in nine", kind="storyline")
    backend = ScriptedBackend()
    backend.always("notes", NoteSheet(notes=[*some_notes(), stray]))
    researcher = Researcher(backend)
    with caplog.at_level("WARNING"):
        updated = await researcher.write_notes(a_pack())
    assert "Les Bleus" not in [note.about for note in updated.notes]
    assert "Les Bleus" in caplog.text


def test_the_notes_prompt_shows_every_name_a_note_may_be_filed_under() -> None:
    """A model cannot copy a spelling it was not shown."""
    pack = a_pack()
    body = "\n".join(
        block["text"] for block in notes_blocks(pack) if block.get("type") == "text"
    )
    for player in list(pack.home.squad) + list(pack.away.squad):
        assert player.name in body, player.name
    assert pack.home.name in body
    assert pack.away.name in body
    assert "under fourteen words" in notes_system()


def test_the_notes_rules_are_the_same_bytes_every_time() -> None:
    assert notes_system() == notes_system()


# -- forty notes instead of thirteen -----------------------------------------
#
# The pack for the 2022 final carried thirteen notes about seven players, and
# on both traces of that match the colour seat went quiet: the man on the ball
# was somebody nobody had written a line about. The researcher's brief now
# asks for forty to sixty, one about every starter, and that changes three
# things a test can hold on to — a note carries its own confidence and a tick
# box, a research pass merges rather than overwrites, and what comes out is a
# list a human reads before any of it is allowed on air.


def a_checked_note() -> Note:
    return Note(
        about="Bukayo Saka",
        text="four goals in this competition",
        kind="stat",
        source="Premier League records",
        counts="goals",
        checked=True,
    )


def test_a_note_is_unchecked_and_fully_confident_until_told_otherwise() -> None:
    """The two defaults, and the asymmetry between them.

    ``checked`` defaults to False because the safe assumption about a note is
    that nobody has read it. ``confidence`` defaults to 1.0 because it is the
    researcher's own estimate, and a note written by hand into a pack — or
    into a test — has no researcher to doubt.
    """
    note = Note(about="Bukayo Saka", text="four goals in this competition")
    assert note.checked is False
    assert note.confidence == 1.0


def test_every_pack_in_the_repository_still_loads() -> None:
    """Two new fields with defaults, and twelve packs on disk that predate them."""
    packs = sorted(Path("clips").glob("pack-*.json"))
    assert len(packs) >= 12
    for path in packs:
        pack = load_pack(path)
        assert pack.home.name and pack.away.name


def test_a_note_written_before_these_fields_existed_loads(tmp_path: Path) -> None:
    path = tmp_path / "old.json"
    payload = a_pack().model_dump()
    payload["notes"] = [{"about": "Bukayo Saka", "text": "four goals in this competition"}]
    path.write_text(json.dumps(payload), encoding="utf-8")
    note = load_pack(path).notes[0]
    assert (note.checked, note.confidence, note.clause) == (False, 1.0, "")


def test_confidence_outside_nought_to_one_is_refused() -> None:
    with pytest.raises(ValidationError):
        Note(about="Bukayo Saka", text="four goals", confidence=1.4)


# -- the unchecked are not said ----------------------------------------------


def test_only_checked_keeps_the_ticked_notes_and_says_how_many_went() -> None:
    fresh = Note(about="Cole Palmer", text="eleven goals this season", confidence=0.8)
    pack = a_pack().model_copy(update={"notes": [a_checked_note(), fresh]})
    trimmed, skipped = only_checked(pack)
    assert [note.about for note in trimmed.notes] == ["Bukayo Saka"]
    assert skipped == 1


def test_a_pack_whose_notes_are_all_checked_comes_back_untouched() -> None:
    """The common case on matchday allocates nothing and compares equal."""
    pack = a_pack().model_copy(update={"notes": [a_checked_note()]})
    trimmed, skipped = only_checked(pack)
    assert trimmed is pack
    assert skipped == 0


def test_the_hand_checked_pack_on_disk_is_ticked() -> None:
    """The thirteen in the 2022 pack were checked by hand and now say so.

    Without this the new default would silence the one pack in the
    repository whose notes somebody actually looked up.
    """
    pack = load_pack(Path("clips/pack-argfra-2022.json"))
    assert len(pack.notes) == 13
    assert all(note.checked for note in pack.notes)


# -- merging over notes somebody has checked ---------------------------------


def test_a_hand_checked_note_survives_a_second_research_pass() -> None:
    fresh = Note(about="Cole Palmer", text="eleven goals this season", confidence=0.8)
    merge = merge_notes([a_checked_note()], [fresh])
    assert [note.about for note in merge.notes] == ["Bukayo Saka", "Cole Palmer"]
    assert merge.notes[0] == a_checked_note()
    assert (merge.kept, merge.added) == (1, 1)


def test_a_fresh_note_arrives_unchecked_however_it_was_marked() -> None:
    """A model does not get to tick its own work."""
    eager = Note(about="Cole Palmer", text="eleven goals this season", checked=True)
    merge = merge_notes([], [eager])
    assert merge.notes[0].checked is False


def test_an_unchecked_note_already_in_the_pack_is_replaced_not_kept() -> None:
    """What keeps a second run over the same pack idempotent."""
    stale = Note(about="Cole Palmer", text="ten goals this season")
    fresh = Note(about="Cole Palmer", text="eleven goals this season")
    merge = merge_notes([stale], [fresh])
    assert [note.text for note in merge.notes] == ["eleven goals this season"]


def test_the_same_note_back_again_donates_its_no_number_form() -> None:
    """The one thing a repeat may contribute, and the reason to ask for it.

    The thirteen notes in the 2022 pack were written before ``clause``
    existed, so the colour seat — which may not say a number — cannot say any
    of them. The prompt asks for those back with the text copied exactly, and
    all that comes across is the missing wording.
    """
    again = Note(
        about="Bukayo Saka",
        text="four goals in this competition",
        clause="scoring in every round of this competition",
        source="a worse source",
        confidence=0.4,
    )
    merge = merge_notes([a_checked_note()], [again])
    assert len(merge.notes) == 1
    kept = merge.notes[0]
    assert kept.clause == "scoring in every round of this competition"
    assert kept.checked is True
    assert kept.source == "Premier League records"
    assert kept.confidence == 1.0
    assert merge.clauses == 1


def test_a_repeat_of_a_note_that_already_has_a_clause_is_dropped() -> None:
    checked = a_checked_note().model_copy(update={"clause": "scoring in every round"})
    merge = merge_notes([checked], [checked.model_copy(update={"clause": "in form"})])
    assert merge.notes == [checked]
    assert len(merge.duplicates) == 1
    assert merge.clauses == 0


@pytest.mark.asyncio
async def test_write_notes_keeps_what_a_human_checked() -> None:
    pack = a_pack().model_copy(update={"notes": [a_checked_note()]})
    backend = ScriptedBackend()
    backend.always("notes", NoteSheet(notes=some_notes()))
    researcher = Researcher(backend)

    updated = await researcher.write_notes(pack)

    assert updated.notes[0] == a_checked_note()
    assert researcher.last_merge is not None
    assert researcher.last_merge.kept == 1
    # "four goals in this competition" is in some_notes() too, under the same
    # name, so it comes back as the repeat rather than as a fourth note.
    assert len(updated.notes) == len(some_notes())


# -- the list a human ticks --------------------------------------------------


def a_pack_to_check() -> KnowledgePack:
    return a_pack().model_copy(
        update={
            "notes": [
                Note(
                    about="Cole Palmer",
                    text="eleven goals this season",
                    kind="stat",
                    source="https://example.test/palmer",
                    counts="goals",
                    clause="among the league's leading scorers",
                    confidence=0.95,
                ),
                a_checked_note(),
                Note(
                    about="Declan Rice",
                    text="seven set-piece assists since January",
                    kind="stat",
                    source="counted from match reports",
                    confidence=0.5,
                ),
                Note(
                    about="Arsenal",
                    text="have not lost at home since April",
                    kind="storyline",
                    source="Premier League results",
                    confidence=0.9,
                ),
            ]
        }
    )


def test_the_hand_check_list_groups_by_subject_in_team_sheet_order() -> None:
    """One player, every claim about him, one source page open."""
    printed = hand_check_list(a_pack_to_check())
    order = [printed.index(name) for name in ("Arsenal", "Bukayo Saka", "Declan Rice")]
    assert order == sorted(order)
    assert printed.index("Declan Rice") < printed.index("Cole Palmer")


def test_the_hand_check_list_shows_the_source_and_the_tick_box() -> None:
    printed = hand_check_list(a_pack_to_check())
    assert "[x] four goals in this competition" in printed
    assert "[ ] eleven goals this season" in printed
    assert "source: https://example.test/palmer" in printed
    assert "counts goals" in printed


def test_the_hand_check_list_flags_the_shaky_ones_and_the_missing_clauses() -> None:
    """The two things it exists to put in front of somebody.

    A note the researcher was unsure of is where the checking time is worth
    most, and a note with a figure and no no-number form is one the colour
    seat cannot say at all.
    """
    printed = hand_check_list(a_pack_to_check())
    shaky = [line for line in printed.splitlines() if "<< LOW" in line]
    assert len(shaky) == 1
    assert "set-piece assists" in shaky[0]
    assert "no number: among the league's leading scorers" in printed
    assert printed.count("MISSING, the colour seat cannot say this one") == 2
    assert "4 notes about 4 subjects, 1 already checked, 1 under 0.7 confidence" in printed


def test_the_hand_check_list_does_not_ask_for_a_clause_where_there_is_no_number() -> None:
    pack = a_pack().model_copy(
        update={
            "notes": [
                Note(
                    about="Cole Palmer",
                    text="always goes to the keeper's left from the spot",
                    kind="habit",
                    source="Chelsea penalties",
                )
            ]
        }
    )
    assert "MISSING" not in hand_check_list(pack)


def test_a_researched_pack_is_written_beside_the_one_it_read_never_over_it() -> None:
    source = Path("clips/pack-argfra-2022.json")
    assert researched_path(source) == Path("clips/pack-argfra-2022-researched.json")
    assert researched_path(source) != source


# -- what the brief now asks for ---------------------------------------------


def test_the_notes_prompt_asks_for_a_pack_that_covers_the_whole_pitch() -> None:
    rules = notes_system()
    assert "forty to sixty" in rules.lower()
    assert "EVERY STARTER, BOTH SIDES" in rules
    assert "confidence" in rules
    assert "NEVER INVENT A NUMBER" in rules


def test_the_notes_prompt_shows_the_shirt_number_and_the_position() -> None:
    """A researcher told to write about every starter has to know who they are."""
    body = "\n".join(
        block["text"] for block in notes_blocks(a_pack()) if block.get("type") == "text"
    )
    assert "David Raya  (number 22, GK)" in body
    assert "Marc Cucurella  (LB)" in body  # no number confirmed, so none shown
    assert "at least one" in body


def test_the_notes_prompt_puts_the_checked_notes_in_front_of_the_model() -> None:
    """Twice over: do not repeat them, except to fill in the missing clause."""
    pack = a_pack().model_copy(update={"notes": [a_checked_note()]})
    body = "\n".join(
        block["text"] for block in notes_blocks(pack) if block.get("type") == "text"
    )
    assert "ALREADY CHECKED BY A HUMAN" in body
    assert "four goals in this competition  [needs a no-number form]" in body
    assert "copied character for character" in body


def test_a_pack_with_no_checked_notes_is_not_told_about_any() -> None:
    body = "\n".join(
        block["text"] for block in notes_blocks(a_pack()) if block.get("type") == "text"
    )
    assert "ALREADY CHECKED" not in body


def test_the_prompt_only_asks_for_a_clause_where_there_is_a_figure() -> None:
    """A habit has no number to take out, and asking for one invites invention."""
    habit = Note(
        about="Cole Palmer",
        text="always goes to the keeper's left from the spot",
        kind="habit",
        source="Chelsea penalties",
        checked=True,
    )
    pack = a_pack().model_copy(update={"notes": [a_checked_note(), habit]})
    body = "\n".join(
        block["text"] for block in notes_blocks(pack) if block.get("type") == "text"
    )
    assert "four goals in this competition  [needs a no-number form]" in body
    marked = [line for line in body.splitlines() if line.startswith("  Cole Palmer: ")]
    assert marked == ["  Cole Palmer: always goes to the keeper's left from the spot"]
