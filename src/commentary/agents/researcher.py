"""The researcher: the only agent allowed outside the broadcast, and only before it.

The rule this project is built on is that no live data reaches any agent at
runtime. This module is the single exception, and it is an exception in time
rather than in kind. It runs once, before kickoff, with a web search tool, and
writes a :class:`~commentary.schemas.KnowledgePack`. When the whistle goes the
pack is passed through :func:`freeze` and every agent downstream reads that
frozen object. There is no code path from a running match back into this
module — the caller holds a pack, not a researcher, and the analyst's tool
surface in ``tools.py`` is seven read-only questions over the same pack.

The freeze is real rather than rhetorical. :func:`freeze` returns a pack whose
every model in the tree rejects attribute assignment, so "the pack is read-only
after kickoff" is something the interpreter enforces at 63 minutes, not
something a reader has to take on trust from a docstring.

Two practical consequences shape the rest of the file.

A pack costs a real Opus call with web search behind it, and a live match is
the worst possible moment to discover that. So packs persist: research once the
day before, save the JSON, and :func:`Researcher.load_or_research` reads the
file on every subsequent run without touching the network.

Substitutions bring on players the pack already knows but the *state* does not,
and occasionally a name that is on no sheet at all. The plan calls for
re-running the researcher when a substitution board appears on screen; that
would be a live outside call, so it is not what happens here.
:func:`update_from_substitution` does the cheap, offline half instead — it
folds a name and number read off a broadcast graphic into a copy of the pack,
so the fact gate will accept that name afterwards, and returns a new pack
rather than reaching into the frozen one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from commentary.config import RESEARCHER_MODEL
from commentary.gate import fold
from commentary.llm.base import LLMBackend, Usage
from commentary.prompts.researcher import researcher_blocks, researcher_system
from commentary.schemas import KnowledgePack, Player, Side, TeamSheet

#: Anthropic's server-side web search tool. The model issues the searches and
#: the results never pass through this process, which is why the researcher can
#: reach the open web without this repository growing an HTTP client.
#:
#: ``web_search_20260209`` is the current version and the one Opus 5 supports;
#: ``web_search_20250305`` is the older basic variant, kept below because a
#: model that rejects the new type needs it and the version string is the only
#: thing that differs.
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 8,
}

#: The older tool version, for a model that does not take the one above.
BASIC_WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 8,
}

#: Where packs live. Relative on purpose: a pack belongs to a checkout, not to
#: a machine, and the same path works in the container.
PACKS_DIR = Path(os.getenv("PACKS_DIR", "packs"))


# -- freezing ------------------------------------------------------------
#
# Pydantic's frozen config is per class, not per instance, so freezing an
# object means rebuilding it as a frozen type. These three subclasses exist for
# no other reason. They are private because nothing should ever annotate
# against them: a frozen pack is a KnowledgePack everywhere it is used, and the
# only thing that changes is that assigning to it raises.


class _FrozenPlayer(Player):
    model_config = ConfigDict(frozen=True)


class _FrozenTeamSheet(TeamSheet):
    model_config = ConfigDict(frozen=True)


class _FrozenPack(KnowledgePack):
    model_config = ConfigDict(frozen=True)


def freeze(pack: KnowledgePack) -> KnowledgePack:
    """Return the pack as it exists after kickoff: read-only, all the way down.

    Every model in the returned tree — the pack, both team sheets, every player
    — raises on attribute assignment. That is the whole contract of this
    system stated in a way the interpreter can enforce: once the whistle has
    gone, nothing may edit the notes, because an edit during a match could only
    have come from outside the broadcast.

    What this does not freeze is the list and dict objects themselves;
    ``pack.storylines.append(...)`` still works, because Pydantic validates
    ``list[str]`` into a list and there is no immutable list to validate into
    without changing ``schemas.py``. That is the lesser hole by a distance: a
    pack is corrupted by somebody reassigning ``pack.home`` or
    ``sheet.starters`` wholesale, not by an append, and the reassignment is
    exactly what is now impossible.

    One sharp edge, since it is invisible: Pydantic's ``__eq__`` compares
    classes, so ``freeze(pack) != pack`` even though every field matches.
    Compare ``model_dump()`` if two packs ever need comparing. Nothing at
    runtime does, and serialising a frozen pack is unaffected — it writes and
    reloads as an ordinary :class:`KnowledgePack`.

    Calling this twice is harmless — it rebuilds an equal, equally frozen pack.
    """
    fields = dict(pack)
    fields["home"] = _freeze_sheet(pack.home)
    fields["away"] = _freeze_sheet(pack.away)
    return _FrozenPack(**fields)


def _freeze_sheet(sheet: TeamSheet) -> _FrozenTeamSheet:
    fields = dict(sheet)
    fields["starters"] = [_FrozenPlayer(**dict(p)) for p in sheet.starters]
    fields["bench"] = [_FrozenPlayer(**dict(p)) for p in sheet.bench]
    return _FrozenTeamSheet(**fields)


def is_frozen(pack: KnowledgePack) -> bool:
    """Whether this pack has been through :func:`freeze`, i.e. whether we are live."""
    return bool(pack.model_config.get("frozen", False))


# -- persistence ---------------------------------------------------------


def pack_path(home: str, away: str, directory: Path | str = PACKS_DIR) -> Path:
    """A stable filename for a fixture, so a second run finds the first one's work.

    Only the two team names go into the name. Adding the date would be more
    precise and would also mean a pack researched last night is not found this
    afternoon, which defeats the point of saving it at all.
    """
    return Path(directory) / f"{_slug(home)}-v-{_slug(away)}.json"


def _slug(name: str) -> str:
    return "-".join(fold(name).split()) or "unknown"


def save_pack(pack: KnowledgePack, path: Path | str) -> Path:
    """Write the pack to disk as JSON, and return where it went.

    Written to a neighbouring temporary file and renamed into place, because
    the one thing worse than no pack on matchday is half a pack: a process
    killed mid-write would otherwise leave a file that exists, fails to parse,
    and sends the run back to the model at kickoff.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(pack.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return target


def load_pack(path: Path | str) -> KnowledgePack:
    """Read a pack back. Validation is the point — a stale schema fails loudly here.

    The pack comes back unfrozen, because loading is a pre-match act and the
    freeze belongs at kickoff. Call :func:`freeze` when the whistle goes.
    """
    return KnowledgePack.model_validate_json(Path(path).read_text(encoding="utf-8"))


# -- the agent -----------------------------------------------------------


class Researcher:
    """One model call, with the web behind it, that produces the match notes.

    Unlike every other agent here this one is not on a clock. It runs before
    kickoff with nobody waiting, so it asks for a large output budget and takes
    as long as it takes; the tight timeouts on :class:`AnthropicBackend` are
    tuned for a caller that must answer inside four seconds, and if a pack ever
    times out the fix is a longer-lived backend, not a shorter prompt.

    THE WEB SEARCH LIMITATION, WRITTEN DOWN RATHER THAN PAPERED OVER.
    ``LLMBackend.parse`` takes no ``tools`` argument, and widening that
    protocol for the one agent that needs it would put a tool parameter on the
    caller, the board reader and the fact gate's model-free path, where it
    means nothing. So the tool configuration lives here, on
    :attr:`search_tool`, and reaches the request through one narrow hook: if
    the backend exposes a mutable ``extra_params`` mapping that it merges into
    the request, this agent sets ``tools`` in it for the duration of the call
    and restores it afterwards. ``AnthropicBackend`` does not expose that hook
    today. Until it does — a one-line ``params.update(self.extra_params)`` in
    ``anthropic_backend.py``, which this agent does not own — a real run
    researches from the model's own knowledge with no live search, which is
    worse for squad numbers after a transfer window and fine for most else.
    :attr:`used_search` records which of the two actually happened, so a pack
    can be read knowing whether anything was looked up to build it. A
    ``ScriptedBackend`` has no such hook either and is unaffected.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        model: str = RESEARCHER_MODEL,
        max_tokens: int = 8192,
        effort: str | None = None,
        search_tool: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        #: Two full squads of JSON is a lot of output, and a truncated pack is
        #: a failed pack. This is the one agent where generosity is free.
        self.max_tokens = max_tokens
        self.effort = effort
        #: Override to pin :data:`BASIC_WEB_SEARCH_TOOL`, to add a domain
        #: filter, or set to ``None`` to research with no search at all.
        self.search_tool: dict[str, Any] | None = (
            WEB_SEARCH_TOOL if search_tool is None else search_tool
        )
        #: Whether the last call actually had the search tool attached.
        self.used_search = False
        #: What the last pack cost. A pack is the most expensive single call
        #: this system makes, and it is the one nobody sees happen.
        self.last_usage: Usage | None = None

    async def research(
        self,
        home: str,
        away: str,
        *,
        competition: str = "",
        when: str = "",
    ) -> KnowledgePack:
        """Go and read about a fixture, once, and come back with the notes.

        Errors are not caught. A caller that fails mid-match is a missed line
        and the director shrugs; a researcher that fails means there are no
        notes, and starting a match with no notes should be a decision somebody
        makes rather than a default that arrives quietly.
        """
        system = researcher_system()
        blocks = researcher_blocks(home, away, competition, when)
        tools = [self.search_tool] if self.search_tool else None
        with _tools_enabled(self.backend, tools) as attached:
            self.used_search = attached
            parsed = await self.backend.parse(
                model=self.model,
                system=system,
                blocks=blocks,
                output_format=KnowledgePack,
                max_tokens=self.max_tokens,
                effort=self.effort,
                cache_system=True,
                tag="researcher",
            )
        self.last_usage = parsed.usage
        return parsed.value

    async def load_or_research(
        self,
        path: Path | str | None,
        home: str,
        away: str,
        *,
        competition: str = "",
        when: str = "",
    ) -> KnowledgePack:
        """The pack on disk if there is one, otherwise a fresh one, saved.

        This is the entry point a run should use. Research is idempotent from
        the outside and expensive on the inside, so the file is the cache and
        deleting it is how you ask for a re-read — team news lands an hour
        before kickoff, and a pack built yesterday has yesterday's eleven.

        ``path`` may be ``None`` for the default name under ``packs/``.
        """
        target = Path(path) if path is not None else pack_path(home, away)
        if target.exists():
            return load_pack(target)
        pack = await self.research(home, away, competition=competition, when=when)
        save_pack(pack, target)
        return pack


@contextmanager
def _tools_enabled(backend: LLMBackend, tools: list[dict[str, Any]] | None) -> Iterator[bool]:
    """Attach server-side tools to one call if the backend has anywhere to put them.

    The narrow escape hatch described on :class:`Researcher`. The duck-typed
    contract is a single attribute: a mutable ``extra_params`` mapping that the
    backend merges into its request. Setting it is scoped to the call and
    always undone, because leaving a web search tool attached to a backend
    shared with the caller would hand a live match an outside line — the one
    thing this system is built not to have.

    Yields whether the tools were attached, so the researcher can record
    honestly what produced the pack.
    """
    extra = getattr(backend, "extra_params", None)
    if not tools or not isinstance(extra, dict):
        yield False
        return
    previous = extra.get("tools")
    extra["tools"] = tools
    try:
        yield True
    finally:
        if previous is None:
            extra.pop("tools", None)
        else:
            extra["tools"] = previous


# -- learning a name mid-match -------------------------------------------


def update_from_substitution(
    pack: KnowledgePack,
    number: int | None,
    name: str,
    side: Side,
) -> KnowledgePack:
    """Fold a player read off a substitution graphic into a copy of the pack.

    The plan says the researcher runs again when a substitution appears on
    screen. It cannot: that would be an outside call during a match. This is
    the half that can be done offline, and it is the half that matters — the
    fact gate rejects any name not on a sheet, so a substitute the researcher
    missed is a player the commentators are structurally unable to name, no
    matter how clearly the broadcast has just captioned him. Folding the
    caption in restores that ability and costs nothing.

    The new player goes on the bench, which is where a substitute belongs
    anyway and which keeps the researched starting eleven untouched. A name
    already on either the starters or the bench is a no-op: the pack comes back
    unchanged, because the researcher's spelling and position are better
    sourced than a name read off a caption at 15 fps.

    Returns a new pack. If the input was frozen the output is frozen too, so a
    pack updated at 70 minutes is exactly as immutable as the one that kicked
    off — the update replaces the reference, it never edits in place.
    """
    sheet = pack.team(side)
    if sheet is None:
        raise ValueError(f"a substitution needs a known side, not {side.value!r}")
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("a substitution needs a name")
    if _already_known(sheet, cleaned):
        return pack

    sheet_fields = dict(sheet)
    sheet_fields["bench"] = [*sheet.bench, Player(name=cleaned, number=number)]
    pack_fields = dict(pack)
    pack_fields["home" if side is Side.HOME else "away"] = TeamSheet(**sheet_fields)
    updated = KnowledgePack(**pack_fields)
    return freeze(updated) if is_frozen(pack) else updated


def _already_known(sheet: TeamSheet, name: str) -> bool:
    """Is this name on the sheet already, allowing for accents and initials?

    Folded with the fact gate's own :func:`~commentary.gate.fold`, deliberately:
    the question being asked is "would the gate already accept this name", so
    it has to be asked in the gate's spelling and not in a second, subtly
    different one of this module's invention.
    """
    wanted = fold(name)
    if not wanted:
        return False
    surname = wanted.rsplit(" ", 1)[-1]
    for player in sheet.squad:
        folded = fold(player.name)
        if wanted == folded or surname == fold(player.surname):
            return True
    return False
