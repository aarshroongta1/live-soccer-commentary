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

import logging
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ConfigDict

from commentary.config import RESEARCHER_MODEL
from commentary.gate import fold, is_the_same_name
from commentary.llm.base import LLMBackend, Usage
from commentary.prompts.researcher import (
    notes_blocks,
    notes_system,
    researcher_blocks,
    researcher_system,
)
from commentary.schemas import KnowledgePack, Note, NoteSheet, Player, Side, TeamSheet

log = logging.getLogger(__name__)

#: Anthropic's server-side web search tool. The model issues the searches and
#: the results never pass through this process, which is why the researcher can
#: reach the open web without this repository growing an HTTP client.
WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 8,
    # Without this the first real notes pass died on "no text block in
    # response" after several minutes of Opus and a billed call. This tool
    # version defaults to being callable from inside a programmatic tool loop,
    # and a model that takes that route answers in tool blocks and never
    # writes the structured output this code reads. "direct" is the plain
    # server-side search: the model asks, the server answers, and the final
    # text block is the pack. A model that cannot do programmatic calling at
    # all — Haiku — refuses the request outright without it.
    "allowed_callers": ["direct"],
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


class _FrozenNote(Note):
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
    fields["notes"] = [_FrozenNote(**dict(note)) for note in pack.notes]
    return _FrozenPack(**fields)


def _freeze_sheet(sheet: TeamSheet) -> _FrozenTeamSheet:
    fields = dict(sheet)
    fields["starters"] = [_FrozenPlayer(**dict(p)) for p in sheet.starters]
    fields["bench"] = [_FrozenPlayer(**dict(p)) for p in sheet.bench]
    return _FrozenTeamSheet(**fields)


def is_frozen(pack: KnowledgePack) -> bool:
    """Whether this pack has been through :func:`freeze`, i.e. whether we are live."""
    return bool(pack.model_config.get("frozen", False))


# -- notes, and who they are about ---------------------------------------
#
# A note is only usable if its subject can be found again. The runtime looks
# notes up by the names on the screen, and the fact gate looks them up by the
# names in a line, so a note filed under "Kylian" or "the French captain" is
# a note nobody will ever be handed. That is not worth failing a pack over —
# a pack with one unusable note is still a pack — so the mismatch is reported
# and, on a freshly researched pack, dropped.


def note_subject(pack: KnowledgePack, about: str) -> str | None:
    """The exact pack name a note's ``about`` refers to, or ``None``.

    Exact first, then the fact gate's own name matching, so that a researcher
    that wrote "Mbappe" for "Kylian Mbappé" is understood rather than thrown
    away. The gate's matcher is deliberately the one used: the question being
    asked is "will this note ever be found again by the code that looks notes
    up", and that code has to agree with the gate about who a surname is.
    """
    wanted = about.strip()
    if not wanted:
        return None
    known = [pack.home.name, pack.away.name] + [
        player.name for sheet in (pack.home, pack.away) for player in sheet.squad
    ]
    for name in known:
        if wanted == name:
            return name
    folded = fold(wanted)
    for name in known:
        if folded and folded == fold(name):
            return name
    for name in known:
        if is_the_same_name(wanted, name):
            return name
    return None


def check_notes(pack: KnowledgePack) -> list[Note]:
    """Every note whose subject is on neither team sheet and is neither team.

    Returns rather than raises. Old packs have no notes and pass trivially;
    a hand-edited pack with one bad subject should still load.
    """
    return [note for note in pack.notes if note_subject(pack, note.about) is None]


def settle_notes(pack: KnowledgePack) -> tuple[KnowledgePack, list[Note]]:
    """Drop the notes nobody can look up, and rewrite the rest onto exact names.

    The second half matters as much as the first. ``about`` is a key, and a
    key that is nearly right is a key that misses: the runtime asks for the
    notes about "Kylian Mbappé" because that is the spelling on the sheet,
    and a note filed under "Mbappe" would sit in the pack unread.

    Returns the settled pack and the notes that were dropped, so a caller can
    say what it lost rather than losing it quietly.
    """
    kept: list[Note] = []
    dropped: list[Note] = []
    for note in pack.notes:
        subject = note_subject(pack, note.about)
        if subject is None:
            dropped.append(note)
            continue
        kept.append(note if subject == note.about else note.model_copy(update={"about": subject}))
    if not dropped and all(a.about == b.about for a, b in zip(kept, pack.notes, strict=True)):
        return pack, []
    fields = dict(pack)
    fields["notes"] = kept
    return KnowledgePack(**fields), dropped


# -- merging, checking, and the list a human ticks -----------------------
#
# Two facts about a pack of forty-odd notes that a pack of thirteen never had
# to face. The first is that nobody has read them. A researcher writes what it
# can source and marks how sure it was; a human reads the list before kickoff
# and ticks the ones that survive, because the fact gate downstream treats
# every note as verified truth and cannot tell a checked one from a guess.
# Until somebody ticks them they are `checked=False`, and the runtime leaves
# them alone.
#
# The second is that a pack already has notes worth more than anything a fresh
# call will produce — the ones a human checked last time. A second research
# pass must not be able to lose them, or to quietly replace one with a model's
# rewording of the same fact.


#: Below this, a note goes to the top of the hand-check list. The researcher
#: sets ``confidence`` itself and is the only thing that knows how hard the
#: number was to find, so this is not a filter — nothing is dropped for being
#: shaky — it is an ordering, so the twenty minutes a human has go where they
#: are worth most.
SHAKY_CONFIDENCE = 0.7


@dataclass(frozen=True)
class Merge:
    """What a research pass did to a pack that already had notes in it."""

    #: The merged list, hand-checked notes first, in pack order.
    notes: list[Note]
    #: Hand-checked notes carried through untouched.
    kept: int
    #: Fresh notes taken.
    added: int
    #: Hand-checked notes that gained a no-number form from the fresh pass.
    clauses: int
    #: Fresh notes thrown away for saying what a hand-checked note already said.
    duplicates: list[Note]


def merge_notes(existing: Sequence[Note], fresh: Sequence[Note]) -> Merge:
    """Hand-checked notes survive a research pass; everything else is replaced.

    Three rules, and the order matters.

    A note somebody has checked is kept exactly as it is. It is the only part
    of a pack anybody has verified, and a fresh call has no standing to
    reword it, re-source it or change its number.

    A fresh note that repeats a hand-checked one is dropped — except for the
    one thing it may contribute. Notes written before the ``clause`` field
    existed have no no-number form, and the colour seat cannot say any of
    them; the prompt asks for those back with the text copied exactly, and a
    fresh note that matches a hand-checked one word for word donates its
    clause and nothing else.

    Everything else on the pack that nobody checked is dropped rather than
    merged, which is what keeps a second run over the same pack idempotent:
    twice gives one set of notes, not two.
    """
    kept = [note for note in existing if note.checked]
    by_key: dict[tuple[str, str], int] = {
        (fold(note.about), fold(note.text)): index for index, note in enumerate(kept)
    }
    clauses = 0
    duplicates: list[Note] = []
    added: list[Note] = []
    for note in fresh:
        index = by_key.get((fold(note.about), fold(note.text)))
        if index is None:
            added.append(note if not note.checked else note.model_copy(update={"checked": False}))
            continue
        # The same note back again. All it may give us is the no-number form
        # the older one was written without.
        if note.clause and not kept[index].clause:
            kept[index] = kept[index].model_copy(update={"clause": note.clause})
            clauses += 1
        else:
            duplicates.append(note)
    return Merge(
        notes=[*kept, *added],
        kept=len(kept),
        added=len(added),
        clauses=clauses,
        duplicates=duplicates,
    )


def above_trust_floor(pack: KnowledgePack, *, floor: float) -> tuple[KnowledgePack, int]:
    """The pack under ``--trust-unchecked``, with the shakiest notes still held back.

    A hand-checked note is untouched here whatever its confidence — checking
    is the verification, and a floor on top of it would be judging a person's
    look at the source rather than the researcher's own guess. An unchecked
    note is kept only when its ``confidence`` clears ``floor``; below it, it
    is exactly as absent as it would be with ``--trust-unchecked`` off. A note
    at confidence 0.1 that the researcher itself was not sure of, and that
    turned out to be wrong, is what this stops from reaching air on a
    rehearsal.

    Returns the pack unchanged, and zero, when nothing falls below the floor,
    so the common case allocates nothing and compares equal.
    """
    below = [note for note in pack.notes if not note.checked and note.confidence < floor]
    if not below:
        return pack, 0
    fields = dict(pack)
    fields["notes"] = [note for note in pack.notes if note.checked or note.confidence >= floor]
    return KnowledgePack(**fields), len(below)


def only_checked(pack: KnowledgePack) -> tuple[KnowledgePack, int]:
    """The pack with every unchecked note removed, and how many went.

    The default on the way into a match. A note is a thing said out loud in a
    confident voice with a number in it, and the gate accepts it because it is
    in the pack — so "in the pack" has to mean somebody looked at it. A
    researched pack is forty notes nobody has read yet; ``--trust-unchecked``
    is how a rehearsal says it does not care.

    Returns the pack unchanged, and zero, when every note is checked, so the
    common case allocates nothing and compares equal.
    """
    unchecked = [note for note in pack.notes if not note.checked]
    if not unchecked:
        return pack, 0
    fields = dict(pack)
    fields["notes"] = [note for note in pack.notes if note.checked]
    return KnowledgePack(**fields), len(unchecked)


def hand_check_list(pack: KnowledgePack) -> str:
    """Every note, grouped by who it is about, in the order a human reads them.

    This is the deliverable of a research pass as much as the JSON is. Forty
    notes in a file are forty unverified claims; the same forty printed under
    the player's name, with the source beside each one and the shaky ones
    flagged, is twenty minutes of work with a definite end.

    Grouped by subject in team-sheet order — home team, then its starters in
    the order they were listed, then its bench, then the same for the away
    side — because that is how a person checks: one player, every claim about
    him, one source page open.
    """
    order = _subject_order(pack)
    grouped: dict[str, list[Note]] = {}
    for note in pack.notes:
        grouped.setdefault(note.about, []).append(note)
    known = [name for name in order if name in grouped]
    stray = [name for name in grouped if name not in set(order)]

    total = len(pack.notes)
    shaky = sum(1 for note in pack.notes if note.confidence < SHAKY_CONFIDENCE)
    lines = [
        f"HAND-CHECK LIST — {total} notes about {len(grouped)} subjects, "
        f"{sum(1 for n in pack.notes if n.checked)} already checked, "
        f"{shaky} under {SHAKY_CONFIDENCE:g} confidence",
        "Tick a note by setting \"checked\": true on it in the pack.",
    ]
    for name in [*known, *stray]:
        lines.append("")
        lines.append(f"{name}")
        for note in grouped[name]:
            lines.extend(_hand_check_rows(note))
    return "\n".join(lines)


def _hand_check_rows(note: Note) -> list[str]:
    """One note as a person reads it: the claim, then what backs it."""
    box = "[x]" if note.checked else "[ ]"
    flag = "  << LOW" if note.confidence < SHAKY_CONFIDENCE else ""
    tags = note.kind + (f", counts {note.counts}" if note.counts else "")
    rows = [f"  {box} {note.text}   [{tags}, confidence {note.confidence:.2f}]{flag}"]
    if note.clause:
        rows.append(f"      no number: {note.clause}")
    elif note.has_figure:
        rows.append("      no number: — MISSING, the colour seat cannot say this one")
    rows.append(f"      source: {note.source or '— none given'}")
    return rows


def _subject_order(pack: KnowledgePack) -> list[str]:
    """Every name a note may be filed under, in team-sheet order."""
    order: list[str] = []
    for sheet in (pack.home, pack.away):
        order.append(sheet.name)
        order.extend(player.name for player in sheet.starters)
        order.extend(player.name for player in sheet.bench)
    return order


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


def researched_path(pack: Path | str) -> Path:
    """Where a research pass over an existing pack writes its result.

    Beside the pack it read, with ``-researched`` on the end, and never over
    the top of it. A pack with hand-checked notes in it is the most expensive
    object in this repository — somebody sat and looked forty claims up — and
    the way to lose it is to make a model call's output the default
    destination.
    """
    path = Path(pack)
    return path.with_name(f"{path.stem}-researched{path.suffix or '.json'}")


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

    Notes are checked but never enforced here. A pack on disk may have been
    written by hand, by an older version of this code, or by a researcher
    that spelled a name its own way; none of those is a reason to refuse to
    call the match. The unusable ones are logged and left where they are, and
    the runtime simply never finds them.
    """
    pack = KnowledgePack.model_validate_json(Path(path).read_text(encoding="utf-8"))
    for note in check_notes(pack):
        log.warning("note about %r is on no team sheet: %s", note.about, note.text)
    return pack


# -- the agent -----------------------------------------------------------


class Researcher:
    """One model call, with the web behind it, that produces the match notes.

    Unlike every other agent here this one is not on a clock. It runs before
    kickoff with nobody waiting, so it asks for a large output budget and takes
    as long as it takes; the tight timeouts on :class:`AnthropicBackend` are
    tuned for a caller that must answer inside four seconds, and if a pack ever
    times out the fix is a longer-lived backend, not a shorter prompt.

    HOW WEB SEARCH GETS TURNED ON, GIVEN THE PROTOCOL DOES NOT CARRY IT.
    ``LLMBackend.parse`` takes no ``tools`` argument, and widening the protocol
    for the one agent that needs it would put a tool parameter on the caller,
    the board reader and the fact gate's model-free path, where it means
    nothing. So the tool configuration lives here, on :attr:`search_tool`,
    where the agent that understands it can own it, and reaches the request
    through one narrow hook: a backend may expose a mutable ``extra_params``
    mapping that it merges into every request, and this agent sets ``tools``
    in it for the duration of the call and restores it afterwards.

    ``AnthropicBackend`` exposes that mapping, and merges it *first*, so the
    hook can add a tool but cannot reach the model, the output format or the
    caching — which is what keeps it narrow enough to be worth having. A
    backend without the mapping, ``ScriptedBackend`` included, is unaffected
    and simply researches with no search rather than failing.

    :attr:`used_search` records which of the two actually happened on the last
    call, so a pack can be read knowing whether anything was looked up to build
    it — a pack built without search has the model's own knowledge of squad
    numbers behind it, which is the part that goes stale after a transfer
    window.
    """

    def __init__(
        self,
        backend: LLMBackend,
        *,
        model: str = RESEARCHER_MODEL,
        max_tokens: int = 32000,
        effort: str | None = None,
        search_tool: dict[str, Any] | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        #: Two full squads of JSON is a lot of output, and a truncated pack is
        #: a failed pack. This is the one agent where generosity is free.
        #: Raised twice on the way to the first fifty-note pack, and the
        #: second time is the instructive one: adaptive thinking is on for
        #: this model and spends the same budget the answer is written from,
        #: so a 16,384 ceiling produced minutes of reasoning, six searches,
        #: and JSON that stopped mid-string. The cap is not a cost control —
        #: nothing is billed for room that goes unused — and a truncated pack
        #: costs the whole call.
        self.max_tokens = max_tokens
        self.effort = effort
        #: Override to add a domain filter, or set to ``None`` to research
        #: with no search at all.
        self.search_tool: dict[str, Any] | None = (
            WEB_SEARCH_TOOL if search_tool is None else search_tool
        )
        #: Whether the last call actually had the search tool attached.
        self.used_search = False
        #: What the last pack cost. A pack is the most expensive single call
        #: this system makes, and it is the one nobody sees happen.
        self.last_usage: Usage | None = None
        #: Notes the last call filed under a name that is on neither sheet,
        #: and which were therefore thrown away. Kept so the command that
        #: wrote the pack can print how many were lost rather than pretending
        #: the model did as it was told.
        self.dropped_notes: list[Note] = []
        #: What the last notes pass did to the notes that were already in the
        #: pack: how many hand-checked ones it kept, how many it added, how
        #: many missing no-number forms it filled in, what it threw away as a
        #: repeat. ``None`` until :meth:`write_notes` has run.
        self.last_merge: Merge | None = None

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
        # A note about a name that is on no sheet is unreachable by every
        # consumer — the runtime looks notes up by roster name and the fact
        # gate checks a claim against notes about the names in the line — so
        # it is not a note, it is a sentence nobody will ever read. Dropped
        # here rather than downstream, because this is the last moment at
        # which anything knows it existed.
        pack, dropped = settle_notes(parsed.value)
        self.dropped_notes = dropped
        for note in dropped:
            log.warning(
                "dropping note about %r, who is on no team sheet: %s", note.about, note.text
            )
        return pack

    async def write_notes(self, pack: KnowledgePack) -> KnowledgePack:
        """Ask for notes about a pack that already exists, and return it with them.

        A second, much cheaper research pass for a pack whose team sheets are
        already right. It exists because the sheets are the expensive part and
        the part that goes stale: a pack built the day before kickoff has
        correct squad numbers in it, and re-running the whole researcher to
        add context would risk them for no reason.

        Replaces the notes rather than appending to them, so the command is
        idempotent — running it twice gives one set of notes, not two — with
        the one exception that is the whole reason this is a merge and not an
        assignment: a note somebody has ticked survives. See
        :func:`merge_notes`. What the merge did is left on
        :attr:`last_merge`, so the command can say it rather than the user
        having to diff two files.
        """
        tools = [self.search_tool] if self.search_tool else None
        with _tools_enabled(self.backend, tools) as attached:
            self.used_search = attached
            parsed = await self.backend.parse(
                model=self.model,
                system=notes_system(),
                blocks=notes_blocks(pack),
                output_format=NoteSheet,
                max_tokens=self.max_tokens,
                effort=self.effort,
                cache_system=True,
                tag="notes",
            )
        self.last_usage = parsed.usage
        merge = merge_notes(pack.notes, parsed.value.notes)
        self.last_merge = merge
        fields = dict(pack)
        fields["notes"] = merge.notes
        settled, dropped = settle_notes(KnowledgePack(**fields))
        self.dropped_notes = dropped
        for note in dropped:
            log.warning(
                "dropping note about %r, who is on no team sheet: %s", note.about, note.text
            )
        return settled

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

    The narrow escape hatch described on :class:`Researcher`. The contract is
    one attribute, checked for rather than required: a mutable ``extra_params``
    mapping that the backend merges into its request, which
    :class:`~commentary.llm.anthropic_backend.AnthropicBackend` has and a
    scripted backend does not.

    Setting it is scoped to the call and always undone, including when the call
    raises, because leaving a web search tool attached to a backend the caller
    shares would hand a live match an outside line — the one thing this system
    is built not to have.

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
