"""What this broadcast has seen happen, counted in code.

``docs/research/real-commentary-corpus.md`` section 5.1: one real utterance in
six carries a number, about two a minute for the whole ninety. Section 5.2
breaks those numbers down, and after the scoreline and the clock the next
categories are all *counts* — a side's form, a player's tally, a run of
something happening again. The corpus says them plainly:

    "In the end it comes off the legs of Danny Simpson and out for a United
     corner, the second of the game."           (lei-mun-2015 23:08)
    "...forced to head it behind for a seventh corner of the game."
                                                (lei-mun-2015 80:29)
    "...should potentially have been sent off before Arturo Vidal was shown a
     second yellow card."                       (clasico-2017 17:35)

This system had nowhere for a number like that to come from. The pack's notes
are researched before kickoff and :mod:`commentary.tallies` moves the ones the
match can move, but neither knows how many corners France have had tonight. So
the model was left with two choices, invent one or say none; the gate exists to
make the first impossible, which leaves none. Two lines in the whole 27-line
Mbappé rephrase carried a number outside the scoreline.

The ledger is the third source, and it keeps the rule the handoff's section 5
sets for all of them: **code writes numbers, the model writes words.** Every
count here is arithmetic over things the system already recorded — the caller's
forms and the state's incidents — handed out as a finished clause the phraser
may reword but not renumber, and checked back against this object afterwards by
:meth:`commentary.gate.FactGate.judge` under ``ledger_claim``.

**These are not official statistics and the docstrings have to keep saying so.**
A ledger count is *what this system saw*: the caller files a form every couple
of seconds off a broadcast camera, so a corner taken under a replay, a foul off
screen and a shot in a wide shot the caller read as build-up are all invisible
here. Against a real statistician the shot count will be low and the touch
count a small fraction. That is honest and it is sayable — "France's fourth
corner" is true of the four this broadcast showed — but nothing here may be
presented as a match stat. :meth:`Ledger.override` is the documented hook where
a wire feed (``commentary feed``, the ESPN adapter) replaces a count with the
statistician's own; until something calls it, every number is the picture's.

**And which side a thing belongs to is the caller's ``side`` field**, which
says which team the line is about and not who was awarded what. It flips
inside one event — on the Mbappé trace the penalty was filed `away` four times
and then `home` — so a run of forms is one occurrence whatever the side says
partway through, and the side is the one the first look gave it. Where even
that is not safe the count is simply not offered per side: see
:data:`SIDE_KINDS` on saves.

One counter, not two. :func:`~commentary.agents.colour.patterns_in` used to
count the same things over a window of forms for the colour seat's material; it
asks :meth:`Ledger.patterns` now, so the spell count and the match count come
out of one list of occurrences and cannot disagree. Goals are the same story
from the other end: a player's goals in this match are
:class:`~commentary.tallies.Tallies`' business, because a note that counts goals
has to move by exactly the number the ledger holds — so :attr:`tallies` is
shared and :meth:`goals` delegates rather than counting twice.

Both the runtime and the offline rephrase build one of these as they walk the
match, the way they both build a :class:`~commentary.threads.Threads`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace

from commentary.gate import COUNT_NOUNS, fold, is_the_same_name, noun_for
from commentary.schemas import CallerLine, Event, KnowledgePack, MatchState, Scene, Side
from commentary.tallies import Tallies

#: The two sides of the pitch, as the words a caller writes. Read off the
#: lead's own prose, because a caller form carries a team and a scene and no
#: geography at all.
FLANKS = ("left", "right")

#: Which ledger count each caller event moves. Events not here — a pass, a
#: carry, an interception, build-up, a kickoff, a stoppage — are the ball
#: moving rather than something happening, and nobody counts those out loud.
#: ``switch`` is absent for the same reason: the corpus says it as a
#: direction, never as a tally.
KIND_OF_EVENT: dict[Event, str] = {
    Event.SHOT: "shot",
    Event.SAVE: "save",
    Event.GOAL: "goal",
    Event.CORNER: "corner",
    Event.FREE_KICK: "free_kick",
    Event.PENALTY: "penalty",
    Event.FOUL: "foul",
    Event.OFFSIDE: "offside",
    Event.THROW_IN: "throw_in",
    Event.CARD: "card",
    Event.SUBSTITUTION: "substitution",
    Event.CLEARANCE: "clearance",
    Event.TACKLE: "tackle",
    Event.CROSS: "cross",
}

#: What a side may be given a count of. A goal is absent on purpose: a side's
#: goals are the scoreline, the board owns it, and
#: :meth:`commentary.gate.FactGate._check_scoreline` and the ordinal rule
#: beside it already judge every way of saying it. A second counter for the
#: same number would be a second thing to be wrong.
#:
#: A save is absent too, and for a duller reason: which side a save belongs to
#: is the one thing the caller's form is unreliable about — it is the keeper's
#: side and the form is usually filled in for the side attacking — and nobody
#: says "France's fourth save" anyway. A keeper's own saves are counted, under
#: :data:`PLAYER_KINDS`, where the name settles the side. Clearances and
#: tackles are left off for the second half of that: they are counted per man
#: and never per team.
SIDE_KINDS: tuple[str, ...] = (
    "shot",
    "shot_on_target",
    "corner",
    "free_kick",
    "penalty",
    "foul",
    "offside",
    "throw_in",
    "card",
    "substitution",
    "cross",
)

#: What a player may be given a count of. Narrower than the side list,
#: because these are the ones a picture can credit to a man: he took the
#: shot, he gave the foul away, he was shown the card, he made the save. A
#: throw-in or a substitution belongs to a team.
PLAYER_KINDS: tuple[str, ...] = (
    "goal",
    "shot",
    "shot_on_target",
    "save",
    "foul",
    "card",
    "tackle",
)

#: A shot is on target when a save or a goal follows it this soon. The caller
#: files the shot and then, a look or two later, the keeper holding it or the
#: net bulging; eight seconds is the window ``state.py`` already uses to
#: decide that two goal reports are one goal, and it covers the two or three
#: seconds a caller round trip takes.
ON_TARGET_S = 8.0

#: How close two reports of one event may be and still be one event, when
#: they come from different places. The board sees a graphic, the wire sees
#: the ball cross, the caller sees the celebration, and all three are the same
#: goal. Two *pictures* are never merged by this: a run of consecutive forms
#: carrying one thing has already been folded into one occurrence on the way
#: in, and two shots six seconds apart are a shot and a rebound.
SAME_EVENT_S = 8.0

#: How long a spell is, for the colour seat's patterns. The window used to be
#: a count of forms — twelve, about forty-five seconds at the caller's rate —
#: and it is a clock here because the ledger holds occurrences, not forms.
SPELL_S = 45.0

#: How many times something has to happen in a spell before it is a pattern.
#: Two, which is the floor the corpus's own examples sit on: "unlike the
#: corners there ... they've gone zonally" is about the second corner.
PATTERN_FLOOR = 2

#: How long a side has to go without a shot before the drought is worth
#: saying. **A guess, written down as a guess.** The corpus restates the clock
#: and the score on timers this module can point at (section 5.3) and says
#: nothing about how long a shotless spell has to run. Three minutes is longer
#: than any gap between chances on the clips this system has run, so the
#: clause fires on a quiet passage and not on a normal one. Replace it with a
#: measured figure if one ever exists.
DROUGHT_S = 180.0

#: How many counts the lead's context block may carry. Two, where the notes
#: get three (:data:`commentary.threads.CONTEXT_NOTES`), because the block is
#: an offer and a menu of six is not one: the study's rate is about one number
#: every forty seconds of build-up, and a voice shown half a dozen either
#: picks at random or says none of them.
CONTEXT_FACTS = 2

#: How recently a count has to have moved to be worth saying at all.
#:
#: This used to gate only a *first*, and the first measured walk of a real
#: trace showed why that is not enough: "Second corner for Argentina." was
#: offered on thirty consecutive forms over two minutes, long after the
#: corner, because the count was still two. A count is news at the moment it
#: moves — the corpus says "out for a United corner, the second of the game"
#: *at the corner* — and half a minute later it is arithmetic.
#:
#: Twenty-five seconds, which is
#: :data:`commentary.agents.colour.EVENT_FRESH_S`: the window that module
#: already draws around "how old a thing may be and still be worth talking
#: about", off section 4.3's 21.4-second median delay to the second voice.
#: One window, both seats.
COUNT_FRESH_S = 25.0

_ORDINALS = (
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
)

_NUMBERS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)


def ordinal(value: int) -> str:
    """A count as the ordinal a commentator says. "fourth", not "4th"."""
    if 1 <= value <= len(_ORDINALS):
        return _ORDINALS[value - 1]
    if 11 <= value % 100 <= 13:
        return f"{value}th"
    return f"{value}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')}"


@dataclass(frozen=True)
class Occurrence:
    """One thing this broadcast was seen to do, once.

    Occurrences, not forms. The caller files the same corner on four looks in
    a row, and counting forms would give four corners — which is false, and is
    the kind of false a listener notices. It is the finding
    :func:`~commentary.agents.colour.patterns_in` was built around. A run of
    consecutive forms carrying the same thing is one occurrence, and
    :attr:`players` is every name read across that run.
    """

    kind: str
    ts: float
    side: Side = Side.UNKNOWN
    team: str = ""
    players: tuple[str, ...] = ()
    flank: str = ""
    #: ``"picture"`` for something the caller saw, ``"board"`` or ``"wire"``
    #: for something the state was told, ``"call"`` for a scorer the goal call
    #: named. Kept so that a trace can say which counts would survive without
    #: a feed, and so that :meth:`Ledger.override` has something to override.
    source: str = "picture"


@dataclass(frozen=True)
class Fact:
    """One count, and the clause that says it.

    ``about`` is a key the way :attr:`commentary.schemas.Note.about` is: the
    exact name of a player or of one of the two teams, so the gate can ask
    whether a line carrying a number also says whose number it is.

    ``text`` is written the way the corpus says it, and is what a voice is
    shown. The figure in it is the figure the gate holds the line to.
    """

    about: str
    kind: str
    count: int
    text: str
    #: Seconds since the thing this fact is about last happened, where the
    #: interval is part of the claim — a drought — or simply worth knowing.
    #: ``None`` where there is no earlier one to measure from.
    since_s: float | None = None
    side: Side = Side.UNKNOWN
    #: True when :attr:`about` is a player rather than a team.
    player: bool = False
    #: When the count last moved.
    ts: float = 0.0
    #: The same fact with the figure left out, for the colour seat, which may
    #: not say a number at all
    #: (:func:`~commentary.agents.colour.says_a_number`). Exactly what
    #: :attr:`commentary.schemas.Note.clause` is for and put here for the
    #: same measured reason: told "say the fact, never the figure" a foot
    #: away from the figure, the seat said the figure — five times across two
    #: traces, every one of them struck out. Empty on a count of one, because
    #: a first is not a repetition and this seat only speaks about what has
    #: happened again.
    clause: str = ""


@dataclass
class Ledger:
    """Every count this broadcast has been seen to hold, so far.

    Fed :meth:`saw_form` with every caller form in order, spoken or not — a
    form the caller filled in and chose not to speak still saw the corner —
    and :meth:`see_state` with the state as it changes, which is where goals,
    cards and anything a wire reported come from.

    Nothing here is an official statistic. See the module docstring.
    """

    home: str = ""
    away: str = ""
    #: Shared with :class:`~commentary.threads.Threads` wherever one exists,
    #: so a player's goals in this match are counted in one place only. See
    #: :meth:`goals`.
    tallies: Tallies = field(default_factory=Tallies)
    occurrences: list[Occurrence] = field(default_factory=list)
    #: Counts a statistician has overridden, keyed by folded subject and kind.
    #: See :meth:`override`.
    wire: dict[tuple[str, str], int] = field(default_factory=dict)
    #: Both squads, as the team sheets spell them. Every name read off the
    #: picture is resolved through this before it is counted or written into
    #: a clause. Without it one man is three counts — the caller read "T.
    #: Hernandez", "T. Hernández" and "Theo Hernández" on one 165-second
    #: trace — and, worse, the clause says whatever the shirt said: "DE
    #: PAUL's first tackle". Empty when there is no pack, which is a ledger
    #: counting whatever it is handed.
    roster: tuple[str, ...] = ()

    #: The run state of the ingest: what the last non-replay form was about,
    #: so that a run of consecutive forms carrying one thing is one thing.
    last_kind: str = ""
    #: When the run of forms carrying :attr:`last_kind` was last added to. A
    #: run is bounded in time as well as by what comes between: the colour
    #: seat's old counter read a window of a dozen forms filed seconds apart,
    #: where "consecutive" and "the same event" meant the same thing. The
    #: ledger reads the whole match, and two corners half a minute apart with
    #: nothing filed in between are two corners.
    last_kind_ts: float = 0.0
    last_players: frozenset[str] = frozenset()
    last_flanks: frozenset[tuple[str, str]] = frozenset()
    seen_reports: set[tuple[str, str, int]] = field(default_factory=set)

    @classmethod
    def from_pack(cls, pack: KnowledgePack | None, *, tallies: Tallies | None = None) -> Ledger:
        """Seed the two team names off a pack, when there is one."""
        return cls(
            home="" if pack is None else pack.home.name,
            away="" if pack is None else pack.away.name,
            tallies=tallies if tallies is not None else Tallies(),
            roster=()
            if pack is None
            else tuple(
                player.name
                for sheet in (pack.home, pack.away)
                for player in sheet.squad
                if player.name
            ),
        )

    # -- what it is told --------------------------------------------------

    def saw_form(self, ts: float, line: CallerLine) -> None:
        """File one caller form, spoken or not.

        A replay is the same thing again, not another one, so a replay form is
        skipped entirely — and does not break a run either, so a corner, its
        replay and the corner again is one corner. That is the rule
        :func:`~commentary.agents.colour.patterns_in` arrived at after one
        penalty and its replay were counted as two goals.
        """
        if line.scene is Scene.REPLAY:
            return
        side, team = self._side_of(line)
        players = frozenset(
            self.who(s.name) for s in line.sightings if s.name and s.name.strip()
        )
        said = _words(line)
        flanks = frozenset((team, flank) for flank in FLANKS if team and flank in said)

        kind = KIND_OF_EVENT.get(line.event, "")
        # The run is keyed on the kind alone and not on the side, because the
        # caller's ``side`` is which team the *line* is about and flips inside
        # one event: on the Mbappé trace the penalty was filed `away` over
        # four looks and then `home` on the fifth, which as two occurrences
        # read "Argentina's first penalty" about a kick France were taking.
        # One dead ball is one occurrence, and the side is the one the first
        # look gave it.
        again = kind == self.last_kind and ts - self.last_kind_ts <= SAME_EVENT_S
        if kind and not again:
            self._add(
                Occurrence(
                    kind=kind,
                    ts=ts,
                    side=side,
                    team=team,
                    players=tuple(sorted(players)),
                    flank=next((flank for _team, flank in sorted(flanks)), ""),
                )
            )
        elif kind:
            self._fold_in(kind, players)
        self.last_kind = kind
        if kind:
            self.last_kind_ts = ts

        for name in sorted(players - self.last_players):
            self._add(Occurrence(kind="touch", ts=ts, side=side, team=team, players=(name,)))
        self.last_players = players
        for team_name, flank in sorted(flanks - self.last_flanks):
            self._add(Occurrence(kind="flank", ts=ts, side=side, team=team_name, flank=flank))
        self.last_flanks = flanks

    def see_state(self, state: MatchState) -> None:
        """Take in the goals, cards and subs the state holds, and the wire's.

        The two team names come from here the first time, because a caller
        form carries a team only when the picture said so and the ledger has
        to be able to write "France" into a clause.

        Every report is taken once, keyed by what it is rather than by its
        place in the list, because the state rebuilds that list as the board
        and the wire disagree and re-agree about a goal.
        """
        self.home = self.home or state.home
        self.away = self.away or state.away
        self.tallies.see_state(state)
        reports: list[tuple[Event, Side, tuple[str, ...], float, str]] = [
            (
                incident.event,
                incident.side,
                (incident.player,) if incident.player else (),
                incident.video_ts,
                incident.source,
            )
            for incident in state.incidents
        ] + [
            (
                named.event,
                named.side,
                tuple(name for name in (named.player, named.recipient) if name),
                named.video_ts,
                "wire",
            )
            for named in state.named
        ]
        for event, side, players, video_ts, source in reports:
            kind = KIND_OF_EVENT.get(event, "")
            if not kind:
                continue
            key = (kind, fold(" ".join(players)), int(video_ts))
            if key in self.seen_reports:
                continue
            self.seen_reports.add(key)
            self._add(
                Occurrence(
                    kind=kind,
                    ts=video_ts,
                    side=side,
                    team=self._team_name(side),
                    players=players,
                    source=source,
                )
            )

    def credit_goal(self, player: str | None, ts: float, side: Side = Side.UNKNOWN) -> bool:
        """This man has scored, as the goal call named him.

        The board sees a graphic and never knows who, so on the ordinary path
        the goal *call* is the only thing that puts a name to a goal — which is
        why :class:`~commentary.goalfollow.GoalFollowup` works it out, and why
        the runtime and the offline rephrase both call this beside
        :meth:`commentary.threads.Threads.credit_goal`. The tally itself is not
        duplicated: :attr:`tallies` counts, here as there.
        """
        moved = self.tallies.credit_goal(player, ts)
        if moved and player:
            self._add(
                Occurrence(
                    kind="goal",
                    ts=ts,
                    side=side,
                    team=self._team_name(side),
                    players=(player,),
                    source="call",
                )
            )
        return moved

    def override(self, about: str, kind: str, count: int) -> None:
        """Replace a count with a statistician's own. The wire hook.

        Nothing calls this yet, and that is the honest state of it: the ESPN
        adapter (``commentary feed``) reports events, has never been run
        against a live fixture, and every number this object holds is what the
        broadcast showed — the picture missed whatever it missed. When a feed
        does arrive this is where its figure goes: keyed by subject and kind,
        consulted by :meth:`count` ahead of anything the caller saw, and never
        written back over the occurrences, so a trace can still say what the
        pictures alone were worth.
        """
        self.wire[(fold(about), kind)] = count

    # -- what it knows ----------------------------------------------------

    def count(self, about: str, kind: str) -> int:
        """How many of this thing belong to this player or this side."""
        override = self.wire.get((fold(about), kind))
        if override is not None:
            return override
        if kind == "goal" and not self.is_team(about):
            return self.goals(about)
        return len(self.matching(about, kind))

    def goals(self, player: str) -> int:
        """This man's goals in this match, from the one place that counts them."""
        return self.tallies.count(player, "goals")

    def last(self, about: str, kind: str) -> Occurrence | None:
        """The most recent one of these, or nothing."""
        found = self.matching(about, kind)
        return found[-1] if found else None

    def since(self, about: str, kind: str, now: float) -> float | None:
        """Seconds since this last happened, or ``None`` if it never has."""
        latest = self.last(about, kind)
        return None if latest is None else max(0.0, now - latest.ts)

    def matching(self, about: str, kind: str) -> list[Occurrence]:
        """Every occurrence of this kind belonging to this subject, oldest first."""
        subject = (about or "").strip()
        if not subject:
            return []
        team = self.is_team(subject)
        out = [
            item
            for item in self.occurrences
            if item.kind == kind
            and (
                (item.team and _same(item.team, subject))
                if team
                else any(_same(player, subject) for player in item.players)
            )
        ]
        return sorted(out, key=lambda item: item.ts)

    @property
    def teams(self) -> tuple[str, ...]:
        return tuple(name for name in (self.home, self.away) if name)

    def is_team(self, name: str) -> bool:
        return any(_same(name, team) for team in self.teams)

    # -- what goes in front of a voice ------------------------------------

    def facts(self, now: float, about: Iterable[str] = (), *, limit: int = 0) -> list[Fact]:
        """Every count worth saying about these people, best first.

        ``about`` arrives in the order that matters — the man on the ball,
        then whoever else was read, then the two teams — and the order is
        kept, so a clause about the player being watched wins a place over one
        about his country. That is the ordering
        :func:`~commentary.state.notes_for` uses for the same reason. Within
        one subject the freshest count comes first: a number is worth saying
        while the thing that moved it is still what the listener is thinking
        about.
        """
        out: list[Fact] = []
        seen: set[tuple[str, str]] = set()
        for name in about:
            cleaned = (name or "").strip()
            if not cleaned:
                continue
            found = (
                self._team_facts(self._canonical_team(cleaned), now)
                if self.is_team(cleaned)
                else self._player_facts(cleaned, now)
            )
            for fact in sorted(found, key=lambda item: -item.ts):
                key = (fold(fact.about), fact.kind)
                if key in seen:
                    continue
                seen.add(key)
                out.append(fact)
        return out[:limit] if limit else out

    def changed_since(self, since_ts: float, now: float, *, floor: int = 2) -> list[Fact]:
        """Side counts of ``floor`` or more that have moved since ``since_ts``.

        The colour seat's half of the material rule: a number that has not
        moved since the seat last spoke is not news, and a side's *first*
        anything is the lead's to say rather than the second voice's.
        """
        out: list[Fact] = []
        for team in self.teams:
            for kind in SIDE_KINDS:
                latest = self.last(team, kind)
                if latest is None or latest.ts <= since_ts:
                    continue
                total = self.count(team, kind)
                if total < floor:
                    continue
                out.append(self._side_fact(team, kind, total, latest))
        return sorted(out, key=lambda fact: -fact.ts)

    def patterns(self, since_ts: float, now: float) -> list[str]:
        """What has happened twice or more in this spell, with a name on it.

        The colour seat's material, and the reason this used to be counted in
        :mod:`commentary.agents.colour` over a window of forms. It is here now
        so the spell count and the match count are one arithmetic over one
        list: a seat told "3 corners in this spell" and a lead told "France's
        fourth corner" cannot contradict each other.

        Only what the caller saw. A goal the board reported and a goal the
        call named are the same goal as the one on the form, and counting all
        three would hand the seat "3 goals" off one penalty.

        The count stays in the string because the prompt has to be able to say
        "that is the third one" to itself in order to write "again"; the
        seat's rules and :func:`~commentary.agents.colour.says_a_number`
        between them stop the figure reaching air.
        """
        start = max(since_ts, now - SPELL_S)
        spell = [
            item for item in self.occurrences if item.ts > start and item.source == "picture"
        ]
        found: list[tuple[int, str]] = []
        here_kinds = {item.kind for item in spell} - {"touch", "flank"}
        for kind in (kind for kind in KINDS if kind in here_kinds):
            here = [item for item in spell if item.kind == kind]
            if len(here) < PATTERN_FLOOR:
                continue
            who = _shared_name(here)
            noun = noun_for(kind, plural=True)
            found.append(
                (
                    len(here),
                    f"{len(here)} {noun} on {who}" if who else f"{len(here)} {noun} in this spell",
                )
            )
        looks: dict[str, int] = {}
        flanks: dict[tuple[str, str], int] = {}
        for item in spell:
            if item.kind == "touch":
                for name in item.players:
                    looks[name] = looks.get(name, 0) + 1
            elif item.kind == "flank" and item.team and item.flank:
                flanks[(item.team, item.flank)] = flanks.get((item.team, item.flank), 0) + 1
        for name, count in sorted(looks.items(), key=lambda pair: (-pair[1], pair[0])):
            if count >= PATTERN_FLOOR:
                found.append((count, f"{name} in the picture on {count} separate looks"))
        for (team, flank), count in sorted(flanks.items()):
            if count >= PATTERN_FLOOR:
                found.append((count, f"{team} down the {flank} {count} times in this spell"))
        return [text for _count, text in sorted(found, key=lambda item: -item[0])]

    # -- the arithmetic ---------------------------------------------------

    def _team_facts(self, team: str, now: float) -> list[Fact]:
        out: list[Fact] = []
        for kind in SIDE_KINDS:
            latest = self.last(team, kind)
            if latest is None:
                continue
            if now - latest.ts > COUNT_FRESH_S:
                # Two corners, two minutes ago. That is not "their second",
                # it is arithmetic, and nobody says it.
                continue
            out.append(self._side_fact(team, kind, self.count(team, kind), latest))
        drought = self.since(team, "shot", now)
        if drought is not None and drought >= DROUGHT_S:
            out.append(
                Fact(
                    about=team,
                    kind="shot_drought",
                    count=_whole_minutes(drought),
                    text=f"{team} have not had a shot in {_minutes(drought)}.",
                    clause=f"{team} have not had a shot in a long while",
                    since_s=drought,
                    side=self._side_of_team(team),
                    ts=now,
                )
            )
        # A run of corners is only worth its own clause when the other side
        # has had one too. Where every corner in the match has been this
        # side's, "second corner in a row" and "second corner" are the same
        # sentence twice, and on the first measured walk they took both of
        # the two places the context block has.
        run = self._run_of(team, "corner")
        latest = self.last(team, "corner")
        if (
            run >= PATTERN_FLOOR
            and run < self.count(team, "corner")
            and latest is not None
            and now - latest.ts <= COUNT_FRESH_S
        ):
            out.append(
                Fact(
                    about=team,
                    kind="corner_run",
                    count=run,
                    text=f"{ordinal(run).capitalize()} corner in a row for {team}.",
                    clause=f"{team} back for another corner",
                    side=self._side_of_team(team),
                    ts=latest.ts if latest else now,
                )
            )
        return out

    def _side_fact(self, team: str, kind: str, total: int, latest: Occurrence) -> Fact:
        noun = noun_for(kind)
        first = total == 1
        return Fact(
            about=team,
            kind=kind,
            count=total,
            text=(
                f"{team}'s first {noun}."
                if first
                else f"{ordinal(total).capitalize()} {noun} for {team}."
            ),
            clause="" if first else f"another {noun} for {team}",
            since_s=self._gap_before(team, kind),
            side=self._side_of_team(team),
            ts=latest.ts,
        )

    def _player_facts(self, name: str, now: float) -> list[Fact]:
        out: list[Fact] = []
        for kind in PLAYER_KINDS:
            latest = self.last(name, kind)
            if latest is None or now - latest.ts > COUNT_FRESH_S:
                continue
            total = self.count(name, kind)
            if total < 1:
                continue
            out.append(
                Fact(
                    about=name,
                    kind=kind,
                    count=total,
                    text=f"{self.who(name)}'s {ordinal(total)} {noun_for(kind)}.",
                    clause=""
                    if total == 1
                    else f"another {noun_for(kind)} from {self.who(name)}",
                    since_s=self._gap_before(name, kind),
                    side=latest.side if latest is not None else Side.UNKNOWN,
                    player=True,
                    ts=latest.ts if latest is not None else now,
                )
            )
        return out

    def _gap_before(self, about: str, kind: str) -> float | None:
        """How long this subject went without one, before the latest."""
        found = self.matching(about, kind)
        if len(found) < 2:
            return None
        return max(0.0, found[-1].ts - found[-2].ts)

    def _run_of(self, team: str, kind: str) -> int:
        """How many of these in a row are this side's, counting back."""
        run = 0
        for item in reversed([o for o in self.occurrences if o.kind == kind and o.team]):
            if not _same(item.team, team):
                break
            run += 1
        return run

    def _add(self, occurrence: Occurrence) -> None:
        """Record one occurrence, unless something else already reported it.

        Two reports of one event from two places — the board's graphic, the
        wire's event, the caller's picture, the goal call's name — are one
        event, and the names on them are pooled. Two *pictures* are never
        merged here: a run of consecutive forms about one thing has already
        been folded on the way in, and two shots six seconds apart are a shot
        and a rebound.
        """
        for index in range(len(self.occurrences) - 1, -1, -1):
            item = self.occurrences[index]
            if item.kind != occurrence.kind:
                continue
            if abs(item.ts - occurrence.ts) > SAME_EVENT_S:
                break
            if item.source == occurrence.source == "picture":
                continue
            if item.source == occurrence.source or not _compatible(item.side, occurrence.side):
                continue
            self.occurrences[index] = replace(
                item,
                players=tuple(sorted(set(item.players) | set(occurrence.players))),
                side=item.side if item.side is not Side.UNKNOWN else occurrence.side,
                team=item.team or occurrence.team,
            )
            return
        self.occurrences.append(occurrence)
        self._maybe_on_target(occurrence)

    def _fold_in(self, kind: str, players: frozenset[str]) -> None:
        """Fold a name read late in a run into the occurrence it belongs to.

        The caller files a goal over several looks and names the scorer on
        only some of them — on the Mbappé trace the form carrying "steps up
        and strikes it" read no name and the next one read Mbappé — so a name
        that turns up part-way through a run belongs to the whole of it.
        """
        if not players:
            return
        for index in range(len(self.occurrences) - 1, -1, -1):
            item = self.occurrences[index]
            if item.kind != kind:
                continue
            self.occurrences[index] = replace(
                item, players=tuple(sorted(set(item.players) | players))
            )
            return

    def _maybe_on_target(self, occurrence: Occurrence) -> None:
        """A shot a save or a goal followed was a shot on target.

        Which side a *save* belongs to is a thing the caller's form is not
        reliable about — it is the keeper's side, and the form is usually
        filled in for the side attacking — so the credit goes to the shot: its
        side, its players, its clock. That end of it is the one the picture is
        sure about.
        """
        if occurrence.kind not in ("save", "goal"):
            return
        for item in reversed(self.occurrences):
            if item.kind != "shot":
                continue
            if occurrence.ts - item.ts > ON_TARGET_S:
                return
            done = any(
                other.kind == "shot_on_target" and other.ts == item.ts
                for other in self.occurrences
            )
            if not done:
                self.occurrences.append(replace(item, kind="shot_on_target"))
            return

    # -- names ------------------------------------------------------------

    def _side_of(self, line: CallerLine) -> tuple[Side, str]:
        side = line.side
        team = (line.team or "").strip()
        if side is Side.UNKNOWN and team:
            side = self._side_of_team(team)
        named = self._team_name(side)
        return side, named or team

    def _team_name(self, side: Side) -> str:
        if side is Side.HOME:
            return self.home
        if side is Side.AWAY:
            return self.away
        return ""

    def _side_of_team(self, team: str) -> Side:
        if self.home and _same(team, self.home):
            return Side.HOME
        if self.away and _same(team, self.away):
            return Side.AWAY
        return Side.UNKNOWN

    def who(self, read: str) -> str:
        """The team sheet's spelling of whoever this reading is of.

        The read itself when nobody on either squad answers to it, which is
        what a ledger with no pack does with everything. A name the roster
        does not know is a name the gate will not let through anyway, so
        keeping it costs nothing and losing it would silently drop a count.
        """
        name = (read or "").strip()
        for full in self.roster:
            if is_the_same_name(name, full):
                return full
        return name

    def _canonical_team(self, name: str) -> str:
        for team in self.teams:
            if _same(name, team):
                return team
        return name


def _compatible(one: Side, other: Side) -> bool:
    """Two reports of one event agree about the side, or one of them cannot say."""
    return one is Side.UNKNOWN or other is Side.UNKNOWN or one is other


def _shared_name(items: Sequence[Occurrence]) -> str:
    """The one name on every one of these, or nothing.

    "the third foul on Otamendi" needs Otamendi to have been on all three of
    them. One name across two fouls is an observation; a different name each
    time is a coincidence.
    """
    seen = [set(item.players) for item in items if item.players]
    if not seen:
        return ""
    shared = set.intersection(*seen)
    return sorted(shared)[0] if shared else ""


def _whole_minutes(seconds: float) -> int:
    return max(1, int(round(seconds / 60.0)))


def _minutes(seconds: float) -> str:
    """An interval as a commentator says it: whole minutes, in words."""
    count = _whole_minutes(seconds)
    return "a minute" if count == 1 else f"{_number(count)} minutes"


def _number(value: int) -> str:
    return _NUMBERS[value - 1] if 1 <= value <= len(_NUMBERS) else str(value)


#: The surname is deliberately not computed here. ``rsplit`` on a space
#: turns "Alexis Mac Allister" into "Allister" and "Rodrigo De Paul" into
#: "Paul", which is the reason ``docs/HANDOFF.md`` deviation 1 has the team
#: sheets printing full names in the first place — and a clause the phraser
#: copies verbatim is the last place to be guessing one. So a ledger clause
#: about a player carries the roster spelling, the phraser's own rules
#: shorten it, and the gate's subject check accepts any tail of the name.


def _same(one: str, other: str) -> bool:
    """Either reading of the gate's name rule, because neither side is a roster.

    The two-way match :mod:`commentary.tallies` makes, for the same reason: a
    sighting reads "Mbappé", a team sheet says "Kylian Mbappé", and a count
    has to find the man whichever way round it arrives.
    """
    return is_the_same_name(one, other) or is_the_same_name(other, one)


def _words(line: CallerLine) -> set[str]:
    """Every word the caller wrote about this form, folded.

    The flank is read off the lead's own prose and the ``detail`` field, which
    is where "down the right" and "out on the left" actually live: the form
    itself carries a team and a scene and no geography at all.
    """
    return set(fold(f"{line.line or ''} {line.detail or ''}").split())


#: Every kind the ledger counts, for anything that wants to walk them. The
#: names are the keys of :data:`commentary.gate.COUNT_NOUNS`, which is the one
#: vocabulary: the ledger writes a clause out of it and ``ledger_claim`` reads
#: a clause back through it.
KINDS: tuple[str, ...] = tuple(COUNT_NOUNS)
