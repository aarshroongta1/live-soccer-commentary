"""A fact picked back up, which is the difference between a thread and a list.

``docs/research/real-commentary-corpus.md`` section 7 is the evidence. Every
match in the corpus runs three to five threads, and each has the same three
slots: a *setup* early with the number in it, a *payoff* within a few seconds
of the event, and one to four *callbacks* spread over the rest of the match,
each restating the same number in a new form. Vardy's record is mentioned
thirteen times over 94 minutes; Cucurella's slips fifteen times over 99; the
snow nine times over 98. The study's own conclusion: **a thread that fires once
is not a thread.**

This system had no thread at all. Every phraser call saw the last four lines
and the notes about whoever was on the form, and nothing else. In the Mbappé
trace "Five in the tournament" is said once at 59.2 s and never again — not
even at 176.7 s, when he scores the goal that makes it seven.

So the pack's notes get a memory. One entry per note, seeded at kickoff, and
when a phrased line uses a note the entry records that it was said and when.
What that memory buys is an ordering: the note offered to the next quiet
moment is, first, one about a player this line names that has been said
*exactly once* and not recently — the callback — and only then an unused one.
That single rule is what turns a list of facts into a thread.

The counting is not here. :mod:`commentary.tallies` owns what the match does to
a running number, and :class:`Threads` holds one so that every clause it hands
out is already true as of now: a callback that restates a stale number is worse
than no callback at all.

Both the runtime and the offline rephrase own one of these, the way they both
own a :class:`~commentary.goalfollow.GoalFollowup`, so a thread that runs in
one runs in the other.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from commentary.gate import fold, is_the_same_name, notes_used
from commentary.ledger import Fact
from commentary.schemas import KnowledgePack, MatchState, Note
from commentary.tallies import Tallies

#: How long a note has to rest before it is worth picking back up. The corpus
#: callbacks are minutes apart — Vardy at 2:12, 24:44, 35:35, 84:44 — and the
#: shortest interval anywhere in section 7 that is not part of one continuous
#: burst is about five minutes. Under this the fact is still in the air and
#: saying it again is repetition, not a callback.
CALLBACK_QUIET_S = 300.0

#: And how long before a note said twice may go out a third time. The corpus
#: does reach four and five mentions of one thread, and always with a long
#: gap: ten minutes is the floor here.
REPEAT_QUIET_S = 600.0

#: How many clauses the context block may carry. Three is what the phraser's
#: block already held and it is the cap the study's rate implies: one number
#: reaches air about every forty seconds of build-up, and a block of six is a
#: menu, not a prompt.
CONTEXT_NOTES = 3


@dataclass
class Thread:
    """One pack note, and what this match has done with it so far."""

    note: Note
    subject: str
    #: True when the note is filed under a team rather than a player. Team
    #: notes lose to player notes in the ordering: the corpus's threads are
    #: overwhelmingly about a man, and a fact about Argentina is what you say
    #: when there is nothing to say about anybody on the ball.
    team: bool = False
    times_said: int = 0
    last_said_ts: float | None = None
    #: The clause as it was the last time it went out, figure and all. What
    #: :func:`_tier` compares the adjusted clause against: "five goals in
    #: this tournament" said at 59 s and "seven goals in this tournament"
    #: offered at 176 s are not the same fact, and resting the second one
    #: because the first was said is how the second goal lost its third beat.
    last_said_text: str = ""

    def quiet_for(self, ts: float) -> float:
        """How long since this was last said. Forever, if it never has been."""
        if self.last_said_ts is None:
            return float("inf")
        return max(0.0, ts - self.last_said_ts)

    def said(self, ts: float, text: str = "") -> None:
        self.times_said += 1
        self.last_said_ts = ts
        if text.strip():
            self.last_said_text = text.strip()


@dataclass(frozen=True)
class Offered:
    """One clause put in front of the phraser, and what it is.

    ``note`` is already adjusted for anything the match has changed, so the
    number in it is the number that is true now and the number the gate will
    check the line against.
    """

    note: Note
    subject: str
    callback: bool
    times_said: int
    #: Which thread this is: the note's place in the pack, and the only handle
    #: on it that survives the number in its text changing. Counting "how many
    #: notes were used more than once" off the text would call "five goals in
    #: this tournament" and "six goals in this tournament" two different
    #: threads, when they are one thread paid off.
    index: int = -1


@dataclass
class Threads:
    """Every pack note, with a memory and a running count behind it."""

    entries: list[Thread] = field(default_factory=list)
    tallies: Tallies = field(default_factory=Tallies)

    @classmethod
    def from_pack(cls, pack: KnowledgePack | None) -> Threads:
        """Seed one entry per note at kickoff, said nothing, never.

        A pack with no notes — every pack written before notes existed —
        gives an empty list and everything downstream behaves as it did.
        """
        if pack is None or not pack.notes:
            return cls()
        teams = {pack.home.name, pack.away.name}
        return cls(
            entries=[
                Thread(note=note, subject=note.about, team=note.about in teams)
                for note in pack.notes
            ]
        )

    # -- what the match does to the numbers -------------------------------

    def credit_goal(self, player: str | None, ts: float) -> bool:
        """This man has scored, so every tally about him moves."""
        return self.tallies.credit_goal(player, ts)

    def see_state(self, state: MatchState) -> None:
        """Take in any goal the state can put a name to. See :class:`Tallies`."""
        self.tallies.see_state(state)

    def notes(self) -> list[Note]:
        """Every note as it is true now. What the gate checks a line against."""
        return self.tallies.adjusted(entry.note for entry in self.entries)

    # -- what goes in front of the phraser --------------------------------

    def offer(
        self,
        names: Iterable[str],
        *,
        ts: float,
        limit: int = CONTEXT_NOTES,
        payoff: bool = False,
    ) -> list[Offered]:
        """The clauses worth putting in front of this line, best first.

        ``names`` arrives in the order that matters and the order is kept
        inside each tier: the man on the ball, then whoever else was read,
        then the two teams.

        ``payoff`` is the goal. It does two things, and both are section 7's
        middle slot: it moves the scorer's running count ahead of his history,
        because the corpus puts a number about the scorer inside fifteen
        seconds of every goal; and it lets a tally through the resting rule,
        because a tally said a minute ago is *not* the same clause any more.
        He has scored since. "Five in the tournament" became "six", and the
        payoff of a thread is exactly the moment the number moves.

        The tiers, and they are the whole of section 7's finding:

        0. a note about a player named here, said once, and rested — the
           callback, and the reason this class exists;
        1. a note about a player named here that has never been said;
        2. a team note never said;
        3. a team note said once and rested;
        4. anything said twice or more, once ten minutes have passed.

        A note said once but recently is not offered at all: repeating a fact
        thirty seconds after it went out is what the last four lines in the
        prompt are already there to stop.
        """
        ranked: list[tuple[int, int, int, int, Thread]] = []
        taken: set[int] = set()
        for order, name in enumerate(names):
            cleaned = (name or "").strip()
            if not cleaned:
                continue
            for index, entry in enumerate(self.entries):
                if index in taken or not is_the_same_name(cleaned, entry.subject):
                    continue
                tier = _tier(entry, ts, payoff=payoff, moved=self._moved(entry))
                if tier is None:
                    continue
                taken.add(index)
                first = 0 if (payoff and entry.note.counts is not None) else 1
                ranked.append((tier, first, order, index, entry))
        ranked.sort(key=lambda row: row[:4])
        return [
            Offered(
                note=self.tallies.adjust(entry.note),
                subject=entry.subject,
                callback=entry.times_said > 0,
                times_said=entry.times_said,
                index=index,
            )
            for _, _, _, index, entry in ranked[:limit]
        ]

    def _moved(self, entry: Thread) -> bool:
        """Has the count in this clause changed since the clause last went out?

        Only a counting note can move, and only one that has been said has
        anything to move from. The comparison is on the adjusted text — the
        clause the phraser was actually shown — because that is the sentence
        a listener heard, figure and all.
        """
        if entry.note.counts is None or not entry.last_said_text:
            return False
        return self.tallies.adjust(entry.note).text.strip() != entry.last_said_text

    # -- what comes back --------------------------------------------------

    def said(
        self, text: str, *, ts: float, pack: KnowledgePack | None = None
    ) -> list[Offered]:
        """Record every note this line used. Returns them as they stood.

        Matched against the *adjusted* notes, because the adjusted clause is
        what the phraser was shown and what it reworded. The returned entries
        carry ``times_said`` as it was before this line, so that a trace row
        can say whether what just went out was a setup or a callback.
        """
        if not text.strip() or not self.entries:
            return []
        adjusted = self.notes()
        out: list[Offered] = []
        for index in notes_used(text, adjusted, pack):
            entry = self.entries[index]
            out.append(
                Offered(
                    note=adjusted[index],
                    subject=entry.subject,
                    callback=entry.times_said > 0,
                    times_said=entry.times_said,
                    index=index,
                )
            )
            entry.said(ts, adjusted[index].text)
        return out

    # -- what a trace and a summary want ----------------------------------

    def used(self) -> list[Thread]:
        """Every note that has reached air at least once."""
        return [entry for entry in self.entries if entry.times_said]

    def recurring(self) -> list[Thread]:
        """Every note that has reached air more than once: the actual threads."""
        return [entry for entry in self.entries if entry.times_said > 1]


def _tier(
    entry: Thread, ts: float, *, payoff: bool = False, moved: bool = False
) -> int | None:
    """Which band this note is in, or ``None`` if it is not on offer at all.

    ``moved`` is a counting clause whose figure has changed since it last
    went out, and it is the top of the ordering rather than an exemption
    inside it. The payoff rule already let such a clause *through* the
    resting check, and on the Mbappé trace that was not enough: at the second
    goal the tally had been said twice, which put it in tier 4, and three
    notes nobody had said yet took the three places in front of it. The
    number moving is the news; an unsaid fact about the same man is not.
    """
    quiet = entry.quiet_for(ts)
    if moved:
        return 3 if entry.team else 0
    if entry.times_said == 0:
        return 2 if entry.team else 1
    # The payoff exemption: a count that has moved is new information however
    # recently the old figure went out.
    rested = quiet >= CALLBACK_QUIET_S or (payoff and entry.note.counts is not None)
    if entry.times_said == 1:
        if not rested:
            return None
        return 3 if entry.team else 0
    if quiet >= REPEAT_QUIET_S or (payoff and entry.note.counts is not None):
        return 4
    return None


@dataclass
class SaidCounts:
    """Which counts off this match have already gone out, at which figure.

    The ledger is a different kind of clause from a note and it had no memory
    at all: on ``runs/rephrased/r5a/offside`` "Argentina\'s first corner of
    the match" went out at 10.3 s and again at 14.3 s, four seconds apart,
    because both lines were offered the same fact and neither knew the other
    had said it.

    So a count that has been said is not offered again until it moves —
    which is exactly the rule :class:`Threads` applies to a note that counts,
    and for the same reason: the second corner is news and the first one is
    not, twice.

    Kept here rather than in :mod:`commentary.ledger` because it is a memory
    of what was *said*, which is this module\'s subject; the ledger counts
    what happened.
    """

    said: dict[tuple[str, str], int] = field(default_factory=dict)

    def note(self, facts: Iterable[Fact]) -> None:
        """Record every count an aired line carried."""
        for fact in facts:
            self.said[self._key(fact)] = fact.count

    def fresh(self, facts: Iterable[Fact]) -> list[Fact]:
        """The counts worth offering: the ones that have moved, or never gone out."""
        return [fact for fact in facts if self.said.get(self._key(fact)) != fact.count]

    @staticmethod
    def _key(fact: Fact) -> tuple[str, str]:
        return (fold(fact.about), fact.kind)


def context_notes(offered: Sequence[Offered]) -> tuple[list[Note], list[bool]]:
    """The two parallel lists the phraser prompt takes: clauses, and which
    of them have been said before in this match."""
    return [item.note for item in offered], [item.callback for item in offered]
