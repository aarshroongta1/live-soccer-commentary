"""Running counts, and what the match does to them.

A pack note is researched before kickoff and true at kickoff. Most of them
stay true for ninety minutes — a 2018 final, an all-time record, who takes the
free kicks — and one kind does not. "Five goals in this tournament" is true
until the man scores, and then it is a wrong number said in a confident voice,
which is the one failure this system has never allowed itself.

It is not hypothetical. On ``runs/rephrased/mbappe-goal`` the line at 184.5 s
reads "Five in the tournament now for Mbappé", said after he has scored twice
inside the same three-minute clip, so the true figure is seven. The note was
right; the match had moved.

``docs/research/real-commentary-corpus.md`` section 7 shows what real
commentary does instead. Every thread in the corpus restates its number after
the event and restates it moved: "13 goals from 13 games" before kickoff
becomes "11 CONSECUTIVE GOALS" at the goal and "the 11th consecutive Premier
League game" at full time. The number is never stale, and nobody on air is
left doing the arithmetic.

So the arithmetic is done here, in code, before the clause reaches either the
phraser's context block or the gate's ``note_claim`` rule. Both see the same
adjusted text, which is the point: the line the model writes and the notes the
gate checks it against agree about what the number is.

The note on disk is never rewritten. :class:`Tallies` produces an adjusted
copy; the pack keeps the researched, kickoff-true figure and its source.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from commentary.gate import is_the_same_name
from commentary.schemas import Event, MatchState, Note, TallyKind
from commentary.scoreline import in_words

#: Two goal reports this close together, credited to the same man, are one
#: goal seen twice: the board's graphic and the wire's event, or the goal call
#: and the wire behind it. The same window ``state.py`` uses to decide that a
#: wire goal is the missing half of a board goal rather than a second one.
SAME_GOAL_S = 8.0

_UNITS = (
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
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_ORDINAL_UNITS = (
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
)
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ORDINAL_TENS = (
    "twentieth",
    "thirtieth",
    "fortieth",
    "fiftieth",
    "sixtieth",
    "seventieth",
    "eightieth",
    "ninetieth",
)

_CARDINAL_VALUES: dict[str, int] = {word: index + 1 for index, word in enumerate(_UNITS)}
_CARDINAL_VALUES.update({word: (index + 2) * 10 for index, word in enumerate(_TENS)})
_ORDINAL_VALUES: dict[str, int] = {word: index + 1 for index, word in enumerate(_ORDINAL_UNITS)}
_ORDINAL_VALUES.update({word: (index + 2) * 10 for index, word in enumerate(_ORDINAL_TENS)})

_ALL_WORDS = sorted([*_CARDINAL_VALUES, *_ORDINAL_VALUES], key=len, reverse=True)
_WORD_ALT = "|".join(_ALL_WORDS)

#: Every shape of number a note can carry: figures, figures with an ordinal
#: tail, a word, a hyphenated compound. Longest alternatives first, so that
#: "twenty-first" is one token rather than "twenty" and a leftover.
_NUMBER = re.compile(
    rf"\b(?:\d+(?:st|nd|rd|th)?|(?:{_WORD_ALT})(?:[\s-](?:{_WORD_ALT}))?)\b",
    re.IGNORECASE,
)

#: What each kind counts, as the noun that follows the number in a note. Used
#: to pick *which* number moves in a note that carries two of them.
_COUNTED: dict[str, str] = {
    "goals": r"goals?",
    "assists": r"assists?",
    "games_scoring": r"games?|matches?",
}


def _figure_ordinal(value: int) -> str:
    if 11 <= value % 100 <= 13:
        return f"{value}th"
    return f"{value}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th')}"


def _ordinal_in_words(value: int) -> str:
    """A number as the ordinal a commentator says. "his sixth", not "his 6th"."""
    if 1 <= value <= len(_ORDINAL_UNITS):
        return _ORDINAL_UNITS[value - 1]
    tens, units = divmod(value, 10)
    if 2 <= tens <= 9:
        if units == 0:
            return _ORDINAL_TENS[tens - 2]
        return f"{_TENS[tens - 2]}-{_ORDINAL_UNITS[units - 1]}"
    return _figure_ordinal(value)


def _value_of(token: str) -> int | None:
    """What this number token means, whatever shape it is written in."""
    word = token.lower().strip()
    if word.isdigit():
        return int(word)
    figures = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", word)
    if figures:
        return int(figures.group(1))
    if word in _CARDINAL_VALUES:
        return _CARDINAL_VALUES[word]
    if word in _ORDINAL_VALUES:
        return _ORDINAL_VALUES[word]
    parts = re.split(r"[\s-]", word)
    if len(parts) == 2 and _CARDINAL_VALUES.get(parts[0], 0) % 10 == 0:
        head = _CARDINAL_VALUES.get(parts[0], 0)
        tail = _CARDINAL_VALUES.get(parts[1]) or _ORDINAL_VALUES.get(parts[1])
        if head and tail is not None:
            return head + tail
    return None


def _render(value: int, like: str) -> str:
    """The new number written the way the old one was.

    Figures stay figures and words stay words, because a note that reads
    "five goals in this tournament" has to come back reading "six goals in
    this tournament" and not "6 goals" — the phraser copies what it is shown,
    and a figure in the context block is a figure in somebody's mouth.
    """
    word = like.lower()
    in_figures = bool(re.fullmatch(r"\d+(?:st|nd|rd|th)?", word))
    ordinal = bool(re.fullmatch(r"\d+(?:st|nd|rd|th)", word)) or word in _ORDINAL_VALUES
    if in_figures:
        written = _figure_ordinal(value) if ordinal else str(value)
    else:
        written = _ordinal_in_words(value) if ordinal else in_words(value, zero="no")
    if like[:1].isupper():
        written = written[:1].upper() + written[1:]
    return written


def _moving_number(text: str, kind: TallyKind) -> re.Match[str] | None:
    """Which number in the note the match moves.

    The one in front of the noun this note counts — "five *goals*", "ten in
    consecutive *games*" — so that a note carrying two figures moves the
    right one. Failing that, the first number in the clause, which is what
    every note in the packs on disk has.
    """
    noun = _COUNTED.get(kind)
    if noun:
        anchored = re.compile(
            rf"(?P<n>{_NUMBER.pattern})(?:\s+\w+){{0,2}}\s+(?:{noun})\b",
            re.IGNORECASE,
        )
        found = anchored.search(text)
        if found:
            return _NUMBER.search(text, found.start("n"), found.end("n"))
    return _NUMBER.search(text)


@dataclass(frozen=True)
class Credit:
    """One thing a player did in this match that a running count has to know."""

    player: str
    ts: float


@dataclass
class Tallies:
    """What the players on this pitch have done since kickoff.

    Fed from two places and deduplicated between them, because a goal is
    reported twice on a good day. The statistician's own event carries a name
    (an :class:`~commentary.schemas.Incident` with ``source="wire"``); the
    board only sees a graphic change and never knows who, so on the ordinary
    path the goal *call* is the only thing that names the scorer — which is
    exactly what :class:`~commentary.goalfollow.GoalFollowup` already works
    out in order to write beat 3.
    """

    goals: list[Credit] = field(default_factory=list)
    assists: list[Credit] = field(default_factory=list)

    # -- what it is told --------------------------------------------------

    def credit_goal(self, player: str | None, ts: float) -> bool:
        """Say that this man has scored. Returns whether it was news."""
        return _credit(self.goals, player, ts)

    def credit_assist(self, player: str | None, ts: float) -> bool:
        """Say that this man made one.

        Nothing reports assists yet — no feed is on by default and no picture
        shows one — so this is the hook a wire with pass events would use,
        and until something calls it an assist tally never moves. That is the
        honest behaviour: an unmoved tally is the researched number, which is
        the best anything knows.
        """
        return _credit(self.assists, player, ts)

    def see_state(self, state: MatchState) -> None:
        """Take in every goal the state can put a name to.

        Board-sourced incidents carry ``player=None`` and are ignored here:
        they are already counted by the goal call that named the scorer, and
        a goal nobody could name cannot move any one player's tally.
        """
        for incident in state.incidents:
            if incident.event is Event.GOAL and incident.player:
                self.credit_goal(incident.player, incident.video_ts)

    # -- what it knows ----------------------------------------------------

    def count(self, about: str, kind: TallyKind) -> int:
        """How far this match has moved this man's count of this thing."""
        if kind == "goals":
            return _count(self.goals, about)
        if kind == "assists":
            return _count(self.assists, about)
        # A run of consecutive games scored in moves by one the first time he
        # scores and never again, however many he gets: it counts games, not
        # goals, and this is one game.
        return 1 if _count(self.goals, about) else 0

    # -- what it produces -------------------------------------------------

    def adjust(self, note: Note) -> Note:
        """The note as it is true now, or the note itself if nothing moved.

        A note with no ``counts`` comes back unchanged and untouched, which is
        every note in every pack written before this existed.
        """
        if note.counts is None:
            return note
        moved = self.count(note.about, note.counts)
        if moved <= 0:
            return note
        found = _moving_number(note.text, note.counts)
        if found is None:
            return note
        value = _value_of(found.group())
        if value is None:
            return note
        text = (
            note.text[: found.start()]
            + _render(value + moved, found.group())
            + note.text[found.end() :]
        )
        return note.model_copy(update={"text": text})

    def adjusted(self, notes: Iterable[Note]) -> list[Note]:
        """Every note as it is true now, in the order given."""
        return [self.adjust(note) for note in notes]


def _credit(into: list[Credit], player: str | None, ts: float) -> bool:
    name = (player or "").strip()
    if not name:
        return False
    for credit in into:
        if abs(credit.ts - ts) <= SAME_GOAL_S and _same(name, credit.player):
            return False
    into.append(Credit(player=name, ts=ts))
    return True


def _count(credits: list[Credit], about: str) -> int:
    """How many of these belong to this man.

    Matched with the gate's own name rule, because a goal is credited to
    whatever the line called him — "Mbappé" — and the note is filed under the
    roster spelling, "Kylian Mbappé".
    """
    subject = about.strip()
    if not subject:
        return 0
    return sum(1 for credit in credits if _same(credit.player, subject))


def _same(one: str, other: str) -> bool:
    """Either reading of the gate's rule, because neither side is the roster.

    :func:`~commentary.gate.is_the_same_name` asks whether a short thing read
    off a picture is a long thing off a team sheet. Here both ends can be
    either: the credit is whatever the goal call said, the note is filed under
    whatever the researcher wrote, and "Mbappé" against "Kylian Mbappé" has to
    match whichever way round it arrives.
    """
    return is_the_same_name(one, other) or is_the_same_name(other, one)
