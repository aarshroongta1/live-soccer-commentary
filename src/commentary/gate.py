"""The fact gate: the last thing between a model's sentence and a microphone.

The claim this project makes is that a vision model can call a match without
inventing things, and this module is where that claim is cashed. It is
deterministic and it makes no model call, because a checker that hallucinates
is not a checker. Every rejection carries a short tag plus the detail, so the
eval can report rejection rate broken down by reason and a human can grep a
90-minute trace for the one line that got through.

Four rules, in the order a sceptic would apply them.

* A replay is called as a replay, never as live. Narrating a replay as
  though it were happening is the most embarrassing failure available to this
  system — and refusing every replay line outright, which is what this rule
  used to do, is how the run went thirty-six seconds without a word after the
  Mbappé penalty while the caller wrote four accurate replay lines nobody
  said. So the scene is not the refusal any more: a replay line passes when
  it is of something the match has actually had, carries no score, and is not
  written in the present tense of a goal going in.
* Names must be on a roster in the knowledge pack. A name the caller claims to
  have read off a graphic gets *less* latitude, not more: if it is not on a
  roster then the caller did not read a graphic, it imagined one, and the rest
  of that line is suspect with it.
* A number put on the score must be a number the state holds. The scoreline
  said out loud is the obvious way of putting one there and the ordinal is the
  quiet one: "Argentina's third" claims a score exactly as hard as "3-0" does,
  and only the state is entitled to settle either. The one latitude is a goal
  being called while the graphic catches up, which is the whole reason the
  caller is allowed to speak ahead of the board at all.
* A goal is only a goal once the board says so. Nothing else gets to claim
  one. "The board says so" is the
  caller's own runtime's judgement and it is broader than a settled change:
  a board part-way through agreeing counts, and so does a goal the state has
  already taken in, which is what every line *about* a goal comes after.

Where a line can be saved it is saved. Dropping a name the gate cannot verify
and leaving "the cross comes in and the winger cuts inside" is better
commentary than silence, so an unverifiable name in the body of a line is
trimmed rather than fatal — unless trimming leaves too little to be worth
saying.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from typing import Protocol

from commentary.config import SETTINGS, GateConfig
from commentary.schemas import (
    CallerLine,
    Event,
    GateVerdict,
    KnowledgePack,
    MatchState,
    Note,
    Scene,
    Side,
)

# Words that open sentences or describe football, not people. A capitalised
# token in here is never treated as a name, which is what stops the gate
# trimming "Brilliant from the far post" down to "from the far post".
#
# The kit colours are in here for a sharper reason: the caller is told to say
# "the near-post runner in blue" when it cannot read a number, so a line
# opening "Blue shirts crowd it out" would otherwise have the one word that
# identified the team cut out of it as an unverifiable name.
_STOPWORD_TEXT = """
    a an and as at back but by for from in into of off on onto or out over the then to
    towards under up with within without
    i he she it they we you him her them his hers its their our your me my mine
    that this these those there here what when where which who whom whose why how
    all any both each every few more most no none not now nor only other same some such
    still too very yes oh ah well just again almost nearly already
    is am are was were be been being do does did done has have had having
    can could may might must shall should will would let
    goal goals ball cross corner corners penalty penalties header headers keeper goalkeeper
    referee ref offside card cards foul fouls throw box area pitch half time
    red blue white black green yellow orange purple claret navy maroon gold grey amber
    stripes stripe shirt shirts sleeves kit
    full whistle replay touch pass passes shot shots save saves free kick kicks run runs
    side left right centre center middle midfield defence defense attack striker winger
    captain sub subs substitution bench minute minutes second seconds injury stoppage var
    play played plays playing goes going get gets got take takes taken drive drives driven
    win wins won lose loses lost find finds found send sends sent turn turns turned
    break breaks broken clip clips whip whips cut cuts chip chips slot slots
    look looks looking wait waits try tries come comes coming go gone
    great good brilliant lovely superb terrific poor quick sharp clever strong big small
    late early first third final last long short high low deep wide narrow tight
    another one two three four five six seven eight nine ten nil nought zero level
    away home down forward forwards square above below behind beyond against between
    inside outside around along across straight
    after before while since until than so if because though although unless even never
    always surely much nice easy lucky unlucky dangerous clear close magnificent outstanding
    fine smart neat tidy calm cool sweet huge massive well ooh wow yeah okay
"""
#: Public because the grader needs the same list. A grader with a shorter one
#: marks as an invented name every ordinary word the gate deliberately let
#: through — the real runs produced "Play breaks down by the touchline" and
#: had "Play" counted as a person.
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

#: The trim used to drop an ordinary word from the front of a line, on the
#: theory that an invented surname would most often be put first. Nineteen
#: firings across nine runs on six clips — Tears, Hands, Arms, Fist, Ice,
#: Thousands, Whole, Pure, Emotion, Sky, Restart, Round, Grimacing — and not
#: one of them was a name the caller invented. A rule that has never once
#: caught what it is for, and has cost about two true lines a run, is not a
#: rule. Removed; a capitalised first word is just a word.
#:
#: Prepositions left dangling by a trim ("comes in from  and the winger"), so
#: they go with the name rather than staying behind as debris.
_DANGLERS = "from|by|to|for|off|with|of|onto|into|at|on|through|past"

_WORD = re.compile(r"[^\W\d_]+(?:['’‐-][^\W\d_]+)*", re.UNICODE)
_NUMBER_WORDS: dict[str, int] = {
    "nil": 0,
    "nought": 0,
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_WORD_ALT = "|".join(_NUMBER_WORDS)
_DIGIT_PAIR = re.compile(r"\b([0-9])\s*[-–—:]\s*([0-9])\b")
_WORD_PAIR = re.compile(rf"\b({_WORD_ALT})[\s-]+({_WORD_ALT})\b", re.IGNORECASE)
_ALL_PAIR = re.compile(rf"\b({_WORD_ALT})[\s-]+all\b", re.IGNORECASE)
#: A measurement is not a scoreline. "Eight to ten yards out" has the shape of
#: one and the meaning of a distance, and the gate is the last thing that
#: should be rejecting a line for saying where the ball was.
_NOT_A_UNIT = r"(?!\s+(?:yards?|metres?|meters?|feet|foot|minutes?|seconds?|men|man|players?))"
#: "3 nil". The word pair above wants both halves spelled out and the pair
#: above that wants both in figures; a commentator mixes them freely.
_MIXED_PAIR = re.compile(rf"\b([0-9])[\s-]+({_WORD_ALT})\b{_NOT_A_UNIT}", re.IGNORECASE)
#: "two to one", "2 to 1". Not "one to one", which is a duel with a keeper: a
#: level score is said "one all", and _ALL_PAIR already has that one.
_TO_PAIR = re.compile(
    rf"\b([0-9]|{_WORD_ALT})\s+to\s+([0-9]|{_WORD_ALT})\b{_NOT_A_UNIT}", re.IGNORECASE
)

#: How a line says a goal was scored. The word "goal" is not on the list.
#: It used to be, with a second expression stripping the innocent uses first
#: — "towards the goal", "goal kick", "goalkeeper" — and that list can never
#: be finished: the first real run lost "France scrambling back towards their
#: own goal" to it, and "his goal" and "the French goal" were next. The form
#: already carries an unambiguous answer in ``event``, so the prose only has
#: to catch the phrasings that mean a goal and nothing else.
_GOAL_CLAIMS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bscores\b(?!\s+(?:are|level|tied))",
        r"\bscored\b",
        r"\bit'?s\s+in\b",
        r"\b(?:in|into)\s+the\s+(?:back\s+of\s+the\s+)?net\b",
        r"\bfinds?\s+the\s+net\b",
        r"\bmakes?\s+it\s+\d\b",
        r"\bequalis\w*\b",
        r"\bequaliz\w*\b",
    )
)


def fold(text: str) -> str:
    """Accents off, case off, punctuation to spaces.

    The caller reads names off a broadcast graphic and the roster comes off a
    web page; the two agree on the person and disagree on the diacritics far
    more often than is comfortable, so neither is compared in its own spelling.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    # The possessive goes as a unit, before the apostrophe is dropped. "De
    # Gea's right" folded to "de geas" and matched nobody, so the gate trimmed
    # the goalkeeper's name out of a penalty being scored past him — twice in
    # one clip. Whose right it was is the possessive; who it was is the name.
    plain = re.sub(r"['’]s\b", "", plain)
    plain = plain.replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^0-9A-Za-z]+", " ", plain).lower().split())


def is_the_same_name(said: str, full_name: str) -> bool:
    """Is what was read off the picture this player's name?

    Folded, and generous about which part was read, because a shirt shows a
    surname, a graphic shows whatever it likes, and neither agrees with a
    team sheet about spacing or diacritics. Four forms count, and the last
    two are here because five lines on the real clips were killed by them:

    - the whole name, or the surname alone;
    - a suffix of the whole name on a word boundary, so "Di María" matches
      even though ``rsplit`` calls the surname "María";
    - **the initial form.** A graphic writes "T. Hernández", and matched
      against "Theo Hernández" that is neither the surname nor a suffix. The
      initial has to be the initial: "L. Martínez" is not Theo.
    - **the same name spaced differently.** The shirt reads MAC ALLISTER and
      the team sheet says "MacAllister"; three lines went for that one.
    """
    wanted, full = fold(said), fold(full_name)
    if not wanted:
        return False
    if _is_tail_of(wanted, full):
        return True
    head, _, rest = wanted.partition(" ")
    return len(head) == 1 and bool(rest) and full.startswith(head) and _is_tail_of(rest, full)


def _is_tail_of(said: str, full: str) -> bool:
    """Is this the whole name or any run of words ending it, spacing aside?

    Spacing aside because a shirt reads MAC ALLISTER and a team sheet says
    "MacAllister"; word by word rather than character by character, so
    "ister" is not a match for anybody.
    """
    words = full.split()
    tails = {" ".join(words[i:]) for i in range(len(words))}
    return said in tails or said.replace(" ", "") in {t.replace(" ", "") for t in tails}


def _is_that_player(pack: KnowledgePack, number: int, name: str) -> bool:
    """Do the number and the name on one sighting describe the same person?"""
    return any(
        is_the_same_name(name, player.name)
        for sheet in (pack.home, pack.away)
        for player in sheet.squad
        if player.number == number
    )


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _surname(name: str) -> str:
    return name.rsplit(" ", 1)[-1]


@dataclass(frozen=True)
class _Roster:
    """Every string the gate is willing to let a line say out loud."""

    people: frozenset[str]
    teams: frozenset[str]
    numbers: frozenset[str]

    @property
    def known(self) -> frozenset[str]:
        return self.people | self.teams


def _roster_of(state: MatchState, pack: KnowledgePack | None) -> _Roster:
    """Fold the knowledge pack and the match state into one bag of allowed names.

    The state contributes the two team names and whatever the graphics have
    taught it about shirt numbers; the pack contributes the squads, the
    managers, the competition and the ground.
    """
    people: set[str] = set()
    teams: set[str] = set()
    numbers: set[str] = set()

    for team_name in (state.home, state.away):
        folded = fold(team_name)
        teams.add(folded)
        teams.update(token for token in folded.split() if len(token) >= 3)
    for number, name in state.on_pitch.items():
        numbers.add(fold(number))
        people.add(fold(name))

    if pack is not None:
        for sheet in (pack.home, pack.away):
            # The demonym is a team word, not a name. "The French lines" was
            # trimmed to "the lines" on the first real run, because "French"
            # is capitalised, is on no roster, and is exactly the word a
            # commentator reaches for when they cannot name anybody.
            for label in (sheet.name, sheet.short, sheet.demonym):
                if label:
                    teams.add(fold(label))
                    teams.update(token for token in fold(label).split() if len(token) >= 3)
            if sheet.demonym:
                # "the Dutchman steps up" cost two lines on the shootout clip:
                # the demonym was a team word and the word a commentator
                # actually says was not. One man, two men, same nationality.
                folded = fold(sheet.demonym)
                teams.update({f"{folded}man", f"{folded}men", f"{folded}woman", f"{folded}s"})
            if sheet.manager:
                people.add(fold(sheet.manager))
                people.add(fold(_surname(sheet.manager)))
            for player in sheet.squad:
                people.add(fold(player.name))
                people.add(fold(player.surname))
                people.update(token for token in fold(player.name).split() if len(token) >= 3)
                if player.number is not None:
                    numbers.add(str(player.number))
        # The vocabulary that is not people. A commentator says where they
        # are, what they are playing for, and where a player is from, and
        # none of those is a name to be checked against a squad. The fifth
        # run lost "a final goal for the man from Rosario" to exactly this:
        # "Rosario" is in the pack's own storylines and was on no roster, and
        # "World Cup" failed because "Cup" is three letters and the old
        # threshold was four.
        for label in (pack.competition, pack.venue):
            teams.update(token for token in fold(label).split() if len(token) >= 3)
        for story in (*pack.storylines, *pack.key_matchups):
            teams.update(
                fold(word) for word in re.findall(r"\b[A-Z][\w'’-]+", story) if len(word) >= 3
            )

    people.discard("")
    teams.discard("")
    return _Roster(frozenset(people), frozenset(teams), frozenset(numbers))


@dataclass(frozen=True)
class _Candidate:
    """A capitalised run in the line that looks like it names somebody."""

    text: str
    start: int
    end: int


def _opens_a_sentence(line: str, start: int) -> bool:
    """Is the word at ``start`` the first word of the line or of a sentence in it?"""
    before = line[:start].rstrip()
    return not before or before[-1] in ".!?"


def _candidates(line: str) -> list[_Candidate]:
    """Consecutive capitalised words, minus the ones that are just English.

    Runs rather than single words, so "Jude Bellingham" is checked against the
    roster as one person instead of as two unknown halves.

    Nothing special happens at position 0 any more. The rule that stripped an
    ordinary word off the front of a line never caught an invented name in
    nine runs and cost nineteen true words; the roster check on the run
    itself is what stops a name nobody is called, wherever it sits.
    """
    runs: list[_Candidate] = []
    current: list[re.Match[str]] = []

    def flush() -> None:
        # A run that starts a sentence is not a name claim at all. The rule
        # that stripped an ordinary word off the front fired nineteen times
        # in nine runs and never once caught an invented name, so the trim
        # there is gone rather than made cleverer — and with it goes any
        # checking of that run, which is the same decision stated the other
        # way round. A second sentence inside the line starts the same way:
        # once the caller was allowed "Save. The deflection flies wide", the
        # word after the full stop was read as a name, and "Everything in
        # this final waits on him" went to air as "in this final waits on him".
        if current and _opens_a_sentence(line, current[0].start()):
            current.clear()
            return
        while current and fold(current[0].group()) in STOPWORDS:
            current.pop(0)
        while current and fold(current[-1].group()) in STOPWORDS:
            current.pop()
        if current:
            runs.append(
                _Candidate(
                    line[current[0].start() : current[-1].end()],
                    current[0].start(),
                    current[-1].end(),
                )
            )
        current.clear()

    previous_end = -1
    for match in _WORD.finditer(line):
        word = match.group()
        adjacent = previous_end >= 0 and match.start() - previous_end <= 1
        if len(word) > 1 and word[0].isupper():
            if not adjacent:
                flush()
            current.append(match)
        else:
            flush()
        previous_end = match.end()
    flush()
    return [c for c in runs if fold(c.text) not in STOPWORDS]


def _matches_roster(name: str, roster: _Roster, threshold: float) -> bool:
    """Exact after folding, or close enough that it is the same person misspelt.

    The suffix rule is the one that earns its place. ``Player.surname`` is a
    split on the last space, so Di María's surname is "María", De Paul's is
    "Paul" and Mac Allister's is "Allister"; the roster's token list drops
    anything under three letters, so "di" and "de" are never known; and the
    similarity of "di maria" to "angel di maria" is 0.73, well under the
    threshold. Every compound surname therefore failed — five lines of one
    Sonnet run died on exactly this. A candidate that ends a known full name
    on a word boundary is that person, and nothing else is.
    """
    folded = fold(name)
    if not folded:
        return True
    known = roster.known
    if folded in known:
        return True
    parts = folded.split()
    if len(parts) > 1 and all(part in known for part in parts):
        return True
    suffix = f" {folded}"
    if any(entry.endswith(suffix) for entry in known):
        return True
    return any(_similar(folded, entry) >= threshold for entry in known)


@dataclass(frozen=True)
class ScoreSpan:
    """A scoreline a line says out loud, and where in the line it sits."""

    start: int
    end: int
    home: int
    away: int

    @property
    def pair(self) -> tuple[int, int]:
        return self.home, self.away


def score_spans(line: str) -> list[ScoreSpan]:
    """Every scoreline the line says out loud, in digits or in words, located.

    Split out of :func:`_stated_scores` so that a scoreline can be *removed*
    as well as judged. ``commentary.scoreline`` strips whatever number the
    phraser wrote before code appends the one the state supports, and a strip
    with its own regexes would drift from these: a shape the strip missed and
    the gate caught is a line dropped whole, which is the fault this whole
    change exists to remove.
    """
    found: list[ScoreSpan] = []
    for match in _DIGIT_PAIR.finditer(line):
        found.append(
            ScoreSpan(match.start(), match.end(), int(match.group(1)), int(match.group(2)))
        )
    for match in _WORD_PAIR.finditer(line):
        first, second = match.group(1).lower(), match.group(2).lower()
        if (first, second) == ("one", "two"):
            continue  # "a lovely one-two" is a give-and-go, not a scoreline.
        found.append(
            ScoreSpan(match.start(), match.end(), _NUMBER_WORDS[first], _NUMBER_WORDS[second])
        )
    for match in _ALL_PAIR.finditer(line):
        value = _NUMBER_WORDS[match.group(1).lower()]
        found.append(ScoreSpan(match.start(), match.end(), value, value))
    for match in _MIXED_PAIR.finditer(line):
        found.append(
            ScoreSpan(
                match.start(),
                match.end(),
                int(match.group(1)),
                _NUMBER_WORDS[match.group(2).lower()],
            )
        )
    for match in _TO_PAIR.finditer(line):
        first, second = match.group(1).lower(), match.group(2).lower()
        if first == second == "one":
            continue  # "one to one with the keeper" is a duel, not a draw.
        found.append(
            ScoreSpan(match.start(), match.end(), _as_number(first), _as_number(second))
        )
    found.sort(key=lambda span: (span.start, span.end))
    return found


def ordinal_score_spans(text: str) -> list[tuple[int, int]]:
    """Where the line counts a *side's* goals: "Argentina's third", "their fourth".

    The scorer's own tally — "his third" — is deliberately not here, for the
    reason :data:`_ORD_POSSESSIVE` gives: it counts one man's goals, which the
    scoreboard does not know and a pack note might. That claim belongs to the
    note rule, and it is the third beat of a goal in the corpus, so a strip
    that removed it would remove the beat this change is adding.
    """
    found: list[tuple[int, int]] = []
    for pattern in (_ORD_POSSESSIVE, _ORD_THEIR, _ORD_FOR_TEAM):
        for match in pattern.finditer(text):
            if _is_aspiration(text, match.start()):
                continue
            found.append((match.start(), match.end()))
    return sorted(found)


def level_claim_spans(text: str) -> list[tuple[int, int]]:
    """Where the line says the scores are equal without saying a number."""
    found: list[tuple[int, int]] = []
    for pattern in _LEVEL_CLAIMS:
        found += [(match.start(), match.end()) for match in pattern.finditer(text)]
    return sorted(found)


def _stated_scores(line: str) -> list[tuple[int, int]]:
    """Every scoreline the line says out loud, in digits or in words."""
    return [span.pair for span in score_spans(line)]


def _as_number(token: str) -> int:
    return int(token) if token.isdigit() else _NUMBER_WORDS[token]


#: Ordinals a commentator actually says, spelled out or in figures.
_ORDINAL_WORDS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "1st": 1,
    "2nd": 2,
    "3rd": 3,
    "4th": 4,
    "5th": 5,
    "6th": 6,
    "7th": 7,
    "8th": 8,
    "9th": 9,
    "10th": 10,
}
_ORD_ALT = "|".join(_ORDINAL_WORDS)

#: An ordinal counts goals only where a noun would go: ending the sentence, in
#: front of "goal", or in front of "of the night". Everywhere else it is
#: describing something that is not the score — "their first real chance", "the
#: third man on the overlap" — and a rule that counted those would spend the
#: first half rejecting true lines.
_ORD_TAIL = r"(?=\s*(?:[.,!?;:]|[-–—]|$)|\s+goals?\b|\s+of\b)"

#: One or two ordinary words, checked against the team names afterwards rather
#: than by their capitals. "Argentina", "Northvale United", "the French".
_TEAM_RUN = r"[^\W\d_][\w-]*(?:\s+[^\W\d_][\w-]*)?"

#: A number somebody is chasing is not a number they hold. "Looking for their
#: third" at two-nil is both true and unsayable under plain arithmetic, so a
#: wish is read as a wish and left alone.
_ASPIRING = frozenset(
    (
        "for",
        "chasing",
        "chase",
        "seeking",
        "search",
        "searching",
        "hunting",
        "hunt",
        "after",
        "need",
        "needs",
        "want",
        "wants",
    )
)

#: "Argentina's third", "their fourth", "a third for Argentina". A hat-trick is
#: deliberately absent: it counts one man's goals, not the team's, and so is
#: "his third" — neither says anything the board can be held to.
_ORD_POSSESSIVE = re.compile(
    rf"\b(?P<team>{_TEAM_RUN})['’]s\s+(?P<ord>{_ORD_ALT}){_ORD_TAIL}", re.IGNORECASE
)
_ORD_THEIR = re.compile(rf"\btheir\s+(?P<ord>{_ORD_ALT}){_ORD_TAIL}", re.IGNORECASE)
_ORD_FOR_TEAM = re.compile(
    rf"\b(?:a|an|the)\s+(?P<ord>{_ORD_ALT})\s+(?:goals?\s+)?for\s+(?P<team>{_TEAM_RUN})\b",
    re.IGNORECASE,
)


def _team_words(state: MatchState, pack: KnowledgePack | None) -> dict[Side, frozenset[str]]:
    """The words that name each side, with the words that name both removed.

    Two clubs called United share the word, and a word both sides answer to
    names neither: better to skip the check than to count a goal for the wrong
    team.
    """
    labels: dict[Side, list[str]] = {Side.HOME: [state.home], Side.AWAY: [state.away]}
    if pack is not None:
        for side, sheet in ((Side.HOME, pack.home), (Side.AWAY, pack.away)):
            labels[side] += [sheet.name, sheet.short, sheet.demonym]
    words = {
        side: {token for label in names for token in fold(label).split() if len(token) >= 3}
        for side, names in labels.items()
    }
    shared = words[Side.HOME] & words[Side.AWAY]
    return {side: frozenset(found - shared) for side, found in words.items()}


def _side_named(phrase: str, teams: dict[Side, frozenset[str]]) -> Side | None:
    """Which side does this run of words name, if exactly one of them?"""
    said = {token for token in fold(phrase).split() if len(token) >= 3}
    hit = [side for side, words in teams.items() if said & words]
    return hit[0] if len(hit) == 1 else None


def _side_of_line(line: CallerLine, teams: dict[Side, frozenset[str]]) -> Side | None:
    """Whose goal "their third" is counting, from the form the caller filled in.

    The form first, because the caller writes the team name it read; the bare
    ``side`` field after it. When neither answers, the claim is left alone
    rather than guessed at: a rule that picked a side would reject true lines
    about the other one.
    """
    if line.team:
        named = _side_named(line.team, teams)
        if named is not None:
            return named
    return line.side if line.side in (Side.HOME, Side.AWAY) else None


def _is_aspiration(text: str, start: int) -> bool:
    before = _WORD.findall(text[:start])
    return bool(before) and before[-1].lower() in _ASPIRING


def _ordinal_claims(
    text: str, line: CallerLine, teams: dict[Side, frozenset[str]]
) -> list[tuple[str, Side, int]]:
    """Every claim the line makes about how many goals a side has."""
    found: list[tuple[str, Side, int]] = []
    for pattern in (_ORD_POSSESSIVE, _ORD_THEIR, _ORD_FOR_TEAM):
        for match in pattern.finditer(text):
            if _is_aspiration(text, match.start()):
                continue
            named = match.groupdict().get("team")
            side = _side_named(named, teams) if named else _side_of_line(line, teams)
            if side is None:
                continue
            found.append((match.group().strip(), side, _ORDINAL_WORDS[match.group("ord").lower()]))
    return found


#: Every way the real clips have said a goal, on top of the patterns the
#: confirmation rule uses. This is the *broad* reading, and it has two
#: consumers: the grader, which must not miss a goal that was called, and the
#: runtime, which needs to know a line is about a goal so the director cannot
#: drop it on a camera cut.
#:
#: It is deliberately not the list the `unconfirmed_goal` rule below uses.
#: The two want opposite errors. A claim detector that misses a goal hides a
#: real call from the table and lets a cut kill it; a confirmation rule that
#: over-fires silences a true line, and on the shootout clip — where the score
#: bug is absent for three minutes — that would have silenced the best-naming
#: run of the set. Narrow to decide whether the board must agree, broad to
#: decide what the line is about.
_SAID_A_GOAL = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        # The confirmation rule's pattern is the contraction only, and the
        # caller writes it out: "and it is in — Argentina have their second".
        r"\bit\s+is\s+in\b",
        r"\bburie[sd]\b",
        r"\bsquirms?\s+in\b",
        r"\bit\s+home\b",
        r"\b(?:top|bottom)\s+corner\b",
        r"\bempty\s+net\b",
        r"\bwheels?\s+away\b",
        r"\bkeeper\s+the\s+wrong\s+way\b",
        # The confirmation rule wants a digit here; a commentator says the
        # number out loud, and the broad reading has to hear it.
        rf"\bmakes?\s+it\s+(?:\d|{_WORD_ALT})\b",
    )
)


#: The present-tense half of the goal shapes above: the words that only ever
#: mean the ball is crossing the line *now*. On a replay they are the whole
#: of what "called as live" means, and they are cheap to find, which is why
#: the replay rule reads for them rather than trying to parse tense. The past
#: tenses of the same verbs — "buried", "scored", "found the net" — are
#: deliberately absent: a past-tense rebuild of the move is the line this
#: mode exists to let out.
_REPLAY_AS_LIVE = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bit'?s\s+in\b",
        r"\bit\s+is\s+in\b",
        r"\bscores\b(?!\s+(?:are|level|tied))",
        r"\bfinds\s+the\s+net\b",
        r"\bburies\b",
        r"\bsquirms\s+in\b",
        rf"\bmakes\s+it\s+(?:\d|{_WORD_ALT})\b",
    )
)

#: The events a replay may only be a replay *of* if this match has had one.
#: A replay is by definition a second look at something that already
#: happened, so a replay-scene line filed under one of these with nothing of
#: the kind anywhere in the state is not a second look at anything: it is the
#: caller inventing an incident behind a slow-motion picture, which is the
#: failure the blanket refusal was really guarding against. Build-up, a pass,
#: a carry and a cross are left out — a replay of the move is not a claim
#: that anything was given.
REPLAY_NEEDS_PRECEDENT = frozenset(
    {
        Event.GOAL,
        Event.SHOT,
        Event.SAVE,
        Event.FOUL,
        Event.PENALTY,
        Event.CARD,
        Event.OFFSIDE,
    }
)


def _state_has_seen(event: Event, state: MatchState) -> bool:
    """Has anything in the state's memory been an event of this kind?

    Three places, because three things write to the state and none of them
    writes to the others: the caller's own recent events, the incidents the
    board and the wire applied, and the statistician's named events.
    """
    return (
        event in state.last_events
        or any(incident.event is event for incident in state.incidents)
        or any(named.event is event for named in state.named)
    )


# -- notes, and the claims that need one ---------------------------------
#
# The score rules above answer "is this number the scoreboard's number". This
# one answers a different question: a line that says "three in the tournament"
# is not talking about the scoreboard at all, and nothing in the state can
# confirm or deny it. Only the pack can. So a numeric or record claim that is
# not a scoreline has to point at a note somebody actually looked up, and a
# line that reaches for one of its own is struck out whole — there is no
# trimming a statistic out of a sentence and leaving commentary behind.


def notes_for_name(pack: KnowledgePack | None, said: str) -> list[Note]:
    """Every note about the player or team this name refers to.

    The gate's own name matching, so a note filed under "Kylian Mbappé" is
    found by a line that says "Mbappé" — which is how every line says it.
    """
    if pack is None:
        return []
    return [note for note in pack.notes if is_the_same_name(said, note.about)]


def _name_words(pack: KnowledgePack | None) -> frozenset[str]:
    """Every folded word that is part of somebody's name, or a team's.

    Claims are compared word by word, and a name in the claim would otherwise
    count as a word the note is missing — the note says "always goes to the
    keeper's left" and the line says "Martínez always goes to his left",
    which agree about everything except the one word the note deliberately
    leaves out. Names are the roster rule's business, not this one's.
    """
    if pack is None:
        return frozenset()
    labels = [pack.home.name, pack.away.name, pack.home.short, pack.away.short]
    labels += [pack.home.demonym, pack.away.demonym]
    labels += [p.name for sheet in (pack.home, pack.away) for p in sheet.squad]
    return frozenset(word for label in labels for word in fold(label).split())


#: Words that carry nothing a claim can be checked on. Deliberately short:
#: "not" and "never" and every number stay in, because they are the words a
#: rephrase changes when it turns a true note into a false line.
_NOTE_FILLER_TEXT = """
    a an the this that these those his her their its our your my
    of in on at to for from with by as and or but
    he she it they we you him them
    is am are was were be been being do does did
    here there now then very quite really
"""
_NOTE_FILLER = frozenset(_NOTE_FILLER_TEXT.split())

#: What a note-backed claim is allowed to say that the note does not. Two
#: words, because a commentator says "three in the tournament already" for a
#: note that reads "three goals in this tournament", and the slack is what
#: lets a rephrase be a rephrase. Numbers are exempt from it entirely.
NOTE_SLACK = 2

#: Periods a count can be counted over. "Night" and "half" are absent on
#: purpose: "their third of the night" is a scoreline with a word missing and
#: belongs to the ordinal rule above, and "the second half" is a time of day.
_NOTE_PERIOD = (
    r"tournaments?|competitions?|world\s+cups?|seasons?|campaigns?|"
    r"qualifying|group\s+stage|careers?|finals?|cups?"
)
#: Things a commentator counts that are not the scoreline.
_NOTE_COUNTED = (
    r"goals?|assists?|finals?|caps?|titles?|troph(?:y|ies)|wins?|"
    r"clean\s+sheets?|shutouts?|penalties|appearances?|games?|matches?"
)
#: Numbers the scoreline rules never need and a statistic always does. A
#: scoreline stops at ten and the words above are all about something else —
#: "unbeaten in nineteen", "thirty-six without defeat", "his hundredth cap" —
#: which is why they are kept apart from ``_NUMBER_WORDS`` rather than added
#: to it. Widening that map would widen every scoreline pattern with it.
_NOTE_NUMBER_WORDS: dict[str, int] = {
    **_NUMBER_WORDS,
    **{
        word: value
        for word, value in (
            ("eleven", 11),
            ("twelve", 12),
            ("thirteen", 13),
            ("fourteen", 14),
            ("fifteen", 15),
            ("sixteen", 16),
            ("seventeen", 17),
            ("eighteen", 18),
            ("nineteen", 19),
            ("twenty", 20),
            ("thirty", 30),
            ("forty", 40),
            ("fifty", 50),
            ("sixty", 60),
            ("seventy", 70),
            ("eighty", 80),
            ("ninety", 90),
            ("hundred", 100),
            ("eleventh", 11),
            ("twelfth", 12),
            ("thirteenth", 13),
            ("fourteenth", 14),
            ("fifteenth", 15),
            ("twentieth", 20),
            ("hundredth", 100),
        )
    },
}
_NOTE_WORD_ALT = "|".join(sorted(_NOTE_NUMBER_WORDS, key=len, reverse=True))
_NOTE_NUM = rf"[0-9]+|{_NOTE_WORD_ALT}"

#: Every shape of claim that needs a note behind it. A hat-trick is here
#: rather than with the score rules for the reason those rules give for
#: leaving it out: it counts one man's goals, which the scoreboard does not
#: know and the pack might.
_NOTE_CLAIMS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bhat[\s-]?tricks?\b",
        rf"\b(?:his|her)\s+(?:{_ORD_ALT}){_ORD_TAIL}",
        rf"\b(?:{_ORD_ALT})\s+(?:goals?\s+)?of\s+(?:the|this)\s+(?:{_NOTE_PERIOD})\b",
        rf"\b(?:{_NOTE_NUM})\s+(?:{_NOTE_COUNTED})\b",
        rf"\b(?:{_NOTE_NUM})\s+in\s+(?:this|the)\s+(?:{_NOTE_PERIOD})\b",
        r"\bunbeaten\b",
        r"\bundefeated\b",
        r"\b(?:has|have|had)\s+not\s+(?:lost|won|conceded|scored|been\s+beaten)\b",
        r"\b(?:has|have|had)n[’']?t\s+(?:lost|won|conceded|scored|been\s+beaten)\b",
        r"\bnever\s+(?:lost|won|conceded|scored|beaten|been\s+beaten)\b",
        rf"\b(?:first|last)\s+(?:{_NOTE_PERIOD}|title|troph(?:y|ies))\b",
        r"\bsince\s+(?:19|20)[0-9]{2}\b",
        r"\bin\s+a\s+row\b",
        r"\bconsecutive\b",
        rf"\b(?:{_NOTE_NUM})\s+straight\b",
        r"\balways\b",
        r"\bevery\s+time\b",
    )
)

#: Where one clause ends and the next begins. A claim is checked as the
#: clause it sits in rather than as the words the pattern happened to match:
#: "always" on its own is not a claim anybody could check, and "always goes
#: to the keeper's left" is.
_CLAUSE_BREAK = re.compile(r"[.,;:!?]|[-–—]{1,2}\s")


def _clause_around(text: str, start: int, end: int) -> str:
    """The run of words between the nearest clause breaks either side."""
    left = 0
    right = len(text)
    for match in _CLAUSE_BREAK.finditer(text):
        if match.end() <= start:
            left = match.end()
        elif match.start() >= end:
            right = match.start()
            break
    return text[left:right].strip() or text[start:end]


def _note_tokens(text: str, names: frozenset[str]) -> list[str]:
    """The words of a claim or a note that a comparison can stand on.

    Numbers fold together so that "three" and "3" and "third" are one token:
    a note written in figures has to cover a line said in words, because that
    is the direction every line goes.
    """
    tokens: list[str] = []
    for word in fold(text).split():
        if word in names or word in _NOTE_FILLER:
            continue
        if word in _NOTE_NUMBER_WORDS:
            tokens.append(str(_NOTE_NUMBER_WORDS[word]))
        elif word in _ORDINAL_WORDS:
            tokens.append(str(_ORDINAL_WORDS[word]))
        elif word.isdigit():
            tokens.append(str(int(word)))
        else:
            tokens.append(word)
    return tokens


def _is_quantity(token: str) -> bool:
    return token.isdigit()


def _note_covers(claim: str, note: Note, names: frozenset[str]) -> bool:
    """Does this note say what the claim says, give or take a word or two?

    Every number in the claim has to be in the note. That is the whole rule:
    the failure this exists to catch is a rephrase that keeps the shape of a
    researched fact and changes the figure in it, and a changed figure is
    indistinguishable from a true one to everything downstream.
    """
    wanted = _note_tokens(claim, names)
    if not wanted:
        return False
    held = set(_note_tokens(note.text, names))
    missing = [token for token in wanted if token not in held]
    if any(_is_quantity(token) for token in missing):
        return False
    if len(missing) == len(wanted):
        return False
    return len(missing) <= NOTE_SLACK


def _note_claims(text: str) -> list[str]:
    """Every clause in the line that asserts something only a note can back."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _NOTE_CLAIMS:
        for match in pattern.finditer(text):
            clause = _clause_around(text, match.start(), match.end())
            if clause in seen:
                continue
            seen.add(clause)
            found.append(clause)
    return found


def _notes_in_play(
    text: str, pack: KnowledgePack | None, notes: Sequence[Note] | None = None
) -> list[Note]:
    """The notes about somebody this line actually names.

    A statistic attached to nobody is not checkable and is not commentary; a
    line that says "three in the tournament" without saying whose three is
    rejected here, by having no notes to match against.

    ``notes`` stands in for the pack's own list when a running count has been
    brought up to date; the filtering by name is the same either way.
    """
    if pack is None:
        return []
    available = list(notes) if notes is not None else pack.notes
    if not available:
        return []
    joined = f" {' '.join(fold(text).split())} "
    return [note for note in available if _names_the_subject(joined, note)]


def _names_the_subject(joined: str, note: Note) -> bool:
    """Does this line say whose note this is?"""
    return _names_subject(joined, note.about)


def _names_subject(joined: str, about: str) -> bool:
    """Does this line name this player or team?

    ``joined`` is the folded line with a space at each end. Any tail of the
    subject counts, because a line says "Mbappé" where the pack says "Kylian
    Mbappé", and "Di María" has to match from either word.
    """
    parts = fold(about).split()
    tails = {" ".join(parts[index:]) for index in range(len(parts))}
    return any(f" {tail} " in joined for tail in tails)


def notes_used(text: str, notes: Sequence[Note], pack: KnowledgePack | None) -> list[int]:
    """Which of these notes this line actually said. Indices, in order.

    The same match ``note_claim`` makes, read the other way round. The gate
    asks "is there a note behind this claim" and rejects the line if there is
    not; a thread asks "which note was that" and remembers it was spoken
    (``docs/research/real-commentary-corpus.md`` section 7: a fact that fires
    once is not a thread).

    Two ways a note counts as used, and the second is here because a note
    without a number in it never trips a claim pattern at all. A habit — "takes
    Argentina's penalties" — is a thread in the corpus exactly as much as a
    tally is, and it would otherwise be offered again every time.
    """
    if not notes:
        return []
    names = _name_words(pack)
    claims = _note_claims(text)
    said = set(_note_tokens(text, names))
    joined = f" {' '.join(fold(text).split())} "
    used: list[int] = []
    for index, note in enumerate(notes):
        # Whose note it is has to be in the line. Two men can hold the same
        # figure — Messi and Mbappé both went into that final on five — and
        # without this a line about one of them marks the other's thread said
        # and takes it off offer for five minutes.
        if not _names_the_subject(joined, note):
            continue
        if any(_note_covers(claim, note, names) for claim in claims):
            used.append(index)
            continue
        wanted = _note_tokens(note.text, names)
        # Every content word of the note, in the line. A high bar on purpose:
        # a thread counted as said when it was not is a thread that never
        # comes back, and the cost of missing one is only that it stays on
        # offer for a while longer.
        if len(wanted) >= 2 and all(token in said for token in wanted):
            used.append(index)
    return used



#: What the match ledger counts, and the nouns a commentator says for each.
#: The vocabulary lives here, beside the note rules, because two things need
#: it and neither may keep its own copy: :mod:`commentary.ledger` writes a
#: clause out of it, and ``ledger_claim`` below reads a clause back through
#: it. The first two entries of each row are the singular and the plural the
#: ledger writes with; the rest are only ever read.
#:
#: Deliberately short. "Effort", "attempt" and "stop" are all real words for
#: a shot and a save, and all three turn true lines into count claims — "his
#: second effort" is what a commentator says about a rebound, and a rule that
#: demanded a ledger fact for it would refuse a line about something the
#: caller never filed. The list is the nouns a count is actually attached to.
COUNT_NOUNS: dict[str, tuple[str, ...]] = {
    "shot_on_target": ("shot on target", "shots on target"),
    "shot": ("shot", "shots"),
    "save": ("save", "saves"),
    "goal": ("goal", "goals"),
    "corner": ("corner", "corners"),
    "free_kick": ("free kick", "free kicks", "free-kick", "free-kicks"),
    "penalty": ("penalty", "penalties"),
    "foul": ("foul", "fouls"),
    "offside": ("offside", "offsides"),
    "throw_in": ("throw-in", "throw-ins", "throw in", "throw ins"),
    "card": ("booking", "bookings", "card", "cards"),
    "substitution": ("substitution", "substitutions"),
    "clearance": ("clearance", "clearances"),
    "tackle": ("tackle", "tackles"),
    "cross": ("cross", "crosses"),
}


def noun_for(kind: str, *, plural: bool = False) -> str:
    """What a commentator calls this count. "booking", not "card"."""
    nouns = COUNT_NOUNS.get(kind, ())
    if not nouns:
        return kind.replace("_", " ")
    return nouns[1] if plural and len(nouns) > 1 else nouns[0]


#: Folded noun back to the kind it counts, so a claim can be read.
_NOUN_KIND: dict[str, str] = {
    fold(noun): kind for kind, nouns in COUNT_NOUNS.items() for noun in nouns
}

#: Longest first, so "shots on target" is never read as "shots" with a
#: leftover, and "free kick" never as "free".
_COUNT_NOUN_ALT = "|".join(sorted(_NOUN_KIND, key=len, reverse=True))
_COUNT_WORD_ALT = "|".join(
    sorted(set(_NOTE_NUMBER_WORDS) | set(_ORDINAL_WORDS), key=len, reverse=True)
)

#: A count of something that happened in this match: a number, cardinal or
#: ordinal, in figures or in words, directly in front of the thing it counts.
#: "Fourth corner", "his second foul", "3 shots".
#:
#: Directly in front, and that is the whole of why this rule is safe to have.
#: A pattern that fired on every number in a line would spend the match
#: rejecting "first time", "second ball", "one-two" and "eight yards out" —
#: the trap ``_ORD_TAIL`` was drawn to avoid for the scoreline, and the one
#: the position-zero name trim fell into and was deleted for.
_COUNTED = re.compile(
    rf"\b(?P<n>[0-9]+(?:st|nd|rd|th)?|{_COUNT_WORD_ALT})\s+(?P<noun>{_COUNT_NOUN_ALT})\b",
    re.IGNORECASE,
)

_LEDGER_CLAIMS = (_COUNTED,)


class CountFact(Protocol):
    """What ``ledger_claim`` needs of a :class:`commentary.ledger.Fact`.

    A protocol rather than the class itself, because
    :mod:`commentary.ledger` imports this module for its name matching and
    the dependency may only run one way.
    """

    @property
    def about(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def count(self) -> int: ...


def _count_value(token: str) -> int | None:
    """What this number token means, whatever shape it is written in."""
    word = token.lower().strip()
    if word.isdigit():
        return int(word)
    figures = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", word)
    if figures:
        return int(figures.group(1))
    if word in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[word]
    return _NOTE_NUMBER_WORDS.get(word)


def counts_in(text: str) -> list[tuple[int, str]]:
    """Every "number, thing counted" pair in this text, as value and kind."""
    found: list[tuple[int, str]] = []
    for match in _COUNTED.finditer(text):
        value = _count_value(match.group("n"))
        kind = _NOUN_KIND.get(fold(match.group("noun")))
        if value is not None and kind:
            found.append((value, kind))
    return found


def _ledger_claims(text: str) -> list[str]:
    """Every clause in the line that asserts a count of something this match."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern in _LEDGER_CLAIMS:
        for match in pattern.finditer(text):
            clause = _clause_around(text, match.start(), match.end())
            if clause in seen:
                continue
            seen.add(clause)
            found.append(clause)
    return found


def fact_covers(claim: str, fact: CountFact) -> bool:
    """Does this ledger fact hold the number this clause puts on this thing?

    Exact, both ways round. The claim's figure has to be the fact's count and
    the claim's noun has to be the fact's kind — "France's fourth corner"
    against four corners for France, and nothing looser. A count is the one
    kind of claim where being nearly right is the same as being wrong, and
    the whole reason the numbers are written in code is that a model asked to
    copy one adds to it.
    """
    return any(value == fact.count and kind == fact.kind for value, kind in counts_in(claim))


def facts_used(
    text: str, facts: Sequence[CountFact], pack: KnowledgePack | None = None
) -> list[int]:
    """Which of these ledger facts this line actually said. Indices, in order.

    The same match ``ledger_claim`` makes, read the other way round: the gate
    asks "is there a count behind this clause", and the trace asks "which
    count was that", so a row can say that a fact reached air. The pairing
    :func:`notes_used` has with ``note_claim``.
    """
    if not facts:
        return []
    joined = f" {' '.join(fold(text).split())} "
    claims = _ledger_claims(text)
    return [
        index
        for index, fact in enumerate(facts)
        if _subject_said(joined, fact.about, pack)
        and any(fact_covers(claim, fact) for claim in claims)
    ]


def _subject_said(joined: str, about: str, pack: KnowledgePack | None) -> bool:
    """Does this line say whose count this is?

    A team answers to more than one word — Argentina, the Argentines, whatever
    the team sheet's ``short`` is — and a count about a side said with the
    demonym is the same count. A player answers to any tail of his name, which
    is :func:`_names_subject`'s rule and the one every line already uses.
    """
    if _names_subject(joined, about):
        return True
    if pack is None:
        return False
    for sheet in (pack.home, pack.away):
        if not is_the_same_name(about, sheet.name):
            continue
        for label in (sheet.short, sheet.demonym, f"{sheet.demonym}s"):
            if label.strip() and _names_subject(joined, label):
                return True
    return False


#: How long a card stays cover for a line that mentions it. A booking is
#: talked about for the next few seconds — the protest, the walk away, the
#: manager on the touchline — and stops being news well before a minute is up.
CARD_RECENT_S = 60.0

#: Saying the scores are equal without saying a number. "Levels it" is a
#: scoreline claim with both numbers left out, in exactly the way
#: "Argentina's third" is one with a number left out.
#:
#: Narrow on purpose, because "level" is a word football uses for other
#: things. "Kane is level with the last man" is an offside and "the back four
#: are level" is a defensive line, so the word on its own never fires: only
#: the phrasings that need an object, and the object has to be the score.
_LEVEL_CLAIMS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bequali[sz]\w*\b",
        r"\blevell?(?:s|ed)?\s+(?:it|things|matters|the\s+(?:scores?|game|tie|match))\b",
        r"\ball\s+square\b",
        r"\blevel\s+(?:terms|pegging)\b",
        r"\bthe\s+scores?\s+(?:are|is)\s+level\b",
        r"\bit(?:'?s|\s+is)\s+(?:all\s+)?level\b(?!\s+with\b)",
    )
)

#: Saying somebody has been booked or sent off. The colour words are only
#: ever read next to a card noun or a showing verb, because a kit is yellow
#: and red far more often than a referee's hand is: the caller is told to say
#: "the near-post runner in red" when it cannot read a number.
_CARD_CLAIMS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bbooked\b",
        r"\bbooking\b",
        r"\bin(?:to)?\s+the\s+book\b",
        r"\b(?:yellow|red|second\s+yellow|straight\s+red)\s+cards?\b",
        r"\bshown\s+(?:a|the)\s+(?:yellow|red)\b",
        r"\bsecond\s+yellow\b",
        r"\bsent\s+off\b",
        r"\bmarching\s+orders\b",
        r"\bcautioned\b",
    )
)


def _first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    """The first of these the line says, as the line spells it."""
    for pattern in patterns:
        found = pattern.search(text)
        if found is not None:
            return found.group().strip()
    return None


def claims_goal(text: str, event: Event | None = None) -> bool:
    """Does this line say a goal was scored, in the form or in the words?

    The one definition. The grader asks it so the table counts a goal that
    was called however the commentator phrased it, and the runtime asks it so
    a goal arriving from a free kick or a penalty is treated as a goal by the
    director whatever the caller wrote in the event field.
    """
    if event is Event.GOAL:
        return True
    return any(p.search(text) for p in (*_GOAL_CLAIMS, *_SAID_A_GOAL))


#: The role nouns a caller reaches for instead of a name. Not an error and
#: never trimmed — "the taker" is a perfectly good phrase — but when the
#: line's own sightings already say who it is, it is the system declining to
#: use something it has, and the trace should say so. On the Messi penalty
#: the caller reported `10 Messi` in the same call that wrote "the keeper
#: goes the wrong way".
_ROLE_NOUNS = re.compile(
    r"\bthe\s+(?:taker|runner|scorer|striker|keeper|goalkeeper|forward|winger|"
    r"full[- ]back|centre[- ]half|midfielder|defender|substitute)\b",
    re.IGNORECASE,
)


def _name_withheld(line: CallerLine, spoken: str) -> str:
    """A name the caller read, reported, and then did not say.

    Logged, never acted on: the gate does not rewrite lines. It is here so
    the grader can count how often the system knows who it is watching and
    says "the taker" anyway.
    """
    if not _ROLE_NOUNS.search(spoken):
        return ""
    said = fold(spoken)
    held = [
        sighting.name
        for sighting in line.sightings
        if sighting.name and fold(_surname(sighting.name)) not in said
    ]
    return ", ".join(held)


def _claims_goal(line: CallerLine) -> bool:
    """Whether this line asserts that a goal has been scored, form or prose.

    The form field is the half that is not guesswork: a caller describing a
    goal marks the event as one. The phrases are for the line that says it
    without the form agreeing.
    """
    if line.event is Event.GOAL:
        return True
    return any(pattern.search(line.line) for pattern in _GOAL_CLAIMS)


#: Goal words in the past tense that history, not the live board, might
#: explain. Deliberately narrow: present tense ("scores") and the phrasings
#: in ``_GOAL_CLAIMS`` that describe the ball going in right now ("it's in",
#: "finds the net", "makes it 2") are never on this list, because those are
#: never about anything but this match.
_PAST_TENSE_GOAL_WORDS = re.compile(
    r"\bscored\b|\bhad\s+scored\b|\bnetted\b|\bgot\s+the\s+winner\b", re.IGNORECASE
)

#: A word that says the past-tense goal above happened somewhere other than
#: this match: a year, a competition or tournament word (the same list
#: ``note_claim`` reads for "three in the tournament"), "last season", "for
#: the club", "career", or "his/her Nth" (the ordinal note-claim pattern,
#: reused rather than re-derived).
_ELSEWHERE_WORD = re.compile(
    rf"\b(?:19|20)\d{{2}}\b"
    rf"|\b(?:{_NOTE_PERIOD})\b"
    rf"|\blast\s+season\b"
    rf"|\bfor\s+the\s+club\b"
    rf"|\bcareer\b"
    rf"|\b(?:his|her)\s+(?:{_ORD_ALT}){_ORD_TAIL}",
    re.IGNORECASE,
)

#: "in Russia" — the place half of the same marker, kept apart from
#: ``_ELSEWHERE_WORD`` because it has to stay case-sensitive: a capitalised
#: word after "in" is what makes it a place and not a preposition, and
#: folding the case away would lose exactly that. Captured on its own so
#: ``_elsewhere_place`` can hand just the word to the roster, below.
_ELSEWHERE_PLACE = re.compile(r"\bin\s+([A-Z][a-zA-Z]+)\b")


def _is_historical_goal_reference(
    text: str, pack: KnowledgePack | None, notes: Sequence[Note] | None
) -> bool:
    """A past-tense goal word about some other match, not a claim about this one.

    "The man who scored in Russia." matches ``_GOAL_CLAIMS`` on the same word
    "Mbappé scored" does, and the two need opposite verdicts: the second is
    happening now and needs the board's say-so, the first is a career fact a
    board could never confirm in the first place. Tense alone is not enough
    — "he's scored before tonight" says nothing about when — so this asks for
    either a marker that places the goal elsewhere, or a pack note the line is
    simply restating, checked the same way ``note_claim`` checks it.
    """
    if not _PAST_TENSE_GOAL_WORDS.search(text):
        return False
    if _ELSEWHERE_WORD.search(text) or _ELSEWHERE_PLACE.search(text):
        return True
    names = _name_words(pack)
    return any(_note_covers(text, note, names) for note in _notes_in_play(text, pack, notes))


def _elsewhere_place(text: str) -> str | None:
    """The capitalised place word that excuses a past-tense goal, if any.

    The same word the roster check would otherwise read as an unverified
    name and trim: on the real Mbappé trace, "The man who scored in Russia."
    passed ``unconfirmed_goal`` and then lost "Russia" to
    ``trimmed_name``, coming out as "The man who scored in." — the fix this
    line is here for, at the one call site that needs it.
    """
    if not _PAST_TENSE_GOAL_WORDS.search(text):
        return None
    found = _ELSEWHERE_PLACE.search(text)
    return found.group(1) if found else None


# -- decoration: the ban only the model enforces --------------------------
#
# src/commentary/prompts/phraser.py tells the model never to decorate — "no
# adjectives for atmosphere, no 'the crowd rises'" — and spells out the one
# shape that keeps sneaking past it: "any group of people made to erupt. The
# whole crowd erupts, and the corner erupts, the bench erupts — the noun
# changes and the tell does not. Nobody who is not the ball or a player gets
# a verb at all." A prompt is not a checker. On ``runs/rephrased/mbappe-researched``
# "Mbappé! Into the net! The whole bench erupts!" went out at 86.8 because
# nothing downstream of the model was holding that rule to it. This is that
# checker: deterministic, and — like the rest of the gate — never rewriting a
# line, only cutting the sentence that breaks the rule and judging what is
# left.

#: The nouns that are never allowed a verb. "Corner" is here for its crowd
#: sense only ("the corner erupts"), which is safe to catch because nothing
#: on this list is also on ``_ATMOSPHERE_VERB`` — a corner *kick* is never
#: said to erupt, rise, roar, go wild, or sit on its feet.
#:
#: ``bench``, ``fans``, ``supporters`` and ``end`` carry an extra, optional
#: word in front of the article: "the *Argentina* bench", "*their own*
#: supporters", "the *away* end" are all still a crowd noun, and a side's
#: name in front of one is not what turns it into a player. Nothing on the
#: other nouns takes that qualifier — "the France corner" is not a shape
#: real commentary uses and widening the match there buys nothing.
_DECORATION_SIDE_WORD = (
    r"(?:the\s+(?:whole\s+|entire\s+)?|their\s+own\s+|his\s+own\s+|her\s+own\s+)?"
)
_DECORATION_SUBJECT = (
    rf"{_DECORATION_SIDE_WORD}(?:[A-Za-z][\w'’-]*\s+)?(?:bench(?:es)?|fans|supporters|end)"
    rf"|(?:the\s+(?:whole\s+|entire\s+)?)?(?:crowd|stadium|ground|dugout|corner|stands?|place)"
    r"|everyone"
)

#: The verbs a crowd noun is never allowed. Spelled out rather than stemmed,
#: because a stem wide enough to catch "erupted" is wide enough to catch
#: "erupting into song" being sung by a player, which is nobody's decoration.
#:
#: "Up" is bare rather than three separate phrases because the claim only
#: has to match where the sentence *opens*, not where it ends: "up", "up on
#: their feet" and "up as one" are all caught by the one word. "The whole
#: bench is up!" (86.8) and "the whole bench celebrates in front of their
#: own supporters" (193.7) are the two real lines this list was missing.
_ATMOSPHERE_VERB = (
    r"erupt(?:s|ed)?|rises?|roars?|go(?:es|ing)?\s+(?:wild|mad|crazy|berserk)|"
    r"on\s+their\s+feet|up|off\s+their\s+seats|"
    r"in\s+raptures|in\s+dreamland|in\s+full\s+voice|bouncing|silenced|stunned|"
    r"celebrat(?:es?|ing)|jumping|waving|dancing|singing"
)

#: A decoration claim only if the crowd noun *opens* the sentence — the
#: subject seat, not just a word that turns up somewhere in it. "Messi rises
#: to meet it" has the same verb and a player in the subject seat instead,
#: and anchoring the pattern at the start of the sentence is what tells the
#: two apart without parsing the rest of the grammar.
_DECORATION_CLAIM = re.compile(
    rf"^(?:(?:and|but)\s+)?(?:{_DECORATION_SUBJECT})\b(?:['’]s)?\s+(?:is\s+|are\s+|was\s+|were\s+)?"
    rf"(?:{_ATMOSPHERE_VERB})\b",
    re.IGNORECASE,
)

#: A commentator's own sentence boundaries — the three marks the gate's own
#: module docstring uses to describe a line. Not ``_CLAUSE_BREAK``, which also
#: splits on commas and dashes for the note rule's purposes: a decoration
#: claim is judged by the sentence it sits in, whole, because trimming it is
#: trimming the sentence, not the clause.
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def decoration_claim(text: str) -> list[tuple[int, int]]:
    """Every sentence in this line that is atmosphere, not commentary.

    A sentence — split on ``.``, ``!``, ``?`` — is a decoration claim when it
    opens with a crowd noun (the crowd, the fans, the stadium, the bench, the
    dugout, the corner, the stands, an end, a side's own supporters,
    everyone...), an optional "and" or "but" allowed in front of it the way a
    commentator actually opens a sentence, holding an atmosphere verb (erupts,
    rises, roars, goes wild, up, off their seats, in raptures, in dreamland,
    in full voice, bouncing, celebrating, jumping, waving, dancing, singing,
    silenced, stunned). "Mbappé! The volley, buried!" has neither sentence
    open that way and is untouched; "Mbappé! Into the net! The whole bench
    erupts!" has its third sentence flagged, and so does "And the whole bench
    celebrates in front of their own supporters."

    Spans are of the whole sentence, punctuation included, so a caller can
    cut one out of the line and be left with a line rather than a stub with
    a stray full stop.
    """
    found: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        _record_decoration(text, start, match.end(), found)
        start = match.end()
    _record_decoration(text, start, len(text), found)
    return found


def _record_decoration(text: str, start: int, end: int, found: list[tuple[int, int]]) -> None:
    sentence = text[start:end]
    stripped = sentence.lstrip()
    if not stripped:
        return
    lead = len(sentence) - len(stripped)
    if _DECORATION_CLAIM.match(stripped):
        found.append((start + lead, end))


#: The floor a decoration trim leaves the sentence at, and it is a lower
#: floor than ``min_words_after_trim``'s default of three on purpose: two
#: words is "Mbappé! Into the net!" with the crowd cut off the end of it,
#: and that line is worth keeping. What makes it worth keeping is not the
#: word count alone — ``_has_content_after_trim`` below asks the second
#: question a bare count cannot: is one of those words a name or an action,
#: or is a stub of grammar all that is left.
_DECORATION_MIN_WORDS = 2

#: The football vocabulary that says something happened, for a sentence that
#: has no verified name in it to lean on. Deliberately the actions and set
#: pieces, not the grammar: a decoration trim that left "It is" behind should
#: not pass for having two words, and neither list is the stopword list
#: above, which exists to say what is *not* a name rather than what is an
#: event.
_EVENT_WORDS_TEXT = """
    goal goals shot shots save saves cross crosses corner corners penalty
    penalties header headers volley volleys chip chips slot slots strike
    strikes tackle tackles chance chances offside foul fouls card cards
    booking bookings free kick kicks throw pass passes clearance clearances
    block blocks blocked net nets ball
"""
_EVENT_WORDS = frozenset(_EVENT_WORDS_TEXT.split())


def _has_content_after_trim(kept: str, roster: _Roster, threshold: float) -> bool:
    """Is there still commentary here, or only the scaffolding a name sat in?

    Two words and either a verified name or a football action word — the
    two ways a sentence says something. "Mbappé! Into the net!" clears it on
    the name alone; a line with the decoration cut off and nothing else in
    it does not.
    """
    if _word_count(kept) < _DECORATION_MIN_WORDS:
        return False
    if any(
        _matches_roster(candidate.text, roster, threshold) for candidate in _candidates(kept)
    ):
        return True
    return any(fold(word) in _EVENT_WORDS for word in kept.split())


def _tidy(text: str) -> str:
    """Repair a line that has had a name cut out of the middle of it."""
    out = re.sub(rf"\b(?:{_DANGLERS})\s+(?=(?:[,.;!?]|and\b|as\b|but\b|who\b|then\b|$))", "", text)
    out = re.sub(r"\s+", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", out)
    out = re.sub(r"^[\s,;:.!?-]+", "", out)
    out = re.sub(r"^(?:and|but|as)\s+", "", out, flags=re.IGNORECASE)
    out = out.strip()
    return out[:1].upper() + out[1:] if out else out


def _word_count(text: str) -> int:
    return len([word for word in text.split() if any(ch.isalnum() for ch in word)])


@dataclass
class GateStats:
    """Running totals, so a run can print the table without re-deriving it."""

    judged: int = 0
    passed: int = 0
    trimmed: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    @property
    def rejected(self) -> int:
        return self.judged - self.passed

    def record(self, verdict: GateVerdict) -> None:
        self.judged += 1
        tags = [reason.split(":", 1)[0] for reason in verdict.reasons]
        if verdict.passed:
            self.passed += 1
            if "trimmed_name" in tags or "trimmed_decoration" in tags:
                self.trimmed += 1
            return
        for tag in tags:
            if tag not in ("trimmed_name", "trimmed_decoration"):
                self.by_reason[tag] += 1

    def table(self) -> str:
        header = (
            f"judged {self.judged}  passed {self.passed} ({self.trimmed} trimmed)  "
            f"rejected {self.rejected}"
        )
        rows = [f"  {tag:<28} {count}" for tag, count in self.by_reason.most_common()]
        return "\n".join([header, *rows])


class FactGate:
    """Judges one caller line against the roster, the board, and the picture."""

    def __init__(self, cfg: GateConfig = SETTINGS.gate) -> None:
        self.cfg = cfg
        self.stats = GateStats()

    def judge(
        self,
        line: CallerLine,
        state: MatchState,
        pack: KnowledgePack | None,
        *,
        board_changed: bool = False,
        wire_confirmed: bool = False,
        goal_in_state: bool = False,
        carried: str | None = None,
        at: float | None = None,
        notes: Sequence[Note] | None = None,
        ledger: Sequence[CountFact] = (),
    ) -> GateVerdict:
        """Pass, trim, or reject — and always say why.

        ``board_changed`` is whether the scoreboard supports a goal claimed
        at this moment: a settled change near the cursor, a change the board
        reader is still confirming, or a goal already in the state.
        ``wire_confirmed`` is a statistician having said so, which is only
        ever true in the ablation that runs one.

        ``goal_in_state`` is the third of those three, on its own, and it is
        what tells the arithmetic which situation it is in. A goal the state
        has already counted is in the score; the next line about it may say
        that score and no other. A goal the board is only now agreeing to is
        not in the score yet, and the line calling it is allowed to be one
        ahead.

        ``carried`` is the player the last line had on the ball, seconds ago.
        A name the caller can no longer read is still the name of the man it
        is describing, and the runtime decides whether the carry applies; the
        gate's part is to accept it as verified when it does.

        ``at`` is the cursor this line is about, and only the card rule uses
        it: a card the state was told about is cover for a line mentioning
        one, but only while it is still the thing that just happened. Left
        out, a card anywhere in the state counts, which is the right default
        for a caller that cannot say when.

        ``notes`` replaces the pack's own notes for the statistic rule, and
        exists for one reason: a note that counts something is only true as of
        kickoff. :class:`commentary.tallies.Tallies` adds what has happened
        since, and the adjusted clauses are what the phraser was shown — so
        they have to be what the line is checked against, or the gate rejects
        the model for using the number it was handed. Left out, the pack's own
        notes are used, which is every caller of this before tallies existed.

        ``ledger`` is what :class:`commentary.ledger.Ledger` holds right now:
        the counts this broadcast has seen for itself, as the clauses the
        voice was offered. It backs ``ledger_claim`` the way ``notes`` backs
        ``note_claim``, and the two are checked together — a number covered by
        either passes — because a researched fact and a count off the pictures
        are two sources for the same sentence. Left out, no line may put a
        number on anything the pack does not already say, which is every
        caller of this before the ledger existed.
        """
        verdict = self._judge(
            line,
            state,
            pack,
            board_changed=board_changed,
            wire_confirmed=wire_confirmed,
            goal_in_state=goal_in_state,
            carried=carried,
            at=at,
            notes=notes,
            ledger=ledger,
        )
        self.stats.record(verdict)
        return verdict

    def _judge(
        self,
        line: CallerLine,
        state: MatchState,
        pack: KnowledgePack | None,
        *,
        board_changed: bool,
        wire_confirmed: bool = False,
        goal_in_state: bool = False,
        carried: str | None = None,
        at: float | None = None,
        notes: Sequence[Note] | None = None,
        ledger: Sequence[CountFact] = (),
    ) -> GateVerdict:
        text = line.line.strip()
        if not line.speak:
            return GateVerdict(passed=False, reasons=["not_speaking: caller chose silence"])
        if not text:
            return GateVerdict(passed=False, reasons=["empty_line: nothing to say"])

        roster = _roster_of(state, pack)
        if carried:
            roster = replace(
                roster,
                people=roster.people | {fold(carried), fold(_surname(carried))},
            )
        # A place that excuses a past-tense goal ("in Russia") is not a name
        # claim, but it is a run of capitalised words like any other and the
        # roster check below cannot otherwise tell it apart from one — it is
        # exactly what invented "The man who scored in." out of "The man who
        # scored in Russia." on the real trace this rule exists for.
        place = _elsewhere_place(text)
        if place:
            roster = replace(roster, people=roster.people | {fold(place)})
        # The decoration ban the phraser's prompt states and only the model
        # enforces: cut the sentence that gives a crowd noun a verb, or
        # refuse the line if cutting it leaves nothing worth saying. Ahead of
        # every other check because it is the same question ``_trim_unverified``
        # asks of names, asked here of atmosphere, and everything below should
        # see the line the caller would actually be left with.
        decorated, decoration_reasons = self._check_decoration(text, pack, notes, roster)
        if decorated is None:
            return GateVerdict(passed=False, reasons=decoration_reasons)
        text = decorated
        # A goal is being called that the score does not yet include: the ball
        # has crossed the line, something outside the caller agrees, and the
        # board has not caught up. That and only that buys a line the right to
        # be one goal ahead of the state.
        goal_incoming = (board_changed or wire_confirmed) and not goal_in_state
        fatal: list[str] = []
        if line.scene is Scene.REPLAY:
            fatal += self._check_replay(text, line, state, pack, notes, goal_in_state=goal_in_state)
        fatal += self._check_sightings(line, roster, pack)
        fatal += self._check_scoreline(text, state, goal_incoming=goal_incoming)
        fatal += self._check_score_claims(text, line, state, pack, goal_incoming=goal_incoming)
        fatal += self._check_level_claim(text, state, goal_incoming=goal_incoming)
        fatal += self._check_card_claim(text, line, state, at=at)
        fatal += self._check_counts(text, pack, notes, ledger)
        # A replay's goal claim is backed by the score already counting it,
        # which is the only thing a replay can be evidence of: the board moved
        # while the live call was going out, seconds ago. Without this a
        # past-tense rebuild over the replay of a goal this system has already
        # called dies on a rule about calling goals nobody has confirmed.
        goal_backed = (
            board_changed
            or wire_confirmed
            or (line.scene is Scene.REPLAY and goal_in_state)
        )
        if (
            self.cfg.require_board_for_goal
            and _claims_goal(line)
            and not goal_backed
            and not (
                line.event is not Event.GOAL
                and _is_historical_goal_reference(text, pack, notes)
            )
        ):
            fatal.append("unconfirmed_goal: no board change, no wire")
        if fatal:
            return GateVerdict(passed=False, reasons=fatal)

        verdict = self._trim_unverified(text, roster)
        verdict.reasons = decoration_reasons + verdict.reasons
        withheld = _name_withheld(line, verdict.line)
        if withheld:
            verdict.reasons.append(f"name_withheld: {withheld}")
        return verdict

    def _check_replay(
        self,
        text: str,
        line: CallerLine,
        state: MatchState,
        pack: KnowledgePack | None,
        notes: Sequence[Note] | None,
        *,
        goal_in_state: bool,
    ) -> list[str]:
        """What a replay-scene line is allowed to be.

        This replaces the blanket ``scene_replay`` refusal. That rule was
        right about the failure it named and wrong about the cost: on
        ``runs/trigger/mbappe/file-20260913-185228.jsonl`` the penalty was
        conceded at 12.9 s and the next line spoken was at 49.0 s, and in
        between the caller had written four accurate replay lines — "The
        replay: driving across, the leg in behind him, and down he goes" — and
        every one of them died here. Real commentary does the opposite: the
        corpus's half-minute after a goal is seven utterances and sixty words
        (``docs/research/real-commentary-corpus.md`` section 2.4) and section
        3.2 shows most of them are over replays.

        So the scene is not the refusal. Three things are:

        ``replay_of_nothing``
            A replay is a second look at something that happened. A line
            filed under :data:`REPLAY_NEEDS_PRECEDENT` with nothing of that
            kind anywhere in the state is not a second look; it is a new
            claim wearing a slow-motion picture.

        ``replay_score``
            A replay line never restates the score, in figures, in an ordinal
            or in "levels it". The score went out on the live call, and a
            replay is the one place a commentator is furthest from the
            scoreboard — the bug is usually not even on screen.

        ``replay_as_live``
            The present-tense shapes that mean the ball is crossing the line
            now. The rest of the tense question is left to the phraser's
            prompt: this is the cheap, certain half, and it is the half that
            embarrasses a broadcast.

        A goal claim is allowed on a replay only when the score already
        counts that goal, which is the ``goal_in_state`` flag the runtime and
        the rephrase both already compute.
        """
        problems: list[str] = []
        if score_spans(text) or ordinal_score_spans(text) or level_claim_spans(text):
            problems.append("replay_score: a replay line never restates the score")
        said = _first_match(text, _REPLAY_AS_LIVE)
        if said is not None:
            problems.append(f"replay_as_live: {said}")
        historical = line.event is not Event.GOAL and _is_historical_goal_reference(
            text, pack, notes
        )
        if claims_goal(text, line.event) and not goal_in_state and not historical:
            problems.append("replay_goal: a goal on a replay the score does not count")
        unseen = line.event in REPLAY_NEEDS_PRECEDENT and not _state_has_seen(line.event, state)
        if unseen and not (line.event is Event.GOAL and goal_in_state):
            problems.append(
                f"replay_of_nothing: no {line.event.value} in the state for this replay"
            )
        return problems

    def _check_decoration(
        self,
        text: str,
        pack: KnowledgePack | None,
        notes: Sequence[Note] | None,
        roster: _Roster,
    ) -> tuple[str | None, list[str]]:
        """Cut the atmosphere out of the line, or say the line cannot survive it.

        Returns the text with any decoration sentences removed and the
        reasons to record, or ``None`` for the text when cutting them leaves
        too little of the line to be worth saying — the whole-line refusal
        the task's own rule calls for, tagged ``decoration`` the way every
        other rejection in this gate is tagged with the question it failed.

        A sentence backed by a pack note is not decoration at all: "the home
        end have been bouncing all night" restates something somebody
        actually looked up, and that is ``note_claim``'s question, asked here
        with the same coverage check it uses everywhere else.
        """
        spans = decoration_claim(text)
        if not spans:
            return text, []
        names = _name_words(pack)
        in_play = _notes_in_play(text, pack, notes)
        cut: list[tuple[int, int, str]] = []
        for start, end in sorted(spans, reverse=True):
            sentence = text[start:end].strip()
            if any(_note_covers(sentence, note, names) for note in in_play):
                continue
            cut.append((start, end, sentence))
        if not cut:
            return text, []
        kept = text
        for start, end, _sentence in cut:
            kept = kept[:start] + kept[end:]
        kept = _tidy(kept)
        problems = [f"decoration: {sentence}" for _, _, sentence in cut]
        if not _has_content_after_trim(kept, roster, self.cfg.name_match_threshold):
            return None, problems
        trims = [f"trimmed_decoration: {sentence}" for _, _, sentence in cut]
        return kept, problems + trims

    def _check_sightings(
        self, line: CallerLine, roster: _Roster, pack: KnowledgePack | None
    ) -> list[str]:
        """A number or a name claimed off the picture is held to the roster.

        This is the one place the gate refuses to trim. A read that is on no
        roster means the caller did not misjudge a face, it invented one, and
        a line built on an invented reading is not worth saving.

        Both halves are checked and so is their agreement: the number has to
        be in that squad, the name has to be on the roster, and when a
        sighting carries both they have to be the same player. Two readings
        of one shirt that cannot both be right is neither.

        The tag is not checked at all. It is a letter this system drew over a
        body itself, not a claim about the world, and it has its own field so
        that it can never arrive here dressed as a name.
        """
        problems: list[str] = []
        for sighting in line.sightings:
            if sighting.number is not None:
                problems += self._check_number(str(sighting.number), roster, pack)
            name = (sighting.name or "").strip()
            if not name:
                continue
            if not _matches_roster(name, roster, self.cfg.name_match_threshold):
                problems.append(f"name_read_not_on_roster: {name}")
            elif (
                sighting.number is not None
                and pack is not None
                and not _is_that_player(pack, sighting.number, name)
            ):
                problems.append(f"sighting_disagrees: {name} is not number {sighting.number}")
        return problems

    def _check_number(self, number: str, roster: _Roster, pack: KnowledgePack | None) -> list[str]:
        if pack is None:
            return []
        squad = {known.lstrip("0") for known in roster.numbers}
        if number.lstrip("0") in squad:
            return []
        return [f"number_not_in_squad: {number}"]

    def _check_scoreline(self, text: str, state: MatchState, *, goal_incoming: bool) -> list[str]:
        """Either orientation is allowed: the gate cannot know which team was meant."""
        board = (state.home_score, state.away_score)
        allowed = {board, board[::-1]}
        if goal_incoming:
            # "And that's three-nil!" as the ball hits the net, seconds before
            # the graphic says so. One goal on one side, and only while a goal
            # is actually being called.
            for bumped in ((board[0] + 1, board[1]), (board[0], board[1] + 1)):
                allowed |= {bumped, bumped[::-1]}
        problems: list[str] = []
        for home, away in _stated_scores(text):
            if (home, away) not in allowed:
                problems.append(
                    f"scoreline_mismatch: said {home}-{away}, board {board[0]}-{board[1]}"
                )
        return problems

    def _check_score_claims(
        self,
        text: str,
        line: CallerLine,
        state: MatchState,
        pack: KnowledgePack | None,
        *,
        goal_incoming: bool,
    ) -> list[str]:
        """An ordinal is a scoreline with one number left out, and it is checked.

        On the Di María clip the caller called the second goal twice more
        after the state already held it — "Messi, from the rebound!
        Argentina's third", then "De Paul! Argentina's fourth!" — and both
        went out. Neither was a phantom goal by the board rule: a goal *was*
        in the state, which is the cover every celebration line needs. What
        made them lies was the counting, and counting is arithmetic the gate
        can do on its own.

        So: a side that has scored *n* may be said to have scored *n*, and one
        more than *n* only while a goal is being called that the state has not
        taken in yet. Nothing else, and the whole line goes — there is no
        trimming a number out of a sentence and leaving commentary behind.
        """
        board = (state.home_score, state.away_score)
        teams = _team_words(state, pack)
        problems: list[str] = []
        for said, side, count in _ordinal_claims(text, line, teams):
            held = board[0] if side is Side.HOME else board[1]
            allowed = {held, held + 1} if goal_incoming else {held}
            if count not in allowed:
                problems.append(f"score_claim: {said} vs state {board[0]}-{board[1]}")
        return problems

    def _check_level_claim(
        self, text: str, state: MatchState, *, goal_incoming: bool
    ) -> list[str]:
        """"Levels it" is a scoreline with both numbers left out.

        The phrasing stage found the hole. The caller had written "Mbappé is
        already into the net for the ball, hauling it back to the centre
        circle" at two-one, and the rewrite said "Mbappé! Levels it!" — a
        claim about the score, at a score that was not level, through a gate
        whose arithmetic only reads digits and ordinals. It went out.

        Same latitude as every other score rule and for the same reason: a
        goal being called that the state has not taken in yet is allowed to
        be one ahead of the graphic, so a side a goal behind may be said to
        be levelling while it scores. Nothing else, and the whole line goes.
        """
        said = _first_match(text, _LEVEL_CLAIMS)
        if said is None:
            return []
        home, away = state.home_score, state.away_score
        if home == away:
            return []
        if goal_incoming and abs(home - away) == 1:
            return []
        return [f"level_claim: {said} vs state {home}-{away}"]

    def _check_card_claim(
        self, text: str, line: CallerLine, state: MatchState, *, at: float | None
    ) -> list[str]:
        """A booking is a thing that happened, not a thing to reach for.

        The same rewrite turned "Otamendi protests, and the referee is
        already waving him away" — a foul, given, nobody booked — into
        "Otamendi in the book." Reaching for the bigger word is what a model
        does when it is asked to be vivid, and a card nobody was shown is a
        fact about the match that this system invented.

        Three things are cover, in the order they are available. The form
        says ``card``, which is the caller having seen one. The state was
        told about a card by a statistician, recently. Or the state has a
        card among the events it has just seen, which is the only one of the
        three a pictures-only run ever produces — it carries no side and no
        time, and it is what lets the line after the booking still mention
        it.
        """
        said = _first_match(text, _CARD_CLAIMS)
        if said is None:
            return []
        if line.event is Event.CARD:
            return []
        if Event.CARD in state.last_events:
            return []
        if self._card_in_state(line, state, at):
            return []
        return [f"card_claim: {said}, and no card in the form or the state"]

    def _check_counts(
        self,
        text: str,
        pack: KnowledgePack | None,
        notes: Sequence[Note] | None = None,
        ledger: Sequence[CountFact] = (),
    ) -> list[str]:
        """Every number in the line that is not the score, against its source.

        Two rules and one pass, because a clause can only be judged once. A
        line may put a number on something for exactly two reasons: somebody
        researched it before kickoff, which is ``note_claim`` below, or this
        broadcast counted it for itself, which is ``ledger_claim``. A clause
        either of them covers passes; a clause neither covers is refused
        whole, and the tag says which rule found it.

        Checking them together rather than one after the other is the whole
        of the coordination. "Mbappé's second goal" is a count off the ledger
        and a tally off the pack, and a rule that ran alone would refuse it
        for not being the other kind.

        **note_claim.** Everything else the gate checks can be checked against
        something the system saw for itself: the roster came off a team sheet,
        the score came off the scoreboard, the card was on the form. A career
        count is different. "Mbappé, three in the tournament" is unfalsifiable
        from inside the broadcast — no camera shows it, no scoreboard carries
        it — and it is exactly the kind of sentence a model writes when it is
        asked to sound like a commentator. So the rule is a lookup, not
        arithmetic: find the clauses that assert a count, an ordinal, a streak
        or a habit, find the notes about the people the line names, and
        require each clause to be one of those notes, reworded by a word or
        two but not renumbered.

        **ledger_claim.** The other half of the same thought, and the half
        that was missing. A count of this match — a fourth corner, a second
        foul, a first shot on target — is not in the pack and cannot be,
        because nobody knew it before kickoff. It is in
        :mod:`commentary.ledger`, which counted it off the caller's own forms,
        and the clause the voice was shown came out of there with the figure
        already written. A number the model wrote instead of the one it was
        handed is caught here by being a number the match does not hold.

        A line with no such clause reaches none of this and is untouched.
        Rejection is whole-line, like the score rules and for the same reason:
        a sentence with the statistic cut out of it is not a shorter sentence,
        it is a different one.
        """
        note_claims = _note_claims(text)
        claims: list[tuple[str, str]] = [(clause, "note_claim") for clause in note_claims]
        claims += [
            (clause, "ledger_claim")
            for clause in _ledger_claims(text)
            if clause not in note_claims
        ]
        if not claims:
            return []
        in_play = _notes_in_play(text, pack, notes)
        names = _name_words(pack)
        joined = f" {' '.join(fold(text).split())} "
        held = [fact for fact in ledger if _subject_said(joined, fact.about, pack)]
        problems: list[str] = []
        for claim, tag in claims:
            if any(_note_covers(claim, note, names) for note in in_play):
                continue
            if any(fact_covers(claim, fact) for fact in held):
                continue
            problems.append(
                f"note_claim: {claim} not in the pack"
                if tag == "note_claim"
                else f"ledger_claim: {claim} is not a count this match holds"
            )
        return problems

    @staticmethod
    def _card_in_state(line: CallerLine, state: MatchState, at: float | None) -> bool:
        """Has a statistician reported a card for this side, recently?"""
        reported: list[tuple[Event, Side, float]] = [
            (event.event, event.side, event.video_ts) for event in state.named
        ] + [(item.event, item.side, item.video_ts) for item in state.incidents]
        for event, side, video_ts in reported:
            if event is not Event.CARD:
                continue
            if line.side in (Side.HOME, Side.AWAY) and side not in (line.side, Side.UNKNOWN):
                continue
            if at is None or 0.0 <= at - video_ts <= CARD_RECENT_S:
                return True
        return False

    def _trim_unverified(self, text: str, roster: _Roster) -> GateVerdict:
        """Cut the names that cannot be verified and keep whatever still stands up."""
        reasons: list[str] = []
        kept = text
        for candidate in sorted(_candidates(text), key=lambda c: c.start, reverse=True):
            if _matches_roster(candidate.text, roster, self.cfg.name_match_threshold):
                continue
            reasons.append(f"name_not_on_roster: {candidate.text}")
            reasons.append(f"trimmed_name: {candidate.text}")
            kept = kept[: candidate.start] + kept[candidate.end :]
        if reasons:
            # Only a line that lost a name is measured. A line the caller
            # wrote short is a line — "Modrić, Perišić." is how the build-up
            # is called — and it goes out as written.
            kept = _tidy(kept)
            words = _word_count(kept)
            if words < self.cfg.min_words_after_trim:
                reasons.append(f"too_short_after_trim: {words} words left")
                return GateVerdict(passed=False, reasons=reasons)
        return GateVerdict(passed=True, reasons=reasons, line=kept)
