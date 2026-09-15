"""The numbers, written by code. The model writes the words.

The rule this module enforces is the one agreed in ``docs/HANDOFF.md``
section 3d: **the model never writes a number.** The phraser writes the name
and the how; the score is composed here, off the state, and appended to the
goal-calling line by the broadcast rather than by the model.

Why it had to stop being the model's job. Version 3 of the phraser was told
the score and told to copy it. It copied it twice and did arithmetic on it
twice, and the gate refused both:

    82.5  Mbappé! Buried past Martínez! Two-one.   passed
    86.8  Mbappé! Three-two.                       scoreline_mismatch
   176.7  Mbappé! Off the ground! Two-two.         passed
   180.5  Mbappé! The volley! Three-two.           scoreline_mismatch

Both refusals are celebration lines, and a refused line is dropped whole, so
the second beat of each goal never reached the replay. Version 4 handed the
model the score as words to copy and was discarded unmerged, because copying
is still something a model can get wrong. What is left is this: code composes
the number, code appends it to exactly one line per goal, and
:func:`strip_score` takes out any number the model wrote anyway so that the
line survives instead of being refused.

Three things live here.

:func:`say_score`
    The scoreline as a commentator says it — "Two-two.", "One-nil to
    Argentina.", "Three-two, France." The forms are lifted from
    ``docs/research/real-commentary-corpus.md`` section 8.4 slot 4 and section
    5.3, and they rotate on the number of goals in the match so a listener
    does not hear the same shape all evening.

:func:`strip_score`
    Every scoreline, team goal-count ordinal and level claim out of a line,
    using the gate's own regexes (:func:`commentary.gate.score_spans` and its
    two neighbours) so that a shape the strip misses is never a shape the gate
    catches. The scorer's own tally — "his third of the campaign" — is left
    alone on purpose: that is a note claim, the gate checks it against the
    pack, and it is the corpus's *third* beat of a goal.

:func:`restatement`
    The score-and-clock line a club feed says every few minutes ("Ten minutes
    gone, two-nil to Barcelona"), which Gap 8 item 4 asks for as code rather
    than as a model line for exactly the same reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from commentary.gate import (
    is_a_hole,
    level_claim_spans,
    ordinal_score_spans,
    score_spans,
)
from commentary.schemas import MatchState, Side

#: How many goals a scoreline is spelled out to. Above this the figures go out
#: as digits, which no real match will ever reach and no synthesiser will
#: stumble over.
SPELLED_TO = 19

_UNITS = (
    "nil",
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
_TENS = {
    2: "twenty",
    3: "thirty",
    4: "forty",
    5: "fifty",
    6: "sixty",
    7: "seventy",
    8: "eighty",
    9: "ninety",
}


def in_words(value: int, *, zero: str = "nil") -> str:
    """A number as a commentator says it, not as a scoreboard prints it.

    ``zero`` is a parameter because the same digit is two different words in
    the two places this module needs it: a score of nothing is "nil" and a
    clock of nothing is "no minutes", and "nil minutes gone" is not English.
    """
    if value < 0 or value > 99:
        return str(value)
    if value == 0:
        return zero
    if value <= SPELLED_TO:
        return _UNITS[value]
    tens, units = divmod(value, 10)
    head = _TENS[tens]
    return head if units == 0 else f"{head}-{_UNITS[units]}"


def effective_score(
    state: MatchState, side: Side, *, goal_in_state: bool
) -> tuple[int, int] | None:
    """The score this line is about: the board's, plus the goal arriving.

    The board lags the ball. A goal call goes out while the graphic still
    reads 2-1 and the ball is in the net for 2-2, and ``goal_in_state`` is the
    runtime's own answer to whether the number has caught up — the same flag
    the gate is handed, so the score composed here and the score the gate
    allows cannot disagree.

    ``None`` when the arithmetic cannot be done: a goal the board has not
    taken in, scored by a side nobody could name, is a score this module
    declines to guess at. The corpus's own instruction, in the phraser's
    prompt: if you are not sure of the score, the name and the how are a whole
    line.
    """
    home, away = state.home_score, state.away_score
    if goal_in_state:
        return home, away
    if side is Side.HOME:
        return home + 1, away
    if side is Side.AWAY:
        return home, away + 1
    return None


def say_score(
    state: MatchState,
    side: Side = Side.UNKNOWN,
    *,
    goal_in_state: bool = False,
    index: int | None = None,
) -> str:
    """The scoreline as its own short utterance, or ``""`` if it cannot be said.

    Section 8.4 slot 4 of the corpus study: the score arrives as a separate
    short utterance within about five seconds of the strike, in one of a small
    closed set of shapes. Three of them are here — the bare pair, the pair
    with the leading side named after "to", and the pair with the side in
    apposition — and a level score takes the two the corpus uses for a draw.

    The leading side goes first, which is how every example in the corpus
    reads it ("It's 1-0 to Barcelona", "SUáREZ HAS MADE IT 4-1"), and the gate
    accepts either orientation, so naming the leader can never put the line
    and the board into an argument.

    Empty when the board is not to be trusted — ``bug_visible`` false is the
    score bug having been off screen longer than any replay lasts — or when
    :func:`effective_score` declines to guess.
    """
    if not state.bug_visible:
        return ""
    pair = effective_score(state, side, goal_in_state=goal_in_state)
    if pair is None:
        return ""
    home, away = pair
    if index is None:
        index = home + away
    forms: tuple[str, ...]
    if home == away:
        forms = (f"{in_words(home)}-{in_words(away)}.", f"{in_words(home)}-all.")
    else:
        if home > away:
            lead, trail, team = home, away, state.home
        else:
            lead, trail, team = away, home, state.away
        figures = f"{in_words(lead)}-{in_words(trail)}"
        forms = (f"{figures} to {team}.", f"{figures}, {team}.", f"{figures}.")
    said = forms[index % len(forms)]
    return said[:1].upper() + said[1:]


def restatement(state: MatchState, *, index: int | None = None) -> str:
    """"Twenty minutes gone, two-one to France." The club feed's five-minute line.

    Section 5.3 of the corpus study. A club channel restates the score for
    viewers joining late on something close to a literal five-minute timer,
    always as clock then score, out of a small closed vocabulary: ``N minutes
    gone``, ``N-N to <team>``, ``still N-N``, ``N minutes into the second
    half``. Two of those shapes are here.

    Empty when the state cannot support it: no clock, no trustworthy bug, or a
    replay on screen. Nothing in the line comes from a model, so nothing in it
    needs a gate — every word is the state's own.
    """
    if not state.bug_visible or state.in_replay or state.clock_s is None:
        return ""
    minutes = int(state.clock_s // 60)
    if minutes <= 0:
        return ""
    if index is None:
        index = minutes // 5
    clock = f"{in_words(minutes, zero='no')} minutes"
    home, away = state.home_score, state.away_score
    if home == away:
        figures = f"{in_words(home)}-{in_words(away)}"
    else:
        if home > away:
            lead, trail, team = home, away, state.home
        else:
            lead, trail, team = away, home, state.away
        figures = f"{in_words(lead)}-{in_words(trail)} to {team}"
    forms = (
        f"{clock} gone, {figures}.",
        f"Still {figures}, {clock} played.",
    )
    said = forms[index % len(forms)]
    return said[:1].upper() + said[1:]


#: Words a fragment can be made of and still be nothing but a scoreline. A
#: fragment whose every remaining word is in here is dropped rather than left
#: behind as debris: "Mbappé! Into the net! And that's." is worse than
#: "Mbappé! Into the net!"
_FILLER_TEXT = """
a an and as at but for in into is it it's its now of on or so still that
that's the then there they to up was we well with he she his her him make
makes made making take takes taking taken has have had all
"""
_FILLER = frozenset(_FILLER_TEXT.split())

#: What introduces a scoreline and has to go with it. "It's two-one" leaves
#: "It's" behind; "Mbappé makes it three-two" leaves "Mbappé makes it". Both
#: read as a sentence that lost its object, which is worse than either the
#: number or its absence.
_LEAD_IN = re.compile(
    r"(?:\b(?:and|now|so|well)\b[\s,]*)?"
    r"(?:\b(?:it|that|there|he|she|they)(?:'s|’s|\s+is|\s+are|\s+was)?\b\s*)?"
    r"(?:\b(?:has|have|had)\b\s*)?"
    r"(?:\b(?:makes?|made|making)\s+it\b\s*)?"
    r"(?:\b(?:takes?|took|taking)\s+(?:a|the)\b\s*)?"
    r"(?:\b(?:lead|leads|leading|winning|wins)\b\s*(?:by\b\s*)?)?$",
    re.IGNORECASE,
)

#: And what trails one. "Villa take a two-nil lead" leaves "lead" hanging, and
#: "two-one to France" leaves a preposition pointing at nothing. The team only
#: goes with the number where a preposition puts it there: everywhere else a
#: name after a number is a name, and this module does not take names out of
#: commentary.
_TRAIL = re.compile(
    r"^\s*,?\s*(?:"
    r"\b(?i:lead|leads|up|now|here|again)\b"
    r"|\b(?i:to|for)\s+(?:(?i:the)\s+)?[^\W\d_][\w'’-]*(?:\s+[^\W\d\sa-z_][\w'’-]*)?"
    r")"
)

_SENTENCE = re.compile(r"[^.!?]*[.!?]+|[^.!?]+$")
_SPACES = re.compile(r"\s{2,}")
_ORPHAN_PUNCT = re.compile(r"\s+([.,!?;:])")
#: Punctuation left at either end once a number has been taken out of the
#: middle: "— Two-two." strips to "—", and ", France." to ", France".
_EDGE_PUNCT = re.compile(r"^[\s,;:—–-]+|[\s,;:—–-]+$")


@dataclass(frozen=True)
class Stripped:
    """A line with the numbers taken out of it, and what came out."""

    text: str
    removed: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.removed)


#: A goal's "how" that belongs to exactly one kind of kick, and the word the
#: form's own description has to carry before that how is allowed to stand.
#: The Mbappé penalty at 82.5 was called "Mbappé! Over the wall! Two-one to
#: Argentina." — the phraser reached for the free kick's own worked example
#: on a penalty, because the shape of the last goal it wrote is a stronger
#: pull than the form in front of it. "Over the wall" is a free kick's how,
#: "the volley" and "the header" are a struck or headed goal's, and none of
#: the three is a penalty's — which is why this only ever fires beside a
#: penalty, never generally: a free kick that really was over the wall keeps
#: saying so.
_PENALTY_BORROWED_HOW: dict[str, str] = {
    "over the wall": "wall",
    "the volley": "volley",
    "the header": "header",
    "from the corner": "corner",
}

_HOW_PHRASE = re.compile(
    "|".join(re.escape(phrase) + r"!?" for phrase in _PENALTY_BORROWED_HOW), re.IGNORECASE
)


def strip_how_not_in_form(text: str, *, description: str, penalty: bool) -> Stripped:
    """Cut a goal's "how" when the caller's own form never said it.

    ``penalty`` is the caller's answer to "is this goal a penalty" — its own
    form said so, or the form just before it did within the ten seconds a
    penalty's kick and its goal sit apart. Nothing here runs otherwise: a
    goal the caller called a header keeps "the header" without this function
    ever looking at it, because a wrong how on a goal that was never a
    penalty is not the failure this backstop exists for.

    Deterministic and no model call, the same promise the rest of this
    module and :mod:`commentary.gate` make: a how the form never gave a
    reason for is cut outright rather than guessed at, whatever the model
    wrote.
    """
    if not penalty or not text:
        return Stripped(text)
    folded = description.lower()
    removed: list[str] = []

    def _cut(match: re.Match[str]) -> str:
        phrase = match.group().rstrip("!").lower()
        keyword = _PENALTY_BORROWED_HOW.get(phrase)
        if keyword is not None and keyword in folded:
            return match.group()
        removed.append(match.group().rstrip("!"))
        return ""

    cut = _HOW_PHRASE.sub(_cut, text)
    if not removed:
        return Stripped(text)
    return Stripped(_tidy(cut), tuple(removed))


def _spans(text: str, teams: Sequence[str] = ()) -> list[tuple[int, int]]:
    """Every run of characters in the line that asserts a score.

    ``teams`` are the two sides' names, and they are what makes the gate's
    ``<Side> level`` shape strippable rather than fatal: "France level!" needs
    no verb and matches nothing built out of one, so without the names it
    reaches the gate, and the gate refuses the whole line.
    """
    found = [(span.start, span.end) for span in score_spans(text)]
    found += ordinal_score_spans(text)
    found += level_claim_spans(text, teams)
    if not found:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(found):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _tidy(text: str) -> str:
    text = _SPACES.sub(" ", text)
    text = _ORPHAN_PUNCT.sub(r"\1", text)
    return _EDGE_PUNCT.sub("", text).strip()


def strip_score(text: str, teams: Sequence[str] = ()) -> Stripped:
    """Take every score claim out of the line, and say what was taken.

    Sentence by sentence, because that is how a commentator says a score: its
    own short utterance. A fragment that is nothing but the number goes
    entirely; a fragment that is a sentence with a number in it keeps the
    sentence and loses the number, along with whatever was leaning on it.

    This is the half of the change that stops a line being *dropped*. The gate
    refuses a whole line for one wrong figure in it, which is right — there is
    no trimming a number out of a sentence and leaving commentary behind, as
    the gate's own docstring says — but by the time a line reaches the gate it
    is too late to do anything but lose it. Here is early enough.

    ``teams`` are the two sides' names, for the score claim that is a side and
    an adjective: "Mbappé! The ball back to centre. France level!" went out on
    a follow-up beat at 2-1 and the gate refused all three fragments of it,
    because ``<Side> level`` is the one level claim the strip could not see.
    Every caller inside this system passes them; the default is empty so that
    a caller with no state in front of it behaves as it always did.
    """
    spans = _spans(text, teams)
    if not spans:
        return Stripped(text)
    removed = tuple(text[start:end].strip() for start, end in spans)
    kept: list[str] = []
    for fragment in _SENTENCE.finditer(text):
        base, stop = fragment.start(), fragment.end()
        inside = [(a, b) for a, b in spans if a >= base and b <= stop]
        if not inside:
            kept.append(fragment.group())
            continue
        out: list[str] = []
        cursor = base
        for start, end in inside:
            before = text[cursor:start]
            lead = _LEAD_IN.search(before)
            if lead is not None:
                before = before[: lead.start()]
            out.append(before)
            cursor = end
            trail = _TRAIL.match(text[cursor:stop])
            if trail is not None:
                cursor += trail.end()
        out.append(text[cursor:stop])
        cleaned = _tidy("".join(out))
        words = [word.strip(".,!?;:'’\"").lower() for word in cleaned.split()]
        if not [word for word in words if word and word not in _FILLER]:
            continue
        if is_a_hole(cleaned):
            # What is left is the tail of the claim rather than a line:
            # "Portugal's first of the night." lost its count and went out as
            # "Of the night." Only ever asked of a sentence something was cut
            # out of, which is why it can be as blunt as it is.
            continue
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        # The fragment has lost its opening words along with the number, so
        # whatever is in front now is the start of an utterance.
        kept.append(cleaned[:1].upper() + cleaned[1:] + " ")
    return Stripped(_tidy(" ".join(part.strip() for part in kept if part.strip())), removed)


@dataclass(frozen=True)
class Numbers:
    """One line, after code has taken its numbers out and put its own in."""

    line: str
    stripped: tuple[str, ...] = ()
    appended: str = ""
    #: A goal's how, taken out because the form never earned it — see
    #: :func:`strip_how_not_in_form`. Empty on every line this never runs on.
    how_removed: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.stripped or self.appended or self.how_removed)


def settle_numbers(
    text: str,
    *,
    state: MatchState,
    side: Side = Side.UNKNOWN,
    goal_in_state: bool = False,
    append: bool = False,
    index: int | None = None,
    description: str = "",
    penalty: bool = False,
) -> Numbers:
    """Strip whatever number the model wrote; append the one the state supports.

    The single call site for the rule, so that the runtime and the offline
    rephrase cannot drift apart on it. ``append`` is the caller's answer to
    "is this the goal-calling line?" — true on the first line of a goal and
    false on every line after it, which is what keeps the scoreline to once
    per goal.

    ``description`` and ``penalty`` are the same question asked of the how
    rather than the score: ``penalty`` is whether this goal's own form, or
    the form just before it, was a penalty, and ``description`` is that
    form's own words, checked by :func:`strip_how_not_in_form` before
    anything else runs. Left at their defaults, no line is touched — every
    caller of this before the how-backstop existed.
    """
    how = strip_how_not_in_form(text, description=description, penalty=penalty)
    stripped = strip_score(how.text, (state.home, state.away))
    line = stripped.text
    score = (
        say_score(state, side, goal_in_state=goal_in_state, index=index) if append else ""
    )
    if score:
        line = f"{line} {score}".strip() if line else score
    return Numbers(line=line, stripped=stripped.removed, appended=score, how_removed=how.removed)


@dataclass
class Restatements:
    """The timer behind :func:`restatement`, on the match clock.

    Periods are counted off the clock rather than off wall time or the video
    cursor, so a restatement lands on the round numbers a commentator actually
    says — "ten minutes gone", "half an hour gone" — and a stoppage does not
    push the whole schedule sideways.

    A period that comes due while the lead is speaking is not lost. It waits,
    and goes out at the first moment that is clear: filler that speaks over
    the game is worse than filler that is late, and section 5.3's own examples
    arrive late all over the match ("Back to Valjent, and then good back to
    the scoreline here, 2-0 to Barcelona").
    """

    every_s: float = 300.0
    #: The last period index that has been said. -1 is "nothing yet".
    said: int = -1
    #: A period that has come due and has not found a clear moment.
    pending: bool = False

    @property
    def enabled(self) -> bool:
        return self.every_s > 0

    def note(self, state: MatchState) -> None:
        """Read the clock. Arms the line when a new period has been reached."""
        if not self.enabled or state.clock_s is None:
            return
        period = int(state.clock_s // self.every_s)
        if self.said < 0:
            # The first clock this has ever seen is where the schedule
            # starts, not a period already missed. A match joined at 78
            # minutes owes fifteen restatements on the arithmetic alone, and
            # nobody says fifteen restatements in a row.
            self.said = period
            return
        if period > self.said:
            self.pending = True

    def take(self, state: MatchState) -> str:
        """The line, if one is owed and the state can support it. Then it is spent."""
        if not (self.enabled and self.pending):
            return ""
        said = restatement(state)
        if not said or state.clock_s is None:
            return ""
        self.said = int(state.clock_s // self.every_s)
        self.pending = False
        return said
