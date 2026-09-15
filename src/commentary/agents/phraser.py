"""The speaking voice: one cheap call that turns a form into six words.

The caller's perception is good and its prose is not. Across sixty-three runs
on real footage it has never put a wrong player name on air, and it has also
never written a line under seven words — with fragment examples in its prompt
and a cadence that pays for short lines. Asked to look and to talk at once it
does the looking well and writes a caption:

    Ronaldo walks back into position, hands on hips, waiting for Portugal to
    work something forward in these closing minutes.

Real commentary is five words. So this module is the second half of that job,
split off: the caller keeps the pictures, the form and the rules about what
may be claimed, and the phraser gets the finished form and writes the line.
It never sees a frame, which is deliberate — a phraser with a picture starts
describing the picture, and that is the failure being fixed.

What it costs. One Haiku call per spoken line, with the register's two
hundred real utterances in a cached system prefix and a few hundred bytes of
form in the body. A tenth of a cent, give or take, against the caller's three
cents. ``PHRASER_MODEL=off`` removes the stage entirely and the runtime is
what it was before this file existed.

What it is not allowed to do. Everything it writes goes through the same fact
gate the caller's line went through — same roster check, same scoreline
arithmetic, same goal rule — so a name it invents is a name that never
reaches the speaker. The prompt tells it this; the gate is what enforces it.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from commentary.agents.caller import clean_line, trim_words
from commentary.agents.colour import mentions, says_a_number
from commentary.config import PHRASER_MODEL, DeadBallConfig, PhraserConfig, SilenceConfig
from commentary.gate import possessive_swap
from commentary.ledger import Fact as LedgerFact
from commentary.llm.base import Block, LLMBackend, LLMError, Parsed, Usage, text_block
from commentary.prompts.phraser import (
    BUILD_UP_FORMS,
    is_long_line,
    phraser_blocks,
    phraser_system,
    strip_replay_marker,
)
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    Note,
    PhrasedLine,
    Scene,
)

#: The beat that is one number about the scorer — see ``GOAL_BEATS[3]`` in
#: ``prompts/phraser.py``. The only beat this module ever needs to
#: distinguish, because it is the only one where a name is mandatory and a
#: number is expected in the same line.
SCORER_BEAT = 3

#: The beats that are *not* the call, and so may not be shouted on a name.
#: Beat 1 is the goal call and "<Scorer>!" is exactly what it should be; every
#: beat after it is a line about a goal the listener has already been told
#: about. On ``runs/rephrased/r1-replay/mbappe`` beats 2 and 3 both opened
#: "Mbappé!"; beat 4 was left out of this set because it is written in the
#: past tense and had never done it, and on ``runs/rephrased/r5b`` it did it
#: twice — "Mbappé! Over the keeper!" and "Mbappé! Off the ground!".
UNSHOUTED_BEATS = frozenset({2, 3, 4})

#: The value of ``PHRASER_MODEL`` that means "do not run this stage".
OFF = "off"

#: How many of the most recently spoken lines an opener is checked against.
#: Matches the prompt's own "last two" rule with slack: the prompt asks the
#: model not to open on either of the last two, and real commentary repeats
#: an opener about one line in eight looking back further than that, so
#: checking five catches the failure (35-48% repeats) without punishing the
#: normal, occasional one-in-eight echo from further back.
_OPENER_LOOKBACK = 5

#: And the same window for the last word. The tail is the same tell as the
#: opener and it is the one this voice actually commits: "Through midfield
#: now." / "Wide on the right now." / "Into the corner now." / "Striding out
#: now, France in no hurry to move it on." is four lines in five ending on
#: one word and no two of them opening on one, so the opener check waved
#: every one of them through.
_CLOSER_LOOKBACK = 5

#: How far back the "now" tail is looked for. Three: the four lines it went
#: out on were consecutive, and a word a commentator used two minutes ago is
#: not a tic.
_NOW_LOOKBACK = 3

_LEADING_TRAILING_PUNCT = re.compile(r"^[^\w]+|[^\w]+$")
_POSSESSIVE = re.compile(r"['’]s$", re.IGNORECASE)

#: A line that opens by shouting a name: one or two capitalised words and an
#: exclamation mark, which is the goal call's own shape. Whether the words
#: are actually a *name* is decided against the form and the scorer, not by
#: this pattern — "Buried!" and "Save!" open the same way and are not it.
_OPENING_SHOUT = re.compile(r"^\s*([A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)?)\s*!")


def _opening_word(text: str) -> str:
    """The line's first word, as it would be judged for a repeat: punctuation
    and a trailing possessive stripped, case-folded.
    """
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[0])
    core = _POSSESSIVE.sub("", core)
    return core.casefold()


def _closing_word(text: str) -> str:
    """The line's last word, judged the same way the first one is."""
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[-1])
    core = _POSSESSIVE.sub("", core)
    return core.casefold()


def _display_word(text: str) -> str:
    """The same word, kept in its written case, for naming in the retry note."""
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[0])
    return _POSSESSIVE.sub("", core)


def _display_closer(text: str) -> str:
    """The closing word, kept in its written case, for naming in the note."""
    tokens = text.strip().split()
    if not tokens:
        return ""
    core = _LEADING_TRAILING_PUNCT.sub("", tokens[-1])
    return _POSSESSIVE.sub("", core)


def _is_bare_name(text: str) -> bool:
    """One word and nothing else — a legitimate repeat, not a repeated frame."""
    return len(text.strip().split()) == 1


def _with_note(blocks: Sequence[Block], note: str) -> list[Block]:
    """The same call's body, with a note appended to its last text block."""
    new_blocks = [dict(block) for block in blocks]
    for block in reversed(new_blocks):
        if block.get("type") == "text":
            block["text"] = f"{block['text']}\n\n{note}"
            return new_blocks
    new_blocks.append(text_block(note))
    return new_blocks


def _register_retry_note(opener: str, closer: str) -> str:
    """Name the repeat — the first word, the last word, or both — and re-ask.

    One note and one re-ask for the pair, rather than two calls for one line.
    Both faults are the same fault measured at opposite ends of the sentence,
    the fix for either is to write a different sentence, and a model asked
    twice about one line costs twice and answers worse the second time.
    """
    faults = []
    if opener:
        faults.append(f'it OPENED on "{opener}", which one of the last five lines already used')
    if closer:
        faults.append(f'it ENDED on "{closer}", which one of the last five lines already ended on')
    return (
        "THAT LINE REPEATS ONE OF THE LAST FIVE: "
        + ", and ".join(faults)
        + ".\nSay it differently this time — a different subject, a different verb, or the\n"
        "detail instead — or return an empty line. Do not simply move the word."
    )


def _shout_retry_note(name: str, beat: int | None) -> str:
    """Say which shape was written, why it is beat 1's and not this one's."""
    this = f"Beat {beat}" if beat is not None else "A line over a replay"
    return (
        f'THAT OPENED ON "{name}!", WHICH IS THE GOAL CALL\'S OWN SHAPE. The call has '
        "already gone out with the score on the end of it, and a second line in that "
        f"shape is a second goal to whoever is listening. {this} is not a shout.\n"
        "Write it again without the name-and-exclamation-mark at the front: the name "
        "may be anywhere else in the line."
    )


#: Every way a number reaches a line, and what each one is worth. Digits are
#: read as themselves; the words and the ordinals are the ones a commentator
#: actually says out loud.
_FIGURES: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "hundred": 100,
}


def figures_in(text: str) -> set[int]:
    """Every number this line says, in figures or in words.

    "a second title" and "his second goal" are the same number here, which is
    the point: the check this feeds asks whether the figure came off the
    clause the line was handed, and a model that reuses the ordinal from a
    clause about something else has still written a number nobody looked up.
    """
    found: set[int] = set()
    for token in _WORD.findall(text):
        word = token.casefold()
        if word.isdigit():
            found.add(int(word))
        elif word in _FIGURES:
            found.add(_FIGURES[word])
    return found


def _figure_retry_note(extra: set[int], clauses: Sequence[Note]) -> str:
    """Say which number is not in the clause, and print the clause again."""
    said = ", ".join(str(number) for number in sorted(extra))
    rows = "\n".join(f"  - {note.text.strip()}" for note in clauses) or "  (none)"
    return (
        f"THAT LINE SAYS {said}, AND THAT FIGURE IS IN NONE OF THE CLAUSES YOU WERE "
        "HANDED. A number this stage writes for itself is a number nobody looked up, "
        "and it is the one mistake here that reaches air sounding right.\n"
        f"The clauses, in full:\n{rows}\n"
        "Say one of them, with its own figure, or return an empty line."
    )


def _swap_retry_note(swapped: str) -> str:
    """Name the thing hung on the wrong man, and say whose it was."""
    return (
        f"THAT LINE PUTS {swapped.upper()}. The person watching wrote down whose it "
        "was and your line has given it to somebody else — which is not a wording "
        "mistake, it is a different account of what happened.\n"
        "Write it again with the description's own owners, or return an empty line."
    )


#: The tail this voice tacks onto a line that was finished without it. Caught
#: where it ends a clause as well as where it ends the line, because that is
#: how it actually went out: "Through midfield now, halfway line reached." /
#: "Argentina through the middle now, numbers forward at pace." / "White
#: shirts streaming towards goal now." — four lines of twenty-one on the
#: offside clip, and the closer check sees only the last word of the three.
_NOW_TAIL = re.compile(r"\s+now(?=\s*(?:[,.;:!?]|$))", re.IGNORECASE)


def has_the_now_tail(text: str) -> bool:
    """Does this line end a clause on "now"?"""
    return _NOW_TAIL.search(text) is not None


def strip_now_tail(text: str) -> str:
    """Take the tail off and close the punctuation up behind it.

    The line is where the ball is and the ball is always now: the word adds
    nothing to any of the four it went out on. Only ever called when a line
    already aired has the same tail, because one of them is a commentator and
    four in a row is a jingle.
    """
    out = _NOW_TAIL.sub("", text)
    return re.sub(r"\s+", " ", out).strip()


def _thin_call_note(described: str) -> str:
    """Ask for the how the form is holding, and quote the form back."""
    return (
        "THAT IS THE NAME AND NOTHING ELSE, AND THE FORM IS HOLDING THE REST OF IT. "
        "A goal is the name, then how, and the how is whatever the person watching "
        "wrote down — said back in three words or fewer.\n"
        f"What they wrote: {described.strip()}\n"
        "Write the name and the how. The score goes on after you, in code."
    )


def is_a_thin_call(text: str, described: str) -> bool:
    """Is this goal call the scorer's name with the how thrown away?

    Round four's first call was "Mbappé!" where round three's was "Mbappé!
    The penalty, buried!" — same form, same description, the how gone. The
    shout is right and it is half a line: the corpus's goal call is the name,
    then how, then the score, and the score is code's.

    Only ever asked of a goal-calling line whose form carried something to
    keep. A description as short as the line is a description with no how in
    it, and then the name alone is the right answer.
    """
    said = [word for word in _WORD.findall(text)]
    return len(said) <= 2 and len(_WORD.findall(described)) >= 6


#: Words that may drop out of a protected phrase without making it a
#: different phrase. "Off the ground" and "off ground" are one thing said
#: twice; the corpus's own repeats never turn on an article.
_DROPPABLE = frozenset({"the", "a", "an", "of", "in", "on", "at", "it", "and"})


def protected_phrases(call: str, detail: str = "") -> list[str]:
    """The phrases the call spent, which no line after it may spend again.

    Two sources and both are the same thing said twice. The form's own
    ``detail`` is the one concrete thing the eyes picked out, and the goal
    call is built to carry it — so once the call has said it, it is on air.
    And the call itself, minus the name it shouts and the score the broadcast
    wrote on the end: what is left of "Mbappé! Off the ground! Two-two." is
    "off the ground", and beat 4 said it again ninety-four seconds in.

    Shorter than :data:`REPEAT_RUN` on purpose. "Off the ground" is three
    words and two of them are ordinary, so the shared-run check waves it
    through; as the call's own detail it is the most memorable phrase in the
    sequence and the one a listener notices twice.
    """
    found: list[str] = []
    if detail.strip():
        found.append(detail.strip())
    for fragment in re.split(r"[.!?]", call):
        words = _WORD.findall(fragment)
        if len(words) < 2 or _score_words(words):
            continue
        found.append(" ".join(words))
    return found


def _score_words(words: Sequence[str]) -> bool:
    """Is this fragment the score the broadcast appended, rather than words?

    Any figure in it is enough. A goal call may not write a number at all —
    the broadcast puts the score on the end in code — so a fragment carrying
    one is the score, and the score is not a detail anybody spent.
    """
    # Split on the hyphen as well: a scoreline arrives as one token,
    # "Two-one", and neither half of it is a word anybody spent.
    return any(
        part.casefold() in _FIGURES or part.isdigit()
        for word in words
        for part in word.split("-")
    )


def repeats_the_detail(text: str, phrases: Sequence[str]) -> str:
    """The protected phrase this line says again, or ``""``.

    Verbatim, or with one droppable word left out of either side: a line that
    says the call's detail back is saying the call back, whatever it does
    with the articles.
    """
    said = [word.casefold() for word in _WORD.findall(text)]
    for phrase in phrases:
        wanted = [word.casefold() for word in _WORD.findall(phrase)]
        if len(wanted) < 2:
            continue
        bare = [word for word in said if word not in _DROPPABLE]
        for variant in _variants(wanted):
            if _contains(said, variant) or _contains(bare, variant):
                return phrase
    return ""


def _variants(words: list[str]) -> list[list[str]]:
    """The phrase, and the phrase with one droppable word taken out."""
    out = [words]
    for index, word in enumerate(words):
        if word in _DROPPABLE and len(words) > 2:
            out.append(words[:index] + words[index + 1 :])
    return out


def _contains(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[index : index + len(needle)] == needle
        for index in range(len(haystack) - len(needle) + 1)
    )


#: Phrases that describe a group of people. One man is not "numbers", and a
#: line that gives him a side's verb is a line about the wrong subject: the
#: caller wrote "navy shirts arriving in numbers — and Mbappé is the danger"
#: and what went out was "Mbappé, arriving in numbers."
PLURAL_ONLY = (
    "in numbers",
    "as a unit",
    "pour forward",
    "pouring forward",
    "swarm",
    "swarming",
    "stream forward",
    "streaming forward",
    "flood forward",
    "flooding forward",
)


def plural_on_one_man(text: str, *, names: Sequence[str], sides: Sequence[str]) -> str:
    """The plural phrase this line hangs on a single player, or ``""``.

    A side may do all of these and a man may not. The test for "a single
    player" is that the line names one man on the sheets and no side at all:
    with a side in it, the plural has something to belong to.
    """
    lowered = text.casefold()
    found = next((phrase for phrase in PLURAL_ONLY if phrase in lowered), "")
    if not found:
        return ""
    if any(side.strip() and side.casefold() in lowered for side in sides):
        return ""
    # Distinct men, not distinct spellings: the same player arrives here from
    # the form and from the team sheet, and counting him twice would let the
    # rule pass every line it exists for.
    named = {
        name.strip().rsplit(" ", 1)[-1].casefold()
        for name in names
        if name.strip() and mentions(text, name)
    }
    return found if len(named) == 1 else ""


def _plural_retry_note(phrase: str, name: str) -> str:
    """Say which phrase belongs to eleven people, and who it was given to."""
    return (
        f'THAT GIVES "{phrase}" TO ONE MAN. {name} is one player; that phrase is about '
        "a group of them, and the description said so — the side is doing it and he is "
        "the one on the end of it.\n"
        "Write it about the side, or write what he is doing, or return an empty line."
    )


def _repeat_retry_note(phrase: str) -> str:
    """Name the phrase that is already on air, and ask for the other thing."""
    return (
        f'THAT SAYS "{phrase}" AGAIN, AND IT HAS ALREADY GONE OUT ABOUT THIS GOAL. '
        "The listener has those words; saying them a second time tells them nothing "
        "they do not have.\n"
        "Write the part of it nobody has heard yet — the run-up, the keeper, where "
        "the ball came from, the man who made it — or return an empty line."
    )


def opening_shout(text: str, names: Sequence[str]) -> str | None:
    """The name this line opens by shouting, or ``None``.

    A name, not a word: "Save!" and "Buried!" open the same way and neither is
    the fault. So the capitalised opener is matched against the people this
    call actually knows about — the scorer, the names on the form, the two
    team sheets — by surname, the way every other name check in this system
    works, because the form spells a man "Kylian Mbappé" and the line says
    "Mbappé".
    """
    match = _OPENING_SHOUT.match(text)
    if match is None:
        return None
    opener = match.group(1).strip()
    wanted = {opener.casefold(), opener.rsplit(" ", 1)[-1].casefold()}
    for name in names:
        clean = (name or "").strip()
        if not clean:
            continue
        if clean.casefold() in wanted or clean.rsplit(" ", 1)[-1].casefold() in wanted:
            return opener
    return None


def unshout(text: str, *, keep_name: bool = False, names: Sequence[str] = ()) -> str:
    """Take a "Name!" off the front of a follow-up beat, leaving a line behind.

    Two shapes, and which one is used is decided by the word after the
    exclamation mark. A lower-case one is a clause the name is the subject or
    the addressee of, and the comma is what a commentator would have written:
    "Mbappé! and away he goes" becomes "Mbappé, and away he goes". An
    upper-case one is a sentence of its own with a shout stuck in front of
    it, so the shout comes off: "Mbappé! The keeper sent the wrong way."
    becomes "The keeper sent the wrong way."

    ``keep_name`` is the third shape and it is the free-kick trace's: the goal
    was called "Over the wall, into the top corner!" and named nobody, so the
    listener has not been told whose goal it is and beat 2 is where the name
    goes. Dropping the shout there would drop the only naming of the scorer
    in the sequence, so the name is kept and the shout is not —
    "Ronaldo! Over the wall" becomes "Ronaldo, over the wall". ``names`` stops
    the lower-casing where the next word is somebody else's name: "Ronaldo!
    De Gea beaten." keeps its capital.

    A line that is *only* the shout is returned untouched. There is nothing
    under it to promote, and an empty line here would drop the beat — which
    is the one thing this must not do, because the beat is what the thirty
    seconds after a goal are made of.
    """
    match = _OPENING_SHOUT.match(text)
    if match is None:
        return text
    name = match.group(1).strip()
    rest = text[match.end() :].lstrip()
    if not rest:
        return text
    if rest[0].islower():
        return f"{name}, {rest}"
    if keep_name:
        head = rest.split(maxsplit=1)[0].strip(".,!?;:'\u2019\"")
        if not _is_a_name(head, names):
            rest = rest[0].lower() + rest[1:]
        return f"{name}, {rest}"
    return rest


def _lead_with(text: str, scorer: str, names: Sequence[str]) -> str:
    """Put the scorer's name on the front of a line that does not carry it.

    The surname, a comma, and the line lower-cased behind it, which is the
    shape the corpus uses and the one beat 2 is asked for when the call named
    nobody. A line that already opens on somebody else's name keeps its
    capital.
    """
    said = text.strip()
    if not said:
        return said
    surname = scorer.rsplit(" ", 1)[-1].strip() or scorer.strip()
    head = said.split(maxsplit=1)[0].strip(".,!?;:'\u2019\"")
    if not _is_a_name(head, names):
        said = said[0].lower() + said[1:]
    return f"{surname}, {said}"


def _ends_on_a_stop(text: str) -> str:
    """The line with a full stop on it, if it had no end of its own."""
    said = text.strip()
    if not said or said.endswith((".", "!", "?", "…")):
        return said
    return f"{said}."


def _is_a_name(word: str, names: Sequence[str]) -> bool:
    """Is this word somebody on the sheet, by any part of their name?"""
    folded = word.casefold()
    return any(
        folded == part.casefold() for name in names for part in name.split() if part
    )


#: A run of words this long, shared with something already said about the
#: same goal, is the same thing said twice. Three: two is ordinary English
#: ("and the", "off the") and four would let "over the wall, into the top
#: corner" through on a single changed word. Kept beside the prompt's own
#: statement of the rule in :data:`commentary.prompts.phraser.REPEAT_RUN`.
REPEAT_RUN = 3

_WORD = re.compile(r"[\w'\u2019-]+")

#: Words a run of three may be made entirely of without meaning anything.
#: "and he has" repeated is English; "into the top" repeated is the same
#: piece of information twice.
_FUNCTION_TEXT = """
a an and are as at be been but by for from had has have he her here him his
i if in into is it its me my no not now of off on one or our out she so than
that the their them then there they this to too up us was we were what when
which who will with you your
"""
_FUNCTION_WORDS = frozenset(_FUNCTION_TEXT.split())


def _tokens(text: str) -> list[str]:
    return [match.group().casefold() for match in _WORD.finditer(text)]


def shared_run(text: str, said: Sequence[str], *, run: int = REPEAT_RUN) -> str:
    """The run of words this line repeats from something already said, or ``""``.

    The fault it is for is on ``runs/rephrased/r2-colour/freekick``: the free
    kick is called "Over the wall, into the top corner!" at 21.2 s, beat 2
    says "Ronaldo! Over the wall, into the top corner!" at 25.2, and the
    rebuild says "Ronaldo took his steps back and whipped it over the wall,
    into the top corner." at 32.5. One piece of information, three times,
    eleven seconds. Every rule the prompt has about not repeating itself is
    about the *last lines spoken*; inside a goal window every beat is handed
    the same description of the same move and reaches for the same phrase in
    it.

    A run made only of function words is not a repeat — "and he has" is how
    English works — so at least one word in the run has to carry something.
    """
    mine = _tokens(text)
    if len(mine) < run:
        return ""
    before: set[tuple[str, ...]] = set()
    for old in said:
        tokens = _tokens(old)
        before.update(
            tuple(tokens[index : index + run]) for index in range(len(tokens) - run + 1)
        )
    for index in range(len(mine) - run + 1):
        gram = tuple(mine[index : index + run])
        if gram in before and not all(word in _FUNCTION_WORDS for word in gram):
            return " ".join(gram)
    return ""


def roster_names(pack: KnowledgePack | None) -> list[str]:
    """Every name on either team sheet, for :func:`opening_shout`.

    A goal call's sightings sometimes carry a shirt number and no name at all
    — on the Mbappé penalty every one of them did — so the man the follow-up
    beats are about can be a name that is on the roster and nowhere on the
    form. The shout check needs to recognise him there.
    """
    if pack is None:
        return []
    return [
        player.name
        for sheet in (pack.home, pack.away)
        for player in sheet.squad
        if player.name.strip()
    ]


#: A dash holding two halves of a line together. The phraser used it to weld
#: a researched fact onto a picture — "Scaloni, arms flung wide, roaring at
#: his players — and Argentina chasing a first World Cup since 1986." — which
#: the judge marked on ``runs/rephrased/r4-shape/mbappe`` as a stat bolted on
#: rather than said. The corpus attaches a fact with a relative clause on the
#: man ("Kenate, who's missed eight games with a knee injury") or gives it a
#: sentence of its own.
_WELDING_DASH = re.compile(r"\s*[\u2014\u2013]\s*|\s+-\s+")


def unweld(text: str) -> str:
    """Take the dash out and put the join a commentator would write in its place.

    A lower-case word after it is a continuation, and that is a comma. A
    capital is a new sentence, and that is a full stop. Either way the line
    comes out as two things said rather than one thing with a fact bolted on
    the end of it.
    """
    out = text
    while True:
        match = _WELDING_DASH.search(out)
        if match is None:
            return re.sub(r"\s+", " ", out).strip()
        rest = out[match.end() :]
        if not rest:
            return out[: match.start()].strip()
        before = out[: match.start()].rstrip(" ,;:")
        join = ", " if rest[0].islower() else ". "
        out = f"{before}{join}{rest}"


def names_nobody(line: CallerLine, *, on_the_ball: str | None = None) -> bool:
    """Is there a person on this form at all?

    The event is one of the words the caller has for the ball going forward
    with nothing happening to it, no sighting bound to a roster name, and no
    name carried on the ball from a few seconds ago. What it does *not* ask
    about is the detail — see :func:`nameless_build_up`, which is this plus
    an empty detail, and :meth:`Phraser.passes_over`, which wants both.
    """
    if line.scene is Scene.REPLAY:
        return False
    if line.event not in BUILD_UP_FORMS and line.event is not Event.NONE:
        return False
    if (on_the_ball or "").strip():
        return False
    return not any((sighting.name or "").strip() for sighting in line.sightings)


def is_a_bare_name(text: str, names: Sequence[str]) -> bool:
    """Is this line one name and nothing else, shouted or not?

    A whole line for the lead in build-up — the corpus says a bare surname is
    one line in twenty-five — and nothing at all for a goal follow-up beat.
    On ``runs/rephrased/r5a/mbappe`` beat 4 went out as "Mbappé!" on its own
    twelve seconds after the goal, because the repeat check had taken the
    rest of the line off and the shout rewrite leaves a line that is only a
    shout alone. The listener is told the man's name for the third time and
    nothing else.
    """
    words = _WORD.findall(text)
    if not words or len(words) > 2:
        return False
    return _known_name(" ".join(words), names) or _known_name(words[-1], names)


def _known_name(word: str, names: Sequence[str]) -> bool:
    folded = word.casefold()
    for name in names:
        clean = (name or "").strip()
        if clean and folded in (clean.casefold(), clean.rsplit(" ", 1)[-1].casefold()):
            return True
    return False


def nameless_build_up(line: CallerLine, *, on_the_ball: str | None = None) -> bool:
    """Is this form the ball moving between nobody in particular?

    Three conditions, all of them the caller's own record of what it saw: the
    event is one of the words it has for the ball going forward with nothing
    happening to it, no sighting bound to a roster name and no name carried
    on the ball from a few seconds ago, and no ``detail`` — the one concrete
    thing it is asked to pick out of the picture.

    ``docs/research/real-commentary-corpus.md`` section 3.1a: 24% of carries
    and passes in build-up have nothing said within ±3 s of them, and even
    counting generously only 38% of carries have their player named. This is
    the shape of the ones that are not said, and
    :meth:`Phraser.passes_over` is where it is acted on.
    """
    return names_nobody(line, on_the_ball=on_the_ball) and not (line.detail or "").strip()


def _name_retry_note(scorer: str) -> str:
    """Two lines: say the number reached nobody, then say who it is about."""
    return (
        "THAT HAD A NUMBER IN IT AND NOBODY'S NAME ON IT — a fact about nobody, which the "
        f"gate refuses. The line must name {scorer}.\n"
        "Keep the number and the fact; add his name, in the same number of words."
    )


@dataclass(frozen=True)
class Said:
    """One line that went out, and the three things about it worth keeping.

    The words are what the prompt shows back. The kind is what lets the model
    tell a second line about the same carry from a new moment. ``nameless``
    and ``ts`` are what :meth:`Phraser.passes_over` reads, and neither can be
    recovered from the words afterwards.
    """

    text: str
    event: Event | None = None
    nameless: bool = False
    ts: float | None = None


def phraser_enabled(model: str = PHRASER_MODEL) -> bool:
    """Is the phrasing stage switched on at all?"""
    return model.strip().lower() not in (OFF, "", "none")


class Phraser:
    """A form in, a line a commentator would say out.

    Holds its own memory of what has been said, like the caller does and for
    the same reason: the prompt shows the last few lines back so the voice
    does not repeat itself. It is a separate memory from the caller's because
    the two are looking at different text — the caller sees its own
    descriptions, the phraser sees what actually went out.
    """

    def __init__(
        self,
        backend: LLMBackend,
        config: PhraserConfig | None = None,
        *,
        model: str | None = None,
        home: str = "Home",
        away: str = "Away",
        silence: SilenceConfig | None = None,
        dead_ball: DeadBallConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or PhraserConfig()
        #: When a form is passed over without a model call at all. See
        #: :meth:`passes_over`.
        self.silence = silence or SilenceConfig()
        #: The restart slot: what the body block asks for and how far the
        #: word cap is lifted for it. See
        #: :func:`commentary.prompts.phraser.is_long_line`.
        self.dead_ball = dead_ball or DeadBallConfig()
        self.model = model if model is not None else self.config.model
        self.home = home
        self.away = away
        #: Built once. Identical bytes every call is what makes two hundred
        #: real utterances affordable to send on each line.
        self.system = phraser_system(
            max_words=self.config.max_words,
            examples_per_kind=self.config.examples_per_kind,
        )
        #: What was said, and what each line was about. The kind is carried
        #: beside the words because the two rules that need it — do not open
        #: the same way twice, and say nothing when the last line was this
        #: same moment about this same man — cannot be checked from the text
        #: alone. A model shown "Upamecano works it forward." has no way to
        #: know whether that was a carry or a tackle.
        #:
        #: The third field is whether the line went out over a form with
        #: nobody on it — see :func:`nameless_build_up`. It cannot be read
        #: back off the words either: "Through midfield now." names nobody
        #: and neither does "Away.", and only one of the two came off a form
        #: with a bound name.
        self._recent: deque[Said] = deque(maxlen=max(1, self.config.recent_lines))
        #: Why the last call produced nothing, in words, for the error row.
        self.last_reason = ""
        #: Did the last call choose to say nothing? A model that returns an
        #: empty line has decided this moment is one of the ones real
        #: commentary passes over — 43% of goal kicks, 37% of throw-in
        #: deliveries, 33% of free-kick deliveries, 31% of kickoffs and a
        #: quarter of all build-up touches
        #: (``docs/research/real-commentary-corpus.md`` section 3). That is a
        #: different event from a call that failed or a line that was
        #: nothing but a label, and the two used to be one flag: both came
        #: back as an empty line and both fell back to the caller's words,
        #: so the phraser could never choose silence. Read it beside
        #: :meth:`phrase` returning ``None``, which is still a failure.
        self.chose_silence = False
        #: What the last call cost, so a rephrase can price itself per line.
        self.last_usage = Usage()

    @property
    def enabled(self) -> bool:
        return phraser_enabled(self.model)

    @property
    def recent(self) -> list[str]:
        """What has actually been said, oldest first, each tagged with its kind.

        The tag is what the prompt reads as "a pass, 2 lines ago": without it
        the model cannot tell a second line about the same carry from a
        genuinely new moment, and the corpus says the second one is silence
        a quarter of the time.
        """
        # The kind goes after the words, not in front of them. In front, the
        # model read the tag as the opener and stopped obeying the rule about
        # not opening two lines the same way: two adjacent lines both
        # starting "Argentina" went out in the round that tried it.
        return [
            f"{said.text}   ({said.event.value if said.event else 'no kind'})"
            for said in self._recent
        ]

    @property
    def said_lines(self) -> list[str]:
        """What has actually gone out, oldest first, without the kind tags.

        :attr:`recent` is what the prompt is shown and carries "(a carry)" on
        the end of every line. This is the same lines as words, for the checks
        that compare one line against another.
        """
        return [said.text for said in self._recent]

    def accept(
        self,
        line: str,
        event: Event | None = None,
        *,
        ts: float | None = None,
        nameless: bool = False,
    ) -> None:
        """Record a line as spoken. Only call this when it really is going out.

        ``ts`` and ``nameless`` are what :meth:`passes_over` reads back: when
        the last thing said was a line with nobody in it, and how long ago.
        Both default to the answer every caller of this gave before the
        silence rule existed — no time, and named — so an older caller keeps
        asking the model about every form, which is what it used to do.
        """
        text = line.strip()
        if text:
            self._recent.append(Said(text=text, event=event, nameless=nameless, ts=ts))

    def passes_over(
        self,
        line: CallerLine,
        *,
        ts: float,
        on_the_ball: str | None = None,
    ) -> str | None:
        """The reason to say nothing about this form, or ``None`` to ask the model.

        The one decision this stage makes without spending a call. The corpus
        (``docs/research/real-commentary-corpus.md`` section 3.1a) is that
        24% of carries and passes in build-up pass with nothing said and only
        38% of carries name anybody; the prompt has said so for four rounds
        and the measured result was **one chosen silence in 35 calls**, with
        "Through midfield now. / Wide on the right now. / Into the corner
        now." going out instead. A model asked to choose silence about a
        picture somebody has just handed it will write something.

        So the narrowest version of the corpus's own rule is decided here:
        this form has nobody on it and nothing on it
        (:func:`nameless_build_up`), and the last thing that went out was the
        same, recently. Never the first of a pair — the first nameless
        build-up line is ordinary commentary — and never when a name is bound,
        a name is carried, or the eyes picked out a detail.
        """
        if not self.silence.enabled or not names_nobody(line, on_the_ball=on_the_ball):
            return None
        if self._after_a_nameless_line(line, ts):
            return "silence: nameless build-up after nameless build-up"
        last = self._recent[-1] if self._recent else None
        gap = float("inf") if last is None or last.ts is None else ts - last.ts
        if gap < self.silence.nameless_gap_s or not (line.detail or "").strip():
            # A line with nobody on it has to earn its place twice over: the
            # gap has to have opened, and the eyes have to have picked out
            # something a listener could not guess. Real commentary passes
            # over a quarter of all touches and names most of the rest.
            return (
                "silence: nobody on the form and "
                + (
                    f"only {gap:.0f}s since the last line"
                    if gap < self.silence.nameless_gap_s
                    else "nothing the eyes picked out"
                )
            )
        return None

    def _after_a_nameless_line(self, line: CallerLine, ts: float) -> bool:
        """Is this the second nameless build-up form in a row, recently?

        The narrower of the two rules and the older one: the pair the corpus
        is most emphatic about, and the one whose reason says so in the
        trace. Asked first because "after another one like it" tells whoever
        reads the row more than "nobody on the form" does.
        """
        if not nameless_build_up(line):
            return False
        recent = list(self._recent)[-max(1, self.silence.after_nameless) :]
        if len(recent) < self.silence.after_nameless:
            return False
        if not all(said.nameless for said in recent):
            return False
        # Older than the window, or from a caller that does not stamp what it
        # says. A quiet passage answered with more quiet is how a system goes
        # mute, and the corpus's longest silences are bounded.
        return all(
            said.ts is not None and 0.0 <= ts - said.ts <= self.silence.within_s
            for said in recent
        )

    async def phrase(
        self,
        line: CallerLine,
        state_summary: str,
        *,
        on_the_ball: str | None = None,
        notes: Sequence[Note] = (),
        callbacks: Sequence[bool] = (),
        ledger: Sequence[LedgerFact] = (),
        followup: str = "",
        goal_beat: int | None = None,
        scorer: str | None = None,
        roster: Sequence[str] = (),
        already_said: Sequence[str] = (),
        replay_first: bool = True,
    ) -> PhrasedLine | None:
        """Rewrite one caller line, or return ``None`` if the call failed.

        ``None`` is a failure and the caller's own words go out instead: a
        line the gate was about to pass is worth more spoken badly than
        lost. An empty ``line`` on the returned object is the other thing —
        the phraser deciding this is a moment to say nothing — and
        :attr:`chose_silence` tells the two apart.

        ``notes`` are the pack's clauses about the people on this form, and
        they are the only outside information this stage has ever been given.
        The prompt decides whether to show them — a goal is no moment for a
        statistic — and the fact gate checks whatever comes back against the
        same notes, so a figure the model adjusts on its way out is a line
        that never reaches the speaker.

        ``ledger`` is the same offer made out of this match's own counts —
        a fourth corner, a second foul — written by
        :class:`commentary.ledger.Ledger` rather than researched, and checked
        afterwards by the gate's ``ledger_claim`` exactly as a note is checked
        by ``note_claim``.

        ``callbacks`` marks which of those clauses have already been said
        once in this match, one flag a note, so the prompt can ask for a new
        form of an old fact rather than the same words again. It is
        :class:`commentary.threads.Threads` that knows, and empty is the
        answer everywhere nothing has been said twice yet.

        ``followup`` is the block :class:`commentary.goalfollow.GoalFollowup`
        writes in the thirty seconds after a goal, naming which of the corpus's
        beats is due — the moment again, the scorer's tally, the move rebuilt
        in past tense. Empty everywhere else, which is most of a match.

        ``goal_beat`` and ``scorer`` are what :class:`~commentary.goalfollow.
        GoalFollowup` already knows and ``followup`` only says in prose: which
        beat this call is under and who scored. Beat 3 is one number about the
        scorer, and the prompt already says "Name him" — about half the time
        the first answer does not, and comes back a fact about nobody, which
        ``note_claim`` in the gate refuses outright. So this stage checks its
        own beat-3 answers the same way it checks a repeated opener: once,
        same call, and whatever comes back is what goes out.

        ``roster`` is every name this match can legitimately carry, and it is
        used for one thing only: telling a line that opens by shouting a name
        from a line that opens "Save!" or "Buried!". Beats 2 and 3 may not be
        shouted on a name — see :data:`UNSHOUTED_BEATS` — and the check needs
        to know which capitalised words are people. Empty is safe; the scorer
        and the names on the form are checked either way.

        ``already_said`` is every line that has gone out about the moment
        this line is also about: inside a goal window, the call and the beats
        since it (:attr:`commentary.goalfollow.GoalFollowup.spoken`); over a
        replay, the live call and the earlier lines of the sequence. A line
        that repeats three words of it is re-asked once and then dropped, and
        a goal's beat 2 whose call named nobody keeps the scorer's name at the
        front when the shout comes off it.

        ``replay_first`` matters only on a replay form and is the one thing
        the model cannot see for itself: whether an earlier line in this same
        replay sequence has already named it as a replay. Whoever is counting
        the sequence — the runtime, or the rephrase — says so here. See
        :func:`commentary.prompts.phraser.replay_block`. When it is false, a
        leading "as we see it again" is taken off the answer in code: the
        model was told the sequence had already been named and named it again
        in two lines of three.
        """
        self.last_reason = ""
        self.chose_silence = False
        self.last_usage = Usage()
        blocks = phraser_blocks(
            line,
            state_summary,
            self.recent,
            home=self.home,
            away=self.away,
            on_the_ball=on_the_ball,
            notes=notes,
            callbacks=callbacks,
            ledger=ledger,
            last_event=self._recent[-1].event if self._recent else None,
            followup=followup,
            replay_first=replay_first,
            dead_ball=self.dead_ball,
        )
        try:
            parsed = await self.backend.parse(
                model=self.model,
                system=self.system,
                blocks=blocks,
                output_format=PhrasedLine,
                max_tokens=self.config.max_tokens,
                effort="low",
                cache_system=True,
                tag="phraser",
            )
        except LLMError as exc:
            self.last_reason = f"model call failed: {exc}"
            return None

        usage = parsed.usage
        proposed = parsed.value

        # The register repeat, both ends of the sentence, one re-ask. Asked
        # once: whatever comes back — even the same word again — is what goes
        # out, and a second re-ask is never made.
        opener = self._repeated_opener(proposed)
        closer = self._repeated_closer(proposed)
        opener_retry = False
        closer_retry = False
        if opener is not None or closer is not None:
            retry = await self._reask(blocks, _register_retry_note(opener or "", closer or ""))
            if retry is not None:
                usage = usage + retry.usage
                proposed = retry.value
                opener_retry = opener is not None
                closer_retry = closer is not None

        # Whose the thing was. The gate refuses this outright — it is a true
        # sentence about the wrong player and there is nothing to trim — so
        # the re-ask here is the only chance the line gets.
        swap_retry = False
        swapped = possessive_swap(proposed.line, line.line, self._names_here(line, scorer, roster))
        if swapped:
            retry = await self._reask(blocks, _swap_retry_note(swapped))
            if retry is not None:
                usage = usage + retry.usage
                proposed = retry.value
                swap_retry = True

        name_retry = False
        if goal_beat == SCORER_BEAT and scorer and self._needs_a_name(proposed, scorer):
            retry = await self._reask(blocks, _name_retry_note(scorer))
            if retry is not None:
                usage = usage + retry.usage
                proposed = retry.value
                name_retry = True
                # Asked once. Whatever came back — even nameless again — is
                # what goes out; the gate is the backstop from here.

        # Beat 3's number, against the clause it was handed. The fault is on
        # ``runs/rephrased/r4-shape/mbappe``: "That's his second World Cup
        # goal, and he's chasing a second title." — built out of a clause
        # about chasing a second World Cup, true of nothing, and the seventh
        # goal of the tournament it should have said went unsaid.
        figure_retry = False
        if goal_beat == SCORER_BEAT and notes:
            allowed = set().union(*(figures_in(note.text) for note in notes))
            extra = figures_in(proposed.line) - allowed
            if extra:
                retry = await self._reask(blocks, _figure_retry_note(extra, notes))
                if retry is not None:
                    usage = usage + retry.usage
                    proposed = retry.value
                    figure_retry = True
                if figures_in(proposed.line) - allowed:
                    self.last_usage = usage
                    self.chose_silence = True
                    self.last_reason = (
                        "figure_not_in_the_clause: "
                        + ", ".join(str(number) for number in sorted(extra))
                    )
                    return PhrasedLine(
                        line="",
                        excitement=0.0,
                        opener_retry=opener_retry,
                        closer_retry=closer_retry,
                        name_retry=name_retry,
                        swap_retry=swap_retry,
                        figure_retry=True,
                    )

        # The goal call's own shape, on a line that is not the call. Unlike
        # every other check here this one does not stop at a re-ask: a beat
        # that comes back shouting the name again is rewritten rather than
        # dropped, because a dropped beat is a hole in the thirty seconds
        # after a goal and the shout is the one thing wrong with the line.
        names = self._names_here(line, scorer, roster)
        # The call named nobody — "Over the wall, into the top corner!" — so
        # the shout coming off beat 2 must not take the only naming of the
        # scorer with it.
        keep_name = bool(
            scorer and already_said and not mentions(already_said[0], scorer)
        )
        shout_retry = False
        shout_rewritten = False
        # A replay line is past-tense and about something already called, so
        # the shout is beat 1's there too.
        if goal_beat in UNSHOUTED_BEATS or line.scene is Scene.REPLAY:
            shouted = opening_shout(proposed.line, names)
            if shouted is not None:
                retry = await self._reask(blocks, _shout_retry_note(shouted, goal_beat))
                if retry is not None:
                    usage = usage + retry.usage
                    proposed = retry.value
                    shout_retry = True
                if opening_shout(proposed.line, names) is not None:
                    proposed = proposed.model_copy(
                        update={
                            "line": unshout(proposed.line, keep_name=keep_name, names=names)
                        }
                    )
                    shout_rewritten = True

        # The same thing said twice about one goal. Unlike every other check
        # here this one ends in the line being dropped: a beat that repeats
        # the call carries nothing, and an empty beat is better than a third
        # saying of one phrase.
        repeat_retry = False
        if (goal_beat is not None or line.scene is Scene.REPLAY) and already_said:
            # The call's own detail is protected whatever its length: "off the
            # ground" is three words and two of them ordinary, so the shared
            # run waves it through, and it is the most memorable phrase in the
            # sequence.
            spent = protected_phrases(already_said[0], line.detail or "")
            repeated = shared_run(proposed.line, already_said) or repeats_the_detail(
                proposed.line, spent
            )
            if repeated:
                retry = await self._reask(blocks, _repeat_retry_note(repeated))
                if retry is not None:
                    usage = usage + retry.usage
                    proposed = retry.value
                    repeat_retry = True
                    # The fresh answer has not been through the shout check,
                    # and it is not worth a third call: fixed in code.
                unshouted = goal_beat in UNSHOUTED_BEATS or line.scene is Scene.REPLAY
                if unshouted and opening_shout(proposed.line, names):
                    proposed = proposed.model_copy(
                        update={"line": unshout(proposed.line, keep_name=keep_name, names=names)}
                    )
                    shout_rewritten = True
                still = shared_run(proposed.line, already_said) or repeats_the_detail(
                    proposed.line, spent
                )
                if still:
                    self.last_usage = usage
                    self.chose_silence = True
                    self.last_reason = (
                        f'repeat: "{still}" had already gone out about this moment'
                    )
                    return PhrasedLine(
                        line="",
                        excitement=0.0,
                        opener_retry=opener_retry,
                        closer_retry=closer_retry,
                        name_retry=name_retry,
                        swap_retry=swap_retry,
                        figure_retry=figure_retry,
                        shout_retry=shout_retry,
                        shout_rewritten=shout_rewritten,
                        repeat_retry=True,
                    )

        # And the antecedent. A call of "Over the wall, into the top corner!"
        # names nobody, so "He knew it from the moment it left his boot." is a
        # line about a man the listener has never been told about. The shout
        # rule takes "<Scorer>!" off the front; this puts "<Scorer>," on it.
        if goal_beat == 2 and keep_name and scorer and not mentions(proposed.line, scorer):
            proposed = proposed.model_copy(
                update={"line": _lead_with(proposed.line, scorer, names)}
            )
            shout_rewritten = True

        # The name and nothing else, on a form holding a how. Asked once;
        # what comes back is what goes out, because a shout is a line even
        # when it is half of one.
        thin_retry = False
        if (
            goal_beat is None
            and line.event is Event.GOAL
            and is_a_thin_call(proposed.line, line.line)
        ):
            retry = await self._reask(blocks, _thin_call_note(line.line))
            if retry is not None:
                usage = usage + retry.usage
                proposed = retry.value
                thin_retry = True

        # No fact welded on with a dash.
        welded = _WELDING_DASH.search(proposed.line) is not None
        if welded:
            proposed = proposed.model_copy(update={"line": unweld(proposed.line)})

        # The replay is named once a sequence, and this is the line after
        # the one that named it.
        marker = ""
        if line.scene is Scene.REPLAY and not replay_first:
            without, marker = strip_replay_marker(proposed.line)
            if marker:
                proposed = proposed.model_copy(update={"line": without})

        # A beat that is one name is not a beat. Unlike the goal call, where
        # the shout is the line, a follow-up that has come down to the man's
        # name says nothing that was not said twelve seconds ago.
        if goal_beat is not None and is_a_bare_name(proposed.line, names):
            self.last_usage = usage
            self.chose_silence = True
            self.last_reason = f"bare_name: beat {goal_beat} came down to the name alone"
            return PhrasedLine(
                line="",
                excitement=0.0,
                opener_retry=opener_retry,
                closer_retry=closer_retry,
                name_retry=name_retry,
                swap_retry=swap_retry,
                figure_retry=figure_retry,
                shout_retry=shout_retry,
                shout_rewritten=shout_rewritten,
                repeat_retry=repeat_retry,
            )

        # A side's verb on one man. Asked once, then dropped: the line is
        # about the wrong subject and there is no half of it to keep.
        plural_retry = False
        plural = plural_on_one_man(
            proposed.line, names=names, sides=[self.home, self.away]
        )
        if plural:
            retry = await self._reask(
                blocks, _plural_retry_note(plural, scorer or "one player")
            )
            if retry is not None:
                usage = usage + retry.usage
                proposed = retry.value
                plural_retry = True
            if plural_on_one_man(proposed.line, names=names, sides=[self.home, self.away]):
                self.last_usage = usage
                self.chose_silence = True
                self.last_reason = f'plural_on_one_man: "{plural}" is a side, not a player'
                return PhrasedLine(
                    line="",
                    excitement=0.0,
                    opener_retry=opener_retry,
                    closer_retry=closer_retry,
                    name_retry=name_retry,
                    swap_retry=swap_retry,
                    figure_retry=figure_retry,
                    shout_retry=shout_retry,
                    shout_rewritten=shout_rewritten,
                    repeat_retry=repeat_retry,
                    plural_retry=True,
                )

        # The jingle, taken off in code. One line ending a clause on "now" is
        # a commentator; four in twenty-one is a tic, and the closer check
        # cannot see it because three of the four end on another word
        # entirely.
        now_stripped = False
        if has_the_now_tail(proposed.line) and any(
            has_the_now_tail(said.text) for said in list(self._recent)[-_NOW_LOOKBACK:]
        ):
            without = strip_now_tail(proposed.line)
            if without:
                proposed = proposed.model_copy(update={"line": without})
                now_stripped = True

        self.last_usage = usage
        settled = self._settle(proposed, self._word_cap(line, goal_beat))
        return settled.model_copy(
            update={
                "opener_retry": opener_retry,
                "closer_retry": closer_retry,
                "now_stripped": now_stripped,
                "plural_retry": plural_retry,
                "thin_retry": thin_retry,
                "unwelded": welded,
                "swap_retry": swap_retry,
                "figure_retry": figure_retry,
                "name_retry": name_retry,
                "shout_retry": shout_retry,
                "shout_rewritten": shout_rewritten,
                "repeat_retry": repeat_retry,
                "replay_marker_stripped": marker[:64],
            }
        )

    async def _reask(self, blocks: Sequence[Block], note: str) -> Parsed[PhrasedLine] | None:
        """The same call again with a note on the end of it, or ``None``.

        ``None`` is the re-ask itself failing to come back, and the answer
        already in hand is kept rather than the line lost over it. Every
        check in :meth:`phrase` re-asks exactly once and through here, so
        one line can never cost more than one extra call per fault.
        """
        try:
            return await self.backend.parse(
                model=self.model,
                system=self.system,
                blocks=_with_note(blocks, note),
                output_format=PhrasedLine,
                max_tokens=self.config.max_tokens,
                effort="low",
                cache_system=True,
                tag="phraser",
            )
        except LLMError:
            return None

    @staticmethod
    def _names_here(
        line: CallerLine, scorer: str | None, roster: Sequence[str]
    ) -> list[str]:
        """Everybody this call could legitimately be shouting at."""
        names = [scorer or "", *((s.name or "") for s in line.sightings), *roster]
        return [name for name in names if name.strip()]

    def _word_cap(self, line: CallerLine, goal_beat: int | None) -> int:
        """The hard trim for this moment, which is not one number any more.

        28 is the backstop everywhere the ball is live: study section 1 puts
        club football's 95th percentile at 22-27 words. At a restart and in
        the thirty seconds after a goal the corpus goes past it — section 2.3
        has one restart line in five over sixteen words, section 2.4 has 60
        words in 30 seconds — and a line asked for at 12 to 22 words that is
        trimmed at 28 has the trim deciding how it ends. So those kinds get
        :attr:`~commentary.config.DeadBallConfig.max_words` instead, and the
        trim is still there.
        """
        if is_long_line(line, followup_beat=goal_beat):
            return max(self.config.max_words, self.dead_ball.max_words)
        return self.config.max_words

    @staticmethod
    def _needs_a_name(proposed: PhrasedLine, scorer: str) -> bool:
        """A number with nobody's name on it — beat 3's own failure mode.

        Checked against the model's raw answer, not the trimmed one: cleaning
        never adds or removes a name, so there is nothing the settle step
        could change this by, and checking the raw answer means the retry
        fires on exactly what the gate is about to see.
        """
        text = proposed.line.strip()
        if not text or not says_a_number(text):
            return False
        return not mentions(text, scorer)

    def _repeated_opener(self, proposed: PhrasedLine) -> str | None:
        """The offending opener word if this line needs a re-ask, else ``None``.

        Stacked repetition is how a goal sounds (excitement >= 0.9), and a
        bare surname is a legitimate repeat rather than a repeated frame, so
        both are waved through without a retry.
        """
        if proposed.excitement >= 0.9:
            return None
        text = proposed.line.strip()
        if not text or _is_bare_name(text):
            return None
        word = _opening_word(text)
        if not word:
            return None
        recent = list(self._recent)[-_OPENER_LOOKBACK:]
        if any(_opening_word(said.text) == word for said in recent):
            return _display_word(text)
        return None

    def _repeated_closer(self, proposed: PhrasedLine) -> str | None:
        """The offending closing word if this line needs a re-ask, else ``None``.

        The same rule as :meth:`_repeated_opener` at the other end of the
        sentence, and the same two exemptions for the same reasons: a goal is
        shouted in repeated fragments, and a bare surname is a whole line
        that happens to be one word.

        What it is for is the tail this voice actually writes. On
        ``runs/rephrased/r1-replay/mbappe``: "Through the middle at speed." /
        "Through midfield now." / "Wide on the right now." / "Into the corner
        now." / "Striding out now, France in no hurry to move it on." Four of
        the five end on "now" and no two of them open on the same word, so
        the opener check passed every one. The register judge counts a closer
        repeat the same way it counts an opener repeat.
        """
        if proposed.excitement >= 0.9:
            return None
        text = proposed.line.strip()
        if not text or _is_bare_name(text):
            return None
        word = _closing_word(text)
        if not word:
            return None
        recent = list(self._recent)[-_CLOSER_LOOKBACK:]
        if any(_closing_word(said.text) == word for said in recent):
            return _display_closer(text)
        return None

    def _settle(self, proposed: PhrasedLine, max_words: int) -> PhrasedLine:
        """The same two post-conditions the caller applies, for the same reasons.

        A "Commentary:" label or a pair of quotation marks is harmless on a
        page and ruinous through a speech synthesiser, and a model that
        ignores a word cap must not be able to hold the voice channel while
        the next chance goes past.

        ``max_words`` is the moment's cap rather than the config's, because
        the cap is per kind now: see :meth:`_word_cap`.

        And the line ends on a stop. "France through the middle at speed"
        went out without one on ``runs/rephrased/r5b/offside``: a trim or a
        strip had taken the end of the sentence with the thing it removed,
        and a synthesiser reads an unpunctuated line straight into the next.
        """
        text = _ends_on_a_stop(trim_words(clean_line(proposed.line), max_words))
        if not text:
            # An empty answer is a choice; an answer that was only a label
            # or a pair of quotation marks is a failed one. The difference
            # decides whether the moment passes in silence or the caller's
            # own line goes out, so it is drawn on what the model actually
            # returned rather than on what survived the cleaning.
            self.chose_silence = not proposed.line.strip()
            self.last_reason = (
                "the phraser chose silence"
                if self.chose_silence
                else "the phraser wrote nothing sayable"
            )
        return proposed.model_copy(update={"line": text})
