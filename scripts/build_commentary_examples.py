"""Regenerate ``commentary.prompts.commentary_examples`` from real captions.

The phraser is not taught a register by adjectives. It is shown what a
commentator actually said, in the caption file a broadcaster shipped with the
video, and asked to sound like that. So the example set has to come out of
captions rather than out of anybody's idea of how football is called, and it
has to be regenerable: point this at more caption files later and the set
grows.

    uv run python scripts/build_commentary_examples.py

With no arguments that rebuilds the file from :data:`POOL`: six of the seven
caption files in ``clips/``, each with its own pack and its own kickoff
offset. Which six, and why not the seventh, is on :data:`POOL` — the short
version is that `lei-mun-2015` is radio commentary at 195 words a minute and
would drag every length in the set (study section 10.2), and that
`argfra-dimaria`, the 2022 World Cup final the whole example set used to be
built from, is capped at one sixth of each bucket by :data:`SHARE_CAP`
because study section 1 shows it is the odd feed out in this corpus.

**Several matches at once.** ``--captions`` and ``--pack`` each take one or
more paths and are paired in order — the first ``--captions`` file with the
first ``--pack``, the second with the second, and so on — rather than a
combined ``captions.json3:pack.json`` token, because that pairing is already
how argparse's ``nargs="+"`` works and needed no new syntax:

    uv run python scripts/build_commentary_examples.py \
        --captions clips/argfra-dimaria.en.json3 clips/lei-mun-2015.en.json3 \
        --pack clips/pack-argfra-2022.json clips/pack-3754186.json \
        --after 300 78

Each caption file is filtered against *its own* pack's team sheet — a name
true on one match's roster is not automatically true on another's — but the
"already said in lower case somewhere" vocabulary that separates a real word
from a mangled surname (see :func:`names_are_clean`) is pooled across every
caption file given, the same as when a single match's commentary was split
across several files. ``--after`` is one kickoff offset per file, same order
as ``--captions``; give fewer values than files and the last one repeats, so
a single ``--after 300`` still applies to every file the way it always did.

What it does, in order.

1. **Utterances, not caption segments.** YouTube writes a rolling caption:
   half-sentences, stamped when they appeared on screen, with the sentence
   finishing two segments later. The stream is reassembled and split again on
   the three things that really end an utterance — a sentence-final stop, a
   ``>>`` speaker change, and a gap of two seconds or more. This is the same
   reconstruction ``runs/prompt-name/REAL_COMMENTARY.md`` measured, so the
   medians the phraser's prompt quotes are medians of these strings.

2. **Live play only, as far as captions can say.** Anything bracketed is
   crowd noise the ASR gave up on. Anything in the studio vocabulary is the
   build-up or the half-time panel. Anything before ``--after`` is neither.

3. **No garbled names.** Auto-captions mangle surnames constantly, and a
   mangled surname in a prompt is a name the phraser may repeat out loud. So
   every capitalised word that does not start the utterance has to be on the
   team sheet or in a short list of words a broadcast says. One word that is
   not drops the utterance whole — there are thousands left.

4. **A kind, by keyword.** Loose on purpose: the kinds exist so the prompt can
   show a spread rather than forty bare surnames, not so that anything
   downstream can trust the label. There are twenty-one of them, which is
   Gap 8 of the corpus study: a corner, a free kick, a throw-in and a goal
   kick used to share one `dead_ball` bucket, and a card, an offside, a
   substitution, a kickoff, an injury, a cross, a switch of play, a
   restatement of the score and a line carrying a number had no examples at
   all. The keywords for each are read off the verbatim quotations in study
   sections 3.2, 5.2 and 5.3.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from commentary.grading.captions import load_json3
from commentary.grading.transcripts import Segment
from commentary.schemas import KnowledgePack

#: A gap this long between two caption segments ends the utterance.
GAP_S = 2.0

#: Longest utterance kept. 28 words is the caller's cap and now the
#: phraser's: study section 1 puts club football's 95th percentile at 22-27
#: words and one utterance in five at sixteen or more, so a shorter cap here
#: would teach the length the system already has.
#:
#: The character cap used to be 88, which is what fits on one 100-column
#: source line, and it bit long before the word cap did — 88 characters is
#: about fifteen words, so the top fifth of the distribution could not reach
#: the file whatever MAX_WORDS said. :func:`_quoted` wraps instead.
MAX_WORDS = 28
MAX_CHARS = 170

_SPEAKER = re.compile(r">>\s*")
_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*$")

#: Words a broadcast says out loud that are on nobody's team sheet. Anything
#: capitalised, mid-utterance and not here or on the sheet is treated as a
#: mangled name and costs the whole utterance.
_ALLOWED_TEXT = """
var fifa world cup qatar lusail doha final finals tournament europe
america american south african european premier league champions
i i'm i've i'll i'd o oh ah ok okay yes no now here there and but so
monday tuesday wednesday thursday friday saturday sunday
"""
ALLOWED_PROPER = frozenset(_ALLOWED_TEXT.split())

#: The studio, the adverts and the panel. None of it is live play.
STUDIO = (
    "brought to you by",
    "sponsor",
    "coming up",
    "after the break",
    "welcome back",
    "welcome to",
    "join us",
    "half-time",
    "half time",
    "full-time",
    "full time",
    "team news",
    "line-ups",
    "lineups",
    "studio",
    "let's take a look",
    "highlights",
    "subscribe",
)

#: First match wins, so the order is the order of specificity. Two kinds are
#: decided by pattern rather than by substring and are checked before this
#: table — see :func:`kind_of`.
#:
#: The new kinds below `dead_ball` are Gap 8 of
#: ``docs/research/real-commentary-corpus.md``: a corner, a free kick, a
#: throw-in and a goal kick used to draw from one `dead_ball` bucket, and
#: there were no card, offside, substitution, kickoff, injury, cross, switch
#: or restatement examples at all. Every keyword list here is read off the
#: verbatim quotations in section 3.2, which is why some of them are turns of
#: phrase rather than event words: 8% of free-kick deliveries are called a
#: free kick and 1% of goal kicks are called a goal kick (section 3), so the
#: vocabulary that finds those lines is the vocabulary of what is said
#: instead.
KIND_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        # First, because a replay line is about something else — a goal, a
        # foul, a save — and every other rule would claim it for that thing
        # and teach the phraser to call it live. Section 3.2's "VAR and
        # replay talk": 26 utterances across the four aligned matches, and
        # the commonest openers are "As we see …", "Having seen the replay
        # …", "Watch this."
        "replay",
        (
            "as we see",
            "having seen the replay",
            "the replay",
            "in the replay",
            "watch this",
            "see it again",
            "see that again",
            # Not a bare "once more": it is the commonest replay opener in
            # the corpus and also an ordinary thing to say about live play —
            # "Iniesta once more spreads play out to the left" — and six of
            # the first sixteen this bucket found were that, present tense,
            # in the one bucket that must not teach it.
            "look at that incident",
            "look at that again",
            "look at it again",
            "we've just seen",
            "being checked",
            "little check",
        ),
    ),
    (
        "substitution",
        (
            "replaced by",
            "coming on",
            "comes on",
            "will come on",
            "on will come",
            "substitution",
            "substitute",
            "taken off",
            "forced off",
            "his debut",
            "first appearance",
            "on loan from",
        ),
    ),
    (
        "card",
        (
            "yellow card",
            "red card",
            "second yellow",
            "booked",
            "booking",
            "cautioned",
            "in the book",
            "referee's book",
            "sent off",
            "shown a",
        ),
    ),
    ("offside", ("offside", "flag is up", "flag for", "linesman")),
    (
        "injury",
        (
            "treatment",
            "clash of heads",
            "physio",
            "stretcher",
            "injury",
            "injured",
            "down on the turf",
            "getting to his feet",
            "back to his feet",
            "casualty",
            "down at the minute",
            "water break",
        ),
    ),
    (
        "goal",
        (
            "goal",
            "scores",
            "scored",
            "it's in",
            "in the net",
            "buries",
            "back of the net",
            "equaliser",
            "equalizer",
        ),
    ),
    (
        "save",
        ("save", "saves", "saved", "keeper", "goalkeeper", "parried", "tipped", "denied", "palms"),
    ),
    (
        "shot",
        (
            "shot",
            "shoots",
            "strikes",
            "fires",
            "fired",
            "curls",
            "drives it",
            "volley",
            "header",
            "heads",
            # "wide" on its own claimed every tactical line about a winger
            # staying wide. The shot is the ball going wide of something.
            "wide of",
            "goes wide",
            "well wide",
            "over the bar",
            "the post",
            "crossbar",
            "chance",
            "effort",
            "blocked",
        ),
    ),
    (
        "foul",
        (
            "foul",
            "fouled",
            "penalty",
            "handball",
            "hand ball",
            "whistle",
            "goes down",
            "knocked over",
            "pulling players back",
            "challenge from",
            "push in the back",
        ),
    ),
    (
        "corner",
        ("corner", "flag kick"),
    ),
    (
        "free_kick",
        ("free kick", "free-kick", "the wall", "concedes a free"),
    ),
    (
        "throw_in",
        ("throw in", "throw-in", "a throw", "the throw", "long throw"),
    ),
    (
        "goal_kick",
        ("goal kick",),
    ),
    (
        "kickoff",
        (
            "underway",
            "we're off",
            "kick-off",
            "kickoff",
            "ready to go",
            "centre circle",
            "center circle",
            "right to left",
            "left to right",
            "blown his whistle",
            "second half of this",
        ),
    ),
    (
        "cross",
        (
            "cross",
            "crosses",
            "crossed",
            "ball in",
            "ball into",
            "swung in",
            "whipped",
            "all the way across",
            "back post",
            "near post",
            "into the box",
            "into the area",
            "delivery",
            "hangs it up",
        ),
    ),
    (
        "switch",
        (
            "switch",
            "switches",
            "switched",
            "switching",
            "sprays",
            "far side",
            "other side",
            "down the left",
            "down the right",
            "out to the right",
            "out to the left",
            "diagonal",
            "across the pitch",
        ),
    ),
    (
        "dead_ball",
        ("restart", "the spot", "set piece", "drop ball", "taken quickly", "advantage"),
    ),
    (
        "pass",
        (
            "played by",
            "on by",
            "out by",
            "in by",
            "flicked",
            "collected",
            "regathered",
            "moved on",
            "passes",
            "pass",
            "finds",
            "square",
            "threaded",
            "knocks",
            "knocked",
            "sifted",
            "spelled",
            "touched",
            "chased",
            "taken up",
            "carries",
            "runs",
            "away from",
            "wriggl",
            "dispossessed",
            "intercept",
            "cleared",
            "clearance",
            "tackle",
            "forward",
            "backwards",
        ),
    ),
)

#: The score and the clock, restated. Section 5.3: a small closed set, said
#: as two short utterances, and `bar-mal-2019` does it every 150 seconds.
#: Checked before :data:`KIND_RULES` because "1-0 to Barcelona" carries no
#: event word at all and "10 minutes into the second half" would be read as a
#: kickoff.
_SCORE = r"(?:\d{1,2}|nil|one|two|three|four|five|six|seven)"
RESTATEMENT = re.compile(
    r"\b\d+ minutes? (?:gone|played|left|remaining)\b"
    r"|\b\d+ minutes? into the (?:first|second) half\b"
    r"|\bminutes gone\b|\bhalf hour gone\b|\bhalfway through the\b"
    r"|\binto the last\b|\bapproach(?:ing)? the last\b|\blast minute\b"
    r"|\bstoppage time\b|\badditional time\b"
    rf"|\b{_SCORE}[-– ]{_SCORE}\s+to\b|\bstill\s+{_SCORE}[-– ]{_SCORE}\b"
    rf"|\bleading\b[^.]{{0,40}}\b{_SCORE}[-– ]{_SCORE}\b",
    re.IGNORECASE,
)

#: A number that is doing a commentator's work: a tally, a run, a record.
#: Gap 1 item 2 asks for these shapes by name, because the system's v3 traces
#: carried two numbers in twenty-seven lines and one of them was the clock.
#: Both halves have to be there — a digit on its own is a shirt number and a
#: phrase on its own is ordinary talk.
#: "first" and "second" are left out on purpose: in this corpus they are
#: nearly always the first half or the second goal, and letting them in
#: filled the bucket with ordinary talk. The two shapes worth keeping —
#: a first goal and a first appearance — are named instead.
_COUNT = (
    r"\b(?:\d+(?:st|nd|rd|th)?|third|fourth|fifth|sixth|seventh|eighth|ninth"
    r"|tenth|hat[- ]?trick|treble|first (?:ever |league |premier league )?goal"
    r"|first appearance)\b"
)
_NUMBER_ABOUT = (
    r"\b(?:of the season|of the campaign|this season|last season|of the game|in a row"
    r"|consecutive|unbeaten|in all competitions|out of|appearances|assists|goals?"
    r"|league goal|points|games|matches|since|record|his \w+ goal|in the tournament"
    r"|years|birthday|wage bill|million)\b"
)
NUMBERS = re.compile(
    rf"(?:{_COUNT}[^.]{{0,60}}{_NUMBER_ABOUT}|{_NUMBER_ABOUT}[^.]{{0,40}}{_COUNT})"
)

#: Every keyword above, compiled with a word boundary in front of it. As
#: plain substrings "cross" matched "across the surface" and filed a colour
#: line about the pitch as a ball into the box; a prefix match still catches
#: the inflections that matter ("crosses", "crossing", "wriggling").
KIND_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (kind, re.compile("|".join(rf"\b{re.escape(needle)}" for needle in needles)))
    for kind, needles in KIND_RULES
)

#: The second voice, by its first word. Study section 4.1 validated exactly
#: this list against the `>>` speaker markers and found a three-fold lift on
#: every file that has them. It is not a speaker label, but a line that opens
#: "Well," or "Yeah," is the analyst three times out of ten rather than one,
#: and the lead's own buckets are where that does damage: the phraser is
#: being taught to call the ball, not to agree with a colleague.
COLOUR_OPENER = re.compile(
    r"^(?:well|yeah|yes|i think|i mean|you know|absolutely|exactly|for me|listen)\b",
    re.IGNORECASE,
)

#: The longest unmatched-but-named utterance still filed as build-up. The
#: median caption window around a build-up carry or pass is fourteen words
#: (study section 3) and the median utterance is eight (section 1).
BUILD_UP_MAX_WORDS = 14

#: The kinds that are the lead calling the ball. A colour-opener line that
#: matches one of these is filed as an aside instead. The kinds left out are
#: the ones the study shows genuinely are the analyst talking — a card, an
#: injury and an offside are argued about rather than called (section 3.2),
#: and the colour seat is five times likelier to speak at a stoppage than in
#: an attacking move (section 4.2).
LEAD_ONLY = frozenset(
    {
        "build_up",
        "pass",
        "cross",
        "switch",
        "shot",
        "save",
        "goal",
        "numbers",
        "corner",
        "free_kick",
        "throw_in",
        "goal_kick",
        "kickoff",
        "restatement",
    }
)

#: A capitalised sentence-opener in the middle of an utterance: the caption
#: stream glued two of them together and no full stop came through to split
#: them. "The header from Benzema, and it's Well, it's a brilliant save" is
#: two lines, and as one it teaches a shape nobody said. "I" is left out
#: because it is capitalised mid-sentence for real.
GLUED = re.compile(
    r"(?<=[a-z,]) (?:Well|There's|And|It's|Oh|So|But|He|They|Now|That's|This|What|Yeah|Yes|We)\b"
)

#: The ASR's rendering of a hesitation. It marks spontaneous talk — the
#: analyst thinking aloud — and not the called line, and read back in a
#: prompt it is noise that the phraser can only copy.
FILLER = re.compile(r"\b(?:uh|um|erm|uhh|mm)\b", re.IGNORECASE)

#: Kinds in the order the prompt shows them: the four that are most of live
#: commentary, then the loud ones, then the restarts, then the rest.
KIND_ORDER = (
    "build_up",
    "pass",
    "cross",
    "switch",
    "shot",
    "save",
    "goal",
    "numbers",
    "foul",
    "card",
    "offside",
    "corner",
    "free_kick",
    "throw_in",
    "goal_kick",
    "kickoff",
    "dead_ball",
    "substitution",
    "injury",
    "restatement",
    "replay",
    "aside",
)

#: How many of each to keep in the generated file. Build-up and passing are
#: most of live commentary and most of what the caller writes badly, so they
#: get the room.
#:
#: The loud three — goal, shot, save — are set above anything these captions
#: can supply, so every chance utterance that survives the filters is kept.
#: The kinds added for Gap 8 are held to 20: enough that the stride through
#: them is a spread over the six matches rather than one afternoon, and few
#: enough that twenty-one buckets still fit in a cacheable prompt.
QUOTA = {
    "build_up": 60,
    "pass": 55,
    "cross": 25,
    "switch": 20,
    "shot": 40,
    "save": 25,
    "goal": 30,
    "numbers": 25,
    "foul": 30,
    "card": 20,
    "offside": 20,
    "corner": 20,
    "free_kick": 20,
    "throw_in": 20,
    "goal_kick": 20,
    "kickoff": 20,
    "dead_ball": 20,
    "substitution": 20,
    "injury": 20,
    "restatement": 20,
    # Everything the corpus has. 26 utterances match the replay vocabulary
    # across four aligned matches (study section 3.2) and the bucket here is
    # wider than that vocabulary, so a quota above what is found keeps all
    # of it: this is the rarest kind in the set and the one the prompt has
    # nothing else to teach from.
    "replay": 20,
    "aside": 25,
}


@dataclass(frozen=True)
class Utterance:
    """One thing a commentator said, with the second it started."""

    ts: float
    text: str

    @property
    def words(self) -> int:
        return len(self.text.split())


def utterances(segments: Sequence[Segment]) -> Iterator[Utterance]:
    """Caption segments back into the sentences somebody actually spoke."""
    start: float | None = None
    parts: list[str] = []
    previous_end = 0.0

    for segment in segments:
        raw = segment.text
        gap = start is not None and segment.start - previous_end >= GAP_S
        if (gap or raw.lstrip().startswith(">>")) and parts and start is not None:
            yield Utterance(ts=start, text=" ".join(parts))
            parts, start = [], None
        for piece in _SPEAKER.split(raw):
            for word in piece.split():
                if start is None:
                    start = segment.start
                parts.append(word)
                if _SENTENCE_END.search(word):
                    yield Utterance(ts=start, text=" ".join(parts))
                    parts, start = [], None
        previous_end = segment.end

    if parts and start is not None:
        yield Utterance(ts=start, text=" ".join(parts))


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )


def roster_words(pack: KnowledgePack) -> set[str]:
    """Every word of every name on either sheet, folded to plain lower case."""
    words: set[str] = set()
    for sheet in (pack.home, pack.away):
        for label in (sheet.name, sheet.short, sheet.demonym, sheet.manager or ""):
            words.update(_letters(label).split())
        for player in sheet.squad:
            words.update(_letters(player.name).split())
    return {word for word in words if word}


def _letters(text: str) -> str:
    return re.sub(r"[^a-z' ]+", " ", strip_accents(text.lower()))


def names_are_clean(
    text: str, allowed: frozenset[str] | set[str], common: frozenset[str] | set[str]
) -> bool:
    """Is every capitalised word here a word a broadcast really says?

    Every capitalised word is a proper noun on the team sheet, a word in the
    allow-list, an ordinary word that happens to start a sentence, or the
    ASR's guess at a surname. The last is the thing this exists to keep out of
    the prompt, and telling it from the third is what ``common`` is for: the
    set of words the same captions use in lower case somewhere else. "Spread"
    opening a sentence is safe because "spread" appears lower-cased a hundred
    times; "MacAllister" opening one is not, because it never does — and the
    team sheet spells him Mac Allister.

    Without that set the first word had to be exempt, and a bare surname is
    exactly the shape of line worth having, so every mangled one got in.
    """
    for token in text.split():
        word = re.sub(r"[^A-Za-z'\-]", "", token)
        if len(word) < 2 or not word[0].isupper():
            continue
        folded = strip_accents(word.lower())
        base = folded[:-2] if folded.endswith("'s") else folded
        if folded in common or base in common:
            continue
        for part in base.replace("-", " ").split():
            if part and part not in allowed:
                return False
    return True


def kind_of(text: str) -> str:
    """A loose label, by keyword, first rule wins."""
    lowered = text.lower()
    # "Goal kick given." is a restart, and the goal rule would otherwise
    # claim it on the word "goal" before goal_kick ever gets a look.
    if "goal kick" in lowered:
        return "goal_kick"
    # Two patterns before the keyword table. A restatement of the score
    # carries no event word at all — "1-0 to Barcelona" — and a tally would
    # otherwise be filed under whatever it is a tally of, so that "his 12th
    # goal of the season" lands in the goal bucket and teaches the phraser
    # to shout it. Both are study sections 5.2 and 5.3.
    if RESTATEMENT.search(lowered):
        return "restatement"
    colour = bool(COLOUR_OPENER.match(lowered))
    if NUMBERS.search(lowered) and not colour:
        return "numbers"
    for kind, pattern in KIND_PATTERNS:
        if pattern.search(lowered):
            return "aside" if colour and kind in LEAD_ONLY else kind
    if colour:
        return "aside"
    # Anything that matched no keyword at all. It used to go to build_up
    # only if it was four words or shorter, which made every build-up
    # example in the prompt four words or shorter by construction — and the
    # phraser wrote four-word build-up because that is all it had ever been
    # shown. Club football's build-up utterance is a median eight words and
    # its window a median fourteen (study sections 1 and 3), so the cut is
    # there instead. :func:`gather` still drops a build_up line with nobody's
    # name in it, which is what keeps the clock-watching out.
    return "build_up" if len(text.split()) <= BUILD_UP_MAX_WORDS else "aside"


def keep(
    utterance: Utterance, allowed: set[str], common: set[str], *, after_s: float
) -> bool:
    """Is this a line of live commentary with nothing invented in it?"""
    text = utterance.text
    if utterance.ts < after_s:
        return False
    if "[" in text or "]" in text or '"' in text or "\\" in text:
        return False
    if not 1 <= utterance.words <= MAX_WORDS or len(text) > MAX_CHARS:
        return False
    if not re.search(r"[a-z]", text):
        return False
    # A line that opens in lower case lost its beginning to a caption
    # boundary, and a line with no full stop never reached its end.
    if not text[0].isalnum() or text[0].islower():
        return False
    if not _SENTENCE_END.search(text):
        return False
    if GLUED.search(text) or FILLER.search(text):
        return False
    lowered = text.lower()
    if any(phrase in lowered for phrase in STUDIO):
        return False
    words = lowered.split()
    if any(a == b for a, b in zip(words, words[1:], strict=False)):
        # "That is a a new record" — the ASR stuttering, not the commentator.
        return False
    return names_are_clean(text, allowed, common)


def lowercase_vocabulary(paths: Iterable[Path]) -> set[str]:
    """Every word these captions ever write in lower case.

    This is what separates an ordinary word opening a sentence from a surname
    the ASR invented. See :func:`names_are_clean`.
    """
    words: set[str] = set()
    for path in paths:
        for segment in load_json3(path).segments:
            for token in segment.text.split():
                word = re.sub(r"[^A-Za-z']", "", token)
                if word and word[0].islower():
                    words.add(strip_accents(word.lower()))
    return words


#: The most of one file's utterances any kind may be built from, by the
#: file's stem. Study Gap 5 item 2: the 2022 final is one seventh of the
#: corpus and the odd one out in it — 72 words a minute against club
#: football's 115, a bare surname three times as often, the
#: participle-plus-`by` form four times as often — and the example set used
#: to be built from it alone. One sixth is its share of the corpus, and the
#: cap is what holds it there when a kind happens to be commonest in it.
SHARE_CAP = {"argfra-dimaria": 1.0 / 6.0}


def choose(entries: list[tuple[str, str]], quota: int, caps: dict[str, float]) -> list[str]:
    """``quota`` of these, with no capped file over its share.

    Entries are ``(file stem, text)`` in the order the matches were read.
    A capped file supplies at most ``quota * cap`` of the bucket, strided
    across its own contribution; everything else fills the rest. Both halves
    go back into the order they arrived in, so the generated file is a diff
    of the last one rather than a reshuffle.
    """
    picked: set[str] = set()
    room = quota
    for stem, cap in caps.items():
        mine = [text for key, text in entries if key == stem]
        if not mine:
            continue
        allowed = max(1, int(quota * cap))
        taken = spread(mine, min(allowed, len(mine)))
        picked.update(taken)
        room -= len(taken)
    rest = [text for key, text in entries if key not in caps]
    picked.update(spread(rest, max(0, room)))
    return [text for _, text in entries if text in picked]


def spread(found: list[str], quota: int) -> list[str]:
    """``quota`` of these, drawn evenly across the match rather than off the top.

    The head of the list is the first ten minutes after kickoff. A prompt full
    of one team's opening spell is a prompt about that spell; an even stride
    over ninety minutes is a prompt about football.
    """
    if len(found) <= quota:
        return list(found)
    step = len(found) / quota
    return [found[int(i * step)] for i in range(quota)]


def surnames_of(pack: KnowledgePack) -> set[str]:
    """Every surname on either sheet, folded, for the build-up filter."""
    return {
        strip_accents(player.surname.lower())
        for sheet in (pack.home, pack.away)
        for player in sheet.squad
    }


@dataclass(frozen=True)
class Source:
    """One caption file, the pack that says which of its names are real, and
    when its live play starts."""

    path: Path
    pack: KnowledgePack
    after_s: float

    @property
    def key(self) -> str:
        """The file stem, which is what :data:`SHARE_CAP` is keyed by."""
        return self.path.name.split(".")[0]


def gather(sources: Sequence[Source]) -> dict[str, list[tuple[str, str]]]:
    """Every kept utterance, by kind, in the order each broadcast said them.

    Each utterance is tagged with the file it came from, which is what
    :func:`choose` needs to hold one file to a share of a bucket.

    Roster words — what makes a capitalised word a real name rather than a
    mangled one — come from each source's own pack, so a name true on one
    match's team sheet cannot wave a different match's utterance through.
    The lower-case vocabulary that backs that same check (see
    :func:`names_are_clean`) is pooled across every caption file, exactly as
    it was when one match's commentary arrived split across several files.
    """
    common = lowercase_vocabulary(source.path for source in sources)
    seen: set[str] = set()
    found: dict[str, list[tuple[str, str]]] = {kind: [] for kind in KIND_ORDER}
    for source in sources:
        allowed = set(ALLOWED_PROPER) | roster_words(source.pack)
        surnames = surnames_of(source.pack)
        for utterance in utterances(load_json3(source.path).segments):
            if not keep(utterance, allowed, common, after_s=source.after_s):
                continue
            key = utterance.text.lower()
            if key in seen:
                continue
            kind = kind_of(utterance.text)
            # Build-up has no keyword of its own — it is whatever matched
            # nothing else and stayed short — so without a name in it the
            # bucket fills with clock-watching and crowd noise. Real build-up
            # commentary *is* the names: "De Paul." "Now Di María."
            if kind == "build_up" and not _has_surname(key, surnames):
                continue
            seen.add(key)
            found[kind].append((source.key, utterance.text))
    return found


def _has_surname(folded_text: str, surnames: set[str]) -> bool:
    words = {re.sub(r"[^a-z']", "", word) for word in strip_accents(folded_text).split()}
    return bool(words & surnames)


def _quoted(text: str, indent: str = "        ") -> str:
    """One example as source, wrapped to fit the line length.

    A long utterance becomes several adjacent string literals, which Python
    joins back into one at compile time. Without this the file could only
    hold what fits on one line, which is about fifteen words — and the top
    fifth of real commentary is longer than that (study section 1).
    """
    room = 100 - len(indent) - 3
    if len(text) <= room:
        return f'{indent}"{text}",'
    chunks: list[str] = []
    line = ""
    for word in text.split():
        candidate = f"{line} {word}" if line else word
        if len(candidate) > room - 1:
            chunks.append(line)
            line = word
        else:
            line = candidate
    chunks.append(line)
    body = "\n".join(
        f'{indent}"{chunk}{"" if i == len(chunks) - 1 else " "}"'
        for i, chunk in enumerate(chunks)
    )
    return body + ","


def render(found: dict[str, list[tuple[str, str]]], sources: Sequence[Path]) -> str:
    """The generated module, byte-stable for a given input."""
    chosen = {kind: choose(found[kind], QUOTA[kind], SHARE_CAP) for kind in KIND_ORDER}
    total = sum(len(v) for v in chosen.values())
    plural = "s" if len(sources) > 1 else ""
    listed = "\n".join(f"  {path}" for path in sources)
    head = f'''"""Real commentary, as a broadcaster's own captions recorded it.

Generated by ``scripts/build_commentary_examples.py``. Do not edit by hand:
run the script again, against more caption files if there are any.

These are not illustrations of a style. They are {total} things a professional
actually said during live play, reassembled into utterances and filtered down
to the ones whose every name is on a team sheet — an auto-caption mangles a
surname about as often as it gets one right, and a mangled surname in a
prompt is a name the phraser may say out loud.

The kinds are keyword guesses and nothing downstream should trust them. They
exist so the prompt can show a spread, a few of each, instead of forty bare
surnames in a row, which is what an unsorted sample of live commentary is.

Source{plural}:
{listed}
"""

from __future__ import annotations

#: Utterances by kind, in the order the prompt shows them.
EXAMPLES: dict[str, tuple[str, ...]] = {{
'''
    body: list[str] = []
    for kind in KIND_ORDER:
        body.append(f'    "{kind}": (')
        body.extend(_quoted(text) for text in chosen[kind])
        body.append("    ),")
    tail = '''}

#: The kinds, in prompt order.
KINDS: tuple[str, ...] = tuple(EXAMPLES)
'''
    return head + "\n".join(body) + "\n" + tail


#: The register pool: the caption files the examples are built from, the
#: pack that says which of each file's capitalised words are real names, and
#: the second its live play starts (study section 1's live window, section
#: 10.1's kickoff offsets).
#:
#: **`lei-mun-2015` is not here, on purpose.** It is radio commentary from
#: the Leicester City club channel, 195 words a minute against 72-115 for
#: every television feed in the corpus, with a median utterance of 11 words
#: against club football's 8 and a third of its lines running to sixteen
#: words or more. Study section 10.2 and Gap 5 item 3: it is the best file
#: here for event coverage and for threads, and it would drag the length
#: distribution of anything built from it. It stays in the corpus and out of
#: this pool.
#:
#: `argfra-dimaria` is here and capped — see :data:`SHARE_CAP`.
POOL: tuple[tuple[str, str, float], ...] = (
    ("clips/pl-liv-mun-2025.en.json3", "clips/pack-pl-liv-mun-2025.json", 152.0),
    ("clips/pl-tot-che-2024.en.json3", "clips/pack-pl-tot-che-2024.json", 168.0),
    ("clips/lei-avl-2015.en.json3", "clips/pack-3754106.json", 15.0),
    ("clips/clasico-2017.en.json3", "clips/pack-267569.json", 282.0),
    ("clips/bar-mal-2019.en.json3", "clips/pack-303451.json", 243.0),
    ("clips/argfra-dimaria.en.json3", "clips/pack-argfra-2022.json", 300.0),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="rebuild the phraser's example set")
    parser.add_argument(
        "--captions",
        nargs="+",
        default=[path for path, _, _ in POOL],
        help="yt-dlp .en.json3 caption files; defaults to the register pool above",
    )
    parser.add_argument(
        "--pack",
        nargs="+",
        default=[pack for _, pack, _ in POOL],
        help="one knowledge pack per --captions file, same order: each file's own pack "
        "says which of its capitalised words are real names",
    )
    parser.add_argument(
        "--after",
        type=float,
        nargs="+",
        default=[after for _, _, after in POOL],
        help="one kickoff offset per --captions file, same order — the build-up is not "
        "live play; give fewer values than files and the last one repeats",
    )
    parser.add_argument("--out", default="src/commentary/prompts/commentary_examples.py")
    parser.add_argument("--dry-run", action="store_true", help="print the tally and stop")
    args = parser.parse_args(argv)

    captions = [Path(p) for p in args.captions]
    pack_paths = [Path(p) for p in args.pack]
    if len(pack_paths) != len(captions):
        parser.error(
            f"--pack must give one pack per --captions file: "
            f"{len(captions)} captions, {len(pack_paths)} packs"
        )
    afters = list(args.after)
    if len(afters) > len(captions):
        parser.error(
            f"--after gives more values ({len(afters)}) than --captions files ({len(captions)})"
        )
    if len(afters) < len(captions):
        afters += [afters[-1]] * (len(captions) - len(afters))

    sources = [
        Source(
            path=cap,
            pack=KnowledgePack.model_validate(json.loads(pack_path.read_text(encoding="utf-8"))),
            after_s=after,
        )
        for cap, pack_path, after in zip(captions, pack_paths, afters, strict=True)
    ]

    found = gather(sources)

    total = 0
    for kind in KIND_ORDER:
        kept = len(choose(found[kind], QUOTA[kind], SHARE_CAP))
        total += kept
        print(f"{kind:<14} kept {kept:>3} of {len(found[kind]):>4} found")
    print(f"total {total}")
    if args.dry_run:
        return 0

    out = Path(args.out)
    out.write_text(render(found, captions), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
