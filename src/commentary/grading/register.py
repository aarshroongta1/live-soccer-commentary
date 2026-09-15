"""Is it in the register? A number, so two rephrase iterations can be compared.

Everything decided about phrasing so far was decided by reading twenty-seven
lines and counting faults by hand: detail thrown away in 7 of 27, the wrong
subject in 2, flat build-up in 9. Those counts were right and they are not
repeatable. A rewrite of the phraser prompt is worth doing only if the next
pass is better than the last one, and "it reads better" cannot answer that
across a week.

So two layers, kept apart on purpose.

**The free layer** counts the things a commentator's register shows up in and
a machine can see without an opinion: how long the lines are, how often one
is nothing but a surname, whether the openers repeat, how the gaps between
lines are distributed, whether numbers and names reach air at all. Each is
printed beside what real commentary measures, from
:mod:`commentary.grading.register_reference`. This layer costs nothing, runs
on any trace on disk, and is the one to watch iteration to iteration: a
phraser change that moves the median line from fourteen words to five moves
it here, visibly, for free.

**The paid layer** is an Opus judge reading the whole line sequence in order,
with the timestamps and the events, against forty things a professional
actually said. It answers the part no counter reaches — whether the words fit
the moment, whether the goal call has the shape of a goal call, whether
anything sounds like a second voice, and whether any line reads like a claim
the pictures could not support. It scores nine dimensions out of ten and
quotes the three worst lines back.

The two are never averaged. The free layer is arithmetic over a sample of
twenty-seven lines; the paid layer is one model's opinion, read once. A single
blended number would hide which one moved, and it is always the question of
which one moved that decides what to change next.

Nothing here is imported by the runtime and nothing here may be.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from commentary.bus import Topic
from commentary.config import JUDGE_MODEL
from commentary.gate import fold
from commentary.grading.judge import Asker, JudgeError, Question, SequentialAsker
from commentary.grading.register_reference import BANDS, ORDER, REFERENCE, SOURCE, Band
from commentary.llm.base import LLMBackend, Usage, text_block
from commentary.prompts.commentary_examples import EXAMPLES, KINDS
from commentary.schemas import KnowledgePack, MatchState, Note
from commentary.tallies import Tallies
from commentary.trace import read_trace, rows_of

#: How many real utterances the judge is shown as the reference register.
#: Forty is about as many as can be read in one pass without the sample
#: becoming the bulk of the prompt, and it covers every kind with a few of
#: each rather than forty bare surnames, which is what an unsorted sample of
#: live commentary is.
REFERENCE_N = 40

#: How far back an opener is compared for repetition. Five, because that is
#: how many previous lines the phraser is shown, so a repeat inside this
#: window is one the model had in front of it and produced anyway.
OPENER_WINDOW = 5

#: Gap thresholds, in seconds. The first is real commentary's own long-gap
#: mark; the second is the length of silence a broadcast tolerates in build-up
#: and this system has never once produced.
GAP_MARKS = (4.0, 8.0)

#: Name particles. They belong to a name but do not prove one was said: a
#: line containing "de" is not a line containing De Paul.
_PARTICLES = frozenset(
    {
        # fmt: off
        "de", "di", "da", "do", "dos", "del", "della", "van", "der", "den",
        "ten", "la", "le", "el", "il", "bin", "al", "of", "the", "and",
        # fmt: on
    }
)

_NUMBER_WORDS = frozenset(
    {
        # fmt: off
        "nil", "nought", "zero", "one", "two", "three", "four", "five", "six",
        "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
        "ninety", "hundred", "thousand", "first", "second", "third", "fourth",
        "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "once",
        "twice", "double", "treble",
        # fmt: on
    }
)

#: A number written as digits or as a word. Ordinal digit suffixes included,
#: so "3rd" counts and "3rd-minute" counts once.
NUMBER = re.compile(
    r"\b(?:\d+(?:st|nd|rd|th)?|"
    # Longest first, then alphabetical. The second key is not cosmetic: a set
    # iterates in whatever order the hash seed gives it, and a regex whose
    # alternation order changes between runs is a metric that changes between
    # runs.
    + "|".join(sorted(_NUMBER_WORDS, key=lambda word: (-len(word), word)))
    + r")\b",
    re.IGNORECASE,
)

_SMALL = r"(?:\d{1,2}|nil|nought|zero|one|two|three|four|five|six|seven|eight|nine|ten)"

#: A scoreline as it is said or written: "2-1", "two-one", "three nil". Used
#: to subtract the score from a line before asking whether it carries a
#: number, because the score is the one number the system already says and
#: the open question is whether anything else ever does. A one-two played
#: between two players would be caught by this and read as a scoreline; that
#: is rare enough to accept and is noted here rather than defended.
SCORELINE_SAID = re.compile(rf"\b{_SMALL}\s*(?:[-–—]\s*|\s+){_SMALL}\b", re.IGNORECASE)


class Source(StrEnum):
    """Where the lines being scored came from.

    Printed with the table because it changes what the numbers mean: a
    caller-row reading is scoring the form the caller filled in, not the words
    anyone would have heard.
    """

    #: A rephrased trace: the beats carry the phrased line that passed the
    #: gate, which is exactly what would have been spoken.
    PHRASED = "phrased beats that passed the gate"
    #: A run trace with a voice attached: what actually came out.
    SPOKEN = "spoken rows"
    #: A run trace with neither: the caller's own lines, unphrased.
    CALLER = "caller lines, unphrased"


#: Trace voices to seats. The lead calls the action; the colour seat is
#: whatever else spoke.
_SEATS = {"caller": "lead", "analyst": "colour", "lead": "lead", "colour": "colour"}


@dataclass(frozen=True)
class Utterance:
    """One line that would have been heard, on the cursor clock."""

    ts: float
    seat: str
    text: str
    event: str = "none"

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass(frozen=True)
class Refusal:
    """A line the gate struck out. It never reached air, and it is evidence."""

    ts: float
    text: str
    reasons: tuple[str, ...] = ()

    @property
    def tag(self) -> str:
        return self.reasons[0].split(":", 1)[0].strip() if self.reasons else "rejected"


# -- reading a trace ---------------------------------------------------


def lines_of(rows: list[dict[str, Any]]) -> tuple[list[Utterance], Source]:
    """Every line that would be spoken, in cursor order, and where it came from.

    Three sources in preference order, because three kinds of trace exist on
    disk. A rephrased trace's caller beats hold the phrased line the gate
    passed, so those are the words; a run trace that had a voice has
    ``spoken`` rows, which are what came out after the director cut some of
    it; and a run trace with neither leaves only the caller's own lines,
    which are the form rather than the phrasing and are marked as such.
    """
    if rows_of(rows, "phrased"):
        beats = [
            Utterance(
                ts=float(row.get("ts", 0.0)),
                seat=_SEATS.get(str(row.get("voice", "caller")), "lead"),
                text=str(row.get("text", "")).strip(),
                event=str(row.get("event", "none")),
            )
            for row in rows_of(rows, "beat")
        ]
        return _tidy(beats), Source.PHRASED
    spoken = [
        Utterance(
            ts=float(row.get("ts", 0.0)),
            seat=_SEATS.get(str(row.get("voice", "caller")), "lead"),
            text=str(row.get("spoken", "")).strip(),
            event=str(row.get("event", "none")),
        )
        for row in rows_of(rows, "spoken")
    ]
    if _tidy(spoken):
        return _tidy(spoken), Source.SPOKEN
    caller = [
        Utterance(
            ts=float(row.get("ts", 0.0)),
            seat="lead",
            text=str(row.get("line", "")).strip(),
            event=str(row.get("event", "none")),
        )
        for row in rows_of(rows, "caller")
        if row.get("speak", True)
    ]
    return _tidy(caller), Source.CALLER


def _tidy(lines: list[Utterance]) -> list[Utterance]:
    return sorted((line for line in lines if line.text), key=lambda line: line.ts)


def refusals_of(rows: list[dict[str, Any]]) -> list[Refusal]:
    """What the gate struck out on the pass that produced these lines.

    A rephrased trace carries the original run's gate verdicts *and* the
    rephrase pass's own, and only the second lot are about the lines being
    scored here — the first lot judged words that no longer exist in the
    file. ``rephrase`` stamps its own rows with ``where``, so they separate
    cleanly; a trace with no phrased rows has only one pass in it and every
    refusal in it is that pass's.
    """
    gates = rows_of(rows, "gate")
    mine = [row for row in gates if row.get("where") == "rephrase"]
    pool = mine if rows_of(rows, "phrased") else gates
    return [
        Refusal(
            ts=float(row.get("ts", 0.0)),
            text=str(row.get("line", "")).strip(),
            reasons=tuple(str(r) for r in row.get("reasons", [])),
        )
        for row in pool
        if not row.get("passed", False)
    ]


def people_of(rows: list[dict[str, Any]], pack: KnowledgePack | None = None) -> frozenset[str]:
    """Every player's name this trace could legitimately have said.

    Three sources, because no one of them is complete. The pack is the team
    sheet and is the authority when there is one. The state rows carry
    ``on_pitch``, which is whatever the graphics taught the tracker. The
    sighting rows carry what the caller read off a shirt. Without a pack the
    last two are all there is, and they are enough for the two traces this
    was built on — said out loud in the table, because a thin name set
    undercounts the bare-name share rather than overcounting it.
    """
    names: set[str] = set()
    if pack is not None:
        names.update(player.name for sheet in (pack.home, pack.away) for player in sheet.squad)
    for row in rows_of(rows, "state"):
        pitch = row.get("on_pitch")
        if isinstance(pitch, dict):
            names.update(str(v) for v in pitch.values() if v)
        ball = row.get("ball")
        if isinstance(ball, dict) and ball.get("player"):
            names.add(str(ball["player"]))
    for topic in ("sighting", "caller"):
        for row in rows_of(rows, topic):
            for sighting in row.get("sightings", []) or []:
                if isinstance(sighting, dict) and sighting.get("name"):
                    names.add(str(sighting["name"]))
    return frozenset(n.strip() for n in names if n and n.strip())


@dataclass(frozen=True)
class Names:
    """The two word sets a name check needs, which are not the same set.

    ``distinctive`` is what proves a line said somebody's name: "De Paul"
    contributes ``paul`` and not ``de``, because a line containing "de" is
    not a line containing a name and leaving the particle in turns half the
    build-up into a false positive.

    ``whole`` is every word of every name including the particles, and it is
    what decides whether a line is *nothing but* a name. "De Paul." has to
    come out as a bare name — it is one — and it cannot if ``de`` is missing
    from the set each of its words must belong to.
    """

    distinctive: frozenset[str] = frozenset()
    whole: frozenset[str] = frozenset()


def name_words(people: frozenset[str]) -> Names:
    """Fold every known name into the two sets :class:`Names` describes."""
    distinctive: set[str] = set()
    whole: set[str] = set()
    for name in people:
        for word in fold(name).split():
            whole.add(word)
            if len(word) > 2 and word not in _PARTICLES:
                distinctive.add(word)
    return Names(distinctive=frozenset(distinctive), whole=frozenset(whole))


# -- the free layer ----------------------------------------------------


@dataclass(frozen=True)
class EventShape:
    """The same measures again, over the lines about one kind of moment.

    The whole-trace median hides the thing a commentator's register is most
    obviously made of: a shot, a foul and a goal are called differently from
    build-up and from each other. A pass sequence that is all bare surnames
    and a goal call that is a long sentence average out to something that
    looks fine and sounds wrong, and only the split shows it.

    Nothing in :mod:`commentary.grading.register_reference` sets a target per
    event yet. That is the gap the corpus study is expected to close, and it
    is the reason this is a table with no reference column rather than one
    with a guessed one.
    """

    event: str
    lines: int
    median_words: float
    share_le_4: float
    bare_name_share: float
    name_share: float


@dataclass
class Shape:
    """What a trace's commentary measures. No opinions, no model, no cost."""

    source: Source = Source.PHRASED
    lines: int = 0
    colour_lines: int = 0
    median_words: float = 0.0
    share_le_2: float = 0.0
    share_le_4: float = 0.0
    share_ge_9: float = 0.0
    share_ge_16: float = 0.0
    bare_name_share: float = 0.0
    name_share: float = 0.0
    opener_repeat_share: float = 0.0
    median_gap_s: float = 0.0
    share_gap_gt_4: float = 0.0
    share_gap_gt_8: float = 0.0
    number_share: float = 0.0
    number_share_off_score: float = 0.0
    gate_refused_share: float = 0.0
    #: The same measures per kind of moment, commonest first. See
    #: :class:`EventShape`.
    by_event: list[EventShape] = field(default_factory=list)
    #: The lead's lines, in order. Everything above is computed off these
    #: except the cadence, which is the whole broadcast's.
    lead: list[Utterance] = field(default_factory=list)
    colour: list[Utterance] = field(default_factory=list)
    refused: list[Refusal] = field(default_factory=list)
    #: Refusal counts by reason tag, the way ``metrics.gate_table`` does it.
    refusal_reasons: dict[str, int] = field(default_factory=dict)
    #: How many names were available to check a line against, for the reader
    #: who wants to know whether a low bare-name share is the system or the
    #: name set.
    known_people: int = 0
    duration_s: float = 0.0

    @property
    def spoken(self) -> list[Utterance]:
        """Both seats, in cursor order. What a listener would hear."""
        return sorted(self.lead + self.colour, key=lambda line: line.ts)

    def value(self, key: str) -> float:
        got = getattr(self, key)
        return float(got)


def measure(rows: list[dict[str, Any]], pack: KnowledgePack | None = None) -> Shape:
    """Layer (a): the register, counted. Free, deterministic, no model."""
    spoken, source = lines_of(rows)
    lead = [line for line in spoken if line.seat == "lead"]
    colour = [line for line in spoken if line.seat != "lead"]
    people = people_of(rows, pack)
    names = name_words(people)
    refused = refusals_of(rows)

    shape = Shape(
        source=source,
        lines=len(lead),
        colour_lines=len(colour),
        lead=lead,
        colour=colour,
        refused=refused,
        known_people=len(people),
        duration_s=max((line.ts for line in spoken), default=0.0),
    )
    for refusal in refused:
        shape.refusal_reasons[refusal.tag] = shape.refusal_reasons.get(refusal.tag, 0) + 1
    judged = len(lead) + len(refused)
    shape.gate_refused_share = len(refused) / judged if judged else 0.0

    if lead:
        counts = [line.words for line in lead]
        shape.median_words = float(statistics.median(counts))
        shape.share_le_2 = _share(counts, lambda n: n <= 2)
        shape.share_le_4 = _share(counts, lambda n: n <= 4)
        shape.share_ge_9 = _share(counts, lambda n: n >= 9)
        # One club utterance in five is sixteen words or longer (study
        # section 1). The phraser's old cap was sixteen, so this row read
        # zero by construction and nothing said so.
        shape.share_ge_16 = _share(counts, lambda n: n >= 16)
        shape.bare_name_share = _share(lead, lambda line: is_bare_name(line.text, names))
        shape.name_share = _share(lead, lambda line: says_a_name(line.text, names))
        shape.opener_repeat_share = opener_repeat_share(lead)
        shape.number_share = _share(lead, lambda line: bool(NUMBER.search(line.text)))
        shape.number_share_off_score = _share(lead, lambda line: says_a_number_off_score(line.text))
        shape.by_event = by_event(lead, names)

    gaps = [b.ts - a.ts for a, b in zip(spoken, spoken[1:], strict=False)]
    if gaps:
        shape.median_gap_s = float(statistics.median(gaps))
        shape.share_gap_gt_4 = _share(gaps, lambda g: g > GAP_MARKS[0])
        shape.share_gap_gt_8 = _share(gaps, lambda g: g > GAP_MARKS[1])
    return shape


def _share[T](items: list[T], predicate: Any) -> float:
    return sum(1 for item in items if predicate(item)) / len(items) if items else 0.0


def by_event(lead: list[Utterance], names: Names) -> list[EventShape]:
    """The lead's lines split by the kind of moment they were written about.

    Commonest first, then alphabetically, so that two runs of the same clip
    print their rows in the same order and the table can be read as a diff.
    The event is the one the caller's form filed the moment under, which is
    also what the director cut on, so a line's row here is the same row the
    rest of the system thought it was writing.
    """
    grouped: dict[str, list[Utterance]] = {}
    for line in lead:
        grouped.setdefault(line.event, []).append(line)
    shapes = [
        EventShape(
            event=event,
            lines=len(group),
            median_words=float(statistics.median([line.words for line in group])),
            share_le_4=_share(group, lambda line: line.words <= 4),
            bare_name_share=_share(group, lambda line: is_bare_name(line.text, names)),
            name_share=_share(group, lambda line: says_a_name(line.text, names)),
        )
        for event, group in grouped.items()
    ]
    return sorted(shapes, key=lambda shape: (-shape.lines, shape.event))


def is_bare_name(text: str, names: Names) -> bool:
    """Is this line nothing but a name, or a comma-separated pair of them?

    The strict reading, deliberately: every word in the line has to be part
    of somebody's name. "Messi." counts, "De Paul." counts, and "And Messi
    again." does not. Real commentary's 19% was measured the same strict way,
    so a looser rule here would beat a reference it was not compared against.
    """
    got = fold(text).split()
    return bool(got) and all(word in names.whole for word in got)


def says_a_name(text: str, names: Names) -> bool:
    return any(word in names.distinctive for word in fold(text).split())


def says_a_number_off_score(text: str) -> bool:
    """A number that is not the scoreline — the one the pack notes are for.

    The score is subtracted first because the system says it often and by
    design, and counting it would hide the thing actually being asked: does
    anything researched ever reach air.
    """
    return bool(NUMBER.search(SCORELINE_SAID.sub(" ", text)))


def opener_repeat_share(lines: list[Utterance]) -> float:
    """Lines opening on a word one of the last five also opened on.

    The phraser is shown the previous five lines and told not to repeat an
    opener, so a repeat inside this window is one it had in front of it. The
    first line of a trace cannot repeat and is counted in the denominator
    anyway: a five-line clip that opens the same way twice should read worse
    than a fifty-line one that does.
    """
    if not lines:
        return 0.0
    openers = [fold(line.text).split()[0] if fold(line.text).split() else "" for line in lines]
    repeats = sum(
        1
        for i, word in enumerate(openers)
        if word and word in openers[max(0, i - OPENER_WINDOW) : i]
    )
    return repeats / len(openers)


# -- the paid layer ----------------------------------------------------


class Score(BaseModel):
    """One dimension, out of ten, with the reason on the same line."""

    score: float = Field(description="0 to 10. Whole or half numbers only.")
    why: str = Field(
        description="One line, under 20 words. Quote from the trace if a line decides it."
    )


class Worst(BaseModel):
    """One of the three lines that most needs rewriting."""

    ts: float = Field(description="The timestamp of the line, copied from the trace.")
    line: str = Field(description="The line, quoted exactly.")
    why: str = Field(description="One line. What is wrong with it and what it should sound like.")


class RegisterVerdict(BaseModel):
    """The judge's whole answer. Nine scores and the three worst lines.

    ``voice_register`` is the dimension everything else is named after and is
    spelled long here because ``register`` on a Pydantic model shadows an
    attribute of ``BaseModel`` and warns at import. It prints as "register".
    """

    voice_register: Score
    economy: Score
    variety: Score
    event_fit: Score
    goal_call: Score
    build_up: Score
    colour: Score
    invention: Score
    overall: Score
    worst: list[Worst] = Field(description="Exactly three, worst first.")


#: Field, printed name, and one-line gloss for each dimension, in the order
#: they print. The gloss is what the rubric expands; keeping all three in one
#: place is what stops the table, the JSON and the prompt drifting apart.
DIMENSIONS: tuple[tuple[str, str, str], ...] = (
    ("voice_register", "register", "commentator, not description"),
    ("economy", "economy", "says less, not more"),
    ("variety", "variety", "openers and shapes differ"),
    ("event_fit", "event fit", "words match the moment"),
    ("goal_call", "goal call", "name, how, score; fragments"),
    ("build_up", "build-up", "bare names, participles, silence"),
    ("colour", "colour", "a second voice, timed right"),
    ("invention", "invention", "10 = no unsupportable claim"),
    ("overall", "overall", "would it pass as a broadcast"),
)

RUBRIC = """You are grading the commentary an AI system produced over one \
passage of a football match, against how a British television commentator \
actually speaks. You are shown forty things a professional said during live \
play as the reference, then the whole sequence of lines the system produced, \
in order, with timestamps in seconds and the kind of moment each line was \
written about.

Score nine dimensions from 0 to 10. On every one of them 10 is best, \
including invention, where 10 means nothing in the passage reads as a claim \
the pictures could not support.

Anchor the scale here and do not drift off it:

- 9-10: indistinguishable from a real broadcast. Almost nothing scores here.
- 8: a real broadcast with one weak passage.
- 6-7: recognisably commentary, but a listener would notice it is not a person.
- 4-5: the facts are right and the voice is wrong — even lengths, even \
rhythm, description rather than calling.
- 2-3: reads like a caption track or a feature description.
- 0-1: not commentary at all.

Grade harshly. A system that is merely fluent, accurate and evenly paced \
belongs at 4 to 6, not at 7. Do not reward correctness — correctness is the \
gate's job and is assumed. You are grading whether it sounds like the man on \
the television.

The dimensions:

1. register. Does it sound like someone calling a match to a live audience, \
or like someone describing a video? Real commentary is fragmentary, \
present-tense, and often has no finite verb at all. A line that could appear \
under a photograph is a failure of register however accurate it is.

2. economy. Half of live commentary is four words or fewer and about one \
line in five is nothing but a surname. Score the whole passage on whether it \
says less than it could. A line that adds a clause the pictures already \
carried costs marks.

3. variety. Openers, shapes and lengths. Several lines starting on the same \
word, or the same subject-verb-object frame line after line, is what a \
generated passage looks like from the outside. A run of identical shapes is \
worse than any single bad line.

4. event fit. A shot, a save, a foul, a corner and a goal are each called \
differently, and build-up is called differently again. Does each line match \
the kind of moment it is against? A build-up line's words on a shot, or a \
shot's words on a pass, is the failure here.

5. goal call. When the ball goes in a real commentator gives the name, then \
how, then the score, in fragments, and the excitement is in the delivery \
rather than in extra words. Score the goal calls in this passage on that \
shape. If the passage has no goal, score 5 and say so.

6. build-up. The long stretches where nothing is happening. Real build-up is \
bare surnames, participle phrases with no subject ("Played by Molina", \
"Flicked on by Rakitic"), and silence — the commentator says nothing for \
seconds at a time. Score whether this system can be quiet and whether it \
names people rather than narrating shape.

7. colour. Whether anything in the passage reads as a genuine second voice — \
an observation, an opinion, something the pictures do not say — and whether \
it lands in a gap rather than over an action call. If there is no second \
voice at all, score low and say so.

8. invention. Anything that reads like a claim the pictures could not \
support: a score nobody gave it, a card that may not have been shown, an \
outcome asserted before it happened, a statistic from nowhere. 10 is clean. \
Lines shown as refused by the gate did not reach air; mention them but do \
not score them as if they had. When a block labelled "FACTS THE BROADCAST \
WAS GIVEN BEFORE KICKOFF" is present, a number that matches one of those \
facts — or a plainly advanced version of a running count — was researched \
and handed to the system, not invented; score it as invention only if it \
contradicts what that block says.

9. overall. Not an average. What a listener would say about the passage as a \
whole.

Then quote the three worst lines, worst first, with their timestamps copied \
from the trace, and say for each what is wrong and what it should have \
sounded like.

Every justification is one line, under twenty words. No preamble."""


def reference_utterances(n: int = REFERENCE_N) -> list[tuple[str, str]]:
    """``n`` real utterances as ``(kind, text)``, spread across the kinds.

    Round-robin rather than random: the sample has to be the same on every
    run or two judgements a week apart are not comparable, and it has to
    cover goals and fouls and dead balls rather than the bare surnames that
    dominate an unsorted sample of live play.
    """
    picked: list[tuple[str, str]] = []
    depth = 0
    while len(picked) < n and depth < max(len(pool) for pool in EXAMPLES.values()):
        for kind in KINDS:
            pool = EXAMPLES[kind]
            if depth < len(pool):
                picked.append((kind, pool[depth]))
                if len(picked) == n:
                    return picked
        depth += 1
    return picked


def reference_block(n: int = REFERENCE_N) -> str:
    sample = reference_utterances(n)
    rows = "\n".join(f"  [{kind}] {text}" for kind, text in sample)
    return (
        f"{len(sample)} things a professional commentator actually said during "
        "live play, from a broadcaster's own captions. This is the register.\n\n"
        f"{rows}"
    )


def passage_block(shape: Shape, *, name: str = "") -> str:
    """The whole line sequence with its clocks, its events and its refusals."""
    head = [
        f"The passage: {name or 'one trace'}, {shape.duration_s:.0f} seconds of match video.",
        f"Lines are {shape.source.value}. {shape.lines} from the lead, "
        f"{shape.colour_lines} from a second voice.",
        "",
        "Timestamps are seconds from the start of the passage.",
        "",
    ]
    marks: list[tuple[float, str]] = [
        (line.ts, f"  {line.ts:7.1f}  [{line.seat}] [{line.event}]  {line.text}")
        for line in shape.spoken
    ]
    marks += [
        (
            refusal.ts,
            f"  {refusal.ts:7.1f}  [REFUSED by the fact gate, never spoken: "
            f"{'; '.join(refusal.reasons)}]  {refusal.text}",
        )
        for refusal in shape.refused
    ]
    body = [text for _, text in sorted(marks, key=lambda pair: pair[0])]
    return "\n".join(head + body)


def _final_state(rows: list[dict[str, Any]]) -> MatchState | None:
    """The last state row on the trace, if any — the match as it ended."""
    found: MatchState | None = None
    for row in rows_of(rows, Topic.STATE.value):
        payload = {k: v for k, v in row.items() if k not in ("topic", "ts")}
        try:
            found = MatchState.model_validate(payload)
        except Exception:
            continue
    return found


def trace_tallies(rows: list[dict[str, Any]]) -> Tallies:
    """What the match's own state rows say every credited goal was worth.

    Cheap: no model call, just the incidents already sitting in the last
    state row on the trace. Built here rather than inside
    :func:`judge_register` so a caller with no rows at hand — a test, or a
    pass that only ever had the shape — can still ask for a judgement.
    """
    tallies = Tallies()
    state = _final_state(rows)
    if state is not None:
        tallies.see_state(state)
    return tallies


def pack_facts_block(pack: KnowledgePack, tallies: Tallies | None = None) -> str:
    """"FACTS THE BROADCAST WAS GIVEN BEFORE KICKOFF" — the judge's own pack.

    The judge marks a gate-verified statistic as invention because it never
    sees what the pack handed the system before kickoff: "five goals in this
    tournament" reads like a number asserted from nowhere unless something
    tells the judge it was researched. This is that something.

    ``tallies``, when the caller has it cheaply (:func:`trace_tallies` off
    the trace's own rows), advances a ``counts`` note the way the phraser and
    the gate already see it. Without one the notes go in as researched — the
    kickoff-true figure — with a line telling the judge a running count may
    have moved by the time a later line in the passage says it.
    """
    notes: list[Note] = tallies.adjusted(pack.notes) if tallies is not None else list(pack.notes)
    if not notes:
        return ""
    rows = "\n".join(f"  {note.about}: {note.text} ({note.kind})" for note in notes)
    caveat = (
        ""
        if tallies is not None
        else (
            "\n\nThese are the figures at kickoff. A note with a running count — goals, "
            "assists, games scoring — moves as the match does, so a later line may give a "
            "higher number than the one above without inventing anything."
        )
    )
    return (
        "FACTS THE BROADCAST WAS GIVEN BEFORE KICKOFF\n\n"
        "Researched by the production and handed to the system before the match; not "
        "something it worked out or made up. A number in the passage that matches one of "
        "these, or a plainly advanced version of a running count, is a researched fact being "
        "read out, not an invented statistic — score it as invention only if it contradicts "
        "what is here.\n\n"
        f"{rows}"
        f"{caveat}"
    )


async def judge_register(
    shape: Shape,
    backend: LLMBackend,
    *,
    name: str = "",
    model: str = JUDGE_MODEL,
    asker: Asker | None = None,
    reference_n: int = REFERENCE_N,
    pack: KnowledgePack | None = None,
    tallies: Tallies | None = None,
) -> tuple[RegisterVerdict, Usage]:
    """Layer (b): one call, the whole passage, nine scores back.

    One call rather than one per line, because the faults this is looking for
    are properties of the sequence — repeated openers, unvaried shapes, a
    build-up that never goes quiet — and a judge shown one line at a time
    cannot see any of them.

    ``pack`` adds the "FACTS THE BROADCAST WAS GIVEN BEFORE KICKOFF" block
    (:func:`pack_facts_block`) after the reference examples, so the cached
    prefix those examples sit behind is unaffected by whether a pack was
    given. ``tallies`` is the match's own, if the caller has it cheaply
    (:func:`trace_tallies`); left out, the pack's own researched notes go in
    with a caveat instead.

    The usage returned is the difference the call made to the backend's own
    running total, which is where every other spend in this project is
    counted from, so a report's cost and a session's cost cannot disagree.
    """
    if not shape.lead and not shape.colour:
        raise JudgeError("nothing to judge: the trace has no lines")
    blocks = [text_block(reference_block(reference_n), cache=True)]
    if pack is not None:
        facts = pack_facts_block(pack, tallies)
        if facts:
            blocks.append(text_block(facts))
    blocks.append(text_block(passage_block(shape, name=name)))
    question = Question(
        key="register-0000",
        tag="judge_register",
        system=RUBRIC,
        blocks=blocks,
        output_format=RegisterVerdict,
        # Room for adaptive thinking plus nine justifications and three
        # quoted lines. Generous on purpose: a truncated answer fails to
        # parse and wastes the whole call, and unused headroom costs nothing.
        max_tokens=8192,
        effort="high",
    )
    before = backend.total
    answers = await (asker or SequentialAsker(backend, model=model)).ask([question])
    spent = _spent(before, backend.total)
    verdict = answers[0]
    if not isinstance(verdict, RegisterVerdict):
        raise JudgeError(f"judge returned {type(verdict).__name__}, wanted RegisterVerdict")
    return verdict, spent


def _spent(before: Usage, after: Usage) -> Usage:
    """What one call added to a running total."""
    return Usage(
        input_tokens=after.input_tokens - before.input_tokens,
        output_tokens=after.output_tokens - before.output_tokens,
        cache_read_tokens=after.cache_read_tokens - before.cache_read_tokens,
        cache_write_tokens=after.cache_write_tokens - before.cache_write_tokens,
        cost_usd=after.cost_usd - before.cost_usd,
        latency_s=after.latency_s,
    )


# -- the write-up ------------------------------------------------------


@dataclass
class RegisterReport:
    """Both layers, one trace, ready to print and ready to write."""

    name: str
    shape: Shape
    verdict: RegisterVerdict | None = None
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    #: A judgement of this same trace from an earlier run, read back off
    #: disk. See :func:`carried_judgement` for why it is here.
    carried: dict[str, Any] | None = None

    def table(self) -> str:
        out = [f"== {self.name}", f"lines from: {self.shape.source.value}"]
        if not self.shape.known_people:
            out.append("no names available: bare-name and name shares are floors, not measurements")
        out += ["", f"{'counted':<34}{'trace':>8}{'real':>8}{'delta':>8}", "-" * 62]
        for key in ORDER:
            out.append(_row(key, self.shape.value(key), BANDS[key]))
        if self.shape.refusal_reasons:
            counted = sorted(self.shape.refusal_reasons.items())
            reasons = ", ".join(f"{tag} {n}" for tag, n in counted)
            out.append(f"{'  refused because':<34}{reasons}")
        if self.shape.by_event:
            out += [
                "",
                f"{'by event':<16}{'lines':>7}{'median':>8}{'<=4w':>7}{'bare':>7}{'name':>7}",
                "-" * 62,
            ]
            for event in self.shape.by_event:
                out.append(
                    f"{event.event:<16}{event.lines:>7}{event.median_words:>8.1f}"
                    f"{_fmt(event.share_le_4, 'share'):>7}"
                    f"{_fmt(event.bare_name_share, 'share'):>7}"
                    f"{_fmt(event.name_share, 'share'):>7}"
                )
            out.append(
                "no reference per event: the study's per-kind table (section 3) counts "
                "a window around a StatsBomb event, both voices, not a line about one"
            )
        out += ["", f"reference: {REFERENCE}, {SOURCE}"]

        if self.verdict is None:
            out += ["", "no model run (--no-model): the judge's half of this table is missing"]
            if self.carried is not None:
                out.append(
                    f"an earlier judgement of this trace by {self.carried.get('model', '?')} "
                    "is kept in the report and not reprinted here"
                )
            return "\n".join(line.rstrip() for line in out)

        out += ["", f"judged by {self.model}", "-" * 62]
        for key, label, gloss in DIMENSIONS:
            score: Score = getattr(self.verdict, key)
            out.append(f"{label:<12}{_clamp(score.score):>5.1f}  {gloss}")
            out.append(f"{'':<12}{'':>5}  {score.why}")
        out += ["", "the three worst lines"]
        for worst in self.verdict.worst[:3]:
            out.append(f"  {worst.ts:7.1f}  {worst.line}")
            out.append(f"           {worst.why}")
        out += [
            "",
            f"cost ${self.usage.cost_usd:.4f} on {self.model} "
            f"(in {self.usage.input_tokens}, out {self.usage.output_tokens}, "
            f"cache read {self.usage.cache_read_tokens})",
        ]
        return "\n".join(line.rstrip() for line in out)

    def as_dict(self) -> dict[str, Any]:
        shape: dict[str, Any] = {
            "source": self.shape.source.value,
            "known_people": self.shape.known_people,
            "duration_s": round(self.shape.duration_s, 2),
            "refusal_reasons": self.shape.refusal_reasons,
            "by_event": [
                {
                    "event": event.event,
                    "lines": event.lines,
                    "median_words": round(event.median_words, 2),
                    "share_le_4": round(event.share_le_4, 4),
                    "bare_name_share": round(event.bare_name_share, 4),
                    "name_share": round(event.name_share, 4),
                }
                for event in self.shape.by_event
            ],
            "refused": [
                {"ts": r.ts, "line": r.text, "reasons": list(r.reasons)} for r in self.shape.refused
            ],
            "lines": [
                {"ts": round(line.ts, 2), "seat": line.seat, "event": line.event, "text": line.text}
                for line in self.shape.spoken
            ],
        }
        measured: dict[str, Any] = {}
        for key in ORDER:
            band = BANDS[key]
            measured[key] = {
                "trace": round(self.shape.value(key), 4),
                "real": band.value,
                "unit": band.unit,
                "what": band.what,
                "basis": band.basis,
            }
        out: dict[str, Any] = {
            "trace": self.name,
            "reference_source": SOURCE,
            "reference": REFERENCE,
            "reference_is_provisional": False,
            "shape": shape,
            "measured": measured,
            "judge": self.carried,
            # What this invocation spent, which is zero on a carried report.
            # What the judgement itself cost stays inside the judge block and
            # travels with it.
            "usd": round(self.usage.cost_usd, 6),
        }
        if self.verdict is not None:
            out["judge"] = {
                "model": self.model,
                "usd": round(self.usage.cost_usd, 6),
                "scores": {
                    label: {
                        "score": _clamp(getattr(self.verdict, key).score),
                        "why": getattr(self.verdict, key).why,
                    }
                    for key, label, _ in DIMENSIONS
                },
                "worst": [w.model_dump() for w in self.verdict.worst[:3]],
                "tokens": {
                    "in": self.usage.input_tokens,
                    "out": self.usage.output_tokens,
                    "cache_read": self.usage.cache_read_tokens,
                    "cache_write": self.usage.cache_write_tokens,
                },
            }
        return out

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return path


def report_path(trace: Path) -> Path:
    """Beside the trace, named after it. One report per trace, overwritten."""
    return trace.with_suffix(".register.json")


def carried_judgement(path: Path, shape: Shape) -> dict[str, Any] | None:
    """A judgement of this same trace from an earlier run, if there is one.

    The free layer is meant to be run constantly — after every phraser change,
    on every trace, for nothing — and the paid layer perhaps once a week. If a
    free run overwrote the report it would throw away the verdict somebody
    paid for, which makes the cheap command the expensive one to use.

    A trace on disk is finished and never rewritten, so an earlier judgement
    of it is still a judgement of it. The line counts are checked anyway
    before carrying anything forward, because a report whose numbers describe
    a different set of lines is worse than no report.
    """
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    judge = stored.get("judge")
    if not isinstance(judge, dict):
        return None
    measured = stored.get("measured", {})
    counts = (measured.get("lines", {}).get("trace"), measured.get("colour_lines", {}).get("trace"))
    if counts != (shape.lines, shape.colour_lines):
        return None
    # Reports written before the cost moved inside the judge block keep it at
    # the top level, where this run is about to overwrite it with zero.
    return {"usd": stored.get("usd", 0.0), **judge, "carried_forward": True}


def _clamp(score: float) -> float:
    """A judge that answers 11 is answering 10, loudly. Not an error."""
    return max(0.0, min(10.0, float(score)))


def _row(key: str, got: float, band: Band) -> str:
    shown = _fmt(got, band.unit)
    if not band.measured or band.value is None:
        return f"{band.what:<34}{shown:>8}{'—':>8}{'—':>8}"
    delta = got - band.value
    mark = "ok" if abs(delta) <= band.tolerance else ("high" if delta > 0 else "low")
    return (
        f"{band.what:<34}{shown:>8}{_fmt(band.value, band.unit):>8}"
        f"{_fmt(delta, band.unit, signed=True):>8}  {mark}"
    )


def _fmt(value: float, unit: str, *, signed: bool = False) -> str:
    sign = "+" if signed and value >= 0 else ""
    if unit == "share":
        return f"{sign}{value * 100:.0f}%"
    if unit == "s":
        return f"{sign}{value:.1f}s"
    if unit == "count":
        return f"{sign}{value:.0f}"
    return f"{sign}{value:.1f}"


async def score_trace(
    path: Path,
    *,
    pack: KnowledgePack | None = None,
    backend: LLMBackend | None = None,
    model: str = JUDGE_MODEL,
    asker: Asker | None = None,
) -> RegisterReport:
    """One trace, both layers unless ``backend`` is left out. The whole job."""
    rows = read_trace(path)
    shape = measure(rows, pack)
    report = RegisterReport(name=path.name, shape=shape, model=model if backend else "")
    if backend is None:
        report.carried = carried_judgement(report_path(path), shape)
        return report
    report.verdict, report.usage = await judge_register(
        shape,
        backend,
        name=path.name,
        model=model,
        asker=asker,
        pack=pack,
        tallies=trace_tallies(rows) if pack is not None else None,
    )
    return report
