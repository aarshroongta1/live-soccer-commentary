"""What real commentary measures, as numbers a trace can be held against.

Every number here is now measured off ``docs/research/real-commentary-corpus.md``:
seven broadcast feeds, 12.4 hours of live-window caption, 7,878 reconstructed
utterances, 78,963 words. The default reference is :data:`CLUB` — the six club
matches pooled, 6,378 utterances and 67,993 words (study section 1) — because
the fixture this system is being built for is a league match.

**What changed and why.** The values before this were seeded from
``runs/prompt-name/REAL_COMMENTARY.md``: one broadcast, the 2022 World Cup
final, with its live-play figures taken from ten hand-picked windows totalling
115 utterances. The corpus study measured that same file over the whole match
and found it to be the odd one out — study section 1, "What is different about
British club feeds". Club football puts half again as many words into each
turn (median 8 against 6), says a bare surname a third as often, and runs one
utterance in five to sixteen words or more against one in thirteen. Holding a
league-match system to the final's shape was holding it to the wrong shape, so
the final is kept as a second named reference, :data:`FINAL_2022`, and nothing
is compared against it by default.

Where the study states a number, the ``basis`` names the section it is from.
Four numbers the study does not state — two words or fewer, gaps over eight
seconds, numbers off the scoreline, and a repeated opener — were measured here
by the study's own method (``scripts/build_commentary_examples.utterances``
over each file's live window from section 1, brackets stripped). That
reproduction returns section 1's pooled club row exactly (6,378 utterances,
67,993 words, median 8, mean 10.7, 4.0% one word, 23.3% four or fewer, 47.3%
nine or more, 20.2% sixteen or more) and section 2.2's pooled gap row (median
4.3 s, 53% over four seconds), which is what makes the four extra figures
trustworthy. They are marked ``measured here`` in their ``basis``.

Replace the values here and nothing else changes:
:mod:`commentary.grading.register` reads this module and never hard-codes a
target of its own.

Where no number has been measured, the band's ``value`` is ``None`` and the
table prints a dash rather than a delta. A missing reference is the honest
state of an unmeasured thing and must not be filled in by guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Where every value in this module came from.
SOURCE = "docs/research/real-commentary-corpus.md"


@dataclass(frozen=True)
class Band:
    """One reference number, what it is, and how close counts as close.

    ``tolerance`` is not a confidence interval. It is the distance at which
    the difference stops being worth arguing about, chosen so that the table
    can mark a row ``ok`` without implying the number is precise. The corpus
    is seven whole matches rather than one, so these are tighter than the
    provisional ones were — but the per-file spread in the study is wide (the
    median line runs 7 to 11 words across the six club feeds) and the
    tolerances carry that spread rather than the pooled figure's precision.
    """

    #: The measured value, or ``None`` when nobody has measured it yet.
    value: float | None
    #: Human-readable unit, for the table: ``"share"``, ``"words"``, ``"s"``.
    unit: str
    #: One line: what it counts, over what.
    what: str
    #: How far from ``value`` still reads as the same register.
    tolerance: float = 0.0
    #: Which part of the study it came from, when that matters.
    basis: str = ""

    @property
    def measured(self) -> bool:
        return self.value is not None


#: British club football, pooled: the six club feeds of study section 1,
#: 6,378 utterances over 589 minutes. This is the reference.
#:
#: Keyed by the field name on :class:`~commentary.grading.register.Shape`.
#: The figures are whole-match and both voices, which the study's own tables
#: are: a lead-only cut of a caption file cannot be made without the speaker
#: separation that two of the six files do not carry (section 10.3). So the
#: name and length figures are a ceiling for the lead seat alone, and the
#: ``basis`` says so where it matters.
CLUB: dict[str, Band] = {
    "median_words": Band(
        value=8.0,
        unit="words",
        what="median words a line",
        tolerance=1.5,
        basis="section 1, pooled club row; per file 7 to 11",
    ),
    "share_le_2": Band(
        value=0.095,
        unit="share",
        what="two words or fewer",
        tolerance=0.06,
        basis="measured here by section 1's method; the study states one word "
        "(4.0%) and four or fewer (23.3%), not two",
    ),
    "share_le_4": Band(
        value=0.233,
        unit="share",
        what="four words or fewer",
        tolerance=0.08,
        basis="section 1, pooled club row",
    ),
    "share_ge_9": Band(
        value=0.473,
        unit="share",
        what="nine words or more",
        tolerance=0.10,
        basis="section 1, pooled club row; the system's v3 traces have zero",
    ),
    "share_ge_16": Band(
        value=0.202,
        unit="share",
        what="sixteen words or more",
        tolerance=0.08,
        basis="section 1 and Gap 4: one club utterance in five, which a "
        "sixteen-word cap excluded by construction",
    ),
    "bare_name_share": Band(
        value=0.04,
        unit="share",
        what="bare name only",
        tolerance=0.03,
        basis="section 6a: one line in twenty-five, 1.2-5.1% across the four "
        "aligned matches, fuzzy-matched. This module's stricter predicate "
        "measures 2.1% on the same six files, and the tolerance covers both",
    ),
    "name_share": Band(
        value=0.45,
        unit="share",
        what="carries a player's name",
        tolerance=0.12,
        basis="section 6: 23-56% of utterances carry a roster surname, pooled "
        "46.5% over the four aligned matches; a floor, because auto-captions "
        "mangle surnames (section 10.3)",
    ),
    "opener_repeat_share": Band(
        value=0.125,
        unit="share",
        what="opener repeats the last five",
        tolerance=0.06,
        basis="measured here with this module's own window of five; real "
        "commentary repeats an opener far more than the prompt's ban implies",
    ),
    "median_gap_s": Band(
        value=4.3,
        unit="s",
        what="median gap between lines",
        tolerance=0.8,
        basis="section 2.2, pooled club, utterance to utterance. The handoff's "
        "2.4 s was caption segments, of which there are 1.75 an utterance",
    ),
    "share_gap_gt_4": Band(
        value=0.53,
        unit="share",
        what="gaps over four seconds",
        tolerance=0.10,
        basis="section 2.2, pooled club",
    ),
    "share_gap_gt_8": Band(
        value=0.213,
        unit="share",
        what="gaps over eight seconds",
        tolerance=0.08,
        basis="measured here; the study brackets it, 33% over six seconds and "
        "14% over ten. A ceiling: a caption gap cannot tell a silent "
        "commentator from an ASR that gave up in crowd noise",
    ),
    "number_share": Band(
        value=0.166,
        unit="share",
        what="carries a number",
        tolerance=0.05,
        basis="section 5.1: one utterance in six, 16.1-19.8% per file, about "
        "two a minute. Measured here with this module's regex",
    ),
    "number_share_off_score": Band(
        value=0.155,
        unit="share",
        what="carries a number, not the score",
        tolerance=0.05,
        basis="measured here with this module's regex; section 5.2 puts the "
        "scoreline at 22.9% of number-carrying lines, the rest being the "
        "clock, form, a player's tally and history",
    ),
    "lines": Band(value=None, unit="count", what="lines spoken"),
    "colour_lines": Band(value=None, unit="count", what="lines from the second voice"),
    "colour_share": Band(
        value=0.31,
        unit="share",
        what="share said by the second voice",
        tolerance=0.06,
        basis="section 4, measured three ways and all three at about 31%: "
        "325 colour entries over 4,613 utterances (4.2) at a mean run of "
        "4.4 (4.4) is 31.0%; the four files with >> speaker markers, "
        "segmented into turns and classified by 4.1's openers, give 1,251 "
        "of 3,949, 31.7%; the six-match opener rate of 7.24 per 100 at the "
        "same run length gives 31.8%. The brief's floor is 30%",
    ),
    "gate_refused_share": Band(
        value=None,
        unit="share",
        what="refused by the gate",
        basis="not a property of real commentary; a diagnostic, printed beside them",
    ),
}

#: The 2022 World Cup final, kept because the earlier study measured it and
#: because it is what this system was built to sound like. Nothing compares
#: against it by default. Read it as the control: every length figure here is
#: shorter and every name figure looser than the club row above, and the gap
#: between the two columns is the size of the recalibration.
FINAL_2022: dict[str, Band] = {
    key: band
    for key, band in (
        (
            "median_words",
            Band(6.0, "words", "median words a line", 1.0, "section 1, argfra-dimaria row"),
        ),
        ("share_le_2", Band(0.154, "share", "two words or fewer", 0.06, "measured here")),
        ("share_le_4", Band(0.357, "share", "four words or fewer", 0.08, "section 1")),
        ("share_ge_9", Band(0.307, "share", "nine words or more", 0.10, "section 1")),
        ("share_ge_16", Band(0.079, "share", "sixteen words or more", 0.05, "section 1")),
        (
            "bare_name_share",
            Band(0.078, "share", "bare name only", 0.05, "section 8.3, bare name of two words"),
        ),
        (
            "name_share",
            Band(0.45, "share", "carries a player's name", 0.12, "measured here, 44.9%"),
        ),
        (
            "opener_repeat_share",
            Band(0.128, "share", "opener repeats the last five", 0.06, "measured here"),
        ),
        ("median_gap_s", Band(4.6, "s", "median gap between lines", 0.8, "section 2.1 and 2.2")),
        ("share_gap_gt_4", Band(0.567, "share", "gaps over four seconds", 0.10, "section 2.2")),
        ("share_gap_gt_8", Band(0.229, "share", "gaps over eight seconds", 0.08, "measured here")),
        ("number_share", Band(0.135, "share", "carries a number", 0.05, "section 5.1, 14.6%")),
        (
            "number_share_off_score",
            Band(0.131, "share", "carries a number, not the score", 0.05, "measured here"),
        ),
        ("lines", Band(None, "count", "lines spoken")),
        ("colour_lines", Band(None, "count", "lines from the second voice")),
        (
            "colour_share",
            Band(
                0.31,
                "share",
                "share said by the second voice",
                0.06,
                "club football's figure; the 2022 final is not separately measured "
                "for it and the study's own marker counts put it in the same place",
            ),
        ),
        ("gate_refused_share", Band(None, "share", "refused by the gate")),
    )
}

#: Every reference this module holds, by the name the table prints.
REFERENCES: dict[str, dict[str, Band]] = {
    "club football, six matches": CLUB,
    "2022 World Cup final": FINAL_2022,
}

#: Which of them a trace is held against unless something says otherwise.
REFERENCE = "club football, six matches"

#: The reference itself. Imported by :mod:`commentary.grading.register`.
BANDS: dict[str, Band] = REFERENCES[REFERENCE]

#: The order the table prints them in. Shape first, then cadence, then the
#: two diagnostics that are ours rather than the broadcast's.
ORDER: tuple[str, ...] = (
    "lines",
    "colour_lines",
    "colour_share",
    "median_words",
    "share_le_2",
    "share_le_4",
    "share_ge_9",
    "share_ge_16",
    "bare_name_share",
    "name_share",
    "opener_repeat_share",
    "median_gap_s",
    "share_gap_gt_4",
    "share_gap_gt_8",
    "number_share",
    "number_share_off_score",
    "gate_refused_share",
)
