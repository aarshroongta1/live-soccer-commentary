"""What real commentary measures, as numbers a trace can be held against.

**Every number in this module is provisional.** They are seeded from
``runs/prompt-name/REAL_COMMENTARY.md``, which is one match's auto-captions
(Argentina v France 2022, 1,537 reconstructed utterances) plus three
whisper'd clips, and whose live-play figures come from ten hand-picked
windows totalling 115 utterances. That is a sample of one broadcast pair on
one evening, and the live-play subset was separated from the co-commentator
by reading it. It is enough to say the system talks too much and too evenly;
it is not enough to defend a delta of three points.

They are to be replaced wholesale from ``docs/research/real-commentary-corpus.md``
when that study lands. Replace the values here and nothing else changes:
:mod:`commentary.grading.register` reads this module and never hard-codes a
target of its own.

Where no number has been measured, the band's ``value`` is ``None`` and the
table prints a dash rather than a delta. A missing reference is the honest
state of an unmeasured thing and must not be filled in by guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Where every seeded value in this module came from.
SOURCE = "runs/prompt-name/REAL_COMMENTARY.md"

#: What will replace it. Nothing reads this; it is here so that the next
#: person greps for the corpus study and finds the file that has to change.
SUCCESSOR = "docs/research/real-commentary-corpus.md"


@dataclass(frozen=True)
class Band:
    """One reference number, what it is, and how close counts as close.

    ``tolerance`` is not a confidence interval — nothing here has one. It is
    the distance at which the difference stops being worth arguing about
    given a sample this size, chosen so that the table can mark a row
    ``ok`` without implying the number is precise.
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


#: Keyed by the field name on :class:`~commentary.grading.register.Shape`.
#:
#: The live-play figures are the ones that matter for the lead seat: they
#: exclude the analyst and exclude the long summary asides between passages,
#: which is exactly the voice the phraser is being written for. The
#: whole-match figure is kept only where the live-play window did not measure
#: the same thing, and says so in ``basis``.
BANDS: dict[str, Band] = {
    "median_words": Band(
        value=5.0,
        unit="words",
        what="median words a line",
        tolerance=1.0,
        basis="115 live-play utterances",
    ),
    "share_le_2": Band(
        value=0.24,
        unit="share",
        what="two words or fewer",
        tolerance=0.08,
        basis="115 live-play utterances",
    ),
    "share_le_4": Band(
        value=0.48,
        unit="share",
        what="four words or fewer",
        tolerance=0.10,
        basis="115 live-play utterances",
    ),
    "share_ge_9": Band(
        value=0.26,
        unit="share",
        what="nine words or more",
        tolerance=0.10,
        basis="115 live-play utterances",
    ),
    "bare_name_share": Band(
        value=0.19,
        unit="share",
        what="bare name only",
        tolerance=0.08,
        basis="115 live-play utterances",
    ),
    "name_share": Band(
        value=0.46,
        unit="share",
        what="carries a player's name",
        tolerance=0.12,
        basis="704 of 1,537 whole-match utterances; both voices, so a ceiling "
        "for the lead and a floor for nothing",
    ),
    "median_gap_s": Band(
        value=2.4,
        unit="s",
        what="median gap between lines",
        tolerance=0.8,
        basis="whole match, both voices",
    ),
    "share_gap_gt_4": Band(
        value=0.22,
        unit="share",
        what="gaps over four seconds",
        tolerance=0.10,
        basis="whole match, both voices",
    ),
    # -- not measured yet. The corpus study is expected to settle all four.
    "share_gap_gt_8": Band(
        value=None,
        unit="share",
        what="gaps over eight seconds",
        basis="unmeasured: the caption stream cannot tell a silent commentator "
        "from an ASR that gave up in crowd noise, so a long gap is a ceiling",
    ),
    "opener_repeat_share": Band(
        value=None,
        unit="share",
        what="opener repeats the last five",
        basis="unmeasured",
    ),
    "number_share": Band(
        value=None,
        unit="share",
        what="carries a number",
        basis="unmeasured",
    ),
    "number_share_off_score": Band(
        value=None,
        unit="share",
        what="carries a number, not the score",
        basis="unmeasured; this is the one the pack notes are supposed to move",
    ),
    "lines": Band(value=None, unit="count", what="lines spoken"),
    "colour_lines": Band(value=None, unit="count", what="lines from the second voice"),
    "gate_refused_share": Band(
        value=None,
        unit="share",
        what="refused by the gate",
        basis="not a property of real commentary; a diagnostic, printed beside them",
    ),
}

#: The order the table prints them in. Shape first, then cadence, then the
#: two diagnostics that are ours rather than the broadcast's.
ORDER: tuple[str, ...] = (
    "lines",
    "colour_lines",
    "median_words",
    "share_le_2",
    "share_le_4",
    "share_ge_9",
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
