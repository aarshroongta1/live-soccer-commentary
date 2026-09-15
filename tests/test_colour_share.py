"""The ratio governor, the prompt leak, and the repeat check.

Three things are protected here and they are separate problems that share a
seat.

**The share.** Club football gives the colour voice about 31% of the
utterances and this system was giving it 14% (``runs/rephrased/night1``, 80
colour lines against 505). The first test in this file is the measurement
itself, off the caption files, because ``ColourConfig.colour_share_target``
is a number in a config and a number in a config with nothing behind it is
a guess. The rest hold the three things that loosen when the seat is short:
the rate it is offered at, the lead's build-up cap, and the length of a
turn.

**The leak.** Seventeen of those eighty colour lines said "has been here
before", which was a phrase out of the rules' own worked example, said
about whichever pack was loaded. The prompt tests are the colour seat's
version of the phraser's
``test_no_worked_example_pairs_an_invented_detail_with_a_name_from_the_measured_pack``.

**The repeat.** Taking a phrase out of the prompt fixes that phrase. The
four-gram check is what catches the next one.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from commentary.agents.colour import (
    AT_A_DEAD_BALL,
    IN_BUILD_UP,
    REPEAT_GRAM,
    REPEAT_HISTORY,
    ColourSeat,
    FormAt,
    Material,
    Moment,
    Offer,
    Share,
    colour_pass,
    gap_when_behind,
    judge_utterance,
    may_speak,
    repeats_itself,
)
from commentary.config import CallerConfig, ColourConfig, PredictorConfig
from commentary.gate import FactGate
from commentary.grading import register as reg
from commentary.llm.fake import ScriptedBackend
from commentary.predictor import SpeakPredictor
from commentary.prompts.colour import COLOUR_EXAMPLES, COLOUR_RULES
from commentary.schemas import Event, KnowledgePack, MatchState, Player, Scene, TeamSheet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import rephrase_all as ra  # noqa: E402

# -- 1. where the target came from ------------------------------------------

#: The six club matches and their live windows, from study section 1.
CORPUS: tuple[tuple[str, float, float], ...] = (
    ("pl-liv-mun-2025", 152, 6110),
    ("pl-tot-che-2024", 168, 6150),
    ("lei-mun-2015", 70, 5760),
    ("lei-avl-2015", 15, 6100),
    ("clasico-2017", 282, 6145),
    ("bar-mal-2019", 243, 6000),
)

#: Section 4.1's colour openers, verbatim.
CORPUS_OPENERS = (
    "well",
    "yeah",
    "yes",
    "i think",
    "i mean",
    "you know",
    "absolutely",
    "exactly",
    "for me",
    "listen",
)

#: The four files YouTube wrote ``>>`` speaker-change markers into (4.1).
MARKED = {"pl-liv-mun-2025", "pl-tot-che-2024", "lei-mun-2015", "lei-avl-2015"}

_BRACKET = re.compile(r"\[[^\]]*\]")
_SPEAKER = re.compile(r">>+")
_SENTENCE_END = re.compile(r"[.!?]$")

clips_present = pytest.mark.skipif(
    not all(Path(f"clips/{name}.en.json3").exists() for name, _, _ in CORPUS),
    reason="clips/ holds the broadcast captions and is not in the repository",
)


def _marked_utterances(segments: Any) -> list[tuple[str, bool]]:
    """The study's own reconstruction, with the ``>>`` marker kept.

    ``scripts.build_commentary_examples.utterances`` splits on the marker
    and then throws it away, and which utterance opened a marked turn is
    exactly what is needed to segment the stream into turns. Everything
    else here — the 2 s gap, the sentence-final stop, the bracket strip —
    is that function's rule, and the proof it is the same rule is that the
    counts below reproduce section 1's table.
    """
    out: list[tuple[str, bool]] = []
    start: float | None = None
    parts: list[str] = []
    previous_end = 0.0
    marked = False
    pending = False

    def flush() -> None:
        nonlocal parts, start, marked
        if parts and start is not None:
            out.append((" ".join(parts), marked))
        parts, start, marked = [], None, False

    for segment in segments:
        raw = segment.text
        gap = start is not None and segment.start - previous_end >= 2.0
        if raw.lstrip().startswith(">>"):
            flush()
            pending = True
        elif gap:
            flush()
        for index, piece in enumerate(_SPEAKER.split(raw)):
            if index:
                flush()
                pending = True
            for word in piece.split():
                if start is None:
                    start = segment.start
                    if pending:
                        marked, pending = True, False
                parts.append(word)
                if _SENTENCE_END.search(word):
                    flush()
        previous_end = segment.end
    flush()
    # Bracketed non-speech is stripped and an utterance that was nothing
    # but a bracket is dropped, which is the study's method paragraph.
    cleaned = [(" ".join(_BRACKET.sub(" ", text).split()), m) for text, m in out]
    return [(text, m) for text, m in cleaned if text]


def _is_an_entry(text: str) -> bool:
    low = re.sub(r"[^a-z' ]+", " ", text.lower()).strip()
    return any(low == cue or low.startswith(cue + " ") for cue in CORPUS_OPENERS)


@clips_present
def test_the_target_is_club_footballs_own_lead_to_colour_split() -> None:
    """``ColourConfig.colour_share_target`` is measured, not chosen.

    Two ways, both off the caption files, and they have to agree with each
    other and with the study's own arithmetic (325 entries x 4.4 utterances
    a turn over 4,613 utterances = 31.0%, sections 4.2 and 4.4).
    """
    from commentary.grading.captions import load_json3

    marked_total = marked_colour = 0
    pooled_total = pooled_entries = 0
    for name, low, high in CORPUS:
        segments = [
            s for s in load_json3(Path(f"clips/{name}.en.json3")).segments if low <= s.start <= high
        ]
        utterances = _marked_utterances(segments)
        pooled_total += len(utterances)
        pooled_entries += sum(1 for text, _ in utterances if _is_an_entry(text))
        if name not in MARKED:
            continue
        # A turn runs from one marker to the next and is the colour seat's
        # when its first utterance opens on one of 4.1's cues.
        colour_turn: bool | None = None
        for text, opens_a_turn in utterances:
            if opens_a_turn or colour_turn is None:
                colour_turn = _is_an_entry(text)
            marked_total += 1
            marked_colour += int(colour_turn)

    # The reconstruction is the study's: section 1 counts 6,378 utterances
    # pooled over the six, and this comes back within a handful.
    assert abs(pooled_total - 6378) <= 20

    by_markers = marked_colour / marked_total
    by_entry_rate = (pooled_entries / pooled_total) * 4.4  # mean run, section 4.4
    by_the_study = 325 * 4.4 / 4613  # sections 4.2 and 4.4

    for measurement in (by_markers, by_entry_rate, by_the_study):
        assert 0.28 <= measurement <= 0.34, (by_markers, by_entry_rate, by_the_study)
    assert ColourConfig.colour_share_target == pytest.approx(0.31, abs=0.005)
    # The brief's floor is 30% and the corpus is above it, so the corpus wins.
    assert ColourConfig.colour_share_target >= 0.30


# -- 2. the governor itself -------------------------------------------------


def test_a_thin_window_is_not_evidence_that_the_seat_is_behind() -> None:
    """One lead line does not mean the second voice is 31% short."""
    share = Share()
    share.said(1.0, colour=False)
    share.said(2.0, colour=False)
    assert share.share(3.0) is None
    assert share.stretch(3.0) == 0.0


def test_the_share_is_over_a_window_and_the_stretch_is_the_shortfall() -> None:
    share = Share(window_s=300.0, target=0.31, min_sample=4)
    for ts in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0):
        share.said(ts, colour=False)
    share.said(80.0, colour=True)
    assert share.share(100.0) == pytest.approx(1 / 8)
    assert share.stretch(100.0) == pytest.approx((0.31 - 0.125) / 0.31, abs=1e-6)

    # At the target nothing is loosened at all.
    at_target = Share(target=0.31, min_sample=4)
    for ts in range(10):
        at_target.said(float(ts), colour=ts % 3 == 0)
    assert at_target.share(20.0) == pytest.approx(0.4)
    assert at_target.stretch(20.0) == 0.0

    # And the window forgets. Everything above rolls out past 300 s.
    assert share.share(1000.0) is None


def test_the_offered_rate_shortens_in_proportion_to_the_shortfall() -> None:
    cfg = ColourConfig()
    assert gap_when_behind(0.0, cfg) == pytest.approx(cfg.min_gap_s)
    assert gap_when_behind(1.0, cfg) == pytest.approx(cfg.min_gap_behind_s)
    assert gap_when_behind(0.5, cfg) == pytest.approx((cfg.min_gap_s + cfg.min_gap_behind_s) / 2)
    # Nothing outside 0 to 1 moves it further.
    assert gap_when_behind(4.0, cfg) == pytest.approx(cfg.min_gap_behind_s)


def _a_dead_ball_moment(now: float, *, last_turn_ts: float, stretch: float) -> Moment:
    dead = FormAt(
        ts=now - 1.0,
        scene=Scene.STOPPAGE,
        event=Event.FREE_KICK,
        team="Argentina",
        said="A free kick, just outside the area.",
    )
    return Moment(
        now=now,
        forms=(dead, dead),
        last_big=None,
        lead_lines=6,
        last_turn_ts=last_turn_ts,
        stretch=stretch,
    )


def test_a_seat_at_its_share_still_waits_the_full_build_up_gap() -> None:
    """The governor is a corrective, not a new default."""
    offer = may_speak(_a_dead_ball_moment(120.0, last_turn_ts=100.0, stretch=0.0))
    assert not offer.allowed
    assert "45 s" in offer.reason


def test_a_seat_behind_its_share_is_offered_the_next_dead_ball() -> None:
    """Twenty seconds after its last turn, which the 45 s rate refuses."""
    offer = may_speak(_a_dead_ball_moment(120.0, last_turn_ts=100.0, stretch=1.0))
    assert offer.allowed
    assert offer.situation == AT_A_DEAD_BALL

    # Still not instantly: an offer costs a model call even when it comes
    # back silent, so twelve seconds is the floor.
    too_soon = may_speak(_a_dead_ball_moment(105.0, last_turn_ts=100.0, stretch=1.0))
    assert not too_soon.allowed
    assert "behind its share" in too_soon.reason


def test_the_twelve_seconds_after_a_goal_belong_to_the_lead_at_any_share() -> None:
    """Section 4.3 is not negotiable against a ratio."""
    moment = Moment(
        now=105.0,
        forms=(FormAt(ts=104.0, scene=Scene.REPLAY, event=Event.GOAL, team="France"),),
        last_big=(Event.GOAL, 100.0),
        lead_lines=8,
        lead_lines_since_big=0,
        last_turn_ts=40.0,
        stretch=1.0,
    )
    offer = may_speak(moment)
    assert not offer.allowed
    assert "belong to the lead" in offer.reason


def test_one_turn_per_big_event_survives_the_governor() -> None:
    moment = Moment(
        now=118.0,
        forms=(FormAt(ts=117.0, scene=Scene.REPLAY, event=Event.SHOT, team="France"),),
        last_big=(Event.SHOT, 100.0),
        lead_lines=8,
        last_turn_ts=113.0,
        turns_since_big=1,
        stretch=1.0,
    )
    offer = may_speak(moment)
    assert not offer.allowed
    assert "one turn per big event" in offer.reason


def test_the_leads_build_up_cap_stretches_towards_six_seconds() -> None:
    """Study section 2.3's 4.5 s in build-up, opened to make a hole."""
    caller = CallerConfig()
    predictor = SpeakPredictor(PredictorConfig(), caller)
    assert predictor.cap_for(Event.BUILD_UP, 0.0) == pytest.approx(caller.min_gap_build_up_s)
    assert predictor.cap_for(Event.BUILD_UP, 1.0) == pytest.approx(
        caller.min_gap_build_up_stretched_s
    )
    assert predictor.cap_for(Event.BUILD_UP, 0.5) == pytest.approx(5.25)

    # And only build-up. The box is the lead's and a restart already gives
    # the colour seat five times the rate (section 4.2).
    assert predictor.cap_for(Event.SHOT, 1.0) == pytest.approx(caller.min_gap_attacking_s)
    assert predictor.cap_for(Event.FREE_KICK, 1.0) == pytest.approx(caller.min_gap_dead_ball_s)
    assert predictor.cap_for(None, 1.0) == pytest.approx(caller.min_gap_s)


def test_the_stretched_cap_holds_the_lead_where_a_long_line_earned_the_silence() -> None:
    predictor = SpeakPredictor()
    # A line that took five seconds to say earns 5.8 s, capped at the phase.
    assert predictor.gap_after(5.0, Event.BUILD_UP, 0.0) == pytest.approx(4.5)
    assert predictor.gap_after(5.0, Event.BUILD_UP, 1.0) == pytest.approx(5.8)
    # A fragment still buys only a fragment's silence, governor or not.
    assert predictor.gap_after(0.4, Event.BUILD_UP, 1.0) == pytest.approx(1.5)

    at_target = predictor.decide(
        now_ts=105.0,
        triggers=[],
        last_spoken_ts=100.0,
        last_spoken_seconds=5.0,
        last_event=Event.BUILD_UP,
    )
    behind = predictor.decide(
        now_ts=105.0,
        triggers=[],
        last_spoken_ts=100.0,
        last_spoken_seconds=5.0,
        last_event=Event.BUILD_UP,
        colour_stretch=1.0,
    )
    assert "rate_cap" not in at_target.reason
    assert "rate_cap" in behind.reason


def test_a_turn_may_run_to_four_on_one_item_when_the_seat_is_behind() -> None:
    """Section 4.4's median run is 4; the one-item clamp halves it."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=None)
    one_item = Material(patterns=("France down the left again",))
    offer = Offer(True, IN_BUILD_UP, "the ball is dead")

    assert seat._how_many(offer, one_item, stretch=0.0) == 2
    assert seat._how_many(offer, one_item, stretch=0.6) == 4
    # The lead still wins: room is how many fit before he comes back.
    cramped = Offer(True, IN_BUILD_UP, "the ball is dead", room=2)
    assert seat._how_many(cramped, one_item, stretch=1.0) == 2
    # And the sanctioned goal reaction is one fragment at any share.
    reaction = Offer(True, "after a goal", "the lead has had his follow-up", reaction=True)
    assert seat._how_many(reaction, one_item, stretch=1.0) == 1


def test_the_seat_feeds_its_own_governor_from_both_voices() -> None:
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=None)
    for ts in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        seat.saw_lead_line(ts, "Messi has it.")
    seat.spoke_colour(7.0)
    assert seat.share.counts(10.0) == (6, 1)
    assert seat.moment(10.0).stretch > 0.0


# -- 3. the prompt leak -----------------------------------------------------

#: Every surname on ``clips/pack-argfra-2022.json``, restated here rather
#: than read off disk for the same reason ``tests/test_phraser.py`` does:
#: ``clips/`` is the broadcast footage and is not in the repository.
ARGFRA_2022_SURNAMES = frozenset(
    {
        "Acuña",
        "Almada",
        "Areola",
        "Armani",
        "Camavinga",
        "Coman",
        "Correa",
        "Dembélé",
        "Disasi",
        "Dybala",
        "Fernandez",
        "Fofana",
        "Foyth",
        "Giroud",
        "Griezmann",
        "Guendouzi",
        "Gómez",
        "Hernández",
        "Konaté",
        "Koundé",
        "Lloris",
        "MacAllister",
        "Mandanda",
        "Martínez",
        "María",
        "Mbappé",
        "Messi",
        "Molina",
        "Montiel",
        "Muani",
        "Otamendi",
        "Palacios",
        "Paredes",
        "Paul",
        "Pavard",
        "Pezzella",
        "Rabiot",
        "Rodríguez",
        "Romero",
        "Rulli",
        "Saliba",
        "Tagliafico",
        "Tchouaméni",
        "Thuram",
        "Upamecano",
        "Varane",
        "Veretout",
        "Álvarez",
    }
)

#: The one name from that roster the rules may still carry, and why: it is
#: the record of a line this seat really produced ("France keeping it tight,
#: not rushing things" while Messi had the ball), not a made-up outcome
#: hung on a real player. Pinned so a new example cannot slip in beside it.
ARGFRA_2022_EXEMPT_IN_COLOUR_RULES = {"Messi": 1}

#: The corpus utterances in ``COLOUR_EXAMPLES``, which are exempt because
#: every one of them is a thing a real commentator said about a match this
#: system will never call. Pinned by count so a rules example cannot arrive
#: dressed as one.
COLOUR_EXAMPLE_COUNT = 52


def _surname_counts(text: str, surnames: frozenset[str]) -> dict[str, int]:
    return {
        surname: len(re.findall(r"\b" + re.escape(surname) + r"\b", text))
        for surname in surnames
        if re.search(r"\b" + re.escape(surname) + r"\b", text)
    }


def _rules_text() -> str:
    return COLOUR_RULES.format(min_words=3, max_words=ColourConfig.max_words)


def test_no_colour_worked_example_pairs_a_detail_with_a_name_on_the_measured_pack() -> None:
    assert (
        _surname_counts(_rules_text(), ARGFRA_2022_SURNAMES) == ARGFRA_2022_EXEMPT_IN_COLOUR_RULES
    )


def test_the_verbatim_corpus_utterances_are_exempt_and_pinned_by_count() -> None:
    every = [line for group in COLOUR_EXAMPLES.values() for line in group]
    assert len(every) == COLOUR_EXAMPLE_COUNT
    assert _surname_counts("\n".join(every), ARGFRA_2022_SURNAMES) == {}


def test_the_phrase_the_model_copied_seventeen_times_is_out_of_the_prompt() -> None:
    """17 of 80 pooled colour lines said it; it was a worked example."""
    assert "been here before" not in _rules_text()
    assert "been here before" not in "\n".join(
        line for group in COLOUR_EXAMPLES.values() for line in group
    )


def test_the_rules_write_invented_names_as_a_placeholder() -> None:
    rules = _rules_text()
    assert "<PLAYER>" in rules
    assert "a line that reaches air with a pointed bracket in it is not a line" in rules.lower()


# -- 4. the repeat check ----------------------------------------------------


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            demonym="Argentine",
            starters=[Player(name="Ángel Di María", number=11)],
        ),
        away=TeamSheet(
            name="France",
            short="FRA",
            demonym="French",
            starters=[Player(name="Kylian Mbappé", number=10)],
        ),
    )


def test_a_four_word_run_the_seat_has_already_used_is_refused() -> None:
    said = ["Well, Di María has been here before."]
    assert repeats_itself("Yeah, Mbappé has been here before.", said) == "has been here before"
    assert repeats_itself("Yeah, Mbappé takes the free kicks.", said) == ""


def test_three_shared_words_are_not_a_repeat() -> None:
    """A gram of three would strike out two different sentences."""
    assert REPEAT_GRAM == 4
    assert repeats_itself("That is the finish.", ["That is the pass of the night."]) == ""


def test_an_utterance_shorter_than_the_gram_is_never_a_repeat() -> None:
    assert repeats_itself("Morris is the man.", ["Morris is the man."]) != ""
    assert repeats_itself("What a beauty.", ["What a beauty."]) == ""


def test_the_gate_refuses_a_repeat_as_colour_repeat_and_says_which_words() -> None:
    state = MatchState(home="Argentina", away="France")
    verdict = judge_utterance(
        "Yeah, Mbappé has been here before.",
        state,
        a_pack(),
        FactGate(),
        said_before=["Well, Di María has been here before."],
    )
    assert not verdict.passed
    assert verdict.reasons[0].startswith("colour_repeat:")
    assert "has been here before" in verdict.reasons[0]


def test_a_line_the_seat_has_not_used_still_passes() -> None:
    state = MatchState(home="Argentina", away="France")
    verdict = judge_utterance(
        "Yeah, Mbappé takes the free kicks.",
        state,
        a_pack(),
        FactGate(),
        said_before=["Well, Di María has been here before."],
    )
    assert verdict.passed


def test_the_seat_keeps_its_last_ten_and_no_more() -> None:
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    for n in range(14):
        seat.accept([f"line number {n} about the game"])
    assert len(seat.history) == REPEAT_HISTORY
    assert "line number 0 about the game" not in seat.history
    assert "line number 13 about the game" in seat.history


@pytest.mark.asyncio
async def test_the_offline_pass_refuses_the_second_line_of_a_turn_that_repeats_the_first() -> None:
    """Within one turn, not only across turns: the history updates as it goes."""
    from test_colour import _settings, _trace, a_pack, a_turn, speaking

    backend = speaking(
        a_turn(
            "Well, Molina has been here before.",
            "Yeah, Messi has been here before.",
        )
    )
    out = await colour_pass(
        _trace(), backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    refusals = [
        row
        for row in out.rows
        if row.get("topic") == "gate"
        and row.get("where") == "colour"
        and any("colour_repeat" in reason for reason in row.get("reasons", []))
    ]
    assert refusals, "the second utterance repeated four words and was not refused"
    spoken = [
        str(row["text"])
        for row in out.rows
        if row.get("topic") == "beat" and row.get("voice") == "analyst"
    ]
    assert "Yeah, Messi has been here before." not in spoken


@pytest.mark.asyncio
async def test_every_colour_row_carries_the_share_it_was_offered_against() -> None:
    from test_colour import _settings, _trace, a_pack, a_turn, speaking

    backend = speaking(a_turn("Well, that is a corner they have earned.", "Molina is the man."))
    out = await colour_pass(
        _trace(), backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    rows = [row for row in out.rows if row.get("topic") == "colour"]
    assert rows
    for row in rows:
        assert "share" in row
        assert row["share_target"] == pytest.approx(ColourConfig.colour_share_target)
    counts = out.counts()
    assert counts["colour_share_target"] == pytest.approx(ColourConfig.colour_share_target)
    assert 0.0 <= counts["colour_share"] <= 1.0


# -- 5. the judge, one trace at a time --------------------------------------


def test_reruns_over_one_clip_collapse_to_one_clip() -> None:
    assert ra.clip_of("mbappe") == "mbappe"
    assert ra.clip_of("cadence/mbappe") == "mbappe"
    assert ra.clip_of("trigger/mbappe") == "mbappe"
    assert ra.clip_of("A") == ra.clip_of("r5") == ra.clip_of("c12b") == "dimaria"
    assert ra.clip_of("w1280/e09-offside") == ra.clip_of("e09-offside") == "e09-offside"
    assert ra.clip_of("abl/marks-e01-counter-r2") == "e01-counter"
    # Different cuts stay different: e01b is its own piece of football.
    assert ra.clip_of("e01b-counter") != ra.clip_of("e01-counter")


def _lines(n: int) -> list[reg.Utterance]:
    return [
        reg.Utterance(ts=float(i), seat="lead", text="Messi drives forward again", event="build_up")
        for i in range(n)
    ]


def _pooled(*rows: tuple[str, int]) -> ra.Pooled:
    per_trace = [
        (
            run_key,
            Path(f"runs/rephrased/x/{run_key}.jsonl"),
            reg.Shape(lines=lines, lead=_lines(lines)),
            0.01,
        )
        for run_key, lines in rows
    ]
    total = sum(n for _, n in rows)
    return ra.Pooled(
        shape=reg.Shape(lines=total, lead=_lines(total)),
        per_trace=per_trace,
        total_usd=0.1,
    )


def test_the_judge_sees_the_longest_of_each_distinct_clip() -> None:
    pooled = _pooled(
        ("r2", 17),
        ("A", 25),
        ("mbappe", 26),
        ("cadence/mbappe", 30),
        ("e09-offside", 5),
        ("w1280/e09-offside", 4),
        ("e12-foul", 7),
    )
    picked = ra.judge_sample(pooled, 3)
    assert [run_key for run_key, _clip, _shape in picked] == ["cadence/mbappe", "A", "e12-foul"]
    assert [clip for _run_key, clip, _shape in picked] == ["mbappe", "dimaria", "e12-foul"]

    # Asking for more than there are clips gives every clip once.
    everything = ra.judge_sample(pooled, 99)
    assert len(everything) == 4
    assert ra.judge_sample(pooled, 0) == []


class _Score:
    def __init__(self, score: float, why: str = "because") -> None:
        self.score = score
        self.why = why


class _Worst:
    def __init__(self, line: str) -> None:
        self.ts = 1.0
        self.line = line
        self.why = "flat"


class _Verdict:
    """A judge's answer with none of the model behind it."""

    def __init__(self, score: float, worst: str) -> None:
        for key, _label, _gloss in reg.DIMENSIONS:
            setattr(self, key, _Score(score))
        self.worst = [_Worst(worst)]


def test_the_pooled_report_prints_the_mean_and_the_spread_and_the_bill(tmp_path: Path) -> None:
    """Five separate scores, summarised. Not one score over five passages."""
    judged = [
        ra.Judged(
            "cadence/mbappe", "mbappe", reg.Shape(lines=30), _Verdict(6.0, "Flat line A"), 0.11
        ),
        ra.Judged("A", "dimaria", reg.Shape(lines=25), _Verdict(4.0, "Flat line B"), 0.09),
        ra.Judged("e12-foul", "e12-foul", reg.Shape(lines=7), _Verdict(5.0, "Flat line C"), 0.04),
    ]
    table = ra.judge_table(judged)
    assert "mean" in table and "low" in table and "high" in table
    for label in ("register", "colour", "overall"):
        assert label in table
    # Mean 5.0 across 4.0 and 6.0, not a single pooled number.
    assert re.search(r"overall\s+5\.0\s+4\.0\s+6\.0\s+3", table)
    assert "cadence/mbappe" in table and "0.1100" in table
    assert "Flat line B" in ra.worst_lines(judged)


def test_the_pooled_json_carries_every_traces_score_and_the_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ra, "REPHRASED_ROOT", tmp_path)
    pooled = _pooled(("cadence/mbappe", 30), ("A", 25))
    pooled.shape.colour_lines = 5
    pooled.shape.colour_share = 5 / 60

    judged = [
        ra.Judged("cadence/mbappe", "mbappe", reg.Shape(lines=30), _Verdict(6.0, "Flat A"), 0.11),
        ra.Judged("A", "dimaria", reg.Shape(lines=25), _Verdict(4.0, "Flat B"), 0.09),
    ]

    async def fake_judge_each(*_args: Any, **_kwargs: Any) -> tuple[list[ra.Judged], float]:
        return judged, 0.20

    monkeypatch.setattr(ra, "judge_each", fake_judge_each)

    import asyncio

    asyncio.run(ra.write_pooled_report(pooled, "t", judge=True, timeout_s=1.0, sample=5))

    md = (tmp_path / "t" / "POOLED.md").read_text(encoding="utf-8")
    assert "2 trace(s) pooled over 2 distinct clip(s)" in md
    assert "The judge, one trace at a time (2 of 2 clips)" in md
    assert "Judge spend: $0.2000 over 2 call(s)." in md
    assert "8% of what was said, against 31% in real club football" in md

    payload = json.loads((tmp_path / "t" / "POOLED.json").read_text(encoding="utf-8"))
    block = payload["judge_per_trace"]
    assert block["sample"] == 2
    assert block["usd"] == pytest.approx(0.20)
    assert block["scores"]["overall"] == {"mean": 5.0, "low": 4.0, "high": 6.0}
    assert [t["run_key"] for t in block["traces"]] == ["cadence/mbappe", "A"]
    assert payload["shape"]["source"]


def test_a_report_with_no_judge_run_carries_no_judge_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ra, "REPHRASED_ROOT", tmp_path)
    import asyncio

    asyncio.run(
        ra.write_pooled_report(_pooled(("A", 25)), "t", judge=False, timeout_s=1.0, sample=5)
    )
    payload = json.loads((tmp_path / "t" / "POOLED.json").read_text(encoding="utf-8"))
    assert "judge_per_trace" not in payload
    assert "The judge, one trace at a time" not in (tmp_path / "t" / "POOLED.md").read_text()


# -- 6. the register prints a share, not a count ----------------------------


def test_the_register_measures_the_second_voices_share_against_the_corpus() -> None:
    rows: list[dict[str, Any]] = [{"topic": "phrased", "ts": 0.0, "usd": 0.0}]
    rows += [
        {
            "topic": "beat",
            "ts": float(n),
            "id": f"b{n}",
            "voice": "caller" if n % 4 else "analyst",
            "text": "Messi drives forward with it again",
            "event": "build_up",
        }
        for n in range(1, 13)
    ]
    shape = reg.measure(rows)
    assert (shape.lines, shape.colour_lines) == (9, 3)
    assert shape.colour_share == pytest.approx(0.25)
    assert reg.BANDS["colour_share"].value == pytest.approx(0.31)
    assert "colour_share" in reg.ORDER

    printed = reg.passage_block(shape, name="t")
    assert "25% of what was said, against 31% in real club football" in printed
