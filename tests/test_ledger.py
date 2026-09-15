"""The match ledger: counts this broadcast made for itself, and who may say them.

``docs/research/real-commentary-corpus.md`` section 5.2: after the scoreline
and the clock, the numbers real commentary says are counts — a side's form, a
player's tally, something happening again. "out for a United corner, the second
of the game"; "a seventh corner of the game"; "shown a second yellow card".

What is protected here is not the register. It is that the arithmetic is right,
that the figure the model is shown is the figure the gate holds it to, and that
the same occurrence is never counted twice however many places report it.
"""

from __future__ import annotations

from typing import Any

import pytest

from commentary.agents.colour import ColourSeat, patterns_in, says_a_number
from commentary.config import ColourConfig
from commentary.gate import FactGate, counts_in, facts_used
from commentary.ledger import COUNT_FRESH_S, DROUGHT_S, Fact, Ledger, ordinal
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.phraser import phraser_blocks
from commentary.rephrase import rephrase
from commentary.schemas import (
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    MatchState,
    Note,
    PhrasedLine,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
)


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            demonym="Argentine",
            starters=[
                Player(name="Nicolás Otamendi", number=19),
                Player(name="Emiliano Martínez", number=23),
                Player(name="Lionel Messi", number=10),
            ],
        ),
        away=TeamSheet(
            name="France",
            short="FRA",
            demonym="French",
            starters=[Player(name="Kylian Mbappé", number=10)],
        ),
    )


def a_form(
    event: Event = Event.BUILD_UP,
    *,
    side: Side = Side.AWAY,
    team: str | None = "France",
    names: tuple[str, ...] = (),
    line: str = "",
    scene: Scene = Scene.LIVE_PLAY,
    detail: str | None = None,
) -> CallerLine:
    return CallerLine(
        scene=scene,
        event=event,
        side=side,
        team=team,
        sightings=[Sighting(name=name) for name in names],
        confidence=0.8,
        speak=bool(line),
        line=line,
        detail=detail,
    )


def a_ledger() -> Ledger:
    return Ledger.from_pack(a_pack())


def walk(ledger: Ledger, forms: list[tuple[float, CallerLine]]) -> float:
    for ts, form in forms:
        ledger.saw_form(ts, form)
    return forms[-1][0] if forms else 0.0


def state(**kwargs: Any) -> MatchState:
    return MatchState(home="Argentina", away="France", **kwargs)


# -- counting ----------------------------------------------------------------


def test_a_run_of_forms_carrying_one_corner_is_one_corner() -> None:
    """Occurrences, not forms. The caller files the same corner on four looks.

    Counting forms is what handed the colour seat "5 penalties" off one spot
    kick, and a ledger that did it would put the same number on the lead's
    line where nothing downstream could catch it.
    """
    led = a_ledger()
    walk(led, [(float(n), a_form(Event.CORNER)) for n in range(4)])
    assert led.count("France", "corner") == 1

    walk(led, [(10.0, a_form(Event.BUILD_UP)), (12.0, a_form(Event.CORNER))])
    assert led.count("France", "corner") == 2


def test_a_replay_of_a_corner_is_not_another_corner() -> None:
    """And it does not break the run either: the corner either side is one."""
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.CORNER)),
            (3.0, a_form(Event.CORNER, scene=Scene.REPLAY)),
            (6.0, a_form(Event.CORNER)),
        ],
    )
    assert led.count("France", "corner") == 1


def test_the_counts_are_per_side_and_move_as_the_match_does() -> None:
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.CORNER)),
            (5.0, a_form(Event.FOUL, side=Side.HOME, team="Argentina")),
            (10.0, a_form(Event.CORNER)),
            (15.0, a_form(Event.THROW_IN, side=Side.HOME, team="Argentina")),
        ],
    )
    assert led.count("France", "corner") == 2
    assert led.count("Argentina", "corner") == 0
    assert led.count("Argentina", "foul") == 1
    assert led.count("Argentina", "throw_in") == 1


def test_a_shot_a_save_follows_is_a_shot_on_target() -> None:
    """Credited to the shot, not to the save: the shot's side is the sure one."""
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.SHOT, names=("Kylian Mbappé",))),
            (3.0, a_form(Event.SAVE, side=Side.HOME, team="Argentina")),
        ],
    )
    assert led.count("France", "shot") == 1
    assert led.count("France", "shot_on_target") == 1
    assert led.count("Kylian Mbappé", "shot_on_target") == 1
    assert led.count("Argentina", "shot_on_target") == 0


def test_a_shot_nothing_follows_is_not_on_target() -> None:
    led = a_ledger()
    walk(led, [(0.0, a_form(Event.SHOT)), (30.0, a_form(Event.SAVE))])
    assert led.count("France", "shot") == 1
    assert led.count("France", "shot_on_target") == 0


def test_a_foul_on_the_same_man_twice_is_his_second() -> None:
    led = a_ledger()
    walk(
        led,
        [
            (
                0.0,
                a_form(Event.FOUL, side=Side.HOME, team="Argentina", names=("Nicolás Otamendi",)),
            ),
            (5.0, a_form(Event.BUILD_UP)),
            (
                30.0,
                a_form(Event.FOUL, side=Side.HOME, team="Argentina", names=("Nicolás Otamendi",)),
            ),
        ],
    )
    assert led.count("Nicolás Otamendi", "foul") == 2
    # And under the name a line would actually say.
    assert led.count("Otamendi", "foul") == 2


def test_the_side_flipping_inside_one_event_does_not_make_it_two() -> None:
    """Off the Mbappé trace: the penalty was filed away, away, away, home.

    Two occurrences out of that read "Argentina's first penalty" about a kick
    France were taking — the one class of error this system has never let on
    air, arriving through the one door nothing downstream can check.
    """
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.PENALTY, side=Side.AWAY, team="France")),
            (4.0, a_form(Event.PENALTY, side=Side.AWAY, team="France")),
            (7.0, a_form(Event.PENALTY, side=Side.HOME, team="Argentina")),
        ],
    )
    assert led.count("France", "penalty") == 1
    assert led.count("Argentina", "penalty") == 0


def test_a_name_read_late_in_a_run_belongs_to_the_whole_of_it() -> None:
    """The Mbappé penalty: one look said "steps up", the next one read the name."""
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.SHOT)),
            (2.0, a_form(Event.SHOT, names=("Kylian Mbappé",))),
        ],
    )
    assert led.count("France", "shot") == 1
    assert led.count("Kylian Mbappé", "shot") == 1


# -- firsts, streaks and droughts --------------------------------------------


def test_a_count_is_news_when_it_moves_and_not_a_minute_later() -> None:
    """The first walk of a real trace offered "Second corner" for two minutes."""
    led = a_ledger()
    walk(led, [(0.0, a_form(Event.CORNER))])
    assert any(fact.kind == "corner" for fact in led.facts(5.0, ["France"]))
    assert not [
        fact for fact in led.facts(COUNT_FRESH_S + 10.0, ["France"]) if fact.kind == "corner"
    ]

    walk(led, [(60.0, a_form(Event.BUILD_UP)), (70.0, a_form(Event.CORNER))])
    assert any(fact.count == 2 for fact in led.facts(72.0, ["France"]))
    assert not led.facts(70.0 + COUNT_FRESH_S + 10.0, ["France"])


def test_a_side_taking_the_corners_has_a_run_of_them() -> None:
    """And a run is only its own clause once the other side has had one too.

    Where every corner so far is this side's, "second corner in a row" and
    "second corner" are the same sentence twice, and on the first measured
    walk they took both of the two places the context block has.
    """
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.CORNER)),
            (5.0, a_form(Event.BUILD_UP)),
            (10.0, a_form(Event.CORNER)),
        ],
    )
    assert not [fact for fact in led.facts(11.0, ["France"]) if fact.kind == "corner_run"]

    walk(
        led,
        [
            (40.0, a_form(Event.CORNER, side=Side.HOME, team="Argentina")),
            (60.0, a_form(Event.BUILD_UP)),
            (70.0, a_form(Event.CORNER)),
            (80.0, a_form(Event.BUILD_UP)),
            (90.0, a_form(Event.CORNER)),
        ],
    )
    runs = [fact for fact in led.facts(91.0, ["France"]) if fact.kind == "corner_run"]
    assert runs and runs[0].count == 2
    assert runs[0].text == "Second corner in a row for France."


def test_the_other_side_breaks_the_run() -> None:
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.CORNER)),
            (5.0, a_form(Event.CORNER, side=Side.HOME, team="Argentina")),
        ],
    )
    assert not [fact for fact in led.facts(6.0, ["France"]) if fact.kind == "corner_run"]


def test_a_side_that_has_not_had_a_shot_in_minutes_has_a_drought() -> None:
    led = a_ledger()
    walk(led, [(0.0, a_form(Event.SHOT))])
    assert not [fact for fact in led.facts(60.0, ["France"]) if fact.kind == "shot_drought"]
    late = [fact for fact in led.facts(DROUGHT_S + 60.0, ["France"]) if fact.kind == "shot_drought"]
    assert late and late[0].text == "France have not had a shot in four minutes."
    assert late[0].since_s == pytest.approx(DROUGHT_S + 60.0)


# -- the words ---------------------------------------------------------------


def test_the_clause_is_how_a_commentator_says_it() -> None:
    """The corpus's own shapes. "the second of the game", "a second yellow card"."""
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.CORNER)),
            (5.0, a_form(Event.BUILD_UP)),
            (10.0, a_form(Event.CORNER)),
            (15.0, a_form(Event.BUILD_UP)),
            (20.0, a_form(Event.CORNER)),
            (25.0, a_form(Event.BUILD_UP)),
            (30.0, a_form(Event.CORNER)),
        ],
    )
    said = {fact.kind: fact.text for fact in led.facts(31.0, ["France"])}
    assert said["corner"] == "Fourth corner for France."

    led = a_ledger()
    walk(
        led,
        [(0.0, a_form(Event.CARD, side=Side.HOME, team="Argentina", names=("Nicolás Otamendi",)))],
    )
    player = {fact.kind: fact.text for fact in led.facts(1.0, ["Nicolás Otamendi"])}
    # The team sheet's spelling, never a surname this module guessed at:
    # ``rsplit`` makes "Mac Allister" into "Allister". The phraser shortens it.
    assert player["card"] == "Nicolás Otamendi's first booking."


def test_a_count_is_written_in_words_not_figures() -> None:
    assert ordinal(1) == "first"
    assert ordinal(7) == "seventh"
    assert ordinal(20) == "twentieth"
    assert ordinal(21) == "21st"


# -- one counter, not two ----------------------------------------------------


def test_a_goal_is_counted_once_however_many_places_report_it() -> None:
    """The picture, the board and the goal call are all the same goal."""
    led = a_ledger()
    walk(led, [(40.0, a_form(Event.GOAL, names=("Kylian Mbappé",)))])
    led.see_state(
        state(
            away_score=1,
            incidents=[
                Incident(
                    event=Event.GOAL, side=Side.AWAY, player=None, video_ts=42.0, source="board"
                )
            ],
        )
    )
    led.credit_goal("Kylian Mbappé", 43.0, Side.AWAY)
    assert len([item for item in led.occurrences if item.kind == "goal"]) == 1
    assert led.count("Kylian Mbappé", "goal") == 1


def test_the_goals_come_from_the_tallies_and_are_not_counted_twice() -> None:
    """A note that counts goals has to move by the number the ledger holds."""
    led = a_ledger()
    led.credit_goal("Kylian Mbappé", 40.0, Side.AWAY)
    assert led.goals("Kylian Mbappé") == 1
    assert led.tallies.count("Kylian Mbappé", "goals") == 1
    note = Note(about="Kylian Mbappé", text="five goals in this tournament", counts="goals")
    assert led.tallies.adjust(note).text == "six goals in this tournament"


def test_the_pattern_counter_is_the_ledger() -> None:
    """The colour seat's spell count and the lead's match count are one list."""
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.CORNER, names=("Kylian Mbappé",))),
            (5.0, a_form(Event.BUILD_UP)),
            (10.0, a_form(Event.CORNER, names=("Kylian Mbappé",))),
        ],
    )
    found = patterns_in([], led, -1.0)
    assert any("2 corners on Kylian Mbappé" in text for text in found)
    assert led.count("France", "corner") == 2


def test_a_wire_figure_overrides_what_the_picture_saw() -> None:
    """The hook the ESPN adapter will use, and nothing else does yet."""
    led = a_ledger()
    walk(led, [(0.0, a_form(Event.CORNER)), (10.0, a_form(Event.BUILD_UP))])
    assert led.count("France", "corner") == 1
    led.override("France", "corner", 6)
    assert led.count("France", "corner") == 6


def test_a_wire_event_the_picture_missed_is_still_counted() -> None:
    led = a_ledger()
    led.see_state(
        state(
            named=[
                {
                    "event": Event.FOUL,
                    "side": Side.HOME,
                    "player": "Nicolás Otamendi",
                    "video_ts": 12.0,
                }
            ]
        )
    )
    assert led.count("Nicolás Otamendi", "foul") == 1
    assert led.count("Argentina", "foul") == 1


# -- the gate ----------------------------------------------------------------


@pytest.fixture
def gate() -> FactGate:
    return FactGate()


def judge(gate: FactGate, text: str, ledger: list[Fact], notes: list[Note] | None = None):
    return gate.judge(
        CallerLine(
            scene=Scene.LIVE_PLAY, event=Event.CORNER, confidence=1.0, speak=True, line=text
        ),
        state(),
        a_pack(),
        board_changed=False,
        notes=notes,
        ledger=ledger,
    )


def a_fact(about: str, kind: str, count: int, text: str, *, player: bool = False) -> Fact:
    return Fact(about=about, kind=kind, count=count, text=text, player=player)


def test_a_count_the_match_holds_passes(gate: FactGate) -> None:
    facts = [a_fact("France", "corner", 4, "Fourth corner for France.")]
    assert judge(gate, "Corner. Fourth corner for France.", facts).passed


def test_a_count_the_match_does_not_hold_is_refused(gate: FactGate) -> None:
    facts = [a_fact("France", "corner", 4, "Fourth corner for France.")]
    verdict = judge(gate, "Corner. Fifth corner for France.", facts)
    assert not verdict.passed
    assert any(reason.startswith("ledger_claim") for reason in verdict.reasons)


def test_a_count_about_nobody_is_refused(gate: FactGate) -> None:
    """A statistic attached to nobody is not checkable and is not commentary."""
    facts = [a_fact("France", "corner", 4, "Fourth corner for France.")]
    verdict = judge(gate, "And that is the fourth corner.", facts)
    assert not verdict.passed
    assert any(reason.startswith("ledger_claim") for reason in verdict.reasons)


def test_a_count_with_no_ledger_at_all_is_refused(gate: FactGate) -> None:
    verdict = judge(gate, "Corner. Fourth corner for France.", [])
    assert not verdict.passed


def test_a_players_count_passes_under_the_name_a_line_says(gate: FactGate) -> None:
    facts = [a_fact("Nicolás Otamendi", "foul", 2, "Nicolás Otamendi's second foul.", player=True)]
    assert judge(gate, "Otamendi again. His second foul.", facts).passed


def test_a_side_count_passes_under_the_demonym(gate: FactGate) -> None:
    facts = [a_fact("France", "corner", 3, "Third corner for France.")]
    assert judge(gate, "Third corner, and the French keep coming.", facts).passed


def test_a_number_that_is_not_a_count_is_left_alone(gate: FactGate) -> None:
    """The rule fires on a number in front of the thing it counts, and nowhere else.

    A pattern that fired on every figure in a line would spend the match
    refusing true ones, which is what the position-zero name trim did and
    what it was deleted for.
    """
    for text in (
        "First time from eight yards.",
        "A one-two on the edge.",
        "Second ball, and Messi has it.",
        "Two men over on the far side.",
        "Ten minutes gone.",
    ):
        assert judge(gate, text, []).passed, text


def test_a_note_and_a_count_are_one_check(gate: FactGate) -> None:
    """A clause either source covers passes, and the tag says which found it."""
    note = Note(about="Kylian Mbappé", text="five goals in this tournament")
    assert judge(gate, "Mbappé. Five goals in this tournament.", [], [note]).passed

    facts = [a_fact("Kylian Mbappé", "shot", 3, "Mbappé's third shot.", player=True)]
    assert judge(gate, "Mbappé. His third shot.", facts, [note]).passed

    verdict = judge(gate, "Mbappé. His fourth shot.", facts, [note])
    assert not verdict.passed
    assert any(reason.startswith("ledger_claim") for reason in verdict.reasons)


def test_the_note_rule_still_answers_for_a_note_shaped_claim(gate: FactGate) -> None:
    verdict = judge(gate, "Mbappé, unbeaten in nineteen.", [])
    assert not verdict.passed
    assert any(reason.startswith("note_claim") for reason in verdict.reasons)


def test_a_count_that_reached_air_can_be_found_again() -> None:
    """The row a trace writes when a number the system worked out was said."""
    facts = [
        a_fact("France", "corner", 4, "Fourth corner for France."),
        a_fact("Nicolás Otamendi", "foul", 2, "Otamendi's second foul.", player=True),
    ]
    assert facts_used("Corner. Fourth corner for France.", facts, a_pack()) == [0]
    assert facts_used("Otamendi. His second foul.", facts, a_pack()) == [1]
    assert facts_used("Molina drives forward.", facts, a_pack()) == []


def test_the_count_parser_reads_words_figures_and_ordinals() -> None:
    assert counts_in("Fourth corner") == [(4, "corner")]
    assert counts_in("4th corner") == [(4, "corner")]
    assert counts_in("4 corners") == [(4, "corner")]
    assert counts_in("two bookings") == [(2, "card")]
    assert counts_in("first shot on target") == [(1, "shot_on_target")]


# -- the colour seat ---------------------------------------------------------


def a_seat(ledger: Ledger) -> ColourSeat:
    return ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack(), ledger=ledger)


def test_a_count_about_a_man_the_lead_named_is_material() -> None:
    """A second foul on one man, half a match after the first.

    Spread out on purpose: inside one spell the pattern counter already says
    "2 fouls on Otamendi" off the same occurrences, and a fact the patterns
    are already making is not offered twice.
    """
    led = a_ledger()
    otamendi = a_form(Event.FOUL, side=Side.HOME, team="Argentina", names=("Nicolás Otamendi",))
    walk(led, [(0.0, otamendi), (5.0, a_form(Event.BUILD_UP))])
    seat = a_seat(led)
    seat.answered(100.0)
    led.saw_form(200.0, otamendi)
    seat.saw_lead_line(200.5, "Otamendi again.")

    material = seat.material(201.0)
    assert any(fact.kind == "foul" and fact.count == 2 for fact in material.ledger)
    # Shown without the figure, because this seat may not say one.
    row = next(row for row in material.lines() if "Otamendi" in row)
    assert row == "REPEATED: another foul from Nicolás Otamendi"
    assert not says_a_number(row)
    assert material


def test_a_count_the_patterns_already_make_is_not_offered_twice() -> None:
    led = a_ledger()
    otamendi = a_form(Event.FOUL, side=Side.HOME, team="Argentina", names=("Nicolás Otamendi",))
    walk(led, [(0.0, otamendi), (5.0, a_form(Event.BUILD_UP)), (10.0, otamendi)])
    seat = a_seat(led)
    seat.saw_lead_line(11.0, "Otamendi again.")
    material = seat.material(12.0)
    assert any("Otamendi" in text for text in material.patterns)
    assert not [fact for fact in material.ledger if fact.player]
    # The side's own count is a different fact and survives.
    assert [fact for fact in material.ledger if fact.about == "Argentina"]


def test_a_first_is_never_this_seats_to_say() -> None:
    """A count of one has no number-free form, so it is not offered at all."""
    led = a_ledger()
    walk(
        led, [(0.0, a_form(Event.FOUL, side=Side.HOME, team="Argentina", names=("Lionel Messi",)))]
    )
    seat = a_seat(led)
    seat.saw_lead_line(1.0, "Messi gives it away.")
    assert not seat.material(2.0).ledger


def test_a_side_count_that_has_not_moved_since_the_last_turn_is_not_material() -> None:
    """A number said at four and said again at four is the weather."""
    led = a_ledger()
    walk(
        led,
        [
            (0.0, a_form(Event.CORNER)),
            (5.0, a_form(Event.BUILD_UP)),
            (10.0, a_form(Event.CORNER)),
        ],
    )
    seat = a_seat(led)
    seat.answered(11.0)
    assert not seat.material(12.0).ledger

    led.saw_form(20.0, a_form(Event.BUILD_UP))
    led.saw_form(25.0, a_form(Event.CORNER))
    assert any(fact.count == 3 for fact in seat.material(26.0).ledger)


def test_a_first_of_something_is_the_leads_to_say_and_not_this_seats() -> None:
    led = a_ledger()
    walk(led, [(0.0, a_form(Event.CORNER))])
    seat = a_seat(led)
    seat.answered(0.5)
    assert not seat.material(1.0).ledger


# -- the prompt --------------------------------------------------------------


def test_the_context_block_carries_the_counts() -> None:
    blocks = phraser_blocks(
        a_form(Event.CORNER, line="Corner."),
        "Argentina 0-0 France",
        [],
        home="Argentina",
        away="France",
        ledger=[a_fact("France", "corner", 4, "Fourth corner for France.")],
    )
    body = "\n".join(block["text"] for block in blocks if block.get("type") == "text")
    assert "ledger:" in body
    assert "Fourth corner for France." in body


def test_a_goal_is_no_moment_for_a_count() -> None:
    blocks = phraser_blocks(
        a_form(Event.GOAL, line="It's in!"),
        "Argentina 0-1 France",
        [],
        home="Argentina",
        away="France",
        ledger=[a_fact("France", "corner", 4, "Fourth corner for France.")],
    )
    body = "\n".join(block["text"] for block in blocks if block.get("type") == "text")
    assert "Fourth corner for France." not in body
    assert "ledger:\n  (none" in body


# -- the offline pass --------------------------------------------------------


def a_corner_trace() -> list[dict[str, Any]]:
    """Two corners for France, each called, with the state rows around them."""
    rows: list[dict[str, Any]] = [
        {
            "topic": "state",
            "ts": 0.5,
            "home": "Argentina",
            "away": "France",
            "home_score": 0,
            "away_score": 0,
            "clock": "20:00",
            "period": 1,
        }
    ]
    for index, ts in enumerate((10.0, 40.0), start=1):
        rows += [
            {
                "topic": "caller",
                "ts": ts,
                "scene": "live_play",
                "event": "corner",
                "side": "away",
                "team": "France",
                "sightings": [],
                "confidence": 0.8,
                "speak": True,
                "line": "France have a corner on the right.",
            },
            {
                "topic": "gate",
                "ts": ts,
                "passed": True,
                "reasons": [],
                "line": "France have a corner on the right.",
                "event": "corner",
            },
            {
                "topic": "beat",
                "ts": ts,
                "id": f"c{index}",
                "voice": "caller",
                "text": "France have a corner on the right.",
                "video_ts": ts,
                "created_ts": float(index),
                "live_ts": ts + 3.0,
                "event": "corner",
                "preemptable": True,
            },
        ]
    return rows


@pytest.mark.asyncio
async def test_the_rephrase_builds_the_ledger_as_it_walks() -> None:
    """The count offered on the second corner is two, not two both times.

    Progressively, the way :class:`commentary.threads.Threads` and the goal
    follow-up are built: a pass that counted the whole trace up front would
    offer the final figure on the first line of the match.
    """
    backend = ScriptedBackend()
    backend.queue(
        "phraser",
        [
            PhrasedLine(line="Corner, France.", excitement=0.2),
            PhrasedLine(line="Second corner for France.", excitement=0.2),
        ],
    )
    result = await rephrase(a_corner_trace(), backend, pack=a_pack(), colour=False)

    offered = [
        row
        for row in result.rows
        if row.get("topic") == "ledger" and row.get("action") == "offered"
    ]
    assert [row["count"] for row in offered if row["kind"] == "corner"] == [1, 2]

    used = [row for row in result.rows if row.get("topic") == "ledger" and row["action"] == "used"]
    assert [row["text"] for row in used] == ["Second corner for France."]
    assert [row["text"] for row in result.rows if row.get("topic") == "beat"][-1] == (
        "Second corner for France."
    )


@pytest.mark.asyncio
async def test_a_count_the_walk_does_not_hold_is_refused_in_the_rephrase() -> None:
    backend = ScriptedBackend()
    backend.queue(
        "phraser",
        [
            PhrasedLine(line="Corner, France.", excitement=0.2),
            PhrasedLine(line="Sixth corner for France.", excitement=0.2),
        ],
    )
    result = await rephrase(a_corner_trace(), backend, pack=a_pack(), colour=False)
    refused = [
        row
        for row in result.rows
        if row.get("topic") == "gate" and not row.get("passed") and row.get("where") == "rephrase"
    ]
    assert refused
    assert any("ledger_claim" in reason for reason in refused[0]["reasons"])
