"""A researched number is true at kickoff and the match moves it.

The fault these exist for is on disk: ``runs/rephrased/mbappe-goal`` says "Five
in the tournament now for Mbappé" at 184.5 s, after he has scored twice in the
same clip. The note was right and the match had gone past it.
"""

import pytest

from commentary.gate import FactGate
from commentary.goalfollow import GoalFollowup
from commentary.schemas import (
    CallerLine,
    Event,
    Incident,
    KnowledgePack,
    MatchState,
    Note,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
)
from commentary.tallies import Tallies


@pytest.fixture
def pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="Argentina",
            demonym="Argentine",
            starters=[Player(name="Lionel Messi", number=10)],
        ),
        away=TeamSheet(
            name="France",
            short="France",
            demonym="French",
            starters=[Player(name="Kylian Mbappé", number=10)],
        ),
        notes=[
            Note(
                about="Kylian Mbappé",
                text="five goals in this tournament",
                kind="stat",
                counts="goals",
            ),
            Note(
                about="Lionel Messi",
                text="lost the 2014 World Cup final to Germany",
                kind="storyline",
            ),
        ],
    )


@pytest.fixture
def state() -> MatchState:
    return MatchState(home="Argentina", away="France", home_score=2, away_score=1)


def call(text: str) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.GOAL,
        side=Side.AWAY,
        sightings=[Sighting(number=10, name="Kylian Mbappé", team=Side.AWAY, confidence=0.9)],
        confidence=0.9,
        speak=True,
        line=text,
    )


# -- the arithmetic ------------------------------------------------------


def test_five_becomes_six_after_one_goal_and_seven_after_two(pack: KnowledgePack) -> None:
    note = pack.notes[0]
    tallies = Tallies()
    assert tallies.adjust(note).text == "five goals in this tournament"

    tallies.credit_goal("Mbappé", 82.5)
    assert tallies.adjust(note).text == "six goals in this tournament"

    tallies.credit_goal("Mbappé", 176.7)
    assert tallies.adjust(note).text == "seven goals in this tournament"


def test_the_note_on_the_pack_is_never_rewritten(pack: KnowledgePack) -> None:
    """The pack keeps the kickoff-true figure and its source. Only a copy moves."""
    tallies = Tallies()
    tallies.credit_goal("Mbappé", 82.5)
    moved = tallies.adjust(pack.notes[0])
    assert moved.text == "six goals in this tournament"
    assert pack.notes[0].text == "five goals in this tournament"
    assert moved.source == pack.notes[0].source


def test_a_note_that_counts_nothing_is_untouched(pack: KnowledgePack) -> None:
    """Every pack written before this loads with ``counts`` unset and nothing
    downstream behaves differently."""
    tallies = Tallies()
    tallies.credit_goal("Messi", 30.0)
    assert tallies.adjust(pack.notes[1]) is pack.notes[1]


def test_one_goal_moves_only_the_man_who_scored_it(pack: KnowledgePack) -> None:
    other = Note(about="Lionel Messi", text="five goals in this tournament", counts="goals")
    tallies = Tallies()
    tallies.credit_goal("Mbappé", 82.5)
    assert tallies.adjust(pack.notes[0]).text.startswith("six")
    assert tallies.adjust(other).text.startswith("five")


def test_the_same_goal_reported_twice_counts_once(pack: KnowledgePack) -> None:
    """The board's graphic and the statistician's event are one goal."""
    tallies = Tallies()
    tallies.credit_goal("Kylian Mbappé", 82.5)
    assert tallies.credit_goal("Mbappé", 84.0) is False
    assert tallies.adjust(pack.notes[0]).text.startswith("six")


def test_a_wire_goal_in_the_state_is_counted(pack: KnowledgePack, state: MatchState) -> None:
    """Where the statistician names the scorer, nothing else has to."""
    state.incidents.append(
        Incident(
            event=Event.GOAL,
            side=Side.AWAY,
            player="Kylian Mbappé",
            video_ts=82.5,
            source="wire",
        )
    )
    tallies = Tallies()
    tallies.see_state(state)
    assert tallies.adjust(pack.notes[0]).text.startswith("six")


def test_a_board_goal_with_no_name_moves_nobodys_tally(
    pack: KnowledgePack, state: MatchState
) -> None:
    """The board sees a graphic, never a scorer, so it cannot credit anyone."""
    state.incidents.append(
        Incident(event=Event.GOAL, side=Side.AWAY, player=None, video_ts=82.5, source="board")
    )
    tallies = Tallies()
    tallies.see_state(state)
    assert tallies.adjust(pack.notes[0]).text.startswith("five")


def test_figures_stay_figures_and_words_stay_words() -> None:
    tallies = Tallies()
    tallies.credit_goal("Vardy", 10.0)
    assert (
        tallies.adjust(Note(about="Vardy", text="13 goals this season", counts="goals")).text
        == "14 goals this season"
    )
    assert (
        tallies.adjust(Note(about="Vardy", text="thirteen goals this season", counts="goals")).text
        == "fourteen goals this season"
    )


def test_an_ordinal_comes_back_an_ordinal() -> None:
    tallies = Tallies()
    tallies.credit_goal("Messi", 30.0)
    assert (
        tallies.adjust(Note(about="Messi", text="his 499th Barcelona goal", counts="goals")).text
        == "his 500th Barcelona goal"
    )
    assert (
        tallies.adjust(Note(about="Messi", text="his fourth of the season", counts="goals")).text
        == "his fifth of the season"
    )


def test_a_run_of_games_moves_by_one_however_many_he_scores() -> None:
    """Eleven consecutive games is still eleven if he gets a hat-trick in one."""
    note = Note(about="Vardy", text="ten consecutive games scored in", counts="games_scoring")
    tallies = Tallies()
    tallies.credit_goal("Vardy", 10.0)
    assert tallies.adjust(note).text.startswith("eleven")
    tallies.credit_goal("Vardy", 40.0)
    tallies.credit_goal("Vardy", 70.0)
    assert tallies.adjust(note).text.startswith("eleven")


def test_an_assist_tally_never_moves_on_its_own() -> None:
    """Nothing reports assists yet, and an unmoved tally is the honest answer."""
    note = Note(about="Di María", text="three assists in this tournament", counts="assists")
    tallies = Tallies()
    tallies.credit_goal("Di María", 20.0)
    assert tallies.adjust(note).text.startswith("three")
    tallies.credit_assist("Di María", 20.0)
    assert tallies.adjust(note).text.startswith("four")


# -- and what the gate does with it --------------------------------------


def test_the_gate_takes_the_moved_number_and_refuses_the_stale_one(
    pack: KnowledgePack, state: MatchState
) -> None:
    """The point of the whole exercise: the line and the check agree.

    Before the goal the note says five and five is what may be said. After it
    the note says six, so six passes and five — the number that was true two
    minutes ago and is a wrong number now — is refused.
    """
    gate = FactGate()
    tallies = Tallies()

    before = gate.judge(
        call("Mbappé. Five goals in this tournament."),
        state,
        pack,
        board_changed=True,
        notes=tallies.adjusted(pack.notes),
    )
    assert before.passed, before.reasons

    tallies.credit_goal("Mbappé", 82.5)
    notes = tallies.adjusted(pack.notes)

    moved = gate.judge(
        call("Mbappé. Six goals in this tournament."),
        state,
        pack,
        board_changed=True,
        notes=notes,
    )
    assert moved.passed, moved.reasons

    stale = gate.judge(
        call("Mbappé. Five goals in this tournament."),
        state,
        pack,
        board_changed=True,
        notes=notes,
    )
    assert not stale.passed
    assert any("note_claim" in reason for reason in stale.reasons)


def test_without_the_adjusted_notes_the_gate_behaves_exactly_as_it_did(
    pack: KnowledgePack, state: MatchState
) -> None:
    """``notes`` left out is the pack's own list, which is every older caller."""
    gate = FactGate()
    assert gate.judge(
        call("Mbappé. Five goals in this tournament."), state, pack, board_changed=True
    ).passed
    assert not gate.judge(
        call("Mbappé. Six goals in this tournament."), state, pack, board_changed=True
    ).passed


# -- who gets credited ---------------------------------------------------


def goal_form(side: Side = Side.AWAY, named: bool = False) -> CallerLine:
    """The shape the caller actually filed at the Mbappé penalty: four shirt
    numbers, no names on any of them."""
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.GOAL,
        side=side,
        sightings=[
            Sighting(
                number=10,
                name="Kylian Mbappé" if named else None,
                team=side,
                confidence=0.8,
            ),
            Sighting(number=23, name=None, team=Side.HOME, confidence=0.8),
        ],
        confidence=0.85,
        speak=True,
        line="Mbappé steps up and strikes it — Martínez goes the other way.",
    )


def test_the_roster_names_the_scorer_when_the_picture_could_not(
    pack: KnowledgePack,
) -> None:
    """The fault on the trace: every sighting on the goal form came back a
    shirt number with no name, so the goal was credited to nobody and the
    tally never moved."""
    follow = GoalFollowup()
    follow.arm(82.5, goal_form(), "Mbappé! On the volley! Two-one to Argentina.", pack)
    assert follow.scorer == "Kylian Mbappé"


def test_the_beaten_goalkeeper_is_never_the_scorer(pack: KnowledgePack) -> None:
    """ "Buried past Martínez" names two men and only one of them scored."""
    pack = pack.model_copy(
        update={
            "home": pack.home.model_copy(
                update={
                    "starters": [*pack.home.starters, Player(name="Emiliano Martínez", number=23)]
                }
            )
        }
    )
    follow = GoalFollowup()
    follow.arm(82.5, goal_form(), "Buried past Martínez. Mbappé.", pack)
    assert follow.scorer == "Kylian Mbappé"


def test_a_named_sighting_still_wins(pack: KnowledgePack) -> None:
    """The roster is the last resort, not the first."""
    follow = GoalFollowup()
    follow.arm(82.5, goal_form(named=True), "Mbappé! Two-one.", pack)
    assert follow.scorer == "Kylian Mbappé"


def test_no_pack_and_no_names_credits_nobody() -> None:
    follow = GoalFollowup()
    follow.arm(82.5, goal_form(), "And it is in!", None)
    assert follow.scorer is None
