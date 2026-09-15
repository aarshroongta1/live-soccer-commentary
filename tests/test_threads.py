"""A fact said once is not a thread. These are the ordering that makes one.

``docs/research/real-commentary-corpus.md`` section 7: every match in the
corpus runs three to five threads, each set up early, paid off at the event and
called back one to four times, and each callback restates the same number in a
new form.
"""

import pytest

from commentary.prompts.phraser import phraser_blocks
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    Note,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
)
from commentary.threads import CALLBACK_QUIET_S, REPEAT_QUIET_S, Threads


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
            starters=[
                Player(name="Kylian Mbappé", number=10),
                Player(name="Olivier Giroud", number=9),
            ],
        ),
        notes=[
            Note(
                about="Kylian Mbappé",
                text="five goals in this tournament",
                kind="stat",
                counts="goals",
            ),
            Note(
                about="Kylian Mbappé",
                text="a goal in the 2018 World Cup final at nineteen",
                kind="storyline",
            ),
            Note(about="Olivier Giroud", text="France's all-time leading scorer", kind="storyline"),
            Note(
                about="France",
                text="no side has won back-to-back World Cups since 1962",
                kind="storyline",
            ),
        ],
    )


def texts(threads: Threads, names: list[str], ts: float, **kwargs: object) -> list[str]:
    return [item.note.text for item in threads.offer(names, ts=ts, **kwargs)]  # type: ignore[arg-type]


def build_up(text: str, name: str = "Kylian Mbappé") -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.BUILD_UP,
        side=Side.AWAY,
        sightings=[Sighting(number=10, name=name, team=Side.AWAY, confidence=0.9)],
        confidence=0.9,
        speak=True,
        line=text,
    )


# -- seeding -------------------------------------------------------------


def test_one_entry_a_note_seeded_at_kickoff(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    assert len(threads.entries) == len(pack.notes)
    assert all(entry.times_said == 0 for entry in threads.entries)
    assert all(entry.last_said_ts is None for entry in threads.entries)
    assert [entry.team for entry in threads.entries] == [False, False, False, True]


def test_a_pack_with_no_notes_offers_nothing(pack: KnowledgePack) -> None:
    empty = Threads.from_pack(pack.model_copy(update={"notes": []}))
    assert empty.entries == []
    assert empty.offer(["Kylian Mbappé"], ts=10.0) == []
    assert Threads.from_pack(None).entries == []


# -- the ordering --------------------------------------------------------


def test_a_player_note_beats_a_team_note(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    offered = texts(threads, ["Kylian Mbappé", "France"], 10.0)
    assert offered[0] == "five goals in this tournament"
    assert offered[-1] == "no side has won back-to-back World Cups since 1962"


def test_the_block_is_capped_at_three(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    offered = threads.offer(["Kylian Mbappé", "Olivier Giroud", "France"], ts=10.0)
    assert len(offered) == 3


def test_a_rested_callback_comes_before_an_unused_note(pack: KnowledgePack) -> None:
    """The finding: picking a fact back up beats reaching for a new one."""
    threads = Threads.from_pack(pack)
    threads.entries[1].said(10.0)  # the 2018 final, said early

    offered = texts(threads, ["Kylian Mbappé"], 10.0 + CALLBACK_QUIET_S + 1)
    assert offered[0] == "a goal in the 2018 World Cup final at nineteen"
    assert offered[1] == "five goals in this tournament"


def test_a_note_said_recently_is_not_offered_at_all(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    threads.entries[1].said(10.0)
    offered = texts(threads, ["Kylian Mbappé"], 40.0)
    assert "a goal in the 2018 World Cup final at nineteen" not in offered


def test_nothing_said_twice_comes_back_inside_ten_minutes(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    entry = threads.entries[1]
    entry.said(10.0)
    entry.said(10.0 + CALLBACK_QUIET_S + 1)
    at = entry.last_said_ts or 0.0

    assert entry.note.text not in texts(threads, ["Kylian Mbappé"], at + REPEAT_QUIET_S - 1)
    assert entry.note.text in texts(threads, ["Kylian Mbappé"], at + REPEAT_QUIET_S + 1)


def test_the_name_order_is_kept_inside_a_tier(pack: KnowledgePack) -> None:
    """The man on the ball first, which is the order the caller hands over."""
    threads = Threads.from_pack(pack)
    assert texts(threads, ["Olivier Giroud", "Kylian Mbappé"], 10.0)[0] == (
        "France's all-time leading scorer"
    )


# -- the goal, and the payoff -------------------------------------------


def test_a_goal_puts_the_scorers_running_count_first(pack: KnowledgePack) -> None:
    """Beat 3 is a number about the scorer, and the corpus fills it every time."""
    threads = Threads.from_pack(pack)
    assert texts(threads, ["Kylian Mbappé"], 90.0, payoff=True)[0] == (
        "five goals in this tournament"
    )


def test_the_payoff_restates_a_count_the_goal_has_just_moved(pack: KnowledgePack) -> None:
    """Said at 59 s, said again at 90 s, and it is a different number now.

    The quiet rule would hold this back — half a minute is not a callback — but
    a tally that has moved is new information, which is exactly what section
    7's payoff slot is.
    """
    threads = Threads.from_pack(pack)
    threads.entries[0].said(59.2)
    threads.credit_goal("Mbappé", 82.5)

    assert texts(threads, ["Kylian Mbappé"], 90.0) == [
        "a goal in the 2018 World Cup final at nineteen"
    ]
    offered = threads.offer(["Kylian Mbappé"], ts=90.0, payoff=True)
    assert offered[0].note.text == "six goals in this tournament"
    assert offered[0].callback is True


def test_a_second_goal_moves_it_again(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    threads.credit_goal("Mbappé", 82.5)
    threads.credit_goal("Mbappé", 176.7)
    offered = threads.offer(["Kylian Mbappé"], ts=184.5, payoff=True)
    assert offered[0].note.text == "seven goals in this tournament"


# -- what comes back -----------------------------------------------------


def test_a_line_that_uses_a_note_marks_the_thread(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    used = threads.said("Mbappé steps up. Five goals in this tournament.", ts=59.2, pack=pack)

    assert [item.note.text for item in used] == ["five goals in this tournament"]
    assert threads.entries[0].times_said == 1
    assert threads.entries[0].last_said_ts == 59.2
    assert threads.used() and not threads.recurring()


def test_a_line_that_uses_nothing_marks_nothing(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    assert threads.said("Mbappé, and France arriving in numbers.", ts=30.0, pack=pack) == []
    assert threads.used() == []


def test_the_moved_number_is_what_the_line_is_matched_against(pack: KnowledgePack) -> None:
    """The phraser was shown "six", so "six" is what counts as using the note."""
    threads = Threads.from_pack(pack)
    threads.credit_goal("Mbappé", 82.5)

    assert threads.said("Six in the tournament now for Mbappé.", ts=90.0, pack=pack)
    assert threads.entries[0].times_said == 1


def test_saying_it_twice_makes_it_a_thread(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    threads.said("Mbappé. Five goals in this tournament.", ts=59.2, pack=pack)
    threads.credit_goal("Mbappé", 82.5)
    threads.said("Six in the tournament now for Mbappé.", ts=90.0, pack=pack)

    assert threads.entries[0].times_said == 2
    assert [entry.note.text for entry in threads.recurring()] == ["five goals in this tournament"]


def test_the_gate_checks_against_the_same_list_the_threads_hold(pack: KnowledgePack) -> None:
    threads = Threads.from_pack(pack)
    threads.credit_goal("Mbappé", 82.5)
    assert "six goals in this tournament" in [note.text for note in threads.notes()]


# -- and what the phraser is told ---------------------------------------


def test_a_callback_is_marked_in_the_context_block(pack: KnowledgePack) -> None:
    body = phraser_blocks(
        build_up("Mbappé picks it up on the left"),
        "Argentina 2-1 France",
        [],
        home="Argentina",
        away="France",
        notes=[pack.notes[0]],
        callbacks=[True],
    )[0]["text"]
    assert "[SAID BEFORE]" in body
    assert "same number, new words" in body


def test_an_unsaid_note_carries_no_mark_and_no_rule(pack: KnowledgePack) -> None:
    body = phraser_blocks(
        build_up("Mbappé picks it up on the left"),
        "Argentina 2-1 France",
        [],
        home="Argentina",
        away="France",
        notes=[pack.notes[0]],
        callbacks=[False],
    )[0]["text"]
    assert "SAID BEFORE" not in body
    assert "five goals in this tournament" in body


def test_the_block_is_unchanged_when_nothing_passes_callbacks(pack: KnowledgePack) -> None:
    """Every caller of this from before threads existed still gets what it got."""
    args: dict[str, object] = {
        "home": "Argentina",
        "away": "France",
        "notes": [pack.notes[0]],
    }
    line = build_up("Mbappé picks it up on the left")
    old = phraser_blocks(line, "Argentina 2-1 France", [], **args)[0]["text"]  # type: ignore[arg-type]
    new = phraser_blocks(line, "Argentina 2-1 France", [], callbacks=[], **args)[0]["text"]  # type: ignore[arg-type]
    assert old == new
    assert "SAID BEFORE" not in old


def test_a_line_about_one_man_does_not_mark_another_mans_thread(
    pack: KnowledgePack,
) -> None:
    """Messi and Mbappé both went into that final on five goals.

    Without the subject check, "Mbappé. Five in the tournament." marked both
    threads said and took Messi's off offer for five minutes.
    """
    pack = pack.model_copy(
        update={
            "notes": [
                *pack.notes,
                Note(
                    about="Lionel Messi",
                    text="five goals in this tournament",
                    kind="stat",
                    counts="goals",
                ),
            ]
        }
    )
    threads = Threads.from_pack(pack)
    used = threads.said("Mbappé. Five in the tournament already.", ts=59.2, pack=pack)

    assert [item.subject for item in used] == ["Kylian Mbappé"]
    assert threads.entries[-1].times_said == 0
