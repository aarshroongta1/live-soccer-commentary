"""The thirty seconds after a goal: one shout, then a person talking.

``docs/research/real-commentary-corpus.md`` section 2.4 measured 19 goals and
found the same four-part shape every time: a shout of two to five fragments,
then within four seconds a number folded into a sentence, then within fifteen
seconds a past-tense rebuild of the move naming two or three players, then the
colour voice. Sixty words in thirty seconds.

What this system wrote instead, on
``runs/rephrased/r1-replay/mbappe/file-20260913-185228-phrased.jsonl``:

    82.5  Mbappé! Buried. Two-one to Argentina.
    86.8  Mbappé! The ball back to the centre circle!
    90.8  Mbappé! Six in the tournament.

Three consecutive lines opening on the scorer's name and an exclamation mark.
A listener is being told he has scored three times in eight seconds, and the
third of them is the tally beat written as a caption rather than a sentence.

So beat 1 is the only shout, and this file is the enforcement of that: the
instruction on beats 2 and 3, the re-ask when the answer opens that way
anyway, and the code rewrite when the re-ask does it again — because a beat
dropped is a hole in the loudest half-minute in the match, and the shout is
the only thing wrong with the line.
"""

from __future__ import annotations

import pytest

from commentary.agents.phraser import (
    Phraser,
    opening_shout,
    roster_names,
    unshout,
)
from commentary.config import PhraserConfig
from commentary.goalfollow import REBUILD_BEAT, GoalFollowup
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.phraser import GOAL_BEATS
from commentary.schemas import (
    CallerLine,
    Event,
    KnowledgePack,
    PhrasedLine,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
)

# -- fixtures ---------------------------------------------------------------


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            starters=[Player(name="Nahuel Molina", number=26)],
        ),
        away=TeamSheet(
            name="France",
            short="FRA",
            starters=[Player(name="Kylian Mbappé", number=10)],
            bench=[Player(name="Randal Kolo Muani", number=12)],
        ),
    )


def a_goal_form(named: bool = True) -> CallerLine:
    return CallerLine(
        scene=Scene.LIVE_PLAY,
        event=Event.GOAL,
        side=Side.AWAY,
        sightings=(
            [Sighting(number=10, name="Kylian Mbappé", side=Side.AWAY)]
            if named
            else [Sighting(number=10, side=Side.AWAY)]
        ),
        confidence=0.9,
        speak=True,
        line="France drive into the box and it is in, the volley buried.",
    )


def a_phraser(backend: ScriptedBackend) -> Phraser:
    return Phraser(
        backend,
        config=PhraserConfig(model="claude-haiku-4-5"),
        home="Argentina",
        away="France",
    )


def saying(*lines: PhrasedLine) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("phraser", list(lines))
    return backend


# -- what the beats now ask for ---------------------------------------------


def test_beat_two_forbids_the_call_shape_and_asks_for_a_sentence_length() -> None:
    beat = GOAL_BEATS[2]
    assert "DO NOT OPEN ON A NAME AND AN EXCLAMATION MARK" in beat
    assert "eight to fourteen words" in beat
    assert "No number of any kind on this beat" in beat


def test_beat_three_asks_for_a_whole_sentence_and_not_a_shout() -> None:
    beat = GOAL_BEATS[3]
    assert "A WHOLE SENTENCE, WITH A SUBJECT AND A VERB" in beat
    assert "NOT A SHOUT, AND NOT ON HIS NAME" in beat
    # The caption this beat kept writing, named as the fault it is.
    assert '"Six in\nthe tournament." is a caption' in beat


def test_beat_four_asks_for_the_join_that_real_rebuilds_have() -> None:
    """Two flat sentences is a list of facts; a rebuild is one thing happening."""
    beat = GOAL_BEATS[REBUILD_BEAT]
    assert "THE JOIN IS THE POINT" in beat
    assert "Twelve to twenty-four words" in beat
    assert "Not two full stops." in beat
    assert "PAST TENSE FROM THE FIRST WORD" in beat


# -- the shout, recognised ---------------------------------------------------


def test_a_shout_is_a_name_and_not_any_capitalised_word() -> None:
    names = ["Kylian Mbappé"]
    assert opening_shout("Mbappé! Off the ground!", names) == "Mbappé"
    assert opening_shout("Kylian Mbappé! Away he goes.", names) == "Kylian Mbappé"
    assert opening_shout("Save! The deflection flies wide.", names) is None
    assert opening_shout("Buried! And away he goes.", names) is None
    assert opening_shout("Mbappé wheels away, arms wide.", names) is None
    assert opening_shout("", names) is None


def test_the_roster_is_where_a_nameless_goal_form_finds_its_scorer() -> None:
    """Every sighting on the Mbappé penalty came back with a number and no name."""
    assert "Kylian Mbappé" in roster_names(a_pack())
    assert "Randal Kolo Muani" in roster_names(a_pack()), "the bench counts"
    assert roster_names(None) == []


def test_the_rewrite_keeps_the_name_when_what_follows_is_a_clause() -> None:
    assert unshout("Mbappé! and away he goes to the corner flag.") == (
        "Mbappé, and away he goes to the corner flag."
    )


def test_the_rewrite_drops_the_shout_when_what_follows_is_a_sentence() -> None:
    assert unshout("Mbappé! The keeper sent the wrong way.") == (
        "The keeper sent the wrong way."
    )


def test_a_line_that_is_only_the_shout_is_left_alone() -> None:
    """There is nothing underneath to promote, and a dropped beat is a hole."""
    assert unshout("Mbappé!") == "Mbappé!"


# -- the re-ask, and the rewrite behind it -----------------------------------


@pytest.mark.asyncio
async def test_a_follow_up_beat_that_shouts_the_name_is_asked_again() -> None:
    backend = saying(
        PhrasedLine(line="Mbappé! The ball back to the centre circle!", excitement=0.9),
        PhrasedLine(line="And away he goes, straight to the corner flag.", excitement=0.9),
    )
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(
        a_goal_form(),
        "Argentina 2 France 2",
        goal_beat=2,
        scorer="Kylian Mbappé",
        roster=roster_names(a_pack()),
        followup=GOAL_BEATS[2],
    )

    assert phrased is not None
    assert phrased.line == "And away he goes, straight to the corner flag."
    assert phrased.shout_retry
    assert not phrased.shout_rewritten
    note = "\n".join(
        block["text"] for block in backend.calls[-1].blocks if block.get("type") == "text"
    )
    assert 'THAT OPENED ON "Mbappé!"' in note
    assert "Beat 2 is not a shout" in note


@pytest.mark.asyncio
async def test_a_beat_that_shouts_again_is_rewritten_rather_than_dropped() -> None:
    """The re-ask is asked once. What comes back shouting is fixed in code."""
    backend = saying(PhrasedLine(line="Mbappé! The keeper sent the wrong way.", excitement=0.9))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(
        a_goal_form(),
        "Argentina 2 France 2",
        goal_beat=2,
        scorer="Kylian Mbappé",
        roster=roster_names(a_pack()),
        followup=GOAL_BEATS[2],
    )

    assert phrased is not None
    assert phrased.line == "The keeper sent the wrong way."
    assert phrased.shout_retry and phrased.shout_rewritten


@pytest.mark.asyncio
async def test_the_scorer_is_recognised_even_when_the_form_read_no_name() -> None:
    """The penalty whose every sighting was a shirt number and nothing else."""
    backend = saying(PhrasedLine(line="Mbappé! Six in the tournament.", excitement=0.8))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(
        a_goal_form(named=False),
        "Argentina 2 France 2",
        goal_beat=3,
        scorer="Kylian Mbappé",
        roster=roster_names(a_pack()),
        followup=GOAL_BEATS[3],
    )

    assert phrased is not None
    assert phrased.shout_rewritten
    assert not phrased.line.startswith("Mbappé!")


@pytest.mark.asyncio
async def test_the_goal_call_itself_is_still_shouted() -> None:
    """Beat 1 is the shout. Nothing here touches a line with no beat on it."""
    backend = saying(PhrasedLine(line="Mbappé! Off the ground!", excitement=1.0))
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(
        a_goal_form(),
        "Argentina 2 France 2",
        scorer="Kylian Mbappé",
        roster=roster_names(a_pack()),
    )

    assert phrased is not None
    assert phrased.line == "Mbappé! Off the ground!"
    assert not phrased.shout_retry and not phrased.shout_rewritten


@pytest.mark.asyncio
async def test_the_rebuild_beat_may_still_carry_the_name_first() -> None:
    """Beat 4 is past tense and has never written the shout. It is not checked."""
    backend = saying(
        PhrasedLine(line="Mbappé! It was the ball in, and he buried it.", excitement=0.6)
    )
    phraser = a_phraser(backend)

    phrased = await phraser.phrase(
        a_goal_form(),
        "Argentina 2 France 2",
        goal_beat=REBUILD_BEAT,
        scorer="Kylian Mbappé",
        roster=roster_names(a_pack()),
        followup=GOAL_BEATS[REBUILD_BEAT],
    )

    assert phrased is not None
    assert phrased.line.startswith("Mbappé!")
    assert not phrased.shout_retry


# -- the replay is the rebuild ----------------------------------------------


def a_window() -> GoalFollowup:
    follow = GoalFollowup()
    follow.arm(10.0, a_goal_form(), "Mbappé! Off the ground!")
    return follow


def test_a_replay_inside_the_window_spends_the_rebuild_and_not_the_next_beat() -> None:
    """Beat 4 and a replay line are the same line; the shout and the tally are owed."""
    follow = a_window()
    assert follow.beat(12.0) == 2

    follow.rebuilt(14.0)

    assert follow.rebuilt_by_replay
    assert follow.beat(16.0) == 2, "the celebration is still due"
    follow.said(16.0)
    assert follow.beat(18.0) == 3, "and so is the tally"
    follow.said(18.0)
    assert follow.beat(20.0) is None, "but the rebuild has been said"


def test_the_synthesiser_writes_no_second_rebuild_of_the_same_move() -> None:
    follow = a_window()
    follow.said(12.0)
    follow.said(14.0)
    assert follow.synth_times(14.0, None) == [18.0], "beat 4 is due and would be written"

    follow.rebuilt(15.0)

    assert follow.synth_times(15.0, None) == []


def test_a_new_goal_forgets_the_last_one_s_replay() -> None:
    follow = a_window()
    follow.rebuilt(14.0)

    follow.arm(50.0, a_goal_form(), "Mbappé! Again!")

    assert not follow.rebuilt_by_replay
    assert follow.beat(52.0) == 2
