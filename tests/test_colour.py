"""The colour seat: when it is offered a turn, how the turn is spaced, and
what happens to it on the way to air.

What is protected here is the timing, because the timing is the whole change.
A prompt cannot be unit-tested into sounding like a second voice and what it
sounds like is in ``runs/rephrased/``; what can be tested is that the seat is
silent in the twelve seconds after a goal that the corpus gives to the lead,
that a turn is a run rather than a line, that two voices never land on top of
each other, and that every utterance goes through the same fact gate a
phrased caller line goes through.

Every number asserted below is cited to a section of
``docs/research/real-commentary-corpus.md`` in the assertion or the name.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from commentary.agents.colour import (
    AFTER_A_CHANCE,
    AFTER_A_GOAL,
    AT_A_DEAD_BALL,
    IN_BUILD_UP,
    OVER_A_REPLAY,
    ColourSeat,
    FormAt,
    Material,
    Moment,
    Offer,
    colour_pass,
    is_filler,
    may_speak,
    one_subject,
    patterns_in,
    says_a_number,
    says_nothing,
    space_out,
    speaking_for,
    swap_cue,
)
from commentary.config import ColourConfig, PredictorConfig, Settings
from commentary.llm.base import Block
from commentary.llm.fake import ScriptedBackend
from commentary.prompts.colour import (
    COLOUR_EXAMPLES,
    colour_blocks,
    colour_system,
    opens_with_a_cue,
)
from commentary.schemas import (
    Angle,
    CallerLine,
    ColourTurn,
    Event,
    KnowledgePack,
    Note,
    Player,
    Scene,
    Side,
    Sighting,
    TeamSheet,
)
from commentary.tallies import Tallies

# -- fixtures ----------------------------------------------------------------


def a_pack() -> KnowledgePack:
    return KnowledgePack(
        home=TeamSheet(
            name="Argentina",
            short="ARG",
            demonym="Argentine",
            starters=[
                Player(name="Nahuel Molina", number=26),
                Player(name="Lionel Messi", number=10),
                Player(name="Nicolás Otamendi", number=19),
                Player(name="Emiliano Martínez", number=23),
            ],
        ),
        away=TeamSheet(
            name="France",
            short="FRA",
            demonym="French",
            starters=[Player(name="Kylian Mbappé", number=10)],
        ),
        notes=[Note(about="Lionel Messi", text="five in this tournament", kind="stat")],
    )


def a_form(
    ts: float,
    *,
    scene: Scene = Scene.LIVE_PLAY,
    event: Event = Event.BUILD_UP,
    line: str = "Argentina keep it.",
    names: tuple[str, ...] = (),
) -> FormAt:
    return FormAt(ts=ts, scene=scene, event=event, team="Argentina", names=names, said=line)


def a_moment(
    now: float,
    *,
    forms: tuple[FormAt, ...] = (),
    last_big: tuple[Event, float] | None = None,
    lead_lines: int = 6,
    lead_lines_since_big: int = 0,
    last_turn_ts: float | None = None,
    turns_since_big: int = 0,
) -> Moment:
    return Moment(
        now=now,
        forms=forms or (a_form(now - 4.0), a_form(now - 1.0)),
        last_big=last_big,
        lead_lines=lead_lines,
        lead_lines_since_big=lead_lines_since_big,
        last_turn_ts=last_turn_ts,
        turns_since_big=turns_since_big,
    )


def a_turn(*utterances: str, angle: Angle = Angle.TACTICS) -> ColourTurn:
    return ColourTurn(
        angle=angle,
        cites=["the forms since the last turn"],
        speak=bool(utterances),
        utterances=list(utterances),
    )


def speaking(*turns: ColourTurn) -> ScriptedBackend:
    backend = ScriptedBackend()
    backend.queue("colour", list(turns))
    return backend


def text_of(blocks: list[Block]) -> str:
    return "\n".join(b["text"] for b in blocks if b.get("type") == "text")


def a_seat(backend: Any, config: ColourConfig | None = None) -> ColourSeat:
    """A seat with material in front of it: a named man and a repeated corner.

    Since the material gate, a seat that has heard nothing is a seat that
    cannot be asked — which is the point of it and is a nuisance in a fixture.
    This gives it the two cheapest kinds: a note about somebody the lead has
    just named, and two separate corners.
    """
    seat = ColourSeat(backend, config=config or ColourConfig(), pack=a_pack())
    seat.saw_lead_line(1.0, "Messi has it.")
    seat.saw_lead_line(2.0, "Molina overlaps.")
    seat.saw_form(1.0, a_caller(Event.CORNER, "Corner to Argentina."))
    seat.saw_form(2.0, a_caller(Event.BUILD_UP, "Worked short."))
    seat.saw_form(3.0, a_caller(Event.CORNER, "And another corner."))
    return seat


def a_caller(
    event: Event,
    line: str,
    *,
    scene: Scene = Scene.STOPPAGE,
    names: tuple[str, ...] = (),
) -> CallerLine:
    return CallerLine(
        scene=scene,
        event=event,
        side=Side.HOME,
        team="Argentina",
        sightings=[Sighting(name=name) for name in names],
        confidence=0.9,
        speak=True,
        line=line,
    )


# -- the phase gate ----------------------------------------------------------


def test_the_seat_does_not_open_before_the_lead_has_said_anything() -> None:
    """Section 4.6: the colour voice comes in after the lead, never ahead of him."""
    assert not may_speak(a_moment(30.0, lead_lines=0)).allowed
    assert not may_speak(a_moment(30.0, lead_lines=1)).allowed
    assert may_speak(a_moment(30.0, lead_lines=2)).allowed


@pytest.mark.parametrize("event", [Event.GOAL, Event.SHOT, Event.SAVE, Event.PENALTY])
@pytest.mark.parametrize("delay", [0.0, 1.0, 3.9, 9.0, 11.9])
def test_the_first_twelve_seconds_after_a_big_event_belong_to_the_lead(
    event: Event, delay: float
) -> None:
    """Section 4.3: median delay 21.4 s, and only 7% of colour entries inside 6 s.

    The goal's 4-to-8-second reaction window is the one exception and is
    tested on its own below; every other instant inside twelve seconds is a
    no, whatever the picture is doing.
    """
    moment = a_moment(
        100.0 + delay,
        forms=(a_form(100.0, event=event), a_form(100.0 + delay, scene=Scene.CLOSE_UP)),
        last_big=(event, 100.0),
        lead_lines_since_big=2,
    )
    offer = may_speak(moment)
    if event is Event.GOAL and 4.0 <= delay <= 8.0:
        assert offer.allowed
    else:
        assert not offer.allowed
        assert "belong to the lead" in offer.reason


def test_a_goal_reaction_is_allowed_between_four_and_eight_seconds() -> None:
    """Section 4.3: 31% of colour entries after a goal land inside six seconds.

    All four the study prints are reaction fragments — "WELL, it's the first
    goal of the game", "Well, well, well." — so the window is short, it is a
    goal only, and it is one turn.
    """
    offer = may_speak(a_moment(106.0, last_big=(Event.GOAL, 100.0), lead_lines_since_big=1))
    assert offer.allowed
    assert offer.situation == AFTER_A_GOAL
    assert "reaction" in offer.reason


def test_the_goal_reaction_waits_for_the_lead_to_have_his_follow_up() -> None:
    """Section 8.4: the score and the tally are slots 4 and 5; colour is slot 7."""
    assert not may_speak(
        a_moment(106.0, last_big=(Event.GOAL, 100.0), lead_lines_since_big=0)
    ).allowed


def test_the_goal_reaction_happens_once() -> None:
    assert not may_speak(
        a_moment(
            107.0,
            last_big=(Event.GOAL, 100.0),
            lead_lines_since_big=2,
            turns_since_big=1,
            last_turn_ts=105.0,
        )
    ).allowed


def test_a_card_does_not_hold_the_lead_s_window() -> None:
    """The brief's list is goal, shot, save, penalty. A card is a stoppage."""
    moment = a_moment(
        103.0,
        forms=(
            a_form(100.0, event=Event.CARD, scene=Scene.CLOSE_UP),
            a_form(102.0, event=Event.CARD),
        ),
        last_big=(Event.CARD, 100.0),
        lead_lines_since_big=1,
    )
    assert may_speak(moment).allowed


def test_a_dead_ball_is_where_the_seat_speaks() -> None:
    """Section 4.2: 10.5 colour entries per 100 utterances at a dead ball."""
    for event in (Event.CORNER, Event.FREE_KICK, Event.THROW_IN, Event.KICKOFF):
        moment = a_moment(40.0, forms=(a_form(36.0, event=event), a_form(39.0, event=event)))
        offer = may_speak(moment)
        assert offer.allowed, event
        assert offer.situation == AT_A_DEAD_BALL


def test_a_replay_a_close_up_and_a_crowd_shot_are_all_the_ball_being_dead() -> None:
    """Section 4.2: 15.4 per 100 over a replay, 11.8 at a stoppage."""
    for scene in (Scene.REPLAY, Scene.CLOSE_UP, Scene.CROWD, Scene.STOPPAGE):
        moment = a_moment(40.0, forms=(a_form(36.0, scene=scene), a_form(39.0, scene=scene)))
        assert may_speak(moment).allowed, scene


def test_one_live_form_among_two_is_not_a_dead_ball() -> None:
    """Both of the last two, not either: a single close-up mid-move is a cutaway."""
    moment = a_moment(
        14.0,
        forms=(a_form(10.0, scene=Scene.CLOSE_UP), a_form(13.0, event=Event.SHOT)),
        last_big=(Event.SHOT, 13.0),
        lead_lines_since_big=1,
    )
    assert not may_speak(moment).allowed


def test_live_build_up_is_allowed_twenty_seconds_after_the_last_big_event() -> None:
    """Section 4.2's 5.9 per 100 in build-up: it does speak, just far less."""
    quiet = a_moment(125.0, last_big=(Event.SAVE, 100.0))
    offer = may_speak(quiet)
    assert offer.allowed
    # Inside a minute of the save, the examples it leans on are still the
    # ones the corpus recorded after a chance.
    assert offer.situation == AFTER_A_CHANCE
    assert may_speak(a_moment(200.0, last_big=(Event.SAVE, 100.0))).situation == IN_BUILD_UP
    assert not may_speak(a_moment(115.0, last_big=(Event.SAVE, 100.0))).allowed


def test_one_turn_a_minute_in_build_up() -> None:
    """Section 4.2 again, translated: about one turn per 45 s of build-up."""
    cfg = ColourConfig()
    assert not may_speak(a_moment(140.0, last_turn_ts=100.0, last_big=None), cfg).allowed
    assert may_speak(a_moment(146.0, last_turn_ts=100.0, last_big=None), cfg).allowed


def test_one_turn_per_big_event() -> None:
    moment = a_moment(
        118.0,
        forms=(a_form(116.0, event=Event.CORNER), a_form(117.0, event=Event.CORNER)),
        last_big=(Event.SAVE, 100.0),
        last_turn_ts=113.0,
        turns_since_big=1,
    )
    offer = may_speak(moment)
    assert not offer.allowed
    assert "one turn per big event" in offer.reason


# -- the scheduler -----------------------------------------------------------


def test_a_turn_goes_out_two_and_a_half_seconds_apart() -> None:
    assert space_out(10.0, 3, gap=2.5) == [10.0, 12.5, 15.0]


def test_no_colour_utterance_lands_on_top_of_a_caller_beat() -> None:
    """The lead has the ball; the colour voice is the one that moves."""
    lead = [(11.0, 0.0), (13.5, 0.0)]
    when = space_out(10.0, 3, gap=2.5, avoid=lead, clear=2.0)
    assert when == sorted(when)
    for at in when:
        assert min(abs(at - beat) for beat, _ in lead) >= 2.0


def test_the_spacing_survives_two_beats_in_a_row() -> None:
    lead = [(10.5, 0.0), (12.0, 0.0), (13.5, 0.0)]
    when = space_out(10.0, 2, gap=2.0, avoid=lead, clear=2.0)
    for at in when:
        assert min(abs(at - beat) for beat, _ in lead) >= 2.0
    assert when[1] - when[0] >= 2.0


def test_a_lead_beat_is_an_interval_and_not_an_instant() -> None:
    """A beat's timestamp is when the line starts, not when it finishes.

    The first paid run put "Well, Argentina are content to sit in now." 2.0 s
    after a nine-word lead line, which at WORDS_PER_SECOND is 2.8 s before
    the lead had stopped talking.
    """
    text = "Down the left now as Argentina drop deep."
    seconds = speaking_for(text)
    assert seconds == pytest.approx(len(text.split()) / 3.2)
    when = space_out(4.2, 1, gap=2.5, avoid=[(4.2, seconds)], clear=2.0)
    assert when[0] >= 4.2 + seconds + 2.0


def test_a_turn_stops_rather_than_trailing_past_its_span() -> None:
    """Section 4.6: the colour voice hands back by stopping when the ball moves."""
    lead = [(12.0, 4.0), (18.0, 4.0), (24.0, 4.0)]
    when = space_out(10.0, 4, gap=2.5, avoid=lead, clear=2.0, span=10.0)
    assert len(when) < 4
    assert all(at <= 20.0 for at in when)


# -- the prompt --------------------------------------------------------------


def test_the_system_prompt_carries_the_real_utterances_from_every_situation() -> None:
    system = colour_system(a_pack())
    for situation, utterances in COLOUR_EXAMPLES.items():
        assert utterances, situation
        assert utterances[0] in system, situation


def test_the_system_prompt_is_the_same_bytes_every_time() -> None:
    """It is sent on every turn and only earns the squads it carries if it caches."""
    assert colour_system(a_pack()) == colour_system(a_pack())


def test_the_call_shows_the_lead_s_lines_and_the_material_and_nothing_else() -> None:
    """The caller's forms are a description of the passage, and this seat may
    not write another one — so nothing but the material reaches the body."""
    body = text_of(
        colour_blocks(
            AT_A_DEAD_BALL,
            "the ball is dead",
            "Argentina 2-0 France",
            ["Messi.", "Won by Molina."],
            [],
            ["NOTE about Lionel Messi: five in this tournament"],
        )
    )
    assert "Messi." in body
    assert "close up, corner" not in body
    assert "five in this tournament" in body
    assert "the ball is dead" in body
    assert "WHAT THIS TURN IS ABOUT" in body


def test_the_call_has_no_pictures_in_it() -> None:
    """The seat is text-only: it cannot narrate what it has not been shown."""
    blocks = colour_blocks(IN_BUILD_UP, "quiet", "0-0", [], [], [])
    assert not [b for b in blocks if b.get("type") == "image"]


def test_the_cue_classifier_is_the_study_s_own() -> None:
    """Section 4.1's colour-entry definition, which is also the measurement."""
    assert opens_with_a_cue("Well, they should have made more of it.")
    assert opens_with_a_cue("Yeah, unlike the corners there.")
    assert opens_with_a_cue("I think unless there was a offside on Gamez.")
    assert not opens_with_a_cue("Messi drives at the defence.")


# -- the seat ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_turn_is_two_to_four_short_utterances() -> None:
    """Section 4.4: median run 4, each 3 to 12 words."""
    backend = speaking(
        a_turn(
            "Well, they have gone down that side twice now.",
            "And the full-back has had no help at all.",
            "You could see it coming.",
        )
    )
    seat = a_seat(backend)
    turn = await seat.turn(may_speak(a_moment(40.0)), "Argentina 2-0 France", now=40.0)
    assert turn is not None
    assert turn.speak
    assert len(turn.utterances) == 3
    assert all(len(u.split()) <= ColourConfig().max_words for u in turn.utterances)


@pytest.mark.asyncio
async def test_a_long_utterance_is_trimmed_and_a_long_turn_is_cut() -> None:
    backend = speaking(
        a_turn(*[f"Well, this is utterance number {n} of far too many." for n in range(6)])
    )
    seat = a_seat(backend, ColourConfig(max_utterances=4))
    turn = await seat.turn(may_speak(a_moment(40.0)), "", now=40.0)
    assert turn is not None
    assert len(turn.utterances) == 4


@pytest.mark.asyncio
async def test_a_turn_is_cut_to_the_room_the_lead_has_left() -> None:
    """Anything past the room is scheduled into his lines and thrown away.

    "Argentina two up, and this is the moment that changes it" was written,
    paid for, judged and refused as ``pushed_out`` on the measured pass, all
    because the model was asked for four utterances in a hole that fitted
    two.
    """
    backend = speaking(a_turn("Well, Messi again.", "And that is the corner.", "Third one."))
    seat = a_seat(backend)
    offer = replace(may_speak(a_moment(40.0)), room=1)
    turn = await seat.turn(offer, "", now=40.0)
    assert turn is not None
    assert turn.utterances == ["Well, Messi again."]


@pytest.mark.asyncio
async def test_the_goal_reaction_is_one_utterance_and_not_a_turn() -> None:
    """Section 4.3's four in-window examples are all one fragment.

    "WELL, it's the first goal of the game." / "Well, well, well." / "Well,
    they they didn't see that one coming." / "LISTEN TO THE NOISE." The
    analysis comes later, when the ball is dead.
    """
    backend = speaking(
        a_turn("Well, Messi has been waiting for that.", "That is the move of the night.")
    )
    seat = a_seat(backend)
    seat.saw_form(
        100.0,
        a_caller(Event.GOAL, "It's in!", scene=Scene.LIVE_PLAY, names=("Lionel Messi",)),
    )
    seat.saw_lead_line(101.0, "Messi!")
    offer = may_speak(a_moment(106.0, last_big=(Event.GOAL, 100.0), lead_lines_since_big=1))
    assert offer.reaction
    turn = await seat.turn(offer, "", now=106.0)
    assert turn is not None
    assert turn.utterances == ["Well, Messi has been waiting for that."]


@pytest.mark.asyncio
async def test_silence_is_an_answer_and_still_spends_the_rate_cap() -> None:
    """Otherwise a seat that declines once is asked again half a second later."""
    seat = a_seat(speaking(a_turn()))
    for ts in (4.0, 5.0, 6.0, 7.0):
        seat.saw_lead_line(ts, "Messi again.")
        seat.saw_form(ts, a_caller(Event.CORNER, "x"))
    assert seat.offer(40.0).allowed
    turn = await seat.turn(seat.offer(40.0), "")
    assert turn is not None and not turn.speak
    seat.answered(40.0)
    assert not seat.offer(40.5).allowed


def test_the_seat_reads_the_phase_off_forms_the_caller_never_spoke() -> None:
    """A form the caller filled in and chose not to say is still evidence.

    Asked of the phase gate directly, because ``offer`` also asks whether
    there is anything to talk about and two blank replay forms give it
    nothing — which is the next test.
    """
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_lead_line(1.0, "Messi.")
    seat.saw_lead_line(2.0, "Now De Paul.")
    silent = CallerLine(scene=Scene.REPLAY, event=Event.NONE, confidence=0.9, speak=False, line="")
    seat.saw_form(3.0, silent)
    seat.saw_form(4.0, silent)
    assert may_speak(seat.moment(5.0), seat.config).allowed


# -- what it is allowed to talk about ----------------------------------------


def test_a_turn_is_not_offered_when_there_is_nothing_to_make_one_out_of() -> None:
    """The judge's complaint, stopped before the call rather than after it.

    A dead ball with no note, no repeated pattern and no completed event
    produced "Everything turns on what the ref decides next" — a line that
    would fit any match ever played. There is no prompt that reliably turns
    nothing into something, so the seat is not asked.
    """
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=None)
    seat.saw_lead_line(1.0, "Messi.")
    seat.saw_lead_line(2.0, "Now De Paul.")
    blank = CallerLine(scene=Scene.CROWD, event=Event.NONE, confidence=0.9, speak=False, line="")
    seat.saw_form(3.0, blank)
    seat.saw_form(4.0, blank)
    assert may_speak(seat.moment(5.0), seat.config).allowed
    offer = seat.offer(5.0)
    assert not offer.allowed
    assert "nothing specific" in offer.reason
    assert not seat.material(5.0)


def test_a_team_level_note_is_not_material() -> None:
    """The diagnosis of the 5.0: the gate passed on notes about a country.

    A note filed under "Argentina" is true at every instant of the match, so
    it let the seat through at every instant, and what came out was about a
    pitch it cannot see: "France keeping it simple across the back". A note
    is material when it is about a man the lead has just named.
    """
    pack = a_pack().model_copy(
        update={"notes": [Note(about="Argentina", text="unbeaten since the opening game")]}
    )
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=pack)
    seat.saw_lead_line(1.0, "Argentina work it across the back.")
    seat.saw_lead_line(2.0, "Nothing on down the left.")
    seat.saw_form(3.0, a_caller(Event.NONE, "", scene=Scene.CROWD))
    seat.saw_form(4.0, a_caller(Event.NONE, "", scene=Scene.CROWD))
    assert may_speak(seat.moment(5.0), seat.config).allowed
    assert not seat.offer(5.0).allowed


def test_a_note_about_a_man_the_lead_has_just_named_is_material() -> None:
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_lead_line(1.0, "Messi, twenty yards out.")
    seat.saw_lead_line(2.0, "Argentina keep it.")
    seat.saw_form(3.0, a_caller(Event.NONE, "", scene=Scene.CROWD))
    seat.saw_form(4.0, a_caller(Event.NONE, "", scene=Scene.CROWD))
    material = seat.material(5.0)
    assert [note.text for note in material.notes] == ["five in this tournament"]
    assert seat.offer(5.0).allowed


def test_the_seat_gets_the_tallies_adjusted_note_not_the_researched_one() -> None:
    """A running-count note is stale the instant its man scores.

    ``tallies.py`` advances a ``counts=goals`` note by the goals credited to
    that player before it reaches the phraser or the gate; this seat used to
    read the pack's raw notes and never saw the match move. Sharing the same
    :class:`~commentary.tallies.Tallies` fixes it for both :meth:`notes` and
    the goal-reaction material in :meth:`_about_the_scorer`.
    """
    pack = a_pack().model_copy(
        update={
            "notes": [
                Note(
                    about="Lionel Messi",
                    text="five goals in this tournament",
                    kind="stat",
                    counts="goals",
                )
            ]
        }
    )
    tallies = Tallies()
    tallies.credit_goal("Lionel Messi", 10.0)
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=pack, tallies=tallies)
    seat.saw_lead_line(1.0, "Messi, twenty yards out.")
    seat.saw_lead_line(2.0, "Argentina keep it.")
    seat.saw_form(3.0, a_caller(Event.NONE, "", scene=Scene.CROWD))
    seat.saw_form(4.0, a_caller(Event.NONE, "", scene=Scene.CROWD))
    material = seat.material(5.0)
    assert [note.text for note in material.notes] == ["six goals in this tournament"]


def test_a_numbered_note_with_a_clause_shows_the_clause_not_the_figure() -> None:
    """The researcher's own no-number rewrite, when it exists, replaces the figure.

    "a goal in the 2018 World Cup final at nineteen" used to reach the model
    with a warning not to say "nineteen" and nothing else to say instead;
    a clause gives it something instead of a warning.
    """
    note = Note(
        about="Kylian Mbappé",
        text="a goal in the 2018 World Cup final at nineteen",
        kind="storyline",
        clause="a goal in a World Cup final, as a teenager",
    )
    lines = Material(notes=(note,)).lines()
    assert lines == [
        "NOTE about Kylian Mbappé: a goal in a World Cup final, as a teenager"
        " — that was then, not now"
    ]
    assert "nineteen" not in lines[0]
    assert "2018" not in lines[0]


def test_a_numbered_note_with_no_clause_falls_back_to_the_warning() -> None:
    """A pack written before ``clause`` existed behaves exactly as it always did."""
    note = Note(about="Kylian Mbappé", text="five goals in this tournament", kind="stat")
    lines = Material(notes=(note,)).lines()
    assert lines == [
        "NOTE about Kylian Mbappé (has a figure in it: say the fact, never the figure): "
        "five goals in this tournament"
    ]


def test_a_note_with_no_number_ignores_its_own_clause() -> None:
    """A clause is only ever a substitute for a figure; a habit note has none to hide."""
    note = Note(
        about="Kylian Mbappé",
        text="always goes to the keeper's left from the spot",
        kind="habit",
        clause="prefers the keeper's near side",
    )
    lines = Material(notes=(note,)).lines()
    assert lines == ["NOTE about Kylian Mbappé: always goes to the keeper's left from the spot"]


def test_a_completed_big_event_is_enough_to_make_a_turn_out_of() -> None:
    """An opinion about the save that has been made cannot be overtaken."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=None)
    seat.saw_lead_line(1.0, "Messi.")
    seat.saw_lead_line(2.0, "Now De Paul.")
    seat.saw_form(3.0, a_caller(Event.SAVE, "Martínez holds it.", names=("Emiliano Martínez",)))
    seat.saw_form(4.0, a_caller(Event.SAVE, ""))
    material = seat.material(18.0)
    assert "save" in material.last_event
    assert "Martínez holds it." in material.last_event
    # Eighteen seconds, not five: the first twelve after a save belong to the
    # lead, and the material is still fresh at twenty-five.
    assert seat.offer(18.0).allowed


def test_an_event_nobody_could_be_named_on_is_not_material() -> None:
    """ "A save" is an opinion about nothing; "that save from Martínez" is one."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=None)
    seat.saw_lead_line(1.0, "Messi.")
    seat.saw_lead_line(2.0, "Now De Paul.")
    seat.saw_form(3.0, a_caller(Event.SAVE, "Held."))
    assert seat.material(5.0).last_event == ""


def test_a_stale_event_is_not_material() -> None:
    """Past twenty-five seconds the moment has gone and so has the opinion."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=None)
    seat.saw_lead_line(1.0, "Messi.")
    seat.saw_lead_line(2.0, "Now De Paul.")
    seat.saw_form(3.0, a_caller(Event.SAVE, "Martínez holds it.", names=("Emiliano Martínez",)))
    assert seat.material(20.0).last_event
    assert seat.material(40.0).last_event == ""


def test_a_pattern_is_counted_in_code_and_never_by_the_model() -> None:
    """Section 4.5's first kind of colour: "a pattern that has now repeated".

    The count is computed here so that "again" is true when the seat says
    it. The figure itself never reaches air — ``says_a_number`` refuses any
    utterance carrying one.
    """
    forms = [
        a_form(1.0, event=Event.CORNER, names=("Nahuel Molina",)),
        a_form(5.0, event=Event.BUILD_UP),
        a_form(9.0, event=Event.CORNER, names=("Nahuel Molina",)),
        a_form(13.0, event=Event.CORNER),
    ]
    found = patterns_in(forms)
    assert any("2 corners on Nahuel Molina" in text for text in found)
    assert any("Nahuel Molina in the picture on 2" in text for text in found)
    assert says_a_number(found[0]), "the count is in the string for the model, not for air"


def test_the_same_thing_shown_twice_is_not_two_things() -> None:
    """The caller files one penalty over five looks; the seat is not told five.

    Counting forms rather than occurrences handed the last pass "5 penalties"
    off a single spot kick, which is false and is the kind of false a
    listener notices.
    """
    forms = [a_form(float(n), event=Event.PENALTY, names=("Kylian Mbappé",)) for n in range(5)]
    assert not [text for text in patterns_in(forms) if "penalt" in text]


def test_a_team_having_the_ball_is_not_a_pattern() -> None:
    """The whole of the last pass's build-up material, and it said nothing."""
    forms = [a_form(float(n), event=Event.BUILD_UP) for n in range(8)]
    assert patterns_in(forms) == []


def test_a_side_going_down_the_same_flank_is_a_pattern() -> None:
    """Section 4.5's shape, read off the lead's words because a form has no zone."""
    forms = [
        a_form(1.0, line="France break down the left."),
        a_form(5.0, line="Back through the middle."),
        a_form(9.0, line="And down the left again."),
    ]
    assert any("Argentina down the left" in text for text in patterns_in(forms))


def test_one_of_a_thing_is_not_a_pattern() -> None:
    assert patterns_in([a_form(1.0, event=Event.CORNER)]) == []
    assert patterns_in([]) == []


def test_the_prompt_shows_the_material_and_says_there_is_nothing_else() -> None:
    body = text_of(
        colour_blocks(
            AT_A_DEAD_BALL,
            "the ball is dead",
            "Argentina 2-0 France",
            ["Otamendi gets across."],
            [],
            [
                "REPEATED: 3 fouls on Nicolás Otamendi",
                "EVENT, finished, speak about it in the past tense: card, Nicolás Otamendi",
            ],
        )
    )
    assert "WHAT THIS TURN IS ABOUT" in body
    assert "3 fouls on Nicolás Otamendi" in body
    assert "card, Nicolás Otamendi" in body
    assert "NO MORE THAN 4 SHORT UTTERANCES" in body


def test_a_turn_with_room_for_one_utterance_is_asked_for_one() -> None:
    """The lead is still talking, so a second utterance would never be heard."""
    body = text_of(
        colour_blocks(
            AT_A_DEAD_BALL,
            "the ball is dead",
            "",
            [],
            [],
            ["REPEATED: 3 fouls on Nicolás Otamendi", "EVENT: card, Nicolás Otamendi"],
            most=1,
        )
    )
    assert "ONE UTTERANCE" in body


def test_one_item_of_material_asks_for_one_or_two_utterances() -> None:
    """Section 4.4's four-utterance run needs four utterances' worth to say."""
    body = text_of(
        colour_blocks(
            AT_A_DEAD_BALL,
            "the ball is dead",
            "",
            [],
            [],
            ["NOTE about Lionel Messi: five in this tournament"],
        )
    )
    assert "ONE OR TWO UTTERANCES" in body
    assert "why it matters" in body


def test_the_goal_reaction_is_told_whose_goal_it_was() -> None:
    body = text_of(
        colour_blocks(
            AFTER_A_GOAL,
            "4.2 s after the goal call",
            "",
            ["Messi! Into the net!"],
            [],
            ["NOTE about Lionel Messi: five in this tournament"],
            about="Lionel Messi",
        )
    )
    assert "This turn is about Lionel Messi. His name goes in the line." in body


def test_the_rules_forbid_saying_what_is_happening_now() -> None:
    """The judge's worst line: colour about the ball while Messi had it."""
    system = colour_system(a_pack())
    assert "never say what is happening on the pitch" in system
    assert "THE MATERIAL IS ALL OF IT" in system


def test_the_rules_name_the_filler_this_seat_actually_produced() -> None:
    """Every line below came out of this seat on a measured pass."""
    system = colour_system(a_pack())
    for line in ("this is what it comes down to", "keeping it simple at the back"):
        assert line in system


def test_the_seat_reaches_for_the_notes_about_the_people_the_lead_named() -> None:
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_form(
        1.0,
        CallerLine(
            scene=Scene.LIVE_PLAY,
            event=Event.CARRY,
            side=Side.HOME,
            team="Argentina",
            sightings=[Sighting(number=10, name="Lionel Messi")],
            confidence=0.8,
            speak=True,
            line="Messi turns.",
        ),
    )
    assert seat.notes() == [], "the caller read him; the lead has not said his name"
    seat.saw_lead_line(1.5, "Messi turns.")
    assert [note.text for note in seat.notes()] == ["five in this tournament"]


def test_a_note_goes_stale_once_the_lead_has_moved_on() -> None:
    """Three lines, because a note is worth saying while the man is in mind."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_lead_line(1.0, "Messi turns.")
    for ts in (2.0, 3.0, 4.0):
        seat.saw_lead_line(ts, "Argentina work it wide.")
    assert seat.notes() == []


# -- the filler check --------------------------------------------------------


def test_a_line_about_nothing_is_refused_in_code() -> None:
    """The judge's worst line, and the two the seat said either side of it."""
    for line in (
        "I think this is what it comes down to.",
        "That changes everything now.",
        "You know, he keeps finding space in transition.",
    ):
        assert is_filler(line, a_pack()), line


def test_a_line_about_a_man_a_repetition_or_an_event_is_not_filler() -> None:
    for line in (
        "Yeah, Otamendi has got himself into a mess there.",
        "Well, that is a penalty all day.",
        "And France down that side again.",
    ):
        assert not is_filler(line, a_pack()), line


def test_a_side_described_rather_than_observed_is_filler() -> None:
    """The line the judge quoted; a team name is not a subject on its own."""
    assert is_filler("Well, France keeping it simple across the back.", a_pack())
    assert is_filler("Argentina happy to sit deep and let them have it.", a_pack())


# -- the incident and the pictures of it again -------------------------------
#
# Every form below is copied off
# ``runs/rephrased/mbappe-final2/file-20260913-185228-phrased.jsonl``, which
# is the trace this section exists because of: Otamendi conceded a penalty at
# 12.9 s, the caller filed five more looks and four replays of the contact
# between 17.3 s and 37.8 s, and the colour seat was offered nothing over any
# of them and first spoke at 135.8 s. Section 4.2 has the seat at its busiest
# over a replay — 15.4 entries per 100 utterances — and it was silent.


def the_foul() -> ColourSeat:
    """The Mbappé trace to 37.8 s: a foul in the box and four replays of it."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_lead_line(8.5, "France break through the middle at speed.")
    seat.saw_lead_line(12.9, "Otamendi gets across inside the box and the man goes down.")
    seat.saw_form(
        12.9,
        a_caller(
            Event.FOUL,
            "Otamendi gets across inside the box and the France attacker goes down.",
            scene=Scene.LIVE_PLAY,
            names=("Otamendi",),
        ),
    )
    seat.saw_form(
        17.3,
        a_caller(
            Event.FOUL,
            "Otamendi protests, and the referee is already waving him away.",
            names=("Otamendi",),
        ),
    )
    seat.saw_form(
        25.4,
        a_caller(
            Event.FOUL,
            "The replay: Kolo Muani driving across, Otamendi's leg in behind him.",
            scene=Scene.REPLAY,
            names=("Kolo Muani",),
        ),
    )
    seat.saw_form(
        29.1,
        a_caller(
            Event.TACKLE,
            "The replay again: Kolo Muani riding the challenge, Otamendi chasing him.",
            scene=Scene.REPLAY,
            names=("Otamendi", "Kolo Muani"),
        ),
    )
    seat.saw_form(
        37.8,
        a_caller(
            Event.FOUL,
            "The replay shows the contact from Otamendi as the runner goes down.",
            scene=Scene.REPLAY,
            names=("Otamendi",),
        ),
    )
    return seat


def test_a_foul_is_an_event_the_second_voice_has_a_verdict_on() -> None:
    """Section 4.3: 37 fouls drew a colour entry inside 60 s, median 16.7 s.

    The material gate is what said no on the real trace — the phase gate had
    already said yes over every one of the replays — because a foul was not
    a completed event as far as this file was concerned and there was
    nothing else in the block.
    """
    seat = the_foul()
    material = seat.material(30.0)
    assert "foul" in material.last_event
    assert "Otamendi" in material.last_event
    assert material.lines(), "a foul and four replays of it is a turn"


def test_the_first_twelve_seconds_after_a_foul_belong_to_the_lead_too() -> None:
    """Section 4.3 again: after a foul only 5% of colour entries land inside
    six seconds, the lowest share in the table. The seat comes in over the
    replays, not over the referee's whistle."""
    seat = the_foul()
    assert not seat.offer(16.0).allowed
    assert "belong to the lead" in seat.offer(16.0).reason
    assert seat.offer(26.0).allowed


def test_the_replays_the_caller_wrote_are_the_evidence() -> None:
    """A verdict on contact needs the pictures the contact was shown on."""
    lines = the_foul().material(30.0).lines()
    replays = [line for line in lines if line.startswith("REPLAY")]
    assert len(replays) == 2
    assert "Otamendi's leg in behind him" in replays[0]


def test_the_verdict_and_its_evidence_come_first_in_the_block() -> None:
    """What is first in the block is what the first utterance is about."""
    lines = the_foul().material(30.0).lines()
    assert lines[0].startswith("EVENT")
    assert lines[1].startswith("REPLAY")


def test_an_incident_still_being_shown_again_is_still_fresh() -> None:
    """While the broadcast is on it, it is still the subject.

    Twenty-five seconds after the contact the foul would be stale on its own
    timestamp. It is not stale, because the caller filed a replay of it at
    37.8 s and freshness is measured off the last look.
    """
    seat = the_foul()
    assert seat.material(50.0).last_event, "a replay at 37.8 s keeps the 12.9 s foul alive"
    assert seat.material(70.0).last_event == "", "and nothing keeps it alive forever"


def test_the_replays_are_capped_at_what_the_config_says() -> None:
    seat = ColourSeat(
        ScriptedBackend(), config=ColourConfig(replays_shown=2), pack=a_pack()
    )
    seat.saw_lead_line(1.0, "Otamendi in.")
    for index, ts in enumerate((10.0, 12.0, 14.0, 16.0)):
        seat.saw_form(
            ts,
            a_caller(
                Event.FOUL, f"The replay, look {index}.", scene=Scene.REPLAY, names=("Otamendi",)
            ),
        )
    assert len(seat.material(18.0).replays) == 2


def test_a_replay_sequence_is_its_own_situation() -> None:
    """Not "after a chance": a chance is a shot and this is an argument."""
    seat = the_foul()
    assert seat.offer(26.0).situation == OVER_A_REPLAY


def test_a_card_at_a_stoppage_is_the_same_kind_of_moment() -> None:
    """Section 3.2's booking window is the colour voice arguing about it,
    with or without the pictures up."""
    moment = a_moment(
        30.0,
        forms=(a_form(26.0, scene=Scene.STOPPAGE, event=Event.CARD),),
        last_big=(Event.CARD, 25.0),
    )
    assert may_speak(moment).situation == OVER_A_REPLAY


def test_one_piece_of_contact_is_one_incident_however_it_is_filed() -> None:
    """The caller called it a foul, then a tackle, then a foul again.

    Three events would be three turns on one challenge, and would reset the
    rate cap each time.
    """
    seat = the_foul()
    moment = seat.moment(40.0)
    assert moment.last_big is not None
    assert moment.last_big[0] is Event.FOUL
    assert moment.last_big[1] == pytest.approx(12.9), "the look the lead called it on"


def test_an_incident_never_takes_a_fresh_goal_s_place() -> None:
    """A tackle five seconds after a goal is not what the turn is about."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_lead_line(1.0, "Mbappé.")
    seat.saw_form(10.0, a_caller(Event.GOAL, "It's in!", names=("Kylian Mbappé",)))
    seat.saw_form(15.0, a_caller(Event.TACKLE, "Back to the halfway line."))
    moment = seat.moment(16.0)
    assert moment.last_big == (Event.GOAL, 10.0)


def test_the_run_of_looks_reaches_past_the_one_the_event_was_called_on() -> None:
    """The scorer is often only legible on the *next* look.

    On the Mbappé trace the goal form at 82.5 s carried four unreadable
    shirts and the one at 86.8 s carried "Mbappé", so the goal reaction —
    section 4.3's one sanctioned fragment inside the lead's window — was
    refused for having nobody to be about.
    """
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_lead_line(1.0, "France drive in.")
    seat.saw_lead_line(2.0, "It's in.")
    seat.saw_form(10.0, a_caller(Event.GOAL, "Steps up and strikes it.", scene=Scene.LIVE_PLAY))
    seat.saw_lead_line(12.0, "Buried, and the keeper went the other way.")
    seat.saw_form(
        14.0,
        a_caller(Event.GOAL, "Into the net for the ball.", names=("Kylian Mbappé",)),
    )
    offer = seat.offer(15.0)
    assert offer.reaction and offer.allowed
    assert seat.material(15.0, reaction=True).about == "Kylian Mbappé"


# -- the offline pass --------------------------------------------------------


def _trace() -> list[dict[str, Any]]:
    """A tiny synthetic trace: a goal, then a corner twenty seconds later."""
    rows: list[dict[str, Any]] = [
        {
            "topic": "state",
            "ts": 0.0,
            "home": "Argentina",
            "away": "France",
            "home_score": 2,
            "away_score": 0,
        }
    ]
    forms = [
        (1.0, Scene.LIVE_PLAY, Event.BUILD_UP, "Argentina keep it."),
        (5.0, Scene.LIVE_PLAY, Event.CARRY, "Messi turns."),
        (9.0, Scene.LIVE_PLAY, Event.GOAL, "It's in!"),
        (12.0, Scene.CLOSE_UP, Event.GOAL, "Messi again."),
        (30.0, Scene.STOPPAGE, Event.CORNER, "Corner to Argentina."),
        (34.0, Scene.STOPPAGE, Event.CORNER, "Molina to take it."),
    ]
    # Two forms the caller filled in and did not speak: the hole in the lead's
    # cadence that the second voice actually fits into. Without one, a trace
    # whose lead talks every four seconds leaves the colour seat no room at
    # all, which is a true thing about this system and a poor fixture.
    quiet = [(40.0, Scene.STOPPAGE, Event.CORNER), (46.0, Scene.STOPPAGE, Event.CORNER)]
    for ts, scene, event, line in forms:
        rows.append(
            {
                "topic": "caller",
                "ts": ts,
                "scene": scene.value,
                "event": event.value,
                "side": "home",
                "team": "Argentina",
                "sightings": [],
                "confidence": 0.8,
                "speak": True,
                "line": line,
            }
        )
        rows.append(
            {
                "topic": "beat",
                "ts": ts,
                "id": f"b{ts}",
                "voice": "caller",
                "text": line,
                "video_ts": ts,
                "created_ts": 0.0,
                "live_ts": ts + 8.0,
                "event": event.value,
                "preemptable": event is not Event.GOAL,
            }
        )
    for ts, scene, event in quiet:
        rows.append(
            {
                "topic": "caller",
                "ts": ts,
                "scene": scene.value,
                "event": event.value,
                "side": "home",
                "team": "Argentina",
                "sightings": [],
                "confidence": 0.8,
                "speak": False,
                "line": "",
            }
        )
    return rows


def _settings() -> Settings:
    return Settings(colour=ColourConfig(), predictor=PredictorConfig(tick_s=0.5))


@pytest.mark.asyncio
async def test_the_offline_pass_inserts_beats_only_where_the_phase_allows() -> None:
    backend = speaking(a_turn("Well, that is a corner they have earned.", "Molina is the man."))
    out = await colour_pass(
        _trace(), backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    colour_beats = [
        row for row in out.rows if row.get("topic") == "beat" and row.get("voice") == "analyst"
    ]
    assert colour_beats, "the seat never spoke"
    for row in colour_beats:
        ts = float(row["ts"])
        # Nothing inside the goal's twelve seconds, and nothing before the
        # goal at all: the lead had only two lines by then.
        assert not (9.0 <= ts < 21.0) or ts >= 13.0
        assert row["event"] == "none"
        assert row["preemptable"] is True


@pytest.mark.asyncio
async def test_the_offline_pass_keeps_the_colour_voice_off_the_lead_s_beats() -> None:
    backend = speaking(a_turn("Well, they have earned that.", "Molina is the man."))
    out = await colour_pass(
        _trace(), backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    lead = [
        (float(r["ts"]), speaking_for(str(r["text"])))
        for r in out.rows
        if r.get("topic") == "beat" and r.get("voice") == "caller"
    ]
    for row in out.rows:
        if row.get("topic") == "beat" and row.get("voice") == "analyst":
            at = float(row["ts"])
            for start, seconds in lead:
                assert not (start - 2.0 < at < start + seconds + 2.0), (
                    f"colour at {at} talks over the lead line starting at {start}"
                )


@pytest.mark.asyncio
async def test_the_offline_pass_writes_the_trace_in_timestamp_order() -> None:
    """A colour beat stamped 6.2 s must not sit above a caller beat stamped 0.27 s.

    The trace is append-only and steps backwards on its own — a board read at
    cursor 2.0 is published while a caller call that started at 0.27 is still
    in flight — so the merge places each addition after the last row stamped
    at or before it rather than assuming the file goes forwards.
    """
    rows = _trace()
    # A board read published late, the way the real runtime writes them.
    rows.insert(1, {"topic": "board", "ts": 2.0, "bug_visible": True})
    rows.insert(2, {"topic": "board", "ts": 20.0, "bug_visible": True})
    backend = speaking(a_turn("Well, they have earned that.", "Molina is the man."))
    out = await colour_pass(
        rows, backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    added = [
        i
        for i, r in enumerate(out.rows)
        if r.get("topic") == "colour" or (r.get("topic") == "beat" and r.get("voice") == "analyst")
    ]
    assert added
    for i in added:
        before = [float(r.get("ts", 0.0)) for r in out.rows[:i]]
        assert all(ts <= float(out.rows[i]["ts"]) + 1e-6 for ts in before[-1:]), (
            "a colour row was placed above a row stamped later than it"
        )


@pytest.mark.asyncio
async def test_every_colour_utterance_goes_through_the_fact_gate() -> None:
    """A scoreline out of the second voice is struck out, as the prompt says."""
    backend = speaking(a_turn("Well, it is two nil now.", "Molina is the man."))
    out = await colour_pass(
        _trace(), backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    refused = out.refused
    assert refused, "the gate let a scoreline through"
    assert any("score" in reason for u in refused for reason in u.reasons)
    said = [r["text"] for r in out.rows if r.get("topic") == "beat" and r.get("voice") == "analyst"]
    assert "Well, it is two nil now." not in said


@pytest.mark.asyncio
async def test_the_second_voice_does_not_say_numbers() -> None:
    """Section 5.1: numbers are the lead's job, and this seat invents them.

    The share of number-carrying lines that open like the colour voice is
    3.4-12.8% against a base rate of 2.4-14.1% — within noise on every file —
    so nothing the corpus records is lost by taking the figures off this seat
    entirely, and two inventions the fact gate could not check were.
    """
    backend = speaking(a_turn("Well, that has only happened once.", "Molina is the man."))
    out = await colour_pass(
        _trace(), backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    assert any("number_claim" in reason for u in out.refused for reason in u.reasons)
    said = [r["text"] for r in out.rows if r.get("topic") == "beat" and r.get("voice") == "analyst"]
    assert "Well, that has only happened once." not in said


@pytest.mark.asyncio
async def test_a_colour_row_is_written_for_every_turn_with_what_it_cost() -> None:
    backend = speaking(a_turn("Well, they have earned that.", "Molina is the man."))
    out = await colour_pass(
        _trace(), backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    rows = [row for row in out.rows if row.get("topic") == "colour"]
    assert rows
    assert len(rows) == len(out.turns)
    for row in rows:
        assert "usd" in row
        assert row["situation"] in COLOUR_EXAMPLES


@pytest.mark.asyncio
async def test_the_old_silence_timer_analyst_s_beats_are_dropped() -> None:
    """Two colour seats on one channel is one too many.

    A trace made by the old runtime carries ``AnalystConfig``'s thirty-word
    lines as beats. Left in, the judge would score both as the second voice.
    The ``analyst`` rows stay: they are the record of what it said, and so do
    the beats when this seat itself says nothing — a trace with no colour at
    all is worse than a trace with the old colour.
    """
    rows = _trace()
    rows.append(
        {
            "topic": "beat",
            "ts": 20.0,
            "id": "a1",
            "voice": "analyst",
            "text": "Otamendi is taking no chances with the fresh legs off the French bench.",
            "video_ts": 20.0,
            "created_ts": 0.0,
            "live_ts": 28.0,
            "event": "none",
        }
    )
    rows.append({"topic": "analyst", "ts": 20.0, "line": "Otamendi is taking no chances."})
    backend = speaking(a_turn("Well, they have earned that.", "Molina is the man."))
    out = await colour_pass(
        rows, backend, pack=a_pack(), settings=_settings(), model="claude-haiku-4-5"
    )
    said = [r["text"] for r in out.rows if r.get("topic") == "beat" and r.get("voice") == "analyst"]
    assert not any("fresh legs" in text for text in said)
    assert [r for r in out.rows if r.get("topic") == "analyst"]


@pytest.mark.asyncio
async def test_the_pass_does_nothing_at_all_when_the_seat_is_off() -> None:
    rows = _trace()
    out = await colour_pass(
        rows, ScriptedBackend(), pack=a_pack(), settings=_settings(), model="off"
    )
    assert out.rows == rows
    assert out.turns == []
    assert out.cost_usd == 0.0


# -- the live runtime --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_runtime_speaks_the_colour_seat_and_skips_the_old_analyst(
    tmp_path: Any,
) -> None:
    """End to end on the simulator: the turn reaches the channel as beats.

    The old analyst is not asked at all when the seat is on — that is the
    whole of what ``settings.colour.enabled`` does — and the utterances go
    out as separate preemptable beats with ``voice=analyst``, so the
    director drops the rest of a turn the instant a goal arrives.
    """
    from commentary.config import CallerConfig, CaptureConfig, DirectorConfig
    from commentary.runtime import Runtime
    from commentary.sim import MatchSim, SimOracle, SimSource
    from commentary.trace import RunTrace
    from commentary.voice import LogSpeaker

    sim = MatchSim(seed=5, duration_s=120.0)
    settings = Settings(
        capture=CaptureConfig(fps=6, delay_s=4.0, history_s=3.0, present_offset_s=3.0),
        caller=CallerConfig(min_gap_s=2.0),
        predictor=PredictorConfig(tick_s=0.05),
        director=DirectorConfig(max_beat_age_s=30.0),
        colour=ColourConfig(min_lead_lines=1, min_gap_s=1.0, utterance_gap_s=0.05),
    )
    # The oracle answers off the timestamps burned into the frames, and this
    # seat is sent no frames at all, so the colour tag is answered beside it
    # rather than by it. That is a real hole in the simulator and is written
    # down as one: ``--backend oracle`` cannot exercise a text-only seat.
    oracle = SimOracle(sim=sim)
    scripted = speaking(
        a_turn("Well, that corner was there to be won.", "The keeper never moved for it.")
    )

    class _Either:
        """The oracle for everything that looks at a picture, a script for this seat."""

        async def parse(self, **kwargs: Any) -> Any:
            which = scripted if kwargs.get("tag") == "colour" else oracle
            return await which.parse(**kwargs)

        @property
        def total(self) -> Any:
            return oracle.total

    path = tmp_path / "run.jsonl"
    with RunTrace(path=path) as trace:
        runtime = Runtime(
            source=SimSource(sim, settings.capture, realtime=False),
            backend=_Either(),  # type: ignore[arg-type]
            pack=sim.knowledge_pack,
            settings=settings,
            speaker=LogSpeaker(words_per_second=120),
            trace=trace,
        )
        await runtime.run(seconds=6.0)

    rows = [__import__("json").loads(line) for line in path.read_text().splitlines() if line]
    assert [r for r in rows if r.get("topic") == "colour"], "the seat was never asked"
    beats = [r for r in rows if r.get("topic") == "beat" and r.get("voice") == "analyst"]
    assert beats, "the turn never reached the channel"
    assert all(r["preemptable"] for r in beats)
    assert not [r for r in rows if r.get("topic") == "analyst"], "the old analyst was asked"


def _penalty_trace() -> list[dict[str, Any]]:
    """Otamendi concedes, the pictures come back, Mbappé scores it.

    Trimmed off the 13 September trace of the 2022 final: the forms and the
    state row the seat read at the moment it wrote the two lines this
    fixture exists to refuse.
    """
    rows: list[dict[str, Any]] = [
        {
            "topic": "state",
            "ts": 0.0,
            "home": "Argentina",
            "away": "France",
            "home_score": 2,
            "away_score": 1,
        }
    ]
    forms = [
        (2.0, Scene.LIVE_PLAY, Event.FOUL, "Otamendi gets across in the box.", ["Otamendi"], True),
        (8.0, Scene.REPLAY, Event.FOUL, "The replay: the leg in behind him.", ["Otamendi"], False),
        (20.0, Scene.STOPPAGE, Event.PENALTY, "Pointed to the spot.", ["Mbappé"], True),
        (40.0, Scene.LIVE_PLAY, Event.GOAL, "Buried.", ["Mbappé"], True),
        (70.0, Scene.STOPPAGE, Event.STOPPAGE, "", [], False),
        (76.0, Scene.STOPPAGE, Event.STOPPAGE, "", [], False),
    ]
    for ts, scene, event, line, names, spoke in forms:
        rows.append(
            {
                "topic": "caller",
                "ts": ts,
                "scene": scene.value,
                "event": event.value,
                "side": "home",
                "team": "Argentina",
                "sightings": [{"name": name} for name in names],
                "confidence": 0.8,
                "speak": spoke,
                "line": line,
            }
        )
        if spoke:
            rows.append(
                {
                    "topic": "beat",
                    "ts": ts,
                    "id": f"b{ts}",
                    "voice": "caller",
                    "text": line,
                    "video_ts": ts,
                    "created_ts": 0.0,
                    "live_ts": ts + 8.0,
                    "event": event.value,
                    "preemptable": event is not Event.GOAL,
                }
            )
    return rows


def a_wider_pack() -> KnowledgePack:
    pack = a_pack()
    pack.away.starters.append(Player(name="Dayotchanculle Upamecano", number=18))
    pack.notes.append(
        Note(
            about="Dayotchanculle Upamecano",
            text="missed the semi-final ill, back in the side tonight",
            kind="storyline",
        )
    )
    return pack


@pytest.mark.asyncio
async def test_the_offline_pass_refuses_the_turn_that_blamed_the_wrong_man() -> None:
    """The measured turn, put back through the pass it came out of.

    Both utterances went to air on 13 September. The first is the note, said
    plainly, and it is fine; the second hangs the penalty on the man in the
    note, and the man in the note was not on the penalty. What this holds is
    the wiring: the second utterance is judged knowing who the first one
    named, because the second one says "he" and means him.
    """
    backend = speaking(
        a_turn(
            "Well, Upamecano was the man ruled out for the semi.",
            "Back in and he's just conceded the penalty.",
        )
    )
    out = await colour_pass(_penalty_trace(), backend, pack=a_wider_pack(), settings=_settings())
    assert out.spoke, "the incident is offered a turn at all, which is the other half"
    assert out.spoke[0].situation == OVER_A_REPLAY
    judged = list(out.spoke[0].utterances)
    assert len(judged) == 2, "the turn was offered, called and scheduled"
    assert judged[0].passed, "the note itself is a line and survives"
    assert not judged[1].passed
    assert judged[1].reasons[0].startswith("attribution:")
    assert "Upamecano" in judged[1].reasons[0]
    assert not any("conceded the penalty" in line.text for line in out.spoken)


def test_a_fresh_incident_does_not_wait_out_the_build_up_gap() -> None:
    """One turn per incident, and the next incident starts its own.

    The build-up rate is one turn a minute (section 4.2's 5.9 entries per
    100 utterances). A foul twenty seconds after the last turn is not
    build-up, and section 4.2 puts the seat at 11.8 entries per 100 at a
    stoppage — five times its rate in a move — so the gap does not apply.
    """
    forms = (
        a_form(118.0, scene=Scene.STOPPAGE, event=Event.FOUL),
        a_form(124.0, scene=Scene.REPLAY, event=Event.FOUL),
    )
    fresh = a_moment(
        130.0, forms=forms, last_big=(Event.FOUL, 118.0), last_turn_ts=110.0, turns_since_big=0
    )
    assert may_speak(fresh).allowed
    assert may_speak(fresh).situation == OVER_A_REPLAY
    # And exactly one: the second look at the same challenge is not a second
    # turn, which is what stops the seat talking over the whole sequence.
    spent = replace(fresh, turns_since_big=1)
    assert not may_speak(spent).allowed
    assert "one turn per big event" in may_speak(spent).reason


# -- the continuation, the note said twice, and the cue ----------------------
#
# All three are read off ``runs/rephrased/r2-colour/mbappe``, the first pass
# with the incident seat in it. No fabrication in seven turns, and three other
# things wrong: every turn was one utterance long because the second was
# refused every time, one note about a substitute was said three times in
# thirty seconds by two voices, and two turns running opened on "You know,".


def test_a_continuation_may_say_he_when_the_line_before_it_named_one_man() -> None:
    """Section 4.4's runs are full of them: "Morris is the man. / He's the
    man here." The utterances go out two and a half seconds apart and nobody
    is reintroduced in between.

    All four of these were refused as ``colour_filler`` on the measured pass,
    which is why every turn in it was one utterance long.
    """
    first = "Well, Otamendi went through the back of him there."
    carries_on = "He never lets a runner past him, that lad."
    assert not is_filler(carries_on, a_pack(), after=first)
    assert not is_filler("And here he is in a World Cup final.", a_pack(), after=first)


def test_the_utterance_that_opens_a_turn_still_has_to_name_somebody() -> None:
    """A listener coming to the turn cold has nobody to call "he"."""
    assert is_filler("He never lets a runner past him, that lad.", a_pack())
    assert is_filler("And here he is in a World Cup final.", a_pack())


def test_a_continuation_after_a_line_naming_two_men_has_no_referent() -> None:
    """ "He" after a line naming two men points at neither."""
    two = "Otamendi and Messi were both in there."
    assert one_subject(two, a_pack()) == ""
    assert is_filler("And he has done that all night.", a_pack(), after=two)


def test_a_continuation_still_has_to_be_worth_hearing() -> None:
    """The three the seat wrote last time, as second lines, after a good first."""
    first = "You know, Mbappé took that penalty without looking up."
    for empty in (
        "And that is the price of it right there.",
        "Now the question is what he can do again.",
        "Yeah, that's a finish at this moment.",
    ):
        assert says_nothing(empty), empty
        assert is_filler(empty, a_pack(), after=first), empty


def test_a_stock_phrase_is_refused_wherever_it_sits() -> None:
    """Naming the right man does not redeem a line that says nothing."""
    assert says_nothing("Well, for Mbappé this is what it comes down to.")
    assert is_filler("Well, for Mbappé this is what it comes down to.", a_pack())


def test_pointing_at_itself_is_only_filler_at_the_end() -> None:
    assert says_nothing("And that is the moment right there.")
    assert not says_nothing("Otamendi's leg was right there for him to fall over.")


def test_a_note_either_voice_has_just_said_is_not_material_again() -> None:
    """The measured repetition: the lead at 133.2 s, the seat at 141.0 s and
    again at 162.8 s, one note about a substitute, thirty seconds.

    ``threads.CALLBACK_QUIET_S`` is the corpus's own number — section 7's
    callbacks are minutes apart — and it is what the lead's notes are already
    rested by.
    """
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_wider_pack())
    seat.saw_lead_line(120.0, "France work it out through Upamecano.")
    assert [note.about for note in seat.notes(121.0)] == ["Dayotchanculle Upamecano"]
    seat.saw_lead_line(133.2, "Upamecano, back in the side tonight.")
    assert seat.notes(141.0) == [], "the lead has just said it"
    assert seat.notes(460.0), "and five minutes on it is a callback again"


def test_the_seats_own_utterance_rests_the_note_too() -> None:
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_wider_pack())
    seat.saw_lead_line(120.0, "France work it out through Upamecano.")
    seat.accept(["You know, Upamecano back after missing the semi with illness."], ts=141.0)
    assert seat.notes(162.8) == []


def test_a_line_about_another_man_leaves_the_note_alone() -> None:
    """Two men went into that final level at the top of the scoring charts."""
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_wider_pack())
    seat.saw_lead_line(120.0, "France work it out through Upamecano.")
    seat.saw_lead_line(133.2, "Mbappé, back in the side tonight after the semi.")
    assert seat.notes(141.0)


def test_a_turn_does_not_open_on_the_cue_the_last_one_opened_on() -> None:
    """The rules have asked for this since the seat existed; two turns
    running opened on "You know," and two more on "Yeah,"."""
    assert swap_cue("You know, Upamecano missing that semi.", "you know").startswith("Well,")
    assert swap_cue("Well, Otamendi's leg was there.", "well").startswith("Yeah,")
    assert swap_cue("Yeah, that was soft.", "well") == "Yeah, that was soft."
    assert swap_cue("Otamendi's leg was there.", "well") == "Otamendi's leg was there."


@pytest.mark.asyncio
async def test_the_seat_swaps_the_cue_rather_than_paying_for_a_second_call() -> None:
    """An opener carries no claim, so rewriting it can make nothing false —
    and a re-ask would double what a turn costs for one word."""
    backend = speaking(
        a_turn("Well, Molina has been up that flank again."),
        a_turn("Well, Messi has dropped in again."),
    )
    seat = a_seat(backend)
    first = await seat.turn(Offer(True, AT_A_DEAD_BALL, "dead ball"), "0-0", now=10.0)
    second = await seat.turn(Offer(True, AT_A_DEAD_BALL, "dead ball"), "0-0", now=70.0)
    assert first is not None and second is not None
    assert first.utterances[0].startswith("Well,")
    assert not second.utterances[0].startswith("Well,")
    assert "Messi has dropped in again" in second.utterances[0]
