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

from typing import Any

import pytest

from commentary.agents.colour import (
    AFTER_A_CHANCE,
    AFTER_A_GOAL,
    AT_A_DEAD_BALL,
    IN_BUILD_UP,
    ColourSeat,
    FormAt,
    Moment,
    colour_pass,
    may_speak,
    space_out,
    speaking_for,
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


def test_the_call_shows_the_lead_s_lines_the_forms_and_the_notes() -> None:
    body = text_of(
        colour_blocks(
            AT_A_DEAD_BALL,
            "the ball is dead",
            "Argentina 2-0 France",
            ["Messi.", "Won by Molina."],
            ["close up, corner; Argentina have it"],
            [Note(about="Lionel Messi", text="five in this tournament")],
            [],
        )
    )
    assert "Messi." in body
    assert "close up, corner" in body
    assert "five in this tournament" in body
    assert "the ball is dead" in body


def test_the_call_has_no_pictures_in_it() -> None:
    """The seat is text-only: it cannot narrate what it has not been shown."""
    blocks = colour_blocks(IN_BUILD_UP, "quiet", "0-0", [], [], [], [])
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
    seat = ColourSeat(backend, config=ColourConfig(), pack=a_pack())
    turn = await seat.turn(may_speak(a_moment(40.0)), "Argentina 2-0 France")
    assert turn is not None
    assert turn.speak
    assert len(turn.utterances) == 3
    assert all(len(u.split()) <= ColourConfig().max_words for u in turn.utterances)


@pytest.mark.asyncio
async def test_a_long_utterance_is_trimmed_and_a_long_turn_is_cut() -> None:
    backend = speaking(
        a_turn(*[f"Well, this is utterance number {n} of far too many." for n in range(6)])
    )
    seat = ColourSeat(backend, config=ColourConfig(max_utterances=4), pack=a_pack())
    turn = await seat.turn(may_speak(a_moment(40.0)), "")
    assert turn is not None
    assert len(turn.utterances) == 4


@pytest.mark.asyncio
async def test_the_goal_reaction_is_one_utterance_and_not_a_turn() -> None:
    """Section 4.3's four in-window examples are all one fragment.

    "WELL, it's the first goal of the game." / "Well, well, well." / "Well,
    they they didn't see that one coming." / "LISTEN TO THE NOISE." The
    analysis comes later, when the ball is dead.
    """
    backend = speaking(a_turn("Well, well, well.", "That is the move of the night.", "Lovely."))
    seat = ColourSeat(backend, config=ColourConfig(), pack=a_pack())
    offer = may_speak(a_moment(106.0, last_big=(Event.GOAL, 100.0), lead_lines_since_big=1))
    assert offer.reaction
    turn = await seat.turn(offer, "")
    assert turn is not None
    assert turn.utterances == ["Well, well, well."]


@pytest.mark.asyncio
async def test_silence_is_an_answer_and_still_spends_the_rate_cap() -> None:
    """Otherwise a seat that declines once is asked again half a second later."""
    seat = ColourSeat(speaking(a_turn()), config=ColourConfig(), pack=a_pack())
    for ts in (1.0, 2.0, 3.0, 4.0):
        seat.saw_lead_line(ts, f"Line {ts}.")
        seat.saw_form(
            ts,
            CallerLine(
                scene=Scene.STOPPAGE, event=Event.CORNER, confidence=1.0, speak=True, line="x"
            ),
        )
    assert seat.offer(40.0).allowed
    turn = await seat.turn(seat.offer(40.0), "")
    assert turn is not None and not turn.speak
    seat.answered(40.0)
    assert not seat.offer(40.5).allowed


def test_the_seat_reads_the_phase_off_forms_the_caller_never_spoke() -> None:
    seat = ColourSeat(ScriptedBackend(), config=ColourConfig(), pack=a_pack())
    seat.saw_lead_line(1.0, "Messi.")
    seat.saw_lead_line(2.0, "Now De Paul.")
    silent = CallerLine(scene=Scene.REPLAY, event=Event.NONE, confidence=0.9, speak=False, line="")
    seat.saw_form(3.0, silent)
    seat.saw_form(4.0, silent)
    assert seat.offer(5.0).allowed


def test_the_seat_reaches_for_the_notes_about_the_people_on_the_forms() -> None:
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
    assert [note.text for note in seat.notes()] == ["five in this tournament"]


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
    assert "Molina is the man." in said


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
    scripted = speaking(a_turn("Well, Argentina have settled into this.", "Nothing rushed."))

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
